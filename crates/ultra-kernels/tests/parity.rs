//! Bit-exact parity of every available SIMD backend against the scalar reference
//! (spec §5.5 row 1): random shapes with tails, extreme operands, every packed code,
//! load-time refusals and thread-count invariance (rule N3).

use proptest::prelude::*;
use ultra_core::ternary::validate_t1;
use ultra_core::CoreError;
use ultra_kernels::*;

/// (rows, din, frames) with tails that are not multiples of 32/64/128.
fn shape() -> impl Strategy<Value = (usize, usize, usize)> {
    (
        1usize..=70,
        prop_oneof![1usize..=40, 100usize..=300],
        1usize..=17,
    )
}

fn run<M: Gemv>(m: M, b: Backend, x: &Frames<'_>) -> Vec<i32> {
    let mut out = vec![i32::MIN; m.rows() * x.count()];
    m.gemv(b, x, &mut out).expect("gemv");
    out
}

/// Every backend this host can run, plus the scalar reference first.
fn backends() -> Vec<Backend> {
    let mut v = vec![Backend::Scalar];
    v.extend(
        Backend::available()
            .into_iter()
            .filter(|&b| b != Backend::Scalar),
    );
    v
}

fn check_all_ternary(rows: usize, din: usize, trits: &[i8], acts: &[u8]) {
    let x = Frames::new(acts, din).expect("frames");
    let i8m = I8Matrix::new(rows, din, trits.to_vec()).expect("i8");
    let t1 = T1Matrix::from_trits(rows, din, trits).expect("t1");
    let tl = TLut2Matrix::from_trits(rows, din, trits).expect("tlut2");
    let want = run(i8m.view(), Backend::Scalar, &x);
    for b in backends() {
        assert_eq!(run(i8m.view(), b, &x), want, "i8 {b:?}");
        assert_eq!(run(t1.view(), b, &x), want, "t1 {b:?}");
        assert_eq!(run(tl.view(), b, &x), want, "tlut2 {b:?}");
    }
    assert_eq!(
        run(TLut2Matrix::from_t1(t1.view()).view(), Backend::Scalar, &x),
        want
    );
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(96))]

    #[test]
    fn ternary_kernels_match_scalar(
        (rows, din, f) in shape(),
        seed in any::<u64>(),
    ) {
        let mut s = seed | 1;
        let mut next = || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; s };
        let trits: Vec<i8> = (0..rows * din).map(|_| (next() % 3) as i8 - 1).collect();
        let acts: Vec<u8> = (0..f * din).map(|_| match next() % 4 {
            0 => 0, 1 => 127, _ => (next() & 0x7F) as u8 }).collect();
        check_all_ternary(rows, din, &trits, &acts);
    }

    #[test]
    fn int8_kernel_matches_scalar(
        (rows, din, f) in shape(),
        seed in any::<u64>(),
    ) {
        let mut s = seed | 1;
        let mut next = || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; s };
        let w: Vec<i8> = (0..rows * din).map(|_| match next() % 4 {
            0 => -127, 1 => 127, _ => ((next() % 255) as i16 - 127) as i8 }).collect();
        let a: Vec<u8> = (0..f * din).map(|_| match next() % 4 {
            0 => 0, 1 => 127, _ => (next() & 0x7F) as u8 }).collect();
        let m = I8Matrix::new(rows, din, w).expect("i8");
        let x = Frames::new(&a, din).expect("frames");
        let want = run(m.view(), Backend::Scalar, &x);
        for b in backends() {
            prop_assert_eq!(run(m.view(), b, &x), want.clone(), "{:?}", b);
        }
    }

    #[test]
    fn thread_count_never_changes_a_result(
        (rows, din, f) in shape(),
        threads in 1usize..=8,
        seed in any::<u64>(),
    ) {
        let mut s = seed | 1;
        let mut next = || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; s };
        let trits: Vec<i8> = (0..rows * din).map(|_| (next() % 3) as i8 - 1).collect();
        let acts: Vec<u8> = (0..f * din).map(|_| (next() & 0x7F) as u8).collect();
        let x = Frames::new(&acts, din).expect("frames");
        let t1 = T1Matrix::from_trits(rows, din, &trits).expect("t1");
        let tl = TLut2Matrix::from_trits(rows, din, &trits).expect("tlut2");
        let b = Backend::detect();
        let want = run(t1.view(), Backend::Scalar, &x);
        for (name, got) in [("t1", par(t1.view(), b, &x, threads)), ("tlut2", par(tl.view(), b, &x, threads))] {
            prop_assert_eq!(got, want.clone(), "{} threads={}", name, threads);
        }
    }
}

fn par<M: Gemv>(m: M, b: Backend, x: &Frames<'_>, threads: usize) -> Vec<i32> {
    let mut out = vec![i32::MIN; m.rows() * x.count()];
    gemv_par(m, b, x, &mut out, threads).expect("par");
    out
}

#[test]
fn extreme_operands_at_worst_case_fan_in() {
    // Every pair hits 2 x 127 x 127 in vpmaddubsw; din has a 31-wide tail.
    let (rows, din, f) = (3, 4096 + 31, 5);
    let x_hi = vec![127u8; f * din];
    let x = Frames::new(&x_hi, din).expect("frames");
    for wv in [127i8, -127] {
        let m = I8Matrix::new(rows, din, vec![wv; rows * din]).expect("i8");
        let want = vec![i32::from(wv) * 127 * din as i32; rows * f];
        for b in backends() {
            assert_eq!(run(m.view(), b, &x), want, "{b:?} w={wv}");
        }
    }
    for tv in [1i8, -1, 0] {
        check_all_ternary(rows, din, &vec![tv; rows * din], &x_hi);
    }
}

#[test]
fn every_valid_t1_byte_and_every_tlut2_state() {
    // 81 valid bytes, each repeated across a 128-wide SIMD block plus a tail.
    let valid: Vec<u8> = (0..=255u8).filter(|&b| validate_t1(&[b]).is_ok()).collect();
    assert_eq!(valid.len(), 81);
    let din: usize = 81 * 4 + 6;
    let stride = din.div_ceil(4);
    let mut packed = Vec::new();
    for rot in 0..4 {
        packed.extend(valid.iter().cycle().skip(rot * 20).take(stride));
    }
    let t1 = T1Matrix::from_packed(4, din, packed).expect("t1");
    let acts: Vec<u8> = (0..3 * din).map(|j| ((j * 37) % 128) as u8).collect();
    let x = Frames::new(&acts, din).expect("frames");
    let tl = TLut2Matrix::from_t1(t1.view());
    let want = run(t1.view(), Backend::Scalar, &x);
    for b in backends() {
        assert_eq!(run(t1.view(), b, &x), want, "t1 {b:?}");
        assert_eq!(run(tl.view(), b, &x), want, "tlut2 {b:?}");
    }
    // All nine pair states appear in the re-encoded image.
    let mut seen = [false; 9];
    for &b in tl.view().image() {
        seen[usize::from(b & 15)] = true;
        seen[usize::from(b >> 4)] = true;
    }
    assert!(seen.iter().all(|&s| s));
}

#[test]
fn code_3_and_bad_states_are_refused_at_load() {
    for b in 0..=255u8 {
        let has3 = (0..4).any(|k| (b >> (2 * k)) & 3 == 3);
        let r = T1Matrix::from_packed(1, 4, vec![b]);
        assert_eq!(r.is_err(), has3, "byte {b:#04x}");
        if has3 {
            assert_eq!(r.unwrap_err(), KernelError::Core(CoreError::BadTernaryCode));
        }
    }
    for bad in [0x09u8, 0x0F, 0x90, 0xF4] {
        let mut img = vec![0x44u8; 32];
        img[7] = bad;
        let r = TLut2Matrix::from_raw(1, 4, img);
        assert_eq!(r.unwrap_err(), KernelError::Core(CoreError::BadTernaryCode));
    }
    assert!(TLut2Matrix::from_raw(1, 4, vec![0x88u8; 32]).is_ok());
}

#[test]
fn rule_n1_and_shapes_are_enforced() {
    assert!(Frames::new(&[0, 128], 2).is_err());
    assert!(Frames::new(&[0, 1, 2], 2).is_err());
    assert!(I8Matrix::new(1, 2, vec![0, -128]).is_err());
    assert!(T1Matrix::from_trits(1, 2, &[0, 2]).is_err());
    assert!(TLut2Matrix::from_trits(1, 2, &[-2, 0]).is_err());
    // N2: 133,145 x 127 x 127 >= 2^31.
    assert!(I8Matrix::new(1, 133_145, vec![127; 133_145]).is_err());
    let m = I8Matrix::new(2, 2, vec![1, 2, 3, 4]).expect("i8");
    let x = Frames::new(&[1, 1], 2).expect("frames");
    assert!(m.view().gemv(Backend::Scalar, &x, &mut [0; 3]).is_err());
    let tl = TLut2Matrix::from_trits(40, 2, &[0; 80]).expect("tlut2");
    assert!(tl.view().slice_rows(16, 40).is_err());
    assert!(tl.view().slice_rows(32, 40).is_ok());
}

#[test]
fn unavailable_backend_is_an_error_not_a_crash() {
    let m = I8Matrix::new(1, 1, vec![1]).expect("i8");
    let x = Frames::new(&[1], 1).expect("frames");
    for b in Backend::ALL {
        let r = m.view().gemv(b, &x, &mut [0]);
        assert_eq!(r.is_ok(), b.is_available(), "{b:?}");
    }
    eprintln!("available backends: {:?}", Backend::available());
}

#[test]
fn tlut2_signed_matches_scalar_and_split_a7() {
    use ultra_kernels::{scalar, Backend, Frames, Gemv, TLut2Matrix};
    let (rows, din, frames) = (70usize, 197usize, 5usize);
    let mut seed = 3u64;
    let mut rnd = || {
        seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        (seed >> 33) as i64
    };
    let trits: Vec<i8> = (0..rows * din).map(|_| (rnd() % 3) as i8 - 1).collect();
    let m = TLut2Matrix::from_trits(rows, din, &trits).expect("tlut2");
    let mut x: Vec<i8> = (0..frames * din)
        .map(|_| (rnd() % 255 - 127) as i8)
        .collect();
    x[0] = 127;
    x[1] = -127;
    let mut want = vec![0i32; rows * frames];
    scalar::gemv_tlut2_signed(m.view(), &x, frames, &mut want);
    // Equal to the exact dot products and to the two-pass A7 split.
    for r in 0..rows {
        for f in 0..frames {
            let d: i32 = (0..din)
                .map(|j| i32::from(trits[r * din + j]) * i32::from(x[f * din + j]))
                .sum();
            assert_eq!(want[r * frames + f], d);
        }
    }
    let pos: Vec<u8> = x.iter().map(|&v| v.max(0) as u8).collect();
    let neg: Vec<u8> = x.iter().map(|&v| (-v).max(0) as u8).collect();
    for b in Backend::available() {
        let mut got = vec![0i32; rows * frames];
        m.view().gemv_signed(b, &x, &mut got).expect("signed");
        assert_eq!(got, want, "{b:?}");
        let (mut p, mut n) = (vec![0i32; rows * frames], vec![0i32; rows * frames]);
        m.view()
            .gemv(b, &Frames::new(&pos, din).expect("pos"), &mut p)
            .expect("p");
        m.view()
            .gemv(b, &Frames::new(&neg, din).expect("neg"), &mut n)
            .expect("n");
        let split: Vec<i32> = p.iter().zip(&n).map(|(a, c)| a - c).collect();
        assert_eq!(split, want, "{b:?} split");
    }
    assert!(m
        .view()
        .gemv_signed(Backend::Scalar, &[i8::MIN; 197], &mut vec![0; rows])
        .is_err());
}
