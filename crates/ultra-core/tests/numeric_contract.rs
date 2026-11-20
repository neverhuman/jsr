use proptest::prelude::*;
use ultra_core::numeric::*;
use ultra_core::ternary::*;

#[test]
fn requant_rounds_halves_toward_positive_infinity() {
    // -0.5 -> 0 and -1.5 -> -1 (spec §5.2 item 2), +0.5 -> 1, +1.5 -> 2.
    assert_eq!(requant(-1, 1, 0, 1).unwrap(), 0);
    assert_eq!(requant(-3, 1, 0, 1).unwrap(), -1);
    assert_eq!(requant(1, 1, 0, 1).unwrap(), 1);
    assert_eq!(requant(3, 1, 0, 1).unwrap(), 2);
}

#[test]
fn requant_refuses_shift_zero_and_saturates() {
    assert!(requant(1, 1, 0, 0).is_err());
    assert!(requant(1, 1, 0, 32).is_err());
    assert_eq!(requant(i32::MAX, i32::MAX, 0, 1).unwrap(), i16::MAX);
    assert_eq!(requant(i32::MIN, i32::MAX, 0, 1).unwrap(), i16::MIN);
}

#[test]
fn exp_table_regenerates() {
    for (k, &v) in EXP.iter().enumerate() {
        let want = (65535.0 * (-(k as f64) / 16.0).exp()).round() as u16;
        assert_eq!(v, want, "EXP[{k}]");
    }
    assert_eq!(EXP[0], 65535);
}

#[test]
fn softmax_equal_scores_equal_weights_and_sums_near_one() {
    let mut out = [0u16; 5];
    softmax_q15(&[7, 7, 7, 7, 7], 0, &mut out).unwrap();
    assert!(out.iter().all(|&x| x == out[0]));
    softmax_q15(&[100, -50, 3, 3000, 0], 2, &mut out).unwrap();
    let s: u32 = out.iter().map(|&x| u32::from(x)).sum();
    assert!((32768 - 5..=32768 + 5).contains(&s), "sum {s}");
}

#[test]
fn rms_norm_centres_on_zero_point() {
    let r = [0i16; 8];
    let mut out = [0u8; 8];
    rms_norm(&r, &[1; 8], 20, &mut out).unwrap();
    assert!(out.iter().all(|&x| x == 64));
}

#[test]
fn overflow_prover() {
    assert!(prove_row_no_overflow(4096, 127).is_ok());
    assert!(prove_row_no_overflow(200_000, 127).is_err());
}

#[test]
fn ternary_code_three_refused() {
    assert!(unpack_t1(&[0b11]).is_err());
    assert!(validate_t1(&[0b1100_0000]).is_err());
    assert!(validate_t1(&[0b1010_1010, 0b0101_0101, 0]).is_ok());
}

proptest! {
    #[test]
    fn isqrt_is_exact_floor(n in any::<u64>()) {
        let r = isqrt_u64(n);
        prop_assert!(u128::from(r) * u128::from(r) <= u128::from(n));
        prop_assert!(u128::from(r + 1) * u128::from(r + 1) > u128::from(n));
    }

    #[test]
    fn ternary_roundtrip_and_dot(w in prop::collection::vec(-1i8..=1, 0..64usize).prop_map(|mut v| { v.truncate(v.len() / 4 * 4); v }),
                                 seed in any::<u64>()) {
        let p = pack_t1(&w).unwrap();
        prop_assert!(validate_t1(&p).is_ok());
        prop_assert_eq!(unpack_t1(&p).unwrap(), w.clone());
        let a: Vec<u8> = (0..w.len()).map(|i| ((seed >> (i % 57)) as u8) & 0x7f).collect();
        prop_assert_eq!(dot_t1(&p, &a), dot_i8_a7(&w, &a));
        prop_assert_eq!(row_sum_t1(&p), w.iter().map(|&x| i32::from(x)).sum::<i32>());
    }

    #[test]
    fn a7z_correction_matches_signed_dot(w in prop::collection::vec(-127i8..=127, 1..64), a in prop::collection::vec(0u8..=127, 64)) {
        let a = &a[..w.len()];
        let acc = dot_i8_a7(&w, a);
        let rs: i32 = w.iter().map(|&x| i32::from(x)).sum();
        let truth: i32 = w.iter().zip(a).map(|(&wi, &ai)| i32::from(wi) * (i32::from(ai) - 64)).sum();
        prop_assert_eq!(a7z_correct(acc, rs), truth);
    }
}
