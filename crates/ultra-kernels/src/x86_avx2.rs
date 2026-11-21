//! AVX2 kernels (spec §5.4 row "x86 AVX2", cadence §5.3 item 2).
//!
//! # Safety model
//! Every `#[target_feature(enable = "avx2")]` function here is reached only through the
//! three `pub(crate)` entry points, which are called only from `Gemv::gemv` after
//! `check_call` has confirmed `Backend::Avx2` (or `Avx512Vnni`, which implies AVX2) is
//! available at run time. Raw-pointer loads and stores are unaligned (`loadu`/`storeu`)
//! and stay inside slices whose lengths are asserted at the top of each function.
//!
//! # Arithmetic
//! Int8 and T1 use `vpmaddubsw` (u8 activation x s8 weight, pairs summed to i16) then
//! `vpmaddwd` by 1 into i32. Rule N1 bounds a pair by 2 x 127 x 127 = 32,258 < 32,767, so
//! the saturating instruction never saturates. T-LUT2 sums table entries (|v| <= 254) in
//! i16 lanes for at most 64 column quads (64 x 508 = 32,512) before widening to i32.

use crate::tlut2::build_tables;
use crate::{Frames, Gemv, I8View, T1View, TLut2View, TLUT2_BLOCK};
use std::arch::x86_64::*;
use ultra_core::numeric::dot_i8_a7;
use ultra_core::ternary::dot_t1;

/// Frames handled together by one pass over a row's (decoded) weights.
const FRAME_BLOCK: usize = 4;
/// T1 SIMD block: 32 packed bytes = 128 weights.
const T1_BLOCK: usize = 128;
/// T-LUT2 column quads summed in i16 before widening (64 x 2 x 254 < 2^15).
const QUADS_PER_FLUSH: usize = 64;

pub(crate) fn gemv_i8(m: I8View<'_>, x: &Frames<'_>, out: &mut [i32]) {
    // SAFETY: only called after check_call confirmed AVX2 is available (module docs).
    unsafe { gemv_i8_avx2(m, x, out) }
}

pub(crate) fn gemv_t1(m: T1View<'_>, x: &Frames<'_>, out: &mut [i32]) {
    // SAFETY: as above.
    unsafe { gemv_t1_avx2(m, x, out) }
}

pub(crate) fn gemv_tlut2(m: TLut2View<'_>, x: &Frames<'_>, out: &mut [i32]) {
    // SAFETY: as above.
    unsafe { gemv_tlut2_avx2(m, x, out) }
}

/// T-LUT2 with prebuilt per-frame tables (`f_n` frames), e.g. from signed activations.
pub(crate) fn gemv_tlut2_tables(
    m: TLut2View<'_>,
    tables: &[[u8; 16]],
    f_n: usize,
    out: &mut [i32],
) {
    // SAFETY: only called after Backend::require() confirmed AVX2 (module docs).
    unsafe { gemv_tlut2_tables_avx2(m, tables, f_n, out) }
}

/// Horizontal sum of eight i32 lanes.
#[target_feature(enable = "avx2")]
fn hsum(v: __m256i) -> i32 {
    let s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256::<1>(v));
    let s = _mm_add_epi32(s, _mm_shuffle_epi32::<0b01_00_11_10>(s));
    let s = _mm_add_epi32(s, _mm_shuffle_epi32::<0b10_11_00_01>(s));
    _mm_cvtsi128_si32(s)
}

/// `sum_{j < n} w[j] * a[f][j]` for `N` frames; `n` is a multiple of 32.
#[target_feature(enable = "avx2")]
fn dot_block<const N: usize>(w: &[i8], a: &[&[u8]], n: usize) -> [i32; N] {
    assert!(n.is_multiple_of(32) && w.len() >= n && a.len() == N);
    assert!(a.iter().all(|s| s.len() >= n));
    let ones = _mm256_set1_epi16(1);
    let ptrs: [*const u8; N] = std::array::from_fn(|f| a[f].as_ptr());
    let mut acc = [_mm256_setzero_si256(); N];
    let mut j = 0;
    while j < n {
        // SAFETY: j + 32 <= n <= w.len() and <= every a[f].len() (asserted above).
        let wv = unsafe { _mm256_loadu_si256(w.as_ptr().add(j).cast()) };
        for (acc_f, p) in acc.iter_mut().zip(ptrs) {
            // SAFETY: as above.
            let av = unsafe { _mm256_loadu_si256(p.add(j).cast()) };
            let prod = _mm256_madd_epi16(_mm256_maddubs_epi16(av, wv), ones);
            *acc_f = _mm256_add_epi32(*acc_f, prod);
        }
        j += 32;
    }
    acc.map(|v| hsum(v))
}

/// All frames against one weight vector (first `n` entries, `n` multiple of 32).
#[target_feature(enable = "avx2")]
fn frames_dot(w: &[i8], acts: &[&[u8]], n: usize, out: &mut [i32]) {
    let mut f = 0;
    while f < acts.len() {
        let take = (acts.len() - f).min(FRAME_BLOCK);
        let a = &acts[f..f + take];
        let o = &mut out[f..f + take];
        match take {
            4 => o.copy_from_slice(&dot_block::<4>(w, a, n)),
            3 => o.copy_from_slice(&dot_block::<3>(w, a, n)),
            2 => o.copy_from_slice(&dot_block::<2>(w, a, n)),
            _ => o.copy_from_slice(&dot_block::<1>(w, a, n)),
        }
        f += take;
    }
}

#[target_feature(enable = "avx2")]
fn gemv_i8_avx2(m: I8View<'_>, x: &Frames<'_>, out: &mut [i32]) {
    let (f_n, din) = (x.count(), m.din());
    let n = din - din % 32;
    let acts: Vec<&[u8]> = (0..f_n).map(|f| x.frame(f)).collect();
    for (r, o) in out.chunks_exact_mut(f_n).enumerate() {
        let w = m.row(r);
        o.fill(0);
        if n > 0 {
            frames_dot(w, &acts, n, o);
        }
        if n < din {
            for (v, a) in o.iter_mut().zip(&acts) {
                *v += dot_i8_a7(&w[n..], &a[n..]);
            }
        }
    }
}

/// Unpack `nb` T1 blocks (32 bytes each) by shift-and-mask into signed bytes. Block `b`
/// position `32 k + i` receives the trit at bits `2k` of byte `i`, i.e. weight `4 i + k`.
#[target_feature(enable = "avx2")]
fn decode_t1(p: &[u8], out: &mut [i8], nb: usize) {
    assert!(p.len() >= nb * 32 && out.len() >= nb * T1_BLOCK);
    let (m3, one) = (_mm256_set1_epi8(3), _mm256_set1_epi8(1));
    for b in 0..nb {
        // SAFETY: 32 * b + 32 <= p.len() (asserted).
        let v = unsafe { _mm256_loadu_si256(p.as_ptr().add(32 * b).cast()) };
        let codes = [
            _mm256_and_si256(v, m3),
            _mm256_and_si256(_mm256_srli_epi16::<2>(v), m3),
            _mm256_and_si256(_mm256_srli_epi16::<4>(v), m3),
            _mm256_and_si256(_mm256_srli_epi16::<6>(v), m3),
        ];
        for (k, c) in codes.into_iter().enumerate() {
            // SAFETY: 128 * b + 32 * k + 32 <= nb * 128 <= out.len() (asserted).
            unsafe {
                let dst = out.as_mut_ptr().add(T1_BLOCK * b + 32 * k);
                _mm256_storeu_si256(dst.cast(), _mm256_sub_epi8(c, one));
            }
        }
    }
}

#[target_feature(enable = "avx2")]
fn gemv_t1_avx2(m: T1View<'_>, x: &Frames<'_>, out: &mut [i32]) {
    let (f_n, din) = (x.count(), m.din());
    let nb = din / T1_BLOCK;
    let n = nb * T1_BLOCK;
    // Activations reordered once per call to match the decode order (weight 4i+k at 32k+i).
    let mut perm = vec![0u8; f_n * n];
    for f in 0..f_n {
        let a = x.frame(f);
        for (j, v) in perm[f * n..(f + 1) * n].iter_mut().enumerate() {
            let (b, k, i) = (j / T1_BLOCK, (j % T1_BLOCK) / 32, j % 32);
            *v = a[T1_BLOCK * b + 4 * i + k];
        }
    }
    let acts: Vec<&[u8]> = (0..f_n).map(|f| &perm[f * n..(f + 1) * n]).collect();
    let mut scratch = vec![0i8; n];
    for (r, o) in out.chunks_exact_mut(f_n).enumerate() {
        let row = m.row(r);
        o.fill(0);
        if n > 0 {
            decode_t1(row, &mut scratch, nb);
            frames_dot(&scratch, &acts, n, o);
        }
        if n < din {
            for (f, v) in o.iter_mut().enumerate() {
                *v += dot_t1(&row[n / 4..], &x.frame(f)[n..]);
            }
        }
    }
}

#[target_feature(enable = "avx2")]
fn gemv_tlut2_avx2(m: TLut2View<'_>, x: &Frames<'_>, out: &mut [i32]) {
    let tables = build_tables(x, m.quads());
    gemv_tlut2_tables_avx2(m, &tables, x.count(), out);
}

#[target_feature(enable = "avx2")]
fn gemv_tlut2_tables_avx2(m: TLut2View<'_>, tables: &[[u8; 16]], f_n: usize, out: &mut [i32]) {
    let (quads, rows) = (m.quads(), m.rows());
    let pairs = 2 * quads;
    assert!(tables.len() >= 2 * pairs * f_n && out.len() >= rows * f_n);
    let per_block = quads * TLUT2_BLOCK;
    let mut res = [0i32; TLUT2_BLOCK];
    for (blk, idx) in m.image().chunks_exact(per_block).enumerate() {
        for f in 0..f_n {
            block_frame(idx, &tables[2 * f * pairs..2 * (f + 1) * pairs], &mut res);
            for (lane, &v) in res.iter().enumerate() {
                let r = blk * TLUT2_BLOCK + lane;
                if r < rows {
                    out[r * f_n + f] = v;
                }
            }
        }
    }
}

/// One 32-row block against one frame's tables (`t[4q..4q+4]` serve quad `q`).
#[target_feature(enable = "avx2")]
fn block_frame(idx: &[u8], t: &[[u8; 16]], res: &mut [i32; TLUT2_BLOCK]) {
    let quads = idx.len() / TLUT2_BLOCK;
    assert!(idx.len() == quads * TLUT2_BLOCK && t.len() == 4 * quads);
    let nib = _mm256_set1_epi8(0x0F);
    let table = |k: usize| {
        // SAFETY: k < t.len() (k = 4q + 0..4 with q < quads); each entry is 16 bytes.
        unsafe { _mm256_broadcastsi128_si256(_mm_loadu_si128(t[k].as_ptr().cast())) }
    };
    let mut acc = [_mm256_setzero_si256(); 4];
    let mut q = 0;
    while q < quads {
        let end = (q + QUADS_PER_FLUSH).min(quads);
        let (mut sa, mut sb) = (_mm256_setzero_si256(), _mm256_setzero_si256());
        for qq in q..end {
            // SAFETY: 32 * qq + 32 <= idx.len() (qq < quads).
            let b = unsafe { _mm256_loadu_si256(idx.as_ptr().add(TLUT2_BLOCK * qq).cast()) };
            let il = _mm256_and_si256(b, nib);
            let ih = _mm256_and_si256(_mm256_srli_epi16::<4>(b), nib);
            let l0 = _mm256_shuffle_epi8(table(4 * qq), il);
            let h0 = _mm256_shuffle_epi8(table(4 * qq + 1), il);
            let l1 = _mm256_shuffle_epi8(table(4 * qq + 2), ih);
            let h1 = _mm256_shuffle_epi8(table(4 * qq + 3), ih);
            let lo = _mm256_add_epi16(_mm256_unpacklo_epi8(l0, h0), _mm256_unpacklo_epi8(l1, h1));
            let hi = _mm256_add_epi16(_mm256_unpackhi_epi8(l0, h0), _mm256_unpackhi_epi8(l1, h1));
            sa = _mm256_add_epi16(sa, lo);
            sb = _mm256_add_epi16(sb, hi);
        }
        let widen = |v: __m128i| _mm256_cvtepi16_epi32(v);
        acc[0] = _mm256_add_epi32(acc[0], widen(_mm256_castsi256_si128(sa)));
        acc[1] = _mm256_add_epi32(acc[1], widen(_mm256_extracti128_si256::<1>(sa)));
        acc[2] = _mm256_add_epi32(acc[2], widen(_mm256_castsi256_si128(sb)));
        acc[3] = _mm256_add_epi32(acc[3], widen(_mm256_extracti128_si256::<1>(sb)));
        q = end;
    }
    // unpacklo/hi work within 128-bit lanes: acc0 = rows 0..8, acc2 = 8..16,
    // acc1 = 16..24, acc3 = 24..32.
    for (dst, v) in [(0, acc[0]), (8, acc[2]), (16, acc[1]), (24, acc[3])] {
        // SAFETY: dst + 8 <= 32 = res.len().
        unsafe { _mm256_storeu_si256(res.as_mut_ptr().add(dst).cast(), v) };
    }
}
