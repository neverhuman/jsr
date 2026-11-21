//! Int8 kernels: per-row centred quantisation and the 4x16 `vpmaddubsw` GEMM tile.
//! Safety model and dispatch: see the parent module docs.

use std::arch::x86_64::*;

use super::super::{scalar, NR};

pub(crate) fn quantize_row(x: &[f32], out: &mut [i8]) -> (f32, f32) {
    // SAFETY: reached only after Isa::Avx2Fma.is_available() (module docs).
    unsafe { quantize_row_avx2(x, out) }
}

/// Eight quantised codes of `x[i..i + 8]` as int32 lanes in [-127, 127].
#[target_feature(enable = "avx2,fma")]
fn q8(x: &[f32], i: usize, c: __m256, iv: __m256) -> __m256i {
    assert!(i + 8 <= x.len());
    // SAFETY: i + 8 <= x.len() (asserted).
    let v = unsafe { _mm256_loadu_ps(x.as_ptr().add(i)) };
    // Default MXCSR rounding: nearest, ties to even (= round_ties_even).
    let r = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_sub_ps(v, c), iv));
    _mm256_min_epi32(
        _mm256_max_epi32(r, _mm256_set1_epi32(-127)),
        _mm256_set1_epi32(127),
    )
}

#[target_feature(enable = "avx2,fma")]
fn quantize_row_avx2(x: &[f32], out: &mut [i8]) -> (f32, f32) {
    assert_eq!(x.len(), out.len());
    let n = x.len();
    let main8 = n - n % 8;
    let (mut lo, mut hi) = (
        _mm256_set1_ps(f32::INFINITY),
        _mm256_set1_ps(f32::NEG_INFINITY),
    );
    let mut i = 0;
    while i < main8 {
        // SAFETY: i + 8 <= n.
        let v = unsafe { _mm256_loadu_ps(x.as_ptr().add(i)) };
        lo = _mm256_min_ps(lo, v);
        hi = _mm256_max_ps(hi, v);
        i += 8;
    }
    let (mut l8, mut h8) = ([0.0f32; 8], [0.0f32; 8]);
    // SAFETY: both arrays hold 8 floats.
    unsafe {
        _mm256_storeu_ps(l8.as_mut_ptr(), lo);
        _mm256_storeu_ps(h8.as_mut_ptr(), hi);
    }
    let (tl, th) = scalar::row_range(&x[main8..]);
    let lo_s = l8.iter().fold(tl, |a, &v| a.min(v));
    let hi_s = h8.iter().fold(th, |a, &v| a.max(v));
    let Some((c, scale, inv)) = scalar::row_params(lo_s, hi_s) else {
        return scalar::finish_row(x, out, lo_s, hi_s);
    };
    let (cv, iv) = (_mm256_set1_ps(c), _mm256_set1_ps(inv));
    let perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    let main32 = n - n % 32;
    let mut i = 0;
    while i < main32 {
        let a = _mm256_packs_epi32(q8(x, i, cv, iv), q8(x, i + 8, cv, iv));
        let b = _mm256_packs_epi32(q8(x, i + 16, cv, iv), q8(x, i + 24, cv, iv));
        let p = _mm256_permutevar8x32_epi32(_mm256_packs_epi16(a, b), perm);
        // SAFETY: i + 32 <= n = out.len().
        unsafe { _mm256_storeu_si256(out.as_mut_ptr().add(i).cast(), p) };
        i += 32;
    }
    for (o, &v) in out[main32..].iter_mut().zip(&x[main32..]) {
        *o = ((v - c) * inv).round_ties_even().clamp(-127.0, 127.0) as i8;
    }
    (scale, c)
}

/// Rows per int8 register tile (4 x 16 int32 accumulators).
const MR8: usize = 4;
/// Prefetch distance (bytes) ahead in the packed int8 weight stream.
const I8_PREFETCH: usize = 2048;

pub(crate) fn gemm_i8(
    w: &crate::quant::PackedI8,
    a: &[i8],
    sa: &[f32],
    ca: &[f32],
    rows: usize,
    out: &mut [f32],
) {
    // SAFETY: reached only after Isa::Avx2Fma.is_available() (module docs).
    unsafe { gemm_i8_avx2(w, a, sa, ca, rows, out) }
}

#[target_feature(enable = "avx2,fma")]
fn gemm_i8_avx2(
    w: &crate::quant::PackedI8,
    a: &[i8],
    sa: &[f32],
    ca: &[f32],
    rows: usize,
    out: &mut [f32],
) {
    let (n, kp) = (w.n, w.kp);
    assert!(a.len() >= rows * kp && sa.len() >= rows && ca.len() >= rows && out.len() >= rows * n);
    assert!(w.data.len() == w.panels() * kp * NR && w.scale.len() == w.panels() * NR);
    for p in 0..w.panels() {
        let panel = w.panel(p);
        let sw = &w.scale[p * NR..(p + 1) * NR];
        let ws = &w.wsum[p * NR..(p + 1) * NR];
        let bias = &w.bias[p * NR..(p + 1) * NR];
        let ncols = NR.min(n - p * NR);
        let mut r = 0;
        while r < rows {
            let m = (rows - r).min(MR8);
            let ar = &a[r * kp..(r + m) * kp];
            let sr = &sa[r..r + m];
            let cr = &ca[r..r + m];
            let os = &mut out[r * n + p * NR..];
            match m {
                4 => tile_i8::<4>(ar, kp, panel, sw, ws, bias, sr, cr, os, n, ncols),
                3 => tile_i8::<3>(ar, kp, panel, sw, ws, bias, sr, cr, os, n, ncols),
                2 => tile_i8::<2>(ar, kp, panel, sw, ws, bias, sr, cr, os, n, ncols),
                _ => tile_i8::<1>(ar, kp, panel, sw, ws, bias, sr, cr, os, n, ncols),
            }
            r += m;
        }
    }
}

/// `M` rows x 16 columns of the int8 GEMM plus the dequantising epilogue.
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2,fma")]
fn tile_i8<const M: usize>(
    a: &[i8],
    kp: usize,
    panel: &[i8],
    sw: &[f32],
    wsum: &[f32],
    bias: &[f32],
    sa: &[f32],
    ca: &[f32],
    out: &mut [f32],
    ldo: usize,
    ncols: usize,
) {
    assert!(kp.is_multiple_of(4) && a.len() >= M * kp && panel.len() >= kp * NR);
    assert!(
        sw.len() >= NR && wsum.len() >= NR && bias.len() >= NR && sa.len() >= M && ca.len() >= M
    );
    assert!(ncols <= NR && out.len() >= (M - 1) * ldo + ncols);
    let ones = _mm256_set1_epi16(1);
    let mut acc = [[_mm256_setzero_si256(); 2]; M];
    let ap = a.as_ptr();
    let wp = panel.as_ptr();
    for g in 0..kp / 4 {
        // Software prefetch of the weight stream (the single-stream case is DRAM-bound):
        // a hint only, never dereferenced, so an address past the panel is harmless.
        _mm_prefetch::<_MM_HINT_T0>(wp.wrapping_add(g * 64 + I8_PREFETCH).cast());
        // SAFETY: g * 64 + 64 <= kp * 16 <= panel.len() (asserted).
        let (w0, w1) = unsafe {
            (
                _mm256_loadu_si256(wp.add(g * 64).cast()),
                _mm256_loadu_si256(wp.add(g * 64 + 32).cast()),
            )
        };
        for (m, acc_m) in acc.iter_mut().enumerate() {
            // SAFETY: m * kp + 4 g + 4 <= M * kp <= a.len() (asserted); unaligned read.
            let a4 = unsafe { std::ptr::read_unaligned(ap.add(m * kp + 4 * g).cast::<i32>()) };
            let av = _mm256_set1_epi32(a4);
            let aa = _mm256_abs_epi8(av);
            let p0 = _mm256_maddubs_epi16(aa, _mm256_sign_epi8(w0, av));
            let p1 = _mm256_maddubs_epi16(aa, _mm256_sign_epi8(w1, av));
            acc_m[0] = _mm256_add_epi32(acc_m[0], _mm256_madd_epi16(p0, ones));
            acc_m[1] = _mm256_add_epi32(acc_m[1], _mm256_madd_epi16(p1, ones));
        }
    }
    // SAFETY: sw, wsum and bias hold >= 16 floats (asserted).
    let (s0, s1, b0, b1, u0, u1) = unsafe {
        (
            _mm256_loadu_ps(sw.as_ptr()),
            _mm256_loadu_ps(sw.as_ptr().add(8)),
            _mm256_loadu_ps(bias.as_ptr()),
            _mm256_loadu_ps(bias.as_ptr().add(8)),
            _mm256_loadu_ps(wsum.as_ptr()),
            _mm256_loadu_ps(wsum.as_ptr().add(8)),
        )
    };
    for (m, acc_m) in acc.iter().enumerate() {
        let sv = _mm256_set1_ps(sa[m]);
        let cv = _mm256_set1_ps(ca[m]);
        let base0 = _mm256_fmadd_ps(cv, u0, b0);
        let base1 = _mm256_fmadd_ps(cv, u1, b1);
        let y0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(acc_m[0]), _mm256_mul_ps(sv, s0), base0);
        let y1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(acc_m[1]), _mm256_mul_ps(sv, s1), base1);
        let row = &mut out[m * ldo..];
        if ncols == NR {
            // SAFETY: row.len() >= 16 when ncols == 16 (asserted bound on out).
            unsafe {
                _mm256_storeu_ps(row.as_mut_ptr(), y0);
                _mm256_storeu_ps(row.as_mut_ptr().add(8), y1);
            }
        } else {
            let mut tmp = [0.0f32; NR];
            // SAFETY: tmp holds 16 floats.
            unsafe {
                _mm256_storeu_ps(tmp.as_mut_ptr(), y0);
                _mm256_storeu_ps(tmp.as_mut_ptr().add(8), y1);
            }
            row[..ncols].copy_from_slice(&tmp[..ncols]);
        }
    }
}
