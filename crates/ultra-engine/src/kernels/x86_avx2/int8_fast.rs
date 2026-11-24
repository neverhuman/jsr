//! Fast Zen 2 AVX2 INT8 kernels using precomputed `|a|` (`a_abs`) to eliminate `vpabsb` from
//! the inner GEMM loop, plus multi-panel `M = 1, 2` tiling for joiner / tail chunks.
//!
//! Arithmetic is 100% bit-exact (`int32` exact contract) with `scalar::gemm_i8` and `int8::gemm_i8`.

use std::arch::x86_64::*;

use super::super::NR;

/// Prefetch distances (bytes) ahead in the packed int8 weight stream on Zen 2:
/// - `NTA` (single-pass `rows <= 4` streaming): bypasses L3 victim allocation so the KV cache
///   stays resident in the 16 MiB CCX L3 while weights stream at peak DRAM bandwidth.
/// - `T0` (multi-tile `rows > 4` reuse, e.g. commit `chunk=16`): keeps each panel in L1/L2.
const I8_PF_NTA: usize = 1536;
const I8_PF_T0: usize = 2048;

#[allow(clippy::too_many_arguments)]
pub(crate) fn gemm_i8_with_abs(
    w: &crate::quant::PackedI8,
    a: &[i8],
    a_abs: &[u8],
    sa: &[f32],
    ca: &[f32],
    rows: usize,
    out: &mut [f32],
) {
    // SAFETY: reached only after Isa::Avx2Fma.is_available() (module docs).
    unsafe { gemm_i8_abs_avx2(w, a, a_abs, sa, ca, rows, out) }
}

#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
fn gemm_i8_abs_avx2(
    w: &crate::quant::PackedI8,
    a: &[i8],
    a_abs: &[u8],
    sa: &[f32],
    ca: &[f32],
    rows: usize,
    out: &mut [f32],
) {
    let (n, kp, panels) = (w.n, w.kp, w.panels());
    assert!(rows > 0 && kp.is_multiple_of(4));
    assert!(a.len() >= rows * kp && a_abs.len() >= rows * kp);
    assert!(sa.len() >= rows && ca.len() >= rows && out.len() >= rows * n);
    assert!(w.data.len() == panels * kp * NR && w.scale.len() == panels * NR);

    // Multi-panel path for M = 1, 2 (joiner_out, partial tail chunks).
    if rows <= 2 && panels >= 2 {
        let mut p = 0;
        while p < panels {
            let step = match (rows, panels - p) {
                (1, 4..) => 4,
                (1 | 2, 2..) => 2,
                _ => 1,
            };
            match (rows, step) {
                (1, 4) => tile_mp_abs::<1, 4>(w, a, a_abs, sa, ca, p, out),
                (1, 2) => tile_mp_abs::<1, 2>(w, a, a_abs, sa, ca, p, out),
                (2, 2) => tile_mp_abs::<2, 2>(w, a, a_abs, sa, ca, p, out),
                (2, _) => tile_mp_abs::<2, 1>(w, a, a_abs, sa, ca, p, out),
                _ => tile_mp_abs::<1, 1>(w, a, a_abs, sa, ca, p, out),
            }
            p += step;
        }
        return;
    }

    let stream_nta = rows <= 4;
    for p in 0..panels {
        let panel = w.panel(p);
        let sw = &w.scale[p * NR..(p + 1) * NR];
        let ws = &w.wsum[p * NR..(p + 1) * NR];
        let bias = &w.bias[p * NR..(p + 1) * NR];
        let ncols = NR.min(n - p * NR);
        let mut r = 0;
        while r < rows {
            let m = (rows - r).min(4);
            let ar = &a[r * kp..(r + m) * kp];
            let aar = &a_abs[r * kp..(r + m) * kp];
            let sr = &sa[r..r + m];
            let cr = &ca[r..r + m];
            let os = &mut out[r * n + p * NR..];
            match (m, stream_nta) {
                (4, true) => {
                    tile_abs::<4, true>(ar, aar, kp, panel, sw, ws, bias, sr, cr, os, n, ncols)
                }
                (3, true) => {
                    tile_abs::<3, true>(ar, aar, kp, panel, sw, ws, bias, sr, cr, os, n, ncols)
                }
                (2, true) => {
                    tile_abs::<2, true>(ar, aar, kp, panel, sw, ws, bias, sr, cr, os, n, ncols)
                }
                (_, true) => {
                    tile_abs::<1, true>(ar, aar, kp, panel, sw, ws, bias, sr, cr, os, n, ncols)
                }
                (4, false) => {
                    tile_abs::<4, false>(ar, aar, kp, panel, sw, ws, bias, sr, cr, os, n, ncols)
                }
                (3, false) => {
                    tile_abs::<3, false>(ar, aar, kp, panel, sw, ws, bias, sr, cr, os, n, ncols)
                }
                (2, false) => {
                    tile_abs::<2, false>(ar, aar, kp, panel, sw, ws, bias, sr, cr, os, n, ncols)
                }
                (_, false) => {
                    tile_abs::<1, false>(ar, aar, kp, panel, sw, ws, bias, sr, cr, os, n, ncols)
                }
            }
            r += m;
        }
    }
}

#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
fn epilogue_store<const M: usize>(
    acc: &[[__m256i; 2]; M],
    sw: &[f32],
    wsum: &[f32],
    bias: &[f32],
    sa: &[f32],
    ca: &[f32],
    out: &mut [f32],
    ldo: usize,
    ncols: usize,
) {
    // SAFETY: sw, wsum, bias have >= 16 floats.
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
            // SAFETY: row.len() >= 16 when ncols == 16.
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

/// `M` rows x 16 cols using precomputed `a_abs` (zero `vpabsb` in inner loop).
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2,fma")]
fn tile_abs<const M: usize, const NTA: bool>(
    a: &[i8],
    a_abs: &[u8],
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
    assert!(kp.is_multiple_of(4) && a.len() >= M * kp && a_abs.len() >= M * kp);
    assert!(panel.len() >= kp * NR && sw.len() >= NR && wsum.len() >= NR && bias.len() >= NR);
    assert!(sa.len() >= M && ca.len() >= M && ncols <= NR && out.len() >= (M - 1) * ldo + ncols);
    let ones = _mm256_set1_epi16(1);
    let mut acc = [[_mm256_setzero_si256(); 2]; M];
    let (ap, aap, wp) = (a.as_ptr(), a_abs.as_ptr(), panel.as_ptr());
    let groups = kp / 4;
    for g in 0..groups {
        if NTA {
            _mm_prefetch::<_MM_HINT_NTA>(wp.wrapping_add(g * 64 + I8_PF_NTA).cast());
        } else {
            _mm_prefetch::<_MM_HINT_T0>(wp.wrapping_add(g * 64 + I8_PF_T0).cast());
        }
        // SAFETY: g * 64 + 64 <= kp * 16 <= panel.len().
        let (w0, w1) = unsafe {
            (
                _mm256_loadu_si256(wp.add(g * 64).cast()),
                _mm256_loadu_si256(wp.add(g * 64 + 32).cast()),
            )
        };
        for (m, acc_m) in acc.iter_mut().enumerate() {
            // SAFETY: m * kp + 4 * g + 4 <= M * kp <= a.len() = a_abs.len().
            let (a4, aa4) = unsafe {
                (
                    std::ptr::read_unaligned(ap.add(m * kp + 4 * g).cast::<i32>()),
                    std::ptr::read_unaligned(aap.add(m * kp + 4 * g).cast::<i32>()),
                )
            };
            let av = _mm256_set1_epi32(a4);
            let aa = _mm256_set1_epi32(aa4);
            let p0 = _mm256_maddubs_epi16(aa, _mm256_sign_epi8(w0, av));
            let p1 = _mm256_maddubs_epi16(aa, _mm256_sign_epi8(w1, av));
            acc_m[0] = _mm256_add_epi32(acc_m[0], _mm256_madd_epi16(p0, ones));
            acc_m[1] = _mm256_add_epi32(acc_m[1], _mm256_madd_epi16(p1, ones));
        }
    }
    epilogue_store(&acc, sw, wsum, bias, sa, ca, out, ldo, ncols);
}

/// Multi-panel (`P` panels of 16 cols) tile for small `M = 1, 2` with precomputed `a_abs`.
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
#[target_feature(enable = "avx2,fma")]
fn tile_mp_abs<const M: usize, const P: usize>(
    w: &crate::quant::PackedI8,
    a: &[i8],
    a_abs: &[u8],
    sa: &[f32],
    ca: &[f32],
    ps: usize,
    out: &mut [f32],
) {
    let (kp, stride) = (w.kp, w.kp * NR);
    assert!(ps + P <= w.panels() && a.len() >= M * kp && a_abs.len() >= M * kp);
    let ones = _mm256_set1_epi16(1);
    let mut acc = [[[_mm256_setzero_si256(); 2]; M]; P];
    // SAFETY: ps * stride <= w.data.len().
    let wp = unsafe { w.data.as_ptr().add(ps * stride) };
    let (ap, aap) = (a.as_ptr(), a_abs.as_ptr());
    for g in 0..(kp / 4) {
        let mut av = [_mm256_setzero_si256(); M];
        let mut aa = [_mm256_setzero_si256(); M];
        for m in 0..M {
            // SAFETY: m * kp + 4 * g + 4 <= M * kp <= a.len() = a_abs.len().
            unsafe {
                av[m] = _mm256_set1_epi32(std::ptr::read_unaligned(
                    ap.add(m * kp + 4 * g).cast::<i32>(),
                ));
                aa[m] = _mm256_set1_epi32(std::ptr::read_unaligned(
                    aap.add(m * kp + 4 * g).cast::<i32>(),
                ));
            }
        }
        for q in 0..P {
            let bp = wp.wrapping_add(q * stride + g * 64);
            _mm_prefetch::<_MM_HINT_T0>(bp.wrapping_add(I8_PF_T0).cast());
            for h in 0..2 {
                // SAFETY: ps + q < panels and g * 64 + h * 32 + 32 <= stride.
                let wv = unsafe { _mm256_loadu_si256(bp.add(h * 32).cast()) };
                for m in 0..M {
                    let pr = _mm256_maddubs_epi16(aa[m], _mm256_sign_epi8(wv, av[m]));
                    acc[q][m][h] = _mm256_add_epi32(acc[q][m][h], _mm256_madd_epi16(pr, ones));
                }
            }
        }
    }
    for q in 0..P {
        let off = (ps + q) * NR;
        let ncols = NR.min(w.n - off);
        epilogue_store(
            &acc[q],
            &w.scale[off..off + NR],
            &w.wsum[off..off + NR],
            &w.bias[off..off + NR],
            sa,
            ca,
            &mut out[off..],
            w.n,
            ncols,
        );
    }
}
