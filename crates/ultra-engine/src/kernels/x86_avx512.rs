//! x86-64 AVX-512 VNNI (`vpdpbusd`) int8 GEMM kernel.
//!
//! # Safety model
//! Every `#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni,avx2,fma")]` function
//! here is reached only through the `pub(crate)` wrappers below, which are called only from
//! `kernels/mod.rs` after `Isa::Avx512Vnni.is_available()` returned true at run time. All loads
//! and stores are unaligned (`loadu`/`storeu`) and stay inside slices whose lengths are asserted
//! before the first pointer is formed.
//!
//! # Arithmetic (0-ULP bit-exact to `scalar::gemm_i8` and `x86_avx2::gemm_i8`)
//! `vpdpbusd` multiplies unsigned activation bytes `u in [0, 255]` by signed weight bytes
//! `w in [-127, 127]` and accumulates into wrapping signed `i32` lanes. For any signed `a in
//! [-128, 127]`, `u = (a as u8) ^ 0x80` equals `a + 128` exactly, so:
//!   `sum_k u_k * w_k - 128 * sum_k w_k == sum_k a_k * w_k (mod 2^32)`.
//! Because `k <= MAX_K` guarantees `|sum_k a_k * w_k| <= i32::MAX` and `128 * |sum_k w_k| <=
//! i32::MAX`, the final `i32` difference is exact for all `i8` inputs including `-128`. The
//! epilogue uses the identical FMA order as `scalar::gemm_i8`, so outputs match bit for bit.

use std::arch::x86_64::*;

use super::NR;

/// Rows per AVX-512 VNNI register tile (8 x 16 int32 accumulators = 8 ZMM registers).
const MR512: usize = 8;
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
    // SAFETY: reached only after Isa::Avx512Vnni.is_available() (module docs).
    unsafe { gemm_i8_avx512(w, a, sa, ca, rows, out, 0, w.panels(), w.n) }
}

/// Panels `p0..p1` only, into `out` with row stride `ldo` (column `p0 * 16` at offset 0).
#[allow(clippy::too_many_arguments)]
pub(crate) fn gemm_i8_panels(
    w: &crate::quant::PackedI8,
    a: &[i8],
    sa: &[f32],
    ca: &[f32],
    rows: usize,
    out: &mut [f32],
    p0: usize,
    p1: usize,
    ldo: usize,
) {
    // SAFETY: reached only after Isa::Avx512Vnni.is_available() (module docs).
    unsafe { gemm_i8_avx512(w, a, sa, ca, rows, out, p0, p1, ldo) }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni,avx2,fma")]
#[allow(clippy::too_many_arguments)]
fn gemm_i8_avx512(
    w: &crate::quant::PackedI8,
    a: &[i8],
    sa: &[f32],
    ca: &[f32],
    rows: usize,
    out: &mut [f32],
    p0: usize,
    p1: usize,
    ldo: usize,
) {
    let (n, kp) = (w.n, w.kp);
    let col0 = p0 * NR;
    assert!(p0 < p1 && p1 <= w.panels() && rows > 0);
    let width = (p1 * NR).min(n) - col0;
    assert!(ldo >= width && out.len() >= (rows - 1) * ldo + width);
    assert!(a.len() >= rows * kp && sa.len() >= rows && ca.len() >= rows);
    assert!(
        w.data.len() == w.panels() * kp * NR
            && w.scale.len() == w.panels() * NR
            && w.qsum.len() == w.panels() * NR
    );
    for p in p0..p1 {
        let panel = w.panel(p);
        let sw = &w.scale[p * NR..(p + 1) * NR];
        let ws = &w.wsum[p * NR..(p + 1) * NR];
        let bias = &w.bias[p * NR..(p + 1) * NR];
        let qsum = &w.qsum[p * NR..(p + 1) * NR];
        let ncols = NR.min(n - p * NR);
        let mut r = 0;
        while r < rows {
            let m = (rows - r).min(MR512);
            let ar = &a[r * kp..(r + m) * kp];
            let sr = &sa[r..r + m];
            let cr = &ca[r..r + m];
            let os = &mut out[r * ldo + p * NR - col0..];
            match m {
                8 => tile_vnni::<8>(ar, kp, panel, sw, ws, bias, qsum, sr, cr, os, ldo, ncols),
                7 => tile_vnni::<7>(ar, kp, panel, sw, ws, bias, qsum, sr, cr, os, ldo, ncols),
                6 => tile_vnni::<6>(ar, kp, panel, sw, ws, bias, qsum, sr, cr, os, ldo, ncols),
                5 => tile_vnni::<5>(ar, kp, panel, sw, ws, bias, qsum, sr, cr, os, ldo, ncols),
                4 => tile_vnni::<4>(ar, kp, panel, sw, ws, bias, qsum, sr, cr, os, ldo, ncols),
                3 => tile_vnni::<3>(ar, kp, panel, sw, ws, bias, qsum, sr, cr, os, ldo, ncols),
                2 => tile_vnni::<2>(ar, kp, panel, sw, ws, bias, qsum, sr, cr, os, ldo, ncols),
                _ => tile_vnni::<1>(ar, kp, panel, sw, ws, bias, qsum, sr, cr, os, ldo, ncols),
            }
            r += m;
        }
    }
}

/// `M` rows x 16 columns of int8 GEMM using 512-bit `vpdpbusd` + exact dequantising epilogue.
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni,avx2,fma")]
fn tile_vnni<const M: usize>(
    a: &[i8],
    kp: usize,
    panel: &[i8],
    sw: &[f32],
    wsum: &[f32],
    bias: &[f32],
    qsum: &[i32],
    sa: &[f32],
    ca: &[f32],
    out: &mut [f32],
    ldo: usize,
    ncols: usize,
) {
    assert!(kp.is_multiple_of(4) && a.len() >= M * kp && panel.len() >= kp * NR);
    assert!(
        sw.len() >= NR
            && wsum.len() >= NR
            && bias.len() >= NR
            && qsum.len() >= NR
            && sa.len() >= M
            && ca.len() >= M
    );
    assert!(ncols <= NR && out.len() >= (M - 1) * ldo + ncols);
    const SIGN_FLIP: i32 = 0x8080_8080_u32 as i32;
    let mut acc = [_mm512_setzero_si512(); M];
    let ap = a.as_ptr();
    let wp = panel.as_ptr();
    let groups = kp / 4;
    let mut g = 0;
    while g + 1 < groups {
        _mm_prefetch::<_MM_HINT_T0>(wp.wrapping_add(g * 64 + I8_PREFETCH).cast());
        _mm_prefetch::<_MM_HINT_T0>(wp.wrapping_add(g * 64 + 64 + I8_PREFETCH).cast());
        // SAFETY: (g + 1) * 64 + 64 <= groups * 64 = kp * 16 <= panel.len() (asserted).
        let (w0, w1) = unsafe {
            (
                _mm512_loadu_si512(wp.add(g * 64).cast()),
                _mm512_loadu_si512(wp.add(g * 64 + 64).cast()),
            )
        };
        for (m, acc_m) in acc.iter_mut().enumerate() {
            // SAFETY: m * kp + 4 * g + 8 <= M * kp <= a.len() (asserted); unaligned reads.
            let (a4, b4) = unsafe {
                (
                    std::ptr::read_unaligned(ap.add(m * kp + 4 * g).cast::<i32>()),
                    std::ptr::read_unaligned(ap.add(m * kp + 4 * g + 4).cast::<i32>()),
                )
            };
            let au = _mm512_set1_epi32(a4 ^ SIGN_FLIP);
            let bu = _mm512_set1_epi32(b4 ^ SIGN_FLIP);
            *acc_m = _mm512_dpbusd_epi32(*acc_m, au, w0);
            *acc_m = _mm512_dpbusd_epi32(*acc_m, bu, w1);
        }
        g += 2;
    }
    if g < groups {
        // SAFETY: g * 64 + 64 <= groups * 64 = kp * 16 <= panel.len() (asserted).
        let w0 = unsafe { _mm512_loadu_si512(wp.add(g * 64).cast()) };
        for (m, acc_m) in acc.iter_mut().enumerate() {
            // SAFETY: m * kp + 4 * g + 4 <= M * kp <= a.len() (asserted); unaligned read.
            let a4 = unsafe { std::ptr::read_unaligned(ap.add(m * kp + 4 * g).cast::<i32>()) };
            let au = _mm512_set1_epi32(a4 ^ SIGN_FLIP);
            *acc_m = _mm512_dpbusd_epi32(*acc_m, au, w0);
        }
    }
    // SAFETY: sw, wsum, bias and qsum each hold >= 16 elements (asserted).
    let (sw_v, ws_v, b_v, corr) = unsafe {
        (
            _mm512_loadu_ps(sw.as_ptr()),
            _mm512_loadu_ps(wsum.as_ptr()),
            _mm512_loadu_ps(bias.as_ptr()),
            _mm512_slli_epi32::<7>(_mm512_loadu_si512(qsum.as_ptr().cast())),
        )
    };
    for (m, &acc_m) in acc.iter().enumerate() {
        let exact = _mm512_sub_epi32(acc_m, corr);
        let sv = _mm512_set1_ps(sa[m]);
        let cv = _mm512_set1_ps(ca[m]);
        let base = _mm512_fmadd_ps(cv, ws_v, b_v);
        let y = _mm512_fmadd_ps(_mm512_cvtepi32_ps(exact), _mm512_mul_ps(sv, sw_v), base);
        let row = &mut out[m * ldo..];
        if ncols == NR {
            // SAFETY: row.len() >= 16 when ncols == 16 (asserted bound on out).
            unsafe { _mm512_storeu_ps(row.as_mut_ptr(), y) };
        } else {
            let mut tmp = [0.0f32; NR];
            // SAFETY: tmp holds 16 floats.
            unsafe { _mm512_storeu_ps(tmp.as_mut_ptr(), y) };
            row[..ncols].copy_from_slice(&tmp[..ncols]);
        }
    }
}
