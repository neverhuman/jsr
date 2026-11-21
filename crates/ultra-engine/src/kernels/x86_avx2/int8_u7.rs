//! EXPERIMENTAL (H15 B2, not used by the engine): the unsigned-offset 7-bit int8 tile.
//!
//! Activation codes are the `act7` codes `a` (|a| <= 63) stored offset as `u = a + 64` in
//! 1..=127, so `vpmaddubsw` takes them as its unsigned operand directly: no `vpabsb` and no
//! `vpsignb`. A pair sum is at most 2 * 127 * 127 = 32 258, so the i16 step never saturates.
//! The offset is removed exactly in int32: `sum_k a w = sum_k u w - 64 * colsum[n]`, with
//! `colsum[n] = sum_k w_q[n][k]` (zero padding weights add nothing). The int32 sums, and so
//! the outputs (same epilogue), equal the `act7` path bit for bit.
//!
//! Per 64 MACs and row: 1 broadcast, 2 `vpmaddubsw`, 2 `vpmaddwd`, 2 `vpaddd` (7 ops) versus
//! 10 for the signed tile.

use std::arch::x86_64::*;

use super::NR;

/// Rows per register tile.
const MR8: usize = 4;
const I8_PREFETCH: usize = 2048;

/// `u` holds `rows x kp` offset codes (`a + 64`, padding `64`); `colsum` has `n` entries.
#[allow(clippy::too_many_arguments)]
pub(crate) fn gemm_i8_u7(
    w: &crate::quant::PackedI8,
    colsum: &[i32],
    u: &[u8],
    sa: &[f32],
    ca: &[f32],
    rows: usize,
    out: &mut [f32],
) {
    // SAFETY: reached only after Isa::Avx2Fma.is_available() (module docs).
    unsafe { gemm_i8_u7_avx2(w, colsum, u, sa, ca, rows, out) }
}

#[target_feature(enable = "avx2,fma")]
fn gemm_i8_u7_avx2(
    w: &crate::quant::PackedI8,
    colsum: &[i32],
    u: &[u8],
    sa: &[f32],
    ca: &[f32],
    rows: usize,
    out: &mut [f32],
) {
    let (n, kp) = (w.n, w.kp);
    assert!(rows > 0 && u.len() >= rows * kp && sa.len() >= rows && ca.len() >= rows);
    assert!(out.len() >= rows * n && colsum.len() >= n);
    assert!(w.data.len() == w.panels() * kp * NR && w.scale.len() == w.panels() * NR);
    let mut cs = [0i32; NR];
    for p in 0..w.panels() {
        let ncols = NR.min(n - p * NR);
        cs[..ncols].copy_from_slice(&colsum[p * NR..p * NR + ncols]);
        cs[ncols..].fill(0);
        let mut r = 0;
        while r < rows {
            let m = (rows - r).min(MR8);
            let args = (
                &u[r * kp..(r + m) * kp],
                kp,
                w.panel(p),
                &cs,
                &w.scale[p * NR..(p + 1) * NR],
                &w.wsum[p * NR..(p + 1) * NR],
                &w.bias[p * NR..(p + 1) * NR],
                &sa[r..r + m],
                &ca[r..r + m],
            );
            let os = &mut out[r * n + p * NR..];
            match m {
                4 => tile_u7::<4>(args, os, n, ncols),
                3 => tile_u7::<3>(args, os, n, ncols),
                2 => tile_u7::<2>(args, os, n, ncols),
                _ => tile_u7::<1>(args, os, n, ncols),
            }
            r += m;
        }
    }
}

type TileArgs<'a> = (
    &'a [u8],
    usize,
    &'a [i8],
    &'a [i32; NR],
    &'a [f32],
    &'a [f32],
    &'a [f32],
    &'a [f32],
    &'a [f32],
);

#[target_feature(enable = "avx2,fma")]
fn tile_u7<const M: usize>(t: TileArgs<'_>, out: &mut [f32], ldo: usize, ncols: usize) {
    let (u, kp, panel, cs, sw, wsum, bias, sa, ca) = t;
    assert!(kp.is_multiple_of(4) && u.len() >= M * kp && panel.len() >= kp * NR);
    assert!(sw.len() >= NR && wsum.len() >= NR && bias.len() >= NR);
    assert!(sa.len() >= M && ca.len() >= M && ncols <= NR && out.len() >= (M - 1) * ldo + ncols);
    let ones = _mm256_set1_epi16(1);
    let mut acc = [[_mm256_setzero_si256(); 2]; M];
    let (up, wp) = (u.as_ptr(), panel.as_ptr());
    for g in 0..kp / 4 {
        _mm_prefetch::<_MM_HINT_T0>(wp.wrapping_add(g * 64 + I8_PREFETCH).cast());
        // SAFETY: g * 64 + 64 <= kp * 16 <= panel.len() (asserted).
        let (w0, w1) = unsafe {
            (
                _mm256_loadu_si256(wp.add(g * 64).cast()),
                _mm256_loadu_si256(wp.add(g * 64 + 32).cast()),
            )
        };
        for (m, acc_m) in acc.iter_mut().enumerate() {
            // SAFETY: m * kp + 4 g + 4 <= M * kp <= u.len() (asserted); unaligned read.
            let u4 = unsafe { std::ptr::read_unaligned(up.add(m * kp + 4 * g).cast::<i32>()) };
            let uv = _mm256_set1_epi32(u4);
            let p0 = _mm256_maddubs_epi16(uv, w0);
            let p1 = _mm256_maddubs_epi16(uv, w1);
            acc_m[0] = _mm256_add_epi32(acc_m[0], _mm256_madd_epi16(p0, ones));
            acc_m[1] = _mm256_add_epi32(acc_m[1], _mm256_madd_epi16(p1, ones));
        }
    }
    // SAFETY: cs, sw, wsum and bias hold >= 16 values (asserted / array type).
    let (c0, c1, s0, s1, b0, b1, w0s, w1s) = unsafe {
        (
            _mm256_slli_epi32::<6>(_mm256_loadu_si256(cs.as_ptr().cast())),
            _mm256_slli_epi32::<6>(_mm256_loadu_si256(cs.as_ptr().add(8).cast())),
            _mm256_loadu_ps(sw.as_ptr()),
            _mm256_loadu_ps(sw.as_ptr().add(8)),
            _mm256_loadu_ps(bias.as_ptr()),
            _mm256_loadu_ps(bias.as_ptr().add(8)),
            _mm256_loadu_ps(wsum.as_ptr()),
            _mm256_loadu_ps(wsum.as_ptr().add(8)),
        )
    };
    for (m, acc_m) in acc.iter().enumerate() {
        // Remove the offset: sum (a + 64) w - 64 colsum.
        let a0 = _mm256_sub_epi32(acc_m[0], c0);
        let a1 = _mm256_sub_epi32(acc_m[1], c1);
        let sv = _mm256_set1_ps(sa[m]);
        let cv = _mm256_set1_ps(ca[m]);
        let y0 = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(a0),
            _mm256_mul_ps(sv, s0),
            _mm256_fmadd_ps(cv, w0s, b0),
        );
        let y1 = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(a1),
            _mm256_mul_ps(sv, s1),
            _mm256_fmadd_ps(cv, w1s, b1),
        );
        let mut tmp = [0.0f32; NR];
        // SAFETY: tmp holds 16 floats.
        unsafe {
            _mm256_storeu_ps(tmp.as_mut_ptr(), y0);
            _mm256_storeu_ps(tmp.as_mut_ptr().add(8), y1);
        }
        out[m * ldo..m * ldo + ncols].copy_from_slice(&tmp[..ncols]);
    }
}
