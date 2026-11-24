//! Fast Zen 2 AVX2 INT8 kernels:
//! - Precomputed `|a|` (`a_abs`) eliminates `vpabsb` from signed inner loops.
//! - Unsigned-offset `U128` (`u = a + 128`) and `U127_NEG` (`u = 127 - a`) register-pinned `M = 4`
//!   microkernels eliminate BOTH `vpabsb` AND `vpsignb` on skewed activations (post-SiLU FFN `w2`
//!   and conv `pw2` layers) using exact per-channel integer sums `w.qsum`.
//! - **Single-Call Sentinel-`bad_ptr` Zero-Spill AVX2 `asm!` Microkernels**: before streaming a
//!   weight matrix's panels, we record the rare pair/quad indices whose `|a|` (or `u`) lane sum
//!   exceeds `258` (`258 * 127 = 32_766 <= i16::MAX`) into a reusable per-thread scratch buffer
//!   (keeping the stack frame `< 2 KB` with zero stack-probe page-touching loops). Each panel
//!   executes in ONE `asm!` call with `ymm0..ymm7` pinned in registers for the entire panel,
//!   halving/quartering `vpmaddwd` throughput pressure on Zen 2's single `FP0` multiply pipe,
//!   branching to out-of-line exact cold labels only on rare overflow-risk indices, and using
//!   `_MM_HINT_NTA` on `rows <= 4` so the KV cache stays L3-resident.
//! - Multi-panel `M = 1, 2` tiling for joiner / tail chunks.
//!
//! Arithmetic is 100% bit-exact (`int32` exact contract) with `scalar::gemm_i8` and `int8::gemm_i8`.

use std::arch::x86_64::*;
use std::cell::RefCell;

use super::super::NR;
use super::int8_fast_prep::{
    epilogue_store, prefer_unsigned, prepare_signed_runs, prepare_u_runs, TileMode, I8_PF_NTA,
    I8_PF_T0,
};
use super::int8_fast_signed::tile_signed_pair_4;
use super::int8_fast_u128::tile_u_pair_4;
use super::int8_fast_u128_quad::tile_u_quad_4;

/// Reusable per-thread scratch (allocated once, stays hot in L1/L2, keeps stack frame `< 512B`
/// with zero `__rust_probestack` page-touching loops on every GEMM call).
#[derive(Default)]
struct GemmScratch {
    u: Vec<u8>,
    prefs: Vec<i8>,
    modes: Vec<TileMode>,
    u_offs: Vec<usize>,
    bad_ranges: Vec<(usize, usize)>,
    bad_idx: Vec<u16>,
}

thread_local! {
    static GEMM_SCRATCH: RefCell<GemmScratch> = RefCell::new(GemmScratch::default());
}

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
    GEMM_SCRATCH.with(|cell| {
        let mut sc = cell.borrow_mut();
        // SAFETY: reached only after Isa::Avx2Fma.is_available() (module docs).
        unsafe { gemm_i8_abs_avx2(w, a, a_abs, sa, ca, rows, out, &mut sc) }
    });
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
    sc: &mut GemmScratch,
) {
    let (n, kp, panels) = (w.n, w.kp, w.panels());
    assert!(rows > 0 && kp.is_multiple_of(4));
    let rk = rows.checked_mul(kp).expect("gemm_i8_with_abs a overflow");
    let rn = rows.checked_mul(n).expect("gemm_i8_with_abs out overflow");
    let panel_elems = kp.checked_mul(NR).expect("gemm_i8_with_abs panel overflow");
    let total_w = panels
        .checked_mul(panel_elems)
        .expect("gemm_i8_with_abs w overflow");
    let panel_meta = panels
        .checked_mul(NR)
        .expect("gemm_i8_with_abs meta overflow");
    assert!(a.len() >= rk && a_abs.len() >= rk);
    assert!(sa.len() >= rows && ca.len() >= rows && out.len() >= rn);
    assert!(
        w.data.len() == total_w
            && w.scale.len() == panel_meta
            && w.wsum.len() == panel_meta
            && w.bias.len() == panel_meta
    );

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

    let has_qsum = w.qsum.len() == panel_meta;
    let stream_nta = rows <= 4;
    let full_tiles = rows / 4;

    if sc.prefs.len() < full_tiles {
        sc.prefs.resize(full_tiles, 0);
    }
    let mut u_tiles = 0usize;
    if has_qsum {
        for ti in 0..full_tiles {
            let r = ti * 4;
            if let Some(neg) = prefer_unsigned(&a[r * kp..(r + 4) * kp], kp, true) {
                sc.prefs[ti] = if neg { -1 } else { 1 };
                u_tiles += 1;
            } else {
                sc.prefs[ti] = 0;
            }
        }
    } else {
        sc.prefs[..full_tiles].fill(0);
    }

    let u_need = u_tiles.saturating_mul(4 * kp);
    if sc.u.len() < u_need {
        sc.u.resize(u_need, 0);
    }
    if sc.modes.len() < full_tiles {
        sc.modes.resize(full_tiles, TileMode::SignedFallback);
    }
    if sc.u_offs.len() < full_tiles {
        sc.u_offs.resize(full_tiles, 0);
    }
    if sc.bad_ranges.len() < full_tiles {
        sc.bad_ranges.resize(full_tiles, (0, 0));
    }
    sc.bad_idx.clear();

    let GemmScratch {
        u: u_buf,
        prefs,
        modes: tile_modes,
        u_offs,
        bad_ranges,
        bad_idx,
    } = sc;

    let mut u_cursor = 0usize;
    for ti in 0..full_tiles {
        let r = ti * 4;
        let ar = &a[r * kp..(r + 4) * kp];
        let aar = &a_abs[r * kp..(r + 4) * kp];
        if prefs[ti] != 0 {
            let neg = prefs[ti] < 0;
            let uoff = u_cursor;
            u_cursor += 4 * kp;
            let ut = &mut u_buf[uoff..uoff + 4 * kp];
            let b0 = bad_idx.len();
            if prepare_u_runs(ar, kp, neg, stream_nta, ut, bad_idx) {
                tile_modes[ti] = if neg {
                    TileMode::U127Neg
                } else {
                    TileMode::U128Pos
                };
                u_offs[ti] = uoff;
                bad_ranges[ti] = (b0, bad_idx.len());
                continue;
            }
        }
        let b0 = bad_idx.len();
        if prepare_signed_runs(aar, kp, bad_idx) {
            tile_modes[ti] = TileMode::SignedPair;
            bad_ranges[ti] = (b0, bad_idx.len());
        } else {
            tile_modes[ti] = TileMode::SignedFallback;
            bad_ranges[ti] = (0, 0);
        }
    }

    for p in 0..panels {
        let panel = w.panel(p);
        let sw = &w.scale[p * NR..(p + 1) * NR];
        let ws = &w.wsum[p * NR..(p + 1) * NR];
        let bias = &w.bias[p * NR..(p + 1) * NR];
        let qs = if has_qsum {
            &w.qsum[p * NR..(p + 1) * NR]
        } else {
            &[]
        };
        let ncols = NR.min(n - p * NR);

        for ti in 0..full_tiles {
            let r = ti * 4;
            let sr = &sa[r..r + 4];
            let cr = &ca[r..r + 4];
            let os = &mut out[r * n + p * NR..];
            let (b0, b1) = bad_ranges[ti];
            let b = &bad_idx[b0..b1];
            match (tile_modes[ti], stream_nta) {
                (TileMode::U128Pos, true) => tile_u_pair_4::<false>(
                    &u_buf[u_offs[ti]..u_offs[ti] + 4 * kp],
                    kp,
                    panel,
                    qs,
                    sw,
                    ws,
                    bias,
                    sr,
                    cr,
                    os,
                    n,
                    ncols,
                    b,
                ),
                (TileMode::U127Neg, true) => tile_u_pair_4::<true>(
                    &u_buf[u_offs[ti]..u_offs[ti] + 4 * kp],
                    kp,
                    panel,
                    qs,
                    sw,
                    ws,
                    bias,
                    sr,
                    cr,
                    os,
                    n,
                    ncols,
                    b,
                ),
                (TileMode::U128Pos, false) => tile_u_quad_4::<false>(
                    &u_buf[u_offs[ti]..u_offs[ti] + 4 * kp],
                    kp,
                    panel,
                    qs,
                    sw,
                    ws,
                    bias,
                    sr,
                    cr,
                    os,
                    n,
                    ncols,
                    b,
                ),
                (TileMode::U127Neg, false) => tile_u_quad_4::<true>(
                    &u_buf[u_offs[ti]..u_offs[ti] + 4 * kp],
                    kp,
                    panel,
                    qs,
                    sw,
                    ws,
                    bias,
                    sr,
                    cr,
                    os,
                    n,
                    ncols,
                    b,
                ),
                (TileMode::SignedPair, true) => tile_signed_pair_4::<true>(
                    &a[r * kp..(r + 4) * kp],
                    &a_abs[r * kp..(r + 4) * kp],
                    kp,
                    panel,
                    sw,
                    ws,
                    bias,
                    sr,
                    cr,
                    os,
                    n,
                    ncols,
                    b,
                ),
                (TileMode::SignedPair, false) => tile_signed_pair_4::<false>(
                    &a[r * kp..(r + 4) * kp],
                    &a_abs[r * kp..(r + 4) * kp],
                    kp,
                    panel,
                    sw,
                    ws,
                    bias,
                    sr,
                    cr,
                    os,
                    n,
                    ncols,
                    b,
                ),
                (TileMode::SignedFallback, true) => tile_abs_small::<4, true>(
                    &a[r * kp..(r + 4) * kp],
                    &a_abs[r * kp..(r + 4) * kp],
                    kp,
                    panel,
                    sw,
                    ws,
                    bias,
                    sr,
                    cr,
                    os,
                    n,
                    ncols,
                ),
                (TileMode::SignedFallback, false) => tile_abs_small::<4, false>(
                    &a[r * kp..(r + 4) * kp],
                    &a_abs[r * kp..(r + 4) * kp],
                    kp,
                    panel,
                    sw,
                    ws,
                    bias,
                    sr,
                    cr,
                    os,
                    n,
                    ncols,
                ),
            }
        }

        let rem_r = full_tiles * 4;
        if rem_r < rows {
            let m = rows - rem_r;
            let ar = &a[rem_r * kp..rows * kp];
            let aar = &a_abs[rem_r * kp..rows * kp];
            let sr = &sa[rem_r..rows];
            let cr = &ca[rem_r..rows];
            let os = &mut out[rem_r * n + p * NR..];
            match (m, stream_nta) {
                (3, true) => tile_abs_small::<3, true>(
                    ar, aar, kp, panel, sw, ws, bias, sr, cr, os, n, ncols,
                ),
                (2, true) => tile_abs_small::<2, true>(
                    ar, aar, kp, panel, sw, ws, bias, sr, cr, os, n, ncols,
                ),
                (_, true) => tile_abs_small::<1, true>(
                    ar, aar, kp, panel, sw, ws, bias, sr, cr, os, n, ncols,
                ),
                (3, false) => tile_abs_small::<3, false>(
                    ar, aar, kp, panel, sw, ws, bias, sr, cr, os, n, ncols,
                ),
                (2, false) => tile_abs_small::<2, false>(
                    ar, aar, kp, panel, sw, ws, bias, sr, cr, os, n, ncols,
                ),
                (_, false) => tile_abs_small::<1, false>(
                    ar, aar, kp, panel, sw, ws, bias, sr, cr, os, n, ncols,
                ),
            }
        }
    }
}

/// Tail `M = 1..=4` rows x 16 cols using precomputed `a_abs` (zero `vpabsb` in inner loop).
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2,fma")]
fn tile_abs_small<const M: usize, const NTA: bool>(
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
    assert!(sa.len() >= M && ca.len() >= M && ncols <= NR);
    assert!((M - 1)
        .checked_mul(ldo)
        .and_then(|v| v.checked_add(ncols))
        .is_some_and(|need| out.len() >= need));
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
            // SAFETY: ps + q < panels and g * 64 + 64 <= stride.
            let bp = unsafe { wp.add(q * stride + g * 64) };
            _mm_prefetch::<_MM_HINT_T0>(bp.wrapping_add(I8_PF_T0).cast());
            for h in 0..2 {
                // SAFETY: g * 64 + h * 32 + 32 <= stride.
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
