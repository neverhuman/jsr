//! Cache-tight small-M (`M = 1..=4`) int8 GEMV / small-GEMM kernels targeting `c=1` (`80 ms`)
//! low-latency single-stream decode.
//!
//! Tiling across multiple `NR = 16` weight panels keeps 6–8 vector accumulators active in
//! registers, eliminates repeated activation broadcasts/`vpabsb`, and preserves **0-ULP bit-exact
//! parity** with `scalar::gemm_i8` on every ISA.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::Isa;
#[cfg(target_arch = "x86_64")]
use super::NR;
use crate::quant::PackedI8;

/// Largest row count handled by the specialized small-M multi-panel path.
pub const MAX_SMALL_M: usize = 4;

#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy)]
struct Tile<'a> {
    w: &'a PackedI8,
    a: &'a [i8],
    sa: &'a [f32],
    ca: &'a [f32],
}

/// Specialized `M = 1..=4` int8 GEMM, bit-identical to [`super::gemm_i8`].
pub fn gemm_i8_small_m(
    isa: Isa,
    w: &PackedI8,
    a: &[i8],
    sa: &[f32],
    ca: &[f32],
    rows: usize,
    out: &mut [f32],
) {
    assert!((1..=MAX_SMALL_M).contains(&rows), "small_m rows 1..=4");
    assert!(a.len() >= rows * w.kp && out.len() >= rows * w.n);
    assert!(sa.len() >= rows && ca.len() >= rows);
    match isa {
        #[cfg(target_arch = "x86_64")]
        Isa::Avx512Vnni if isa.is_available() => {
            // SAFETY: Isa::Avx512Vnni.is_available() checked above.
            unsafe { gemm_vnni(Tile { w, a, sa, ca }, rows, out) }
        }
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2Fma if isa.is_available() => {
            // SAFETY: Isa::Avx2Fma.is_available() checked above.
            unsafe { gemm_avx2(Tile { w, a, sa, ca }, rows, out) }
        }
        #[cfg(target_arch = "aarch64")]
        Isa::Neon => {
            super::aarch64_neon::gemm_i8_panels(w, a, sa, ca, rows, out, 0, w.panels(), w.n)
        }
        _ => super::scalar::gemm_i8(w, a, sa, ca, rows, out),
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
fn gemm_avx2(t: Tile<'_>, rows: usize, out: &mut [f32]) {
    let panels = t.w.panels();
    if rows >= 3 || panels < 2 {
        super::x86_avx2::gemm_i8(t.w, t.a, t.sa, t.ca, rows, out, false);
        return;
    }
    let mut p = 0;
    while p < panels {
        let step = match (rows, panels - p) {
            (1, 4..) => 4,
            (1 | 2, 2..) => 2,
            _ => 0,
        };
        match (rows, step) {
            (1, 4) => tile_avx2::<1, 4>(t, p, out),
            (1, 2) => tile_avx2::<1, 2>(t, p, out),
            (2, 2) => tile_avx2::<2, 2>(t, p, out),
            _ => {
                let o = &mut out[p * NR..];
                super::x86_avx2::gemm_i8_panels(
                    t.w, t.a, t.sa, t.ca, rows, o, false, p, panels, t.w.n,
                );
                break;
            }
        }
        p += step;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::needless_range_loop)]
fn tile_avx2<const M: usize, const P: usize>(t: Tile<'_>, ps: usize, out: &mut [f32]) {
    let (w, kp, stride) = (t.w, t.w.kp, t.w.kp * NR);
    assert!(kp.is_multiple_of(4) && ps + P <= w.panels() && t.a.len() >= M * kp);
    let ones = _mm256_set1_epi16(1);
    let mut acc = [[[_mm256_setzero_si256(); 2]; M]; P];
    // SAFETY: ps * stride <= w.data.len().
    let (ap, wp) = (t.a.as_ptr(), unsafe { w.data.as_ptr().add(ps * stride) });
    for g in 0..(kp / 4) {
        let (mut av, mut aa) = ([_mm256_setzero_si256(); M], [_mm256_setzero_si256(); M]);
        for m in 0..M {
            // SAFETY: m * kp + 4 * g + 4 <= M * kp <= t.a.len().
            av[m] = _mm256_set1_epi32(unsafe {
                std::ptr::read_unaligned(ap.add(m * kp + 4 * g).cast::<i32>())
            });
            aa[m] = _mm256_abs_epi8(av[m]);
        }
        for q in 0..P {
            let bp = wp.wrapping_add(q * stride + g * 64);
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
        let (off, ncols) = ((ps + q) * NR, NR.min(w.n - (ps + q) * NR));
        for m in 0..M {
            let (sv, cv) = (_mm256_set1_ps(t.sa[m]), _mm256_set1_ps(t.ca[m]));
            let (row, mut tmp) = (&mut out[m * w.n + off..], [0.0f32; NR]);
            let dst = if ncols == NR {
                row.as_mut_ptr()
            } else {
                tmp.as_mut_ptr()
            };
            for h in 0..2 {
                // SAFETY: off + h * 8 + 8 <= panels * 16 == scale/wsum/bias len; dst holds >= 16.
                unsafe {
                    let s = _mm256_loadu_ps(w.scale.as_ptr().add(off + h * 8));
                    let b = _mm256_loadu_ps(w.bias.as_ptr().add(off + h * 8));
                    let u = _mm256_loadu_ps(w.wsum.as_ptr().add(off + h * 8));
                    let y = _mm256_fmadd_ps(
                        _mm256_cvtepi32_ps(acc[q][m][h]),
                        _mm256_mul_ps(sv, s),
                        _mm256_fmadd_ps(cv, u, b),
                    );
                    _mm256_storeu_ps(dst.add(h * 8), y);
                }
            }
            if ncols < NR {
                row[..ncols].copy_from_slice(&tmp[..ncols]);
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni,avx2,fma")]
fn gemm_vnni(t: Tile<'_>, rows: usize, out: &mut [f32]) {
    let panels = t.w.panels();
    let mut p = 0;
    while p < panels {
        let step = match (rows, panels - p) {
            (1, 8..) => 8,
            (1 | 2, 4..) => 4,
            (_, 2..) => 2,
            _ => 0,
        };
        match (rows, step) {
            (1, 8) => tile_vnni::<1, 8>(t, p, out),
            (1, 4) => tile_vnni::<1, 4>(t, p, out),
            (2, 4) => tile_vnni::<2, 4>(t, p, out),
            (1, 2) => tile_vnni::<1, 2>(t, p, out),
            (2, 2) => tile_vnni::<2, 2>(t, p, out),
            (3, 2) => tile_vnni::<3, 2>(t, p, out),
            (4, 2) => tile_vnni::<4, 2>(t, p, out),
            _ => {
                let o = &mut out[p * NR..];
                super::x86_avx512::gemm_i8_panels(t.w, t.a, t.sa, t.ca, rows, o, p, panels, t.w.n);
                break;
            }
        }
        p += step;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni,avx2,fma")]
#[allow(clippy::needless_range_loop)]
fn tile_vnni<const M: usize, const P: usize>(t: Tile<'_>, ps: usize, out: &mut [f32]) {
    const SIGN_FLIP: i32 = 0x8080_8080_u32 as i32;
    let (w, kp, stride) = (t.w, t.w.kp, t.w.kp * NR);
    assert!(kp.is_multiple_of(4) && ps + P <= w.panels() && w.qsum.len() == w.panels() * NR);
    let mut acc = [[_mm512_setzero_si512(); M]; P];
    // SAFETY: ps * stride <= w.data.len().
    let (ap, wp) = (t.a.as_ptr(), unsafe { w.data.as_ptr().add(ps * stride) });
    for g in 0..(kp / 4) {
        let mut au = [_mm512_setzero_si512(); M];
        for m in 0..M {
            // SAFETY: m * kp + 4 * g + 4 <= M * kp <= t.a.len().
            let a4 = unsafe { std::ptr::read_unaligned(ap.add(m * kp + 4 * g).cast::<i32>()) };
            au[m] = _mm512_set1_epi32(a4 ^ SIGN_FLIP);
        }
        for q in 0..P {
            let bp = wp.wrapping_add(q * stride + g * 64);
            _mm_prefetch::<_MM_HINT_T0>(bp.wrapping_add(1024).cast());
            // SAFETY: ps + q < panels and g * 64 + 64 <= stride.
            let wv = unsafe { _mm512_loadu_si512(bp.cast()) };
            for m in 0..M {
                acc[q][m] = _mm512_dpbusd_epi32(acc[q][m], au[m], wv);
            }
        }
    }
    for q in 0..P {
        let (off, ncols) = ((ps + q) * NR, NR.min(w.n - (ps + q) * NR));
        // SAFETY: off + 16 <= panels * 16 == scale/wsum/bias/qsum len.
        let (sw, ws, bv, corr) = unsafe {
            (
                _mm512_loadu_ps(w.scale.as_ptr().add(off)),
                _mm512_loadu_ps(w.wsum.as_ptr().add(off)),
                _mm512_loadu_ps(w.bias.as_ptr().add(off)),
                _mm512_slli_epi32::<7>(_mm512_loadu_si512(w.qsum.as_ptr().add(off).cast())),
            )
        };
        for m in 0..M {
            let ex = _mm512_cvtepi32_ps(_mm512_sub_epi32(acc[q][m], corr));
            let base = _mm512_fmadd_ps(_mm512_set1_ps(t.ca[m]), ws, bv);
            let y = _mm512_fmadd_ps(ex, _mm512_mul_ps(_mm512_set1_ps(t.sa[m]), sw), base);
            let (row, mut tmp) = (&mut out[m * w.n + off..], [0.0f32; NR]);
            let dst = if ncols == NR {
                row.as_mut_ptr()
            } else {
                tmp.as_mut_ptr()
            };
            // SAFETY: dst holds >= 16 floats.
            unsafe { _mm512_storeu_ps(dst, y) };
            if ncols < NR {
                row[..ncols].copy_from_slice(&tmp[..ncols]);
            }
        }
    }
}
