//! Bit-exact AVX2 elementwise & convolution kernels: unfused RoPE pairs, register-blocked
//! causal depthwise convolution, and register-blocked stem layer-1 3x3 conv + SiLU.
//!
//! Every kernel matches its scalar IEEE-754 rounding order bit for bit (0-ULP exact).

use std::arch::x86_64::*;

use super::super::scalar;
use super::{exp8, neg8};

pub(crate) fn rope_pair(v: &mut [f32], cs: &[f32]) {
    // SAFETY: reached only after Isa::Avx2Fma.is_available() (module docs).
    unsafe { rope_pair_avx2(v, cs) }
}

/// Exact unfused RoPE on pairs `(2i, 2i+1)`: `x1*c - x2*s`, `x1*s + x2*c` (0-ULP bit-identical).
#[target_feature(enable = "avx2,fma")]
fn rope_pair_avx2(v: &mut [f32], cs: &[f32]) {
    let half = v.len() / 2;
    assert_eq!(v.len(), 2 * half);
    assert!(cs.len() >= 2 * half);
    let main = half - half % 8;
    let mut i = 0;
    while i < main {
        // SAFETY: 2*i + 16 <= v.len() and half + i + 8 <= cs.len().
        unsafe {
            let a = _mm256_loadu_ps(v.as_ptr().add(2 * i));
            let b = _mm256_loadu_ps(v.as_ptr().add(2 * i + 8));
            let even_raw = _mm256_shuffle_ps::<0x88>(a, b);
            let odd_raw = _mm256_shuffle_ps::<0xDD>(a, b);
            let x1 = _mm256_castsi256_ps(_mm256_permute4x64_epi64::<0xD8>(_mm256_castps_si256(
                even_raw,
            )));
            let x2 = _mm256_castsi256_ps(_mm256_permute4x64_epi64::<0xD8>(_mm256_castps_si256(
                odd_raw,
            )));
            let c = _mm256_loadu_ps(cs.as_ptr().add(i));
            let s = _mm256_loadu_ps(cs.as_ptr().add(half + i));
            // Unfused mul + sub/add matches torch / scalar::rope bit for bit (no FMA contraction).
            let y1 = _mm256_sub_ps(_mm256_mul_ps(x1, c), _mm256_mul_ps(x2, s));
            let y2 = _mm256_add_ps(_mm256_mul_ps(x1, s), _mm256_mul_ps(x2, c));
            let lo = _mm256_unpacklo_ps(y1, y2);
            let hi = _mm256_unpackhi_ps(y1, y2);
            let out0 = _mm256_permute2f128_ps::<0x20>(lo, hi);
            let out1 = _mm256_permute2f128_ps::<0x31>(lo, hi);
            _mm256_storeu_ps(v.as_mut_ptr().add(2 * i), out0);
            _mm256_storeu_ps(v.as_mut_ptr().add(2 * i + 8), out1);
        }
        i += 8;
    }
    for j in main..half {
        let (c, s) = (cs[j], cs[half + j]);
        let (x1, x2) = (v[2 * j], v[2 * j + 1]);
        v[2 * j] = x1 * c - x2 * s;
        v[2 * j + 1] = x1 * s + x2 * c;
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn depthwise_conv_frame(
    out: &mut [f32],
    bias: &[f32],
    hist: &[f32],
    cur: &[f32],
    dw_w: &[f32],
    frame_i: usize,
    kernel: usize,
    d: usize,
) {
    // SAFETY: reached only after Isa::Avx2Fma.is_available() (module docs).
    unsafe { depthwise_conv_frame_avx2(out, bias, hist, cur, dw_w, frame_i, kernel, d) }
}

/// Register-blocked causal depthwise conv for one frame across all `kernel` taps in exact order
/// `t = 0..kernel` with unfused `add(acc, mul(w, src))` (0-ULP bit-identical to `mul_add_exact`).
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2,fma")]
fn depthwise_conv_frame_avx2(
    out: &mut [f32],
    bias: &[f32],
    hist: &[f32],
    cur: &[f32],
    dw_w: &[f32],
    frame_i: usize,
    kernel: usize,
    d: usize,
) {
    let km1 = kernel - 1;
    assert!(out.len() >= d && bias.len() >= d && dw_w.len() >= kernel * d);
    assert!(hist.len() >= km1 * d && cur.len() >= (frame_i + 1) * d);
    let split = km1.saturating_sub(frame_i).min(kernel);
    let mut c = 0;
    while c + 64 <= d {
        // SAFETY: c + 64 <= d for bias, out, hist rows, cur rows, dw_w rows.
        unsafe {
            let bp = bias.as_ptr().add(c);
            let mut acc = [
                _mm256_loadu_ps(bp),
                _mm256_loadu_ps(bp.add(8)),
                _mm256_loadu_ps(bp.add(16)),
                _mm256_loadu_ps(bp.add(24)),
                _mm256_loadu_ps(bp.add(32)),
                _mm256_loadu_ps(bp.add(40)),
                _mm256_loadu_ps(bp.add(48)),
                _mm256_loadu_ps(bp.add(56)),
            ];
            let mut wp = dw_w.as_ptr().add(c);
            let mut sp = hist.as_ptr().add(frame_i * d + c);
            for _ in 0..split {
                for (u, a) in acc.iter_mut().enumerate() {
                    let off = u * 8;
                    *a = _mm256_add_ps(
                        *a,
                        _mm256_mul_ps(_mm256_loadu_ps(wp.add(off)), _mm256_loadu_ps(sp.add(off))),
                    );
                }
                wp = wp.add(d);
                sp = sp.add(d);
            }
            let mut cp = cur.as_ptr().add((frame_i + split - km1) * d + c);
            for _ in split..kernel {
                for (u, a) in acc.iter_mut().enumerate() {
                    let off = u * 8;
                    *a = _mm256_add_ps(
                        *a,
                        _mm256_mul_ps(_mm256_loadu_ps(wp.add(off)), _mm256_loadu_ps(cp.add(off))),
                    );
                }
                wp = wp.add(d);
                cp = cp.add(d);
            }
            let op = out.as_mut_ptr().add(c);
            for (u, &a) in acc.iter().enumerate() {
                _mm256_storeu_ps(op.add(u * 8), a);
            }
        }
        c += 64;
    }
    while c + 8 <= d {
        // SAFETY: c + 8 <= d.
        unsafe {
            let mut a = _mm256_loadu_ps(bias.as_ptr().add(c));
            for t in 0..kernel {
                let pos = frame_i + t;
                let sp = if pos < km1 {
                    hist.as_ptr().add(pos * d + c)
                } else {
                    cur.as_ptr().add((pos - km1) * d + c)
                };
                let wp = dw_w.as_ptr().add(t * d + c);
                a = _mm256_add_ps(a, _mm256_mul_ps(_mm256_loadu_ps(wp), _mm256_loadu_ps(sp)));
            }
            _mm256_storeu_ps(out.as_mut_ptr().add(c), a);
        }
        c += 8;
    }
    for j in c..d {
        let mut a = bias[j];
        for t in 0..kernel {
            let pos = frame_i + t;
            let sv = if pos < km1 {
                hist[pos * d + j]
            } else {
                cur[(pos - km1) * d + j]
            };
            a += dw_w[t * d + j] * sv;
        }
        out[j] = a;
    }
}

pub(crate) fn stem_c1_pos(out: &mut [f32], bias: &[f32], w: &[f32], x9: [f32; 9], ch: usize) {
    // SAFETY: reached only after Isa::Avx2Fma.is_available() (module docs).
    unsafe { stem_c1_pos_avx2(out, bias, w, x9, ch) }
}

/// Register-blocked 9-tap stem layer-1 conv + exact SiLU (0-ULP bit-identical to scalar + silu).
#[target_feature(enable = "avx2,fma")]
fn stem_c1_pos_avx2(out: &mut [f32], bias: &[f32], w: &[f32], x9: [f32; 9], ch: usize) {
    assert!(out.len() >= ch && bias.len() >= ch && w.len() >= 9 * ch);
    let one = _mm256_set1_ps(1.0);
    let mut c = 0;
    while c + 64 <= ch {
        // SAFETY: c + 64 <= ch for bias, out, and each tap row of w.
        unsafe {
            let bp = bias.as_ptr().add(c);
            let mut acc = [
                _mm256_loadu_ps(bp),
                _mm256_loadu_ps(bp.add(8)),
                _mm256_loadu_ps(bp.add(16)),
                _mm256_loadu_ps(bp.add(24)),
                _mm256_loadu_ps(bp.add(32)),
                _mm256_loadu_ps(bp.add(40)),
                _mm256_loadu_ps(bp.add(48)),
                _mm256_loadu_ps(bp.add(56)),
            ];
            for (t, &x) in x9.iter().enumerate() {
                let xv = _mm256_set1_ps(x);
                let wp = w.as_ptr().add(t * ch + c);
                for (u, a) in acc.iter_mut().enumerate() {
                    let wv = _mm256_loadu_ps(wp.add(u * 8));
                    *a = _mm256_add_ps(*a, _mm256_mul_ps(wv, xv));
                }
            }
            let op = out.as_mut_ptr().add(c);
            for (u, &v) in acc.iter().enumerate() {
                let y = _mm256_div_ps(v, _mm256_add_ps(one, exp8(neg8(v))));
                _mm256_storeu_ps(op.add(u * 8), y);
            }
        }
        c += 64;
    }
    while c + 8 <= ch {
        // SAFETY: c + 8 <= ch.
        unsafe {
            let mut v = _mm256_loadu_ps(bias.as_ptr().add(c));
            for (t, &x) in x9.iter().enumerate() {
                let wv = _mm256_loadu_ps(w.as_ptr().add(t * ch + c));
                v = _mm256_add_ps(v, _mm256_mul_ps(wv, _mm256_set1_ps(x)));
            }
            let y = _mm256_div_ps(v, _mm256_add_ps(one, exp8(neg8(v))));
            _mm256_storeu_ps(out.as_mut_ptr().add(c), y);
        }
        c += 8;
    }
    for j in c..ch {
        let mut v = bias[j];
        for (t, &x) in x9.iter().enumerate() {
            v += w[t * ch + j] * x;
        }
        out[j] = scalar::silu(v);
    }
}
