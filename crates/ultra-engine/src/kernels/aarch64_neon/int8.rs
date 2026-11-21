//! Int8 kernels: per-row centred quantisation (bit-identical to the scalar and AVX2 paths)
//! and the integer GEMM with the dequantising epilogue. The products are exact integers, so
//! the `sdot` path (ARMv8.2 dot product, used when `dotprod` is detected) and the baseline
//! widening path (`smull` + pairwise adds) give identical results.
//! Safety model: see the parent module docs.

use std::arch::aarch64::*;
use std::sync::OnceLock;

use super::super::{scalar, NR};

pub(crate) fn quantize_row(x: &[f32], out: &mut [i8], qmax: f32) -> (f32, f32) {
    assert_eq!(x.len(), out.len());
    let (lo, hi) = scalar::row_range(x);
    let Some((c, scale, inv)) = scalar::row_params(lo, hi, qmax) else {
        return scalar::finish_row(x, out, lo, hi, qmax);
    };
    let qm = qmax as i32;
    let main = x.len() - x.len() % 4;
    let mut i = 0;
    while i < main {
        // SAFETY: i + 4 <= x.len() = out.len(); register ops otherwise.
        unsafe {
            let v = vld1q_f32(x.as_ptr().add(i));
            let t = vmulq_f32(vsubq_f32(v, vdupq_n_f32(c)), vdupq_n_f32(inv));
            // Round to nearest, ties to even (= round_ties_even / AVX2 cvtps with default MXCSR).
            let r = vcvtnq_s32_f32(t);
            let r = vminq_s32(vmaxq_s32(r, vdupq_n_s32(-qm)), vdupq_n_s32(qm));
            for (l, o) in out[i..i + 4].iter_mut().enumerate() {
                *o = match l {
                    0 => vgetq_lane_s32::<0>(r),
                    1 => vgetq_lane_s32::<1>(r),
                    2 => vgetq_lane_s32::<2>(r),
                    _ => vgetq_lane_s32::<3>(r),
                } as i8;
            }
        }
        i += 4;
    }
    for (o, &v) in out[main..].iter_mut().zip(&x[main..]) {
        *o = ((v - c) * inv).round_ties_even().clamp(-qmax, qmax) as i8;
    }
    (scale, c)
}

fn has_dotprod() -> bool {
    static D: OnceLock<bool> = OnceLock::new();
    *D.get_or_init(|| std::arch::is_aarch64_feature_detected!("dotprod"))
}

/// Rows per int8 register tile.
const MR8: usize = 4;

/// Panels `p0..p1` of the int8 GEMM into `out` (row stride `ldo`, column `16 p0` at 0).
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
    let (n, kp) = (w.n, w.kp);
    let col0 = p0 * NR;
    assert!(p0 < p1 && p1 <= w.panels() && rows > 0);
    let width = (p1 * NR).min(n) - col0;
    assert!(ldo >= width && out.len() >= (rows - 1) * ldo + width);
    assert!(a.len() >= rows * kp && sa.len() >= rows && ca.len() >= rows);
    assert!(w.data.len() == w.panels() * kp * NR && w.scale.len() == w.panels() * NR);
    let dot = has_dotprod();
    for p in p0..p1 {
        let panel = w.panel(p);
        let ncols = NR.min(n - p * NR);
        let ep = Epilogue {
            sw: &w.scale[p * NR..(p + 1) * NR],
            wsum: &w.wsum[p * NR..(p + 1) * NR],
            bias: &w.bias[p * NR..(p + 1) * NR],
        };
        let mut r = 0;
        while r < rows {
            let m = (rows - r).min(MR8);
            let ar = &a[r * kp..(r + m) * kp];
            let acc = if dot {
                // SAFETY: dotprod availability checked by has_dotprod().
                unsafe { acc_sdot(ar, kp, panel, m) }
            } else {
                acc_widen(ar, kp, panel, m)
            };
            let os = &mut out[r * ldo + p * NR - col0..];
            ep.store(&acc[..m], &sa[r..r + m], &ca[r..r + m], os, ldo, ncols);
            r += m;
        }
    }
}

struct Epilogue<'a> {
    sw: &'a [f32],
    wsum: &'a [f32],
    bias: &'a [f32],
}

impl Epilogue<'_> {
    /// `f32(acc) * (sa * sw) + (ca * wsum + bias)` with the AVX2 roundings.
    fn store(
        &self,
        acc: &[[int32x4_t; 4]],
        sa: &[f32],
        ca: &[f32],
        out: &mut [f32],
        ldo: usize,
        ncols: usize,
    ) {
        assert!(self.sw.len() >= NR && self.wsum.len() >= NR && self.bias.len() >= NR);
        assert!(ncols <= NR && out.len() >= (acc.len() - 1) * ldo + ncols);
        for (m, am) in acc.iter().enumerate() {
            let mut tmp = [0.0f32; NR];
            for (c, &av) in am.iter().enumerate() {
                // SAFETY: c * 4 + 4 <= 16 for sw, wsum, bias (asserted) and tmp.
                unsafe {
                    let s = vmulq_f32(vdupq_n_f32(sa[m]), vld1q_f32(self.sw.as_ptr().add(4 * c)));
                    let base = vfmaq_f32(
                        vld1q_f32(self.bias.as_ptr().add(4 * c)),
                        vdupq_n_f32(ca[m]),
                        vld1q_f32(self.wsum.as_ptr().add(4 * c)),
                    );
                    let y = vfmaq_f32(base, vcvtq_f32_s32(av), s);
                    vst1q_f32(tmp.as_mut_ptr().add(4 * c), y);
                }
            }
            out[m * ldo..m * ldo + ncols].copy_from_slice(&tmp[..ncols]);
        }
    }
}

/// Exact int32 sums for up to 4 rows x 16 columns with `sdot` (4 columns x 4 k per lane
/// group: the packed layout `[kg][16 cols][4]` is exactly the instruction's operand shape).
#[target_feature(enable = "dotprod")]
fn acc_sdot(a: &[i8], kp: usize, panel: &[i8], m: usize) -> [[int32x4_t; 4]; MR8] {
    assert!(m <= MR8 && a.len() >= m * kp && panel.len() >= kp * NR && kp.is_multiple_of(4));
    let mut acc = [[vdupq_n_s32(0); 4]; MR8];
    for g in 0..kp / 4 {
        // SAFETY: g * 64 + 64 <= kp * 16 <= panel.len() (asserted).
        let wv = unsafe {
            let p = panel.as_ptr().add(g * 64);
            [
                vld1q_s8(p),
                vld1q_s8(p.add(16)),
                vld1q_s8(p.add(32)),
                vld1q_s8(p.add(48)),
            ]
        };
        for (r, ar) in acc.iter_mut().enumerate().take(m) {
            // SAFETY: r * kp + 4 g + 4 <= m * kp <= a.len(); unaligned 4-byte read.
            let a4 =
                unsafe { std::ptr::read_unaligned(a.as_ptr().add(r * kp + 4 * g).cast::<i32>()) };
            let av = vreinterpretq_s8_s32(vdupq_n_s32(a4));
            for (acc_c, &w) in ar.iter_mut().zip(&wv) {
                // SAFETY: `sdot` is available (target feature enabled and detected by the
                // caller); register-only instruction, no memory access.
                unsafe {
                    std::arch::asm!(
                        "sdot {acc:v}.4s, {w:v}.16b, {a:v}.16b",
                        acc = inout(vreg) *acc_c,
                        w = in(vreg) w,
                        a = in(vreg) av,
                        options(pure, nomem, nostack, preserves_flags)
                    );
                }
            }
        }
    }
    acc
}

/// Baseline NEON twin of `acc_sdot`: `smull` products (|a w| <= 128 * 127 fits i16), then
/// pairwise widening adds give each column's 4-term sum exactly.
fn acc_widen(a: &[i8], kp: usize, panel: &[i8], m: usize) -> [[int32x4_t; 4]; MR8] {
    assert!(m <= MR8 && a.len() >= m * kp && panel.len() >= kp * NR && kp.is_multiple_of(4));
    // SAFETY: register-only NEON op.
    let mut acc = [[unsafe { vdupq_n_s32(0) }; 4]; MR8];
    for g in 0..kp / 4 {
        for (r, ar) in acc.iter_mut().enumerate().take(m) {
            // SAFETY: r * kp + 4 g + 4 <= a.len(); g * 64 + 64 <= panel.len() (asserted).
            unsafe {
                let a4 = std::ptr::read_unaligned(a.as_ptr().add(r * kp + 4 * g).cast::<i32>());
                let av = vreinterpretq_s8_s32(vdupq_n_s32(a4));
                for (c, acc_c) in ar.iter_mut().enumerate() {
                    let w = vld1q_s8(panel.as_ptr().add(g * 64 + 16 * c));
                    let lo = vpaddlq_s16(vmull_s8(vget_low_s8(w), vget_low_s8(av)));
                    let hi = vpaddlq_s16(vmull_high_s8(w, av));
                    *acc_c = vaddq_s32(*acc_c, vpaddq_s32(lo, hi));
                }
            }
        }
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sdot_and_widening_paths_agree() {
        if !has_dotprod() {
            return;
        }
        let (kp, m) = (64usize, 4usize);
        let a: Vec<i8> = (0..m * kp)
            .map(|i| (i as i32 * 37 % 255 - 127) as i8)
            .collect();
        let panel: Vec<i8> = (0..kp * NR)
            .map(|i| (i as i32 * 53 % 255 - 127) as i8)
            .collect();
        // SAFETY: dotprod detected above.
        let s = unsafe { acc_sdot(&a, kp, &panel, m) };
        let w = acc_widen(&a, kp, &panel, m);
        for r in 0..m {
            for c in 0..4 {
                let (mut x, mut y) = ([0i32; 4], [0i32; 4]);
                // SAFETY: each array holds 4 i32.
                unsafe {
                    vst1q_s32(x.as_mut_ptr(), s[r][c]);
                    vst1q_s32(y.as_mut_ptr(), w[r][c]);
                }
                assert_eq!(x, y);
            }
        }
    }
}
