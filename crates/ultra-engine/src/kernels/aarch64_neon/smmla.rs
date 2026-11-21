//! The i8mm `smmla` int8 accumulation (ARMv8.6), via stable inline asm.
//! Safety model: see the parent module docs.

use std::arch::aarch64::*;

use super::super::NR;
use super::int8::MR8;

/// Exact int32 sums with `smmla` (ARMv8.6 i8mm): one instruction multiplies a 2-row x 8-k
/// activation block by a 2-column x 8-k weight block into a 2x2 int32 tile; the i8mm
/// packing (`PackedI8::mm`) stores each column pair's 8-k blocks contiguously.
#[target_feature(enable = "neon,i8mm")]
pub(super) fn acc_smmla(a: &[i8], kp: usize, mm: &[i8], m: usize) -> [[int32x4_t; 4]; MR8] {
    assert!(m <= MR8 && a.len() >= m * kp && mm.len() >= kp * NR && kp.is_multiple_of(8));
    let pairs = m.div_ceil(2);
    // acc[rp][cp] lanes: (2rp, 2cp), (2rp, 2cp+1), (2rp+1, 2cp), (2rp+1, 2cp+1).
    let mut acc = [[vdupq_n_s32(0); 8]; 2];
    for g in 0..kp / 8 {
        // SAFETY: g * 128 + 128 <= kp * 16 <= mm.len() (asserted).
        let wv: [int8x16_t; 8] =
            std::array::from_fn(|cp| unsafe { vld1q_s8(mm.as_ptr().add(g * 128 + 16 * cp)) });
        for (rp, acc_r) in acc.iter_mut().enumerate().take(pairs) {
            let r0 = 2 * rp;
            // SAFETY: row r0 (and r0 + 1 when < m) spans 8 bytes at 8 g inside a (asserted).
            let av = unsafe {
                let lo = vld1_s8(a.as_ptr().add(r0 * kp + 8 * g));
                let hi = if r0 + 1 < m {
                    vld1_s8(a.as_ptr().add((r0 + 1) * kp + 8 * g))
                } else {
                    vdup_n_s8(0)
                };
                vcombine_s8(lo, hi)
            };
            for (acc_c, &w) in acc_r.iter_mut().zip(&wv) {
                // SAFETY: `smmla` is available (target feature enabled and detected by the
                // caller); register-only instruction, no memory access.
                unsafe {
                    std::arch::asm!(
                        "smmla {acc:v}.4s, {a:v}.16b, {w:v}.16b",
                        acc = inout(vreg) *acc_c,
                        a = in(vreg) av,
                        w = in(vreg) w,
                        options(pure, nomem, nostack, preserves_flags)
                    );
                }
            }
        }
    }
    // Re-tile to [row][4-column block] as the epilogue expects.
    let mut out = [[vdupq_n_s32(0); 4]; MR8];
    for rp in 0..pairs {
        for cb in 0..4 {
            let (x, y) = (
                vreinterpretq_s64_s32(acc[rp][2 * cb]),
                vreinterpretq_s64_s32(acc[rp][2 * cb + 1]),
            );
            out[2 * rp][cb] = vreinterpretq_s32_s64(vzip1q_s64(x, y));
            if 2 * rp + 1 < MR8 {
                out[2 * rp + 1][cb] = vreinterpretq_s32_s64(vzip2q_s64(x, y));
            }
        }
    }
    out
}
