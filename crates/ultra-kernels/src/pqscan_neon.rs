//! ARM64 NEON 4-row codebook-scan (`vqtbl1q_u8`), SIMD table builder, and `FWHT16` (Wave H24).
//!
//! # Safety model
//! Compiled only on `aarch64` where NEON is mandatory. Every pointer load/store stays within
//! slices whose exact lengths are asserted up front.

use crate::pqscan::{CodebookMatrix, CodebookTables, PQSCAN_BLOCK, PQSCAN_FLUSH_PAIRS};
use std::arch::aarch64::*;

pub(crate) fn scan(m: &CodebookMatrix, t: &CodebookTables, sparse: bool, out: &mut [i32]) {
    // SAFETY: NEON is always available on aarch64.
    unsafe { scan_neon(m, t, sparse, out) }
}

pub(crate) fn fwht16(x: &mut [f32]) {
    // SAFETY: NEON is always available on aarch64.
    unsafe { fwht16_neon(x) }
}

#[target_feature(enable = "neon")]
fn scan_neon(m: &CodebookMatrix, t: &CodebookTables, sparse: bool, out: &mut [i32]) {
    let (rows, f_n, m_n, c_n) = (m.rows(), t.frames, m.stages_m(), m.subspaces_c());
    let pairs = c_n / 2;
    let mc = m_n * c_n;
    let blocks = rows.div_ceil(PQSCAN_BLOCK);
    let blk_stride = m_n * pairs * PQSCAN_BLOCK;
    assert_eq!(m.codes().len(), blocks * blk_stride);
    assert_eq!(t.entries.len(), f_n * mc);
    assert_eq!(out.len(), rows * f_n);

    let mut f = 0;
    while f < f_n {
        let take = (f_n - f).min(4);
        match take {
            4 => scan_blocks_neon::<4>(m, t, f, sparse, out),
            3 => scan_blocks_neon::<3>(m, t, f, sparse, out),
            2 => scan_blocks_neon::<2>(m, t, f, sparse, out),
            _ => scan_blocks_neon::<1>(m, t, f, sparse, out),
        }
        f += take;
    }
}

#[target_feature(enable = "neon")]
fn scan_blocks_neon<const F: usize>(
    m: &CodebookMatrix,
    t: &CodebookTables,
    f0: usize,
    sparse: bool,
    out: &mut [i32],
) {
    let (codes, rows, m_n, c_n) = (m.codes(), m.rows(), m.stages_m(), m.subspaces_c());
    let pairs = c_n / 2;
    let mc = m_n * c_n;
    let f_total = t.frames;
    let blk_stride = m_n * pairs * PQSCAN_BLOCK;
    let blocks = rows.div_ceil(PQSCAN_BLOCK);
    let nib = vdupq_n_u8(0x0F);

    for blk in 0..blocks {
        let blk_codes = &codes[blk * blk_stride..(blk + 1) * blk_stride];
        let mut acc32 = [[vdupq_n_u32(0); 8]; F];
        let mut acc16 = [[vdupq_n_u16(0); 4]; F];
        let mut in_flush = 0usize;

        for st in 0..m_n {
            for p in 0..pairs {
                if sparse && (0..F).all(|i| t.pair_nonzero[(f0 + i) * pairs + p] == 0) {
                    continue;
                }
                let off_c = (st * pairs + p) * PQSCAN_BLOCK;
                // SAFETY: off_c + 32 <= blk_codes.len().
                let (b0, b1) = unsafe {
                    (
                        vld1q_u8(blk_codes.as_ptr().add(off_c)),
                        vld1q_u8(blk_codes.as_ptr().add(off_c + 16)),
                    )
                };
                let il0 = vandq_u8(b0, nib);
                let ih0 = vshrq_n_u8::<4>(b0);
                let il1 = vandq_u8(b1, nib);
                let ih1 = vshrq_n_u8::<4>(b1);

                for i in 0..F {
                    let base = (f0 + i) * mc + st * c_n + 2 * p;
                    // SAFETY: base + 1 < t.entries.len(), each entry is 16 bytes.
                    let (te, to) = unsafe {
                        (
                            vld1q_u8(t.entries[base].as_ptr()),
                            vld1q_u8(t.entries[base + 1].as_ptr()),
                        )
                    };
                    let s0 = vaddq_u8(vqtbl1q_u8(te, il0), vqtbl1q_u8(to, ih0));
                    let s1 = vaddq_u8(vqtbl1q_u8(te, il1), vqtbl1q_u8(to, ih1));
                    acc16[i][0] = vaddw_u8(acc16[i][0], vget_low_u8(s0));
                    acc16[i][1] = vaddw_high_u8(acc16[i][1], s0);
                    acc16[i][2] = vaddw_u8(acc16[i][2], vget_low_u8(s1));
                    acc16[i][3] = vaddw_high_u8(acc16[i][3], s1);
                }
                in_flush += 1;
                if in_flush == PQSCAN_FLUSH_PAIRS {
                    for i in 0..F {
                        for g in 0..4 {
                            acc32[i][2 * g] = vaddw_u16(acc32[i][2 * g], vget_low_u16(acc16[i][g]));
                            acc32[i][2 * g + 1] = vaddw_high_u16(acc32[i][2 * g + 1], acc16[i][g]);
                            acc16[i][g] = vdupq_n_u16(0);
                        }
                    }
                    in_flush = 0;
                }
            }
        }
        if in_flush > 0 {
            for i in 0..F {
                for g in 0..4 {
                    acc32[i][2 * g] = vaddw_u16(acc32[i][2 * g], vget_low_u16(acc16[i][g]));
                    acc32[i][2 * g + 1] = vaddw_high_u16(acc32[i][2 * g + 1], acc16[i][g]);
                }
            }
        }

        let r0 = blk * PQSCAN_BLOCK;
        let r_end = (r0 + PQSCAN_BLOCK).min(rows);
        let mut tmp = [0u32; PQSCAN_BLOCK];
        for i in 0..F {
            for g in 0..8 {
                // SAFETY: 4 * g + 4 <= 32 = tmp.len().
                unsafe { vst1q_u32(tmp.as_mut_ptr().add(4 * g), acc32[i][g]) };
            }
            let sh = t.shifts[f0 + i];
            let off = t.offsets[f0 + i];
            for lane in 0..(r_end - r0) {
                out[(r0 + lane) * f_total + (f0 + i)] = ((tmp[lane] as i64) << sh) as i32 + off;
            }
        }
    }
}

/// In-register NEON block-16 orthonormal Walsh-Hadamard transform (exact match to AVX2/scalar).
#[target_feature(enable = "neon")]
fn fwht16_neon(x: &mut [f32]) {
    let scale = vdupq_n_f32(0.25);
    for blk in x.chunks_exact_mut(16) {
        // SAFETY: blk.len() == 16.
        let (mut v0, mut v1, mut v2, mut v3) = unsafe {
            (
                vld1q_f32(blk.as_ptr()),
                vld1q_f32(blk.as_ptr().add(4)),
                vld1q_f32(blk.as_ptr().add(8)),
                vld1q_f32(blk.as_ptr().add(12)),
            )
        };
        // Stage 4 (stride 8)
        let (a0, a2) = (vaddq_f32(v0, v2), vsubq_f32(v0, v2));
        let (a1, a3) = (vaddq_f32(v1, v3), vsubq_f32(v1, v3));
        // Stage 3 (stride 4)
        v0 = vaddq_f32(a0, a1);
        v1 = vsubq_f32(a0, a1);
        v2 = vaddq_f32(a2, a3);
        v3 = vsubq_f32(a2, a3);

        // Stage 2 (stride 2) + Stage 1 (stride 1) inside each 4-lane vector.
        let inner = |v: float32x4_t| {
            let sw2 = vextq_f32::<2>(v, v);
            let add2 = vaddq_f32(v, sw2);
            let sub2 = vsubq_f32(sw2, v);
            let u = vcombine_f32(vget_low_f32(add2), vget_high_f32(sub2));
            let sw1 = vrev64q_f32(u);
            let add1 = vaddq_f32(u, sw1);
            let sub1 = vsubq_f32(u, sw1);
            let z = vtrn1q_f32(add1, sub1);
            // vtrn1q picks [add1[0], sub1[0], add1[2], sub1[2]] -> exact [u0+u1, u0-u1, u2+u3, u2-u3]!
            vmulq_f32(z, scale)
        };
        v0 = inner(v0);
        v1 = inner(v1);
        v2 = inner(v2);
        v3 = inner(v3);

        // SAFETY: blk.len() == 16.
        unsafe {
            vst1q_f32(blk.as_mut_ptr(), v0);
            vst1q_f32(blk.as_mut_ptr().add(4), v1);
            vst1q_f32(blk.as_mut_ptr().add(8), v2);
            vst1q_f32(blk.as_mut_ptr().add(12), v3);
        }
    }
}
