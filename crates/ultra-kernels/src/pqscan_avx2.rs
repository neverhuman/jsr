//! AVX2 zero-spill 4-row codebook-scan (`vpshufb`), SIMD table builder, and `FWHT16` (Wave H24).
//!
//! # Register budget (16 YMM registers on x86_64 AVX2, ZERO spills in inner loop)
//! - `acc0[0..4]`, `acc1[0..4]`: 8 YMMs (wrapping u16 accumulators; `even = acc0 - (acc1 << 8)`
//!   is algebraically exact mod 2^16 whenever `pairs_since_flush <= 256` since `256 * 254 < 2^16`).
//! - `nib` (1), `il` (1), `ih` (1), `te` (1), `to` (1), `s` (1): 6 YMMs.
//! - Total = 14 YMMs <= 16 YMMs. The i32 totals live in L1 stack arrays and are touched only at
//!   the 256-pair flush boundary.

use crate::pqscan::{CodebookMatrix, CodebookTables, PQSCAN_BLOCK, PQSCAN_FLUSH_PAIRS, PQSCAN_K};
use std::arch::x86_64::*;

pub(crate) fn build_tables_into(
    m: &CodebookMatrix,
    x: &[i8],
    sparse: bool,
    out: &mut CodebookTables,
) {
    // SAFETY: caller verified Backend::Avx2 availability.
    unsafe { build_tables_avx2(m, x, sparse, out) }
}

pub(crate) fn scan(m: &CodebookMatrix, t: &CodebookTables, sparse: bool, out: &mut [i32]) {
    // SAFETY: caller verified Backend::Avx2 availability.
    unsafe { scan_avx2(m, t, sparse, out) }
}

pub(crate) fn fwht16(x: &mut [f32]) {
    // SAFETY: caller verified Backend::Avx2 availability.
    unsafe { fwht16_avx2(x) }
}

/// Fast SIMD table builder reusing preallocated `CodebookTables` buffers (zero heap allocs).
#[target_feature(enable = "avx2")]
fn build_tables_avx2(m: &CodebookMatrix, x: &[i8], sparse: bool, out: &mut CodebookTables) {
    let f_n = x.len() / m.din();
    let (m_n, c_n, s) = (m.stages_m(), m.subspaces_c(), m.subspace_s());
    let pairs = c_n / 2;
    let mc = m_n * c_n;
    out.resize(f_n, m_n, c_n);

    let cb = m.codebooks();
    let ones16 = _mm256_set1_epi16(1);

    for f in 0..f_n {
        let a = &x[f * m.din()..(f + 1) * m.din()];
        if sparse {
            for p in 0..pairs {
                let seg = &a[2 * p * s..(2 * p + 2) * s];
                out.pair_nonzero[f * pairs + p] = u8::from(seg.iter().any(|&v| v != 0));
            }
        }
        let mut max_rng = 0i32;
        for st in 0..m_n {
            for c in 0..c_n {
                let idx = st * c_n + c;
                if sparse && out.pair_nonzero[f * pairs + (c / 2)] == 0 {
                    out.raw_scratch[idx] = [0; PQSCAN_K];
                    out.mins_scratch[idx] = 0;
                    continue;
                }
                let ax = &a[c * s..(c + 1) * s];
                let cb_base = idx * PQSCAN_K * s;
                let dst = &mut out.raw_scratch[idx];
                match s {
                    4 => {
                        // Exact signed dot product of 16 codewords of width 4 via abs(a) * sign(cb, a).
                        // 2 * 127 * 127 = 32,258 < 32,767 -> vpmaddubsw never saturates!
                        let a4 = i32::from_le_bytes([
                            ax[0] as u8,
                            ax[1] as u8,
                            ax[2] as u8,
                            ax[3] as u8,
                        ]);
                        let av = _mm256_set1_epi32(a4);
                        let abs_a = _mm256_abs_epi8(av);
                        // SAFETY: cb_base + 64 <= cb.len() (16 * 4 = 64 bytes).
                        let w0 = unsafe { _mm256_loadu_si256(cb.as_ptr().add(cb_base).cast()) };
                        // SAFETY: as above.
                        let w1 =
                            unsafe { _mm256_loadu_si256(cb.as_ptr().add(cb_base + 32).cast()) };
                        let p0 = _mm256_madd_epi16(
                            _mm256_maddubs_epi16(abs_a, _mm256_sign_epi8(w0, av)),
                            ones16,
                        );
                        let p1 = _mm256_madd_epi16(
                            _mm256_maddubs_epi16(abs_a, _mm256_sign_epi8(w1, av)),
                            ones16,
                        );
                        // SAFETY: dst holds 16 i32s (64 bytes).
                        unsafe {
                            _mm256_storeu_si256(dst.as_mut_ptr().cast(), p0);
                            _mm256_storeu_si256(dst.as_mut_ptr().add(8).cast(), p1);
                        }
                    }
                    2 => {
                        let a2 = i16::from_le_bytes([ax[0] as u8, ax[1] as u8]);
                        let av = _mm256_set1_epi16(a2);
                        let abs_a = _mm256_abs_epi8(av);
                        // SAFETY: cb_base + 32 <= cb.len() (16 * 2 = 32 bytes).
                        let w = unsafe { _mm256_loadu_si256(cb.as_ptr().add(cb_base).cast()) };
                        let p16 = _mm256_maddubs_epi16(abs_a, _mm256_sign_epi8(w, av));
                        let lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(p16));
                        let hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(p16));
                        // SAFETY: dst holds 16 i32s.
                        unsafe {
                            _mm256_storeu_si256(dst.as_mut_ptr().cast(), lo);
                            _mm256_storeu_si256(dst.as_mut_ptr().add(8).cast(), hi);
                        }
                    }
                    _ => {
                        // s == 8: 16 codewords x 8 bytes = 128 bytes = 4 YMMs (4 codewords/YMM).
                        let a8 = i64::from_le_bytes([
                            ax[0] as u8,
                            ax[1] as u8,
                            ax[2] as u8,
                            ax[3] as u8,
                            ax[4] as u8,
                            ax[5] as u8,
                            ax[6] as u8,
                            ax[7] as u8,
                        ]);
                        let av = _mm256_set1_epi64x(a8);
                        let abs_a = _mm256_abs_epi8(av);
                        for g in 0..4 {
                            // SAFETY: cb_base + 32*g + 32 <= cb.len() (16*8 = 128).
                            let w = unsafe {
                                _mm256_loadu_si256(cb.as_ptr().add(cb_base + 32 * g).cast())
                            };
                            let p32 = _mm256_madd_epi16(
                                _mm256_maddubs_epi16(abs_a, _mm256_sign_epi8(w, av)),
                                ones16,
                            );
                            // Sum pairs of adjacent i32 lanes (2 per codeword -> 4 codewords).
                            let mut tmp = [0i32; 8];
                            // SAFETY: tmp holds 8 i32s.
                            unsafe { _mm256_storeu_si256(tmp.as_mut_ptr().cast(), p32) };
                            for k in 0..4 {
                                dst[4 * g + k] = tmp[2 * k] + tmp[2 * k + 1];
                            }
                        }
                    }
                }
                // Vectorized min/max over 16 i32 entries.
                // SAFETY: dst holds 16 i32s.
                let (v0, v1) = unsafe {
                    (
                        _mm256_loadu_si256(dst.as_ptr().cast()),
                        _mm256_loadu_si256(dst.as_ptr().add(8).cast()),
                    )
                };
                let mn_v = _mm256_min_epi32(v0, v1);
                let mx_v = _mm256_max_epi32(v0, v1);
                let mn128 = _mm_min_epi32(
                    _mm256_castsi256_si128(mn_v),
                    _mm256_extracti128_si256::<1>(mn_v),
                );
                let mx128 = _mm_max_epi32(
                    _mm256_castsi256_si128(mx_v),
                    _mm256_extracti128_si256::<1>(mx_v),
                );
                let mn128 = _mm_min_epi32(mn128, _mm_shuffle_epi32::<0b01_00_11_10>(mn128));
                let mx128 = _mm_max_epi32(mx128, _mm_shuffle_epi32::<0b01_00_11_10>(mx128));
                let mn = _mm_cvtsi128_si32(_mm_min_epi32(
                    mn128,
                    _mm_shuffle_epi32::<0b10_11_00_01>(mn128),
                ));
                let mx = _mm_cvtsi128_si32(_mm_max_epi32(
                    mx128,
                    _mm_shuffle_epi32::<0b10_11_00_01>(mx128),
                ));
                out.mins_scratch[idx] = mn;
                let r = mx - mn;
                if r > max_rng {
                    max_rng = r;
                }
            }
        }
        let mut sh = 0u32;
        while (max_rng >> sh) > 127 {
            sh += 1;
        }
        out.shifts[f] = sh;
        let sh_v = _mm_cvtsi32_si128(sh as i32);
        let mut sum_mn = 0i64;
        let mut active_subspaces = 0i64;
        for idx in 0..mc {
            let c = idx % c_n;
            if sparse && out.pair_nonzero[f * pairs + (c / 2)] == 0 {
                out.entries[f * mc + idx] = [0u8; PQSCAN_K];
                continue;
            }
            active_subspaces += 1;
            let mn = out.mins_scratch[idx];
            sum_mn += i64::from(mn);
            let mn_v = _mm256_set1_epi32(mn);
            let src = &out.raw_scratch[idx];
            // SAFETY: src has 16 i32s.
            let (q0, q1) = unsafe {
                (
                    _mm256_sra_epi32(
                        _mm256_sub_epi32(_mm256_loadu_si256(src.as_ptr().cast()), mn_v),
                        sh_v,
                    ),
                    _mm256_sra_epi32(
                        _mm256_sub_epi32(_mm256_loadu_si256(src.as_ptr().add(8).cast()), mn_v),
                        sh_v,
                    ),
                )
            };
            // Pack 16 i32 values in 0..=127 down to 16 u8s in one 128-bit store.
            let p16 = _mm256_packs_epi32(q0, q1);
            let p16_lo = _mm256_castsi256_si128(p16);
            let p16_hi = _mm256_extracti128_si256::<1>(p16);
            // Cross-lane order after packs_epi32: p16_lo = [0..4, 8..12], p16_hi = [4..8, 12..16].
            let v_0_7 = _mm_unpacklo_epi64(p16_lo, p16_hi);
            let v_8_15 = _mm_unpackhi_epi64(p16_lo, p16_hi);
            let packed = _mm_packus_epi16(v_0_7, v_8_15);
            // SAFETY: destination entry is [u8; 16].
            unsafe { _mm_storeu_si128(out.entries[f * mc + idx].as_mut_ptr().cast(), packed) };
        }
        let round_bias = (active_subspaces * ((1i64 << sh) - 1)) >> 1;
        out.offsets[f] = (sum_mn + round_bias) as i32;
    }
}

#[target_feature(enable = "avx2")]
fn scan_avx2(m: &CodebookMatrix, t: &CodebookTables, sparse: bool, out: &mut [i32]) {
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
            4 => scan_blocks::<4>(m, t, f, sparse, out),
            3 => scan_blocks::<3>(m, t, f, sparse, out),
            2 => scan_blocks::<2>(m, t, f, sparse, out),
            _ => scan_blocks::<1>(m, t, f, sparse, out),
        }
        f += take;
    }
}

/// Zero-spill 14-YMM AVX2 codebook-scan inner loop using the wrapping-u16 even/odd identity:
///   acc0 += s (u16), acc1 += (s >> 8) (u16)
///   at flush (<= 256 pairs): even = acc0 - (acc1 << 8) (exact u16!), odd = acc1 (exact u16!).
#[target_feature(enable = "avx2")]
fn scan_blocks<const F: usize>(
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
    let nib = _mm256_set1_epi8(0x0F);

    for blk in 0..blocks {
        let blk_codes = &codes[blk * blk_stride..(blk + 1) * blk_stride];
        // Spill-free: only u16 accumulators stay in YMM registers; i32 sums live on L1 stack.
        let mut sums = [[0u32; PQSCAN_BLOCK]; F];
        let mut acc0 = [_mm256_setzero_si256(); F];
        let mut acc1 = [_mm256_setzero_si256(); F];
        let mut in_flush = 0usize;

        let flush = |acc0: &mut [__m256i; F],
                     acc1: &mut [__m256i; F],
                     sums: &mut [[u32; PQSCAN_BLOCK]; F]| {
            let mut ev = [0u16; 16];
            let mut od = [0u16; 16];
            for i in 0..F {
                let even = _mm256_sub_epi16(acc0[i], _mm256_slli_epi16::<8>(acc1[i]));
                // SAFETY: ev and od each hold 16 u16s (32 bytes).
                unsafe {
                    _mm256_storeu_si256(ev.as_mut_ptr().cast(), even);
                    _mm256_storeu_si256(od.as_mut_ptr().cast(), acc1[i]);
                }
                for k in 0..16 {
                    sums[i][2 * k] += u32::from(ev[k]);
                    sums[i][2 * k + 1] += u32::from(od[k]);
                }
                acc0[i] = _mm256_setzero_si256();
                acc1[i] = _mm256_setzero_si256();
            }
        };

        for st in 0..m_n {
            for p in 0..pairs {
                if sparse && (0..F).all(|i| t.pair_nonzero[(f0 + i) * pairs + p] == 0) {
                    continue;
                }
                let off_c = (st * pairs + p) * PQSCAN_BLOCK;
                // SAFETY: off_c + 32 <= blk_codes.len().
                let b = unsafe { _mm256_loadu_si256(blk_codes.as_ptr().add(off_c).cast()) };
                let il = _mm256_and_si256(b, nib);
                let ih = _mm256_and_si256(_mm256_srli_epi16::<4>(b), nib);

                for i in 0..F {
                    let base = (f0 + i) * mc + st * c_n + 2 * p;
                    // SAFETY: base + 1 < t.entries.len(), each entry is 16 bytes.
                    let te = unsafe {
                        _mm256_broadcastsi128_si256(_mm_loadu_si128(
                            t.entries[base].as_ptr().cast(),
                        ))
                    };
                    // SAFETY: as above.
                    let to = unsafe {
                        _mm256_broadcastsi128_si256(_mm_loadu_si128(
                            t.entries[base + 1].as_ptr().cast(),
                        ))
                    };
                    let s =
                        _mm256_add_epi8(_mm256_shuffle_epi8(te, il), _mm256_shuffle_epi8(to, ih));
                    acc0[i] = _mm256_add_epi16(acc0[i], s);
                    acc1[i] = _mm256_add_epi16(acc1[i], _mm256_srli_epi16::<8>(s));
                }
                in_flush += 1;
                if in_flush == PQSCAN_FLUSH_PAIRS {
                    flush(&mut acc0, &mut acc1, &mut sums);
                    in_flush = 0;
                }
            }
        }
        if in_flush > 0 {
            flush(&mut acc0, &mut acc1, &mut sums);
        }

        let r0 = blk * PQSCAN_BLOCK;
        let r_end = (r0 + PQSCAN_BLOCK).min(rows);
        for i in 0..F {
            let sh = t.shifts[f0 + i];
            let off = t.offsets[f0 + i];
            for lane in 0..(r_end - r0) {
                out[(r0 + lane) * f_total + (f0 + i)] = ((sums[i][lane] as i64) << sh) as i32 + off;
            }
        }
    }
}

/// In-register AVX2 block-16 orthonormal Walsh-Hadamard transform (4 stages, 0 multiplies
/// except final power-of-2 scale `0.25`).
#[target_feature(enable = "avx2")]
fn fwht16_avx2(x: &mut [f32]) {
    let scale = _mm256_set1_ps(0.25);
    for blk in x.chunks_exact_mut(16) {
        // SAFETY: blk.len() == 16.
        let mut a = unsafe { _mm256_loadu_ps(blk.as_ptr()) };
        // SAFETY: blk.len() == 16.
        let mut b = unsafe { _mm256_loadu_ps(blk.as_ptr().add(8)) };

        let (t0, t1) = (_mm256_add_ps(a, b), _mm256_sub_ps(a, b));
        a = t0;
        b = t1;

        let stage4 = |v: __m256| {
            let sw = _mm256_permute2f128_ps::<0x01>(v, v);
            let add = _mm256_add_ps(v, sw);
            let sub = _mm256_sub_ps(sw, v);
            _mm256_blend_ps::<0xF0>(add, sub)
        };
        a = stage4(a);
        b = stage4(b);

        let stage2 = |v: __m256| {
            let sw = _mm256_shuffle_ps::<0b01_00_11_10>(v, v);
            let add = _mm256_add_ps(v, sw);
            let sub = _mm256_sub_ps(sw, v);
            _mm256_blend_ps::<0b1100_1100>(add, sub)
        };
        a = stage2(a);
        b = stage2(b);

        let stage1 = |v: __m256| {
            let sw = _mm256_shuffle_ps::<0b10_11_00_01>(v, v);
            let add = _mm256_add_ps(v, sw);
            let sub = _mm256_sub_ps(sw, v);
            _mm256_blend_ps::<0b1010_1010>(add, sub)
        };
        a = _mm256_mul_ps(stage1(a), scale);
        b = _mm256_mul_ps(stage1(b), scale);

        // SAFETY: blk.len() == 16.
        unsafe {
            _mm256_storeu_ps(blk.as_mut_ptr(), a);
            _mm256_storeu_ps(blk.as_mut_ptr().add(8), b);
        }
    }
}
