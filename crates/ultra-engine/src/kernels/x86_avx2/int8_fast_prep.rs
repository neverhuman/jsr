//! Shared constants, tile-mode preparation, and epilogue helpers for fast AVX2 INT8 GEMM.

use std::arch::x86_64::*;

use super::super::NR;

#[repr(C, align(32))]
pub(super) struct Align32<T>(pub(super) T);

#[repr(C, align(32))]
pub(super) struct Avx2Consts {
    ones: [i16; 16],
    mask7f: [u8; 32],
}

pub(super) static AVX2_CONSTS: Avx2Consts = Avx2Consts {
    ones: [1; 16],
    mask7f: [0x7F; 32],
};

/// Prefetch distances (bytes) ahead in the packed int8 weight stream on Zen 2:
/// - `NTA` (single-pass `rows <= 4` streaming): bypasses L3 victim allocation so the KV cache
///   stays resident in the 16 MiB CCX L3 while weights stream at peak DRAM bandwidth.
/// - `T0` (multi-tile `rows > 4` reuse, e.g. commit `chunk=16`): keeps each panel in L1/L2.
pub(super) const I8_PF_NTA: usize = 2048;
pub(super) const I8_PF_T0: usize = 2048;

/// Maximum sum of unsigned operands contributing to one `i16` lane of `vpmaddubsw` across K-groups
/// without any possibility of `i16` overflow (`258 * 127 = 32_766 <= i16::MAX`).
pub(super) const MAX_SAFE_I16_SUM: u16 = 258;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum TileMode {
    SignedFallback,
    SignedPair,
    U128Pos,
    U127Neg,
}

/// Fast AVX2 horizontal sum of a row of `i8` values to test skewness (`mean <= -48` or `>= 48`).
#[target_feature(enable = "avx2,fma")]
fn row_sum_i8(row: &[i8]) -> i32 {
    let n = row.len();
    let main32 = n - n % 32;
    let x80 = _mm256_set1_epi8(-128i8);
    let zero = _mm256_setzero_si256();
    let mut acc = _mm256_setzero_si256();
    let mut i = 0;
    while i < main32 {
        // SAFETY: i + 32 <= n.
        let v = unsafe { _mm256_loadu_si256(row.as_ptr().add(i).cast()) };
        let u = _mm256_xor_si256(v, x80);
        acc = _mm256_add_epi64(acc, _mm256_sad_epu8(u, zero));
        i += 32;
    }
    let mut lanes = [0u64; 4];
    // SAFETY: lanes holds 4 u64s.
    unsafe { _mm256_storeu_si256(lanes.as_mut_ptr().cast(), acc) };
    let u_sum: i64 = lanes.iter().map(|&v| v as i64).sum();
    let mut s = (u_sum - (main32 as i64) * 128) as i32;
    for &v in &row[main32..] {
        s += i32::from(v);
    }
    s
}

#[target_feature(enable = "avx2,fma")]
pub(super) fn prefer_unsigned(a: &[i8], kp: usize, has_qsum: bool) -> Option<bool> {
    if !has_qsum || kp < 16 || kp / 8 >= 0x7FFF {
        return None;
    }
    let mut all_pos = true;
    let mut all_neg = true;
    let thresh = (kp as i32) * 48;
    for r in 0..4 {
        let sum = row_sum_i8(&a[r * kp..(r + 1) * kp]);
        if sum > -thresh {
            all_pos = false;
        }
        if sum < thresh {
            all_neg = false;
        }
        if !all_pos && !all_neg {
            return None;
        }
    }
    if all_pos {
        Some(false)
    } else if all_neg {
        Some(true)
    } else {
        None
    }
}

/// Transform 4 rows of `a` into `u_dst` (`a + 128` or `127 - a`) and collect
/// encoded bad pair/quad entries `(idx << 1) | needs_split`, terminated by `u16::MAX` sentinel.
#[target_feature(enable = "avx2,fma")]
pub(super) fn prepare_u_runs(
    a: &[i8],
    kp: usize,
    neg: bool,
    pair_mode: bool,
    u_dst: &mut [u8],
    bad: &mut Vec<u16>,
) -> bool {
    let total = 4 * kp;
    assert!(a.len() >= total && u_dst.len() >= total);
    let main32 = total - total % 32;
    let dst_ptr = u_dst.as_mut_ptr();
    if !neg {
        let x80 = _mm256_set1_epi8(-128i8);
        let mut i = 0;
        while i < main32 {
            // SAFETY: i + 32 <= total <= a.len() and u_dst.len().
            unsafe {
                let v = _mm256_loadu_si256(a.as_ptr().add(i).cast());
                _mm256_storeu_si256(dst_ptr.add(i).cast(), _mm256_xor_si256(v, x80));
            }
            i += 32;
        }
        for i in main32..total {
            u_dst[i] = (a[i] as u8) ^ 0x80;
        }
    } else {
        let c127 = _mm256_set1_epi8(127i8);
        let mut i = 0;
        while i < main32 {
            // SAFETY: i + 32 <= total <= a.len() and u_dst.len().
            unsafe {
                let v = _mm256_loadu_si256(a.as_ptr().add(i).cast());
                _mm256_storeu_si256(dst_ptr.add(i).cast(), _mm256_sub_epi8(c127, v));
            }
            i += 32;
        }
        for i in main32..total {
            u_dst[i] = (127i16 - i16::from(a[i])) as u8;
        }
    }
    let u = &u_dst[..total];

    let start = bad.len();
    if pair_mode {
        let pairs = kp / 8;
        let max_bad = (pairs / 3).max(4);
        for pi in 0..pairs {
            let k0 = pi * 8;
            let mut ok = true;
            let mut split = false;
            for m in 0..4 {
                let s = &u[m * kp + k0..m * kp + k0 + 8];
                let e0 = u16::from(s[0]) + u16::from(s[1]);
                let o0 = u16::from(s[2]) + u16::from(s[3]);
                let e1 = u16::from(s[4]) + u16::from(s[5]);
                let o1 = u16::from(s[6]) + u16::from(s[7]);
                if e0 + e1 > MAX_SAFE_I16_SUM || o0 + o1 > MAX_SAFE_I16_SUM {
                    ok = false;
                    if e0 > MAX_SAFE_I16_SUM
                        || o0 > MAX_SAFE_I16_SUM
                        || e1 > MAX_SAFE_I16_SUM
                        || o1 > MAX_SAFE_I16_SUM
                    {
                        split = true;
                        break;
                    }
                }
            }
            if !ok {
                bad.push(((pi as u16) << 1) | u16::from(split));
                if bad.len() - start > max_bad {
                    bad.truncate(start);
                    return false;
                }
            }
        }
    } else {
        let quads = kp / 16;
        let max_bad = (quads / 2).max(4);
        for q in 0..quads {
            let k0 = q * 16;
            let mut ok = true;
            let mut split = false;
            for m in 0..4 {
                let s = &u[m * kp + k0..m * kp + k0 + 16];
                let mut e = 0u16;
                let mut o = 0u16;
                for j in 0..4 {
                    let ej = u16::from(s[4 * j]) + u16::from(s[4 * j + 1]);
                    let oj = u16::from(s[4 * j + 2]) + u16::from(s[4 * j + 3]);
                    if ej > MAX_SAFE_I16_SUM || oj > MAX_SAFE_I16_SUM {
                        split = true;
                    }
                    e += ej;
                    o += oj;
                }
                if e > MAX_SAFE_I16_SUM || o > MAX_SAFE_I16_SUM {
                    ok = false;
                    if split {
                        break;
                    }
                }
            }
            if !ok {
                bad.push(((q as u16) << 1) | u16::from(split));
                if bad.len() - start > max_bad {
                    bad.truncate(start);
                    return false;
                }
            }
        }
    }
    bad.push(u16::MAX);
    true
}

/// Scan `a_abs` (zero copy!) for a signed 4-row tile and collect bad pair indices terminated by
/// `u16::MAX`. Returns `true` if `<= 60%` of pairs are bad (so paired path wins).
#[inline]
pub(super) fn prepare_signed_runs(a_abs: &[u8], kp: usize, bad: &mut Vec<u16>) -> bool {
    let pairs = kp / 8;
    if pairs < 2 || pairs >= u16::MAX as usize {
        return false;
    }
    assert!(a_abs.len() >= 4 * kp);
    let start = bad.len();
    let max_bad = (pairs * 3 / 5).max(2);
    for pi in 0..pairs {
        let k0 = pi * 8;
        let mut ok = true;
        for m in 0..4 {
            let s = &a_abs[m * kp + k0..m * kp + k0 + 8];
            let e = u16::from(s[0]) + u16::from(s[1]) + u16::from(s[4]) + u16::from(s[5]);
            let o = u16::from(s[2]) + u16::from(s[3]) + u16::from(s[6]) + u16::from(s[7]);
            if e > MAX_SAFE_I16_SUM || o > MAX_SAFE_I16_SUM {
                ok = false;
                break;
            }
        }
        if !ok {
            bad.push(pi as u16);
            if bad.len() - start > max_bad {
                bad.truncate(start);
                return false;
            }
        }
    }
    bad.push(u16::MAX);
    true
}

#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
pub(super) fn epilogue_store<const M: usize>(
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
    assert!(sw.len() >= NR && wsum.len() >= NR && bias.len() >= NR);
    assert!(sa.len() >= M && ca.len() >= M && ncols <= NR);
    assert!((M - 1)
        .checked_mul(ldo)
        .and_then(|v| v.checked_add(ncols))
        .is_some_and(|need| out.len() >= need));
    // SAFETY: sw, wsum, bias have >= 16 floats (asserted).
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
            // SAFETY: row.len() >= 16 when ncols == 16 (asserted bound on out).
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

#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
pub(super) fn epilogue_u_store<const NEG: bool>(
    mut acc: [[__m256i; 2]; 4],
    qsum: &[i32],
    sw: &[f32],
    wsum: &[f32],
    bias: &[f32],
    sa: &[f32],
    ca: &[f32],
    out: &mut [f32],
    ldo: usize,
    ncols: usize,
) {
    assert!(qsum.len() >= NR);
    // SAFETY: qsum has >= 16 i32 values (asserted).
    let (q0, q1) = unsafe {
        (
            _mm256_loadu_si256(qsum.as_ptr().cast()),
            _mm256_loadu_si256(qsum.as_ptr().add(8).cast()),
        )
    };
    if !NEG {
        let c0 = _mm256_slli_epi32::<7>(q0);
        let c1 = _mm256_slli_epi32::<7>(q1);
        for acc_m in &mut acc {
            acc_m[0] = _mm256_sub_epi32(acc_m[0], c0);
            acc_m[1] = _mm256_sub_epi32(acc_m[1], c1);
        }
    } else {
        let c0 = _mm256_sub_epi32(_mm256_slli_epi32::<7>(q0), q0);
        let c1 = _mm256_sub_epi32(_mm256_slli_epi32::<7>(q1), q1);
        for acc_m in &mut acc {
            acc_m[0] = _mm256_sub_epi32(c0, acc_m[0]);
            acc_m[1] = _mm256_sub_epi32(c1, acc_m[1]);
        }
    }
    epilogue_store(&acc, sw, wsum, bias, sa, ca, out, ldo, ncols);
}

/// Exact single-group unsigned accumulate for arbitrary `u ∈ [0, 255]` (inline, zero `vzeroupper`).
///
/// # Safety
/// Caller must ensure AVX2+FMA is available and `up` points to at least 4 readable bytes.
#[inline]
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn accum_u_group_exact(
    up: *const u8,
    w0: __m256i,
    w1: __m256i,
    ones: __m256i,
    acc: &mut [__m256i; 2],
) {
    // SAFETY: caller ensures `up` has 4 readable bytes.
    let u4 = unsafe { std::ptr::read_unaligned(up.cast::<[u8; 4]>()) };
    if u16::from(u4[0]) + u16::from(u4[1]) <= MAX_SAFE_I16_SUM
        && u16::from(u4[2]) + u16::from(u4[3]) <= MAX_SAFE_I16_SUM
    {
        let uv = _mm256_set1_epi32(i32::from_ne_bytes(u4));
        acc[0] = _mm256_add_epi32(
            acc[0],
            _mm256_madd_epi16(_mm256_maddubs_epi16(uv, w0), ones),
        );
        acc[1] = _mm256_add_epi32(
            acc[1],
            _mm256_madd_epi16(_mm256_maddubs_epi16(uv, w1), ones),
        );
    } else {
        let lo = [u4[0] >> 1, u4[1] >> 1, u4[2] >> 1, u4[3] >> 1];
        let hi = [u4[0] - lo[0], u4[1] - lo[1], u4[2] - lo[2], u4[3] - lo[3]];
        let u_lo = _mm256_set1_epi32(i32::from_ne_bytes(lo));
        let u_hi = _mm256_set1_epi32(i32::from_ne_bytes(hi));
        acc[0] = _mm256_add_epi32(
            acc[0],
            _mm256_madd_epi16(_mm256_maddubs_epi16(u_lo, w0), ones),
        );
        acc[1] = _mm256_add_epi32(
            acc[1],
            _mm256_madd_epi16(_mm256_maddubs_epi16(u_lo, w1), ones),
        );
        acc[0] = _mm256_add_epi32(
            acc[0],
            _mm256_madd_epi16(_mm256_maddubs_epi16(u_hi, w0), ones),
        );
        acc[1] = _mm256_add_epi32(
            acc[1],
            _mm256_madd_epi16(_mm256_maddubs_epi16(u_hi, w1), ones),
        );
    }
}

/// Vectorized AVX2 computation of per-channel integer weight sums `qsum` (`sum_k w_q[n][k]`),
/// bit-identical to the scalar sum.
pub(crate) fn compute_qsum(data: &[i8], n: usize, kp: usize, qsum: &mut [i32]) {
    // SAFETY: reached only after Isa::Avx2Fma.is_available() (module docs).
    unsafe { compute_qsum_avx2(data, n, kp, qsum) }
}

#[target_feature(enable = "avx2,fma")]
fn compute_qsum_avx2(data: &[i8], n: usize, kp: usize, qsum: &mut [i32]) {
    let panels = n.div_ceil(NR);
    assert!(kp.is_multiple_of(4));
    let panel_elems = kp.checked_mul(NR).expect("compute_qsum panel overflow");
    let total = panels
        .checked_mul(panel_elems)
        .expect("compute_qsum total overflow");
    let meta = panels.checked_mul(NR).expect("compute_qsum meta overflow");
    assert!(data.len() == total && qsum.len() == meta);
    let groups = kp / 4;
    let ones_u8 = _mm256_set1_epi8(1);
    let ones_i16 = _mm256_set1_epi16(1);
    for p in 0..panels {
        let ncols = NR.min(n - p * NR);
        // SAFETY: (p + 1) * panel_elems <= data.len().
        let wp = unsafe { data.as_ptr().add(p * panel_elems) };
        let mut q0 = _mm256_setzero_si256();
        let mut q1 = _mm256_setzero_si256();
        for g in 0..groups {
            // SAFETY: (g + 1) * 64 <= kp * NR = panel_elems.
            unsafe {
                let w0 = _mm256_loadu_si256(wp.add(g * 64).cast());
                let w1 = _mm256_loadu_si256(wp.add(g * 64 + 32).cast());
                let s0 = _mm256_madd_epi16(_mm256_maddubs_epi16(ones_u8, w0), ones_i16);
                let s1 = _mm256_madd_epi16(_mm256_maddubs_epi16(ones_u8, w1), ones_i16);
                q0 = _mm256_add_epi32(q0, s0);
                q1 = _mm256_add_epi32(q1, s1);
            }
        }
        let dst = &mut qsum[p * NR..(p + 1) * NR];
        // SAFETY: dst holds NR = 16 i32 values.
        unsafe {
            _mm256_storeu_si256(dst.as_mut_ptr().cast(), q0);
            _mm256_storeu_si256(dst.as_mut_ptr().add(8).cast(), q1);
        }
        if ncols < NR {
            dst[ncols..].fill(0);
        }
    }
}
