//! AVX2 IEEE-754 finite `f32` tensor validator (8 lanes/instruction, scalar reference & tail).
//!
//! # Safety model
//! [`all_finite_f32_avx2`] is entered exclusively after runtime `is_x86_feature_detected!("avx2")`.
//! All loads are unaligned (`_mm256_loadu_si256`) and bounded by `i + 8 <= slice.len()`.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Portable scalar reference: returns `true` iff every element in `slice` is finite.
#[inline]
pub(crate) fn all_finite_f32_scalar(slice: &[f32]) -> bool {
    slice.iter().all(|v| v.is_finite())
}

/// Runtime-dispatched finite check over `slice`.
#[inline]
pub(crate) fn all_finite_f32(slice: &[f32]) -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: `avx2` verified at runtime above.
            return unsafe { all_finite_f32_avx2(slice) };
        }
    }
    all_finite_f32_scalar(slice)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn all_finite_f32_avx2(slice: &[f32]) -> bool {
    let exp_mask = _mm256_set1_epi32(0x7f80_0000u32 as i32);
    let mut i = 0usize;
    while i + 8 <= slice.len() {
        // SAFETY: `i + 8 <= slice.len()` checked in loop condition.
        let v = unsafe { _mm256_loadu_si256(slice.as_ptr().add(i).cast()) };
        let non_finite = _mm256_cmpeq_epi32(_mm256_and_si256(v, exp_mask), exp_mask);
        if _mm256_movemask_epi8(non_finite) != 0 {
            return false;
        }
        i += 8;
    }
    all_finite_f32_scalar(&slice[i..])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn avx2_finite_validator_matches_scalar_across_adversarial_inputs() {
        for len in [0usize, 1, 7, 8, 9, 15, 16, 31, 32, 37] {
            let mut buf = vec![0.0f32; len];
            for (k, v) in buf.iter_mut().enumerate() {
                *v = match k % 5 {
                    0 => 0.0,
                    1 => -0.0,
                    2 => f32::MIN_POSITIVE * 0.25, // subnormal
                    3 => f32::MAX,
                    _ => f32::MIN,
                };
            }
            assert!(all_finite_f32(&buf));
            assert_eq!(all_finite_f32(&buf), all_finite_f32_scalar(&buf));

            for pos in 0..len {
                for bad in [
                    f32::NAN,
                    f32::from_bits(0xffc0_0001),
                    f32::INFINITY,
                    f32::NEG_INFINITY,
                ] {
                    let saved = buf[pos];
                    buf[pos] = bad;
                    assert!(!all_finite_f32(&buf), "missed non-finite at {pos}/{len}");
                    assert_eq!(all_finite_f32(&buf), all_finite_f32_scalar(&buf));
                    buf[pos] = saved;
                }
            }
        }
    }
}
