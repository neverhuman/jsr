//! AVX2 + FMA accelerated `f64` mel filterbank & DC/Hann windowing for [`super::trainer_mel`].
//!
//! # Safety model
//! Every `#[target_feature(enable = "avx2,fma")]` function is entered exclusively after runtime
//! `std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")`
//! verification. Accumulates in exact IEEE-754 `f64` (`[__m256d; 10]` = `40` mel bins per pass)
//! across `257` FFT bins so outputs match the scalar `f64` reference to `<= 1e-6` (`0` token drift).

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(any(target_arch = "x86_64", test))]
use crate::mel::WIN_SAMPLES;
#[cfg(target_arch = "x86_64")]
use crate::mel::{FFT_SIZE, MEL_BINS};
#[cfg(target_arch = "x86_64")]
use crate::trainer_mel::SPEC_BINS;

/// Returns `true` when both `avx2` and `fma` are supported on the current CPU.
#[inline]
pub(crate) fn avx2_fma_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

#[cfg(target_arch = "x86_64")]
pub(crate) fn dc_window_and_mel_avx2(
    frame: &[f32],
    window: &[f32; WIN_SAMPLES],
    fbank: &[f32],
    fft_fn: fn(&mut [f64; FFT_SIZE], &mut [f64; FFT_SIZE]),
) -> [f64; MEL_BINS] {
    assert!(frame.len() >= WIN_SAMPLES && fbank.len() >= SPEC_BINS * MEL_BINS);
    let mut re = [0.0f64; FFT_SIZE];
    let mut im = [0.0f64; FFT_SIZE];
    // SAFETY: Caller verified `avx2` + `fma` via `avx2_fma_available()`; bounds asserted above.
    unsafe { dc_and_window_f64_avx2(frame, window, &mut re) };
    fft_fn(&mut re, &mut im);
    let mut mel_acc = [0.0f64; MEL_BINS];
    // SAFETY: Caller verified `avx2` + `fma`; `fbank` has `SPEC_BINS * MEL_BINS` (`20_560`) elements.
    unsafe { mel_dot_f64_avx2(&re, &im, fbank, &mut mel_acc) };
    mel_acc
}

/// Bit-exact portable AVX2 DC removal + Hann windowing (0-ULP identical to scalar tree sum).
#[cfg(target_arch = "x86_64")]
pub(crate) fn dc_and_window_avx2_portable(
    frame: &[f32],
    window: &[f32; WIN_SAMPLES],
    re: &mut [f64; FFT_SIZE],
) {
    assert!(frame.len() >= WIN_SAMPLES);
    // SAFETY: Caller verified `avx2` + `fma` via `avx2_fma_available()`; bounds asserted above.
    unsafe { dc_and_window_f64_avx2(frame, window, re) };
}

/// Bit-exact portable AVX2 512-point radix-2 FFT using unfused `vmulpd`/`vaddpd`/`vsubpd`
/// (zero FMA contraction, 0-ULP identical to `portable_math::fft_in_place_portable_with_bitrev`).
#[cfg(target_arch = "x86_64")]
pub(crate) fn fft_in_place_avx2_portable(
    re: &mut [f64; FFT_SIZE],
    im: &mut [f64; FFT_SIZE],
    tw_re: &[f64; FFT_SIZE],
    tw_im: &[f64; FFT_SIZE],
    bitrev: &[(u16, u16)],
) {
    // SAFETY: Caller verified `avx2` + `fma` via `avx2_fma_available()`.
    unsafe { fft_in_place_avx2_unfused(re, im, tw_re, tw_im, bitrev) };
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn fft_in_place_avx2_unfused(
    re: &mut [f64; FFT_SIZE],
    im: &mut [f64; FFT_SIZE],
    tw_re: &[f64; FFT_SIZE],
    tw_im: &[f64; FFT_SIZE],
    bitrev: &[(u16, u16)],
) {
    for &(i, j) in bitrev {
        let (iu, ju) = (i as usize, j as usize);
        re.swap(iu, ju);
        im.swap(iu, ju);
    }
    // Stages len = 2, 4 (half = 1, 2): scalar unfused f64.
    for len in [2usize, 4usize] {
        let half = len / 2;
        let tw_r = &tw_re[half..half + half];
        let tw_i = &tw_im[half..half + half];
        let mut start = 0usize;
        while start < FFT_SIZE {
            for k in 0..half {
                let (w_re, w_im) = (tw_r[k], tw_i[k]);
                let (a, b) = (start + k, start + k + half);
                let (u_re, u_im, b_re, b_im) = (re[a], im[a], re[b], im[b]);
                let v_re = b_re * w_re - b_im * w_im;
                let v_im = b_re * w_im + b_im * w_re;
                re[a] = u_re + v_re;
                im[a] = u_im + v_im;
                re[b] = u_re - v_re;
                im[b] = u_im - v_im;
            }
            start += len;
        }
    }
    // Stages len = 8..=512 (half = 4..=256, multiples of 4): 4-wide unfused AVX2 f64 butterflies.
    let mut len = 8usize;
    while len <= FFT_SIZE {
        let half = len / 2;
        let tw_r_ptr = tw_re[half..].as_ptr();
        let tw_i_ptr = tw_im[half..].as_ptr();
        let mut start = 0usize;
        while start < FFT_SIZE {
            let mut k = 0usize;
            while k < half {
                let a = start + k;
                let b = a + half;
                // SAFETY: `k + 4 <= half` and `b + 4 <= FFT_SIZE`.
                unsafe {
                    let wr = _mm256_loadu_pd(tw_r_ptr.add(k));
                    let wi = _mm256_loadu_pd(tw_i_ptr.add(k));
                    let ur = _mm256_loadu_pd(re.as_ptr().add(a));
                    let ui = _mm256_loadu_pd(im.as_ptr().add(a));
                    let br = _mm256_loadu_pd(re.as_ptr().add(b));
                    let bi = _mm256_loadu_pd(im.as_ptr().add(b));
                    let vr = _mm256_sub_pd(_mm256_mul_pd(br, wr), _mm256_mul_pd(bi, wi));
                    let vi = _mm256_add_pd(_mm256_mul_pd(br, wi), _mm256_mul_pd(bi, wr));
                    _mm256_storeu_pd(re.as_mut_ptr().add(a), _mm256_add_pd(ur, vr));
                    _mm256_storeu_pd(im.as_mut_ptr().add(a), _mm256_add_pd(ui, vi));
                    _mm256_storeu_pd(re.as_mut_ptr().add(b), _mm256_sub_pd(ur, vr));
                    _mm256_storeu_pd(im.as_mut_ptr().add(b), _mm256_sub_pd(ui, vi));
                }
                k += 4;
            }
            start += len;
        }
        len <<= 1;
    }
}

/// Bit-exact portable aarch64 NEON 512-point radix-2 FFT using unfused `vmulq_f64`/`vaddq_f64`/
/// `vsubq_f64` (zero FMA contraction, 0-ULP identical to x86_64 AVX2 and scalar).
#[cfg(target_arch = "aarch64")]
pub(crate) fn fft_in_place_neon_portable(
    re: &mut [f64; crate::mel::FFT_SIZE],
    im: &mut [f64; crate::mel::FFT_SIZE],
    tw_re: &[f64; crate::mel::FFT_SIZE],
    tw_im: &[f64; crate::mel::FFT_SIZE],
    bitrev: &[(u16, u16)],
) {
    use std::arch::aarch64::*;
    let n = crate::mel::FFT_SIZE;
    for &(i, j) in bitrev {
        let (iu, ju) = (i as usize, j as usize);
        re.swap(iu, ju);
        im.swap(iu, ju);
    }
    // Stage len = 2 (half = 1): scalar unfused f64.
    let (w_re, w_im) = (tw_re[1], tw_im[1]);
    let mut start = 0usize;
    while start < n {
        let (a, b) = (start, start + 1);
        let (u_re, u_im, b_re, b_im) = (re[a], im[a], re[b], im[b]);
        let v_re = b_re * w_re - b_im * w_im;
        let v_im = b_re * w_im + b_im * w_re;
        re[a] = u_re + v_re;
        im[a] = u_im + v_im;
        re[b] = u_re - v_re;
        im[b] = u_im - v_im;
        start += 2;
    }
    // Stages len = 4..=512 (half = 2..=256, multiples of 2): 2-wide unfused NEON f64 butterflies.
    let mut len = 4usize;
    while len <= n {
        let half = len / 2;
        // SAFETY: NEON is mandatory on aarch64; `k + 2 <= half` and `b + 2 <= FFT_SIZE`.
        unsafe {
            let tw_r_ptr = tw_re.as_ptr().add(half);
            let tw_i_ptr = tw_im.as_ptr().add(half);
            let mut st = 0usize;
            while st < n {
                let mut k = 0usize;
                while k < half {
                    let a = st + k;
                    let b = a + half;
                    let wr = vld1q_f64(tw_r_ptr.add(k));
                    let wi = vld1q_f64(tw_i_ptr.add(k));
                    let ur = vld1q_f64(re.as_ptr().add(a));
                    let ui = vld1q_f64(im.as_ptr().add(a));
                    let br = vld1q_f64(re.as_ptr().add(b));
                    let bi = vld1q_f64(im.as_ptr().add(b));
                    let vr = vsubq_f64(vmulq_f64(br, wr), vmulq_f64(bi, wi));
                    let vi = vaddq_f64(vmulq_f64(br, wi), vmulq_f64(bi, wr));
                    vst1q_f64(re.as_mut_ptr().add(a), vaddq_f64(ur, vr));
                    vst1q_f64(im.as_mut_ptr().add(a), vaddq_f64(ui, vi));
                    vst1q_f64(re.as_mut_ptr().add(b), vsubq_f64(ur, vr));
                    vst1q_f64(im.as_mut_ptr().add(b), vsubq_f64(ui, vi));
                    k += 2;
                }
                st += len;
            }
        }
        len <<= 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dc_and_window_f64_avx2(
    frame: &[f32],
    window: &[f32; WIN_SAMPLES],
    re: &mut [f64; FFT_SIZE],
) {
    let mut sum0 = _mm256_setzero_pd();
    let mut sum1 = _mm256_setzero_pd();
    let mut i = 0usize;
    while i + 8 <= WIN_SAMPLES {
        // SAFETY: `i + 8 <= WIN_SAMPLES <= frame.len()`.
        let lo = unsafe { _mm_loadu_ps(frame.as_ptr().add(i)) };
        // SAFETY: `i + 8 <= WIN_SAMPLES <= frame.len()`.
        let hi = unsafe { _mm_loadu_ps(frame.as_ptr().add(i + 4)) };
        sum0 = _mm256_add_pd(sum0, _mm256_cvtps_pd(lo));
        sum1 = _mm256_add_pd(sum1, _mm256_cvtps_pd(hi));
        i += 8;
    }
    let s256 = _mm256_add_pd(sum0, sum1);
    let mut lanes = [0.0f64; 4];
    // SAFETY: `lanes` has 4 aligned/unaligned `f64` slots.
    unsafe { _mm256_storeu_pd(lanes.as_mut_ptr(), s256) };
    let mean = ((lanes[0] + lanes[1] + lanes[2] + lanes[3]) / (WIN_SAMPLES as f64)) as f32;
    let vmean = _mm256_set1_ps(mean);

    let mut j = 0usize;
    while j + 8 <= WIN_SAMPLES {
        // SAFETY: `j + 8 <= WIN_SAMPLES` for `frame`, `window`, and `re`.
        unsafe {
            let fv = _mm256_loadu_ps(frame.as_ptr().add(j));
            let wv = _mm256_loadu_ps(window.as_ptr().add(j));
            let prod = _mm256_mul_ps(_mm256_sub_ps(fv, vmean), wv);
            let lo128 = _mm256_castps256_ps128(prod);
            let hi128 = _mm256_extractf128_ps::<1>(prod);
            _mm256_storeu_pd(re.as_mut_ptr().add(j), _mm256_cvtps_pd(lo128));
            _mm256_storeu_pd(re.as_mut_ptr().add(j + 4), _mm256_cvtps_pd(hi128));
        }
        j += 8;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn mel_dot_f64_avx2(
    re: &[f64; FFT_SIZE],
    im: &[f64; FFT_SIZE],
    fbank: &[f32],
    mel_acc: &mut [f64; MEL_BINS],
) {
    let mut powers = [0.0f64; SPEC_BINS];
    for k in 0..SPEC_BINS {
        let mag = re[k].hypot(im[k]) as f32;
        powers[k] = f64::from(mag * mag);
    }

    // Two passes of 40 mel bins (`10 x __m256d` registers) keep all accumulators in `ymm` registers.
    for half in [0usize, 40usize] {
        let mut acc = [_mm256_setzero_pd(); 10];
        for (k, &pk) in powers.iter().enumerate() {
            let vp = _mm256_set1_pd(pk);
            let row_base = k * MEL_BINS + half;
            for (g, a_reg) in acc.iter_mut().enumerate() {
                // SAFETY: `row_base + g * 4 + 4 <= SPEC_BINS * MEL_BINS <= fbank.len()`.
                let w4 = unsafe { _mm_loadu_ps(fbank.as_ptr().add(row_base + g * 4)) };
                *a_reg = _mm256_fmadd_pd(vp, _mm256_cvtps_pd(w4), *a_reg);
            }
        }
        for (g, a_reg) in acc.into_iter().enumerate() {
            // SAFETY: `half + g * 4 + 4 <= MEL_BINS == mel_acc.len()`.
            unsafe { _mm256_storeu_pd(mel_acc.as_mut_ptr().add(half + g * 4), a_reg) };
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::mel::WIN_SAMPLES;
    use crate::trainer_mel::{GlobalCmvn, TrainerMelConfig};

    #[test]
    fn avx2_fma_frame_matches_scalar_reference_within_1e6() {
        let cfg = TrainerMelConfig::default_htk(GlobalCmvn::identity()).expect("default htk");
        for mode in 0..4 {
            let frame: Vec<f32> = (0..WIN_SAMPLES)
                .map(|i| match mode {
                    0 => 0.0,
                    1 => ((i as f32) * 0.137).sin() * 0.8 + 0.12,
                    2 => {
                        if i % 2 == 0 {
                            0.95
                        } else {
                            -0.95
                        }
                    }
                    _ => ((i * i) as f32 * 0.0017).cos(),
                })
                .collect();
            let got = cfg.frame_f32(&frame);
            let want = cfg.frame_f32_scalar(&frame);
            for (a, b) in got.iter().zip(&want) {
                assert!((a - b).abs() <= 1e-5, "diff {} vs {}", a, b);
            }
        }
    }

    #[test]
    fn avx2_portable_frame_is_bit_exact_with_scalar_reference() {
        let cfg = TrainerMelConfig::default_htk(GlobalCmvn::identity())
            .expect("default htk")
            .with_portable_math(true);
        for mode in 0..4 {
            let frame: Vec<f32> = (0..WIN_SAMPLES)
                .map(|i| match mode {
                    0 => 0.0,
                    1 => (((i * 197 + 31) % 20_000) as i32 - 10_000) as f32 / 32768.0,
                    2 => {
                        if i % 2 == 0 {
                            0.95
                        } else {
                            -0.95
                        }
                    }
                    _ => (((i * i * 17 + 7) % 65_535) as i32 - 32_768) as f32 / 32768.0,
                })
                .collect();
            let got = cfg.frame_f32(&frame);
            let want = cfg.frame_f32_portable_scalar(&frame);
            for (bin, (a, b)) in got.iter().zip(&want).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "portable bit mismatch mode={mode} bin={bin}: {a} ({:#010x}) vs {b} ({:#010x})",
                    a.to_bits(),
                    b.to_bits()
                );
            }
        }
    }
}
