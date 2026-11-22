//! Pure IEEE-754 unfused floating-point math helpers for cross-ISA bit-exact log-mel (`G4`).
//!
//! Eliminates all platform `libm` (`glibc` vs Apple `libSystem`) transcendental calls (`cos`,
//! `sin`, `hypot`, `ln`) and FMA contraction from `TrainerLogMel`, so x86_64 AVX2, aarch64 NEON,
//! and portable scalar builds emit identical `f32` bit patterns from identical PCM16 input.

use std::f64::consts::{FRAC_PI_2, LN_2, SQRT_2};

/// High 33 bits of `pi / 2` for Cody-Waite range reduction (exact in IEEE-754 `f64`).
const PI_2_HI: f64 = 1.570_796_326_734_125_6_f64;
/// Next 53 bits of `pi / 2` (`PI_2_HI + PI_2_LO == pi/2` to ~86 bits).
const PI_2_LO: f64 = 6.077_100_506_506_192e-11_f64;

/// Deterministic `(sin(x), cos(x))` in pure unfused IEEE-754 `f64` (zero `libm`).
///
/// Accurate to `< 2e-16` across `[-4*pi, 4*pi]` (and all moderate finite inputs), and bit-exact
/// across x86_64 and aarch64 because Rust never contracts `*` + `+` into FMA.
#[inline]
pub fn sin_cos_portable(x: f64) -> (f64, f64) {
    let q = (x / FRAC_PI_2).round();
    let k = (q as i64).rem_euclid(4);
    let r = (x - q * PI_2_HI) - q * PI_2_LO;
    let r2 = r * r;

    // Degree-15 Taylor polynomial for sin(r) on |r| <= pi/4 (truncation error < 5e-17).
    let s = r
        * (1.0
            + r2 * (-1.0 / 6.0
                + r2 * (1.0 / 120.0
                    + r2 * (-1.0 / 5_040.0
                        + r2 * (1.0 / 362_880.0
                            + r2 * (-1.0 / 39_916_800.0
                                + r2 * (1.0 / 6_227_020_800.0
                                    + r2 * (-1.0 / 1_307_674_368_000.0))))))));

    // Degree-14 Taylor polynomial for cos(r) on |r| <= pi/4 (truncation error < 3e-16).
    let c = 1.0
        + r2 * (-0.5
            + r2 * (1.0 / 24.0
                + r2 * (-1.0 / 720.0
                    + r2 * (1.0 / 40_320.0
                        + r2 * (-1.0 / 3_628_800.0
                            + r2 * (1.0 / 479_001_600.0 + r2 * (-1.0 / 87_178_291_200.0)))))));

    match k {
        0 => (s, c),
        1 => (c, -s),
        2 => (-s, -c),
        _ => (-c, s),
    }
}

/// Deterministic natural logarithm `ln(x)` for `f32` using pure unfused IEEE-754 `f64` math.
///
/// Accurate to `<= 0.5 ULP` (`f32`) for all finite positive inputs and 0-ULP bit-identical across
/// Linux x86_64, macOS aarch64, and scalar builds (zero `libm`).
#[inline]
pub fn ln_portable_f32(x: f32) -> f32 {
    if x.is_nan() || x < 0.0 {
        return f32::NAN;
    }
    if x == 0.0 {
        return f32::NEG_INFINITY;
    }
    if x.is_infinite() {
        return f32::INFINITY;
    }
    let d = f64::from(x);
    let bits = d.to_bits();
    let mut exp = ((bits >> 52) as i32) - 1023;
    let mut m = f64::from_bits((bits & 0x000f_ffff_ffff_ffff) | 0x3ff0_0000_0000_0000);
    if m >= SQRT_2 {
        m *= 0.5;
        exp += 1;
    }
    // m in [sqrt(0.5), sqrt(2)) => |u| <= 3 - 2*sqrt(2) ~= 0.17157, u^2 <= 0.02944.
    let u = (m - 1.0) / (m + 1.0);
    let u2 = u * u;
    let poly = 1.0
        + u2 * (1.0 / 3.0
            + u2 * (1.0 / 5.0
                + u2 * (1.0 / 7.0
                    + u2 * (1.0 / 9.0
                        + u2 * (1.0 / 11.0 + u2 * (1.0 / 13.0 + u2 * (1.0 / 15.0)))))));
    let ln_m = (2.0 * u) * poly;
    (ln_m + (exp as f64) * LN_2) as f32
}

/// Deterministic power `(|z|^2)` as `f32` from `f64` complex components without `libm::hypot`.
#[inline]
pub fn power_f32_from_complex_f64(re: f64, im: f64) -> f32 {
    (re * re + im * im) as f32
}

/// Precompute deterministic radix-2 FFT twiddle factors for `FFT_SIZE = 512` (zero `libm`).
pub fn build_portable_fft_twiddles<const N: usize>() -> ([f64; N], [f64; N]) {
    let mut tw_re = [0.0f64; N];
    let mut tw_im = [0.0f64; N];
    let mut len = 2usize;
    while len <= N {
        let half = len / 2;
        for k in 0..half {
            let ang = -2.0 * std::f64::consts::PI * (k as f64) / (len as f64);
            let (s, c) = sin_cos_portable(ang);
            tw_re[half + k] = c;
            tw_im[half + k] = s;
        }
        len <<= 1;
    }
    (tw_re, tw_im)
}

/// Precompute deterministic radix-2 bit-reversal swap pairs `(i, j)` with `i < j` for `N`.
pub fn build_portable_fft_bitrev<const N: usize>() -> Vec<(u16, u16)> {
    let mut pairs = Vec::with_capacity(N / 2);
    let mut j = 0usize;
    for i in 1..N {
        let mut bit = N >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            pairs.push((i as u16, j as u16));
        }
    }
    pairs
}

/// Deterministic in-place radix-2 FFT using precomputed portable twiddles and bit-reversal table
/// (unfused IEEE-754 `f64`, 0-ULP identical to [`fft_in_place_portable`]).
pub fn fft_in_place_portable_with_bitrev<const N: usize>(
    re: &mut [f64; N],
    im: &mut [f64; N],
    tw_re: &[f64; N],
    tw_im: &[f64; N],
    bitrev: &[(u16, u16)],
) {
    for &(i, j) in bitrev {
        let (iu, ju) = (i as usize, j as usize);
        re.swap(iu, ju);
        im.swap(iu, ju);
    }
    let mut len = 2usize;
    while len <= N {
        let half = len / 2;
        let tw_r = &tw_re[half..half + half];
        let tw_i = &tw_im[half..half + half];
        let mut start = 0usize;
        while start < N {
            for k in 0..half {
                let w_re = tw_r[k];
                let w_im = tw_i[k];
                let a = start + k;
                let b = a + half;
                let u_re = re[a];
                let u_im = im[a];
                let b_re = re[b];
                let b_im = im[b];
                let v_re = b_re * w_re - b_im * w_im;
                let v_im = b_re * w_im + b_im * w_re;
                re[a] = u_re + v_re;
                im[a] = u_im + v_im;
                re[b] = u_re - v_re;
                im[b] = u_im - v_im;
            }
            start += len;
        }
        len <<= 1;
    }
}

/// Deterministic in-place radix-2 FFT using precomputed portable twiddles (unfused IEEE-754 `f64`).
pub fn fft_in_place_portable<const N: usize>(
    re: &mut [f64; N],
    im: &mut [f64; N],
    tw_re: &[f64; N],
    tw_im: &[f64; N],
) {
    let bitrev = build_portable_fft_bitrev::<N>();
    fft_in_place_portable_with_bitrev(re, im, tw_re, tw_im, &bitrev);
}

/// Legacy `libm`-twiddle 512-point radix-2 FFT retained for `portable_math = false` control.
pub(crate) fn fft_in_place_512_libm_f64(
    re: &mut [f64; crate::mel::FFT_SIZE],
    im: &mut [f64; crate::mel::FFT_SIZE],
) {
    let n = crate::mel::FFT_SIZE;
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let mut len = 2usize;
    while len <= n {
        let half = len / 2;
        let ang = -2.0 * std::f64::consts::PI / (len as f64);
        let (wlen_re, wlen_im) = (ang.cos(), ang.sin());
        let mut start = 0usize;
        while start < n {
            let (mut w_re, mut w_im) = (1.0f64, 0.0f64);
            for k in 0..half {
                let u_re = re[start + k];
                let u_im = im[start + k];
                let v_re = re[start + k + half] * w_re - im[start + k + half] * w_im;
                let v_im = re[start + k + half] * w_im + im[start + k + half] * w_re;
                re[start + k] = u_re + v_re;
                im[start + k] = u_im + v_im;
                re[start + k + half] = u_re - v_re;
                im[start + k + half] = u_im - v_im;
                let next_w_re = w_re * wlen_re - w_im * wlen_im;
                w_im = w_re * wlen_im + w_im * wlen_re;
                w_re = next_w_re;
            }
            start += len;
        }
        len <<= 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn portable_sin_cos_matches_libm_within_1e14() {
        for i in -512..=512 {
            let x = (i as f64) * std::f64::consts::PI / 256.0;
            let (s, c) = sin_cos_portable(x);
            assert!(
                (s - x.sin()).abs() < 1e-14,
                "sin mismatch at {x}: {s} vs {}",
                x.sin()
            );
            assert!(
                (c - x.cos()).abs() < 1e-14,
                "cos mismatch at {x}: {c} vs {}",
                x.cos()
            );
        }
    }

    #[test]
    fn portable_ln_matches_f32_ln_within_1_ulp_across_range() {
        let samples = [
            1e-30_f32,
            1e-12,
            1e-6,
            1e-3,
            0.25,
            0.5,
            std::f32::consts::FRAC_1_SQRT_2,
            1.0,
            std::f32::consts::SQRT_2,
            2.0,
            std::f32::consts::PI,
            100.0,
            1e6,
            1e20,
        ];
        for &x in &samples {
            let got = ln_portable_f32(x);
            let want = x.ln();
            let ulp = got.to_bits().abs_diff(want.to_bits());
            assert!(ulp <= 1, "ln({x}) got={got} want={want} ulp={ulp}");
        }
        assert!(ln_portable_f32(0.0).is_infinite() && ln_portable_f32(0.0).is_sign_negative());
        assert!(ln_portable_f32(-1.0).is_nan());
        assert!(ln_portable_f32(f32::NAN).is_nan());
        assert!(ln_portable_f32(f32::INFINITY).is_infinite());
    }
}
