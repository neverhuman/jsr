//! Z1: fixed whole-utterance RMS level normalisation for the final-text (commit) pass.
//!
//! The closed utterance's int16 PCM is scaled by one gain so its RMS lands on
//! [`Z1_TARGET_RMS`], then rounded half-to-even and clipped to int16. The gain is clamped to
//! `[-12 dB, +40 dB]`, which also bounds the all-zero / near-silent case (rms is floored at
//! `1e-9`, so silence gets the +40 dB cap applied to zeros: the output stays all zero).
//!
//! This is a bit-exact port of the Python reference that produced the codex-o3-audit
//! `z1-rc2` / `z1-clean-guard` probes (`run_z1.py`, `run_guard.py`):
//!
//! ```python
//! f = x.astype(np.float64)
//! rms = np.sqrt(np.mean(f * f))
//! gain = float(np.clip(1928 / max(rms, 1e-9), 10 ** (-12 / 20), 10 ** (40 / 20)))
//! y = np.round(f * gain)                       # half to even
//! out = np.clip(y, -32768, 32767).astype(np.int16)
//! ```
//!
//! The sum of squares is accumulated in `f64`; every partial sum is an exact integer below
//! 2^53 for inputs up to about 8.3 M full-scale samples (over 8 minutes at 16 kHz), so the
//! result does not depend on numpy's pairwise summation order for any supported clip length.

/// Target RMS on the int16 scale.
pub const Z1_TARGET_RMS: f64 = 1928.0;
/// Lower gain clamp in dB.
pub const Z1_MIN_GAIN_DB: f64 = -12.0;
/// Upper gain clamp in dB.
pub const Z1_MAX_GAIN_DB: f64 = 40.0;
/// Floor applied to the measured RMS before dividing.
pub const Z1_RMS_FLOOR: f64 = 1e-9;

/// Result of one normalisation: the gain applied and how many samples hit the int16 rails.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RmsNormStats {
    pub gain: f64,
    pub clipped: usize,
}

/// Whole-utterance gain for `pcm`. Empty input returns exactly `1.0`.
#[must_use]
pub fn rms_gain(pcm: &[i16]) -> f64 {
    if pcm.is_empty() {
        return 1.0;
    }
    let sum_sq: f64 = pcm
        .iter()
        .map(|&s| {
            let f = f64::from(s);
            f * f
        })
        .sum();
    let rms = (sum_sq / pcm.len() as f64).sqrt();
    let lo = 10f64.powf(Z1_MIN_GAIN_DB / 20.0);
    let hi = 10f64.powf(Z1_MAX_GAIN_DB / 20.0);
    (Z1_TARGET_RMS / rms.max(Z1_RMS_FLOOR)).clamp(lo, hi)
}

/// Scale `pcm` in place by [`rms_gain`], round half-to-even, clip to int16.
pub fn rms_normalize_in_place(pcm: &mut [i16]) -> RmsNormStats {
    let gain = rms_gain(pcm);
    let mut clipped = 0usize;
    for s in pcm.iter_mut() {
        let y = (f64::from(*s) * gain).round_ties_even();
        if !(-32768.0..=32767.0).contains(&y) {
            clipped += 1;
        }
        *s = y.clamp(-32768.0, 32767.0) as i16;
    }
    RmsNormStats { gain, clipped }
}

/// Allocating form of [`rms_normalize_in_place`].
#[must_use]
pub fn rms_normalize(pcm: &[i16]) -> (Vec<i16>, RmsNormStats) {
    let mut out = pcm.to_vec();
    let stats = rms_normalize_in_place(&mut out);
    (out, stats)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn silence_and_empty_are_finite_and_unchanged() {
        let (y, st) = rms_normalize(&[0i16; 4000]);
        assert!(y.iter().all(|&s| s == 0));
        assert!(st.gain.is_finite());
        assert_eq!(st.gain, 10f64.powf(2.0));
        assert_eq!(st.clipped, 0);
        let (e, st) = rms_normalize(&[]);
        assert!(e.is_empty());
        assert_eq!(st.gain, 1.0);
    }

    #[test]
    fn gain_is_clamped_both_ways() {
        let (_, st) = rms_normalize(&[1i16; 100]);
        assert_eq!(st.gain, 10f64.powf(2.0)); // 1928x wanted, +40 dB cap
        let (y, st) = rms_normalize(&[i16::MAX; 100]);
        assert_eq!(st.gain, 10f64.powf(-0.6)); // -12 dB floor
        assert!(y.iter().all(|&s| s > 0));
    }

    #[test]
    fn rails_clip_and_extreme_values_do_not_wrap() {
        let mut x = vec![0i16; 1000];
        x[0] = i16::MIN;
        x[1] = i16::MAX;
        x[2] = 1;
        let st = rms_normalize_in_place(&mut x);
        assert!(st.gain.is_finite());
        assert_eq!(x[0], i16::MIN);
        assert_eq!(x[1], i16::MAX);
        assert!(st.clipped >= 2);
    }

    #[test]
    fn ties_round_half_to_even() {
        // gain 1928/rms for a constant signal c is 1928/|c|; c=1928*2 -> gain 0.5.
        let (y, st) = rms_normalize(&[3856i16, 3856, -3856, -3856]);
        assert_eq!(st.gain, 0.5);
        assert_eq!(y, [1928, 1928, -1928, -1928]);
        let (y, st) = rms_normalize(&[2570i16; 4]); // gain = 1928/2570
        assert!(st.gain < 1.0);
        assert_eq!(y, [1928; 4]);
    }

    #[test]
    fn idempotent_within_rounding() {
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let x: Vec<i16> = (0..32000)
            .map(|_| {
                state ^= state >> 12;
                state ^= state << 25;
                state ^= state >> 27;
                ((state.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 48) as i16) / 40
            })
            .collect();
        let (once, s1) = rms_normalize(&x);
        let (twice, s2) = rms_normalize(&once);
        assert!((s2.gain - 1.0).abs() < 1e-3, "second gain {}", s2.gain);
        assert!(s1.clipped == 0 && s2.clipped == 0);
        assert!(once
            .iter()
            .zip(&twice)
            .all(|(a, b)| (i32::from(*a) - i32::from(*b)).abs() <= 1));
    }
}
