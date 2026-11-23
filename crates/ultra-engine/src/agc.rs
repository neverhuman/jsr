//! Causal, fixed-point-friendly Warm AGC for the recogniser audio path (`H47`).
//!
//! Runs strictly on the recogniser audio path AFTER any Sentinel-0 gate and pre-roll ring.
//! Default-off (`warm_agc: false`): exact bit-identical passthrough with zero overhead.
//! Tracks power-domain speech level (`E[x^2]`, `tau_s = 5.0` s) toward LibriSpeech median
//! speech level (`-24.6 dBFS`, `target_rms ≈ 0.0589`) with smooth per-frame gain ramp.

use crate::EngineError;

pub const DEFAULT_FRAME_SAMPLES: usize = 160;
/// LibriSpeech training median speech RMS (`10^(-24.6 / 20)` ≈ `0.058884`, `-24.6 dBFS`).
pub const DEFAULT_TARGET_RMS: f32 = 0.058_884_366;
pub const DEFAULT_TAU_S: f32 = 5.0;
pub const DEFAULT_GAIN_SMOOTH_S: f32 = 0.05;
/// Minimum active 10 ms speech frames required in a pre-roll before cold seeding overrides unity.
pub const MIN_SEED_ACTIVE_FRAMES: usize = 3;

const SAMPLE_RATE_HZ: f64 = 16_000.0;
const INV_I16: f32 = 1.0 / 32768.0;
const KNEE: f32 = 0.90;
const HEADROOM: f32 = 1.0 - KNEE;

/// Configuration for [`WarmAgc`]. Default is disabled (`warm_agc: false`, `-24.6 dBFS`, `tau_s: 5.0`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WarmAgcConfig {
    pub warm_agc: bool,
    pub target_rms: f32,
    pub min_gain_db: f32,
    pub max_gain_db: f32,
    pub speech_floor_dbfs: f32,
    pub tau_s: f32,
    pub attack_tau_s: f32,
    pub frame_samples: usize,
}

impl Default for WarmAgcConfig {
    fn default() -> Self {
        Self {
            warm_agc: false,
            target_rms: DEFAULT_TARGET_RMS,
            min_gain_db: -12.0,
            max_gain_db: 40.0,
            speech_floor_dbfs: -65.0,
            tau_s: DEFAULT_TAU_S,
            attack_tau_s: DEFAULT_TAU_S,
            frame_samples: DEFAULT_FRAME_SAMPLES,
        }
    }
}

impl WarmAgcConfig {
    pub fn enabled() -> Self {
        Self {
            warm_agc: true,
            ..Self::default()
        }
    }

    pub fn validate(&self) -> Result<(), EngineError> {
        let p = |v: f32| v.is_finite() && v > 0.0;
        let ok = p(self.target_rms)
            && self.target_rms <= 1.0
            && p(self.tau_s)
            && p(self.attack_tau_s)
            && self.min_gain_db.is_finite()
            && self.max_gain_db.is_finite()
            && self.min_gain_db <= self.max_gain_db
            && self.speech_floor_dbfs.is_finite()
            && self.speech_floor_dbfs <= 0.0
            && self.frame_samples > 0;
        ok.then_some(())
            .ok_or_else(|| EngineError::Format(format!("invalid WarmAgcConfig: {self:?}")))
    }
}

#[inline]
pub fn db_to_linear(db: f32) -> f32 {
    10.0f32.powf(db * 0.05)
}

#[inline]
pub fn linear_to_db(lin: f32) -> f32 {
    20.0 * lin.max(1e-12).log10()
}

#[inline(always)]
fn soft_limit(x: f32) -> f32 {
    let a = x.abs();
    if a <= KNEE {
        return x;
    }
    if !a.is_finite() {
        return if x.is_nan() { 0.0 } else { x.signum() };
    }
    let e = a - KNEE;
    (KNEE + HEADROOM * (e / (e + HEADROOM)))
        .copysign(x)
        .clamp(-1.0, 1.0)
}

/// Causal speech-gated power-domain Warm AGC carrying level state across utterances and sleep.
#[derive(Debug, Clone)]
pub struct WarmAgc {
    cfg: WarmAgcConfig,
    gain_bounds: (f32, f32),
    coeffs: (f64, f64),
    gain_alpha: f32,
    speech_floor_sq: f64,
    gains: (f32, f32),
    level_sq: f64,
    flags: (bool, bool),
    acc: (f64, usize),
}

impl WarmAgc {
    pub fn new(cfg: WarmAgcConfig) -> Result<Self, EngineError> {
        cfg.validate()?;
        let gain_bounds = (db_to_linear(cfg.min_gain_db), db_to_linear(cfg.max_gain_db));
        let floor_rms = f64::from(db_to_linear(cfg.speech_floor_dbfs));
        let dt = (cfg.frame_samples as f64) / SAMPLE_RATE_HZ;
        let alpha = |t: f32| (1.0 - (-dt / f64::from(t)).exp()).clamp(1e-7, 1.0);
        let g = 1.0f32.clamp(gain_bounds.0, gain_bounds.1);
        let rms = f64::from(cfg.target_rms) / f64::from(g);
        Ok(Self {
            cfg,
            gain_bounds,
            coeffs: (alpha(cfg.attack_tau_s), alpha(cfg.tau_s)),
            gain_alpha: alpha(DEFAULT_GAIN_SMOOTH_S) as f32,
            speech_floor_sq: floor_rms * floor_rms,
            gains: (g, g),
            level_sq: rms * rms,
            flags: (false, false),
            acc: (0.0, 0),
        })
    }

    pub fn disabled() -> Self {
        Self::new(WarmAgcConfig::default()).expect("default")
    }
    pub fn enabled_default() -> Self {
        Self::new(WarmAgcConfig::enabled()).expect("enabled")
    }
    pub fn is_enabled(&self) -> bool {
        self.cfg.warm_agc
    }
    pub fn is_warm(&self) -> bool {
        self.flags.0
    }
    pub fn is_sleeping(&self) -> bool {
        self.flags.1
    }
    pub fn config(&self) -> &WarmAgcConfig {
        &self.cfg
    }
    pub fn gain_linear(&self) -> f32 {
        self.gains.1
    }
    pub fn gain_db(&self) -> f32 {
        linear_to_db(self.gains.1)
    }
    pub fn level_rms(&self) -> f32 {
        self.level_sq.sqrt() as f32
    }

    pub fn reset_utterance(&mut self) {
        (self.acc, self.flags.1, self.gains.0) = ((0.0, 0), false, self.gains.1);
    }
    pub fn on_sleep(&mut self) {
        (self.acc, self.flags.1, self.gains.0) = ((0.0, 0), true, self.gains.1);
    }
    pub fn hard_reset(&mut self) {
        let g = 1.0f32.clamp(self.gain_bounds.0, self.gain_bounds.1);
        let rms = f64::from(self.cfg.target_rms) / f64::from(g);
        (self.level_sq, self.gains, self.flags, self.acc) =
            (rms * rms, (g, g), (false, false), (0.0, 0));
    }
    pub fn seed_from_preroll(&mut self, samples: &[f32]) {
        self.seed_slice(samples, |&x| f64::from(x));
    }
    pub fn seed_from_preroll_i16(&mut self, pcm: &[i16]) {
        self.seed_slice(pcm, |&v| f64::from(v) * f64::from(INV_I16));
    }

    fn seed_slice<T>(&mut self, buf: &[T], to_f64: impl Fn(&T) -> f64) {
        if !self.cfg.warm_agc || buf.is_empty() {
            return;
        }
        self.reset_utterance();
        if self.flags.0 {
            return;
        }
        let msq =
            |c: &[T]| c.iter().map(|v| to_f64(v).powi(2)).sum::<f64>() / (c.len().max(1) as f64);
        let peak = buf
            .chunks(self.cfg.frame_samples)
            .map(&msq)
            .filter(|m| m.is_finite())
            .fold(0.0, f64::max);
        if peak <= self.speech_floor_sq {
            return;
        }
        let gate = self.speech_floor_sq.max(peak * 0.0625);
        let (mut sum_sq, mut samples_n, mut active) = (0.0f64, 0usize, 0usize);
        for c in buf.chunks(self.cfg.frame_samples) {
            let m = msq(c);
            if m.is_finite() && m >= gate {
                (sum_sq, samples_n, active) = (
                    sum_sq + m * (c.len() as f64),
                    samples_n + c.len(),
                    active + 1,
                );
            }
        }
        if active < MIN_SEED_ACTIVE_FRAMES || samples_n == 0 {
            return;
        }
        let ms = sum_sq / (samples_n as f64);
        if ms.is_finite() && ms > self.speech_floor_sq {
            let g = (self.cfg.target_rms / (ms.sqrt() as f32))
                .clamp(self.gain_bounds.0, self.gain_bounds.1);
            (self.level_sq, self.gains, self.flags.0) = (ms, (g, g), true);
        }
    }

    /// Process unit-range `f32` audio in place. Exact identity no-op when `!warm_agc`.
    pub fn process_chunk(&mut self, samples: &mut [f32]) {
        self.run_frames(samples, |x, g| {
            let r = *x;
            *x = soft_limit(r * g);
            f64::from(r) * f64::from(r)
        });
    }

    /// Process PCM16 audio in place. Exact identity no-op when `!warm_agc`.
    pub fn process_i16(&mut self, pcm: &mut [i16]) {
        self.run_frames(pcm, |s, g| {
            let r = f32::from(*s) * INV_I16;
            *s = (soft_limit(r * g) * 32768.0)
                .round()
                .clamp(-32768.0, 32767.0) as i16;
            f64::from(r) * f64::from(r)
        });
    }

    /// Convert raw PCM16 to unit-range `f32` into reusable scratch `out`, applying AGC if enabled.
    pub fn process_i16_into_f32(&mut self, pcm: &[i16], out: &mut Vec<f32>) {
        out.clear();
        out.extend(pcm.iter().map(|&s| f32::from(s) * INV_I16));
        if self.cfg.warm_agc {
            self.process_chunk(out);
        }
    }

    #[inline]
    fn run_frames<T>(&mut self, buf: &mut [T], mut step: impl FnMut(&mut T, f32) -> f64) {
        if !self.cfg.warm_agc || buf.is_empty() {
            return;
        }
        self.flags.1 = false;
        let (f, inv_f, mut idx) = (
            self.cfg.frame_samples,
            1.0 / (self.cfg.frame_samples as f32),
            0,
        );
        while idx < buf.len() {
            let (pos, take) = (self.acc.1, (f - self.acc.1).min(buf.len() - idx));
            let dg = (self.gains.1 - self.gains.0) * inv_f;
            let (mut g, mut sq) = (self.gains.0 + dg * (pos as f32), 0.0f64);
            for item in &mut buf[idx..idx + take] {
                sq += step(item, g);
                g += dg;
            }
            self.acc = (self.acc.0 + sq, self.acc.1 + take);
            if self.acc.1 >= f {
                let ms = self.acc.0 / (self.acc.1 as f64);
                self.acc = (0.0, 0);
                if ms.is_finite() && ms > self.speech_floor_sq {
                    let c = if ms > self.level_sq {
                        self.coeffs.0
                    } else {
                        self.coeffs.1
                    };
                    (self.level_sq, self.flags.0) =
                        ((self.level_sq + c * (ms - self.level_sq)).max(1e-18), true);
                }
                let desired = (self.cfg.target_rms / (self.level_sq.sqrt() as f32))
                    .clamp(self.gain_bounds.0, self.gain_bounds.1);
                let next = (self.gains.1 + self.gain_alpha * (desired - self.gains.1))
                    .clamp(self.gain_bounds.0, self.gain_bounds.1);
                self.gains = (self.gains.1, next);
            }
            idx += take;
        }
    }
}
