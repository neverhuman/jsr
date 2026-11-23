//! Causal, fixed-point-friendly Warm AGC for the recogniser audio path (`H47`).
//!
//! Runs strictly on the recogniser audio path AFTER any Sentinel-0 gate and pre-roll ring
//! (the gate and ring keep raw levels). Default-off (`warm_agc: false`): exact identity
//! no-op with zero per-sample overhead. Default level-tracking time constant is `tau_s = 5.0` s
//! (`τ = 5 s`), paired with instant wake lock via [`WarmAgc::seed_from_preroll`].

use crate::EngineError;

pub const DEFAULT_FRAME_SAMPLES: usize = 160;
pub const DEFAULT_TARGET_RMS: f32 = 0.08;
pub const DEFAULT_TAU_S: f32 = 5.0;

const SAMPLE_RATE_HZ: f32 = 16_000.0;
const INV_I16: f32 = 1.0 / 32768.0;
const KNEE: f32 = 0.90;
const HEADROOM: f32 = 1.0 - KNEE;

/// Configuration for [`WarmAgc`]. Default is disabled (`warm_agc: false`, `tau_s: 5.0`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WarmAgcConfig {
    pub warm_agc: bool,
    pub target_rms: f32,
    pub min_gain_db: f32,
    pub max_gain_db: f32,
    pub speech_floor_dbfs: f32,
    /// Level-tracking time constant in seconds (default `5.0 s`, `τ = 5 s`).
    pub tau_s: f32,
    /// Attack time constant in seconds when speech gets louder (default `5.0 s`).
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
        let pos = |v: f32| v.is_finite() && v > 0.0;
        let ok = pos(self.target_rms)
            && self.target_rms <= 1.0
            && self.min_gain_db.is_finite()
            && self.max_gain_db.is_finite()
            && self.min_gain_db <= self.max_gain_db
            && self.speech_floor_dbfs.is_finite()
            && self.speech_floor_dbfs <= 0.0
            && pos(self.tau_s)
            && pos(self.attack_tau_s)
            && self.frame_samples > 0;
        if !ok {
            return Err(EngineError::Format(format!(
                "invalid WarmAgcConfig: {self:?}"
            )));
        }
        Ok(())
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

/// C1-continuous rational soft limiter: exact identity for `|x| <= 0.9`, strictly in `[-1, 1]`.
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

/// Causal speech-gated Warm AGC carrying level state across utterances and sleep cycles.
#[derive(Debug, Clone)]
pub struct WarmAgc {
    cfg: WarmAgcConfig,
    gain_bounds: (f32, f32),
    coeffs: (f32, f32),
    speech_floor_sq: f32,
    gain: f32,
    level_rms: f32,
    warm: bool,
    sleeping: bool,
    acc_sq: f32,
    acc_count: usize,
}

impl WarmAgc {
    pub fn new(cfg: WarmAgcConfig) -> Result<Self, EngineError> {
        cfg.validate()?;
        let gain_bounds = (db_to_linear(cfg.min_gain_db), db_to_linear(cfg.max_gain_db));
        let floor_rms = db_to_linear(cfg.speech_floor_dbfs);
        let dt = (cfg.frame_samples as f32) / SAMPLE_RATE_HZ;
        let coeffs = (
            (1.0 - (-dt / cfg.attack_tau_s).exp()).clamp(1e-6, 1.0),
            (1.0 - (-dt / cfg.tau_s).exp()).clamp(1e-6, 1.0),
        );
        let gain = 1.0f32.clamp(gain_bounds.0, gain_bounds.1);
        Ok(Self {
            cfg,
            gain_bounds,
            coeffs,
            speech_floor_sq: floor_rms * floor_rms,
            gain,
            level_rms: cfg.target_rms / gain,
            warm: false,
            sleeping: false,
            acc_sq: 0.0,
            acc_count: 0,
        })
    }

    pub fn disabled() -> Self {
        Self::new(WarmAgcConfig::default()).expect("valid default config")
    }
    pub fn enabled_default() -> Self {
        Self::new(WarmAgcConfig::enabled()).expect("valid enabled config")
    }
    pub fn is_enabled(&self) -> bool {
        self.cfg.warm_agc
    }
    pub fn is_warm(&self) -> bool {
        self.warm
    }
    pub fn is_sleeping(&self) -> bool {
        self.sleeping
    }
    pub fn config(&self) -> &WarmAgcConfig {
        &self.cfg
    }
    pub fn gain_linear(&self) -> f32 {
        self.gain
    }
    pub fn gain_db(&self) -> f32 {
        linear_to_db(self.gain)
    }
    pub fn level_rms(&self) -> f32 {
        self.level_rms
    }

    /// Utterance boundary: clears partial-frame carry while preserving warm level/gain state.
    pub fn reset_utterance(&mut self) {
        (self.acc_sq, self.acc_count, self.sleeping) = (0.0, 0, false);
    }

    /// Freeze learned level/gain state cleanly across sleep.
    pub fn on_sleep(&mut self) {
        (self.acc_sq, self.acc_count, self.sleeping) = (0.0, 0, true);
    }

    /// Full cold reset back to unity gain (`0 dB`).
    pub fn hard_reset(&mut self) {
        self.gain = 1.0f32.clamp(self.gain_bounds.0, self.gain_bounds.1);
        self.level_rms = self.cfg.target_rms / self.gain;
        (self.warm, self.sleeping, self.acc_sq, self.acc_count) = (false, false, 0.0, 0);
    }

    /// Seed initial or post-sleep gain from a pre-roll ring (e.g. 500 ms) of unit-range `f32`.
    pub fn seed_from_preroll(&mut self, samples: &[f32]) {
        if !self.cfg.warm_agc || samples.is_empty() {
            return;
        }
        self.seed_from_frames(samples.chunks(self.cfg.frame_samples).map(|c| {
            let s: f32 = c.iter().map(|&x| x * x).sum();
            (s / (c.len().max(1) as f32), c.len())
        }));
    }

    /// Seed gain from a raw PCM16 pre-roll ring without modifying the caller's buffer.
    pub fn seed_from_preroll_i16(&mut self, pcm: &[i16]) {
        if !self.cfg.warm_agc || pcm.is_empty() {
            return;
        }
        self.seed_from_frames(pcm.chunks(self.cfg.frame_samples).map(|c| {
            let s: f32 = c.iter().map(|&v| (f32::from(v) * INV_I16).powi(2)).sum();
            (s / (c.len().max(1) as f32), c.len())
        }));
    }

    fn seed_from_frames<I: Iterator<Item = (f32, usize)> + Clone>(&mut self, frames: I) {
        (self.acc_sq, self.acc_count, self.sleeping) = (0.0, 0, false);
        let peak_sq = frames
            .clone()
            .filter_map(|(m, _)| m.is_finite().then_some(m))
            .fold(0.0f32, f32::max);
        if peak_sq <= self.speech_floor_sq {
            return;
        }
        let gate_sq = self.speech_floor_sq.max(peak_sq * 0.0625);
        let (mut sum_sq, mut count) = (0.0f64, 0usize);
        for (m, len) in frames {
            if m.is_finite() && m >= gate_sq {
                sum_sq += f64::from(m) * (len as f64);
                count += len;
            }
        }
        let rms = (sum_sq / (count.max(1) as f64)).sqrt() as f32;
        if count > 0 && rms.is_finite() && rms > 0.0 {
            self.level_rms = rms;
            self.gain = (self.cfg.target_rms / rms).clamp(self.gain_bounds.0, self.gain_bounds.1);
            self.warm = true;
        }
    }

    /// Process unit-range `f32` audio in place. Exact identity no-op when `!warm_agc`.
    pub fn process_chunk(&mut self, samples: &mut [f32]) {
        self.run_frames(samples, |x, g| {
            let raw = *x;
            *x = soft_limit(raw * g);
            raw * raw
        });
    }

    /// Process PCM16 audio in place. Exact identity no-op when `!warm_agc`.
    pub fn process_i16(&mut self, pcm: &mut [i16]) {
        self.run_frames(pcm, |s, g| {
            let raw = f32::from(*s) * INV_I16;
            *s = (soft_limit(raw * g) * 32767.0)
                .round()
                .clamp(-32768.0, 32767.0) as i16;
            raw * raw
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
    fn run_frames<T>(&mut self, buf: &mut [T], mut step: impl FnMut(&mut T, f32) -> f32) {
        if !self.cfg.warm_agc || buf.is_empty() {
            return;
        }
        self.sleeping = false;
        let mut idx = 0;
        while idx < buf.len() {
            let take = (self.cfg.frame_samples - self.acc_count).min(buf.len() - idx);
            let (g, mut sq) = (self.gain, 0.0f32);
            for item in &mut buf[idx..idx + take] {
                sq += step(item, g);
            }
            self.acc_sq += sq;
            self.acc_count += take;
            if self.acc_count >= self.cfg.frame_samples {
                let mean_sq = self.acc_sq / (self.acc_count as f32);
                (self.acc_sq, self.acc_count) = (0.0, 0);
                if mean_sq.is_finite() && mean_sq > self.speech_floor_sq {
                    let rms = mean_sq.sqrt();
                    let c = if rms > self.level_rms {
                        self.coeffs.0
                    } else {
                        self.coeffs.1
                    };
                    self.level_rms = (self.level_rms + c * (rms - self.level_rms)).max(1e-12);
                    self.gain = (self.cfg.target_rms / self.level_rms)
                        .clamp(self.gain_bounds.0, self.gain_bounds.1);
                    self.warm = true;
                }
            }
            idx += take;
        }
    }
}
