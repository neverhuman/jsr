//! Exact R1 trainer-parity streaming 80-bin causal log-mel + global CMVN (`ultra.r1-export/v1`).
//!
//! Matches `python/ultra/train/features.py` and `python/ultra/train/bundle_infer.py`:
//! 1. Causal framing: frame `i` covers samples `[160 i, 160 i + 400)` (`16 kHz`, no padding;
//!    a frame exists only once all 400 samples have arrived).
//! 2. Per-frame DC removal (`frame - mean(frame)` in `f32`).
//! 3. Symmetric 400-point Hann window (`numpy.hanning(400).astype(float32)`).
//! 4. Zero-padded 512-point RFFT power spectrum `|rfft|^2` (`257` bins).
//! 5. Matrix product `mel = power @ mel_fbank` using the bundle's checksummed `257 x 80`
//!    row-major little-endian `mel_fbank.f32` matrix.
//! 6. `logmel = ln(mel + log_floor)` (`log_floor = 1e-6`) followed by global affine CMVN
//!    `(logmel - mean) * istd`.

use std::path::Path;

use crate::mel::{FFT_SIZE, HOP_SAMPLES, MEL_BINS, WIN_SAMPLES};
use crate::FrontendError;

/// Number of unique real-FFT bins (`512 / 2 + 1 = 257`).
pub const SPEC_BINS: usize = FFT_SIZE / 2 + 1;

/// Expected element count in `mel_fbank.f32` (`257 * 80 = 20_560`).
pub const FBANK_LEN: usize = SPEC_BINS * MEL_BINS;

/// Validated global affine CMVN statistics (`(x - mean) * istd`).
#[derive(Debug, Clone, PartialEq)]
pub struct GlobalCmvn {
    mean: [f32; MEL_BINS],
    istd: [f32; MEL_BINS],
}

impl GlobalCmvn {
    /// Construct validated 80-bin CMVN statistics; refuses non-finite values or `istd <= 0`.
    pub fn new(mean_slice: &[f32], istd_slice: &[f32]) -> Result<Self, FrontendError> {
        if mean_slice.len() != MEL_BINS || istd_slice.len() != MEL_BINS {
            return Err(FrontendError::Format(format!(
                "CMVN expects {MEL_BINS} mean/istd values, got {}/{}",
                mean_slice.len(),
                istd_slice.len()
            )));
        }
        let mut mean = [0.0f32; MEL_BINS];
        let mut istd = [0.0f32; MEL_BINS];
        for i in 0..MEL_BINS {
            let m = mean_slice[i];
            let s = istd_slice[i];
            if !m.is_finite() || !s.is_finite() || s <= 0.0 {
                return Err(FrontendError::Format(format!(
                    "non-finite or non-positive CMVN entry at bin {i}: mean={m}, istd={s}"
                )));
            }
            mean[i] = m;
            istd[i] = s;
        }
        Ok(Self { mean, istd })
    }

    /// Identity CMVN (`mean = 0`, `istd = 1`) for raw log-mel inspection.
    pub fn identity() -> Self {
        Self {
            mean: [0.0; MEL_BINS],
            istd: [1.0; MEL_BINS],
        }
    }

    #[inline]
    pub fn apply_in_place(&self, row: &mut [f32; MEL_BINS]) {
        for ((v, &m), &s) in row.iter_mut().zip(&self.mean).zip(&self.istd) {
            *v = (*v - m) * s;
        }
    }
}

/// Immutable configuration + filterbank shared across streams of an R1 bundle.
#[derive(Debug, Clone)]
pub struct TrainerMelConfig {
    pub log_floor: f32,
    pub cmvn: GlobalCmvn,
    window: [f32; WIN_SAMPLES],
    /// Row-major `257 x 80` filterbank (`fbank[k * 80 + b]`).
    fbank: Vec<f32>,
}

impl TrainerMelConfig {
    /// Build from raw little-endian `mel_fbank.f32` bytes (`257 * 80 * 4 = 82_240` bytes).
    pub fn from_fbank_le_bytes(
        bytes: &[u8],
        log_floor: f32,
        cmvn: GlobalCmvn,
    ) -> Result<Self, FrontendError> {
        if bytes.len() != FBANK_LEN * 4 {
            return Err(FrontendError::Format(format!(
                "mel_fbank.f32 must be {} bytes (257x80 f32), got {}",
                FBANK_LEN * 4,
                bytes.len()
            )));
        }
        let mut fbank = Vec::with_capacity(FBANK_LEN);
        for chunk in bytes.chunks_exact(4) {
            let v = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            if !v.is_finite() || v < 0.0 {
                return Err(FrontendError::Format(format!(
                    "invalid mel_fbank weight {v} (must be finite and >= 0)"
                )));
            }
            fbank.push(v);
        }
        Self::from_fbank_vec(fbank, log_floor, cmvn)
    }

    /// Load `mel_fbank.f32` from disk after caller has verified its SHA-256 against `manifest.files`.
    pub fn from_fbank_file(
        path: &Path,
        log_floor: f32,
        cmvn: GlobalCmvn,
    ) -> Result<Self, FrontendError> {
        let bytes = std::fs::read(path)?;
        Self::from_fbank_le_bytes(&bytes, log_floor, cmvn)
    }

    /// Construct with the HTK `f64` mel filterbank formula from `python/ultra/train/features.py`
    /// (for standalone unit tests when no bundle file is needed).
    pub fn default_htk(cmvn: GlobalCmvn) -> Result<Self, FrontendError> {
        let hz_to_mel = |f: f64| 1127.0 * (1.0 + f / 700.0).ln();
        let mel: Vec<f64> = (0..SPEC_BINS)
            .map(|k| {
                let freq = (k as f64) * 8000.0 / ((SPEC_BINS - 1) as f64);
                hz_to_mel(freq)
            })
            .collect();
        let lo_m = hz_to_mel(20.0);
        let hi_m = hz_to_mel(7600.0);
        let edges: Vec<f64> = (0..MEL_BINS + 2)
            .map(|i| lo_m + (hi_m - lo_m) * (i as f64) / ((MEL_BINS + 1) as f64))
            .collect();
        let mut fbank = vec![0.0f32; FBANK_LEN];
        for (k, &mk) in mel.iter().enumerate() {
            for b in 0..MEL_BINS {
                let (lo, mid, hi) = (edges[b], edges[b + 1], edges[b + 2]);
                let up = (mk - lo) / (mid - lo);
                let down = (hi - mk) / (hi - mid);
                fbank[k * MEL_BINS + b] = up.min(down).max(0.0) as f32;
            }
        }
        Self::from_fbank_vec(fbank, 1e-6, cmvn)
    }

    fn from_fbank_vec(
        fbank: Vec<f32>,
        log_floor: f32,
        cmvn: GlobalCmvn,
    ) -> Result<Self, FrontendError> {
        if !log_floor.is_finite() || log_floor <= 0.0 {
            return Err(FrontendError::Format(format!(
                "log_floor must be finite and > 0, got {log_floor}"
            )));
        }
        if fbank.len() != FBANK_LEN {
            return Err(FrontendError::Format(format!(
                "fbank length {} != {FBANK_LEN}",
                fbank.len()
            )));
        }
        let mut window = [0.0f32; WIN_SAMPLES];
        for (n, w) in window.iter_mut().enumerate() {
            let v = 0.5
                - 0.5
                    * (2.0 * std::f64::consts::PI * (n as f64) / ((WIN_SAMPLES - 1) as f64)).cos();
            *w = v as f32;
        }
        Ok(Self {
            log_floor,
            cmvn,
            window,
            fbank,
        })
    }

    /// Compute one normalized 80-bin log-mel frame from exactly `400` unit-range `f32` samples.
    pub fn frame_f32(&self, frame: &[f32]) -> [f32; MEL_BINS] {
        debug_assert_eq!(frame.len(), WIN_SAMPLES);
        let mean = (frame.iter().map(|&x| f64::from(x)).sum::<f64>() / (WIN_SAMPLES as f64)) as f32;
        let mut re = [0.0f64; FFT_SIZE];
        let mut im = [0.0f64; FFT_SIZE];
        for i in 0..WIN_SAMPLES {
            let v = (frame[i] - mean) * self.window[i];
            re[i] = f64::from(v);
        }
        fft_in_place_512_f64(&mut re, &mut im);

        let mut mel_acc = [0.0f64; MEL_BINS];
        for k in 0..SPEC_BINS {
            let mag = re[k].hypot(im[k]) as f32;
            let p = f64::from(mag * mag);
            let row = &self.fbank[k * MEL_BINS..(k + 1) * MEL_BINS];
            for (m, &w) in mel_acc.iter_mut().zip(row) {
                *m += p * f64::from(w);
            }
        }
        let mut mel = [0.0f32; MEL_BINS];
        for (dst, &acc) in mel.iter_mut().zip(&mel_acc) {
            *dst = ((acc as f32) + self.log_floor).ln();
        }
        self.cmvn.apply_in_place(&mut mel);
        mel
    }
}

/// Per-stream causal streaming extractor wrapping a shared [`TrainerMelConfig`].
#[derive(Debug, Clone)]
pub struct TrainerLogMel {
    cfg: std::sync::Arc<TrainerMelConfig>,
    buf: Vec<f32>,
    read_pos: usize,
    frames_emitted: u64,
}

impl TrainerLogMel {
    pub fn new(cfg: std::sync::Arc<TrainerMelConfig>) -> Self {
        Self {
            cfg,
            buf: Vec::with_capacity(WIN_SAMPLES * 4),
            read_pos: 0,
            frames_emitted: 0,
        }
    }

    pub fn frames_emitted(&self) -> u64 {
        self.frames_emitted
    }

    /// Push `16 kHz` `PCM16` samples (`i16 / 32768.0`) and append newly completed frames to `out`.
    pub fn push_i16_into(&mut self, pcm: &[i16], out: &mut Vec<[f32; MEL_BINS]>) {
        if pcm.is_empty() {
            return;
        }
        let scale = 1.0f32 / 32768.0;
        self.buf.extend(pcm.iter().map(|&s| f32::from(s) * scale));
        self.drain_into(out);
    }

    /// Push `16 kHz` unit-range `f32` samples and append newly completed frames to `out`.
    pub fn push_f32_into(&mut self, samples: &[f32], out: &mut Vec<[f32; MEL_BINS]>) {
        if samples.is_empty() {
            return;
        }
        self.buf.extend_from_slice(samples);
        self.drain_into(out);
    }

    pub fn push_i16(&mut self, pcm: &[i16]) -> Vec<[f32; MEL_BINS]> {
        let mut out = Vec::new();
        self.push_i16_into(pcm, &mut out);
        out
    }

    fn drain_into(&mut self, out: &mut Vec<[f32; MEL_BINS]>) {
        while self.buf.len() - self.read_pos >= WIN_SAMPLES {
            let row = self
                .cfg
                .frame_f32(&self.buf[self.read_pos..self.read_pos + WIN_SAMPLES]);
            out.push(row);
            self.read_pos += HOP_SAMPLES;
            self.frames_emitted += 1;
        }
        if self.read_pos >= 4096 {
            self.buf.drain(..self.read_pos);
            self.read_pos = 0;
        }
    }
}

fn fft_in_place_512_f64(re: &mut [f64; FFT_SIZE], im: &mut [f64; FFT_SIZE]) {
    let mut j = 0usize;
    for i in 1..FFT_SIZE {
        let mut bit = FFT_SIZE >> 1;
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
    while len <= FFT_SIZE {
        let half = len / 2;
        let ang = -2.0 * std::f64::consts::PI / (len as f64);
        let wlen_re = ang.cos();
        let wlen_im = ang.sin();
        let mut start = 0usize;
        while start < FFT_SIZE {
            let mut w_re = 1.0f64;
            let mut w_im = 0.0f64;
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
