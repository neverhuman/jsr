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
        let (mut mean, mut istd) = ([0.0f32; MEL_BINS], [0.0f32; MEL_BINS]);
        for i in 0..MEL_BINS {
            let (m, s) = (mean_slice[i], istd_slice[i]);
            if !m.is_finite() || !s.is_finite() || s <= 0.0 {
                return Err(FrontendError::Format(format!(
                    "non-finite or non-positive CMVN entry at bin {i}: mean={m}, istd={s}"
                )));
            }
            (mean[i], istd[i]) = (m, s);
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

    #[inline]
    pub fn mean(&self) -> &[f32; MEL_BINS] {
        &self.mean
    }

    #[inline]
    pub fn istd(&self) -> &[f32; MEL_BINS] {
        &self.istd
    }
}

/// Immutable configuration + filterbank shared across streams of an R1 bundle.
#[derive(Debug, Clone)]
pub struct TrainerMelConfig {
    pub log_floor: f32,
    pub cmvn: GlobalCmvn,
    window: [f32; WIN_SAMPLES],
    portable_window: [f32; WIN_SAMPLES],
    /// Precomputed portable 512-point FFT twiddle factors indexed by `half + k`.
    fft_tw_re: [f64; FFT_SIZE],
    fft_tw_im: [f64; FFT_SIZE],
    /// Precomputed 512-point FFT bit-reversal swap pairs (`i < j`).
    fft_bitrev: Vec<(u16, u16)>,
    /// When `true`, [`Self::frame_f32`] uses the 0-ULP cross-ISA portable path (zero `libm`,
    /// unfused IEEE-754 `f64`, identical 8-lane DC reduction on AVX2 and scalar/ARM).
    portable_math: bool,
    /// Row-major `257 x 80` filterbank (`fbank[k * 80 + b]`, retained for non-finite fallback).
    fbank: Vec<f32>,
    /// Sparse triangular filterbank: nonzero `(k, w_f64)` entries per mel bin in strictly
    /// increasing `k` order (~514 entries vs 20,560 dense = 40x fewer MACs, 0-ULP identical).
    fbank_sparse: Vec<(u16, f64)>,
    /// Start/end offsets into `fbank_sparse` for each mel bin `0..MEL_BINS`.
    fbank_bin_offsets: [u16; MEL_BINS + 1],
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
            .map(|k| hz_to_mel((k as f64) * 8000.0 / ((SPEC_BINS - 1) as f64)))
            .collect();
        let (lo_m, hi_m) = (hz_to_mel(20.0), hz_to_mel(7600.0));
        let edges: Vec<f64> = (0..MEL_BINS + 2)
            .map(|i| lo_m + (hi_m - lo_m) * (i as f64) / ((MEL_BINS + 1) as f64))
            .collect();
        let mut fbank = vec![0.0f32; FBANK_LEN];
        for (k, &mk) in mel.iter().enumerate() {
            for b in 0..MEL_BINS {
                let (lo, mid, hi) = (edges[b], edges[b + 1], edges[b + 2]);
                fbank[k * MEL_BINS + b] = ((mk - lo) / (mid - lo))
                    .min((hi - mk) / (hi - mid))
                    .max(0.0) as f32;
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
        let mut portable_window = [0.0f32; WIN_SAMPLES];
        for n in 0..WIN_SAMPLES {
            let ang = 2.0 * std::f64::consts::PI * (n as f64) / ((WIN_SAMPLES - 1) as f64);
            window[n] = (0.5 - 0.5 * ang.cos()) as f32;
            let (_s, c) = crate::portable_math::sin_cos_portable(ang);
            portable_window[n] = (0.5 - 0.5 * c) as f32;
        }
        let (fft_tw_re, fft_tw_im) =
            crate::portable_math::build_portable_fft_twiddles::<FFT_SIZE>();
        let fft_bitrev = crate::portable_math::build_portable_fft_bitrev::<FFT_SIZE>();
        let mut fbank_sparse = Vec::with_capacity(SPEC_BINS * 3);
        let mut fbank_bin_offsets = [0u16; MEL_BINS + 1];
        for b in 0..MEL_BINS {
            fbank_bin_offsets[b] = fbank_sparse.len() as u16;
            for k in 0..SPEC_BINS {
                let w = fbank[k * MEL_BINS + b];
                if w != 0.0 {
                    fbank_sparse.push((k as u16, f64::from(w)));
                }
            }
        }
        fbank_bin_offsets[MEL_BINS] = fbank_sparse.len() as u16;
        Ok(Self {
            log_floor,
            cmvn,
            window,
            portable_window,
            fft_tw_re,
            fft_tw_im,
            fft_bitrev,
            portable_math: true,
            fbank,
            fbank_sparse,
            fbank_bin_offsets,
        })
    }

    /// Enable or disable the 0-ULP cross-ISA portable math path (`G4`: zero `libm`, unfused
    /// IEEE-754 `f64`, bit-identical across x86_64 AVX2, aarch64 NEON, and scalar builds).
    #[must_use]
    pub fn with_portable_math(mut self, enabled: bool) -> Self {
        self.portable_math = enabled;
        self
    }

    /// Returns whether the cross-ISA bit-exact portable math path is enabled.
    #[inline]
    pub fn is_portable_math(&self) -> bool {
        self.portable_math
    }

    /// Row-major `257 x 80` mel filterbank weights (`FBANK_LEN` floats).
    #[inline]
    pub fn fbank(&self) -> &[f32] {
        &self.fbank
    }

    /// Compute one normalized 80-bin log-mel frame from exactly `400` unit-range `f32` samples.
    pub fn frame_f32(&self, frame: &[f32]) -> [f32; MEL_BINS] {
        if self.portable_math {
            return self.frame_f32_portable(frame);
        }
        if crate::trainer_mel_avx2::avx2_fma_available() {
            #[cfg(target_arch = "x86_64")]
            {
                let mel_acc = crate::trainer_mel_avx2::dc_window_and_mel_avx2(
                    frame,
                    &self.window,
                    &self.fbank,
                    crate::portable_math::fft_in_place_512_libm_f64,
                );
                let mut mel = [0.0f32; MEL_BINS];
                for (dst, &acc) in mel.iter_mut().zip(&mel_acc) {
                    *dst = ((acc as f32) + self.log_floor).ln();
                }
                self.cmvn.apply_in_place(&mut mel);
                return mel;
            }
        }
        self.frame_f32_scalar(frame)
    }

    /// 0-ULP sparse triangular mel projection (`~514` MACs vs `20,560` dense), falling back to
    /// dense summation if any power bin is non-finite so `0 * NaN = NaN` semantics hold.
    fn mel_dot_portable(
        &self,
        re: &[f64; FFT_SIZE],
        im: &[f64; FFT_SIZE],
        use_sparse: bool,
    ) -> [f64; MEL_BINS] {
        let mut powers = [0.0f64; SPEC_BINS];
        let mut all_finite_nonneg = use_sparse;
        for k in 0..SPEC_BINS {
            let p = f64::from(crate::portable_math::power_f32_from_complex_f64(
                re[k], im[k],
            ));
            all_finite_nonneg &= p.is_finite() && p >= 0.0;
            powers[k] = p;
        }
        let mut mel_acc = [0.0f64; MEL_BINS];
        if all_finite_nonneg {
            for (b, dst) in mel_acc.iter_mut().enumerate() {
                let lo = self.fbank_bin_offsets[b] as usize;
                let hi = self.fbank_bin_offsets[b + 1] as usize;
                let mut acc = 0.0f64;
                for &(k, w) in &self.fbank_sparse[lo..hi] {
                    acc += powers[k as usize] * w;
                }
                *dst = acc;
            }
        } else {
            for (k, &p) in powers.iter().enumerate() {
                let row = &self.fbank[k * MEL_BINS..(k + 1) * MEL_BINS];
                for (m, &w) in mel_acc.iter_mut().zip(row) {
                    *m += p * f64::from(w);
                }
            }
        }
        mel_acc
    }

    fn finish_portable_mel(&self, mel_acc: &[f64; MEL_BINS]) -> [f32; MEL_BINS] {
        let mut mel = [0.0f32; MEL_BINS];
        for (dst, &acc) in mel.iter_mut().zip(mel_acc) {
            *dst = crate::portable_math::ln_portable_f32((acc as f32) + self.log_floor);
        }
        self.cmvn.apply_in_place(&mut mel);
        mel
    }

    /// Number of nonzero weights in the sparse mel filterbank (`<= 20,560`, typically `~477`).
    #[inline]
    pub fn sparse_fbank_nnz(&self) -> usize {
        self.fbank_sparse.len()
    }

    fn dc_and_window_scalar_portable(&self, frame: &[f32]) -> [f64; FFT_SIZE] {
        debug_assert_eq!(frame.len(), WIN_SAMPLES);
        let (mut sum0, mut sum1) = ([0.0f64; 4], [0.0f64; 4]);
        let mut i = 0usize;
        while i + 8 <= WIN_SAMPLES {
            for l in 0..4 {
                sum0[l] += f64::from(frame[i + l]);
                sum1[l] += f64::from(frame[i + 4 + l]);
            }
            i += 8;
        }
        let mean = (((sum0[0] + sum1[0])
            + (sum0[1] + sum1[1])
            + (sum0[2] + sum1[2])
            + (sum0[3] + sum1[3]))
            / (WIN_SAMPLES as f64)) as f32;
        let mut re = [0.0f64; FFT_SIZE];
        for idx in 0..WIN_SAMPLES {
            re[idx] = f64::from((frame[idx] - mean) * self.portable_window[idx]);
        }
        re
    }

    /// Cross-ISA bit-exact normalized 80-bin log-mel frame (0 ULP between AVX2, NEON, scalar).
    pub fn frame_f32_portable(&self, frame: &[f32]) -> [f32; MEL_BINS] {
        if crate::trainer_mel_avx2::avx2_fma_available() {
            #[cfg(target_arch = "x86_64")]
            {
                let mut re = [0.0f64; FFT_SIZE];
                let mut im = [0.0f64; FFT_SIZE];
                crate::trainer_mel_avx2::dc_and_window_avx2_portable(
                    frame,
                    &self.portable_window,
                    &mut re,
                );
                crate::trainer_mel_avx2::fft_in_place_avx2_portable(
                    &mut re,
                    &mut im,
                    &self.fft_tw_re,
                    &self.fft_tw_im,
                    &self.fft_bitrev,
                );
                let mel_acc = self.mel_dot_portable(&re, &im, true);
                return self.finish_portable_mel(&mel_acc);
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            let mut re = self.dc_and_window_scalar_portable(frame);
            let mut im = [0.0f64; FFT_SIZE];
            crate::trainer_mel_avx2::fft_in_place_neon_portable(
                &mut re,
                &mut im,
                &self.fft_tw_re,
                &self.fft_tw_im,
                &self.fft_bitrev,
            );
            let mel_acc = self.mel_dot_portable(&re, &im, true);
            return self.finish_portable_mel(&mel_acc);
        }
        #[allow(unreachable_code)]
        self.frame_f32_portable_scalar(frame)
    }

    fn frame_f32_portable_scalar_impl(&self, frame: &[f32], use_sparse: bool) -> [f32; MEL_BINS] {
        let mut re = self.dc_and_window_scalar_portable(frame);
        let mut im = [0.0f64; FFT_SIZE];
        if use_sparse {
            crate::portable_math::fft_in_place_portable_with_bitrev::<FFT_SIZE>(
                &mut re,
                &mut im,
                &self.fft_tw_re,
                &self.fft_tw_im,
                &self.fft_bitrev,
            );
        } else {
            crate::portable_math::fft_in_place_portable::<FFT_SIZE>(
                &mut re,
                &mut im,
                &self.fft_tw_re,
                &self.fft_tw_im,
            );
        }
        let mel_acc = self.mel_dot_portable(&re, &im, use_sparse);
        self.finish_portable_mel(&mel_acc)
    }

    /// Pure scalar reference for [`Self::frame_f32_portable`] (0-ULP identical to AVX2 portable).
    pub fn frame_f32_portable_scalar(&self, frame: &[f32]) -> [f32; MEL_BINS] {
        self.frame_f32_portable_scalar_impl(frame, true)
    }

    /// Brute-force dense portable reference (used to prove 0-ULP equivalence of sparse projection).
    pub fn frame_f32_portable_dense_ref(&self, frame: &[f32]) -> [f32; MEL_BINS] {
        self.frame_f32_portable_scalar_impl(frame, false)
    }

    /// Exact portable scalar `f64` reference for [`Self::frame_f32`].
    pub fn frame_f32_scalar(&self, frame: &[f32]) -> [f32; MEL_BINS] {
        debug_assert_eq!(frame.len(), WIN_SAMPLES);
        let mean = (frame.iter().map(|&x| f64::from(x)).sum::<f64>() / (WIN_SAMPLES as f64)) as f32;
        let mut re = [0.0f64; FFT_SIZE];
        let mut im = [0.0f64; FFT_SIZE];
        for i in 0..WIN_SAMPLES {
            let v = (frame[i] - mean) * self.window[i];
            re[i] = f64::from(v);
        }
        crate::portable_math::fft_in_place_512_libm_f64(&mut re, &mut im);

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
    pub fn retained_heap_bytes(&self) -> usize {
        self.buf.capacity() * std::mem::size_of::<f32>()
    }
    pub fn retained_bytes(&self) -> usize {
        self.retained_heap_bytes() + std::mem::size_of::<Self>()
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
