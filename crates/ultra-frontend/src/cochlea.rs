//! Integer Cochlea front-end arm (`isqrt`-PCEN into A7 `0..=127`, CADENCE v3 / ULTRA §4).
//!
//! Uses `ultra_core::numeric::{isqrt_u64, requant, clipped_relu, check_a7}` so dynamic-range
//! compression and quantisation obey the exact integer contract.

use ultra_core::numeric::{check_a7, clipped_relu, isqrt_u64, requant};

use crate::mel::{MelConfig, StreamingLogMel, HOP_SAMPLES, MEL_BINS, WIN_SAMPLES};
use crate::FrontendError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CochleaConfig {
    /// Exponential smoother retention in Q8 (`0..=255`, e.g. `243` ≈ 0.95).
    pub smooth_keep_q8: u16,
    /// Stabiliser floor added inside `isqrt(eps + M)`.
    pub eps_q16: u64,
    /// Offset subtracted after `E * 2^12 / isqrt(eps + M)` (PCEN delta).
    pub delta_q12: i32,
    /// Requantisation multiplier and shift (`1..=31`).
    pub requant_mult: i32,
    pub requant_shift: u32,
    /// Post-requant right shift into A7 (`0..=127`).
    pub relu_shift: u32,
}

impl Default for CochleaConfig {
    fn default() -> Self {
        Self {
            smooth_keep_q8: 243,
            eps_q16: 64,
            delta_q12: 256,
            requant_mult: 64,
            requant_shift: 10,
            relu_shift: 1,
        }
    }
}

/// Stateful integer PCEN Cochlea producing `A7` (`0..=127`) 80-bin feature frames.
#[derive(Debug, Clone)]
pub struct StreamingCochlea {
    cfg: CochleaConfig,
    mel_helper: StreamingLogMel,
    buf: Vec<i16>,
    prev_sample: Option<i16>,
    ema_q16: [u64; MEL_BINS],
    initialized: bool,
}

impl StreamingCochlea {
    pub fn new(cfg: CochleaConfig) -> Result<Self, FrontendError> {
        if !(1..=31).contains(&cfg.requant_shift) {
            return Err(ultra_core::CoreError::BadShift(cfg.requant_shift).into());
        }
        Ok(Self {
            cfg,
            mel_helper: StreamingLogMel::new(MelConfig::default()),
            buf: Vec::new(),
            prev_sample: None,
            ema_q16: [0; MEL_BINS],
            initialized: false,
        })
    }

    /// Push 16 kHz mono PCM16 samples and emit newly completed A7 (`[u8; 80]`) frames.
    pub fn push(&mut self, pcm: &[i16]) -> Result<Vec<[u8; MEL_BINS]>, FrontendError> {
        if pcm.is_empty() {
            return Ok(Vec::new());
        }
        if self.prev_sample.is_none() {
            self.prev_sample = Some(pcm[0]);
        }
        self.buf.extend_from_slice(pcm);
        let mut out = Vec::new();
        let keep = u64::from(self.cfg.smooth_keep_q8.min(256));
        let enter = 256u64 - keep;

        while self.buf.len() >= WIN_SAMPLES {
            let prev = self.prev_sample.unwrap_or(self.buf[0]);
            let energies = self
                .mel_helper
                .mel_energies_for_frame(prev, &self.buf[..WIN_SAMPLES]);
            let mut a7_frame = [0u8; MEL_BINS];
            for b in 0..MEL_BINS {
                // Quantise linear mel energy onto a fixed Q16 integer grid.
                let e_q16 = ((energies[b].max(0.0) * 65_536.0).round() as u64).min(1 << 40);
                if !self.initialized {
                    self.ema_q16[b] = e_q16;
                } else {
                    self.ema_q16[b] = (keep * self.ema_q16[b] + enter * e_q16 + 128) >> 8;
                }
                let denom = isqrt_u64(self.cfg.eps_q16.saturating_add(self.ema_q16[b])).max(1);
                let pcen_q12 =
                    ((e_q16 << 4) / denom).min(i32::MAX as u64) as i32 - self.cfg.delta_q12;
                let rq = requant(pcen_q12, self.cfg.requant_mult, 0, self.cfg.requant_shift)?;
                a7_frame[b] = clipped_relu(rq, self.cfg.relu_shift);
            }
            self.initialized = true;
            check_a7(&a7_frame)?;
            out.push(a7_frame);
            self.prev_sample = Some(self.buf[HOP_SAMPLES - 1]);
            self.buf.drain(..HOP_SAMPLES);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mel::{MelConfig, StreamingLogMel};
    use crate::npz::{load_npz_audio_only, load_npz_references};
    use crate::wav::{read_audio_16k_mono, write_wav_pcm16, Resampler16k};
    use std::path::Path;

    fn synthetic_speech_16k(n: usize) -> Vec<i16> {
        (0..n)
            .map(|i| {
                let t = i as f32 / 16_000.0;
                let s = 0.4 * (2.0 * std::f32::consts::PI * 220.0 * t).sin()
                    + 0.25 * (2.0 * std::f32::consts::PI * 880.0 * t).sin()
                    + 0.1 * (2.0 * std::f32::consts::PI * 2400.0 * t).cos();
                (s * 20_000.0) as i16
            })
            .collect()
    }

    #[test]
    fn logmel_and_cochlea_are_packet_size_invariant() {
        let pcm = synthetic_speech_16k(16_000 + 333); // >1s with non-aligned tail
        let mut ref_mel = StreamingLogMel::new(MelConfig::default());
        let expected_mel = ref_mel.push(&pcm);
        assert_eq!(
            expected_mel.len(),
            (pcm.len() - WIN_SAMPLES) / HOP_SAMPLES + 1
        );

        let mut ref_coch = StreamingCochlea::new(CochleaConfig::default()).expect("cochlea");
        let expected_coch = ref_coch.push(&pcm).expect("cochlea frames");
        assert_eq!(expected_coch.len(), expected_mel.len());

        // Test Spec §38.1 transport packet sizes: 10 ms (160), 20 ms (320), 37 ms (592).
        for step in [160usize, 320, 592, 73] {
            let mut m = StreamingLogMel::new(MelConfig::default());
            let mut c = StreamingCochlea::new(CochleaConfig::default()).expect("cochlea");
            let mut got_mel = Vec::new();
            let mut got_coch = Vec::new();
            for chunk in pcm.chunks(step) {
                got_mel.extend(m.push(chunk));
                got_coch.extend(c.push(chunk).expect("cochlea chunk"));
            }
            assert_eq!(got_mel, expected_mel, "log-mel mismatch at chunk={step}");
            assert_eq!(got_coch, expected_coch, "cochlea mismatch at chunk={step}");
        }
    }

    #[test]
    fn wav_roundtrip_and_resampler_48k_to_16k() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let wav_16k = tmp.path().join("tone16.wav");
        let orig = synthetic_speech_16k(8_000);
        write_wav_pcm16(&wav_16k, 16_000, &orig).expect("write wav");
        let loaded = read_audio_16k_mono(&wav_16k).expect("read wav");
        assert_eq!(loaded, orig);

        // 48 kHz 220 Hz sine resampled in chunks vs whole buffer matches length & energy.
        let pcm_48k: Vec<i16> = (0..48_000)
            .map(|i| {
                ((2.0 * std::f32::consts::PI * 220.0 * (i as f32 / 48_000.0)).sin() * 16_000.0)
                    as i16
            })
            .collect();
        let mut r_whole = Resampler16k::new(48_000).expect("48k");
        let mut y_whole = r_whole.push(&pcm_48k);
        y_whole.extend(r_whole.flush());
        assert!((y_whole.len() as isize - 16_000).abs() <= 2);

        let mut r_stream = Resampler16k::new(48_000).expect("48k");
        let mut y_stream = Vec::new();
        for ch in pcm_48k.chunks(480) {
            y_stream.extend(r_stream.push(ch));
        }
        y_stream.extend(r_stream.flush());
        assert_eq!(y_stream, y_whole);
    }

    #[test]
    fn npz_reader_loads_dev_clean_slice_when_available() {
        let p = Path::new("/mnt/xbabe3-external/jsr/cache/librispeech_pcm/dev-clean.npz");
        if !p.exists() {
            return;
        }
        let audio = load_npz_audio_only(p, 2, Some(3)).expect("audio slice");
        let refs = load_npz_references(p, 2, Some(3)).expect("ref slice");
        assert_eq!(audio.len(), 3);
        assert_eq!(refs.len(), 3);
        for (a, r) in audio.iter().zip(refs.iter()) {
            assert_eq!(a.index, r.index);
            assert!(!a.pcm.is_empty());
            assert!(!r.uid.is_empty());
            assert!(!r.text.is_empty());
            assert!((a.seconds() - r.duration_s).abs() < 1e-6);
        }
    }
}
