//! Audio contract (§14.1): PCM16 WAV / raw `s16le` I/O and stateful anti-aliased resampling
//! into the internal `16_000` Hz signed 16-bit mono timebase.

use std::path::Path;

use crate::FrontendError;

pub const SAMPLE_RATE_16K: u32 = 16_000;

/// Parsed PCM samples from a WAV container.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WavData {
    pub sample_rate_hz: u32,
    pub channels: u16,
    /// Interleaved signed 16-bit PCM samples.
    pub samples: Vec<i16>,
}

/// Parse a RIFF WAVE PCM16 (format tag 1, 16 bits/sample) byte buffer.
pub fn read_wav_pcm(bytes: &[u8]) -> Result<WavData, FrontendError> {
    if bytes.len() < 44 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(FrontendError::Format("missing RIFF/WAVE header".into()));
    }
    let mut pos = 12usize;
    let mut fmt_found = false;
    let mut channels = 0u16;
    let mut sample_rate_hz = 0u32;
    let mut data_slice: Option<&[u8]> = None;

    while pos + 8 <= bytes.len() {
        let tag = &bytes[pos..pos + 4];
        let chunk_len = u32::from_le_bytes([
            bytes[pos + 4],
            bytes[pos + 5],
            bytes[pos + 6],
            bytes[pos + 7],
        ]) as usize;
        pos += 8;
        let end = pos.saturating_add(chunk_len).min(bytes.len());
        let body = &bytes[pos..end];
        if tag == b"fmt " {
            if body.len() < 16 {
                return Err(FrontendError::Format("truncated fmt chunk".into()));
            }
            let audio_format = u16::from_le_bytes([body[0], body[1]]);
            if audio_format != 1 {
                return Err(FrontendError::Format(format!(
                    "unsupported WAV format tag {audio_format} (expected PCM=1)"
                )));
            }
            channels = u16::from_le_bytes([body[2], body[3]]);
            sample_rate_hz = u32::from_le_bytes([body[4], body[5], body[6], body[7]]);
            let bits_per_sample = u16::from_le_bytes([body[14], body[15]]);
            if bits_per_sample != 16 {
                return Err(FrontendError::Format(format!(
                    "unsupported bits_per_sample {bits_per_sample} (expected 16)"
                )));
            }
            fmt_found = true;
        } else if tag == b"data" {
            data_slice = Some(body);
            break;
        }
        // RIFF chunks are word-aligned (padded to even byte boundary).
        pos = end.saturating_add(chunk_len & 1);
    }

    if !fmt_found {
        return Err(FrontendError::Format("missing fmt chunk".into()));
    }
    if channels == 0 || channels > 2 {
        return Err(FrontendError::Channels(channels));
    }
    if sample_rate_hz == 0 {
        return Err(FrontendError::SampleRate(0));
    }
    let raw = data_slice.ok_or_else(|| FrontendError::Format("missing data chunk".into()))?;
    let frame_bytes = usize::from(channels) * 2;
    let valid_bytes = (raw.len() / frame_bytes) * frame_bytes;
    let mut samples = Vec::with_capacity(valid_bytes / 2);
    for pair in raw[..valid_bytes].chunks_exact(2) {
        samples.push(i16::from_le_bytes([pair[0], pair[1]]));
    }
    Ok(WavData {
        sample_rate_hz,
        channels,
        samples,
    })
}

/// Write mono 16-bit PCM samples as a canonical 44-byte header RIFF WAVE file.
pub fn write_wav_pcm16(
    path: &Path,
    sample_rate_hz: u32,
    mono_pcm: &[i16],
) -> Result<(), FrontendError> {
    if sample_rate_hz == 0 {
        return Err(FrontendError::SampleRate(0));
    }
    let data_bytes = (mono_pcm.len() * 2) as u32;
    let riff_size = 36u32.saturating_add(data_bytes);
    let byte_rate = sample_rate_hz.saturating_mul(2);
    let mut out = Vec::with_capacity(44 + mono_pcm.len() * 2);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&riff_size.to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM
    out.extend_from_slice(&1u16.to_le_bytes()); // mono
    out.extend_from_slice(&sample_rate_hz.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&2u16.to_le_bytes()); // block align
    out.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_bytes.to_le_bytes());
    for &s in mono_pcm {
        out.extend_from_slice(&s.to_le_bytes());
    }
    std::fs::write(path, out)?;
    Ok(())
}

/// Downmix stereo to mono by exact rounded midpoint `(L + R) / 2`, or pass mono through.
pub fn to_mono(channels: u16, interleaved: &[i16]) -> Result<Vec<i16>, FrontendError> {
    match channels {
        1 => Ok(interleaved.to_vec()),
        2 => {
            if !interleaved.len().is_multiple_of(2) {
                return Err(FrontendError::Format(
                    "odd sample count in stereo PCM".into(),
                ));
            }
            Ok(interleaved
                .chunks_exact(2)
                .map(|lr| ((i32::from(lr[0]) + i32::from(lr[1])) / 2) as i16)
                .collect())
        }
        other => Err(FrontendError::Channels(other)),
    }
}

/// Stateful windowed-sinc anti-aliased resampler to `16_000` Hz mono (§14.1).
///
/// For `16_000` Hz input it is a zero-delay identity pass. For other supported capture
/// rates (e.g. `48_000`, `44_100`, `32_000`, `24_000`, `8_000` Hz) it applies a causal
/// Hann-windowed low-pass sinc kernel (`HALF_TAPS = 16` input samples) with continuous
/// fractional phase across arbitrary streaming packet boundaries.
#[derive(Debug, Clone)]
pub struct Resampler16k {
    in_rate: u32,
    history: Vec<f32>,
    /// Next output position in input-sample coordinates relative to `history[0]`.
    pos: f64,
}

const HALF_TAPS: usize = 16;

impl Resampler16k {
    pub fn new(in_rate: u32) -> Result<Self, FrontendError> {
        if !(4_000..=192_000).contains(&in_rate) {
            return Err(FrontendError::SampleRate(in_rate));
        }
        Ok(Self {
            in_rate,
            history: vec![0.0; HALF_TAPS],
            pos: HALF_TAPS as f64,
        })
    }

    pub fn in_rate(&self) -> u32 {
        self.in_rate
    }

    /// Algorithmic lookahead delay in input samples (`0` at 16 kHz, `HALF_TAPS` otherwise).
    pub fn delay_input_samples(&self) -> usize {
        if self.in_rate == SAMPLE_RATE_16K {
            0
        } else {
            HALF_TAPS
        }
    }

    /// Algorithmic delay in milliseconds.
    pub fn delay_ms(&self) -> f64 {
        (self.delay_input_samples() as f64) * 1000.0 / f64::from(self.in_rate)
    }

    /// Push mono `in_rate` samples and emit newly available `16_000` Hz samples.
    pub fn push(&mut self, input: &[i16]) -> Vec<i16> {
        if self.in_rate == SAMPLE_RATE_16K {
            return input.to_vec();
        }
        if input.is_empty() {
            return Vec::new();
        }
        self.history.extend(input.iter().map(|&s| f32::from(s)));
        self.drain_available()
    }

    /// Flush the tail filter state at end of stream.
    pub fn flush(&mut self) -> Vec<i16> {
        if self.in_rate == SAMPLE_RATE_16K {
            return Vec::new();
        }
        self.history.resize(self.history.len() + HALF_TAPS, 0.0);
        self.drain_available()
    }

    fn drain_available(&mut self) -> Vec<i16> {
        let step = f64::from(self.in_rate) / f64::from(SAMPLE_RATE_16K);
        // Cutoff below the lower Nyquist to prevent aliasing when downsampling.
        let cutoff = (f64::from(SAMPLE_RATE_16K).min(f64::from(self.in_rate)) * 0.45)
            / f64::from(self.in_rate);
        let mut out = Vec::new();
        while self.pos + (HALF_TAPS as f64) < self.history.len() as f64 {
            let center = self.pos.floor() as isize;
            let mut acc = 0.0f64;
            let mut wsum = 0.0f64;
            for k in -(HALF_TAPS as isize)..=(HALF_TAPS as isize) {
                let idx = (center + k) as usize;
                let x = self.pos - (idx as f64);
                let u = x / (HALF_TAPS as f64);
                if u.abs() >= 1.0 {
                    continue;
                }
                let win = 0.5 * (1.0 + (std::f64::consts::PI * u).cos());
                let arg = 2.0 * cutoff * x;
                let sinc = if arg.abs() < 1e-9 {
                    2.0 * cutoff
                } else {
                    (std::f64::consts::PI * arg).sin() / (std::f64::consts::PI * x)
                };
                let w = win * sinc;
                acc += f64::from(self.history[idx]) * w;
                wsum += w;
            }
            let y = if wsum.abs() > 1e-12 { acc / wsum } else { 0.0 };
            out.push(y.round().clamp(f64::from(i16::MIN), f64::from(i16::MAX)) as i16);
            self.pos += step;
        }
        let drop_n = (self.pos.floor() as usize).saturating_sub(HALF_TAPS);
        if drop_n > 0 && drop_n <= self.history.len() {
            self.history.drain(..drop_n);
            self.pos -= drop_n as f64;
        }
        out
    }
}

/// Read either a `.wav` file (resampling + downmixing to 16 kHz mono if needed) or a raw
/// `.s16le` / `.pcm` 16 kHz mono file.
pub fn read_audio_16k_mono(path: &Path) -> Result<Vec<i16>, FrontendError> {
    let bytes = std::fs::read(path)?;
    let ext = path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    if ext == "s16le" || ext == "pcm" {
        if !bytes.len().is_multiple_of(2) {
            return Err(FrontendError::Format(
                "raw s16le file has odd byte length".into(),
            ));
        }
        return Ok(bytes
            .chunks_exact(2)
            .map(|b| i16::from_le_bytes([b[0], b[1]]))
            .collect());
    }
    let wav = read_wav_pcm(&bytes)?;
    let mono = to_mono(wav.channels, &wav.samples)?;
    if wav.sample_rate_hz == SAMPLE_RATE_16K {
        Ok(mono)
    } else {
        let mut r = Resampler16k::new(wav.sample_rate_hz)?;
        let mut out = r.push(&mono);
        out.extend(r.flush());
        Ok(out)
    }
}
