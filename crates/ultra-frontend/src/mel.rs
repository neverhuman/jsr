//! Streaming 80-bin float log-mel feature extractor (16 kHz, 25 ms Povey window, 10 ms hop).
//!
//! Designed so arbitrary input packet boundaries (`10 ms`, `20 ms`, `37 ms`, or 1 sample at
//! a time) emit bit-identical feature frames.

pub const MEL_BINS: usize = 80;
pub const WIN_SAMPLES: usize = 400; // 25 ms @ 16 kHz
pub const HOP_SAMPLES: usize = 160; // 10 ms @ 16 kHz
pub const FFT_SIZE: usize = 512;
const SPEC_BINS: usize = FFT_SIZE / 2 + 1;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MelConfig {
    pub preemph: f32,
    pub fmin_hz: f32,
    pub fmax_hz: f32,
    pub log_floor: f32,
}

impl Default for MelConfig {
    fn default() -> Self {
        Self {
            preemph: 0.97,
            fmin_hz: 20.0,
            fmax_hz: 7600.0,
            log_floor: 1e-10,
        }
    }
}

/// Stateful streaming 80-bin log-mel extractor.
#[derive(Debug, Clone)]
pub struct StreamingLogMel {
    cfg: MelConfig,
    window: Vec<f32>,
    /// Flattened `MEL_BINS x SPEC_BINS` triangular filterbank weights.
    fb: Vec<f32>,
    buf: Vec<i16>,
    prev_sample: Option<i16>,
    frames_emitted: u64,
}

fn hz_to_mel(hz: f32) -> f32 {
    1127.0 * (1.0 + hz / 700.0).ln()
}

fn mel_to_hz(mel: f32) -> f32 {
    700.0 * ((mel / 1127.0).exp() - 1.0)
}

impl StreamingLogMel {
    pub fn new(cfg: MelConfig) -> Self {
        // Povey window: Hann^0.85, standard in Kaldi / icefall / Zipformer.
        let window: Vec<f32> = (0..WIN_SAMPLES)
            .map(|n| {
                let hann = 0.5
                    - 0.5
                        * (2.0 * std::f32::consts::PI * (n as f32) / ((WIN_SAMPLES - 1) as f32))
                            .cos();
                hann.max(0.0).powf(0.85)
            })
            .collect();

        let mel_lo = hz_to_mel(cfg.fmin_hz);
        let mel_hi = hz_to_mel(cfg.fmax_hz);
        let step = (mel_hi - mel_lo) / ((MEL_BINS + 1) as f32);
        let centers_hz: Vec<f32> = (0..=MEL_BINS + 1)
            .map(|i| mel_to_hz(mel_lo + (i as f32) * step))
            .collect();

        let mut fb = vec![0.0f32; MEL_BINS * SPEC_BINS];
        for b in 0..MEL_BINS {
            let f_left = centers_hz[b];
            let f_center = centers_hz[b + 1];
            let f_right = centers_hz[b + 2];
            for k in 0..SPEC_BINS {
                let freq = (k as f32) * 16_000.0 / (FFT_SIZE as f32);
                let w = if freq >= f_left && freq <= f_center && f_center > f_left {
                    (freq - f_left) / (f_center - f_left)
                } else if freq > f_center && freq <= f_right && f_right > f_center {
                    (f_right - freq) / (f_right - f_center)
                } else {
                    0.0
                };
                fb[b * SPEC_BINS + k] = w;
            }
        }

        Self {
            cfg,
            window,
            fb,
            buf: Vec::new(),
            prev_sample: None,
            frames_emitted: 0,
        }
    }

    pub fn frames_emitted(&self) -> u64 {
        self.frames_emitted
    }

    /// Push 16 kHz mono PCM16 samples and return newly completed `[frame][80]` log-mel rows.
    pub fn push(&mut self, pcm: &[i16]) -> Vec<[f32; MEL_BINS]> {
        if pcm.is_empty() {
            return Vec::new();
        }
        if self.prev_sample.is_none() {
            self.prev_sample = Some(pcm[0]);
        }
        self.buf.extend_from_slice(pcm);
        self.drain_frames()
    }

    /// Power spectrum of one 400-sample frame (exposed to `cochlea` as well).
    pub(crate) fn mel_energies_for_frame(&self, prev: i16, frame_pcm: &[i16]) -> [f32; MEL_BINS] {
        let mut re = [0.0f32; FFT_SIZE];
        let mut im = [0.0f32; FFT_SIZE];
        let scale = 1.0f32 / 32768.0;
        let mut last = f32::from(prev) * scale;
        for (i, &s) in frame_pcm.iter().take(WIN_SAMPLES).enumerate() {
            let cur = f32::from(s) * scale;
            let pe = cur - self.cfg.preemph * last;
            last = cur;
            re[i] = pe * self.window[i];
        }
        fft_in_place_512(&mut re, &mut im);
        let mut power = [0.0f32; SPEC_BINS];
        for k in 0..SPEC_BINS {
            power[k] = re[k] * re[k] + im[k] * im[k];
        }
        let mut mel = [0.0f32; MEL_BINS];
        for (b, out) in mel.iter_mut().enumerate() {
            let row = &self.fb[b * SPEC_BINS..(b + 1) * SPEC_BINS];
            let sum: f32 = row.iter().zip(power.iter()).map(|(&w, &p)| w * p).sum();
            *out = sum;
        }
        mel
    }

    fn drain_frames(&mut self) -> Vec<[f32; MEL_BINS]> {
        let mut out = Vec::new();
        while self.buf.len() >= WIN_SAMPLES {
            let prev = self.prev_sample.unwrap_or(self.buf[0]);
            let mut mel = self.mel_energies_for_frame(prev, &self.buf[..WIN_SAMPLES]);
            for v in &mut mel {
                *v = v.max(self.cfg.log_floor).ln();
            }
            out.push(mel);
            self.prev_sample = Some(self.buf[HOP_SAMPLES - 1]);
            self.buf.drain(..HOP_SAMPLES);
            self.frames_emitted += 1;
        }
        out
    }
}

/// Iterative Radix-2 Cooley-Tukey FFT of length 512.
fn fft_in_place_512(re: &mut [f32; FFT_SIZE], im: &mut [f32; FFT_SIZE]) {
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
        let ang = -2.0 * std::f32::consts::PI / (len as f32);
        let wlen_re = ang.cos();
        let wlen_im = ang.sin();
        let mut start = 0usize;
        while start < FFT_SIZE {
            let mut w_re = 1.0f32;
            let mut w_im = 0.0f32;
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
