//! Deterministic synthetic recogniser matching `ultra-engine`'s exact `400`-sample window,
//! `160`-sample hop, `chunk_bounds`, and `640*T - 240` slice geometry for unit tests and
//! `--verify-identity` checks without external weight files.

use ultra_gp::chunk_bounds;

use crate::{AlwaysOnError, AsrBackend, DecodeStep};

/// Deterministic 18-second synthetic PCM16 stream (silence -> burst 1 -> pause -> burst 2 -> silence).
pub fn synthetic_stream_pcm() -> Vec<i16> {
    let total = 18 * 16_000;
    let mut s: u32 = 12_345;
    (0..total)
        .map(|i| {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let mut v = if !(32_000..224_000).contains(&i) {
                0
            } else {
                i32::from((s >> 16) as u16 as i16) >> 9
            };
            let sq = if (i / 20) % 2 == 0 { 1 } else { -1 };
            if (48_000..72_000).contains(&i) || (160_000..184_000).contains(&i) {
                v += 3_000 * sq;
            }
            v as i16
        })
        .collect()
}

#[derive(Debug, Clone, Default)]
struct MockStream {
    pcm: Vec<i16>,
    t: usize,
    finished: bool,
    tokens: Vec<u32>,
}

/// Deterministic mock recogniser implementing [`AsrBackend`] with batched `max_chunks` stepping.
#[derive(Debug, Clone)]
pub struct MockBackend {
    chunk: usize,
    max_chunks: usize,
    speech_diff_threshold: u64,
    fault_on_call: Option<u64>,
    total_calls: u64,
    stream: Option<MockStream>,
}

impl MockBackend {
    /// Construct a mock recogniser (`chunk = 4`, `max_chunks = 16`, diff threshold `350`).
    pub fn new(chunk: usize, max_chunks: usize, speech_diff_threshold: u64) -> Self {
        Self {
            chunk: chunk.max(1),
            max_chunks: max_chunks.clamp(1, 16),
            speech_diff_threshold,
            fault_on_call: None,
            total_calls: 0,
            stream: None,
        }
    }

    /// Inject a deterministic backend failure on the Nth `accept_and_decode_all` call (1-indexed).
    pub fn with_fault_on_call(mut self, call_index: u64) -> Self {
        self.fault_on_call = Some(call_index);
        self
    }

    fn step_stream(&self, s: &mut MockStream) -> DecodeStep {
        let n_feat = if s.pcm.len() >= 400 {
            1 + (s.pcm.len() - 400) / 160
        } else {
            0
        };
        let before_t = s.t;
        let before_toks = s.tokens.len();
        let mut decode_calls = 0u64;
        while let Some((t0, mut t1)) = chunk_bounds(s.t, n_feat, self.chunk, s.finished) {
            for _ in 1..self.max_chunks {
                if t1 - t0 < self.chunk * (t1 - t0).div_ceil(self.chunk) {
                    break;
                }
                match chunk_bounds(t1, n_feat, self.chunk, s.finished) {
                    Some((_, t2)) => t1 = t2,
                    None => break,
                }
            }
            decode_calls += 1;
            for frame in t0..t1 {
                let start = frame * 640;
                let end = (start + 400).min(s.pcm.len());
                if start < end {
                    if let Some(tok) =
                        Self::classify_window(&s.pcm[start..end], self.speech_diff_threshold)
                    {
                        s.tokens.push(tok);
                    }
                }
            }
            s.t = t1;
        }
        DecodeStep {
            frames_advanced: s.t.saturating_sub(before_t),
            decode_calls,
            new_tokens: s.tokens.len().saturating_sub(before_toks),
        }
    }

    fn classify_window(win: &[i16], threshold: u64) -> Option<u32> {
        if win.len() < 2 {
            return None;
        }
        let mut diff_sum = 0u64;
        for w in win.windows(2) {
            diff_sum += i32::from(w[1]).abs_diff(i32::from(w[0])) as u64;
        }
        if diff_sum / (win.len() as u64 - 1) < threshold {
            return None;
        }
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        for &sample in win {
            for b in sample.to_le_bytes() {
                hash ^= u64::from(b);
                hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        Some(1 + (hash % 499) as u32)
    }
}

impl Default for MockBackend {
    fn default() -> Self {
        Self::new(4, 16, 200)
    }
}

impl AsrBackend for MockBackend {
    fn chunk_frames(&self) -> usize {
        self.chunk
    }

    fn max_chunks(&self) -> usize {
        self.max_chunks
    }

    fn reset_stream(&mut self) {
        self.stream = Some(MockStream::default());
    }

    fn drop_stream(&mut self) {
        self.stream = None;
    }

    fn has_active_stream(&self) -> bool {
        self.stream.is_some()
    }

    fn accept_and_decode_all(&mut self, pcm: &[i16]) -> Result<DecodeStep, AlwaysOnError> {
        self.total_calls += 1;
        if self.fault_on_call == Some(self.total_calls) {
            return Err(AlwaysOnError::Backend(format!(
                "injected fault on call {}",
                self.total_calls
            )));
        }
        let mut stream = self.stream.take().unwrap_or_default();
        stream.pcm.extend_from_slice(pcm);
        let step = self.step_stream(&mut stream);
        self.stream = Some(stream);
        Ok(step)
    }

    fn finish_and_drain(&mut self) -> Result<DecodeStep, AlwaysOnError> {
        let Some(mut stream) = self.stream.take() else {
            return Ok(DecodeStep::default());
        };
        stream.finished = true;
        let step = self.step_stream(&mut stream);
        self.stream = Some(stream);
        Ok(step)
    }

    fn frames_processed(&self) -> usize {
        self.stream.as_ref().map_or(0, |s| s.t)
    }

    fn tokens(&self) -> &[u32] {
        self.stream.as_ref().map_or(&[], |s| &s.tokens)
    }

    fn decode_isolated(&self, pcm: &[i16]) -> Result<Vec<u32>, AlwaysOnError> {
        let mut s = MockStream {
            pcm: pcm.to_vec(),
            t: 0,
            finished: true,
            tokens: Vec::new(),
        };
        self.step_stream(&mut s);
        Ok(s.tokens)
    }
}
