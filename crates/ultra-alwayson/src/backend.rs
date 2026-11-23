//! ASR backend abstraction for `ultra-alwayson`: wraps `ultra-engine` (`EngineBackend`) with
//! one-pass batched replay (`max_chunks = 16`) and fresh per-segment `Engine::create_stream()`.

use ultra_engine::{Engine, EngineStream, Workspace};

use crate::AlwaysOnError;

/// Summary of one `accept_and_decode_all` or `finish_and_drain` backend step.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DecodeStep {
    /// Encoder frames advanced during this call.
    pub frames_advanced: usize,
    /// Number of batched decode invocations executed (`1` for one-pass wake replay).
    pub decode_calls: u64,
    /// Number of newly appended non-blank tokens.
    pub new_tokens: usize,
}

/// Minimal streaming recogniser interface consumed by [`crate::AlwaysOnSession`].
pub trait AsrBackend {
    /// Streaming chunk size in 25 Hz encoder frames (e.g. `4` = `160 ms`).
    fn chunk_frames(&self) -> usize;
    /// Maximum chunks processed per batched decode call (e.g. `16`).
    fn max_chunks(&self) -> usize;
    /// Construct a fresh isolated stream for a new segment (never reuses stale caches/counters).
    fn reset_stream(&mut self);
    /// Release the active stream when entering `Listening` or `Recovery`.
    fn drop_stream(&mut self);
    /// True if a stream is currently open.
    fn has_active_stream(&self) -> bool;
    /// Accept PCM16 samples and decode every newly ready chunk (in batched steps up to `max_chunks`).
    fn accept_and_decode_all(&mut self, pcm: &[i16]) -> Result<DecodeStep, AlwaysOnError>;
    /// Mark input finished on the active stream and flush any final partial chunk.
    fn finish_and_drain(&mut self) -> Result<DecodeStep, AlwaysOnError>;
    /// Total 25 Hz encoder frames processed in the current segment.
    fn frames_processed(&self) -> usize;
    /// Emitted non-blank token IDs in the current segment.
    fn tokens(&self) -> &[u32];
    /// Decode a standalone PCM16 slice in a brand-new stream for exact segment-identity checks.
    fn decode_isolated(&self, pcm: &[i16]) -> Result<Vec<u32>, AlwaysOnError>;
}

/// Real `ultra-engine` backend using fresh per-segment `Engine::create_stream()` and batched
/// `with_max_chunks(16)` wake replay (with `endpoint_reset_ms = 0` so the sleep ladder owns
/// segment boundaries and per-episode reset counters exclusively).
#[derive(Debug)]
pub struct EngineBackend {
    engine: Engine,
    stream: Option<EngineStream>,
    ws: Workspace,
    episode_resets: u64,
}

impl EngineBackend {
    /// Wrap an `Engine`, enabling batched multi-chunk replay (`max_chunks = 16`) and disabling
    /// engine-internal endpoint resets.
    pub fn new(engine: Engine) -> Self {
        let mut eng = engine.with_max_chunks(16).with_endpoint_reset_ms(0);
        eng.endpoint_replay_ms = 0;
        Self {
            engine: eng,
            stream: None,
            ws: Workspace::default(),
            episode_resets: 0,
        }
    }

    /// Borrow the underlying configured `Engine`.
    pub fn engine(&self) -> &Engine {
        &self.engine
    }

    /// Total fresh isolated streams created by this backend.
    pub fn episode_resets(&self) -> u64 {
        self.episode_resets
    }
}

impl AsrBackend for EngineBackend {
    fn chunk_frames(&self) -> usize {
        self.engine.chunk
    }

    fn max_chunks(&self) -> usize {
        self.engine.max_chunks
    }

    fn reset_stream(&mut self) {
        self.stream = Some(self.engine.create_stream());
        self.episode_resets += 1;
    }

    fn drop_stream(&mut self) {
        self.stream = None;
    }

    fn has_active_stream(&self) -> bool {
        self.stream.is_some()
    }

    fn accept_and_decode_all(&mut self, pcm: &[i16]) -> Result<DecodeStep, AlwaysOnError> {
        let stream = self.stream.get_or_insert_with(|| {
            self.episode_resets += 1;
            self.engine.create_stream()
        });
        let before_t = stream.enc.t;
        let before_toks = stream.tokens().len();
        stream.accept_i16(pcm)?;
        let mut decode_calls = 0u64;
        while self.engine.decode(&mut [stream], &mut self.ws)? > 0 {
            decode_calls += 1;
        }
        Ok(DecodeStep {
            frames_advanced: stream.enc.t.saturating_sub(before_t),
            decode_calls,
            new_tokens: stream.tokens().len().saturating_sub(before_toks),
        })
    }

    fn finish_and_drain(&mut self) -> Result<DecodeStep, AlwaysOnError> {
        let Some(stream) = self.stream.as_mut() else {
            return Ok(DecodeStep::default());
        };
        let before_t = stream.enc.t;
        let before_toks = stream.tokens().len();
        stream.input_finished();
        let mut decode_calls = 0u64;
        while self.engine.decode(&mut [stream], &mut self.ws)? > 0 {
            decode_calls += 1;
        }
        Ok(DecodeStep {
            frames_advanced: stream.enc.t.saturating_sub(before_t),
            decode_calls,
            new_tokens: stream.tokens().len().saturating_sub(before_toks),
        })
    }

    fn frames_processed(&self) -> usize {
        self.stream.as_ref().map_or(0, |s| s.enc.t)
    }

    fn tokens(&self) -> &[u32] {
        self.stream.as_ref().map_or(&[], EngineStream::tokens)
    }

    fn decode_isolated(&self, pcm: &[i16]) -> Result<Vec<u32>, AlwaysOnError> {
        let mut stream = self.engine.create_stream();
        let mut ws = Workspace::default();
        stream.accept_i16(pcm)?;
        stream.input_finished();
        while self.engine.decode(&mut [&mut stream], &mut ws)? > 0 {}
        Ok(stream.tokens().to_vec())
    }
}
