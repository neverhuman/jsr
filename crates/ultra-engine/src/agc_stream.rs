//! Recogniser-path [`AgcStream`] adapter pairing [`EngineStream`] with [`WarmAgc`].
//!
//! Runs strictly on the recogniser audio path AFTER any Sentinel-0 gate and pre-roll ring:
//! - When `warm_agc == false` (default), `accept_i16` and `accept_f32` forward straight to
//!   [`EngineStream`] with zero copies, zero scratch allocations, and bit-identical outputs.
//! - When `warm_agc == true`, level state persists across [`AgcStream::on_sleep`] and
//!   [`AgcStream::reset_utterance`] even when the underlying [`EngineStream`] is recreated.

use crate::agc::{WarmAgc, WarmAgcConfig};
use crate::{Engine, EngineError, EngineStream, Workspace};

/// Recogniser audio-path stream combining an [`EngineStream`] with a default-off [`WarmAgc`].
#[derive(Debug, Clone)]
pub struct AgcStream {
    stream: EngineStream,
    agc: WarmAgc,
    scratch: Vec<f32>,
}

impl AgcStream {
    /// Wrap an existing [`EngineStream`] with the given [`WarmAgcConfig`].
    pub fn new(stream: EngineStream, cfg: WarmAgcConfig) -> Result<Self, EngineError> {
        let agc = WarmAgc::new(cfg)?;
        Ok(Self {
            stream,
            agc,
            scratch: Vec::new(),
        })
    }

    /// Create a fresh [`EngineStream`] from `engine` wrapped with `cfg`.
    pub fn from_engine(engine: &Engine, cfg: WarmAgcConfig) -> Result<Self, EngineError> {
        Self::new(engine.create_stream(), cfg)
    }

    pub fn is_agc_enabled(&self) -> bool {
        self.agc.is_enabled()
    }
    pub fn agc(&self) -> &WarmAgc {
        &self.agc
    }
    pub fn agc_mut(&mut self) -> &mut WarmAgc {
        &mut self.agc
    }
    pub fn stream(&self) -> &EngineStream {
        &self.stream
    }
    pub fn stream_mut(&mut self) -> &mut EngineStream {
        &mut self.stream
    }
    pub fn tokens(&self) -> &[u32] {
        self.stream.tokens()
    }
    pub fn input_finished(&mut self) {
        self.stream.input_finished();
    }

    /// Accept 16 kHz PCM16 audio on the recogniser path (identity passthrough when disabled).
    pub fn accept_i16(&mut self, pcm: &[i16]) -> Result<(), EngineError> {
        if !self.agc.is_enabled() {
            return self.stream.accept_i16(pcm);
        }
        self.agc.process_i16_into_f32(pcm, &mut self.scratch);
        self.stream.accept_f32(&self.scratch)
    }

    /// Accept 16 kHz unit-range `f32` audio on the recogniser path (identity when disabled).
    pub fn accept_f32(&mut self, samples: &[f32]) -> Result<(), EngineError> {
        if !self.agc.is_enabled() {
            return self.stream.accept_f32(samples);
        }
        self.scratch.clear();
        self.scratch.extend_from_slice(samples);
        self.agc.process_chunk(&mut self.scratch);
        self.stream.accept_f32(&self.scratch)
    }

    /// Freeze AGC level state across a sleep interval (raw gate/ring remain untouched).
    pub fn on_sleep(&mut self) {
        self.agc.on_sleep();
    }

    /// Start a new utterance on a fresh [`EngineStream`] while preserving warm AGC state.
    pub fn reset_utterance(&mut self, engine: &Engine) {
        self.stream = engine.create_stream();
        self.agc.reset_utterance();
    }

    /// Wake from sleep with a fresh [`EngineStream`] and a raw PCM16 pre-roll slice (e.g. 500 ms).
    ///
    /// Leaves the caller's raw `preroll_pcm` buffer untouched so any Sentinel-0 gate keeps raw
    /// levels, while seeding and processing the recogniser's pre-roll copy.
    pub fn wake_with_preroll_i16(
        &mut self,
        engine: &Engine,
        preroll_pcm: &[i16],
    ) -> Result<(), EngineError> {
        self.stream = engine.create_stream();
        if !self.agc.is_enabled() {
            return self.stream.accept_i16(preroll_pcm);
        }
        self.agc.seed_from_preroll_i16(preroll_pcm);
        self.agc
            .process_i16_into_f32(preroll_pcm, &mut self.scratch);
        self.stream.accept_f32(&self.scratch)
    }

    /// Wake from sleep with a fresh [`EngineStream`] and a raw `f32` pre-roll slice.
    pub fn wake_with_preroll_f32(
        &mut self,
        engine: &Engine,
        preroll: &[f32],
    ) -> Result<(), EngineError> {
        self.stream = engine.create_stream();
        if !self.agc.is_enabled() {
            return self.stream.accept_f32(preroll);
        }
        self.agc.seed_from_preroll(preroll);
        self.scratch.clear();
        self.scratch.extend_from_slice(preroll);
        self.agc.process_chunk(&mut self.scratch);
        self.stream.accept_f32(&self.scratch)
    }

    /// Hard-reset both the underlying [`EngineStream`] and the [`WarmAgc`] back to cold unity.
    pub fn hard_reset(&mut self, engine: &Engine) {
        self.stream = engine.create_stream();
        self.agc.hard_reset();
    }

    /// Advance decoding on the wrapped stream until no full chunk remains ready.
    pub fn decode_ready(
        &mut self,
        engine: &Engine,
        ws: &mut Workspace,
    ) -> Result<usize, EngineError> {
        let mut steps = 0usize;
        while engine.decode(&mut [&mut self.stream], ws)? > 0 {
            steps += 1;
        }
        Ok(steps)
    }
}
