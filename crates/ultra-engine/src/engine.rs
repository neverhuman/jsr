//! Streaming API: per-stream state, and one-chunk-per-ready-stream batched decoding.
//!
//! ```text
//! let engine = Engine::new(Arc::new(Model::load(dir, true)?), Isa::detect(), Precision::F32)?;
//! let mut s = engine.create_stream();
//! s.accept_i16(&pcm); s.input_finished();
//! let mut ws = Workspace::default();
//! while engine.decode(&mut [&mut s], &mut ws)? > 0 {}
//! let text = engine.text(&s);
//! ```

use std::sync::Arc;

use ultra_frontend::{TrainerLogMel, MEL_BINS};
use ultra_gp::{chunk_bounds, FeatureRing};

use crate::decoder::{greedy_chunk, DecState, DecWorkspace};
pub use crate::encoder::Precision;
use crate::encoder::{encode_chunk, ChunkJob, EncState, EncWorkspace, Tap};
use crate::kernels::Isa;
use crate::model::Model;
use crate::EngineError;

/// A loaded model plus the execution choices; cheap to clone and share across workers.
#[derive(Debug, Clone)]
pub struct Engine {
    pub model: Arc<Model>,
    pub isa: Isa,
    pub precision: Precision,
    /// Active encoder chunk length in 25 Hz frames (defaults to `model.arch.chunk`; override at
    /// start-up via [`Engine::with_chunk`] for multi-chunk exports such as 4/8/16 = 160/320/640 ms).
    pub chunk: usize,
    /// Most consecutive ready chunks of one stream run in one step (layer by layer, so one
    /// weight pass serves them all). Outputs are identical for every value.
    pub max_chunks: usize,
    /// Int8 only: 7-bit activation codes with the paired-accumulation kernel (experimental;
    /// re-gate accuracy before use).
    pub act7: bool,
    /// Optional single-stream latency mode: int8 GEMMs split across a thread pool.
    pub pool: Option<Arc<crate::pool::GemmPool>>,
    /// Beam width: 1 = greedy (default), k > 1 = bounded beam search (`crate::beam`).
    pub beam: usize,
    /// Maximum tokens emitted per frame during beam search (`1` default = B6 parity; `2..=8`
    /// recommended for BPE-500 and multirate/V2 models to prevent consecutive-token deletions).
    pub beam_max_symbols: usize,
    /// Entropy-gated Micro-Beam-4 threshold (`0.0` default = always-on exact beam; e.g. `3.5`
    /// collapses to width-1 joiner whenever the top hypothesis & blank logit dominate by `> margin`).
    pub beam_margin: f32,
    /// Opt-in deterministic int8 K/V cache compression (`false` default = fp32 K/V; `true` cuts
    /// per-stream K/V state memory by ~3.5x with per-head/per-slot symmetric int8 quantisation).
    pub kv_int8: bool,
}

/// One utterance/stream: frontend, feature ring, encoder caches and decoder state.
#[derive(Debug, Clone)]
pub struct EngineStream {
    mel: TrainerLogMel,
    rows: Vec<[f32; MEL_BINS]>,
    ring: FeatureRing,
    finished: bool,
    /// Set when a chunk step failed after this stream's features or state were consumed;
    /// the stream is terminal (a retry could silently skip audio).
    failed: bool,
    pub enc: EncState,
    pub dec: DecState,
}

impl EngineStream {
    fn open(&self) -> Result<(), EngineError> {
        if self.failed {
            return Err(EngineError::Lifecycle(
                "stream failed in an earlier chunk".into(),
            ));
        }
        if self.finished {
            return Err(EngineError::Lifecycle("audio after input_finished".into()));
        }
        Ok(())
    }

    pub fn accept_i16(&mut self, pcm: &[i16]) -> Result<(), EngineError> {
        self.open()?;
        self.rows.clear();
        self.mel.push_i16_into(pcm, &mut self.rows);
        self.ring.push_rows(&self.rows);
        Ok(())
    }

    pub fn accept_f32(&mut self, samples: &[f32]) -> Result<(), EngineError> {
        self.open()?;
        self.rows.clear();
        self.mel.push_f32_into(samples, &mut self.rows);
        self.ring.push_rows(&self.rows);
        Ok(())
    }

    /// Feed already-normalised feature rows directly (parity tests bypass the frontend).
    pub fn accept_features(&mut self, rows: &[[f32; MEL_BINS]]) -> Result<(), EngineError> {
        self.open()?;
        self.ring.push_rows(rows);
        Ok(())
    }

    pub fn is_failed(&self) -> bool {
        self.failed
    }

    pub fn input_finished(&mut self) {
        self.finished = true;
    }

    pub fn is_finished(&self) -> bool {
        self.finished
    }

    pub fn n_feat(&self) -> usize {
        self.ring.n_feat()
    }

    pub fn tokens(&self) -> &[u32] {
        &self.dec.tokens
    }

    pub fn forced_advances(&self) -> u64 {
        self.dec.forced
    }
}

/// Per-worker scratch. Tensor buffers are reused (grown on demand, never shrunk); each
/// `decode` call still allocates a few small O(streams) index/reference vectors.
#[derive(Debug, Default)]
pub struct Workspace {
    enc: EncWorkspace,
    dec: DecWorkspace,
    feats: Vec<f32>,
    enc_out: Vec<f32>,
    ready: Vec<(usize, usize)>,
}

impl Workspace {
    /// Turn per-phase accounting on or off (and reset it).
    pub fn set_profiling(&mut self, on: bool) {
        self.enc.q.prof = crate::prof::Prof {
            on,
            ..Default::default()
        };
        self.dec.q.prof = crate::prof::Prof {
            on,
            ..Default::default()
        };
    }

    /// `(encoder, decoder)` phase accounting since the last reset.
    pub fn profile(&self) -> (crate::prof::Prof, crate::prof::Prof) {
        (self.enc.q.prof.clone(), self.dec.q.prof.clone())
    }
}

impl Engine {
    pub fn new(model: Arc<Model>, isa: Isa, precision: Precision) -> Result<Self, EngineError> {
        if !isa.is_available() {
            return Err(EngineError::Unsupported(format!(
                "ISA {} not available on this CPU",
                isa.name()
            )));
        }
        if precision == Precision::F32 && model.blocks.iter().any(|b| !b.qkv.has_fp32()) {
            return Err(EngineError::Unsupported(
                "fp32 requested on a prepared-int8 directory (its quantised layers keep no fp32 \
                 weights); load the fp32 weight directory instead"
                    .into(),
            ));
        }
        if precision == Precision::Int8
            && model
                .blocks
                .iter()
                .all(|b| b.qkv.q.is_none() && b.ff1_w1.q.is_none())
        {
            return Err(EngineError::Unsupported(
                "int8 requested but the model was not quantised (Model::quantize_int8)".into(),
            ));
        }
        let chunk = model.arch.chunk;
        Ok(Self {
            model,
            isa,
            precision,
            chunk,
            max_chunks: 1,
            act7: false,
            pool: None,
            beam: 1,
            beam_max_symbols: 1,
            beam_margin: 0.0,
            kv_int8: false,
        })
    }

    /// Override the streaming chunk length (in 25 Hz encoder frames, `1..=64`; must be even for
    /// multirate models) without re-exporting weights. Streams created afterwards run with the
    /// requested chunk mask (e.g. `4`, `8`, `16` = 160 / 320 / 640 ms). Passing `model.arch.chunk`
    /// is bit-identical to the default engine.
    pub fn with_chunk(mut self, chunk: usize) -> Result<Self, EngineError> {
        if chunk == 0 || chunk > 64 {
            return Err(EngineError::Shape(format!(
                "chunk must be in 1..=64 (got {chunk})"
            )));
        }
        if self.model.arch.mid > 0 && !chunk.is_multiple_of(2) {
            return Err(EngineError::Shape(format!(
                "multirate models require an even chunk length (got {chunk})"
            )));
        }
        self.chunk = chunk;
        Ok(self)
    }

    /// Split each small-batch int8 GEMM across `threads` threads (the caller plus
    /// `threads - 1` pool workers; 1 = off, the default and the density mode). Outputs are
    /// bit-identical to the single-thread engine.
    pub fn with_gemm_threads(mut self, threads: usize) -> Self {
        self.pool = (threads > 1).then(|| Arc::new(crate::pool::GemmPool::new(threads)));
        self
    }

    /// Decode with a bounded beam of width `k` (1..=16; 1 = greedy, the default). Streams
    /// created afterwards carry the beam state; a stream created before gets it at its first
    /// step.
    pub fn with_beam(mut self, k: usize) -> Self {
        self.beam = k.clamp(1, 16);
        self
    }

    /// Set the maximum tokens emitted per encoder frame during beam search (`1..=8`;
    /// `1` = default B6 parity; `2..=4` recommended for BPE-500 / 12.5 Hz / V2 models).
    pub fn with_beam_max_symbols(mut self, k: usize) -> Self {
        self.beam_max_symbols = k.clamp(1, 8);
        self
    }

    /// Enable Entropy-Gated Micro-Beam-4 (`0.0` = off / exact beam; e.g. `3.5` collapses
    /// confident blank frames to width-1 joiner evaluation).
    pub fn with_beam_margin(mut self, margin: f32) -> Self {
        self.beam_margin = if margin.is_finite() && margin > 0.0 {
            margin
        } else {
            0.0
        };
        self
    }

    /// Use 7-bit activation codes in int8 linear layers (see [`Engine::act7`]).
    pub fn with_act7(mut self, on: bool) -> Self {
        self.act7 = on;
        self
    }

    /// Allow up to `k` (1..=16) ready chunks per stream per step. Streams created afterwards
    /// get caches sized for it.
    pub fn with_max_chunks(mut self, k: usize) -> Self {
        self.max_chunks = k.clamp(1, 16);
        self
    }

    /// Enable deterministic int8 K/V cache compression (~3.5x smaller per-stream memory).
    pub fn with_kv_int8(mut self, on: bool) -> Self {
        self.kv_int8 = on;
        self
    }

    pub fn create_stream(&self) -> EngineStream {
        EngineStream {
            mel: TrainerLogMel::new(self.model.mel_cfg.clone()),
            rows: Vec::new(),
            ring: FeatureRing::new(),
            finished: false,
            failed: false,
            enc: EncState::new_with_kv_and_chunk(
                &self.model,
                self.max_chunks,
                self.kv_int8,
                self.chunk,
            ),
            dec: DecState::new(&self.model),
        }
    }

    /// The next step `(t0, t1)` of `s`: up to `max_chunks` consecutive chunks whose features
    /// are complete (limited by the stream's cache capacity).
    pub fn next_chunk(&self, s: &EngineStream) -> Option<(usize, usize)> {
        let c = self.chunk;
        let cap = self.max_chunks.min(s.enc.max_step_frames() / c).max(1);
        let (t0, mut t1) = chunk_bounds(s.enc.t, s.ring.n_feat(), c, s.finished)?;
        for _ in 1..cap {
            if t1 - t0 < c * ((t1 - t0).div_ceil(c)) {
                break; // a partial chunk ends the stream
            }
            match chunk_bounds(t1, s.ring.n_feat(), c, s.finished) {
                Some((_, t2)) => t1 = t2,
                None => break,
            }
        }
        Some((t0, t1))
    }

    pub fn is_ready(&self, s: &EngineStream) -> bool {
        self.next_chunk(s).is_some()
    }

    /// Run one chunk for every ready stream in `streams` (one batched weight pass per
    /// layer). Returns how many streams advanced.
    pub fn decode(
        &self,
        streams: &mut [&mut EngineStream],
        ws: &mut Workspace,
    ) -> Result<usize, EngineError> {
        self.decode_tapped(streams, ws, None)
    }

    /// [`Engine::decode`] with an optional per-layer tap (parity tests).
    pub fn decode_tapped(
        &self,
        streams: &mut [&mut EngineStream],
        ws: &mut Workspace,
        tap: Option<Tap<'_>>,
    ) -> Result<usize, EngineError> {
        let a = &self.model.arch;
        let per = |n: usize| (4 * n + 3) * a.n_mels;
        ws.ready.clear();
        let mut total = 0;
        if streams.iter().any(|s| s.failed) {
            return Err(EngineError::Lifecycle(
                "a stream in the batch has failed".into(),
            ));
        }
        for (i, s) in streams.iter().enumerate() {
            if let Some((t0, t1)) = self.next_chunk(s) {
                ws.ready.push((i, t1 - t0));
                total += per(t1 - t0);
            }
        }
        if ws.ready.is_empty() {
            return Ok(0);
        }
        if ws.feats.len() < total {
            ws.feats.resize(total, 0.0);
        }
        let mut off = 0;
        for &(i, n) in &ws.ready {
            let s = &mut *streams[i];
            let t0 = s.enc.t;
            s.ring
                .extract_chunk_into(t0, n, &mut ws.feats[off..off + per(n)]);
            off += per(n);
        }
        // Split the borrowed streams into encoder and decoder halves, in ready order.
        let mut encs: Vec<(&mut EncState, &mut DecState)> = Vec::with_capacity(ws.ready.len());
        {
            let mut it = streams.iter_mut().enumerate();
            for &(i, _) in &ws.ready {
                for (k, s) in it.by_ref() {
                    if k == i {
                        let s: &mut EngineStream = s;
                        encs.push((&mut s.enc, &mut s.dec));
                        break;
                    }
                }
            }
        }
        let ns: Vec<usize> = ws.ready.iter().map(|&(_, n)| n).collect();
        let mut decs: Vec<&mut DecState> = Vec::with_capacity(encs.len());
        let mut jobs: Vec<ChunkJob<'_>> = Vec::with_capacity(encs.len());
        let mut off = 0;
        for ((enc, dec), &n) in encs.into_iter().zip(&ns) {
            jobs.push(ChunkJob {
                state: enc,
                feats: &ws.feats[off..off + per(n)],
                n,
            });
            decs.push(dec);
            off += per(n);
        }
        ws.enc.q.act7 = self.act7;
        ws.dec.q.act7 = self.act7;
        let split = self.pool.clone().map(|p| (p, self.model.clone()));
        ws.enc.q.split = split.clone();
        ws.dec.q.split = split;
        let m = &self.model;
        let result = encode_chunk(
            m,
            self.isa,
            self.precision,
            &mut jobs,
            &mut ws.enc,
            &mut ws.enc_out,
            tap,
        )
        .and_then(|_| {
            if self.beam > 1 {
                let mut bs: Vec<&mut crate::beam::BeamState> = decs
                    .iter_mut()
                    .map(|d| d.beam.get_or_insert_with(|| crate::beam::BeamState::new(m)))
                    .collect();
                crate::beam::beam_chunk_with_opts(
                    m,
                    self.isa,
                    self.precision,
                    self.beam,
                    self.beam_max_symbols,
                    self.beam_margin,
                    &mut bs,
                    &ns,
                    &ws.enc_out,
                    &mut ws.dec,
                )?;
                for d in decs.iter_mut() {
                    if let Some(b) = &d.beam {
                        d.tokens = b.best().to_vec();
                    }
                }
                Ok(())
            } else {
                greedy_chunk(
                    m,
                    self.isa,
                    self.precision,
                    &mut decs,
                    &ns,
                    &ws.enc_out,
                    &mut ws.dec,
                )
            }
        });
        drop(jobs);
        drop(decs);
        if let Err(e) = result {
            // Features were consumed and caches may have advanced: these streams are terminal.
            for &(i, _) in &ws.ready {
                streams[i].failed = true;
            }
            return Err(e);
        }
        Ok(ns.len())
    }

    /// Detokenised transcript so far.
    pub fn text(&self, s: &EngineStream) -> String {
        let ids: Vec<i64> = s.dec.tokens.iter().map(|&t| i64::from(t)).collect();
        self.model.tokens.decode_ids(&ids)
    }
}
