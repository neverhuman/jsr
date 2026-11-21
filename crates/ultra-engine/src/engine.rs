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
    /// Most consecutive ready chunks of one stream run in one step (layer by layer, so one
    /// weight pass serves them all). Outputs are identical for every value.
    pub max_chunks: usize,
    /// Int8 only: 7-bit activation codes with the paired-accumulation kernel (experimental;
    /// re-gate accuracy before use).
    pub act7: bool,
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
        Ok(Self {
            model,
            isa,
            precision,
            max_chunks: 1,
            act7: false,
        })
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

    pub fn create_stream(&self) -> EngineStream {
        EngineStream {
            mel: TrainerLogMel::new(self.model.mel_cfg.clone()),
            rows: Vec::new(),
            ring: FeatureRing::new(),
            finished: false,
            failed: false,
            enc: EncState::new(&self.model, self.max_chunks),
            dec: DecState::new(&self.model),
        }
    }

    /// The next step `(t0, t1)` of `s`: up to `max_chunks` consecutive chunks whose features
    /// are complete (limited by the stream's cache capacity).
    pub fn next_chunk(&self, s: &EngineStream) -> Option<(usize, usize)> {
        let c = self.model.arch.chunk;
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
            greedy_chunk(
                m,
                self.isa,
                self.precision,
                &mut decs,
                &ns,
                &ws.enc_out,
                &mut ws.dec,
            )
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
