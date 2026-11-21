//! Thin audio-only service over the pure Rust engine; no model arithmetic here.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::Instant;

use sha2::{Digest, Sha256};
use ultra_engine::{Engine, EngineStream, Isa, Precision, Workspace};

use crate::protocol::Packet;
use crate::records::{Observation, Plan};
use crate::{require, Result};

struct Stream {
    raw: EngineStream,
    accepted: u64,
    finished: bool,
    digest: Sha256,
    token_count: usize,
    text: String,
}

pub struct NativeFused {
    engine: Engine,
    streams: BTreeMap<String, Stream>,
    workspace: Workspace,
    failed: bool,
    pub(crate) calls: u64,
    pub(crate) model_batch_ms: Vec<f64>,
}

impl NativeFused {
    pub fn load(plan: &Plan) -> Result<(Self, serde_json::Value)> {
        plan.job.validate()?;
        require(
            plan.job.engine == "r1-fused",
            "wrong fused engine selection",
        )?;
        let started = Instant::now();
        let model = crate::fused_bundle::load(&plan.job.bundle, &plan.bundle_manifest_sha256)?;
        let engine = Engine::new(Arc::new(model), Isa::detect(), Precision::F32)?;
        require(
            ultra_stream::sherpa_ffi::verify_cpu_only_process(),
            "fused CPU-only check failed",
        )?;
        let info = serde_json::json!({
            "manifest_sha256": plan.bundle_manifest_sha256,
            "bundle_schema": "ultra.r1-weights/v1",
            "checkpoint_sha256": engine.model.arch.checkpoint_sha256,
            "weights_sha256": engine.model.arch.weights_sha256,
            "precision": "fp32", "isa": engine.isa.name(), "max_chunks": 1,
            "chunk_frames": engine.model.arch.chunk,
            "left_context_frames": engine.model.arch.left,
            "decoder_context": 2, "blank_id": engine.model.arch.blank,
            "max_symbols_per_frame": engine.model.arch.max_symbols,
            "inference_threads": 1, "provider": "pure Rust CPU kernels",
            "cpu_only_verified": true, "python_runtime_used": false,
            "runtime_weight_quantization": false,
            "runtime_libraries": crate::records::mapped_runtime_libraries()?,
            "load_s": started.elapsed().as_secs_f64(), "tail_pad_s": 0.0,
            "endpoint": "controlled_eof", "processing_watermark": "conservative EOF drain only",
            "qualification": "experimental; FP32 parity and profile gates remain separate"
        });
        Ok((
            Self {
                engine,
                streams: BTreeMap::new(),
                workspace: Workspace::default(),
                failed: false,
                calls: 0,
                model_batch_ms: Vec::new(),
            },
            info,
        ))
    }

    pub fn step(&mut self, packets: &[Packet]) -> Result<Vec<Observation>> {
        require(!self.failed, "fused service already failed")?;
        self.failed = true;
        let rows = self.step_inner(packets)?;
        self.failed = false;
        Ok(rows)
    }

    fn step_inner(&mut self, packets: &[Packet]) -> Result<Vec<Observation>> {
        let mut seen = BTreeSet::new();
        for p in packets {
            require(seen.insert(&p.id), "duplicate fused packet ID")?;
            if p.start {
                require(!self.streams.contains_key(&p.id), "duplicate fused stream")?;
                self.streams.insert(
                    p.id.clone(),
                    Stream {
                        raw: self.engine.create_stream(),
                        accepted: 0,
                        finished: false,
                        digest: Sha256::new(),
                        token_count: 0,
                        text: String::new(),
                    },
                );
            }
            let s = self.streams.get_mut(&p.id).ok_or("unknown fused stream")?;
            require(!s.finished && s.accepted == p.offset, "fused audio order")?;
            for sample in &p.pcm {
                s.digest.update(sample.to_le_bytes());
            }
            s.raw.accept_i16(&p.pcm)?;
            s.accepted += p.pcm.len() as u64;
            if p.finish {
                s.raw.input_finished();
                s.finished = true;
            }
        }
        let mut iterations = 0;
        while self.streams.values().any(|s| self.engine.is_ready(&s.raw)) {
            require(iterations < 1024, "fused decoder made no bounded progress")?;
            let mut ready: Vec<_> = self
                .streams
                .values_mut()
                .filter(|s| self.engine.is_ready(&s.raw))
                .map(|s| &mut s.raw)
                .collect();
            let started = Instant::now();
            let advanced = self.engine.decode(&mut ready, &mut self.workspace)?;
            require(advanced > 0, "fused decoder did not advance")?;
            self.model_batch_ms
                .push(started.elapsed().as_secs_f64() * 1000.0);
            self.calls += 1;
            iterations += 1;
        }
        let mut rows = Vec::with_capacity(packets.len());
        for p in packets {
            let s = self.streams.get_mut(&p.id).ok_or("lost fused stream")?;
            require(
                !s.raw.is_failed() && !self.engine.is_ready(&s.raw),
                "fused work remains",
            )?;
            // Greedy RNN-T appends tokens. Packet-only frontend work cannot change
            // detokenization, so reuse the prior text until new tokens exist.
            if s.token_count != s.raw.tokens().len() {
                s.text = self.engine.text(&s.raw);
                s.token_count = s.raw.tokens().len();
            }
            rows.push(Observation {
                id: p.id.clone(),
                accepted_samples: s.accepted,
                processed_samples: if s.finished { s.accepted } else { 0 },
                terminal: s.finished,
                drained: s.finished,
                hyp: s.text.clone(),
                forced_advances: Some(s.raw.forced_advances()),
                pcm_sha256: s
                    .finished
                    .then(|| format!("{:x}", s.digest.clone().finalize())),
            });
            if s.finished {
                self.streams.remove(&p.id);
            }
        }
        Ok(rows)
    }

    pub fn shutdown(&self) -> Result<()> {
        require(
            !self.failed && self.streams.is_empty(),
            "failed or unfinished fused service",
        )
    }
}
