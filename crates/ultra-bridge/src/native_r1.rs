//! Native R1 behind the same audio-only service envelope as the measured bridge.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Instant;

use sha2::{Digest, Sha256};
use ultra_gp::{R1Bundle, R1Precision};
use ultra_stream::{BatchWorkspace, OrtR1RawStream, OrtTransducerBackend};

use crate::protocol::Packet;
use crate::records::{hash_file, Observation, Plan};
use crate::{require, Result};

struct Stream {
    raw: OrtR1RawStream,
    accepted: u64,
    finished: bool,
    digest: Sha256,
}

pub struct NativeR1 {
    backend: Arc<OrtTransducerBackend>,
    streams: BTreeMap<String, Stream>,
    workspace: BatchWorkspace,
    failed: bool,
    pub(crate) calls: u64,
    pub(crate) model_batch_ms: Vec<f64>,
}

impl NativeR1 {
    pub fn load(plan: &Plan) -> Result<(Self, serde_json::Value)> {
        let started = Instant::now();
        let job = &plan.job;
        let precision = R1Precision::parse(&job.precision)?;
        require(
            hash_file(&job.bundle.join("manifest.json"))? == plan.bundle_manifest_sha256,
            "R1 manifest identity changed",
        )?;
        // Reject malformed/corrupt bundles before opening a native library. The backend
        // validates again when constructing sessions; both checks are inside cold cost.
        let bundle = R1Bundle::load(&job.bundle, precision, true)?;
        require(
            bundle.manifest_sha256 == plan.bundle_manifest_sha256,
            "R1 bundle changed",
        )?;
        let backend = OrtTransducerBackend::load(&job.bundle, precision, job.threads as i32, true)?;
        require(
            backend.bundle.manifest_sha256 == plan.bundle_manifest_sha256,
            "R1 load identity changed",
        )?;
        require(
            ultra_stream::sherpa_ffi::verify_cpu_only_process(),
            "native R1 CPU-only check failed",
        )?;
        let info = serde_json::json!({
            "manifest_sha256": backend.bundle.manifest_sha256,
            "bundle_schema": backend.bundle.schema,
            "int8_batch_invariant": backend.bundle.int8_batch_invariant,
            "precision": job.precision, "load_s": started.elapsed().as_secs_f64(),
            "ort": backend.ort.version, "ort_library_sha256": backend.ort.sha256,
            "ort_library_path": backend.ort.path,
            "provider": "CPUExecutionProvider; native sessions append no additional provider",
            "cpu_only_verified": true, "python_runtime_used": false,
            "runtime_libraries": crate::records::mapped_runtime_libraries()?,
            "tail_pad_s": 0.0, "endpoint": "controlled_eof",
            "processing_watermark": "conservative EOF drain only",
            "qualification": "experimental; same-precision parity and profile gates are separate"
        });
        Ok((
            Self {
                backend,
                streams: BTreeMap::new(),
                workspace: BatchWorkspace::new(),
                failed: false,
                calls: 0,
                model_batch_ms: Vec::new(),
            },
            info,
        ))
    }

    pub fn step(&mut self, packets: &[Packet]) -> Result<Vec<Observation>> {
        require(!self.failed, "native R1 service already failed")?;
        // Any protocol or inference error permanently invalidates this service instance.
        // It can never retry partially mutated state and later emit a successful Final.
        self.failed = true;
        let rows = self.step_inner(packets)?;
        self.failed = false;
        Ok(rows)
    }

    fn step_inner(&mut self, packets: &[Packet]) -> Result<Vec<Observation>> {
        for packet in packets {
            if packet.start {
                require(
                    !self.streams.contains_key(&packet.id),
                    "duplicate native R1 stream",
                )?;
                self.streams.insert(
                    packet.id.clone(),
                    Stream {
                        raw: self.backend.create_raw_stream(),
                        accepted: 0,
                        finished: false,
                        digest: Sha256::new(),
                    },
                );
            }
            let stream = self
                .streams
                .get_mut(&packet.id)
                .ok_or("unknown native R1 stream")?;
            require(
                !stream.finished && stream.accepted == packet.offset,
                "native R1 audio order",
            )?;
            for sample in &packet.pcm {
                stream.digest.update(sample.to_le_bytes());
            }
            stream.raw.accept_i16(&packet.pcm);
            stream.accepted += packet.pcm.len() as u64;
            if packet.finish {
                stream.raw.input_finished();
                stream.finished = true;
            }
        }
        let mut iterations = 0;
        while let Some(n) = self
            .streams
            .values()
            .filter_map(|s| s.raw.chunk_bounds())
            .map(|(lo, hi)| hi - lo)
            .min()
        {
            require(
                iterations < 1024,
                "native R1 decoder made no bounded progress",
            )?;
            let mut ready: Vec<_> = self
                .streams
                .values_mut()
                .filter(|s| s.raw.chunk_bounds().is_some_and(|(lo, hi)| hi - lo == n))
                .map(|s| &mut s.raw)
                .collect();
            let started = Instant::now();
            let (_, calls) = self
                .backend
                .decode_streams(&mut ready, &mut self.workspace)?;
            require(calls == 1, "unexpected native R1 compatible-group count")?;
            self.model_batch_ms
                .push(started.elapsed().as_secs_f64() * 1000.0);
            self.calls += calls;
            iterations += 1;
        }
        let mut rows = Vec::with_capacity(packets.len());
        for packet in packets {
            let stream = self
                .streams
                .get(&packet.id)
                .ok_or("lost native R1 stream")?;
            require(!stream.raw.is_ready(), "native R1 work remains ready")?;
            rows.push(Observation {
                id: packet.id.clone(),
                accepted_samples: stream.accepted,
                processed_samples: if stream.finished { stream.accepted } else { 0 },
                terminal: stream.finished,
                drained: stream.finished,
                hyp: stream.raw.text(),
                forced_advances: Some(stream.raw.rnnt.forced_advances),
                pcm_sha256: stream
                    .finished
                    .then(|| format!("{:x}", stream.digest.clone().finalize())),
            });
            if stream.finished {
                self.streams.remove(&packet.id);
            }
        }
        Ok(rows)
    }

    pub fn shutdown(&self) -> Result<()> {
        require(
            !self.failed && self.streams.is_empty(),
            "failed or unfinished native R1 service",
        )
    }
}
