//! Same orchestration for native R0 and the temporary R1 worker.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Instant;

use sha2::{Digest, Sha256};
use ultra_stream::{RawOnlineStream, SherpaLib, SherpaZipformerBackend, SherpaZipformerConfig};

use crate::native_fused::NativeFused;
use crate::native_r1::NativeR1;
use crate::protocol::{Packet, Worker};
use crate::records::{Observation, Plan};
use crate::{require, Result};

struct NativeStream {
    raw: RawOnlineStream,
    accepted: u64,
    finished: bool,
    digest: Sha256,
}

pub struct Native {
    backend: Arc<SherpaZipformerBackend>,
    streams: BTreeMap<String, NativeStream>,
    calls: u64,
    model_batch_ms: Vec<f64>,
}

pub enum Engine {
    R1(Worker),
    R0(Native),
    R1Native(NativeR1),
    R1Fused(Box<NativeFused>),
}

impl Engine {
    pub fn load(plan: &Plan) -> Result<(Self, serde_json::Value)> {
        if plan.job.engine == "r1-fused" {
            let (native, info) = NativeFused::load(plan)?;
            return Ok((Self::R1Fused(Box::new(native)), info));
        }
        if plan.job.engine == "r1-bridge" {
            let w = Worker::load(plan)?;
            let info = w.info.clone();
            return Ok((Self::R1(w), info));
        }
        if plan.job.engine == "r1-native" {
            let (native, info) = NativeR1::load(plan)?;
            return Ok((Self::R1Native(native), info));
        }
        require(plan.job.engine == "r0-native", "unsupported engine")?;
        let started = Instant::now();
        let j = &plan.job;
        let verified = ultra_model::bundle::verify_bundle_dir(&j.bundle, None)?;
        require(
            verified.manifest_sha256 == plan.bundle_manifest_sha256
                && verified.manifest.precision == j.precision,
            "wrong native bundle/precision",
        )?;
        for isa in &verified.manifest.required_isa {
            let supported = match isa.trim().to_ascii_lowercase().as_str() {
                #[cfg(target_arch = "x86_64")]
                "avx2" => std::is_x86_feature_detected!("avx2"),
                #[cfg(target_arch = "x86_64")]
                "fma" => std::is_x86_feature_detected!("fma"),
                _ => false,
            };
            require(supported, format!("unsupported required native ISA: {isa}"))?;
        }
        let suffix = if j.precision == "int8" {
            ".int8.onnx"
        } else {
            ".onnx"
        };
        for name in ["encoder", "decoder", "joiner"] {
            let p = format!("model/{name}-epoch-99-avg-1-chunk-16-left-128{suffix}");
            require(
                verified.manifest.entries.iter().any(|e| e.path == p),
                "native model absent from verified payloads",
            )?;
        }
        require(
            verified
                .manifest
                .entries
                .iter()
                .any(|e| e.path == "model/tokens.txt"),
            "unbound native tokens",
        )?;
        let lib = SherpaLib::load_default()?;
        let version = lib.version();
        let ort_version = lib.onnxruntime_version();
        let backend = SherpaZipformerBackend::new(
            Arc::clone(&lib),
            SherpaZipformerConfig {
                model_dir: j.bundle.join("model"),
                precision: j.precision.clone(),
                num_threads: j.threads as i32,
                tail_pad_s: 0.8,
            },
        )?;
        require(
            ultra_stream::sherpa_ffi::verify_cpu_only_process(),
            "native CPU-only check failed",
        )?;
        let info = serde_json::json!({"manifest_sha256": verified.manifest_sha256,
            "load_s": started.elapsed().as_secs_f64(), "sherpa": version, "ort": ort_version,
            "provider": "cpu", "tail_pad_s": 0.8, "endpoint": "controlled_eof",
            "python_runtime_used": false,
            "runtime_libraries": crate::records::mapped_runtime_libraries()?, "cpu_only_verified": true});
        Ok((
            Self::R0(Native {
                backend,
                streams: BTreeMap::new(),
                calls: 0,
                model_batch_ms: Vec::new(),
            }),
            info,
        ))
    }

    pub fn step(&mut self, packets: &[Packet]) -> Result<Vec<Observation>> {
        match self {
            Self::R1(w) => w.step(packets),
            Self::R0(n) => n.step(packets),
            Self::R1Native(n) => n.step(packets),
            Self::R1Fused(n) => n.step(packets),
        }
    }

    pub fn shutdown(&mut self) -> Result<()> {
        match self {
            Self::R1(w) => w.shutdown(),
            Self::R0(n) => require(n.streams.is_empty(), "unfinished native streams"),
            Self::R1Native(n) => n.shutdown(),
            Self::R1Fused(n) => n.shutdown(),
        }
    }

    pub fn wakeups(&self) -> u64 {
        match self {
            Self::R1(w) => w.wakeups,
            Self::R0(n) => n.calls,
            Self::R1Native(n) => n.calls,
            Self::R1Fused(n) => n.calls,
        }
    }

    pub fn take_model_batch_ms(&mut self) -> Vec<f64> {
        match self {
            Self::R1(w) => std::mem::take(&mut w.model_batch_ms),
            Self::R0(n) => std::mem::take(&mut n.model_batch_ms),
            Self::R1Native(n) => std::mem::take(&mut n.model_batch_ms),
            Self::R1Fused(n) => std::mem::take(&mut n.model_batch_ms),
        }
    }
}

impl Native {
    fn step(&mut self, packets: &[Packet]) -> Result<Vec<Observation>> {
        for p in packets {
            if p.start {
                require(!self.streams.contains_key(&p.id), "duplicate native stream")?;
                self.streams.insert(
                    p.id.clone(),
                    NativeStream {
                        raw: self.backend.create_raw_stream()?,
                        accepted: 0,
                        finished: false,
                        digest: Sha256::new(),
                    },
                );
            }
            let s = self.streams.get_mut(&p.id).ok_or("unknown native stream")?;
            require(!s.finished && s.accepted == p.offset, "native audio order")?;
            for sample in &p.pcm {
                s.digest.update(sample.to_le_bytes());
            }
            s.raw.accept_i16(&p.pcm, &mut Vec::new());
            s.accepted += p.pcm.len() as u64;
            if p.finish {
                s.raw.accept_f32(&vec![0.0; 12800]);
                s.raw.input_finished();
                s.finished = true;
            }
        }
        let mut iterations = 0;
        loop {
            let ready: Vec<_> = self
                .streams
                .values()
                .filter(|s| s.raw.is_ready())
                .map(|s| s.raw.addr())
                .collect();
            if ready.is_empty() {
                break;
            }
            require(iterations < 1024, "native decoder made no bounded progress")?;
            let start = Instant::now();
            self.backend.decode_batch_addrs(&ready);
            self.model_batch_ms
                .push(start.elapsed().as_secs_f64() * 1000.0);
            self.calls += 1;
            iterations += 1;
        }
        let mut rows = Vec::new();
        for p in packets {
            let s = self.streams.get(&p.id).ok_or("lost native stream")?;
            rows.push(Observation {
                id: p.id.clone(),
                accepted_samples: s.accepted,
                processed_samples: if s.finished { s.accepted } else { 0 },
                terminal: s.finished,
                drained: s.finished,
                hyp: s.raw.try_text()?.trim().to_owned(),
                forced_advances: None,
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
}
