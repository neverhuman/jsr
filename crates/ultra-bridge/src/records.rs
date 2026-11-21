//! Identities and immutable audio-only workloads, prepared outside measured inference.

use std::collections::BTreeSet;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{require, Result};

pub const SCHEMA: &str = "ultra.bridge/v1";

/// M1 admits only the exact DEV caches pinned in docs/ultra/DATA.lock.
pub fn dev_cache_sha256(dataset: &str) -> Result<&'static str> {
    match dataset {
        "dev-clean" => Ok("bfb27f5857411d7cea5fd8a307bab47793955425fbcb7ce5db5b428a919644b6"),
        "dev-other" => Ok("2944dbff8f3f3158e1192143a4aa6e6b4205681efbb12f1d1fd1c433b21d03c8"),
        _ => Err("bridge currently admits pinned DEV caches only".into()),
    }
}

pub fn host_target() -> String {
    let mut isa: Vec<&str> = Vec::new();
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            isa.push("avx2");
        }
        if std::is_x86_feature_detected!("fma") {
            isa.push("fma");
        }
    }
    format!(
        "{}; detected host ISA {}; Rust compiled avx2={}",
        std::env::consts::ARCH,
        isa.join(","),
        cfg!(target_feature = "avx2")
    )
}

pub fn mapped_runtime_libraries() -> Result<std::collections::BTreeMap<String, String>> {
    let maps = std::fs::read_to_string("/proc/self/maps")?;
    let paths: BTreeSet<_> = maps
        .lines()
        .filter_map(|l| l.split_whitespace().last())
        .filter(|p| p.starts_with('/') && (p.contains("sherpa") || p.contains("libonnxruntime")))
        .collect();
    paths
        .into_iter()
        .map(|p| Ok((p.to_owned(), hash_file(Path::new(p))?)))
        .collect()
}

pub fn hash_file(path: &Path) -> Result<String> {
    let mut f = std::fs::File::open(path)?;
    let mut h = Sha256::new();
    let mut buf = [0u8; 65536];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(format!("{:x}", h.finalize()))
}

/// Retain the exact executable/source bytes used, so later builds cannot erase a pilot.
pub fn retain(src: &Path, dst: &Path) -> Result<String> {
    let expected = hash_file(src)?;
    std::fs::create_dir_all(dst.parent().ok_or("artifact directory")?)?;
    let mut output = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(dst)?;
    std::io::copy(&mut std::fs::File::open(src)?, &mut output)?;
    std::fs::set_permissions(dst, std::fs::metadata(src)?.permissions())?;
    require(
        hash_file(dst)? == expected,
        "runtime artifact changed while retaining",
    )?;
    Ok(expected)
}

pub fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
    Ok(serde_json::from_slice(&std::fs::read(path)?)?)
}

/// Never overwrite a previous run, including an invalid one.
pub fn write_json(path: &Path, value: &impl Serialize) -> Result<()> {
    let mut f = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?;
    serde_json::to_writer_pretty(&mut f, value)?;
    f.write_all(b"\n")?;
    Ok(())
}

pub fn json_line(f: &mut impl Write, value: &impl Serialize) -> Result<()> {
    // File writers otherwise issue a syscall for each JSON fragment. Keep each
    // record immediately visible without buffering it until a later observation.
    let mut record = serde_json::to_vec(value)?;
    record.push(b'\n');
    f.write_all(&record)?;
    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Input {
    pub id: String,
    pub index: usize,
    pub path: String,
    pub samples: u64,
    pub sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Workload {
    pub schema: String,
    pub dataset: String,
    pub source_cache_sha256: String,
    pub sample_rate: u32,
    pub inputs: Vec<Input>,
}

impl Workload {
    pub fn validate(&self) -> Result<()> {
        require(
            self.schema == SCHEMA && self.sample_rate == 16000,
            "unsupported workload",
        )?;
        require(
            matches!(self.dataset.as_str(), "dev-clean" | "dev-other"),
            "bridge currently admits DEV only",
        )?;
        require(!self.inputs.is_empty(), "empty workload")?;
        require(
            self.source_cache_sha256 == dev_cache_sha256(&self.dataset)?,
            "dataset/cache identity differs from the admitted DEV source",
        )?;
        let mut ids = BTreeSet::new();
        let mut indices = BTreeSet::new();
        for i in &self.inputs {
            require(
                i.id == format!("u{:05}", i.index) && ids.insert(&i.id) && indices.insert(i.index),
                "duplicate/empty input identity",
            )?;
            ultra_model::bundle::check_safe_relative_path(&i.path)?;
            require(
                i.samples > 0 && i.sha256.len() == 64,
                "empty or unpinned audio",
            )?;
        }
        Ok(())
    }
}

pub fn prepare(cache: &Path, out: &Path, dataset: String, count: usize) -> Result<()> {
    require(count > 0, "count must be positive")?;
    require(
        matches!(dataset.as_str(), "dev-clean" | "dev-other"),
        "DEV only",
    )?;
    let source_cache_sha256 = hash_file(cache)?;
    require(
        source_cache_sha256 == dev_cache_sha256(&dataset)?,
        "dataset/cache identity differs from the admitted DEV source",
    )?;
    // Reject an unadmitted source before opening any NPZ member. This loader
    // does not read reference, source uid, or speaker members.
    let audio = ultra_frontend::npz::load_npz_audio_only(cache, 0, Some(count))?;
    require(audio.len() == count, "requested audio count unavailable")?;
    std::fs::create_dir(out)?;
    let mut inputs = Vec::new();
    for a in audio {
        let id = a.opaque_id();
        let path = format!("{id}.s16le");
        let mut f = std::fs::File::create(out.join(&path))?;
        for s in &a.pcm {
            f.write_all(&s.to_le_bytes())?;
        }
        inputs.push(Input {
            id,
            index: a.index,
            samples: a.pcm.len() as u64,
            sha256: hash_file(&out.join(&path))?,
            path,
        });
    }
    let workload = Workload {
        schema: SCHEMA.into(),
        dataset,
        source_cache_sha256,
        sample_rate: 16000,
        inputs,
    };
    workload.validate()?;
    write_json(&out.join("inputs.json"), &workload)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Job {
    pub inputs: PathBuf,
    pub bundle: PathBuf,
    /// r1-bridge, r1-native or r0-native; identical Rust scheduling and collection.
    pub engine: String,
    pub precision: String,
    pub out: PathBuf,
    pub streams: usize,
    pub packet_ms: u32,
    pub threads: usize,
    pub paced: bool,
    #[serde(default)]
    pub guarded: bool,
    #[serde(default)]
    pub python: PathBuf,
    #[serde(default)]
    pub worker: PathBuf,
    pub measure: PathBuf,
    pub timeout_s: u64,
    pub host_window: String,
    #[serde(default)]
    pub service_cpus: Option<String>,
    #[serde(default)]
    pub observer_cpus: Option<String>,
}

impl Job {
    pub fn validate(&self) -> Result<()> {
        require(
            matches!(
                self.engine.as_str(),
                "r1-bridge" | "r1-native" | "r0-native"
            ),
            "unsupported engine",
        )?;
        require(
            matches!(self.precision.as_str(), "fp32" | "int8"),
            "unsupported precision",
        )?;
        require(
            self.engine != "r1-bridge"
                || (!self.python.as_os_str().is_empty() && !self.worker.as_os_str().is_empty()),
            "Python bridge requires python and worker paths",
        )?;
        require(
            (1..=32).contains(&self.streams)
                && (1..=100).contains(&self.packet_ms)
                && (1..=64).contains(&self.threads)
                && self.timeout_s > 0,
            "invalid scheduling settings",
        )?;
        require(!self.host_window.is_empty(), "host window label required")?;
        measurement_args(self.service_cpus.as_deref(), self.observer_cpus.as_deref())?;
        Ok(())
    }
}

/// Shared whole-service observer placement for native, bridge and completed-file controls.
pub fn measurement_args(service: Option<&str>, observer: Option<&str>) -> Result<Vec<String>> {
    let mut args = vec!["--interval-ms".into(), "20".into()];
    match (service, observer) {
        (None, None) => {}
        (Some(service), Some(observer)) => {
            ultra_eval::affinity::PlacementSpec {
                service_cpus: service.into(),
                observer_cpus: observer.into(),
            }
            .validate()?;
            args.extend([
                "--service-cpus".into(),
                service.into(),
                "--observer-cpus".into(),
                observer.into(),
            ]);
        }
        _ => return Err("service_cpus and observer_cpus must be supplied together".into()),
    }
    args.push("--".into());
    Ok(args)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Plan {
    pub schema: String,
    pub run_id: String,
    pub job: Job,
    pub workload_sha256: String,
    pub bundle_manifest_sha256: String,
    pub executable_sha256: String,
    pub worker_sha256: String,
    pub consumer_sha256: String,
    pub source_commit: String,
    pub source_dirty: bool,
    pub host: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Observation {
    pub id: String,
    pub accepted_samples: u64,
    /// Conservative watermark: streaming work stays uncredited until EOF drain.
    pub processed_samples: u64,
    pub terminal: bool,
    pub drained: bool,
    pub hyp: String,
    pub forced_advances: Option<u64>,
    pub pcm_sha256: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Outcome {
    pub run_id: String,
    pub worker_id: String,
    pub input_id: String,
    pub source_index: usize,
    pub pcm_sha256: String,
    pub accepted_samples: u64,
    pub processed_samples: u64,
    pub final_text: String,
    pub forced_advances: Option<u64>,
    pub source_start_s: f64,
    pub final_observed_s: f64,
}

pub fn validate_outcomes(run_id: &str, work: &Workload, rows: &[Outcome]) -> Result<()> {
    require(rows.len() == work.inputs.len(), "missing/extra outcomes")?;
    let mut seen = BTreeSet::new();
    for row in rows {
        let i = work
            .inputs
            .iter()
            .find(|i| i.id == row.input_id)
            .ok_or("unexpected output ID")?;
        require(
            seen.insert(&row.input_id)
                && row.run_id == run_id
                && row.worker_id == "w0"
                && row.source_index == i.index
                && row.pcm_sha256 == i.sha256
                && row.accepted_samples == i.samples
                && row.processed_samples == i.samples
                && row.source_start_s.is_finite()
                && row.final_observed_s.is_finite()
                && row.final_observed_s >= row.source_start_s,
            "stale, duplicate, malformed, or incomplete outcome",
        )?;
    }
    Ok(())
}
