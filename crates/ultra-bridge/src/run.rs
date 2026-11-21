//! Complete-tree measurement wrapper and separate Rust scoring stage.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use ultra_core::receipt::{ChunkMs, Hashes, HonestyLabel, Partition, Receipt, RECEIPT_SCHEMA};
use ultra_eval::procfs::Measurement;
use ultra_eval::score::ScoreRecord;

use crate::records::{
    hash_file, json_line, read_json, validate_outcomes, write_json, Job, Outcome, Plan, Workload,
    SCHEMA,
};
use crate::{require, Result};

fn absolute(p: &Path) -> Result<PathBuf> {
    Ok(if p.is_absolute() {
        p.into()
    } else {
        std::env::current_dir()?.join(p)
    })
}

fn git(args: &[&str]) -> Result<String> {
    let out = Command::new("git").args(args).output()?;
    require(out.status.success(), "cannot bind source revision")?;
    Ok(String::from_utf8(out.stdout)?.trim().to_owned())
}

pub fn execute(job_file: &Path) -> Result<()> {
    let mut job: Job = read_json(job_file)?;
    job.validate()?;
    job.inputs = absolute(&job.inputs)?;
    job.bundle = absolute(&job.bundle)?;
    job.out = absolute(&job.out)?;
    if job.engine == "r1-bridge" {
        job.python = absolute(&job.python)?;
        job.worker = absolute(&job.worker)?;
    }
    job.measure = absolute(&job.measure)?;
    let work: Workload = read_json(&job.inputs)?;
    work.validate()?;
    std::fs::create_dir(&job.out)?;
    let out = job.out.clone();
    // All failures after allocation retain a concrete invalid-run record.
    let result = run_inner(job, &work);
    if let Err(e) = &result {
        write_json(
            &out.join("invalid.json"),
            &serde_json::json!({"valid": false, "reason": e.to_string()}),
        )?;
    }
    result
}

fn run_inner(mut job: Job, work: &Workload) -> Result<()> {
    let exe = std::env::current_exe()?;
    let retained_exe = job.out.join("runtime/ultra-bridge");
    crate::records::retain(&exe, &retained_exe)?;
    let retained_measure = job.out.join("runtime/ultra-measure");
    crate::records::retain(&job.measure, &retained_measure)?;
    job.measure = retained_measure;
    let (worker_sha256, consumer_sha256) = if job.engine == "r1-bridge" {
        let consumer = job
            .worker
            .parent()
            .ok_or("consumer path")?
            .join("../train/bundle_infer.py");
        let retained_worker = job.out.join("runtime/python/bridge/worker.py");
        let worker_hash = crate::records::retain(&job.worker, &retained_worker)?;
        let consumer_hash = crate::records::retain(
            &consumer,
            &job.out.join("runtime/python/train/bundle_infer.py"),
        )?;
        std::fs::write(job.out.join("runtime/python/train/__init__.py"), b"")?;
        job.worker = retained_worker;
        (worker_hash, consumer_hash)
    } else {
        // These sources are unused by native inference. No Python installation is required.
        let empty = ultra_core::receipt::sha256_hex(b"");
        (empty.clone(), empty)
    };
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_nanos();
    let plan = Plan {
        schema: SCHEMA.into(),
        run_id: format!("bridge-{now}-{}", std::process::id()),
        workload_sha256: hash_file(&job.inputs)?,
        bundle_manifest_sha256: hash_file(&job.bundle_manifest_path())?,
        executable_sha256: hash_file(&exe)?,
        worker_sha256,
        consumer_sha256,
        source_commit: git(&["rev-parse", "HEAD"])?,
        source_dirty: !git(&[
            "status",
            "--porcelain",
            "--",
            "crates/",
            "python/ultra/bridge/",
            "python/ultra/train/bundle_infer.py",
            "Cargo.toml",
            "Cargo.lock",
            ".cargo/",
            "rust-toolchain.toml",
        ])?
        .is_empty(),
        host: serde_json::json!({"loadavg": std::fs::read_to_string("/proc/loadavg")?,
            "cpuinfo_sha256": hash_file(Path::new("/proc/cpuinfo"))?,
            "status": std::fs::read_to_string("/proc/self/status")?,
            "measurement_binary_sha256": hash_file(&job.measure)?,
            "hardware": "H0", "qualification": "exploratory_shared_host"}),
        job,
    };
    let j = &plan.job;
    let plan_path = j.out.join("plan.json");
    write_json(&plan_path, &plan)?;
    let manifest = std::fs::read(j.bundle_manifest_path())?;
    require(
        ultra_core::receipt::sha256_hex(&manifest) == plan.bundle_manifest_sha256,
        "bundle changed during preparation",
    )?;
    std::fs::write(j.out.join("bundle-manifest.json"), manifest)?;
    let stderr = std::fs::File::create(j.out.join("service.stderr"))?;
    let measured = Command::new(&j.measure)
        .args(crate::records::measurement_args(
            j.service_cpus.as_deref(),
            j.observer_cpus.as_deref(),
        )?)
        .arg(&retained_exe)
        .arg("service")
        .arg(&plan_path)
        .stderr(Stdio::from(stderr))
        .output()?;
    std::fs::write(j.out.join("measurement.raw.json"), &measured.stdout)?;
    require(
        measured.status.success(),
        format!("inference service/measurement exited {}", measured.status),
    )?;
    let m: Measurement = serde_json::from_slice(&measured.stdout)?;
    require(
        m.exit_code == Some(0)
            && m.cpu_seconds_user.is_finite()
            && m.cpu_seconds_user >= 0.0
            && m.cpu_seconds_sys.is_finite()
            && m.cpu_seconds_sys >= 0.0
            && m.wall_seconds.is_finite()
            && m.wall_seconds > 0.0
            && m.max_threads > 0
            && m.peak_rss_bytes > 0,
        "invalid whole-service measurement",
    )?;
    let rows = outcomes(&j.out)?;
    validate_outcomes(&plan.run_id, work, &rows)?;
    let service: serde_json::Value = read_json(&j.out.join("service.json"))?;
    require(
        service["run_id"] == plan.run_id
            && service["complete"] == true
            && service["unfinished_samples"] == 0
            && service["expected"] == work.inputs.len(),
        "missing/incomplete service result",
    )?;
    let audio = work.inputs.iter().map(|i| i.samples).sum::<u64>() as f64 / 16000.0;
    let digest = hash_file(&j.out.join("events.jsonl"))?;
    let load: serde_json::Value = read_json(&j.out.join("load.json"))?;
    let cpu_only = load["cpu_only_verified"] == true;
    require(cpu_only, "effective CPU execution check missing/failed")?;
    let receipt = Receipt {
        schema: RECEIPT_SCHEMA.into(),
        bundle: format!("models/sha256-{}", plan.bundle_manifest_sha256),
        source_commit: plan.source_commit.clone(),
        hardware: "H0 (exploratory shared host)".into(),
        target: crate::records::host_target(),
        configured_threads: j.threads as u32,
        observed_threads: Some(m.max_threads.try_into()?),
        cpu_only_verified: cpu_only,
        audio_seconds: audio,
        process_tree_cpu_seconds: Some(m.cpu_seconds_user + m.cpu_seconds_sys),
        peak_rss_bytes: Some(m.peak_rss_bytes),
        events: service["events"].as_u64(),
        wakeups: service["wakeups"].as_u64(),
        cap_hits: service["forced_advances"].as_u64(),
        chunk_ms: if j.guarded {
            ChunkMs::default()
        } else {
            serde_json::from_value::<ChunkMs>(service["chunk_ms"].clone())?
        },
        status_events: vec![
            "exploratory; cold-inclusive; controlled EOF; no profile qualification".into(),
        ],
        gaps: Vec::new(),
        hashes: Hashes {
            events: digest,
            preview: hash_file(&j.out.join("preview.jsonl"))?,
            commit: hash_file(&j.out.join("commit.jsonl"))?,
            final_: hash_file(&j.out.join("final.jsonl"))?,
        },
        label: HonestyLabel::InCorpus,
        partition: Partition::Dev,
        completion_status: "exploratory".into(),
    };
    let mut evidence_gaps = Vec::new();
    if let Err(e) = receipt.validate_as_evidence() {
        evidence_gaps.push(e.to_string());
    }
    if plan.source_dirty {
        evidence_gaps.push("runtime source has uncommitted changes".into());
    }
    if j.guarded {
        evidence_gaps.push("guarded wall timings are unavailable as evidence".into());
    }
    if receipt.cap_hits.is_none() {
        evidence_gaps.push("decoder cap counters unavailable".into());
    }
    let total_cpu = m.cpu_seconds_user + m.cpu_seconds_sys;
    let cold_cpu = service["cold_tree_cpu_s"]
        .as_f64()
        .ok_or("cold CPU snapshot missing")?;
    require(
        cold_cpu.is_finite() && cold_cpu >= 0.0 && cold_cpu <= total_cpu,
        "invalid cold CPU snapshot",
    )?;
    write_json(&j.out.join("receipt.json"), &receipt)?;
    write_json(
        &j.out.join("summary.json"),
        &serde_json::json!({"schema": SCHEMA, "run_id": plan.run_id,
        "valid": true, "qualifying": false, "runtime_path": j.engine, "precision": j.precision,
        "evidence_valid": evidence_gaps.is_empty(), "evidence_gaps": evidence_gaps,
        "dataset": work.dataset, "utterances": rows.len(), "audio_s": audio,
        "cpu_s_per_audio_s": (m.cpu_seconds_user + m.cpu_seconds_sys) / audio,
        "audio_s_per_wall_s": (!j.guarded).then_some(audio / m.wall_seconds), "paced": j.paced, "guarded": j.guarded,
        "wall_timing_evidence": !j.guarded,
        "cost_scope": "whole service tree including cold load, IPC, audio I/O, features, decode, collection and drain",
        "measurement": m, "cold_load": load,
        "latency_cq_fc": null, "backlog_qualification": null,
        "warm_service_cost": {"cpu_s": total_cpu - cold_cpu, "cpu_s_per_audio_s": (total_cpu - cold_cpu) / audio,
            "cold_tree_cpu_s": cold_cpu, "snapshot_resolution_s": service["cpu_snapshot_resolution_s"],
            "scope": "whole-service CPU minus post-load tree snapshot; includes audio I/O, IPC, features, formatting and drain"},
        "model_batch_timing_scope": service["chunk_timing_scope"],
        "tail_padding_s": if j.engine == "r0-native" {0.8} else {0.0},
        "wakeup_scope": "actual model batch calls; variable compatible stream occupancy",
        "decoder_cap_hits": service["forced_advances"],
        "event_hash_scope": "exact observed event-log integrity; timestamps/interleaving may differ on rerun",
        "aa_output_hash_scope": "source-sorted final.jsonl and commit.jsonl only",
        "processing_watermark": "conservative EOF drain only; no streaming backlog claim",
        "training_cost": "not supplied by inference bridge; see training record",
        "host_load_after": std::fs::read_to_string("/proc/loadavg")?}),
    )?;
    let mut files = vec![
        "plan.json",
        "events.jsonl",
        "observations.jsonl",
        "outcomes.jsonl",
        "service.json",
        "load.json",
        "measurement.raw.json",
        "receipt.json",
        "summary.json",
        "service.stderr",
        "preview.jsonl",
        "commit.jsonl",
        "final.jsonl",
        "bundle-manifest.json",
        "runtime/ultra-bridge",
        "runtime/ultra-measure",
    ];
    if j.engine == "r1-bridge" {
        files.extend([
            "runtime/python/bridge/worker.py",
            "runtime/python/train/bundle_infer.py",
            "runtime/python/train/__init__.py",
        ]);
    }
    checksums(&j.out, "checksums.json", &files)
}

pub fn outcomes(out: &Path) -> Result<Vec<Outcome>> {
    std::fs::read_to_string(out.join("outcomes.jsonl"))?
        .lines()
        .map(|line| Ok(serde_json::from_str(line)?))
        .collect()
}

fn checksums(out: &Path, name: &str, files: &[&str]) -> Result<()> {
    let mut hashes = BTreeMap::new();
    for f in files {
        hashes.insert(*f, hash_file(&out.join(f))?);
    }
    write_json(&out.join(name), &hashes)
}

#[derive(serde::Deserialize)]
struct ScoringBinding {
    run_id: String,
    job: ScoringJob,
    workload_sha256: String,
}

#[derive(serde::Deserialize)]
struct ScoringJob {
    inputs: PathBuf,
}

pub fn score(out: &Path, cache: &Path) -> Result<()> {
    require(
        !out.join("invalid.json").exists(),
        "cannot score invalid run",
    )?;
    let checks: BTreeMap<String, String> = read_json(&out.join("checksums.json"))?;
    for (file, digest) in checks {
        ultra_model::bundle::check_safe_relative_path(&file)?;
        require(
            hash_file(&out.join(file))? == digest,
            "run artifact changed after measurement",
        )?;
    }
    let plan: ScoringBinding = read_json(&out.join("plan.json"))?;
    let work: Workload = read_json(&plan.job.inputs)?;
    work.validate()?;
    require(
        hash_file(&plan.job.inputs)? == plan.workload_sha256
            && hash_file(cache)? == work.source_cache_sha256,
        "scorer input differs from prepared audio source",
    )?;
    let rows = outcomes(out)?;
    validate_outcomes(&plan.run_id, &work, &rows)?;
    // Reference members are opened only here, after the inference tree has exited.
    let refs = ultra_frontend::npz::load_npz_references(cache, 0, None)?;
    let mut records = Vec::new();
    let mut source_ids = BTreeSet::new();
    for input in &work.inputs {
        let row = rows
            .iter()
            .find(|r| r.input_id == input.id)
            .ok_or("missing outcome")?;
        let reference = refs.get(input.index).ok_or("missing reference")?;
        require(
            source_ids.insert(reference.uid.clone()),
            "duplicate source reference ID",
        )?;
        require(
            reference.index == input.index
                && reference.duration_s == input.samples as f64 / 16000.0,
            "reference/audio join mismatch",
        )?;
        records.push(ScoreRecord {
            id: reference.uid.clone(),
            speaker: reference.speaker.to_string(),
            dataset: work.dataset.clone(),
            reference: reference.text.clone(),
            hyp: row.final_text.clone(),
            duration_s: Some(reference.duration_s),
        });
    }
    let mut f = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(out.join("scores.jsonl"))?;
    for r in &records {
        json_line(&mut f, r)?;
    }
    let weights =
        ultra_eval::corpus::MacroWeights::equal(ultra_eval::score::scorable_datasets(&records))?;
    let cfg = ultra_eval::bootstrap::BootstrapConfig {
        replicates: 10000,
        seed: 20261120,
        alpha: 0.05,
    };
    let report = ultra_eval::score::score_report(&records, None, &weights, &cfg)?;
    write_json(&out.join("score.json"), &report)?;
    std::fs::write(out.join("score.md"), ultra_eval::score::markdown(&report))?;
    let clusters: BTreeSet<_> = records.iter().map(|r| &r.speaker).collect();
    write_json(
        &out.join("support.json"),
        &serde_json::json!({"utterances": records.len(),
        "independent_speaker_clusters": clusters.len(), "selection": "DEV exploratory",
        "scorer_executable_sha256": hash_file(&std::env::current_exe()?)?,
        "inference_checksums_sha256": hash_file(&out.join("checksums.json"))?}),
    )?;
    checksums(
        out,
        "score-checksums.json",
        &["scores.jsonl", "score.json", "score.md", "support.json"],
    )
}
