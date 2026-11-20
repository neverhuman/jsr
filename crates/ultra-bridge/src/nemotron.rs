//! Completed-file control using the pinned CPU CLI. No live pacing/latency claim.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use ultra_model::candidate::{parse_candidate_docs, CandidateDoc};

use crate::records::{hash_file, read_json, validate_outcomes, write_json, Workload, SCHEMA};
use crate::{require, Result};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Job {
    pub inputs: PathBuf,
    pub candidate: PathBuf,
    pub candidate_id: String,
    pub binary: PathBuf,
    pub model: PathBuf,
    pub out: PathBuf,
    pub concurrency: usize,
    pub threads: usize,
    pub chunk_ms: u32,
    pub measure: PathBuf,
    pub timeout_s: u64,
    pub host_window: String,
    #[serde(default)]
    pub guarded: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Plan {
    pub schema: String,
    pub run_id: String,
    pub job: Job,
    pub workload_sha256: String,
    pub candidate: CandidateDoc,
    pub candidate_sha256: String,
    pub binary_sha256: String,
    pub model_sha256: String,
    pub libraries: BTreeMap<String, String>,
    pub executable_sha256: String,
    pub source_commit: String,
    pub source_dirty: bool,
    pub host: serde_json::Value,
}

pub fn execute(job_file: &Path) -> Result<()> {
    let mut j: Job = read_json(job_file)?;
    require(
        (1..=32).contains(&j.concurrency)
            && (1..=64).contains(&j.threads)
            && matches!(j.chunk_ms, 160 | 560)
            && j.timeout_s > 0
            && !j.host_window.is_empty(),
        "invalid file control configuration",
    )?;
    for p in [
        &mut j.inputs,
        &mut j.candidate,
        &mut j.binary,
        &mut j.model,
        &mut j.out,
        &mut j.measure,
    ] {
        if p.is_relative() {
            *p = std::env::current_dir()?.join(&*p);
        }
    }
    let work: Workload = read_json(&j.inputs)?;
    work.validate()?;
    std::fs::create_dir(&j.out)?;
    let out = j.out.clone();
    let result = run_inner(j, &work);
    if let Err(e) = &result {
        write_json(
            &out.join("invalid.json"),
            &serde_json::json!({"valid": false, "reason": e.to_string()}),
        )?;
    }
    result
}

fn run_inner(mut j: Job, work: &Workload) -> Result<()> {
    let candidate_text = std::fs::read_to_string(&j.candidate)?;
    let candidate = parse_candidate_docs(&candidate_text)?
        .into_iter()
        .find(|c| c.candidate_id == j.candidate_id)
        .ok_or("candidate ID absent")?;
    candidate.validate(true)?;
    require(
        candidate.stream.requested_chunk_ms == j.chunk_ms && candidate.runtime.backend == "cpu",
        "candidate operating point differs",
    )?;
    let model_sha256 = hash_file(&j.model)?;
    require(
        candidate.model.artifact_sha256.as_deref() == Some(&model_sha256),
        "wrong actual GGUF",
    )?;
    require(
        candidate.runtime.build_path.as_deref() == j.binary.to_str(),
        "wrong actual CLI path",
    )?;
    let mut libraries = BTreeMap::new();
    for entry in std::fs::read_dir(j.binary.parent().ok_or("CLI directory")?)? {
        let p = entry?.path();
        if p.is_file()
            && p.file_name()
                .is_some_and(|n| n.to_string_lossy().contains(".so"))
        {
            libraries.insert(p.to_string_lossy().into_owned(), hash_file(&p)?);
        }
    }
    require(
        libraries.keys().any(|p| p.contains("ggml-cpu")),
        "CPU library not pinned",
    )?;
    let exe = std::env::current_exe()?;
    let retained_exe = j.out.join("runtime/ultra-bridge");
    crate::records::retain(&exe, &retained_exe)?;
    let retained_measure = j.out.join("runtime/ultra-measure");
    crate::records::retain(&j.measure, &retained_measure)?;
    j.measure = retained_measure;
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_nanos();
    let git = Command::new("git").args(["rev-parse", "HEAD"]).output()?;
    require(git.status.success(), "source identity unavailable")?;
    let dirty = Command::new("git")
        .args(["status", "--porcelain"])
        .output()?;
    require(dirty.status.success(), "source status unavailable")?;
    let plan = Plan {
        schema: SCHEMA.into(),
        run_id: format!("nemotron-file-{now}-{}", std::process::id()),
        workload_sha256: hash_file(&j.inputs)?,
        candidate,
        candidate_sha256: hash_file(&j.candidate)?,
        binary_sha256: hash_file(&j.binary)?,
        model_sha256,
        libraries,
        executable_sha256: hash_file(&exe)?,
        source_commit: String::from_utf8(git.stdout)?.trim().into(),
        source_dirty: !dirty.stdout.is_empty(),
        host: serde_json::json!({
            "loadavg": std::fs::read_to_string("/proc/loadavg")?, "status": std::fs::read_to_string("/proc/self/status")?,
            "measurement_binary_sha256": hash_file(&j.measure)?}),
        job: j,
    };
    let j = &plan.job;
    write_json(&j.out.join("plan.json"), &plan)?;
    let measured = Command::new(&j.measure)
        .args(["--interval-ms", "20", "--"])
        .arg(retained_exe)
        .arg("service-nemotron")
        .arg(j.out.join("plan.json"))
        .stderr(Stdio::from(std::fs::File::create(
            j.out.join("service.stderr"),
        )?))
        .output()?;
    std::fs::write(j.out.join("measurement.raw.json"), &measured.stdout)?;
    require(
        measured.status.success(),
        "Nemotron service/measurement failed",
    )?;
    let m: ultra_eval::procfs::Measurement = serde_json::from_slice(&measured.stdout)?;
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
        "invalid measurement",
    )?;
    let rows = crate::run::outcomes(&j.out)?;
    validate_outcomes(&plan.run_id, work, &rows)?;
    let service: serde_json::Value = read_json(&j.out.join("service.json"))?;
    require(
        service["run_id"] == plan.run_id && service["complete"] == true,
        "missing service completion",
    )?;
    let cpu_only = service["cpu_check"]["ggml_cpu_seen"] == true
        && service["cpu_check"]["forbidden_gpu_seen"] == false;
    require(cpu_only, "CPU execution evidence unavailable or failed")?;
    let receipt = ultra_core::receipt::Receipt { schema: ultra_core::receipt::RECEIPT_SCHEMA.into(),
        bundle: format!("models/sha256-{}", plan.model_sha256), source_commit: plan.source_commit.clone(),
        hardware: "H0 (exploratory shared host)".into(), target: crate::records::host_target(), configured_threads: j.threads as u32,
        observed_threads: Some(m.max_threads.try_into()?), cpu_only_verified: cpu_only,
        audio_seconds: work.inputs.iter().map(|i| i.samples).sum::<u64>() as f64 / 16000.0,
        process_tree_cpu_seconds: Some(m.cpu_seconds_user + m.cpu_seconds_sys), peak_rss_bytes: Some(m.peak_rss_bytes),
        events: Some(rows.len() as u64), wakeups: None, cap_hits: None,
        chunk_ms: if j.guarded { ultra_core::receipt::ChunkMs::default() } else { serde_json::from_value(service["chunk_ms"].clone())? },
        status_events: vec!["exploratory file-mode; unknown-overlap; cold-inclusive; decoder cap counters unavailable".into()],
        gaps: Vec::new(), hashes: ultra_core::receipt::Hashes {
            events: hash_file(&j.out.join("events.jsonl"))?, preview: hash_file(&j.out.join("preview.jsonl"))?,
            commit: hash_file(&j.out.join("commit.jsonl"))?, final_: hash_file(&j.out.join("final.jsonl"))? },
        label: ultra_core::receipt::HonestyLabel::UnknownOverlap, partition: ultra_core::receipt::Partition::Dev,
        completion_status: "exploratory".into() };
    let mut evidence_gaps = Vec::new();
    if let Err(e) = receipt.validate_as_evidence() {
        evidence_gaps.push(e.to_string());
    }
    evidence_gaps.push("decoder cap counters and model batch count unavailable".into());
    if j.guarded {
        evidence_gaps.push("guarded wall timings are unavailable as evidence".into());
    }
    write_json(&j.out.join("receipt.json"), &receipt)?;
    write_json(
        &j.out.join("summary.json"),
        &serde_json::json!({"schema": SCHEMA, "run_id": plan.run_id,
        "valid": true, "qualifying": false, "runtime_path": "nemotron-completed-file-control", "precision": "Q8_0",
        "evidence_valid": evidence_gaps.is_empty(), "evidence_gaps": evidence_gaps,
        "wakeup_scope": "internal model batch count unavailable; one external CLI invocation",
        "honesty_label": "unknown-overlap", "dataset": work.dataset, "utterances": rows.len(),
        "audio_s": receipt.audio_seconds, "cpu_s_per_audio_s": receipt.process_tree_cpu_seconds.unwrap_or(0.0) / receipt.audio_seconds,
        "audio_s_per_wall_s": (!j.guarded).then_some(receipt.audio_seconds / m.wall_seconds), "measurement": m,
        "guarded": j.guarded, "wall_timing_evidence": !j.guarded,
        "cost_scope": "whole service tree including cold load, WAV formatting, model verification, CLI, collection and drain",
        "paced": false, "latency_cq_fc": null, "warm_service_cost": null, "decoder_cap_hits": null,
        "cold_load": service["cold_load"], "host_load_after": std::fs::read_to_string("/proc/loadavg")?}),
    )?;
    let mut checks = BTreeMap::new();
    for name in [
        "plan.json",
        "service.json",
        "events.jsonl",
        "final.jsonl",
        "preview.jsonl",
        "commit.jsonl",
        "outcomes.jsonl",
        "receipt.json",
        "summary.json",
        "measurement.raw.json",
        "service.stderr",
        "cli.json",
        "cli.stderr",
        "effective-cli.json",
        "wav-checksums.json",
        "runtime/ultra-bridge",
        "runtime/ultra-measure",
    ] {
        checks.insert(name, hash_file(&j.out.join(name))?);
    }
    write_json(&j.out.join("checksums.json"), &checks)
}

pub fn cli_args(plan: &Plan) -> Vec<String> {
    let j = &plan.job;
    vec![
        "--json".into(),
        "bench".into(),
        "asr".into(),
        j.out.join("wav").to_string_lossy().into_owned(),
        "--model".into(),
        j.model.to_string_lossy().into_owned(),
        "--device".into(),
        "cpu".into(),
        "--mode".into(),
        "stream".into(),
        "--asr.backend.threads".into(),
        j.threads.to_string(),
        "--asr.streaming.rnnt_right_context".into(),
        (j.chunk_ms / 80 - 1).to_string(),
        "--concurrency".into(),
        j.concurrency.to_string(),
        "--repetitions".into(),
        "1".into(),
        "--warmup".into(),
        "0".into(),
        "--save".into(),
        j.out.join("hyp").to_string_lossy().into_owned(),
    ]
}

pub fn run_child(plan: &Plan) -> Result<serde_json::Value> {
    use std::os::unix::process::CommandExt;
    let j = &plan.job;
    let mut child = Command::new(&j.binary)
        .args(cli_args(plan))
        .env("OMP_THREAD_LIMIT", j.threads.to_string())
        .env("CUDA_VISIBLE_DEVICES", "")
        .stdout(Stdio::from(std::fs::File::create(j.out.join("cli.json"))?))
        .stderr(Stdio::from(std::fs::File::create(
            j.out.join("cli.stderr"),
        )?))
        .process_group(0)
        .spawn()?;
    let started = Instant::now();
    let (mut samples, mut cpu_seen, mut gpu_seen) = (0u64, false, false);
    loop {
        if let Some(status) = child.try_wait()? {
            require(status.success(), format!("Nemotron exited {status}"))?;
            break;
        }
        if started.elapsed() > Duration::from_secs(j.timeout_s) {
            // SAFETY: the CLI owns this dedicated group; no unrelated process is targeted.
            unsafe {
                libc::kill(-(child.id() as i32), libc::SIGKILL);
            }
            let _ = child.wait();
            return Err("Nemotron file control timed out".into());
        }
        if let Ok(maps) = std::fs::read_to_string(format!("/proc/{}/maps", child.id())) {
            samples += 1;
            cpu_seen |= maps.contains("libggml-cpu");
            gpu_seen |= [
                "libcuda",
                "libcudart",
                "libvulkan",
                "libggml-cuda",
                "libggml-vulkan",
            ]
            .iter()
            .any(|s| maps.contains(s));
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    require(
        samples > 0 && cpu_seen && !gpu_seen,
        "actual CPU library evidence unavailable/failed",
    )?;
    Ok(
        serde_json::json!({"map_samples": samples, "ggml_cpu_seen": cpu_seen, "forbidden_gpu_seen": gpu_seen}),
    )
}
