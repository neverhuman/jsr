//! Whole-service reference-blind process-tree adapter for `NeMo-Speech.cpp` (`Nemotron 0.6B Q8_0`).

use std::collections::BTreeSet;
use std::fs::File;
use std::os::unix::process::CommandExt;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use ultra_core::events::{Emitter, ProcessingEvidence};
use ultra_core::receipt::{
    sha256_hex, ChunkMs, Hashes, HonestyLabel, Partition, Receipt, RECEIPT_SCHEMA,
};
use ultra_eval::corpus::pool;
use ultra_eval::procfs::TreeSampler;
use ultra_eval::score::{score_records, ScoreRecord};
use ultra_frontend::npz::{load_npz_audio_only, load_npz_references};
use ultra_frontend::write_wav_pcm16;

use crate::nemotron_proc::{
    hash_file_streaming, inspect_pid_maps_cpu, kill_and_verify_pgid_gone, self_and_child_cpu_s,
    whole_tree_cpu_s, write_create_new,
};
use crate::AdapterError;

/// Supported Nemotron causal chunk configurations (`160 ms` `R=1` vs `560 ms` `R=6`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NemotronChunkMode {
    Live160,
    Live560,
}

impl NemotronChunkMode {
    pub fn right_context(self) -> u32 {
        match self {
            Self::Live160 => 1,
            Self::Live560 => 6,
        }
    }

    pub fn chunk_ms(self) -> u32 {
        match self {
            Self::Live160 => 160,
            Self::Live560 => 560,
        }
    }

    pub fn candidate_id(self) -> &'static str {
        match self {
            Self::Live160 => "nemotron_en_q8_live_160",
            Self::Live560 => "nemotron_en_q8_live_560",
        }
    }
}

#[derive(Debug, Clone)]
pub struct NemotronConfig {
    pub binary_path: PathBuf,
    pub model_path: PathBuf,
    pub mode: NemotronChunkMode,
    pub threads: u32,
    pub concurrency: usize,
    pub timeout: Duration,
    pub npz_path: PathBuf,
    pub split_name: String,
    pub partition: Partition,
    pub offset: usize,
    pub limit: Option<usize>,
    pub source_commit: String,
    pub hardware_id: String,
    pub completion_status: String,
    pub require_ggml_cpu_map: bool,
    pub retain_dir: Option<PathBuf>,
}

impl NemotronConfig {
    pub fn default_h0(mode: NemotronChunkMode, npz_path: PathBuf, split_name: String) -> Self {
        let root = std::env::var("JSR_DATA")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("/mnt/xbabe3-external/jsr"));
        Self {
            binary_path: root.join("ext/NeMo-Speech.cpp/build/cpu-asr/bin/nemo-speech"),
            model_path: root
                .join("models/nemotron-0.6b/nemotron-speech-streaming-en-0.6b.q8_0.gguf"),
            mode,
            threads: 1,
            concurrency: 1,
            timeout: Duration::from_secs(1800),
            npz_path,
            split_name,
            partition: Partition::Dev,
            offset: 0,
            limit: Some(100),
            source_commit: "HEAD".into(),
            hardware_id: "h0-threadripper-3995wx".into(),
            completion_status: "exploratory".into(),
            require_ggml_cpu_map: true,
            retain_dir: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NemotronRunResult {
    pub candidate_id: String,
    pub chunk_ms_mode: u32,
    pub utterances: usize,
    pub audio_seconds: f64,
    pub wall_seconds: f64,
    pub prep_hash_cpu_seconds: f64,
    pub child_cpu_seconds: f64,
    pub whole_service_cpu_seconds: f64,
    pub rtf_cpu: f64,
    pub streams_per_core_at_full_load: f64,
    pub load_ms: Option<f64>,
    pub wer_pct: Option<f64>,
    pub binary_sha256: String,
    pub model_sha256: String,
    pub receipt: Receipt,
    pub scored_rows: Vec<ScoreRecord>,
}

/// Execute `nemo-speech` reference-blind on an `.npz` split with whole-service CPU/RSS accounting,
/// file-backed `stdout`/`stderr` capture (preventing >64 KB pipe deadlocks), positive `libggml-cpu`
/// verification, strict 1-to-1 `.txt` output checks, and optional retained artifacts.
pub fn run_nemotron_batch(cfg: &NemotronConfig) -> Result<NemotronRunResult, AdapterError> {
    if !cfg.binary_path.is_file() {
        return Err(AdapterError::Unavailable(format!(
            "nemo-speech binary missing at {}",
            cfg.binary_path.display()
        )));
    }
    if !cfg.model_path.is_file() {
        return Err(AdapterError::Unavailable(format!(
            "Nemotron GGUF missing at {}",
            cfg.model_path.display()
        )));
    }
    let prep_cpu_t0 = whole_tree_cpu_s()?;
    let binary_sha256 = hash_file_streaming(&cfg.binary_path)?;
    let model_sha256 = hash_file_streaming(&cfg.model_path)?;
    let utts = load_npz_audio_only(&cfg.npz_path, cfg.offset, cfg.limit)?;
    if utts.is_empty() {
        return Err(AdapterError::Integrity("empty utterance slice".into()));
    }

    let work = tempfile::tempdir()?;
    let wav_dir = work.path().join("wavs");
    let out_dir = work.path().join("out");
    let trace_path = work.path().join("trace.jsonl");
    let stdout_path = work.path().join("stdout.json");
    let stderr_path = work.path().join("stderr.log");
    std::fs::create_dir_all(&wav_dir)?;
    std::fs::create_dir_all(&out_dir)?;

    let mut expected_txt = BTreeSet::new();
    let mut pcm_hash_concat = String::new();
    let mut audio_s = 0.0f64;
    for (i, u) in utts.iter().enumerate() {
        let stem = format!("u{i:05}");
        let wav_path = wav_dir.join(format!("{stem}.wav"));
        write_wav_pcm16(&wav_path, 16_000, &u.pcm)?;
        pcm_hash_concat.push_str(&hash_file_streaming(&wav_path)?);
        expected_txt.insert(format!("{stem}.txt"));
        audio_s += u.pcm.len() as f64 / 16_000.0;
    }
    let audio_sha = sha256_hex(pcm_hash_concat.as_bytes());
    let prep_hash_cpu_s = (whole_tree_cpu_s()? - prep_cpu_t0).max(0.0);

    let (self_before, child_before) = self_and_child_cpu_s()?;
    let cpu_before = self_before + child_before;
    let wall_t0 = Instant::now();
    let mut self_sampler = TreeSampler::new(std::process::id() as i32);
    self_sampler.sample();

    let stdout_file = File::create(&stdout_path)?;
    let stderr_file = File::create(&stderr_path)?;
    let mut cmd = Command::new(&cfg.binary_path);
    cmd.process_group(0)
        .env("OMP_THREAD_LIMIT", cfg.threads.max(1).to_string())
        .args([
            "--json",
            "bench",
            "asr",
            &wav_dir.to_string_lossy(),
            "--model",
            &cfg.model_path.to_string_lossy(),
            "--device",
            "cpu",
            "--mode",
            "stream",
            "--asr.backend.threads",
            &cfg.threads.max(1).to_string(),
            "--asr.streaming.rnnt_right_context",
            &cfg.mode.right_context().to_string(),
            "--concurrency",
            &cfg.concurrency.max(1).to_string(),
            "--repetitions",
            "1",
            "--warmup",
            "0",
            "--save",
            &out_dir.to_string_lossy(),
            "--trace",
            &trace_path.to_string_lossy(),
        ])
        .stdout(Stdio::from(stdout_file))
        .stderr(Stdio::from(stderr_file));

    let mut child = cmd.spawn()?;
    let child_pid = child.id();
    let pgid = child_pid as i32;
    let mut saw_ggml_cpu = false;
    let mut mapped_libs = BTreeSet::new();

    let status = loop {
        self_sampler.sample();
        if let Some(res) = inspect_pid_maps_cpu(child_pid, &mut mapped_libs) {
            match res {
                Ok(ggml_ok) => saw_ggml_cpu = saw_ggml_cpu || ggml_ok,
                Err(msg) => {
                    let _ = kill_and_verify_pgid_gone(pgid);
                    let _ = child.wait();
                    return Err(AdapterError::GpuDetected(msg));
                }
            }
        }
        if let Some(st) = child.try_wait()? {
            break st;
        }
        if wall_t0.elapsed() >= cfg.timeout {
            let kill_res = kill_and_verify_pgid_gone(pgid);
            let _ = child.wait();
            kill_res?;
            return Err(AdapterError::Process(format!(
                "nemo-speech timed out after {:?}",
                cfg.timeout
            )));
        }
        std::thread::sleep(Duration::from_millis(15));
    };

    let _ = child.wait();
    kill_and_verify_pgid_gone(pgid)?;
    let stdout_bytes = std::fs::read(&stdout_path)?;
    let stderr_bytes = std::fs::read(&stderr_path)?;
    if !status.success() {
        return Err(AdapterError::Process(format!(
            "nemo-speech exited with {}: {}",
            status,
            String::from_utf8_lossy(&stderr_bytes)
        )));
    }
    if cfg.require_ggml_cpu_map && !saw_ggml_cpu {
        return Err(AdapterError::Integrity(
            "libggml-cpu shared library was not observed in child /proc/<pid>/maps".into(),
        ));
    }

    let mut got_txt = BTreeSet::new();
    for entry in std::fs::read_dir(&out_dir)? {
        let name = entry?.file_name().to_string_lossy().into_owned();
        if !got_txt.insert(name.clone()) {
            return Err(AdapterError::Integrity(format!("duplicate output {name}")));
        }
    }
    if got_txt != expected_txt {
        let missing: Vec<_> = expected_txt.difference(&got_txt).take(5).cloned().collect();
        let extra: Vec<_> = got_txt.difference(&expected_txt).take(5).cloned().collect();
        return Err(AdapterError::Integrity(format!(
            "output set mismatch (expected {}, got {}; missing={missing:?}, extra={extra:?})",
            expected_txt.len(),
            got_txt.len()
        )));
    }

    let mut hyps = Vec::with_capacity(utts.len());
    let mut event_stream_bytes = Vec::new();
    let cand_id = cfg.mode.candidate_id();
    for (i, u) in utts.iter().enumerate() {
        let raw = std::fs::read(out_dir.join(format!("u{i:05}.txt")))?;
        let hyp = std::str::from_utf8(&raw)
            .map_err(|_| AdapterError::Integrity(format!("u{i:05}.txt invalid UTF-8")))?
            .trim()
            .to_owned();
        let n_samp = u.pcm.len() as u64;
        let mut em = Emitter::new(&format!("nemo-{i}"), cand_id);
        let ev = em.finalize(
            wall_t0.elapsed().as_nanos() as u64,
            &hyp,
            n_samp,
            ProcessingEvidence {
                accepted_through_sample: n_samp,
                required_processed_through_sample: n_samp,
                flushed: true,
            },
        )?;
        let mut line = serde_json::to_vec(&ev)
            .map_err(|e| AdapterError::Integrity(format!("event json: {e}")))?;
        line.push(b'\n');
        event_stream_bytes.extend_from_slice(&line);
        hyps.push(hyp);
    }

    let stdout_json: serde_json::Value = serde_json::from_slice(&stdout_bytes)
        .map_err(|e| AdapterError::Integrity(format!("nemo-speech stdout JSON invalid: {e}")))?;
    let load_ms = stdout_json["load_ms"].as_f64();
    let cl = &stdout_json["runs"][0]["metrics"]["chunk_latency_ms"];
    let chunk_ms = ChunkMs {
        p50: cl["p50"].as_f64(),
        p95: cl["p95"].as_f64(),
        p99: cl["p99"].as_f64(),
    };

    self_sampler.sample();
    let tree_sum = self_sampler.summary();
    let wall_s = wall_t0.elapsed().as_secs_f64().max(1e-6);
    let (self_after, child_after) = self_and_child_cpu_s()?;
    let child_cpu_s = (child_after - child_before).max(0.0);
    let cpu_s = ((self_after + child_after) - cpu_before).max(1e-6);
    let rtf_cpu = cpu_s / audio_s.max(1e-9);

    // Reference-isolated scoring occurs strictly AFTER child exit and output validation.
    let refs = load_npz_references(&cfg.npz_path, cfg.offset, Some(utts.len()))?;
    let mut scored_rows = Vec::with_capacity(utts.len());
    for ((i, r), hyp) in refs.iter().enumerate().zip(&hyps) {
        scored_rows.push(ScoreRecord {
            id: r.uid.clone(),
            speaker: r.speaker.to_string(),
            dataset: cfg.split_name.clone(),
            reference: r.text.clone(),
            hyp: hyp.clone(),
            duration_s: Some(utts[i].pcm.len() as f64 / 16_000.0),
        });
    }
    let utt_results = score_records(&scored_rows);
    let wer_pct = pool(&utt_results)
        .remove(&cfg.split_name)
        .and_then(|ds| ds.wer)
        .map(|w| (w * 100_000.0).round() / 1000.0);

    let final_bytes = hyps.join("\n").into_bytes();
    if let Some(rdir) = &cfg.retain_dir {
        std::fs::create_dir_all(rdir)?;
        write_create_new(&rdir.join("events.jsonl"), &event_stream_bytes)?;
        write_create_new(&rdir.join("final.txt"), &final_bytes)?;
        write_create_new(&rdir.join("stdout.json"), &stdout_bytes)?;
        write_create_new(&rdir.join("stderr.log"), &stderr_bytes)?;
        let prov = serde_json::json!({
            "binary_path": cfg.binary_path.to_string_lossy(),
            "binary_sha256": binary_sha256,
            "model_path": cfg.model_path.to_string_lossy(),
            "model_sha256": model_sha256,
            "pcm_wav_sha256": audio_sha,
            "ggml_cpu_verified": saw_ggml_cpu,
            "prep_hash_cpu_seconds": (prep_hash_cpu_s * 1000.0).round() / 1000.0,
            "child_cpu_seconds": (child_cpu_s * 1000.0).round() / 1000.0,
            "whole_service_cpu_seconds": (cpu_s * 1000.0).round() / 1000.0,
            "mapped_shared_objects": mapped_libs,
        });
        let prov_bytes = serde_json::to_vec_pretty(&prov)
            .map_err(|e| AdapterError::Integrity(format!("provenance json: {e}")))?;
        write_create_new(&rdir.join("provenance.json"), &prov_bytes)?;
    }

    let empty_sha = sha256_hex(b"");
    let final_sha = sha256_hex(&final_bytes);
    let receipt = Receipt {
        schema: RECEIPT_SCHEMA.into(),
        bundle: cand_id.into(),
        source_commit: cfg.source_commit.clone(),
        hardware: cfg.hardware_id.clone(),
        target: std::env::consts::ARCH.into(),
        configured_threads: cfg.threads.max(1),
        observed_threads: Some(tree_sum.max_threads.max(1) as u32),
        cpu_only_verified: saw_ggml_cpu,
        audio_seconds: (audio_s * 1000.0).round() / 1000.0,
        process_tree_cpu_seconds: Some((cpu_s * 1000.0).round() / 1000.0),
        peak_rss_bytes: Some(tree_sum.peak_rss_bytes.max(1)),
        events: Some(utts.len() as u64),
        wakeups: None,
        cap_hits: None,
        chunk_ms,
        status_events: vec![
            format!("cost_scope:whole_service_process_tree_cold_included"),
            format!("binary_sha256:{binary_sha256}"),
            format!("model_sha256:{model_sha256}"),
            format!("pcm_wav_sha256:{audio_sha}"),
            format!("ggml_cpu_verified:{saw_ggml_cpu}"),
        ],
        gaps: Vec::new(),
        hashes: Hashes {
            events: sha256_hex(&event_stream_bytes),
            preview: empty_sha.clone(),
            commit: empty_sha,
            final_: final_sha,
        },
        label: HonestyLabel::UnknownOverlap,
        partition: cfg.partition,
        completion_status: cfg.completion_status.clone(),
    };

    Ok(NemotronRunResult {
        candidate_id: cand_id.into(),
        chunk_ms_mode: cfg.mode.chunk_ms(),
        utterances: utts.len(),
        audio_seconds: (audio_s * 1000.0).round() / 1000.0,
        wall_seconds: (wall_s * 1000.0).round() / 1000.0,
        prep_hash_cpu_seconds: (prep_hash_cpu_s * 1000.0).round() / 1000.0,
        child_cpu_seconds: (child_cpu_s * 1000.0).round() / 1000.0,
        whole_service_cpu_seconds: (cpu_s * 1000.0).round() / 1000.0,
        rtf_cpu: (rtf_cpu * 100_000.0).round() / 100_000.0,
        streams_per_core_at_full_load: ((1.0 / rtf_cpu) * 100.0).round() / 100.0,
        load_ms,
        wer_pct,
        binary_sha256,
        model_sha256,
        receipt,
        scored_rows,
    })
}
