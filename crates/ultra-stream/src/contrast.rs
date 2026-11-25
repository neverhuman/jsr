//! Reference-blind P05 contrast evaluation (`Moonshine v2`, `Whisper small.en`) and shared
//! benchmark receipt / scoring helpers.

use std::path::{Path, PathBuf};
use std::time::Instant;

use nix::sys::resource::{getrusage, UsageWho};
use serde::{Deserialize, Serialize};
use ultra_core::events::{Emitter, Ledger, ProcessingEvidence};
use ultra_core::receipt::{
    sha256_hex, ChunkMs, Hashes, HonestyLabel, Partition, Receipt, RECEIPT_SCHEMA,
};
use ultra_eval::corpus::{pool, DatasetScore};
use ultra_eval::procfs::TreeSampler;
use ultra_eval::score::{score_records, ScoreRecord};
use ultra_eval::wer::EditCounts;
use ultra_frontend::npz::{load_npz_references, AudioUtterance};

use crate::api::StreamError;
use crate::sherpa_ffi::{verify_cpu_only_process, SherpaLib};
use crate::sherpa_offline::{SherpaOfflineBackend, SherpaOfflineConfig};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WerSummary {
    pub wer_pct: f64,
    pub sub: u64,
    pub del: u64,
    pub ins: u64,
    pub ref_words: u64,
}

#[derive(Debug, Clone)]
pub struct ContrastConfig {
    pub npz_path: PathBuf,
    pub split_name: String,
    pub partition: Partition,
    pub honesty_label: HonestyLabel,
    pub offset: usize,
    pub limit: Option<usize>,
    pub selected_indices: Option<Vec<usize>>,
    pub batch_size: usize,
    pub offline: SherpaOfflineConfig,
    pub source_commit: String,
    pub hardware_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContrastResult {
    pub schema: String,
    pub candidate_id: String,
    pub driver: String,
    pub sherpa_version: String,
    pub onnxruntime_version: String,
    pub split: String,
    pub utterances: usize,
    pub batch_size: usize,
    pub threads: i32,
    pub audio_seconds: f64,
    pub cpu_seconds: f64,
    pub wall_seconds: f64,
    pub rtf_cpu: f64,
    pub streams_per_core_at_full_load: f64,
    pub wer: WerSummary,
    pub affinity: Vec<usize>,
    pub load_before: [f64; 3],
    pub load_after: [f64; 3],
    pub receipt: Receipt,
    #[serde(skip)]
    pub scored_rows: Vec<ScoreRecord>,
}

pub(crate) fn self_cpu_user_sys_seconds() -> Result<(f64, f64), StreamError> {
    let ru = getrusage(UsageWho::RUSAGE_SELF)
        .map_err(|e| StreamError::Runtime(format!("getrusage failed: {e}")))?;
    let u = ru.user_time();
    let s = ru.system_time();
    Ok((
        u.tv_sec() as f64 + u.tv_usec() as f64 / 1e6,
        s.tv_sec() as f64 + s.tv_usec() as f64 / 1e6,
    ))
}

pub(crate) fn self_cpu_seconds() -> Result<f64, StreamError> {
    let (user_s, sys_s) = self_cpu_user_sys_seconds()?;
    Ok(user_s + sys_s)
}

pub(crate) fn self_max_rss_bytes() -> u64 {
    let Ok(ru) = getrusage(UsageWho::RUSAGE_SELF) else {
        return 0;
    };
    let raw = ru.max_rss().max(0) as u64;
    #[cfg(target_os = "macos")]
    {
        raw
    }
    #[cfg(not(target_os = "macos"))]
    {
        raw.saturating_mul(1024)
    }
}

pub(crate) fn read_loadavg() -> [f64; 3] {
    let Ok(s) = std::fs::read_to_string("/proc/loadavg") else {
        return [0.0; 3];
    };
    let mut vals = [0.0f64; 3];
    for (dst, tok) in vals.iter_mut().zip(s.split_whitespace().take(3)) {
        if let Ok(v) = tok.parse::<f64>() {
            *dst = v;
        }
    }
    vals
}

pub(crate) fn current_affinity() -> Vec<usize> {
    if let Ok(set) = nix::sched::sched_getaffinity(nix::unistd::Pid::from_raw(0)) {
        let mut out = Vec::new();
        for cpu in 0..nix::sched::CpuSet::count() {
            if set.is_set(cpu).unwrap_or(false) {
                out.push(cpu);
            }
        }
        return out;
    }
    let Ok(status) = std::fs::read_to_string("/proc/thread-self/status")
        .or_else(|_| std::fs::read_to_string("/proc/self/status"))
    else {
        return Vec::new();
    };
    let Some(line) = status
        .lines()
        .find_map(|l| l.strip_prefix("Cpus_allowed_list:"))
    else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for part in line.trim().split(',').filter(|p| !p.is_empty()) {
        if let Some((a, b)) = part.split_once('-') {
            if let (Ok(lo), Ok(hi)) = (a.parse::<usize>(), b.parse::<usize>()) {
                out.extend(lo..=hi);
            }
        } else if let Ok(c) = part.parse::<usize>() {
            out.push(c);
        }
    }
    out
}

pub(crate) fn percentile_sorted(sorted: &[f64], pct: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let pos = (pct / 100.0) * ((sorted.len() - 1) as f64);
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    let frac = pos - (lo as f64);
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

pub(crate) fn build_event_hashes_and_chunks(
    model_id: &str,
    audio_utts: &[AudioUtterance],
    completed: &[(usize, String)],
    tick_compute_ms: &mut Vec<f64>,
) -> Result<(Hashes, ChunkMs), StreamError> {
    let mut emitter = Emitter::new("bench-stream-0", model_id);
    let mut ledger = Ledger::new();
    let mut serialized_events = Vec::new();
    let mut final_concat = String::new();
    let mut cum_samples = 0u64;
    for (utt_pos, hyp) in completed {
        cum_samples += audio_utts[*utt_pos].pcm.len() as u64;
        let ts_ns = (cum_samples * 1_000_000_000) / 16_000;
        let ev = emitter.finalize(
            ts_ns,
            hyp,
            cum_samples,
            ProcessingEvidence {
                accepted_through_sample: cum_samples,
                required_processed_through_sample: cum_samples,
                flushed: true,
            },
        )?;
        ledger.apply(&ev)?;
        let bytes = serde_json::to_vec(&ev).map_err(|e| StreamError::Runtime(e.to_string()))?;
        serialized_events.extend_from_slice(&bytes);
        serialized_events.push(b'\n');
        if !final_concat.is_empty() {
            final_concat.push('\n');
        }
        final_concat.push_str(hyp);
    }
    if tick_compute_ms.is_empty() {
        tick_compute_ms.push(0.0);
    }
    tick_compute_ms.sort_by(f64::total_cmp);
    let chunk_ms = ChunkMs {
        p50: Some((percentile_sorted(tick_compute_ms, 50.0) * 1000.0).round() / 1000.0),
        p95: Some((percentile_sorted(tick_compute_ms, 95.0) * 1000.0).round() / 1000.0),
        p99: Some((percentile_sorted(tick_compute_ms, 99.0) * 1000.0).round() / 1000.0),
    };
    let hashes = Hashes {
        events: sha256_hex(&serialized_events),
        preview: sha256_hex(b""),
        commit: sha256_hex(ledger.committed_text().as_bytes()),
        final_: sha256_hex(final_concat.as_bytes()),
    };
    Ok((hashes, chunk_ms))
}

pub(crate) fn load_selected_audio(
    npz_path: &Path,
    offset: usize,
    limit: Option<usize>,
    selected_indices: Option<&[usize]>,
) -> Result<Vec<ultra_frontend::npz::AudioUtterance>, StreamError> {
    if let Some(idxs) = selected_indices {
        let all = ultra_frontend::npz::load_npz_audio_only(npz_path, 0, None)?;
        let mut out = Vec::with_capacity(idxs.len());
        for &i in idxs {
            let u = all.get(i).ok_or_else(|| {
                StreamError::Protocol(format!(
                    "panel utterance index {i} out of bounds (split has {})",
                    all.len()
                ))
            })?;
            out.push(u.clone());
        }
        Ok(out)
    } else {
        Ok(ultra_frontend::npz::load_npz_audio_only(
            npz_path, offset, limit,
        )?)
    }
}

pub(crate) fn score_completed(
    npz_path: &Path,
    split_name: &str,
    offset: usize,
    limit: Option<usize>,
    selected_indices: Option<&[usize]>,
    unique_seq_ids: bool,
    completed: &[(usize, String)],
) -> Result<(WerSummary, Vec<ScoreRecord>), StreamError> {
    let refs = if selected_indices.is_some() {
        load_npz_references(npz_path, 0, None)?
    } else {
        load_npz_references(npz_path, offset, limit)?
    };
    let mut scored_rows = Vec::with_capacity(completed.len());
    for (seq_k, (utt_pos, hyp)) in completed.iter().enumerate() {
        let ref_idx = match selected_indices {
            Some(idxs) => idxs[*utt_pos],
            None => *utt_pos,
        };
        let r = &refs[ref_idx];
        let row_id = if unique_seq_ids {
            format!("{}#{}", r.uid, seq_k)
        } else {
            r.uid.clone()
        };
        scored_rows.push(ScoreRecord {
            id: row_id,
            speaker: r.speaker.to_string(),
            dataset: split_name.to_owned(),
            reference: r.text.clone(),
            hyp: hyp.clone(),
            duration_s: Some(r.duration_s),
        });
    }
    let utt_results = score_records(&scored_rows);
    let pooled = match pool(&utt_results).remove(split_name) {
        Some(ds) => ds,
        None => DatasetScore {
            counts: EditCounts::default(),
            wer: None,
            utterances: 0,
            speakers: 0,
            silence: Default::default(),
        },
    };
    let wer_pct = match pooled.wer {
        Some(w) => (w * 100.0 * 1000.0).round() / 1000.0,
        None => 0.0,
    };
    Ok((
        WerSummary {
            wer_pct,
            sub: pooled.counts.substitutions,
            del: pooled.counts.deletions,
            ins: pooled.counts.insertions,
            ref_words: pooled.counts.ref_words,
        },
        scored_rows,
    ))
}

pub fn default_npz_for_split(split: &str) -> PathBuf {
    let data = match std::env::var("JSR_DATA") {
        Ok(v) if !v.trim().is_empty() => v,
        _ => String::from("/mnt/xbabe3-external/jsr"),
    };
    Path::new(&data)
        .join("cache")
        .join("librispeech_pcm")
        .join(format!("{split}.npz"))
}

pub fn run_offline_contrast(cfg: &ContrastConfig) -> Result<ContrastResult, StreamError> {
    let audio_utts = load_selected_audio(
        &cfg.npz_path,
        cfg.offset,
        cfg.limit,
        cfg.selected_indices.as_deref(),
    )?;
    if audio_utts.is_empty() {
        return Err(StreamError::Protocol("no audio utterances loaded".into()));
    }
    let lib = SherpaLib::load_default()?;
    let sherpa_ver = lib.version();
    let ort_ver = lib.onnxruntime_version();
    let engine = SherpaOfflineBackend::new(lib, cfg.offline.clone())?;

    let mut sampler = TreeSampler::new(std::process::id() as i32);
    sampler.sample();
    let load_before = read_loadavg();
    let cpu_t0 = self_cpu_seconds()?;
    let wall_t0 = Instant::now();

    let bsz = cfg.batch_size.max(1);
    let mut completed: Vec<(usize, String)> = Vec::with_capacity(audio_utts.len());
    let mut batch_ms = Vec::new();
    let mut wakeups = 0u64;
    for (base_idx, group) in audio_utts.chunks(bsz).enumerate() {
        let slices: Vec<&[i16]> = group.iter().map(|u| u.pcm.as_slice()).collect();
        let t0 = Instant::now();
        let out = engine.transcribe_batch(&slices)?;
        batch_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
        wakeups += 1;
        for (k, hyp) in out.into_iter().enumerate() {
            completed.push((base_idx * bsz + k, hyp));
        }
        sampler.sample();
    }

    let cpu_seconds = (self_cpu_seconds()? - cpu_t0).max(1e-6);
    let wall_seconds = wall_t0.elapsed().as_secs_f64().max(1e-6);
    let res_sum = sampler.summary();
    let load_after = read_loadavg();
    let cpu_only = verify_cpu_only_process();
    let audio_done_s: f64 = audio_utts.iter().map(AudioUtterance::seconds).sum();
    drop(engine);

    let (hashes, chunk_ms) = build_event_hashes_and_chunks(
        &cfg.offline.candidate_id,
        &audio_utts,
        &completed,
        &mut batch_ms,
    )?;
    let receipt = Receipt {
        schema: RECEIPT_SCHEMA.into(),
        bundle: cfg.offline.candidate_id.clone(),
        source_commit: cfg.source_commit.clone(),
        hardware: cfg.hardware_id.clone(),
        target: std::env::consts::ARCH.into(),
        configured_threads: cfg.offline.num_threads.max(1) as u32,
        observed_threads: Some(res_sum.max_threads.max(1) as u32),
        cpu_only_verified: cpu_only,
        audio_seconds: (audio_done_s * 1000.0).round() / 1000.0,
        process_tree_cpu_seconds: Some((cpu_seconds * 1000.0).round() / 1000.0),
        peak_rss_bytes: Some(res_sum.peak_rss_bytes.max(1)),
        events: Some(completed.len() as u64),
        wakeups: Some(wakeups),
        cap_hits: Some(0),
        chunk_ms,
        status_events: Vec::new(),
        gaps: Vec::new(),
        hashes,
        label: cfg.honesty_label,
        partition: cfg.partition,
        completion_status: "complete".into(),
    };
    receipt.validate_as_evidence()?;

    let (wer_summary, scored_rows) = score_completed(
        &cfg.npz_path,
        &cfg.split_name,
        cfg.offset,
        cfg.limit,
        cfg.selected_indices.as_deref(),
        false,
        &completed,
    )?;

    Ok(ContrastResult {
        schema: "ultra.contrast/v1".into(),
        candidate_id: cfg.offline.candidate_id.clone(),
        driver: "rust-sherpa-c-api".into(),
        sherpa_version: sherpa_ver,
        onnxruntime_version: ort_ver,
        split: cfg.split_name.clone(),
        utterances: completed.len(),
        batch_size: bsz,
        threads: cfg.offline.num_threads,
        audio_seconds: (audio_done_s * 100.0).round() / 100.0,
        cpu_seconds: (cpu_seconds * 1000.0).round() / 1000.0,
        wall_seconds: (wall_seconds * 1000.0).round() / 1000.0,
        rtf_cpu: ((cpu_seconds / audio_done_s.max(1e-9)) * 100_000.0).round() / 100_000.0,
        streams_per_core_at_full_load: ((audio_done_s / cpu_seconds.max(1e-9)) * 100.0).round()
            / 100.0,
        wer: wer_summary,
        affinity: current_affinity(),
        load_before,
        load_after,
        receipt,
        scored_rows,
    })
}
