//! Paced real-clock streaming latency scorer (`CQ(250/750/2000ms)`, `stable-word` delay,
//! `FC(800/1500ms)`, and `first_partial_p50_s`) against MFA `ultra.word-timing/v1` alignments.

use std::collections::BTreeMap;
use std::path::Path;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use ultra_core::receipt::{ChunkMs, HonestyLabel, Partition};
use ultra_eval::delay::{
    delay_report, normalise_timed, quantile_nearest_rank, word_delays, DelayReport, RefWord,
};
use ultra_eval::finalize::{
    episodes_from_finals, finalization_report, Episode, FinalizationReport, RefEpisode,
};
use ultra_eval::pace::PacedReplay;
use ultra_eval::stream::{replay, TimedEvent};
use ultra_frontend::npz::{load_npz_audio_only, load_npz_references};
use ultra_stream::{
    AsrStream, Backend, OperatingProfile, OutputPolicy, PollResult, SpAResult, StreamClock,
    StreamConfig, StreamCoordinator,
};

use crate::{jsr_data_root, CliError};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DensityPoint {
    pub workers: usize,
    pub streams_per_worker: usize,
    pub total_streams: usize,
    pub sustained: Option<bool>,
    pub drained: bool,
    pub lateness_p99_ms: Option<f64>,
    pub peak_rss_bytes: Option<u64>,
}

impl DensityPoint {
    pub(crate) fn from_sp_a(res: &SpAResult, streams: usize) -> Self {
        Self {
            workers: 1,
            streams_per_worker: streams,
            total_streams: streams,
            sustained: res.sustained,
            drained: res.drained,
            lateness_p99_ms: res.lateness_ms.as_ref().map(|l| l.p99),
            peak_rss_bytes: res.receipt.peak_rss_bytes,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrontierRow {
    pub candidate: String,
    pub driver: String,
    pub honesty_label: HonestyLabel,
    pub partition: Partition,
    pub completion_status: String,
    pub utterances: usize,
    pub audio_seconds: f64,
    pub wer_pct: f64,
    pub single_rtf_cpu: f64,
    pub single_streams_per_core: f64,
    pub batch_rtf_cpu: Option<f64>,
    pub batch_streams_per_core: Option<f64>,
    pub chunk_ms: ChunkMs,
    pub peak_rss_bytes: Option<u64>,
    pub cap_hits: Option<u64>,
    pub evidence_complete: bool,
    pub density: Option<DensityPoint>,
    pub latency: Option<SuiteLatencySummary>,
    pub sustained_probe: Option<DensityPoint>,
    pub scores_file: String,
    pub receipt_file: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrontierReport {
    pub schema: String,
    pub split: String,
    pub source_commit: String,
    pub hardware_id: String,
    pub completion_status: String,
    pub rows: Vec<FrontierRow>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuiteLatencySummary {
    pub utterances_requested: usize,
    pub utterances_timed: usize,
    pub timing_unavailable_count: usize,
    pub timing_unavailable_ids: Vec<String>,
    pub population_scope: String,
    pub word_timing_available: bool,
    pub first_partial_p50_s: Option<f64>,
    pub delay: Option<DelayReport>,
    pub finalization: FinalizationReport,
}

#[derive(Deserialize)]
struct MfaRow {
    id: String,
    words: Vec<MfaWord>,
}

#[derive(Deserialize)]
struct MfaWord {
    w: String,
    start_s: f64,
    end_s: f64,
}

struct MfaUtterance {
    timed_words: Vec<RefWord>,
    start_s: f64,
    end_s: f64,
}

/// Returns `(mfa_file_exists, aligned_utterances_by_id)`.
/// Rejects non-finite or reversed word timestamps and leaves empty/unaligned rows (such as
/// `test-other` `8131-117029-0017`) absent from the map so the caller records them explicitly in
/// `timing_unavailable_ids` rather than silently claiming whole-split coverage.
fn load_mfa_map(split_name: &str) -> Result<(bool, BTreeMap<String, MfaUtterance>), CliError> {
    let mut map = BTreeMap::new();
    let path = std::path::PathBuf::from(jsr_data_root())
        .join("corpora/librispeech/word_timing")
        .join(format!("{split_name}.jsonl"));
    if !path.is_file() {
        return Ok((false, map));
    }
    let txt = std::fs::read_to_string(&path)?;
    for (line_no, line) in txt.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let row: MfaRow = serde_json::from_str(line).map_err(|e| {
            CliError::Config(format!(
                "malformed MFA timing JSON at {}:{}: {e}",
                path.display(),
                line_no + 1
            ))
        })?;
        if row.words.is_empty() {
            continue;
        }
        for w in &row.words {
            if !w.start_s.is_finite()
                || !w.end_s.is_finite()
                || w.start_s < 0.0
                || w.end_s < w.start_s
            {
                return Err(CliError::Config(format!(
                    "non-finite or reversed MFA timestamp in {} for `{}`",
                    path.display(),
                    row.id
                )));
            }
        }
        let start_s = row.words.first().map_or(0.0, |w| w.start_s);
        let end_s = row.words.last().map_or(0.0, |w| w.end_s);
        let pairs: Vec<(String, f64)> = row.words.into_iter().map(|w| (w.w, w.end_s)).collect();
        let timed_words = normalise_timed(&pairs);
        if !timed_words.is_empty() {
            map.insert(
                row.id,
                MfaUtterance {
                    timed_words,
                    start_s,
                    end_s,
                },
            );
        }
    }
    Ok((true, map))
}

/// Run `limit` utterances at `1.0x` real-time pace through `backend` (`StreamCoordinator`) and
/// compute MFA `CQ(250/750/2000ms)` + `stable-word` delay + `FC(800/1500ms)`.
pub fn measure_paced_latency(
    backend: &dyn Backend,
    npz_path: &Path,
    split_name: &str,
    limit: usize,
) -> Result<SuiteLatencySummary, CliError> {
    let utts = load_npz_audio_only(npz_path, 0, Some(limit))
        .map_err(|e| CliError::Unavailable(e.to_string()))?;
    let refs = load_npz_references(npz_path, 0, Some(limit))
        .map_err(|e| CliError::Unavailable(e.to_string()))?;
    let (mfa_file_exists, mfa) = load_mfa_map(split_name)?;
    let has_mfa = mfa_file_exists && !mfa.is_empty();

    let mut all_word_delays: Vec<Option<f64>> = Vec::new();
    let mut all_episodes: Vec<Episode> = Vec::new();
    let mut first_partials: Vec<f64> = Vec::new();
    let mut timing_unavailable_ids: Vec<String> = Vec::new();
    let mut utterances_timed = 0usize;

    for (u, r) in utts.iter().zip(&refs) {
        if has_mfa && !mfa.contains_key(&r.uid) {
            timing_unavailable_ids.push(r.uid.clone());
            continue;
        }

        let cfg = StreamConfig {
            stream_id: format!("lat-{}", r.uid),
            policy: OutputPolicy::EditableDictation,
            profile: OperatingProfile::Efficient,
            ..StreamConfig::default()
        };
        let shared_clock = StreamClock::real();
        let inner = backend.create_stream_with_clock(&cfg, shared_clock.clone())?;
        let mut coord = StreamCoordinator::with_clock(cfg.clone(), inner, shared_clock.clone());
        let mut timed = Vec::new();

        for pkt in
            PacedReplay::new(&u.pcm, 20, 1.0).map_err(|e| CliError::Internal(e.to_string()))?
        {
            let due_ns = ((pkt.start_sample + pkt.pcm.len() as u64) * 1_000_000_000) / 16_000;
            let now_ns = shared_clock.now_ns();
            if due_ns > now_ns {
                std::thread::sleep(Duration::from_nanos(due_ns - now_ns));
            }
            coord.push_pcm(pkt.start_sample, pkt.pcm)?;
            while let PollResult::Event(ev) = coord.poll_event()? {
                let t_s = (ev.observed_monotonic_ns as f64) * 1e-9;
                timed.push(TimedEvent { t_s, event: *ev });
            }
        }
        coord.finish_input()?;
        while let PollResult::Event(ev) = coord.poll_event()? {
            let t_s = (ev.observed_monotonic_ns as f64) * 1e-9;
            timed.push(TimedEvent { t_s, event: *ev });
        }

        let sc = replay(&timed).map_err(|e| CliError::Internal(e.to_string()))?;
        if let Some(fp) = sc.snapshots.iter().find(|s| !s.text.trim().is_empty()) {
            first_partials.push(fp.t_s);
        }

        let dur_samples = u.pcm.len() as u64;
        let (sp_start, sp_end, ref_end_s) = if let Some(mu) = mfa.get(&r.uid) {
            all_word_delays.extend(word_delays(&mu.timed_words, &sc.words, &sc.stable_t_s));
            let s0 = ((mu.start_s * 16_000.0).round() as u64).min(dur_samples);
            let s1 = ((mu.end_s * 16_000.0).round() as u64)
                .clamp(s0.saturating_add(1), dur_samples.max(1));
            (s0, s1, mu.end_s)
        } else {
            timing_unavailable_ids.push(r.uid.clone());
            (0, dur_samples.max(1), (dur_samples as f64) / 16_000.0)
        };
        let ref_ep = RefEpisode {
            id: r.uid.clone(),
            speech_start_sample: sp_start,
            speech_end_sample: sp_end,
            ref_end_s,
        };
        let eps = episodes_from_finals(&[ref_ep], &sc.finals)
            .map_err(|e| CliError::Internal(e.to_string()))?;
        all_episodes.extend(eps);
        utterances_timed += 1;
    }

    first_partials.sort_by(f64::total_cmp);
    let first_partial_p50_s = quantile_nearest_rank(&first_partials, 0.50);
    let delay = has_mfa.then(|| delay_report(&all_word_delays, &[250, 750, 2000]));
    let finalization = finalization_report(&all_episodes, &[800, 1500]);
    let population_scope = if !has_mfa {
        "waveform_bounds_no_mfa".to_owned()
    } else if timing_unavailable_ids.is_empty() {
        "mfa_complete_slice".to_owned()
    } else {
        format!("mfa_aligned_subset({utterances_timed}/{})", utts.len())
    };

    Ok(SuiteLatencySummary {
        utterances_requested: utts.len(),
        utterances_timed,
        timing_unavailable_count: timing_unavailable_ids.len(),
        timing_unavailable_ids,
        population_scope,
        word_timing_available: has_mfa,
        first_partial_p50_s,
        delay,
        finalization,
    })
}
