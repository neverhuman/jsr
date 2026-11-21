//! Reference-blind paced streaming latency scorer (`CQ(250/750/2000ms)`, `stable-word` delay,
//! `FC(800/1500ms)`, and `first_partial_p50_s`) joining MFA `ultra.word-timing/v1` alignments
//! strictly AFTER all streams finish inference.

use std::path::Path;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use ultra_core::receipt::{ChunkMs, HonestyLabel, Partition};
use ultra_eval::delay::{delay_report, quantile_nearest_rank, word_delays, DelayReport};
use ultra_eval::finalize::{
    episodes_from_finals, finalization_report, Episode, FinalizationReport, RefEpisode,
};
use ultra_eval::pace::PacedReplay;
use ultra_eval::stream::{replay, StreamScore, TimedEvent};
use ultra_frontend::npz::{load_npz_audio_only, load_npz_references};
use ultra_stream::{
    AsrStream, Backend, OperatingProfile, OutputPolicy, PollResult, SpAResult, StreamClock,
    StreamConfig, StreamCoordinator,
};

use crate::suite_mfa::load_mfa_post_inference;
use crate::suite_slices::{SilenceProbeReport, SliceSummaries};
use crate::CliError;

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
    pub loaded_identity: String,
    pub driver: String,
    pub cost_scope: String,
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
    pub slices: SliceSummaries,
    pub silence_probe: Option<SilenceProbeReport>,
    pub job_manifest_file: String,
    pub scores_file: String,
    pub receipt_file: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrontierReport {
    pub schema: String,
    pub split: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub panel_manifest_sha256: Option<String>,
    pub grid_ms: u32,
    pub max_chunks: usize,
    pub source_commit: String,
    pub hardware_id: String,
    pub completion_status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub density_ladder_file: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub abab_timing_file: Option<String>,
    pub rows: Vec<FrontierRow>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuiteLatencySummary {
    pub loaded_identity: String,
    pub finalization_mode: String,
    pub whole_population_claim_eligible: bool,
    pub mfa_sha256: Option<String>,
    pub utterances_inferred: usize,
    pub utterances_timed: usize,
    pub timing_unavailable_count: usize,
    pub timing_unavailable_ids: Vec<String>,
    pub population_scope: String,
    pub word_timing_available: bool,
    pub first_partial_p50_s: Option<f64>,
    pub delay: Option<DelayReport>,
    pub finalization: FinalizationReport,
}

/// Run `limit` utterances reference-blind at `1.0x` real-time pace through `backend`, then join
/// MFA `ultra.word-timing/v1` alignments afterward to compute `CQ(250/750/2000ms)` & `FC(800/1500ms)`.
pub fn measure_paced_latency(
    backend: &dyn Backend,
    npz_path: &Path,
    split_name: &str,
    limit: usize,
) -> Result<SuiteLatencySummary, CliError> {
    let loaded_identity = backend.capabilities().name;
    let utts = load_npz_audio_only(npz_path, 0, Some(limit))
        .map_err(|e| CliError::Unavailable(e.to_string()))?;
    let mut scorecards: Vec<StreamScore> = Vec::with_capacity(utts.len());
    let mut first_partials: Vec<f64> = Vec::with_capacity(utts.len());
    let mut durs_s: Vec<f64> = Vec::with_capacity(utts.len());

    for (idx, u) in utts.iter().enumerate() {
        let cfg = StreamConfig {
            stream_id: format!("lat-{idx:05}"),
            model_config_id: loaded_identity.clone(),
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
        scorecards.push(sc);
        durs_s.push(u.pcm.len() as f64 / 16_000.0);
    }

    let refs = load_npz_references(npz_path, 0, Some(utts.len()))
        .map_err(|e| CliError::Unavailable(e.to_string()))?;
    let (mfa_sha256, mfa) = load_mfa_post_inference(split_name, &refs, &durs_s)?;
    let has_mfa = mfa_sha256.is_some();

    let mut all_word_delays: Vec<Option<f64>> = Vec::new();
    let mut all_episodes: Vec<Episode> = Vec::new();
    let mut timing_unavailable_ids: Vec<String> = Vec::new();
    let mut utterances_timed = 0usize;

    for ((u, r), sc) in utts.iter().zip(&refs).zip(&scorecards) {
        let dur_samples = u.pcm.len() as u64;
        if has_mfa {
            let Some(mu) = mfa.get(&r.uid) else {
                timing_unavailable_ids.push(r.uid.clone());
                continue;
            };
            all_word_delays.extend(word_delays(&mu.timed_words, &sc.words, &sc.stable_t_s));
            let s0 = ((mu.start_s * 16_000.0).round() as u64).min(dur_samples);
            let s1 = ((mu.end_s * 16_000.0).round() as u64)
                .clamp(s0.saturating_add(1), dur_samples.max(1));
            let ref_ep = RefEpisode {
                id: r.uid.clone(),
                speech_start_sample: s0,
                speech_end_sample: s1,
                ref_end_s: mu.end_s,
            };
            let eps = episodes_from_finals(&[ref_ep], &sc.finals)
                .map_err(|e| CliError::Internal(e.to_string()))?;
            all_episodes.extend(eps);
            utterances_timed += 1;
        } else {
            timing_unavailable_ids.push(r.uid.clone());
            let ref_ep = RefEpisode {
                id: r.uid.clone(),
                speech_start_sample: 0,
                speech_end_sample: dur_samples.max(1),
                ref_end_s: (dur_samples as f64) / 16_000.0,
            };
            let eps = episodes_from_finals(&[ref_ep], &sc.finals)
                .map_err(|e| CliError::Internal(e.to_string()))?;
            all_episodes.extend(eps);
        }
    }

    first_partials.sort_by(f64::total_cmp);
    let first_partial_p50_s = quantile_nearest_rank(&first_partials, 0.50);
    let delay = (has_mfa && utterances_timed > 0)
        .then(|| delay_report(&all_word_delays, &[250, 750, 2000]));
    let finalization = finalization_report(&all_episodes, &[800, 1500]);
    let population_scope = if !has_mfa {
        "waveform_bounds_no_mfa".to_owned()
    } else if timing_unavailable_ids.is_empty() {
        "mfa_complete_slice".to_owned()
    } else {
        format!("mfa_aligned_subset({utterances_timed}/{})", utts.len())
    };

    let whole_population_claim_eligible = has_mfa
        && timing_unavailable_ids.is_empty()
        && utterances_timed == utts.len()
        && !utts.is_empty();

    Ok(SuiteLatencySummary {
        loaded_identity,
        finalization_mode: "controlled_eof_finish_input".into(),
        whole_population_claim_eligible,
        mfa_sha256,
        utterances_inferred: utts.len(),
        utterances_timed,
        timing_unavailable_count: timing_unavailable_ids.len(),
        timing_unavailable_ids,
        population_scope,
        word_timing_available: has_mfa && utterances_timed > 0,
        first_partial_p50_s,
        delay,
        finalization,
    })
}
