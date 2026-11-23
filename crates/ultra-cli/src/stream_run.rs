//! Single-file and paced-stream execution (`transcribe`, `stream`) with chunk-coalesced wakeups.

use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use nix::sys::resource::{getrusage, UsageWho};
use ultra_core::events::{Body, TranscriptEvent};
use ultra_core::receipt::{
    sha256_hex, ChunkMs, Hashes, HonestyLabel, Partition, Receipt, RECEIPT_SCHEMA,
};
use ultra_eval::pace::PacedReplay;
use ultra_eval::procfs::TreeSampler;
use ultra_eval::stream::TimedEvent;
use ultra_frontend::read_audio_16k_mono;
use ultra_stream::{
    AsrStream, Backend, NativeEngineBackend, OperatingProfile, OutputPolicy, PollResult,
    StreamClock, StreamConfig, StreamCoordinator,
};

use crate::runner::instantiate_backend;
use crate::stream_commit::StreamCommitCliOptions;
use crate::{git_short_commit, CliError};

/// Wakeup scheduling cadence for paced live streams (`ultra stream`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WakeCadence {
    /// Wake once per model chunk (`chunk_samples`, e.g. 160 ms = 6.25 wakes/s at chunk 4).
    PerChunk,
    /// Legacy control: wake once per transport packet (`--packet-ms`, e.g. 20 ms = 50 wakes/s).
    PerPacket,
}

impl WakeCadence {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::PerChunk => "per_chunk",
            Self::PerPacket => "per_packet",
        }
    }
}

fn self_cpu_s() -> Result<f64, CliError> {
    let ru = getrusage(UsageWho::RUSAGE_SELF)
        .map_err(|e| CliError::Internal(format!("getrusage: {e}")))?;
    let (u, s) = (ru.user_time(), ru.system_time());
    Ok((u.tv_sec() as f64 + u.tv_usec() as f64 / 1e6)
        + (s.tv_sec() as f64 + s.tv_usec() as f64 / 1e6))
}

fn drain_timed(coord: &mut StreamCoordinator, out: &mut Vec<TimedEvent>) -> Result<(), CliError> {
    while let PollResult::Event(ev) = coord.poll_event()? {
        let t_s = (ev.observed_monotonic_ns as f64) * 1e-9;
        out.push(TimedEvent { t_s, event: *ev });
    }
    Ok(())
}

fn quantile_ms(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = (((sorted.len() - 1) as f64) * q).round() as usize;
    (sorted[idx] * 1000.0).round() / 1000.0
}

pub fn transcribe_audio_file(
    candidate_id: &str,
    audio_path: &Path,
    profile: OperatingProfile,
    policy: OutputPolicy,
    packet_ms: u32,
    pace: f64,
    threads: i32,
) -> Result<(String, Vec<TimedEvent>, Receipt), CliError> {
    let (text, timed, receipt, _) = transcribe_audio_file_with_wake(
        candidate_id,
        audio_path,
        profile,
        policy,
        packet_ms,
        pace,
        threads,
        WakeCadence::PerPacket,
    )?;
    Ok((text, timed, receipt))
}

#[allow(clippy::too_many_arguments)]
pub fn transcribe_audio_file_with_wake(
    candidate_id: &str,
    audio_path: &Path,
    profile: OperatingProfile,
    policy: OutputPolicy,
    packet_ms: u32,
    pace: f64,
    threads: i32,
    wake_cadence: WakeCadence,
) -> Result<(String, Vec<TimedEvent>, Receipt, u64), CliError> {
    transcribe_audio_file_with_options(
        candidate_id,
        audio_path,
        profile,
        policy,
        packet_ms,
        pace,
        threads,
        wake_cadence,
        &StreamCommitCliOptions::default(),
    )
}

#[allow(clippy::too_many_arguments)]
pub fn transcribe_audio_file_with_options(
    candidate_id: &str,
    audio_path: &Path,
    profile: OperatingProfile,
    policy: OutputPolicy,
    packet_ms: u32,
    pace: f64,
    threads: i32,
    wake_cadence: WakeCadence,
    commit_opts: &StreamCommitCliOptions,
) -> Result<(String, Vec<TimedEvent>, Receipt, u64), CliError> {
    let pcm = read_audio_16k_mono(audio_path)
        .map_err(|e| CliError::Config(format!("{}: {e}", audio_path.display())))?;
    let clock = StreamClock::real();
    let (caps, cfg, inner) = if commit_opts.enabled {
        let live_cfg = commit_opts.to_live_commit_config()?;
        let (wdir, prec) = crate::suite_engine::resolve_r1_weights_dir(candidate_id, None)
            .ok_or_else(|| {
                CliError::Config(format!(
                    "--commit-pass requires a native ultra-engine candidate (got `{candidate_id}`)"
                ))
            })??;
        let native = NativeEngineBackend::load_with_options(&wdir, prec, threads, true, 1, 1)?;
        let caps = native.capabilities();
        let cfg = StreamConfig {
            stream_id: "cli-stream-0".into(),
            model_config_id: caps.name.clone(),
            policy,
            profile,
            // LiveCommitAsrStream owns bounded PCM retention and idle resets; avoid synthetic
            // coordinator-level `segment_hard_cap` status injections on long silent spans.
            segment_hard_cap_samples: u64::MAX,
            endpoint_hangover_ms: u64::from(live_cfg.hangover_ms),
            ..StreamConfig::default()
        };
        let inner = native.create_live_commit_stream_with_clock(&cfg, &live_cfg, clock.clone())?;
        (caps, cfg, inner)
    } else {
        let backend = instantiate_backend(candidate_id, threads)?;
        let caps = backend.capabilities();
        let cfg = StreamConfig {
            stream_id: "cli-stream-0".into(),
            model_config_id: caps.name.clone(),
            policy,
            profile,
            ..StreamConfig::default()
        };
        let inner = backend.create_stream_with_clock(&cfg, clock.clone())?;
        (caps, cfg, inner)
    };
    let mut sampler = TreeSampler::new(std::process::id() as i32);
    sampler.sample();
    let cpu0 = self_cpu_s()?;
    let mut coord = StreamCoordinator::with_clock(cfg, inner, clock);
    let (mut timed, mut chunk_ms, mut paced_wakeups) = (Vec::new(), Vec::new(), 0u64);

    let replay = match wake_cadence {
        WakeCadence::PerChunk if caps.chunk_samples > 0 => {
            PacedReplay::with_chunk_coalescing(&pcm, packet_ms, caps.chunk_samples, pace)
        }
        _ => PacedReplay::new(&pcm, packet_ms, pace),
    }
    .map_err(|e| CliError::Config(e.to_string()))?;

    for pkt in replay {
        paced_wakeups += 1;
        let t0 = Instant::now();
        let (mut off, mut stalls) = (0usize, 0u32);
        while off < pkt.pcm.len() {
            let rep = coord.push_pcm(pkt.start_sample + off as u64, &pkt.pcm[off..])?;
            drain_timed(&mut coord, &mut timed)?;
            if rep.accepted_samples == 0 {
                stalls += 1;
                if stalls > 10_000 {
                    return Err(CliError::Resource(format!(
                        "stream backpressure stall at sample {}",
                        pkt.start_sample + off as u64
                    )));
                }
                std::thread::sleep(std::time::Duration::from_millis(1));
            } else {
                off += rep.accepted_samples;
                stalls = 0;
            }
        }
        chunk_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    let t0 = Instant::now();
    coord.finish_input()?;
    chunk_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
    drain_timed(&mut coord, &mut timed)?;

    let cpu_used = (self_cpu_s()? - cpu0).max(1e-6);
    sampler.sample();
    let rsum = sampler.summary();
    let m = coord.metrics();
    let committed = coord.ledger().committed_text();
    chunk_ms.sort_by(f64::total_cmp);

    let (mut previews, mut statuses, mut gaps) = (String::new(), Vec::new(), Vec::new());
    for te in &timed {
        match &te.event.body {
            Body::Preview(u) => {
                previews.push_str(&u.snapshot_utf8);
                previews.push('\n');
            }
            Body::Status { code, .. } => statuses.push(code.clone()),
            Body::Gap {
                start_sample: Some(s),
                end_sample: Some(e),
                ..
            } => gaps.push((*s, *e)),
            _ => {}
        }
    }
    let raw: Vec<&TranscriptEvent> = timed.iter().map(|te| &te.event).collect();
    let ev_bytes = serde_json::to_vec(&raw).map_err(|e| CliError::Internal(e.to_string()))?;
    let receipt = Receipt {
        schema: RECEIPT_SCHEMA.into(),
        bundle: caps.name,
        source_commit: git_short_commit(),
        hardware: "H0".into(),
        target: std::env::consts::ARCH.into(),
        configured_threads: threads.max(1) as u32,
        observed_threads: Some(rsum.max_threads.max(1) as u32),
        cpu_only_verified: ultra_stream::sherpa_ffi::verify_cpu_only_process(),
        audio_seconds: (pcm.len() as f64 / 16_000.0 * 1000.0).round() / 1000.0,
        process_tree_cpu_seconds: Some((cpu_used * 1000.0).round() / 1000.0),
        peak_rss_bytes: Some(rsum.peak_rss_bytes.max(1)),
        events: Some(m.events_emitted),
        wakeups: Some(m.wakeups),
        cap_hits: Some(m.cap_hits),
        chunk_ms: ChunkMs {
            p50: Some(quantile_ms(&chunk_ms, 0.50)),
            p95: Some(quantile_ms(&chunk_ms, 0.95)),
            p99: Some(quantile_ms(&chunk_ms, 0.99)),
        },
        status_events: statuses,
        gaps,
        hashes: Hashes {
            events: sha256_hex(&ev_bytes),
            preview: sha256_hex(previews.as_bytes()),
            commit: sha256_hex(committed.as_bytes()),
            final_: sha256_hex(committed.as_bytes()),
        },
        label: HonestyLabel::InCorpus,
        partition: Partition::Dev,
        completion_status: "complete".into(),
    };
    receipt
        .validate_as_evidence()
        .map_err(|e| CliError::Internal(e.to_string()))?;
    Ok((committed, timed, receipt, paced_wakeups))
}

fn create_with_dirs(p: &Path) -> Result<File, CliError> {
    if let Some(parent) = p.parent() {
        std::fs::create_dir_all(parent)?;
    }
    Ok(File::create(p)?)
}

pub(crate) fn write_stream_outputs(
    events_out: Option<PathBuf>,
    receipt_out: Option<PathBuf>,
    timed: &[TimedEvent],
    receipt: &Receipt,
) -> Result<(), CliError> {
    if let Some(ep) = events_out {
        let mut f = create_with_dirs(&ep)?;
        for te in timed {
            let line = serde_json::to_string(te).map_err(|e| CliError::Internal(e.to_string()))?;
            writeln!(f, "{line}")?;
        }
    }
    if let Some(rp) = receipt_out {
        let body =
            serde_json::to_vec_pretty(receipt).map_err(|e| CliError::Internal(e.to_string()))?;
        create_with_dirs(&rp)?.write_all(&body)?;
    }
    Ok(())
}
