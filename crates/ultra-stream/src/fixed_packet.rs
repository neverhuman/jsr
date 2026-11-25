//! Fixed-packet paced drain & cryptographic parity runner (`R10`, `ultra.fixed-packet-paced/v1`).
//!
//! Runs a deterministic per-slot fixed-packet schedule (`duration_s = None`, `paced = true`,
//! zero in-loop `/proc` polling) over [`NativeEngineBackend`] and verifies exact per-stream
//! submitted PCM SHA-256, packet-schedule SHA-256, and transcript SHA-256 parity post-loop.

use std::sync::Arc;
use std::time::Instant;

use serde::{Deserialize, Serialize};
use ultra_core::receipt::{sha256_hex, ChunkMs, Receipt, RECEIPT_SCHEMA};
use ultra_eval::score::ScoreRecord;
use ultra_frontend::npz::AudioUtterance;

use crate::api::StreamError;
use crate::contrast::{
    build_event_hashes_and_chunks, load_selected_audio, score_completed, self_cpu_user_sys_seconds,
    self_max_rss_bytes, WerSummary,
};
use crate::grid_sim::{
    simulate_grid_fixed_slots, sustains_lateness, LatenessQuantiles, SimParams, SustainedSample,
};
use crate::native_engine::NativeEngineBackend;
use crate::native_engine_bank::{engine_status_events, EngineSpAConfig, NativeEngineSlotBank};
use crate::scheduler::RealClock;
use crate::sherpa_ffi::verify_cpu_only_process;

/// Per-stream cryptographic parity digest for fixed-packet paced runs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StreamDigest {
    pub slot: usize,
    pub utterances: usize,
    pub submitted_samples: u64,
    pub submitted_packets: u64,
    pub submitted_pcm_sha256: String,
    pub packet_schedule_sha256: String,
    pub transcript_sha256: String,
}

/// Compute per-slot and aggregate PCM/transcript SHA-256 digests from recorded slot logs post-loop.
pub(crate) fn compute_stream_digests(
    utts: &[AudioUtterance],
    slot_packets: &[Vec<(u32, u32, u32)>],
    slot_hyps: &[Vec<(usize, String)>],
    streams: usize,
) -> (Vec<StreamDigest>, String, String) {
    let mut digests = Vec::with_capacity(streams);
    let mut pcm_buf = Vec::new();
    let mut pkt_buf = Vec::new();
    let mut hyp_buf = Vec::new();
    let mut agg_pcm = String::new();
    let mut all_hyps: Vec<(String, String)> = Vec::new();

    for slot in 0..streams {
        pcm_buf.clear();
        pkt_buf.clear();
        hyp_buf.clear();
        let mut submitted_samples = 0u64;
        let mut submitted_packets = 0u64;
        if let Some(pkts) = slot_packets.get(slot) {
            for &(utt_pos, start, end) in pkts {
                let u = &utts[utt_pos as usize];
                let slice = &u.pcm[start as usize..end as usize];
                submitted_samples += slice.len() as u64;
                submitted_packets += 1;
                let hdr = format!("{}:{start}:{end}:", u.opaque_id());
                pkt_buf.extend_from_slice(hdr.as_bytes());
                for &s in slice {
                    let le = s.to_le_bytes();
                    pcm_buf.extend_from_slice(&le);
                    pkt_buf.extend_from_slice(&le);
                }
                pkt_buf.push(b'\n');
            }
        }
        let mut utterances = 0usize;
        if let Some(hyps) = slot_hyps.get(slot) {
            utterances = hyps.len();
            for (utt_pos, hyp) in hyps {
                let oid = utts[*utt_pos].opaque_id();
                hyp_buf.extend_from_slice(oid.as_bytes());
                hyp_buf.push(b'\t');
                hyp_buf.extend_from_slice(hyp.as_bytes());
                hyp_buf.push(b'\n');
                all_hyps.push((oid, hyp.clone()));
            }
        }
        let submitted_pcm_sha256 = sha256_hex(&pcm_buf);
        let packet_schedule_sha256 = sha256_hex(&pkt_buf);
        let transcript_sha256 = sha256_hex(&hyp_buf);
        agg_pcm.push_str(&format!(
            "{slot}:{submitted_samples}:{submitted_packets}:{submitted_pcm_sha256}:{packet_schedule_sha256}\n"
        ));
        digests.push(StreamDigest {
            slot,
            utterances,
            submitted_samples,
            submitted_packets,
            submitted_pcm_sha256,
            packet_schedule_sha256,
            transcript_sha256,
        });
    }
    all_hyps.sort();
    let mut canon = String::new();
    for (oid, hyp) in &all_hyps {
        canon.push_str(oid);
        canon.push('\t');
        canon.push_str(hyp);
        canon.push('\n');
    }
    (
        digests,
        sha256_hex(agg_pcm.as_bytes()),
        sha256_hex(canon.as_bytes()),
    )
}

/// Output of [`run_fixed_packet_paced_engine`] (`ultra.fixed-packet-paced/v1`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FixedPacketPacedResult {
    pub schema: String,
    pub driver: String,
    pub bundle_id: String,
    pub split: String,
    pub utterances: usize,
    pub streams: usize,
    pub offered_streams: usize,
    pub sustained_streams: usize,
    pub accepted_samples: u64,
    pub processed_samples: u64,
    pub audio_accepted_s: f64,
    pub audio_processed_s: f64,
    pub audio_unfinished_s: f64,
    pub drained: bool,
    pub sustained: bool,
    pub lateness_bound_ms: f64,
    pub accepted_packets: u64,
    pub offered_to_decode_packets: u64,
    pub user_cpu_seconds: f64,
    pub sys_cpu_seconds: f64,
    pub service_cpu_seconds: f64,
    pub wall_seconds: f64,
    pub rtf_service_cpu: f64,
    pub xrt_service_cpu: f64,
    pub peak_rss_bytes: u64,
    pub engine_decode_calls: u64,
    pub paced_wakeups: u64,
    pub decoded_stream_steps: u64,
    pub max_ready_batch: usize,
    pub mean_streams_per_batch: f64,
    pub forced_advances: u64,
    pub model_chunk_ms: u32,
    pub model_right_context_ms: u32,
    pub packet_ms: u32,
    pub grid_ms: u32,
    pub effective_grid_ms: u32,
    pub coalesce_ms: u32,
    pub effective_max_schedule_lookahead_ms: u32,
    pub decode_policy_sha256: String,
    pub weights_sha256: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lm_sha256: Option<String>,
    pub all_streams_pcm_sha256: String,
    pub canonical_transcript_sha256: String,
    pub stream_digests: Vec<StreamDigest>,
    pub chunk_ms: ChunkMs,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lateness_ms: Option<LatenessQuantiles>,
    pub series: Vec<SustainedSample>,
    pub wer: WerSummary,
    pub receipt: Receipt,
    #[serde(skip)]
    pub scored_rows: Vec<ScoreRecord>,
}

/// Execute a fixed-packet paced drain (`duration_s = None`, `paced = true`) over
/// [`NativeEngineBackend`] with deterministic per-slot queues, zero in-loop `/proc` polling,
/// and exact post-loop per-stream submitted PCM & transcript SHA-256 verification.
pub fn run_fixed_packet_paced_engine(
    cfg: &EngineSpAConfig,
) -> Result<FixedPacketPacedResult, StreamError> {
    if cfg.duration_s.is_some() {
        return Err(StreamError::Protocol(
            "run_fixed_packet_paced_engine requires duration_s = None".into(),
        ));
    }
    if !cfg.paced {
        return Err(StreamError::Protocol(
            "run_fixed_packet_paced_engine requires paced = true".into(),
        ));
    }
    let audio_utts = load_selected_audio(
        &cfg.npz_path,
        cfg.offset,
        cfg.limit,
        cfg.selected_indices.as_deref(),
    )?;
    if audio_utts.is_empty() {
        return Err(StreamError::Protocol("no audio utterances loaded".into()));
    }
    let expected_samples: u64 = audio_utts.iter().map(|u| u.pcm.len() as u64).sum();
    let load_cfg = cfg.resolved_load_config();
    let backend = NativeEngineBackend::load_with_config(
        &cfg.weights_dir,
        cfg.precision,
        cfg.threads,
        true,
        &load_cfg,
    )?;
    let bundle_id = backend.bundle_id();
    let mut bank = NativeEngineSlotBank::with_digest_recording(
        Arc::clone(&backend),
        &audio_utts,
        0,
        cfg.streams,
    );

    let (u0, s0) = self_cpu_user_sys_seconds()?;
    let wall_t0 = Instant::now();
    let mut clk = RealClock::new()?;
    let sim = simulate_grid_fixed_slots(
        &mut bank,
        &SimParams {
            streams: cfg.streams,
            decode: cfg.decode,
            max_batch: cfg.max_batch,
            packet_ms: cfg.packet_ms,
            grid_ms: cfg.grid_ms,
            coalesce_ms: cfg.coalesce_ms,
            paced: true,
            duration_s: None,
        },
        &mut clk,
        None,
    )?;
    let (u1, s1) = self_cpu_user_sys_seconds()?;
    let wall_s = wall_t0.elapsed().as_secs_f64().max(1e-6);
    let peak_rss_bytes = self_max_rss_bytes().max(1);
    if let Some(err) = bank.last_error.take() {
        return Err(err);
    }
    if !sim.drained
        || sim.accepted_samples != expected_samples
        || sim.processed_samples != expected_samples
        || sim.completed.len() != audio_utts.len()
    {
        return Err(StreamError::Runtime(format!(
            "fixed-packet drain invariant failed: drained={} accepted={}/{} processed={} utts={}/{}",
            sim.drained,
            sim.accepted_samples,
            expected_samples,
            sim.processed_samples,
            sim.completed.len(),
            audio_utts.len()
        )));
    }

    let forced = bank.forced_advances;
    let (stream_digests, all_streams_pcm_sha256, canonical_transcript_sha256) =
        bank.finalize_stream_digests();
    drop(bank);

    let user_cpu_s = (u1 - u0).max(0.0);
    let sys_cpu_s = (s1 - s0).max(0.0);
    let svc_cpu_s = (user_cpu_s + sys_cpu_s).max(1e-6);
    let audio_denom = sim.audio_accepted_s.max(1e-9);
    let sustained = sim
        .lateness_ms
        .as_ref()
        .is_some_and(|lat| sustains_lateness(lat, sim.drained, cfg.lateness_bound_ms));
    let mean_streams_per_batch = if sim.engine_decode_calls > 0 {
        (sim.decoded_stream_steps as f64) / (sim.engine_decode_calls as f64)
    } else {
        0.0
    };
    let model_chunk_ms = (backend.engine.chunk as u32) * 40;
    let model_right_context_ms = 0u32;
    let effective_max_schedule_lookahead_ms =
        model_chunk_ms + model_right_context_ms + cfg.coalesce_ms;
    let eff = backend.effective_decode_config();
    let decode_policy_sha256 = eff.policy_sha256();
    let weights_sha256 = backend.engine.model.arch.weights_sha256.clone();
    let lm_sha256 = eff.lm_sha256.clone();

    let (wer, scored_rows) = score_completed(
        &cfg.npz_path,
        &cfg.split_name,
        cfg.offset,
        cfg.limit,
        cfg.selected_indices.as_deref(),
        false,
        &sim.completed,
    )?;
    let mut ticks = sim.tick_compute_ms;
    let (hashes, chunk_ms) =
        build_event_hashes_and_chunks(&bundle_id, &audio_utts, &sim.completed, &mut ticks)?;
    let mut status_events = engine_status_events(&backend);
    status_events.extend([
        "fixed_packet_schedule:true".to_string(),
        format!("coalesce_ms:{}", cfg.coalesce_ms),
        format!("model_chunk_ms:{model_chunk_ms}"),
        format!("model_right_context_ms:{model_right_context_ms}"),
        format!("effective_max_schedule_lookahead_ms:{effective_max_schedule_lookahead_ms}"),
        format!("user_cpu_seconds:{user_cpu_s:.6}"),
        format!("sys_cpu_seconds:{sys_cpu_s:.6}"),
        format!("audio_accepted_s:{:.6}", sim.audio_accepted_s),
        format!("audio_processed_s:{:.6}", sim.audio_processed_s),
        format!("audio_unfinished_s:{:.6}", sim.audio_unfinished_s),
        format!("engine_decode_calls:{}", sim.engine_decode_calls),
        format!("paced_wakeups:{}", sim.paced_wakeups),
        format!("decoded_stream_steps:{}", sim.decoded_stream_steps),
        format!("all_streams_pcm_sha256:{all_streams_pcm_sha256}"),
        format!("canonical_transcript_sha256:{canonical_transcript_sha256}"),
    ]);
    let receipt = Receipt {
        schema: RECEIPT_SCHEMA.into(),
        bundle: bundle_id.clone(),
        source_commit: cfg.source_commit.clone(),
        hardware: cfg.hardware_id.clone(),
        target: std::env::consts::ARCH.into(),
        configured_threads: cfg.threads.max(1) as u32,
        observed_threads: Some(cfg.threads.max(1) as u32),
        cpu_only_verified: verify_cpu_only_process(),
        audio_seconds: (sim.audio_accepted_s * 1000.0).round() / 1000.0,
        process_tree_cpu_seconds: Some((svc_cpu_s * 1000.0).round() / 1000.0),
        peak_rss_bytes: Some(peak_rss_bytes),
        wakeups: Some(sim.engine_decode_calls),
        events: Some(sim.completed.len() as u64),
        cap_hits: Some(forced),
        chunk_ms: chunk_ms.clone(),
        status_events,
        gaps: Vec::new(),
        hashes,
        label: cfg.honesty_label,
        partition: cfg.partition,
        completion_status: cfg.completion_status.clone(),
    };

    Ok(FixedPacketPacedResult {
        schema: "ultra.fixed-packet-paced/v1".into(),
        driver: "rust-ultra-engine-fixed-packet-paced-v1".into(),
        bundle_id,
        split: cfg.split_name.clone(),
        utterances: sim.completed.len(),
        streams: cfg.streams,
        offered_streams: cfg.streams,
        sustained_streams: if sustained { cfg.streams } else { 0 },
        accepted_samples: sim.accepted_samples,
        processed_samples: sim.processed_samples,
        audio_accepted_s: sim.audio_accepted_s,
        audio_processed_s: sim.audio_processed_s,
        audio_unfinished_s: sim.audio_unfinished_s,
        drained: sim.drained,
        sustained,
        lateness_bound_ms: cfg.lateness_bound_ms,
        accepted_packets: sim.accepted_packets,
        offered_to_decode_packets: sim.offered_to_decode_packets,
        user_cpu_seconds: (user_cpu_s * 1_000_000.0).round() / 1_000_000.0,
        sys_cpu_seconds: (sys_cpu_s * 1_000_000.0).round() / 1_000_000.0,
        service_cpu_seconds: (svc_cpu_s * 1_000_000.0).round() / 1_000_000.0,
        wall_seconds: (wall_s * 1000.0).round() / 1000.0,
        rtf_service_cpu: svc_cpu_s / audio_denom,
        xrt_service_cpu: audio_denom / svc_cpu_s,
        peak_rss_bytes,
        engine_decode_calls: sim.engine_decode_calls,
        paced_wakeups: sim.paced_wakeups,
        decoded_stream_steps: sim.decoded_stream_steps,
        max_ready_batch: sim.max_ready_batch,
        mean_streams_per_batch,
        forced_advances: forced,
        model_chunk_ms,
        model_right_context_ms,
        packet_ms: cfg.packet_ms,
        grid_ms: cfg.grid_ms,
        effective_grid_ms: sim.effective_grid_ms,
        coalesce_ms: cfg.coalesce_ms,
        effective_max_schedule_lookahead_ms,
        decode_policy_sha256,
        weights_sha256,
        lm_sha256,
        all_streams_pcm_sha256,
        canonical_transcript_sha256,
        stream_digests,
        chunk_ms,
        lateness_ms: sim.lateness_ms,
        series: sim.series,
        wer,
        receipt,
        scored_rows,
    })
}
