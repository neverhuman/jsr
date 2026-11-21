//! Candidate resolution and single-file / paced-stream CLI execution (`transcribe`, `stream`).

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
use ultra_gp::{R1Precision, R1_EXPORT_SCHEMA, R1_EXPORT_SCHEMA_V1_1};
use ultra_model::verify_bundle_dir;
use ultra_stream::sherpa_ffi::verify_cpu_only_process;
use ultra_stream::{
    AsrStream, Backend, FakeBackend, FakeBackendConfig, NativeEngineBackend, OfflineModelKind,
    OperatingProfile, OrtTransducerBackend, OutputPolicy, PollResult, SherpaLib,
    SherpaOfflineBackend, SherpaOfflineConfig, SherpaZipformerBackend, SherpaZipformerConfig,
    StreamClock, StreamConfig, StreamCoordinator,
};

use crate::{git_short_commit, jsr_data_root, CliError};

fn peek_manifest_schema(bundle_dir: &Path) -> Option<String> {
    let bytes = std::fs::read(bundle_dir.join("manifest.json")).ok()?;
    let val: serde_json::Value = serde_json::from_slice(&bytes).ok()?;
    val.get("schema")
        .and_then(|s| s.as_str())
        .map(str::to_owned)
}

/// Resolve an `ultra.r1-export/v1` bundle directory or alias (`r1-d1`, `r1-d2`, `r1-mr`).
pub fn resolve_r1_bundle(
    candidate_id: &str,
    precision_override: Option<&str>,
) -> Option<Result<(PathBuf, R1Precision), CliError>> {
    let data = PathBuf::from(jsr_data_root());
    let candidate_dir = if candidate_id == "r1-d1" {
        let rev = data.join("bundles/r1-d1-reviewed-20261120-v2");
        if rev.join("manifest.json").exists() {
            Some(rev)
        } else {
            Some(data.join("bundles/r1-d1"))
        }
    } else {
        let direct = Path::new(candidate_id);
        let named = data.join("bundles").join(candidate_id);
        if direct.join("manifest.json").exists() {
            Some(direct.to_path_buf())
        } else if named.join("manifest.json").exists() {
            Some(named)
        } else {
            None
        }
    }?;

    if !matches!(
        peek_manifest_schema(&candidate_dir).as_deref(),
        Some(R1_EXPORT_SCHEMA | R1_EXPORT_SCHEMA_V1_1)
    ) {
        return None;
    }
    let prec_s = precision_override.unwrap_or("int8");
    Some(
        R1Precision::parse(prec_s)
            .map(|p| (candidate_dir, p))
            .map_err(|e| CliError::Config(e.to_string())),
    )
}

/// Resolve either a verified `ultra.bundle/v1` directory (`manifest.json` on disk) or a named
/// Zipformer candidate into `(bundle_id, SherpaZipformerConfig)`.
pub fn resolve_bundle_or_zipformer(
    candidate_id: &str,
    precision_override: Option<&str>,
    threads: i32,
    tail_pad_s: f64,
) -> Option<Result<(String, SherpaZipformerConfig), CliError>> {
    let data = PathBuf::from(jsr_data_root());
    let direct_path = Path::new(candidate_id);
    let named_bundle = data.join("bundles").join(candidate_id);
    let bundle_dir = if direct_path.join("manifest.json").exists() {
        Some(direct_path.to_path_buf())
    } else if named_bundle.join("manifest.json").exists() {
        Some(named_bundle)
    } else {
        None
    };

    if let Some(bdir) = bundle_dir {
        if matches!(
            peek_manifest_schema(&bdir).as_deref(),
            Some(R1_EXPORT_SCHEMA | R1_EXPORT_SCHEMA_V1_1)
        ) {
            return None;
        }
        return Some(
            verify_bundle_dir(&bdir, None)
                .map_err(CliError::from)
                .map(|vb| {
                    let prec = match precision_override {
                        Some(p) => p.to_owned(),
                        None => vb.manifest.precision.clone(),
                    };
                    (
                        vb.manifest.bundle_id,
                        SherpaZipformerConfig {
                            model_dir: vb.root.join("model"),
                            precision: prec,
                            num_threads: threads,
                            tail_pad_s,
                        },
                    )
                }),
        );
    }

    let models = data.join("models");
    let prec = match candidate_id {
        "zipformer-int8"
        | "sherpa-onnx-streaming-zipformer-en"
        | "sherpa-onnx-streaming-zipformer-en-int8"
        | "zipformer_en_int8_320" => Some(precision_override.unwrap_or("int8")),
        "zipformer-fp32" | "sherpa-onnx-streaming-zipformer-en-fp32" | "zipformer_en_fp32_320" => {
            Some(precision_override.unwrap_or("fp32"))
        }
        _ if candidate_id.contains("zipformer") => Some(precision_override.unwrap_or(
            if candidate_id.contains("fp32") {
                "fp32"
            } else {
                "int8"
            },
        )),
        _ => None,
    }?;

    Some(Ok((
        format!("sherpa-onnx-streaming-zipformer-en-2023-06-26-{prec}"),
        SherpaZipformerConfig {
            model_dir: models.join("zipformer-en-2023-06-26"),
            precision: prec.to_owned(),
            num_threads: threads,
            tail_pad_s,
        },
    )))
}

pub fn instantiate_backend(candidate_id: &str, threads: i32) -> Result<Box<dyn Backend>, CliError> {
    instantiate_backend_with_precision(candidate_id, None, threads)
}

pub fn instantiate_backend_with_precision(
    candidate_id: &str,
    precision_override: Option<&str>,
    threads: i32,
) -> Result<Box<dyn Backend>, CliError> {
    if matches!(candidate_id, "fake" | "deterministic-fake-v1") {
        return Ok(Box::new(FakeBackend::new(FakeBackendConfig::default())));
    }
    if let Some(res) = crate::suite_engine::resolve_r1_weights_dir(candidate_id, precision_override)
    {
        let (wdir, prec) = res?;
        let eng = NativeEngineBackend::load(&wdir, prec, threads, true)?;
        return Ok(Box::new(eng));
    }
    if let Some(res) = resolve_r1_bundle(candidate_id, precision_override) {
        let (bdir, prec) = res?;
        let eng = OrtTransducerBackend::load(&bdir, prec, threads, true)?;
        return Ok(Box::new(eng));
    }
    if let Some(res) = resolve_bundle_or_zipformer(candidate_id, precision_override, threads, 0.8) {
        let (_bundle_id, zcfg) = res?;
        let lib = SherpaLib::load_default()?;
        let eng = SherpaZipformerBackend::new(lib, zcfg)?;
        return Ok(Box::new(eng));
    }
    if let Some(p) = precision_override.filter(|&p| p != "int8") {
        return Err(CliError::Config(format!(
            "precision override `{p}` unsupported for offline candidate `{candidate_id}`"
        )));
    }
    let models = PathBuf::from(jsr_data_root()).join("models");
    let cfg = resolve_offline_config(candidate_id, &models, threads)?;
    let lib = SherpaLib::load_default()?;
    let eng = SherpaOfflineBackend::new(lib, cfg)?;
    Ok(Box::new(eng))
}

pub fn resolve_offline_config(
    candidate_id: &str,
    models_root: &Path,
    threads: i32,
) -> Result<SherpaOfflineConfig, CliError> {
    match candidate_id {
        "moonshine-v2" | "moonshine-v2-tiny-en-int8" | "moonshine_v2_tiny_en_int8" => {
            let dir = models_root.join("moonshine-tiny-en-quantized-2026-02-27");
            Ok(SherpaOfflineConfig {
                candidate_id: "moonshine_v2_tiny_en_int8".into(),
                kind: OfflineModelKind::MoonshineV2 {
                    encoder: dir.join("encoder_model.ort"),
                    merged_decoder: dir.join("decoder_model_merged.ort"),
                    tokens: dir.join("tokens.txt"),
                },
                num_threads: threads,
            })
        }
        "moonshine-v2-base-en-int8" | "moonshine_v2_base_en_int8" => {
            let dir = models_root.join("moonshine-base-en-quantized-2026-02-27");
            Ok(SherpaOfflineConfig {
                candidate_id: "moonshine_v2_base_en_int8".into(),
                kind: OfflineModelKind::MoonshineV2 {
                    encoder: dir.join("encoder_model.ort"),
                    merged_decoder: dir.join("decoder_model_merged.ort"),
                    tokens: dir.join("tokens.txt"),
                },
                num_threads: threads,
            })
        }
        "whisper-small.en-int8" | "faster-whisper-small.en-int8" | "whisper_small_en_int8" => {
            let dir = models_root.join("sherpa-onnx-whisper-small.en");
            Ok(SherpaOfflineConfig {
                candidate_id: "whisper_small_en_int8".into(),
                kind: OfflineModelKind::Whisper {
                    encoder: dir.join("small.en-encoder.int8.onnx"),
                    decoder: dir.join("small.en-decoder.int8.onnx"),
                    tokens: dir.join("small.en-tokens.txt"),
                },
                num_threads: threads,
            })
        }
        other => Err(CliError::Unavailable(format!(
            "unknown candidate_id `{other}`"
        ))),
    }
}

fn self_cpu_s() -> Result<f64, CliError> {
    let ru = getrusage(UsageWho::RUSAGE_SELF)
        .map_err(|e| CliError::Internal(format!("getrusage: {e}")))?;
    let u = ru.user_time();
    let s = ru.system_time();
    Ok((u.tv_sec() as f64 + u.tv_usec() as f64 / 1e6)
        + (s.tv_sec() as f64 + s.tv_usec() as f64 / 1e6))
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
    let pcm = read_audio_16k_mono(audio_path)
        .map_err(|e| CliError::Config(format!("{}: {e}", audio_path.display())))?;
    let backend = instantiate_backend(candidate_id, threads)?;
    let model_cfg_id = backend.capabilities().name;
    let stream_cfg = StreamConfig {
        stream_id: "cli-stream-0".into(),
        model_config_id: model_cfg_id.clone(),
        policy,
        profile,
        ..StreamConfig::default()
    };
    let mut sampler = TreeSampler::new(std::process::id() as i32);
    sampler.sample();
    let cpu0 = self_cpu_s()?;
    let shared_clock = StreamClock::real();
    let inner = backend.create_stream_with_clock(&stream_cfg, shared_clock.clone())?;
    let mut coord = StreamCoordinator::with_clock(stream_cfg.clone(), inner, shared_clock);
    let mut timed_events = Vec::new();
    let mut chunk_ms_samples = Vec::new();

    for pkt in
        PacedReplay::new(&pcm, packet_ms, pace).map_err(|e| CliError::Config(e.to_string()))?
    {
        let step_t0 = Instant::now();
        coord.push_pcm(pkt.start_sample, pkt.pcm)?;
        chunk_ms_samples.push(step_t0.elapsed().as_secs_f64() * 1000.0);
        while let PollResult::Event(ev) = coord.poll_event()? {
            let t_s = (ev.observed_monotonic_ns as f64) * 1e-9;
            timed_events.push(TimedEvent { t_s, event: *ev });
        }
    }
    let flush_t0 = Instant::now();
    coord.finish_input()?;
    chunk_ms_samples.push(flush_t0.elapsed().as_secs_f64() * 1000.0);
    while let PollResult::Event(ev) = coord.poll_event()? {
        let t_s = (ev.observed_monotonic_ns as f64) * 1e-9;
        timed_events.push(TimedEvent { t_s, event: *ev });
    }

    let cpu_used = (self_cpu_s()? - cpu0).max(1e-6);
    sampler.sample();
    let rsum = sampler.summary();
    let m = coord.metrics();
    let committed = coord.ledger().committed_text();

    chunk_ms_samples.sort_by(f64::total_cmp);
    let pick = |q: f64| -> f64 {
        if chunk_ms_samples.is_empty() {
            return 0.0;
        }
        let idx = (((chunk_ms_samples.len() - 1) as f64) * q).round() as usize;
        (chunk_ms_samples[idx] * 1000.0).round() / 1000.0
    };

    let mut preview_concat = String::new();
    let mut status_events = Vec::new();
    let mut gaps = Vec::new();
    for te in &timed_events {
        match &te.event.body {
            Body::Preview(u) => {
                preview_concat.push_str(&u.snapshot_utf8);
                preview_concat.push('\n');
            }
            Body::Status { code, .. } => status_events.push(code.clone()),
            Body::Gap {
                start_sample: Some(s),
                end_sample: Some(e),
                ..
            } => gaps.push((*s, *e)),
            _ => {}
        }
    }

    let raw_events: Vec<&TranscriptEvent> = timed_events.iter().map(|te| &te.event).collect();
    let ev_bytes =
        serde_json::to_vec(&raw_events).map_err(|e| CliError::Internal(e.to_string()))?;
    let receipt = Receipt {
        schema: RECEIPT_SCHEMA.into(),
        bundle: model_cfg_id,
        source_commit: git_short_commit(),
        hardware: "H0".into(),
        target: std::env::consts::ARCH.into(),
        configured_threads: threads.max(1) as u32,
        observed_threads: Some(rsum.max_threads.max(1) as u32),
        cpu_only_verified: verify_cpu_only_process(),
        audio_seconds: (pcm.len() as f64 / 16_000.0 * 1000.0).round() / 1000.0,
        process_tree_cpu_seconds: Some((cpu_used * 1000.0).round() / 1000.0),
        peak_rss_bytes: Some(rsum.peak_rss_bytes.max(1)),
        events: Some(m.events_emitted),
        wakeups: Some(m.wakeups),
        cap_hits: Some(m.cap_hits),
        chunk_ms: ChunkMs {
            p50: Some(pick(0.50)),
            p95: Some(pick(0.95)),
            p99: Some(pick(0.99)),
        },
        status_events,
        gaps,
        hashes: Hashes {
            events: sha256_hex(&ev_bytes),
            preview: sha256_hex(preview_concat.as_bytes()),
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
    Ok((committed, timed_events, receipt))
}

pub(crate) fn cmd_transcribe(args: &[String], stream_mode: bool) -> Result<(), CliError> {
    let mut candidate: Option<String> = None;
    let mut audio: Option<PathBuf> = None;
    let mut profile = OperatingProfile::Live;
    let mut policy = OutputPolicy::EditableDictation;
    let mut packet_ms = 20u32;
    let mut pace = if stream_mode { 1.0 } else { 0.0 };
    let mut threads = 1i32;
    let mut events_out: Option<PathBuf> = None;
    let mut receipt_out: Option<PathBuf> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--candidate" => {
                i += 1;
                candidate = args.get(i).cloned();
            }
            "--audio" | "--input" => {
                i += 1;
                audio = args.get(i).map(PathBuf::from);
            }
            "--profile" => {
                i += 1;
                profile = match args.get(i).map(String::as_str) {
                    Some("live") => OperatingProfile::Live,
                    Some("efficient") => OperatingProfile::Efficient,
                    Some("accurate") => OperatingProfile::Accurate,
                    _ => return Err(CliError::Config("invalid --profile".into())),
                };
            }
            "--policy" => {
                i += 1;
                policy = match args.get(i).map(String::as_str) {
                    Some("dictate") => OutputPolicy::EditableDictation,
                    Some("commit") => OutputPolicy::Commit,
                    _ => return Err(CliError::Config("invalid --policy".into())),
                };
            }
            "--packet-ms" => {
                i += 1;
                packet_ms = args
                    .get(i)
                    .and_then(|v| v.parse().ok())
                    .ok_or_else(|| CliError::Config("invalid --packet-ms".into()))?;
            }
            "--pace" => {
                i += 1;
                pace = args
                    .get(i)
                    .and_then(|v| v.parse().ok())
                    .ok_or_else(|| CliError::Config("invalid --pace".into()))?;
            }
            "--threads" => {
                i += 1;
                threads = args
                    .get(i)
                    .and_then(|v| v.parse().ok())
                    .ok_or_else(|| CliError::Config("invalid --threads".into()))?;
            }
            "--events" => {
                i += 1;
                events_out = args.get(i).map(PathBuf::from);
            }
            "--receipt" => {
                i += 1;
                receipt_out = args.get(i).map(PathBuf::from);
            }
            x => return Err(CliError::Config(format!("unknown option `{x}`"))),
        }
        i += 1;
    }

    let cand = candidate.ok_or_else(|| CliError::Config("missing --candidate".into()))?;
    let path = audio.ok_or_else(|| CliError::Config("missing --audio / --input".into()))?;
    let (text, timed, receipt) =
        transcribe_audio_file(&cand, &path, profile, policy, packet_ms, pace, threads)?;

    if let Some(ep) = events_out {
        if let Some(parent) = ep.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut f = File::create(ep)?;
        for te in &timed {
            writeln!(
                f,
                "{}",
                serde_json::to_string(te).map_err(|e| CliError::Internal(e.to_string()))?
            )?;
        }
    }
    if let Some(rp) = receipt_out {
        if let Some(parent) = rp.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(
            rp,
            serde_json::to_vec_pretty(&receipt).map_err(|e| CliError::Internal(e.to_string()))?,
        )?;
    }
    let out = serde_json::json!({
        "candidate": cand,
        "text": text,
        "events": timed.len(),
        "audio_seconds": receipt.audio_seconds,
        "cpu_seconds": receipt.process_tree_cpu_seconds,
        "peak_rss_bytes": receipt.peak_rss_bytes,
    });
    println!(
        "{}",
        serde_json::to_string(&out).map_err(|e| CliError::Internal(e.to_string()))?
    );
    Ok(())
}
