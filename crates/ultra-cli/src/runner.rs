//! Candidate resolution and backend instantiation (`transcribe`, `stream`, `bench`).

use std::path::{Path, PathBuf};

use ultra_gp::{R1Precision, R1_EXPORT_SCHEMA, R1_EXPORT_SCHEMA_V1_1};
use ultra_model::verify_bundle_dir;
use ultra_stream::{
    Backend, FakeBackend, FakeBackendConfig, NativeEngineBackend, OfflineModelKind,
    OperatingProfile, OrtTransducerBackend, OutputPolicy, SherpaLib, SherpaOfflineBackend,
    SherpaOfflineConfig, SherpaZipformerBackend, SherpaZipformerConfig,
};

pub use crate::stream_run::{transcribe_audio_file, transcribe_audio_file_with_wake, WakeCadence};
use crate::{jsr_data_root, CliError};

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
    instantiate_backend_with_beam(candidate_id, precision_override, threads, 1)
}

pub fn instantiate_backend_with_beam(
    candidate_id: &str,
    precision_override: Option<&str>,
    threads: i32,
    beam: usize,
) -> Result<Box<dyn Backend>, CliError> {
    if matches!(candidate_id, "fake" | "deterministic-fake-v1") {
        return Ok(Box::new(FakeBackend::new(FakeBackendConfig::default())));
    }
    if let Some(res) = crate::suite_engine::resolve_r1_weights_dir(candidate_id, precision_override)
    {
        let (wdir, prec) = res?;
        let eng = NativeEngineBackend::load_with_options(&wdir, prec, threads, true, 1, beam)?;
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

pub(crate) fn cmd_transcribe(args: &[String], stream_mode: bool) -> Result<(), CliError> {
    let (mut candidate, mut audio, mut wake_override) = (None, None, None);
    let (mut profile, mut policy) = (OperatingProfile::Live, OutputPolicy::EditableDictation);
    let (mut packet_ms, mut pace, mut threads) = (20u32, if stream_mode { 1.0 } else { 0.0 }, 1i32);
    let (mut events_out, mut receipt_out) = (None, None);
    let mut commit_opts = crate::stream_commit::StreamCommitCliOptions::default();
    let mut i = 0;
    while i < args.len() {
        if commit_opts.try_parse_flag(args, &mut i)? {
            i += 1;
            continue;
        }
        let next = |idx: &mut usize, name: &str| -> Result<&str, CliError> {
            *idx += 1;
            args.get(*idx)
                .map(String::as_str)
                .ok_or_else(|| CliError::Config(format!("missing value for {name}")))
        };
        let p_err = |f: &str| CliError::Config(format!("invalid {f}"));
        match args[i].as_str() {
            "--candidate" => candidate = Some(next(&mut i, "--candidate")?.to_owned()),
            "--audio" | "--input" => audio = Some(PathBuf::from(next(&mut i, "--input")?)),
            "--profile" => {
                profile = match next(&mut i, "--profile")? {
                    "live" => OperatingProfile::Live,
                    "efficient" => OperatingProfile::Efficient,
                    "accurate" => OperatingProfile::Accurate,
                    _ => return Err(p_err("--profile")),
                }
            }
            "--policy" => {
                policy = match next(&mut i, "--policy")? {
                    "dictate" => OutputPolicy::EditableDictation,
                    "commit" => OutputPolicy::Commit,
                    _ => return Err(p_err("--policy")),
                }
            }
            "--packet-ms" => {
                packet_ms = next(&mut i, "--packet-ms")?
                    .parse()
                    .map_err(|_| p_err("--packet-ms"))?
            }
            "--pace" => {
                pace = next(&mut i, "--pace")?
                    .parse()
                    .map_err(|_| p_err("--pace"))?
            }
            "--wake-per-chunk" => wake_override = Some(WakeCadence::PerChunk),
            "--wake-per-packet" => wake_override = Some(WakeCadence::PerPacket),
            "--threads" => {
                threads = next(&mut i, "--threads")?
                    .parse()
                    .map_err(|_| p_err("--threads"))?
            }
            "--events" => events_out = Some(PathBuf::from(next(&mut i, "--events")?)),
            "--receipt" => receipt_out = Some(PathBuf::from(next(&mut i, "--receipt")?)),
            x => return Err(CliError::Config(format!("unknown option `{x}`"))),
        }
        i += 1;
    }
    let cand = candidate.ok_or_else(|| CliError::Config("missing --candidate".into()))?;
    let path = audio.ok_or_else(|| CliError::Config("missing --audio / --input".into()))?;
    let cadence = wake_override.unwrap_or(WakeCadence::PerPacket);
    let (text, timed, receipt, paced_wakeups) =
        crate::stream_run::transcribe_audio_file_with_options(
            &cand,
            &path,
            profile,
            policy,
            packet_ms,
            pace,
            threads,
            cadence,
            &commit_opts,
        )?;
    crate::stream_run::write_stream_outputs(events_out, receipt_out, &timed, &receipt)?;
    let out = serde_json::json!({
        "candidate": cand, "text": text, "events": timed.len(),
        "wake_cadence": cadence.as_str(), "paced_wakeups": paced_wakeups,
        "audio_seconds": receipt.audio_seconds, "cpu_seconds": receipt.process_tree_cpu_seconds,
        "peak_rss_bytes": receipt.peak_rss_bytes,
    });
    println!(
        "{}",
        serde_json::to_string(&out).map_err(|e| CliError::Internal(e.to_string()))?
    );
    Ok(())
}
