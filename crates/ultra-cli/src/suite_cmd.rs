//! `ultra bench-suite` (`B3`, `B4`, `B5`): unified Rust benchmark harness running quality,
//! single-stream & batched CPU cost, per-CCX / multi-worker live stream density, MFA `CQ/FC`
//! latency, and 1-hour real-bundle sustained probes across R0, R1 (`S` & `S_MR`), Nemotron,
//! Moonshine v2, and Whisper, emitting `ultra.receipt/v1` receipts and `frontier.json`.

use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::time::Duration;

use ultra_adapters::{run_nemotron_batch, NemotronChunkMode, NemotronConfig};
use ultra_core::receipt::{HonestyLabel, Partition, Receipt};
use ultra_stream::{
    default_npz_for_split, run_offline_contrast, run_sp_a_ort_r1, run_sp_a_shared_workers,
    run_sp_a_zipformer, ContrastConfig, DecodeMode, R1SpAConfig, SpAConfig,
    DEFAULT_LATENESS_BOUND_MS,
};

use crate::runner::{
    instantiate_backend, resolve_bundle_or_zipformer, resolve_offline_config, resolve_r1_bundle,
};
use crate::suite_latency::measure_paced_latency;
pub use crate::suite_latency::{DensityPoint, FrontierReport, FrontierRow};
use crate::{git_short_commit, jsr_data_root, write_score_records_jsonl, BenchLease, CliError};

fn parse_arg<T: FromStr>(args: &[String], idx: &mut usize, flag: &str) -> Result<T, CliError> {
    *idx += 1;
    args.get(*idx)
        .and_then(|v| v.parse().ok())
        .ok_or_else(|| CliError::Config(format!("invalid or missing value for {flag}")))
}

fn partition_of(split: &str) -> Partition {
    if split.starts_with("test") {
        Partition::PublicTest
    } else if split.starts_with("train") {
        Partition::Train
    } else {
        Partition::Dev
    }
}

fn sanitize_slug(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

fn write_receipt(path: &Path, rec: &Receipt) -> Result<(), CliError> {
    std::fs::write(
        path,
        serde_json::to_vec_pretty(rec).map_err(|e| CliError::Internal(e.to_string()))?,
    )?;
    Ok(())
}

pub(crate) fn cmd_bench_suite(args: &[String]) -> Result<(), CliError> {
    let mut candidates_csv =
        String::from("zipformer-int8,r1-d1,moonshine-v2-base-en-int8,moonshine-v2-tiny-en-int8");
    let mut split = String::from("dev-clean");
    let mut npz_override: Option<PathBuf> = None;
    let mut limit: Option<usize> = Some(100);
    let mut batch_streams = 16usize;
    let mut density_workers = 1usize;
    let mut density_streams = 12usize;
    let mut density_seconds = 10.0f64;
    let mut latency_limit = 5usize;
    let mut sustained_seconds: Option<f64> = None;
    let mut precision: Option<String> = None;
    let mut threads = 1i32;
    let mut qualified = false;
    let mut lease_path: Option<PathBuf> = None;
    let mut out_dir: Option<PathBuf> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--candidates" => candidates_csv = parse_arg(args, &mut i, "--candidates")?,
            "--split" => split = parse_arg(args, &mut i, "--split")?,
            "--npz" => npz_override = Some(parse_arg(args, &mut i, "--npz")?),
            "--limit" => {
                let n: usize = parse_arg(args, &mut i, "--limit")?;
                limit = (n > 0).then_some(n);
            }
            "--batch-streams" => batch_streams = parse_arg(args, &mut i, "--batch-streams")?,
            "--density-workers" => density_workers = parse_arg(args, &mut i, "--density-workers")?,
            "--density-streams" => density_streams = parse_arg(args, &mut i, "--density-streams")?,
            "--density-seconds" => density_seconds = parse_arg(args, &mut i, "--density-seconds")?,
            "--latency-limit" => latency_limit = parse_arg(args, &mut i, "--latency-limit")?,
            "--sustained-seconds" => {
                sustained_seconds = Some(parse_arg(args, &mut i, "--sustained-seconds")?)
            }
            "--precision" => precision = Some(parse_arg(args, &mut i, "--precision")?),
            "--threads" => threads = parse_arg(args, &mut i, "--threads")?,
            "--qualified" => qualified = true,
            "--lease" => lease_path = Some(parse_arg(args, &mut i, "--lease")?),
            "--out" | "--out-dir" => out_dir = Some(parse_arg(args, &mut i, "--out-dir")?),
            other => {
                return Err(CliError::Config(format!(
                    "unknown bench-suite option `{other}`"
                )))
            }
        }
        i += 1;
    }

    let out_dir = out_dir.ok_or_else(|| CliError::Config("--out-dir DIR is required".into()))?;
    std::fs::create_dir_all(&out_dir)?;
    let _lease = match &lease_path {
        Some(lp) => Some(BenchLease::acquire(lp)?),
        None => None,
    };
    let status_tag = if qualified { "complete" } else { "exploratory" };

    let npz_path = match npz_override {
        Some(p) => p,
        None => default_npz_for_split(&split),
    };
    if !npz_path.exists() {
        return Err(CliError::Unavailable(format!(
            "split cache `{}` not found",
            npz_path.display()
        )));
    }
    let partition = partition_of(&split);
    let commit = git_short_commit();
    let mut rows = Vec::new();

    for raw_cand in candidates_csv
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        let slug = sanitize_slug(raw_cand);
        let scores_path = out_dir.join(format!("{slug}.scores.jsonl"));
        let receipt_path = out_dir.join(format!("{slug}.receipt.json"));

        // 1. Nemotron process-tree adapter (`nemotron-160`, `nemotron-560`).
        if matches!(
            raw_cand,
            "nemotron-160" | "nemotron_en_q8_live_160" | "nemotron-560" | "nemotron_en_q8_live_560"
        ) {
            let mode = if raw_cand.contains("160") {
                NemotronChunkMode::Live160
            } else {
                NemotronChunkMode::Live560
            };
            let mut ncfg = NemotronConfig::default_h0(mode, npz_path.clone(), split.clone());
            ncfg.partition = partition;
            ncfg.limit = limit;
            ncfg.threads = threads.max(1) as u32;
            ncfg.source_commit = commit.clone();
            ncfg.completion_status = status_tag.into();
            ncfg.timeout = Duration::from_secs(3600);
            let single =
                run_nemotron_batch(&ncfg).map_err(|e| CliError::Internal(e.to_string()))?;
            write_score_records_jsonl(&scores_path, &single.scored_rows)?;
            write_receipt(&receipt_path, &single.receipt)?;
            rows.push(FrontierRow {
                candidate: single.candidate_id,
                driver: "rust-nemo-speech-adapter-v1".into(),
                honesty_label: single.receipt.label,
                partition,
                completion_status: single.receipt.completion_status.clone(),
                utterances: single.utterances,
                audio_seconds: single.audio_seconds,
                wer_pct: single.wer_pct.unwrap_or(0.0),
                single_rtf_cpu: single.rtf_cpu,
                single_streams_per_core: single.streams_per_core_at_full_load,
                batch_rtf_cpu: None,
                batch_streams_per_core: None,
                chunk_ms: single.receipt.chunk_ms.clone(),
                peak_rss_bytes: single.receipt.peak_rss_bytes,
                cap_hits: single.receipt.cap_hits,
                evidence_complete: single.receipt.validate_as_evidence().is_ok(),
                density: None,
                latency: None,
                sustained_probe: None,
                scores_file: scores_path.to_string_lossy().into_owned(),
                receipt_file: receipt_path.to_string_lossy().into_owned(),
            });
            continue;
        }

        // 2. R1 `OrtTransducerBackend` (`r1-d1`, `r1-d2`, `r1-mr`, or bundle path).
        if let Some(r1_res) = resolve_r1_bundle(raw_cand, precision.as_deref()) {
            let (bundle_dir, prec) = r1_res?;
            let mk_cfg = |st: usize, dec: DecodeMode, paced: bool, dur: Option<f64>| R1SpAConfig {
                bundle_dir: bundle_dir.clone(),
                precision: prec,
                threads,
                npz_path: npz_path.clone(),
                split_name: split.clone(),
                partition,
                honesty_label: HonestyLabel::InCorpus,
                offset: 0,
                limit: if dur.is_some() { None } else { limit },
                duration_s: dur,
                streams: st,
                decode: dec,
                max_batch: 1024,
                packet_ms: 20,
                grid_ms: 160,
                lateness_bound_ms: DEFAULT_LATENESS_BOUND_MS,
                paced,
                source_commit: commit.clone(),
                hardware_id: "H0".into(),
                completion_status: status_tag.into(),
            };
            let single = run_sp_a_ort_r1(&mk_cfg(1, DecodeMode::Single, false, None))?;
            let batch = run_sp_a_ort_r1(&mk_cfg(batch_streams, DecodeMode::Batch, false, None))?;
            let dens = if density_seconds > 0.0 {
                let d = run_sp_a_ort_r1(&mk_cfg(
                    density_streams,
                    DecodeMode::Batch,
                    true,
                    Some(density_seconds),
                ))?;
                Some(DensityPoint::from_sp_a(&d, density_streams))
            } else {
                None
            };
            let lat = if latency_limit > 0 {
                let backend = instantiate_backend(raw_cand, threads)?;
                Some(measure_paced_latency(
                    &*backend,
                    &npz_path,
                    &split,
                    latency_limit,
                )?)
            } else {
                None
            };
            let sust = if let Some(sec) = sustained_seconds {
                let s = run_sp_a_ort_r1(&mk_cfg(1, DecodeMode::Single, true, Some(sec)))?;
                write_receipt(
                    &out_dir.join(format!("{slug}.sustained.receipt.json")),
                    &s.receipt,
                )?;
                Some(DensityPoint::from_sp_a(&s, 1))
            } else {
                None
            };

            write_score_records_jsonl(&scores_path, &batch.scored_rows)?;
            write_receipt(&receipt_path, &batch.receipt)?;
            rows.push(FrontierRow {
                candidate: raw_cand.to_owned(),
                driver: batch.driver,
                honesty_label: batch.receipt.label,
                partition,
                completion_status: batch.receipt.completion_status.clone(),
                utterances: batch.utterances,
                audio_seconds: batch.audio_seconds,
                wer_pct: batch.wer.wer_pct,
                single_rtf_cpu: single.rtf_cpu,
                single_streams_per_core: single.streams_per_core_at_full_load,
                batch_rtf_cpu: Some(batch.rtf_cpu),
                batch_streams_per_core: Some(batch.streams_per_core_at_full_load),
                chunk_ms: batch.receipt.chunk_ms.clone(),
                peak_rss_bytes: batch.receipt.peak_rss_bytes,
                cap_hits: batch.receipt.cap_hits,
                evidence_complete: batch.receipt.validate_as_evidence().is_ok(),
                density: dens,
                latency: lat,
                sustained_probe: sust,
                scores_file: scores_path.to_string_lossy().into_owned(),
                receipt_file: receipt_path.to_string_lossy().into_owned(),
            });
            continue;
        }

        // 3. R0 Zipformer (`zipformer-int8` or `ultra.bundle/v1`).
        if let Some(z_res) =
            resolve_bundle_or_zipformer(raw_cand, precision.as_deref(), threads, 0.8)
        {
            let (bundle_id, zcfg) = z_res?;
            let mk_z = |st: usize, dec: DecodeMode, paced: bool, dur: Option<f64>| SpAConfig {
                bundle_id: Some(bundle_id.clone()),
                npz_path: npz_path.clone(),
                split_name: split.clone(),
                partition,
                honesty_label: HonestyLabel::InCorpus,
                offset: 0,
                limit: if dur.is_some() { None } else { limit },
                duration_s: dur,
                streams: st,
                decode: dec,
                max_batch: 1024,
                packet_ms: 20,
                grid_ms: 320,
                lateness_bound_ms: DEFAULT_LATENESS_BOUND_MS,
                paced,
                zipformer: zcfg.clone(),
                source_commit: commit.clone(),
                hardware_id: "H0".into(),
            };
            let single = run_sp_a_zipformer(&mk_z(1, DecodeMode::Single, false, None))?;
            let mut batch =
                run_sp_a_zipformer(&mk_z(batch_streams, DecodeMode::Batch, false, None))?;
            batch.receipt.completion_status = status_tag.into();
            let dens = if density_seconds > 0.0 {
                if density_workers > 1 {
                    let f = run_sp_a_shared_workers(
                        &mk_z(
                            density_streams,
                            DecodeMode::Batch,
                            true,
                            Some(density_seconds),
                        ),
                        density_workers,
                    )?;
                    Some(DensityPoint {
                        workers: density_workers,
                        streams_per_worker: density_streams,
                        total_streams: density_workers * density_streams,
                        sustained: f.all_sustained,
                        drained: f.worker_summaries.iter().all(|w| w.drained),
                        lateness_p99_ms: f
                            .worker_summaries
                            .iter()
                            .filter_map(|w| w.lateness_ms.as_ref().map(|l| l.p99))
                            .max_by(f64::total_cmp),
                        peak_rss_bytes: f.receipt.peak_rss_bytes,
                    })
                } else {
                    let d = run_sp_a_zipformer(&mk_z(
                        density_streams,
                        DecodeMode::Batch,
                        true,
                        Some(density_seconds),
                    ))?;
                    Some(DensityPoint::from_sp_a(&d, density_streams))
                }
            } else {
                None
            };
            let lat = if latency_limit > 0 {
                let backend = instantiate_backend(raw_cand, threads)?;
                Some(measure_paced_latency(
                    &*backend,
                    &npz_path,
                    &split,
                    latency_limit,
                )?)
            } else {
                None
            };
            let sust = if let Some(sec) = sustained_seconds {
                let mut s = run_sp_a_zipformer(&mk_z(1, DecodeMode::Single, true, Some(sec)))?;
                s.receipt.completion_status = status_tag.into();
                write_receipt(
                    &out_dir.join(format!("{slug}.sustained.receipt.json")),
                    &s.receipt,
                )?;
                Some(DensityPoint::from_sp_a(&s, 1))
            } else {
                None
            };

            write_score_records_jsonl(&scores_path, &batch.scored_rows)?;
            write_receipt(&receipt_path, &batch.receipt)?;
            rows.push(FrontierRow {
                candidate: raw_cand.to_owned(),
                driver: batch.driver,
                honesty_label: batch.receipt.label,
                partition,
                completion_status: batch.receipt.completion_status.clone(),
                utterances: batch.utterances,
                audio_seconds: batch.audio_seconds,
                wer_pct: batch.wer.wer_pct,
                single_rtf_cpu: single.rtf_cpu,
                single_streams_per_core: single.streams_per_core_at_full_load,
                batch_rtf_cpu: Some(batch.rtf_cpu),
                batch_streams_per_core: Some(batch.streams_per_core_at_full_load),
                chunk_ms: batch.receipt.chunk_ms.clone(),
                peak_rss_bytes: batch.receipt.peak_rss_bytes,
                cap_hits: batch.receipt.cap_hits,
                evidence_complete: batch.receipt.validate_as_evidence().is_ok(),
                density: dens,
                latency: lat,
                sustained_probe: sust,
                scores_file: scores_path.to_string_lossy().into_owned(),
                receipt_file: receipt_path.to_string_lossy().into_owned(),
            });
            continue;
        }

        // 4. Offline contrast recognizers (`Moonshine v2`, `Whisper`).
        let models_root = PathBuf::from(jsr_data_root()).join("models");
        let off_cfg = resolve_offline_config(raw_cand, &models_root, threads)?;
        let mut res = run_offline_contrast(&ContrastConfig {
            npz_path: npz_path.clone(),
            split_name: split.clone(),
            partition,
            honesty_label: HonestyLabel::UnknownOverlap,
            offset: 0,
            limit,
            batch_size: 1,
            offline: off_cfg,
            source_commit: commit.clone(),
            hardware_id: "H0".into(),
        })?;
        res.receipt.completion_status = status_tag.into();
        write_score_records_jsonl(&scores_path, &res.scored_rows)?;
        write_receipt(&receipt_path, &res.receipt)?;
        rows.push(FrontierRow {
            candidate: res.candidate_id,
            driver: "rust-sherpa-offline-c-api".into(),
            honesty_label: res.receipt.label,
            partition,
            completion_status: res.receipt.completion_status.clone(),
            utterances: res.utterances,
            audio_seconds: res.audio_seconds,
            wer_pct: res.wer.wer_pct,
            single_rtf_cpu: res.rtf_cpu,
            single_streams_per_core: res.streams_per_core_at_full_load,
            batch_rtf_cpu: None,
            batch_streams_per_core: None,
            chunk_ms: res.receipt.chunk_ms.clone(),
            peak_rss_bytes: res.receipt.peak_rss_bytes,
            cap_hits: res.receipt.cap_hits,
            evidence_complete: res.receipt.validate_as_evidence().is_ok(),
            density: None,
            latency: None,
            sustained_probe: None,
            scores_file: scores_path.to_string_lossy().into_owned(),
            receipt_file: receipt_path.to_string_lossy().into_owned(),
        });
    }

    let report = FrontierReport {
        schema: "ultra.frontier-suite/v1".into(),
        split,
        source_commit: commit,
        hardware_id: "H0".into(),
        completion_status: status_tag.into(),
        rows,
    };
    let frontier_path = out_dir.join("frontier.json");
    let json_bytes =
        serde_json::to_vec_pretty(&report).map_err(|e| CliError::Internal(e.to_string()))?;
    std::fs::write(&frontier_path, &json_bytes)?;
    println!("{}", String::from_utf8_lossy(&json_bytes));
    Ok(())
}
