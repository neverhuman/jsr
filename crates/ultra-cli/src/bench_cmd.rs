//! `bench` / `sp-a`, `score`, and `compare` CLI commands.

use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use serde::{Deserialize, Serialize};
use ultra_core::receipt::{HonestyLabel, Partition, Receipt};
use ultra_eval::bootstrap::{recommended_replicates, BootstrapConfig};
use ultra_eval::corpus::MacroWeights;
use ultra_eval::score::{markdown, parse_jsonl, scorable_datasets, score_report, ScoreRecord};
use ultra_stream::{
    default_npz_for_split, run_offline_contrast, run_sp_a_ort_r1, run_sp_a_shared_workers,
    run_sp_a_zipformer, ContrastConfig, DecodeMode, R1SpAConfig, SpAConfig,
    DEFAULT_LATENESS_BOUND_MS,
};

use crate::inventory::write_or_print_json;
use crate::runner::{resolve_bundle_or_zipformer, resolve_offline_config, resolve_r1_bundle};
use crate::{git_short_commit, jsr_data_root, write_score_records_jsonl, BenchLease, CliError};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompareVerdictReport {
    pub schema: String,
    pub verdict: String,
    pub alpha: f64,
    pub margin_wer_pp: f64,
    pub observed_delta_wer_pp: Option<f64>,
    pub upper_bound_wer_pp: Option<f64>,
    pub score_report: ultra_eval::score::ScoreReport,
}

pub fn compare_score_files(
    baseline_jsonl: &Path,
    candidate_jsonl: &Path,
    alpha: f64,
    replicates: Option<usize>,
    margin_wer_pp: f64,
) -> Result<CompareVerdictReport, CliError> {
    let base_txt = std::fs::read_to_string(baseline_jsonl)
        .map_err(|e| CliError::Config(format!("{}: {e}", baseline_jsonl.display())))?;
    let cand_txt = std::fs::read_to_string(candidate_jsonl)
        .map_err(|e| CliError::Config(format!("{}: {e}", candidate_jsonl.display())))?;
    let ra = parse_jsonl(&base_txt).map_err(|e| CliError::Config(e.to_string()))?;
    let rb = parse_jsonl(&cand_txt).map_err(|e| CliError::Config(e.to_string()))?;
    let weights =
        MacroWeights::equal(scorable_datasets(&ra)).map_err(|e| CliError::Config(e.to_string()))?;
    let reps = match replicates {
        Some(n) => n,
        None => recommended_replicates(alpha),
    };
    let cfg = BootstrapConfig {
        replicates: reps,
        seed: 20_260_101,
        alpha,
    };
    let rep = score_report(&ra, Some(&rb), &weights, &cfg)
        .map_err(|e| CliError::Config(e.to_string()))?;
    let cmp = rep
        .comparison
        .as_ref()
        .ok_or_else(|| CliError::Internal("missing comparison section".into()))?;
    let obs_pp = Some(cmp.pooled_delta.delta * 100.0);
    let upper_pp = (!cmp.pooled_delta.limited).then_some(cmp.pooled_delta.upper * 100.0);
    let verdict = match upper_pp {
        None => "INCONCLUSIVE",
        Some(u) if u <= margin_wer_pp => "PASS",
        Some(_) => "NO_LIFT",
    };
    Ok(CompareVerdictReport {
        schema: "ultra.compare-verdict/v1".into(),
        verdict: verdict.into(),
        alpha,
        margin_wer_pp,
        observed_delta_wer_pp: obs_pp,
        upper_bound_wer_pp: upper_pp,
        score_report: rep,
    })
}

fn partition_for_split(split: &str) -> Partition {
    if split.starts_with("test") {
        Partition::PublicTest
    } else if split.starts_with("train") {
        Partition::Train
    } else {
        Partition::Dev
    }
}

fn parse_next<T: FromStr>(args: &[String], idx: &mut usize, flag: &str) -> Result<T, CliError> {
    *idx += 1;
    args.get(*idx)
        .and_then(|v| v.parse().ok())
        .ok_or_else(|| CliError::Config(format!("invalid or missing value for {flag}")))
}

fn emit_bench_artifacts(
    line: &str,
    scored_rows: &[ScoreRecord],
    receipt: &Receipt,
    out_jsonl: Option<&Path>,
    scores_out: Option<&Path>,
    receipt_out: Option<&Path>,
) -> Result<(), CliError> {
    println!("{line}");
    if let Some(op) = out_jsonl {
        if let Some(parent) = op.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut f = OpenOptions::new().create(true).append(true).open(op)?;
        writeln!(f, "{line}")?;
    }
    if let Some(sp) = scores_out {
        write_score_records_jsonl(sp, scored_rows)?;
    }
    if let Some(rp) = receipt_out {
        if let Some(parent) = rp.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(
            rp,
            serde_json::to_vec_pretty(receipt).map_err(|e| CliError::Internal(e.to_string()))?,
        )?;
    }
    Ok(())
}

pub(crate) fn cmd_bench(args: &[String]) -> Result<(), CliError> {
    let mut candidate = String::from("zipformer-int8");
    let mut split = String::from("dev-clean");
    let mut npz_override: Option<PathBuf> = None;
    let mut limit: Option<usize> = None;
    let mut offset = 0usize;
    let mut duration_s: Option<f64> = None;
    let mut workers = 1usize;
    let mut streams = 1usize;
    let mut decode = DecodeMode::Batch;
    let mut max_batch = 1024usize;
    let mut precision: Option<String> = None;
    let mut threads = 1i32;
    let mut packet_ms = 20u32;
    let mut grid_ms = 320u32;
    let mut lateness_bound_ms = DEFAULT_LATENESS_BOUND_MS;
    let mut tail_pad_s = 0.8f64;
    let mut paced = false;
    let mut lease_path: Option<PathBuf> = None;
    let mut out_jsonl: Option<PathBuf> = None;
    let mut scores_out: Option<PathBuf> = None;
    let mut receipt_out: Option<PathBuf> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--candidate" => candidate = parse_next(args, &mut i, "--candidate")?,
            "--split" | "--suite" => split = parse_next(args, &mut i, "--split")?,
            "--npz" => npz_override = Some(parse_next(args, &mut i, "--npz")?),
            "--limit" => limit = Some(parse_next(args, &mut i, "--limit")?),
            "--offset" => offset = parse_next(args, &mut i, "--offset")?,
            "--duration-s" => duration_s = Some(parse_next(args, &mut i, "--duration-s")?),
            "--workers" => workers = parse_next(args, &mut i, "--workers")?,
            "--streams" | "--batch-size" => streams = parse_next(args, &mut i, "--streams")?,
            "--decode" => {
                let s: String = parse_next(args, &mut i, "--decode")?;
                decode = match s.as_str() {
                    "single" => DecodeMode::Single,
                    "batch" => DecodeMode::Batch,
                    _ => return Err(CliError::Config("--decode must be single|batch".into())),
                };
            }
            "--mode" => {
                let s: String = parse_next(args, &mut i, "--mode")?;
                match s.as_str() {
                    "paced" => paced = true,
                    "file" => {
                        paced = false;
                        decode = DecodeMode::Single;
                    }
                    "concurrent" => decode = DecodeMode::Batch,
                    _ => {
                        return Err(CliError::Config(
                            "--mode must be paced|file|concurrent".into(),
                        ))
                    }
                }
            }
            "--max-batch" => max_batch = parse_next(args, &mut i, "--max-batch")?,
            "--precision" => precision = Some(parse_next(args, &mut i, "--precision")?),
            "--threads" => threads = parse_next(args, &mut i, "--threads")?,
            "--packet-ms" => packet_ms = parse_next(args, &mut i, "--packet-ms")?,
            "--grid-ms" => grid_ms = parse_next(args, &mut i, "--grid-ms")?,
            "--lateness-bound-ms" => {
                lateness_bound_ms = parse_next(args, &mut i, "--lateness-bound-ms")?
            }
            "--tail-pad-s" => tail_pad_s = parse_next(args, &mut i, "--tail-pad-s")?,
            "--paced" => paced = true,
            "--lease" => lease_path = Some(parse_next(args, &mut i, "--lease")?),
            "--out" => out_jsonl = Some(parse_next(args, &mut i, "--out")?),
            "--scores-out" => scores_out = Some(parse_next(args, &mut i, "--scores-out")?),
            "--receipt-out" => receipt_out = Some(parse_next(args, &mut i, "--receipt-out")?),
            x => return Err(CliError::Config(format!("unknown option `{x}`"))),
        }
        i += 1;
    }

    let _lease = match &lease_path {
        Some(lp) => Some(BenchLease::acquire(lp)?),
        None => None,
    };
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
    let partition = partition_for_split(&split);
    let commit = git_short_commit();

    if let Some(r1_res) = resolve_r1_bundle(&candidate, precision.as_deref()) {
        let (bundle_dir, prec) = r1_res?;
        let res = run_sp_a_ort_r1(&R1SpAConfig {
            bundle_dir,
            precision: prec,
            threads,
            npz_path,
            split_name: split,
            partition,
            honesty_label: HonestyLabel::InCorpus,
            offset,
            limit,
            duration_s,
            streams,
            decode,
            max_batch,
            packet_ms,
            grid_ms,
            lateness_bound_ms,
            paced,
            source_commit: commit,
            hardware_id: "H0".into(),
            completion_status: "exploratory".into(),
        })?;
        let line = serde_json::to_string(&res).map_err(|e| CliError::Internal(e.to_string()))?;
        return emit_bench_artifacts(
            &line,
            &res.scored_rows,
            &res.receipt,
            out_jsonl.as_deref(),
            scores_out.as_deref(),
            receipt_out.as_deref(),
        );
    }

    if let Some(resolved) =
        resolve_bundle_or_zipformer(&candidate, precision.as_deref(), threads, tail_pad_s)
    {
        let (bundle_id, zcfg) = resolved?;
        let sp_cfg = SpAConfig {
            bundle_id: Some(bundle_id),
            npz_path,
            split_name: split,
            partition,
            honesty_label: HonestyLabel::InCorpus,
            offset,
            limit,
            duration_s,
            streams,
            decode,
            max_batch,
            packet_ms,
            grid_ms,
            lateness_bound_ms,
            paced,
            zipformer: zcfg,
            source_commit: commit,
            hardware_id: "H0".into(),
        };
        if workers > 1 {
            let res = run_sp_a_shared_workers(&sp_cfg, workers)?;
            let line =
                serde_json::to_string(&res).map_err(|e| CliError::Internal(e.to_string()))?;
            emit_bench_artifacts(
                &line,
                &res.scored_rows,
                &res.receipt,
                out_jsonl.as_deref(),
                scores_out.as_deref(),
                receipt_out.as_deref(),
            )
        } else {
            let res = run_sp_a_zipformer(&sp_cfg)?;
            let line =
                serde_json::to_string(&res).map_err(|e| CliError::Internal(e.to_string()))?;
            emit_bench_artifacts(
                &line,
                &res.scored_rows,
                &res.receipt,
                out_jsonl.as_deref(),
                scores_out.as_deref(),
                receipt_out.as_deref(),
            )
        }
    } else {
        let models_root = PathBuf::from(jsr_data_root()).join("models");
        let off_cfg = resolve_offline_config(&candidate, &models_root, threads)?;
        let bsz = match decode {
            DecodeMode::Single => 1,
            DecodeMode::Batch => streams.max(1),
        };
        let res = run_offline_contrast(&ContrastConfig {
            npz_path,
            split_name: split,
            partition,
            honesty_label: HonestyLabel::Fair,
            offset,
            limit,
            batch_size: bsz,
            offline: off_cfg,
            source_commit: commit,
            hardware_id: "H0".into(),
        })?;
        let line = serde_json::to_string(&res).map_err(|e| CliError::Internal(e.to_string()))?;
        emit_bench_artifacts(
            &line,
            &res.scored_rows,
            &res.receipt,
            out_jsonl.as_deref(),
            scores_out.as_deref(),
            receipt_out.as_deref(),
        )
    }
}

pub(crate) fn cmd_score(args: &[String]) -> Result<(), CliError> {
    let mut pred: Option<PathBuf> = None;
    let mut out_path: Option<PathBuf> = None;
    let mut md_out: Option<PathBuf> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--predictions" => pred = Some(parse_next(args, &mut i, "--predictions")?),
            "--out" => out_path = Some(parse_next(args, &mut i, "--out")?),
            "--md-out" => md_out = Some(parse_next(args, &mut i, "--md-out")?),
            f if !f.starts_with("--") && pred.is_none() => pred = Some(PathBuf::from(f)),
            x => return Err(CliError::Config(format!("unknown option `{x}`"))),
        }
        i += 1;
    }
    let p = pred.ok_or_else(|| CliError::Config("missing --predictions".into()))?;
    let text = std::fs::read_to_string(&p)
        .map_err(|e| CliError::Config(format!("{}: {e}", p.display())))?;
    let ra = parse_jsonl(&text).map_err(|e| CliError::Config(e.to_string()))?;
    let weights =
        MacroWeights::equal(scorable_datasets(&ra)).map_err(|e| CliError::Config(e.to_string()))?;
    let cfg = BootstrapConfig {
        replicates: recommended_replicates(0.05),
        seed: 20_260_101,
        alpha: 0.05,
    };
    let rep =
        score_report(&ra, None, &weights, &cfg).map_err(|e| CliError::Config(e.to_string()))?;
    let json = serde_json::to_string_pretty(&rep).map_err(|e| CliError::Internal(e.to_string()))?;
    write_or_print_json(out_path.as_deref(), &json)?;
    let md = markdown(&rep);
    match md_out {
        Some(mp) => std::fs::write(mp, md)?,
        None => eprint!("{md}"),
    }
    Ok(())
}

pub(crate) fn cmd_compare(args: &[String]) -> Result<(), CliError> {
    let mut base: Option<PathBuf> = None;
    let mut cand: Option<PathBuf> = None;
    let mut alpha = 0.05f64;
    let mut replicates: Option<usize> = None;
    let mut margin_wer_pp = 0.10f64;
    let mut out_path: Option<PathBuf> = None;
    let mut require_pass = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--baseline" => base = Some(parse_next(args, &mut i, "--baseline")?),
            "--candidate" => cand = Some(parse_next(args, &mut i, "--candidate")?),
            "--alpha" => alpha = parse_next(args, &mut i, "--alpha")?,
            "--replicates" => replicates = Some(parse_next(args, &mut i, "--replicates")?),
            "--margin-wer-pp" => margin_wer_pp = parse_next(args, &mut i, "--margin-wer-pp")?,
            "--out" => out_path = Some(parse_next(args, &mut i, "--out")?),
            "--require-pass" => require_pass = true,
            x => return Err(CliError::Config(format!("unknown option `{x}`"))),
        }
        i += 1;
    }
    let bp = base.ok_or_else(|| CliError::Config("missing --baseline".into()))?;
    let cp = cand.ok_or_else(|| CliError::Config("missing --candidate".into()))?;
    let verdict = compare_score_files(&bp, &cp, alpha, replicates, margin_wer_pp)?;
    let json =
        serde_json::to_string_pretty(&verdict).map_err(|e| CliError::Internal(e.to_string()))?;
    write_or_print_json(out_path.as_deref(), &json)?;
    if require_pass && verdict.verdict != "PASS" {
        return Err(CliError::GateFailed(format!(
            "verdict={} (upper_bound_wer_pp={:?}, margin={})",
            verdict.verdict, verdict.upper_bound_wer_pp, verdict.margin_wer_pp
        )));
    }
    Ok(())
}
