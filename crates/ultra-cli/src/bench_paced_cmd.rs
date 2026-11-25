//! `ultra bench-paced`: fixed-packet paced drain & cryptographic parity harness (`R10`).
//!
//! Runs ONE fixed-packet paced drain (`duration_s = None`, deterministic per-slot utterance
//! partition, zero in-loop `/proc` polling) and writes `paced_report.json`, `paced.receipt.json`,
//! and `paced.scores.jsonl` atomically via `create_new`.

use std::path::PathBuf;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use ultra_core::receipt::HonestyLabel;
use ultra_engine::{LmConfig, NgramLm};
use ultra_stream::{
    default_npz_for_split, run_fixed_packet_paced_engine, DecodeMode, EngineSpAConfig,
    FixedPacketPacedResult, NativeEngineLoadConfig, DEFAULT_LATENESS_BOUND_MS,
};

use crate::bench_cmd::{parse_next, partition_for_split_and_npz};
use crate::suite_engine::resolve_r1_weights_dir;
use crate::suite_manifest::{
    detect_hardware_id, write_json_create_new, write_receipt_create_new, write_scores_create_new,
    SuiteHostSnapshot,
};
use crate::suite_panel::load_panel_for_split;
use crate::{git_short_commit, CliError};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchPacedReport {
    pub executable_path: String,
    pub executable_sha256: String,
    pub npz_path: String,
    pub npz_sha256: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub panel_manifest_sha256: Option<String>,
    pub hostname: String,
    pub cpu_affinity_list: String,
    pub cpu_governor: String,
    pub cpu_boost: String,
    pub cli_args: Vec<String>,
    #[serde(flatten)]
    pub result: FixedPacketPacedResult,
}

pub(crate) fn cmd_bench_paced(args: &[String]) -> Result<(), CliError> {
    let mut candidate: Option<String> = None;
    let mut split = String::from("dev-clean");
    let mut npz_override: Option<PathBuf> = None;
    let mut panel_path: Option<PathBuf> = None;
    let mut offset = 0usize;
    let mut limit: Option<usize> = Some(64);
    let mut streams = 16usize;
    let mut packet_ms = 20u32;
    let mut grid_ms = 20u32;
    let mut coalesce_ms = 0u32;
    let mut max_chunks = 2usize;
    let mut chunk_override: Option<usize> = None;
    let mut beam = 1usize;
    let mut beam_max_symbols = 1usize;
    let mut beam_margin = 0.0f32;
    let mut lm_path: Option<PathBuf> = None;
    let mut lm_weight = 0.0f32;
    let mut lodr_weight = 0.0f32;
    let mut blank_penalty = 0.0f32;
    let mut precision: Option<String> = None;
    let mut threads = 1i32;
    let mut hardware_id_override: Option<String> = None;
    let mut out_dir: Option<PathBuf> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--candidate" => candidate = Some(parse_next(args, &mut i, "--candidate")?),
            "--split" => split = parse_next(args, &mut i, "--split")?,
            "--npz" => npz_override = Some(parse_next(args, &mut i, "--npz")?),
            "--panel" => panel_path = Some(parse_next(args, &mut i, "--panel")?),
            "--offset" => offset = parse_next(args, &mut i, "--offset")?,
            "--limit" => {
                let n: usize = parse_next(args, &mut i, "--limit")?;
                limit = (n > 0).then_some(n);
            }
            "--streams" => streams = parse_next(args, &mut i, "--streams")?,
            "--packet-ms" => packet_ms = parse_next(args, &mut i, "--packet-ms")?,
            "--grid-ms" => grid_ms = parse_next(args, &mut i, "--grid-ms")?,
            "--coalesce-ms" => coalesce_ms = parse_next(args, &mut i, "--coalesce-ms")?,
            "--max-chunks" => max_chunks = parse_next(args, &mut i, "--max-chunks")?,
            "--chunk" => chunk_override = Some(parse_next(args, &mut i, "--chunk")?),
            "--beam" => beam = parse_next(args, &mut i, "--beam")?,
            "--beam-max-symbols" => {
                beam_max_symbols = parse_next(args, &mut i, "--beam-max-symbols")?;
            }
            "--beam-margin" => beam_margin = parse_next(args, &mut i, "--beam-margin")?,
            "--lm" => lm_path = Some(parse_next(args, &mut i, "--lm")?),
            "--lm-weight" => lm_weight = parse_next(args, &mut i, "--lm-weight")?,
            "--lodr-weight" => lodr_weight = parse_next(args, &mut i, "--lodr-weight")?,
            "--blank-penalty" => blank_penalty = parse_next(args, &mut i, "--blank-penalty")?,
            "--precision" => precision = Some(parse_next(args, &mut i, "--precision")?),
            "--threads" => threads = parse_next(args, &mut i, "--threads")?,
            "--hardware-id" => {
                hardware_id_override = Some(parse_next(args, &mut i, "--hardware-id")?);
            }
            "--out" | "--out-dir" => out_dir = Some(parse_next(args, &mut i, "--out-dir")?),
            other => {
                return Err(CliError::Config(format!(
                    "unknown bench-paced option `{other}`"
                )))
            }
        }
        i += 1;
    }

    let raw_cand =
        candidate.ok_or_else(|| CliError::Config("--candidate PATH|ID is required".into()))?;
    let out_dir = out_dir.ok_or_else(|| CliError::Config("--out-dir DIR is required".into()))?;
    if streams == 0 || packet_ms == 0 {
        return Err(CliError::Config(
            "--streams and --packet-ms must be > 0".into(),
        ));
    }
    if coalesce_ms > 0 && grid_ms > coalesce_ms {
        return Err(CliError::Config(format!(
            "--grid-ms ({grid_ms}) must be <= --coalesce-ms ({coalesce_ms}) when coalescing"
        )));
    }
    if !(1..=16).contains(&beam) || !(1..=8).contains(&beam_max_symbols) {
        return Err(CliError::Config(
            "--beam must be 1..=16 and --beam-max-symbols must be 1..=8".into(),
        ));
    }
    if !beam_margin.is_finite() || beam_margin < 0.0 {
        return Err(CliError::Config(
            "--beam-margin must be finite and >= 0".into(),
        ));
    }

    let lm_cfg = LmConfig {
        lm_weight,
        lodr_weight,
        blank_penalty,
    };
    lm_cfg
        .validate()
        .map_err(|e| CliError::Config(e.to_string()))?;
    let loaded_lm = match &lm_path {
        Some(p) => {
            let loaded = NgramLm::load(p)
                .map_err(|e| CliError::Unavailable(format!("--lm {}: {e}", p.display())))?;
            Some((Arc::new(loaded), lm_cfg))
        }
        None => {
            if lm_weight != 0.0 || lodr_weight != 0.0 || blank_penalty != 0.0 {
                return Err(CliError::Config(
                    "--lm-weight, --lodr-weight, and --blank-penalty require --lm".into(),
                ));
            }
            None
        }
    };

    let (weights_dir, prec) =
        resolve_r1_weights_dir(&raw_cand, precision.as_deref()).ok_or_else(|| {
            CliError::Unavailable(format!(
                "bench-paced requires a native ultra-engine weights dir (got `{raw_cand}`)"
            ))
        })??;
    let npz_path = npz_override.unwrap_or_else(|| default_npz_for_split(&split));
    let partition = partition_for_split_and_npz(&split, &npz_path)?;
    if !npz_path.exists() {
        return Err(CliError::Unavailable(format!(
            "split cache `{}` not found",
            npz_path.display()
        )));
    }
    let hardware_id = hardware_id_override.unwrap_or_else(detect_hardware_id);
    let host_snap = SuiteHostSnapshot::capture(&npz_path, None, &hardware_id)?;
    let resolved_panel = match panel_path.as_deref() {
        Some(pp) => Some(load_panel_for_split(
            pp,
            &split,
            &npz_path,
            &host_snap.npz_sha256,
        )?),
        None => None,
    };

    let cfg = EngineSpAConfig {
        weights_dir,
        precision: prec,
        threads,
        npz_path: npz_path.clone(),
        split_name: split,
        partition,
        honesty_label: HonestyLabel::InCorpus,
        offset,
        limit: if resolved_panel.is_some() {
            None
        } else {
            limit
        },
        selected_indices: resolved_panel.as_ref().map(|p| p.indices.clone()),
        duration_s: None,
        streams,
        decode: DecodeMode::Batch,
        max_batch: 1024,
        max_chunks: max_chunks.clamp(1, 16),
        beam,
        decode_config: NativeEngineLoadConfig {
            max_chunks: max_chunks.clamp(1, 16),
            chunk: chunk_override,
            beam,
            beam_max_symbols,
            beam_margin,
            lm: loaded_lm,
            ..NativeEngineLoadConfig::default()
        },
        packet_ms,
        grid_ms,
        coalesce_ms,
        lateness_bound_ms: DEFAULT_LATENESS_BOUND_MS,
        paced: true,
        source_commit: git_short_commit(),
        hardware_id,
        completion_status: "exploratory".into(),
    };

    let res = run_fixed_packet_paced_engine(&cfg)?;
    std::fs::create_dir_all(&out_dir)?;
    write_receipt_create_new(&out_dir.join("paced.receipt.json"), &res.receipt)?;
    write_scores_create_new(&out_dir.join("paced.scores.jsonl"), &res.scored_rows)?;
    let report = BenchPacedReport {
        executable_path: host_snap.executable_path,
        executable_sha256: host_snap.executable_sha256,
        npz_path: npz_path.to_string_lossy().into_owned(),
        npz_sha256: host_snap.npz_sha256,
        panel_manifest_sha256: resolved_panel.map(|p| p.manifest_sha256),
        hostname: host_snap.hostname,
        cpu_affinity_list: host_snap.cpu_affinity_list,
        cpu_governor: host_snap.cpu_governor,
        cpu_boost: host_snap.cpu_boost,
        cli_args: args.to_vec(),
        result: res,
    };
    write_json_create_new(&out_dir.join("paced_report.json"), &report)?;
    println!(
        "{}",
        serde_json::to_string_pretty(&report).map_err(|e| CliError::Internal(e.to_string()))?
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bench_paced_rejects_grid_gt_coalesce_and_missing_args() {
        let s = |v: &[&str]| v.iter().map(|x| (*x).to_owned()).collect::<Vec<_>>();
        let err = cmd_bench_paced(&s(&["--out-dir", "/tmp/x"])).expect_err("missing candidate");
        assert!(matches!(err, CliError::Config(_)));
        let err2 = cmd_bench_paced(&s(&[
            "--candidate",
            "r1-engine",
            "--out-dir",
            "/tmp/x",
            "--grid-ms",
            "100",
            "--coalesce-ms",
            "80",
        ]))
        .expect_err("grid > coalesce");
        assert!(matches!(err2, CliError::Config(ref m) if m.contains("--grid-ms")));
    }
}
