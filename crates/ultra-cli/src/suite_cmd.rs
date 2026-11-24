//! `ultra bench-suite` (`B3`, `B4`, `B5`, `C4`, `G3`, `FREEZE.md §3-§4`): unified multi-candidate
//! frontier, §4.4 density ladder, `--panel`, sustained series, ABAB timing, and `--qualified` runner.

use std::path::PathBuf;
use std::sync::Arc;

use ultra_core::receipt::Partition;
use ultra_engine::{LmConfig, NgramLm};
use ultra_stream::{default_npz_for_split, NativeEngineLoadConfig};

use crate::bench_cmd::{parse_next, partition_for_split_and_npz};
use crate::runner::{resolve_bundle_or_zipformer, resolve_r1_bundle};
use crate::suite_abab::{
    resolve_abab_baseline_index, run_abab_timing_blocks, validate_qualified_ccost_arms,
    validate_qualified_ccost_resolved,
};
use crate::suite_engine::{resolve_r1_weights_dir, run_engine_candidate};
use crate::suite_ladder::{run_density_ladder, validate_qualified_ladder_preset, LadderSpec};
pub use crate::suite_latency::{DensityPoint, FrontierReport, FrontierRow};
use crate::suite_looks::{
    git_repo_root, guard_and_log_public_test_reads, record_aborted_public_test_reads,
};
use crate::suite_manifest::{
    detect_hardware_id, parse_cpuset, read_cpu_affinity_list, validate_qualified_preflight,
    write_bytes_create_new, SuiteHostSnapshot,
};
use crate::suite_panel::load_panel_for_split;
use crate::suite_pins::{
    is_nemotron_candidate, resolve_candidate_identity, verify_loaded_candidates_against_lock,
    ResolvedCandidate,
};
use crate::suite_r1_z::{run_r1_candidate, run_zipformer_candidate};
use crate::suite_runners::{run_nemotron_candidate, run_offline_candidate, SuiteCtx};
use crate::{git_short_commit, jsr_data_root, BenchLease, CliError};

pub(crate) fn cmd_bench_suite(args: &[String]) -> Result<(), CliError> {
    let mut candidates_csv =
        String::from("zipformer-int8,r1-d1,moonshine-v2-base-en-int8,moonshine-v2-tiny-en-int8");
    let mut split = String::from("dev-clean");
    let mut npz_override: Option<PathBuf> = None;
    let mut panel_path: Option<PathBuf> = None;
    let mut limit: Option<usize> = Some(100);
    let mut batch_streams = 16usize;
    let mut grid_ms = 320u32;
    let mut max_chunks = 2usize;
    let mut beam = 1usize;
    let mut beam_max_symbols = 1usize;
    let mut beam_margin = 0.0f32;
    let mut chunk_override: Option<usize> = None;
    let mut lm_path: Option<PathBuf> = None;
    let mut lm_weight = 0.0f32;
    let mut lodr_weight = 0.0f32;
    let mut blank_penalty = 0.0f32;
    let mut density_workers = 4usize;
    let mut density_workers_explicit = false;
    let mut full_machine_requested = false;
    let mut density_streams = 12usize;
    let mut density_seconds = 10.0f64;
    let mut density_seconds_explicit = false;
    let mut density_ladder: Option<LadderSpec> = None;
    let mut density_repeats = 3usize;
    let mut ladder_confirm_seconds: Option<f64> = None;
    let mut abab_blocks = 0usize;
    let mut abab_baseline: Option<String> = None;
    let mut latency_limit = 5usize;
    let mut sustained_seconds: Option<f64> = None;
    let mut sustained_streams: Option<usize> = None;
    let mut sustained_workers: Option<usize> = None;
    let mut silence_probe_wav: Option<(PathBuf, bool)> = None;
    let mut observer_cpus: Option<String> = None;
    let mut qualified_requested = false;
    let mut sealed_read_flag: Option<String> = None;
    let mut hardware_id_override: Option<String> = None;
    let mut precision: Option<String> = None;
    let mut threads = 1i32;
    let mut lease_path: Option<PathBuf> = None;
    let mut out_dir: Option<PathBuf> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--candidates" => candidates_csv = parse_next(args, &mut i, "--candidates")?,
            "--split" => split = parse_next(args, &mut i, "--split")?,
            "--npz" => npz_override = Some(parse_next(args, &mut i, "--npz")?),
            "--panel" => panel_path = Some(parse_next(args, &mut i, "--panel")?),
            "--limit" => {
                let n: usize = parse_next(args, &mut i, "--limit")?;
                limit = (n > 0).then_some(n);
            }
            "--batch-streams" => batch_streams = parse_next(args, &mut i, "--batch-streams")?,
            "--grid-ms" => grid_ms = parse_next(args, &mut i, "--grid-ms")?,
            "--max-chunks" => max_chunks = parse_next(args, &mut i, "--max-chunks")?,
            "--beam" => beam = parse_next(args, &mut i, "--beam")?,
            "--beam-max-symbols" => {
                beam_max_symbols = parse_next(args, &mut i, "--beam-max-symbols")?;
            }
            "--beam-margin" => beam_margin = parse_next(args, &mut i, "--beam-margin")?,
            "--chunk" => chunk_override = Some(parse_next(args, &mut i, "--chunk")?),
            "--lm" => lm_path = Some(parse_next(args, &mut i, "--lm")?),
            "--lm-weight" => lm_weight = parse_next(args, &mut i, "--lm-weight")?,
            "--lodr-weight" => lodr_weight = parse_next(args, &mut i, "--lodr-weight")?,
            "--blank-penalty" => blank_penalty = parse_next(args, &mut i, "--blank-penalty")?,
            "--density-workers" => {
                density_workers = parse_next(args, &mut i, "--density-workers")?;
                density_workers_explicit = true;
            }
            "--full-machine" => full_machine_requested = true,
            "--density-streams" => density_streams = parse_next(args, &mut i, "--density-streams")?,
            "--density-seconds" => {
                density_seconds = parse_next(args, &mut i, "--density-seconds")?;
                density_seconds_explicit = true;
            }
            "--density-ladder" => {
                let raw: String = parse_next(args, &mut i, "--density-ladder")?;
                density_ladder = Some(LadderSpec::parse(&raw)?);
            }
            "--density-repeats" => density_repeats = parse_next(args, &mut i, "--density-repeats")?,
            "--ladder-confirm-seconds" => {
                ladder_confirm_seconds =
                    Some(parse_next(args, &mut i, "--ladder-confirm-seconds")?);
            }
            "--abab-blocks" => abab_blocks = parse_next(args, &mut i, "--abab-blocks")?,
            "--abab-baseline" => abab_baseline = Some(parse_next(args, &mut i, "--abab-baseline")?),
            "--latency-limit" => latency_limit = parse_next(args, &mut i, "--latency-limit")?,
            "--sustained-seconds" => {
                sustained_seconds = Some(parse_next(args, &mut i, "--sustained-seconds")?)
            }
            "--sustained-streams" => {
                sustained_streams = Some(parse_next(args, &mut i, "--sustained-streams")?)
            }
            "--sustained-workers" => {
                sustained_workers = Some(parse_next(args, &mut i, "--sustained-workers")?)
            }
            "--silence-probe" => {
                let p =
                    PathBuf::from(jsr_data_root()).join("corpora/probes/silence_noise_10min.wav");
                silence_probe_wav = Some((p, true));
            }
            "--silence-probe-wav" => {
                silence_probe_wav = Some((parse_next(args, &mut i, "--silence-probe-wav")?, false));
            }
            "--observer-cpus" => {
                observer_cpus = Some(parse_next(args, &mut i, "--observer-cpus")?);
            }
            "--precision" => precision = Some(parse_next(args, &mut i, "--precision")?),
            "--threads" => threads = parse_next(args, &mut i, "--threads")?,
            "--qualified" => qualified_requested = true,
            "--sealed-read" => {
                sealed_read_flag = Some(parse_next(args, &mut i, "--sealed-read")?);
            }
            "--hardware-id" => {
                hardware_id_override = Some(parse_next(args, &mut i, "--hardware-id")?);
            }
            "--lease" => lease_path = Some(parse_next(args, &mut i, "--lease")?),
            "--out" | "--out-dir" => out_dir = Some(parse_next(args, &mut i, "--out-dir")?),
            other => {
                return Err(CliError::Config(format!(
                    "unknown bench-suite option `{other}`"
                )))
            }
        }
        i += 1;
    }

    if full_machine_requested && density_workers_explicit {
        return Err(CliError::Config(
            "--full-machine and --density-workers are mutually exclusive (re-review N7)".into(),
        ));
    }
    if full_machine_requested {
        density_workers = 60; // 15 CCX x 4 physical cores (16th CCX = observer)
    }

    let cand_list: Vec<&str> = candidates_csv
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect();
    if cand_list.is_empty() {
        return Err(CliError::Config("--candidates must not be empty".into()));
    }
    // N4 nit: refuse Nemotron + --panel upfront before running any candidate or logging a read.
    if panel_path.is_some() {
        if let Some(&nem) = cand_list.iter().find(|&&c| is_nemotron_candidate(c)) {
            return Err(CliError::Config(format!(
                "Nemotron candidate `{nem}` does not support --panel subset filtering; run it separately on the full split without --panel so frontier.json never mixes utterance sets (re-review N4)"
            )));
        }
    }
    let abab_baseline_idx = if abab_blocks > 0 {
        if qualified_requested {
            Some(validate_qualified_ccost_arms(
                &cand_list,
                abab_blocks,
                abab_baseline.as_deref(),
            )?)
        } else {
            Some(resolve_abab_baseline_index(
                &cand_list,
                abab_baseline.as_deref(),
                false,
            )?)
        }
    } else {
        None
    };

    if !(1..=16).contains(&beam) {
        return Err(CliError::Config(format!(
            "--beam must be in 1..=16 (got {beam})"
        )));
    }
    if !(1..=8).contains(&beam_max_symbols) {
        return Err(CliError::Config(format!(
            "--beam-max-symbols must be in 1..=8 (got {beam_max_symbols})"
        )));
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
    let native_decode_cfg = NativeEngineLoadConfig {
        max_chunks: max_chunks.clamp(1, 16),
        chunk: chunk_override,
        beam,
        beam_max_symbols,
        beam_margin,
        lm: loaded_lm,
        ..NativeEngineLoadConfig::default()
    };
    if !density_seconds.is_finite() || density_seconds < 0.0 {
        return Err(CliError::Config(format!(
            "--density-seconds must be finite and >= 0 (got {density_seconds})"
        )));
    }
    if let Some(cs) = ladder_confirm_seconds {
        if !cs.is_finite() || cs < 0.0 {
            return Err(CliError::Config(format!(
                "--ladder-confirm-seconds must be finite and >= 0 (got {cs})"
            )));
        }
    }
    if let Some(ss) = sustained_seconds {
        if !ss.is_finite() || ss <= 0.0 {
            return Err(CliError::Config(format!(
                "--sustained-seconds must be finite and > 0 (got {ss})"
            )));
        }
    }

    let out_dir = out_dir.ok_or_else(|| CliError::Config("--out-dir DIR is required".into()))?;
    let npz_path = match npz_override {
        Some(p) => p,
        None => default_npz_for_split(&split),
    };
    let partition = partition_for_split_and_npz(&split, &npz_path)?;
    if !npz_path.exists() {
        return Err(CliError::Unavailable(format!(
            "split cache `{}` not found",
            npz_path.display()
        )));
    }

    let build_commit = option_env!("ULTRA_BUILD_GIT_COMMIT").unwrap_or("unknown");
    let build_dirty = option_env!("ULTRA_BUILD_GIT_DIRTY") == Some("true");
    let obs_vec: Option<Vec<usize>> = match observer_cpus.as_deref() {
        Some(spec) => Some(
            parse_cpuset(spec)?
                .into_iter()
                .map(|c| c as usize)
                .collect(),
        ),
        None => None,
    };

    let confirm_s = ladder_confirm_seconds.unwrap_or(if qualified_requested { 300.0 } else { 0.0 });
    let status_tag = if qualified_requested {
        // FREEZE.md §3: lock grid_ms=320, max_chunks=2, threads=1, and per-CCX=4 / full-machine=60.
        if grid_ms != 320 || max_chunks != 2 || threads != 1 {
            return Err(CliError::Config(format!(
                "--qualified locks --grid-ms 320, --max-chunks 2, and --threads 1 per FREEZE.md §3 (got grid_ms={grid_ms}, max_chunks={max_chunks}, threads={threads})"
            )));
        }
        let expected_workers = if full_machine_requested { 60 } else { 4 };
        if density_workers != expected_workers {
            return Err(CliError::Config(format!(
                "--qualified requires exactly {expected_workers} workers ({}; got --density-workers {density_workers}; FREEZE.md §3 / N9)",
                if full_machine_requested { "15 CCX x 4" } else { "per-CCX = 4" }
            )));
        }
        if let Some(sw) = sustained_workers {
            if sw != expected_workers {
                return Err(CliError::Config(format!(
                    "--qualified requires --sustained-workers {expected_workers} (got {sw}; FREEZE.md §3 / N9)"
                )));
            }
        }
        let min_step_s = if density_ladder.is_some() {
            120.0
        } else {
            300.0
        };
        if !density_seconds_explicit {
            density_seconds = min_step_s;
        }
        if let Some(ref spec) = density_ladder {
            // B6(a): --qualified forces the frozen preset, repeats=3, 120s rungs, 300s confirm.
            validate_qualified_ladder_preset(spec, density_repeats, density_seconds, confirm_s)?;
        } else if density_seconds > 0.0 && density_seconds < min_step_s {
            return Err(CliError::Config(format!(
                "--qualified density step requires >= {min_step_s} s per FREEZE.md §3/§4.4 (got {density_seconds})"
            )));
        }
        let svc_spec = read_cpu_affinity_list();
        let _freeze_commit = validate_qualified_preflight(
            &svc_spec,
            observer_cpus.as_deref(),
            lease_path.as_deref(),
            build_commit,
            build_dirty,
            "bench-freeze-v1",
        )?;
        // B1 + S4 + N9: verify exact 1:1 physical cores + L3 packing on ANY timed/sustained/ladder mode.
        if density_seconds > 0.0 || density_ladder.is_some() || sustained_seconds.is_some() {
            let svc_cpus: Vec<usize> = parse_cpuset(&svc_spec)?
                .into_iter()
                .map(|c| c as usize)
                .collect();
            let assigned = ultra_stream::assign_worker_cpus(
                expected_workers,
                &svc_cpus,
                obs_vec.as_deref(),
                true,
            )?;
            ultra_stream::verify_worker_ccx_packing(&assigned, full_machine_requested)?;
        }
        "qualified"
    } else {
        "exploratory"
    };

    // Resolve canonical loaded candidate identities upfront for B4 pins, S6 canonical LOOKS ids
    // (both PUBLIC-TEST and PRIVATE-OOD DomainSealed), and resolved-identity C-cost matrix check.
    let resolved_cands: Option<Vec<ResolvedCandidate>> = if qualified_requested
        || matches!(partition, Partition::PublicTest | Partition::DomainSealed)
    {
        let mut v = Vec::with_capacity(cand_list.len());
        for &c in &cand_list {
            v.push(resolve_candidate_identity(
                c,
                precision.as_deref(),
                threads,
            )?);
        }
        Some(v)
    } else {
        None
    };
    if qualified_requested {
        let rc = resolved_cands.as_deref().unwrap_or(&[]);
        if abab_blocks > 0 {
            validate_qualified_ccost_resolved(rc, abab_blocks)?;
        }
        let repo_root = git_repo_root()?;
        verify_loaded_candidates_against_lock(
            &repo_root.join("docs/ultra/CANDIDATES.lock"),
            &repo_root,
            rc,
        )?;
    }

    let hardware_id = hardware_id_override.unwrap_or_else(detect_hardware_id);

    // S6: enforce freeze-tag ancestry + clean build + canonical-id one-read rule BEFORE opening
    // or hashing sealed NPZ reference bytes, and AFTER upfront CLI/pin/partition checks.
    let canonical_pairs: Vec<(String, String)> = resolved_cands
        .as_ref()
        .map(|rc| {
            rc.iter()
                .map(|c| (c.canonical_id.clone(), c.raw_arg.clone()))
                .collect()
        })
        .unwrap_or_default();
    let logged_read = guard_and_log_public_test_reads(
        partition,
        &split,
        &canonical_pairs,
        "bench-freeze-v1",
        build_commit,
        build_dirty,
        status_tag,
        sealed_read_flag.as_deref(),
    )?;

    let run_res = (|| -> Result<(), CliError> {
        let host_snap =
            SuiteHostSnapshot::capture(&npz_path, observer_cpus.as_deref(), &hardware_id)?;
        let resolved_panel = match panel_path.as_deref() {
            Some(pp) => Some(load_panel_for_split(
                pp,
                &split,
                &npz_path,
                &host_snap.npz_sha256,
            )?),
            None => None,
        };
        std::fs::create_dir_all(&out_dir)?;
        let _lease = match &lease_path {
            Some(lp) => Some(BenchLease::acquire(lp)?),
            None => None,
        };
        let commit = git_short_commit();
        let ctx = SuiteCtx {
            out_dir: &out_dir,
            npz_path: &npz_path,
            split: &split,
            partition,
            limit: if resolved_panel.is_some() {
                None
            } else {
                limit
            },
            selected_indices: resolved_panel.as_ref().map(|p| p.indices.as_slice()),
            panel_manifest_sha256: resolved_panel.as_ref().map(|p| p.manifest_sha256.as_str()),
            batch_streams,
            grid_ms,
            max_chunks: max_chunks.clamp(1, 16),
            beam,
            decode_config: &native_decode_cfg,
            density_workers,
            density_streams,
            density_seconds,
            latency_limit,
            sustained_seconds,
            sustained_streams: sustained_streams.unwrap_or(density_streams),
            sustained_workers: sustained_workers.unwrap_or(density_workers),
            silence_probe_wav: silence_probe_wav.as_ref(),
            observer_cpus: obs_vec.as_deref(),
            precision: precision.as_deref(),
            threads,
            commit: &commit,
            hardware_id: &hardware_id,
            status_tag,
            host_snap: &host_snap,
            cli_args: args,
        };

        let ladder_file = if let Some(spec) = density_ladder {
            let _lad = run_density_ladder(&ctx, &cand_list, spec, density_repeats, confirm_s)?;
            Some(
                out_dir
                    .join("density_ladder.json")
                    .to_string_lossy()
                    .into_owned(),
            )
        } else {
            None
        };
        let abab_file = if let Some(base_idx) = abab_baseline_idx {
            let _abab = run_abab_timing_blocks(&ctx, &cand_list, abab_blocks, base_idx)?;
            Some(
                out_dir
                    .join("abab_timing.json")
                    .to_string_lossy()
                    .into_owned(),
            )
        } else {
            None
        };

        let mut rows = Vec::new();
        for raw_cand in cand_list {
            if is_nemotron_candidate(raw_cand) {
                rows.push(run_nemotron_candidate(&ctx, raw_cand)?);
                continue;
            }
            if let Some(eng_res) = resolve_r1_weights_dir(raw_cand, precision.as_deref()) {
                let (weights_dir, prec) = eng_res?;
                rows.push(run_engine_candidate(&ctx, raw_cand, weights_dir, prec)?);
                continue;
            }
            if let Some(r1_res) = resolve_r1_bundle(raw_cand, precision.as_deref()) {
                let (bundle_dir, prec) = r1_res?;
                rows.push(run_r1_candidate(&ctx, raw_cand, bundle_dir, prec)?);
                continue;
            }
            if let Some(z_res) =
                resolve_bundle_or_zipformer(raw_cand, precision.as_deref(), threads, 0.8)
            {
                let (bundle_id, zcfg) = z_res?;
                rows.push(run_zipformer_candidate(&ctx, raw_cand, bundle_id, zcfg)?);
                continue;
            }
            rows.push(run_offline_candidate(&ctx, raw_cand)?);
        }

        let final_max_chunks = ctx.max_chunks;
        let report = FrontierReport {
            schema: "ultra.frontier-suite/v1".into(),
            split: split.clone(),
            panel_manifest_sha256: resolved_panel.map(|p| p.manifest_sha256),
            grid_ms,
            max_chunks: final_max_chunks,
            source_commit: commit,
            hardware_id: hardware_id.clone(),
            completion_status: status_tag.into(),
            density_ladder_file: ladder_file,
            abab_timing_file: abab_file,
            rows,
        };
        let frontier_path = out_dir.join("frontier.json");
        let json_bytes =
            serde_json::to_vec_pretty(&report).map_err(|e| CliError::Internal(e.to_string()))?;
        write_bytes_create_new(&frontier_path, &json_bytes)?;
        println!("{}", String::from_utf8_lossy(&json_bytes));
        Ok(())
    })();

    if let Err(ref e) = run_res {
        if let Some(ref logged) = logged_read {
            record_aborted_public_test_reads(logged, &e.to_string());
        }
    }
    run_res
}
