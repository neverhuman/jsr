//! `ultra bench-suite` (`B3`, `B4`, `B5`, `C4`): CLI argument parser and multi-candidate
//! frontier runner writing exclusively via `create_new`.

use std::path::PathBuf;

use ultra_stream::default_npz_for_split;

use crate::bench_cmd::{parse_next, partition_for_split};
use crate::runner::{resolve_bundle_or_zipformer, resolve_r1_bundle};
pub use crate::suite_latency::{DensityPoint, FrontierReport, FrontierRow};
use crate::suite_manifest::{
    read_cpu_affinity_list, validate_observer_cpus_disjoint, write_bytes_create_new,
    SuiteHostSnapshot,
};
use crate::suite_runners::{
    run_nemotron_candidate, run_offline_candidate, run_r1_candidate, run_zipformer_candidate,
    SuiteCtx,
};
use crate::{git_short_commit, jsr_data_root, BenchLease, CliError};

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
    let mut silence_probe_wav: Option<(PathBuf, bool)> = None;
    let mut observer_cpus: Option<String> = None;
    let mut qualified_requested = false;
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
            "--limit" => {
                let n: usize = parse_next(args, &mut i, "--limit")?;
                limit = (n > 0).then_some(n);
            }
            "--batch-streams" => batch_streams = parse_next(args, &mut i, "--batch-streams")?,
            "--density-workers" => density_workers = parse_next(args, &mut i, "--density-workers")?,
            "--density-streams" => density_streams = parse_next(args, &mut i, "--density-streams")?,
            "--density-seconds" => density_seconds = parse_next(args, &mut i, "--density-seconds")?,
            "--latency-limit" => latency_limit = parse_next(args, &mut i, "--latency-limit")?,
            "--sustained-seconds" => {
                sustained_seconds = Some(parse_next(args, &mut i, "--sustained-seconds")?)
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

    if qualified_requested {
        let Some(ref obs) = observer_cpus else {
            return Err(CliError::Config(
                "--qualified requires explicit --observer-cpus on a separate CCX from the service cpuset".into(),
            ));
        };
        validate_observer_cpus_disjoint(&read_cpu_affinity_list(), obs)?;
        return Err(CliError::Config(
            "qualifying window requires verified lease, host, freeze, and profile gate; bench-suite emits exploratory receipts only".into(),
        ));
    }
    let out_dir = out_dir.ok_or_else(|| CliError::Config("--out-dir DIR is required".into()))?;
    std::fs::create_dir_all(&out_dir)?;
    let _lease = match &lease_path {
        Some(lp) => Some(BenchLease::acquire(lp)?),
        None => None,
    };
    let status_tag = "exploratory";

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
    let host_snap = SuiteHostSnapshot::capture(&npz_path, observer_cpus.as_deref())?;
    let partition = partition_for_split(&split);
    let commit = git_short_commit();
    let ctx = SuiteCtx {
        out_dir: &out_dir,
        npz_path: &npz_path,
        split: &split,
        partition,
        limit,
        batch_streams,
        density_workers,
        density_streams,
        density_seconds,
        latency_limit,
        sustained_seconds,
        silence_probe_wav: silence_probe_wav.as_ref(),
        precision: precision.as_deref(),
        threads,
        commit: &commit,
        status_tag,
        host_snap: &host_snap,
        cli_args: args,
    };

    let mut rows = Vec::new();
    for raw_cand in candidates_csv
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        if matches!(
            raw_cand,
            "nemotron-160" | "nemotron_en_q8_live_160" | "nemotron-560" | "nemotron_en_q8_live_560"
        ) {
            rows.push(run_nemotron_candidate(&ctx, raw_cand)?);
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
    write_bytes_create_new(&frontier_path, &json_bytes)?;
    println!("{}", String::from_utf8_lossy(&json_bytes));
    Ok(())
}
