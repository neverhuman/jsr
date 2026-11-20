//! `inventory`, `candidate validate`, `bundle verify`, and `doctor` CLI commands.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use ultra_core::receipt::sha256_hex;
use ultra_model::{
    parse_candidate_docs, verify_bundle_dir, BundleEntry, BundleFrontend, BundleLimits,
    BundleManifest, HardwareInventory, MachineRecord, BUNDLE_SCHEMA, HARDWARE_SCHEMA,
};

use crate::runner::instantiate_backend;
use crate::CliError;

fn read_trimmed_or(path: &str, default_val: &str) -> String {
    match std::fs::read_to_string(path) {
        Ok(s) if !s.trim().is_empty() => s.trim().to_owned(),
        _ => default_val.to_owned(),
    }
}

pub fn collect_hardware_inventory(machine_id: &str) -> Result<HardwareInventory, CliError> {
    let cpuinfo = std::fs::read_to_string("/proc/cpuinfo")?;
    let meminfo = std::fs::read_to_string("/proc/meminfo")?;

    let mut cpu_model = String::from("unknown-cpu");
    let mut logical = 0u32;
    let mut phys_cores = BTreeSet::new();
    let mut flags_set = BTreeSet::new();
    let mut cur_phys = 0u32;

    for line in cpuinfo.lines() {
        if let Some((_, v)) = line
            .strip_prefix("model name")
            .and_then(|r| r.split_once(':'))
        {
            if cpu_model == "unknown-cpu" {
                cpu_model = v.trim().to_owned();
            }
        } else if line.starts_with("processor") {
            logical += 1;
        } else if let Some((_, v)) = line
            .strip_prefix("physical id")
            .and_then(|r| r.split_once(':'))
        {
            if let Ok(pid) = v.trim().parse::<u32>() {
                cur_phys = pid;
            }
        } else if let Some((_, v)) = line.strip_prefix("core id").and_then(|r| r.split_once(':')) {
            if let Ok(cid) = v.trim().parse::<u32>() {
                phys_cores.insert((cur_phys, cid));
            }
        } else if let Some((_, v)) = line.strip_prefix("flags").and_then(|r| r.split_once(':')) {
            for f in v.split_whitespace() {
                if matches!(
                    f,
                    "avx"
                        | "avx2"
                        | "fma"
                        | "f16c"
                        | "bmi2"
                        | "avx512f"
                        | "avx512bw"
                        | "avx512vbmi"
                        | "avx512vnni"
                ) {
                    flags_set.insert(f.to_owned());
                }
            }
        }
    }

    let cores = if phys_cores.is_empty() {
        logical.max(1)
    } else {
        phys_cores.len() as u32
    };
    let threads_per_core = (logical / cores.max(1)).max(1);

    let mut mem_kb = 1024u64 * 1024;
    for line in meminfo.lines() {
        if let Some(rest) = line.strip_prefix("MemTotal:") {
            if let Some(tok) = rest.split_whitespace().next() {
                if let Ok(v) = tok.parse::<u64>() {
                    mem_kb = v;
                    break;
                }
            }
        }
    }
    let ram_gib = ((mem_kb as f64 / (1024.0 * 1024.0)) * 10.0).round() / 10.0;
    let host = read_trimmed_or("/etc/hostname", "localhost");
    let os_raw = read_trimmed_or("/proc/version", std::env::consts::OS);
    let os = os_raw
        .split_whitespace()
        .take(3)
        .collect::<Vec<_>>()
        .join(" ");
    let governor = read_trimmed_or(
        "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor",
        "unknown",
    );

    Ok(HardwareInventory {
        schema: HARDWARE_SCHEMA.into(),
        machines: vec![MachineRecord {
            id: machine_id.into(),
            host,
            cpu: cpu_model,
            cores,
            threads_per_core,
            l1d: String::new(),
            l2: String::new(),
            l3: String::new(),
            isa: flags_set.into_iter().collect(),
            ram_gib,
            os,
            governor,
            status: "inspected by ultra inventory".into(),
        }],
    })
}

pub(crate) fn write_or_print_json(out_path: Option<&Path>, json: &str) -> Result<(), CliError> {
    match out_path {
        Some(p) => {
            if let Some(parent) = p.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(p, format!("{json}\n"))?;
        }
        None => println!("{json}"),
    }
    Ok(())
}

pub(crate) fn cmd_inventory(args: &[String]) -> Result<(), CliError> {
    let mut id = String::from("H0");
    let mut out_path: Option<PathBuf> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--id" => {
                i += 1;
                id = args
                    .get(i)
                    .cloned()
                    .ok_or_else(|| CliError::Config("--id needs value".into()))?;
            }
            "--out" => {
                i += 1;
                out_path =
                    Some(PathBuf::from(args.get(i).ok_or_else(|| {
                        CliError::Config("--out needs value".into())
                    })?));
            }
            x => return Err(CliError::Config(format!("unknown option `{x}`"))),
        }
        i += 1;
    }
    let inv = collect_hardware_inventory(&id)?;
    let json = serde_json::to_string_pretty(&inv).map_err(|e| CliError::Internal(e.to_string()))?;
    write_or_print_json(out_path.as_deref(), &json)
}

pub(crate) fn cmd_candidate(args: &[String]) -> Result<(), CliError> {
    let Some((action, rest)) = args.split_first() else {
        return Err(CliError::Config(
            "expected `ultra candidate validate --config PATH`".into(),
        ));
    };
    if action != "validate" {
        return Err(CliError::Config(format!(
            "unknown candidate action `{action}`"
        )));
    }
    let mut cfg_path: Option<PathBuf> = None;
    let mut verify_files = false;
    let mut i = 0;
    while i < rest.len() {
        match rest[i].as_str() {
            "--config" => {
                i += 1;
                cfg_path =
                    Some(PathBuf::from(rest.get(i).ok_or_else(|| {
                        CliError::Config("--config needs value".into())
                    })?));
            }
            "--verify-files" => verify_files = true,
            x => return Err(CliError::Config(format!("unknown option `{x}`"))),
        }
        i += 1;
    }
    let p = cfg_path.ok_or_else(|| CliError::Config("missing --config".into()))?;
    let text = std::fs::read_to_string(&p)
        .map_err(|e| CliError::Config(format!("{}: {e}", p.display())))?;
    let docs = parse_candidate_docs(&text)?;
    for d in &docs {
        d.validate(verify_files)?;
    }
    let summary = serde_json::json!({
        "status": "valid",
        "config": p,
        "candidates": docs.iter().map(|d| &d.candidate_id).collect::<Vec<_>>(),
        "files_verified": verify_files,
    });
    println!(
        "{}",
        serde_json::to_string_pretty(&summary).map_err(|e| CliError::Internal(e.to_string()))?
    );
    Ok(())
}

fn pack_zipformer_bundle(
    source_dir: &Path,
    out_dir: &Path,
    precision: &str,
) -> Result<(), CliError> {
    let suffix = match precision {
        "int8" => ".int8.onnx",
        "fp32" => ".onnx",
        other => return Err(CliError::Config(format!("unsupported precision `{other}`"))),
    };
    let stem = "epoch-99-avg-1-chunk-16-left-128";
    let model_files = [
        format!("encoder-{stem}{suffix}"),
        format!("decoder-{stem}{suffix}"),
        format!("joiner-{stem}{suffix}"),
        String::from("tokens.txt"),
    ];
    let dst_model = out_dir.join("model");
    std::fs::create_dir_all(&dst_model)?;
    let mut weight_bytes = 0u64;
    let mut rel_paths = Vec::new();
    for name in &model_files {
        let src = source_dir.join(name);
        if !src.exists() {
            return Err(CliError::Unavailable(format!(
                "missing source file `{}`",
                src.display()
            )));
        }
        let dst = dst_model.join(name);
        std::fs::copy(&src, &dst)?;
        let len = std::fs::metadata(&dst)?.len();
        if name.ends_with(".onnx") {
            weight_bytes += len;
        }
        rel_paths.push(format!("model/{name}"));
    }

    let bundle_id = format!("r0-zipformer-{precision}");
    let card = format!(
        "# Model Card: `{bundle_id}`\n\n\
         - **Architecture**: Streaming Zipformer-Transducer (`chunk-16-left-128`, 320 ms chunk)\n\
         - **Precision**: `{precision}`\n\
         - **Training Corpus**: LibriSpeech 960h only (CC-BY-4.0); LibriSpeech dev/test held out (`in-corpus`, non-contaminated)\n\
         - **Licenses**: Code `Apache-2.0`, Weights `Apache-2.0` (`csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-26`)\n\
         - **Supported Domain**: English 16 kHz mono close-talk / read speech (`H0` x86_64 AVX2/FMA CPU)\n\
         - **Known Limitations**: Trained on read audiobooks; conversational far-field / telephony speech requires domain calibration.\n"
    );
    std::fs::write(out_dir.join("model_card.md"), card)?;
    rel_paths.push(String::from("model_card.md"));

    let provenance = serde_json::json!({
        "schema": "ultra.provenance/v1",
        "bundle_id": bundle_id,
        "upstream_repo": "csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-26",
        "upstream_revision": "4062f0f2dc9f9cbfcdcfbf2ad8fd0ce8b4cc89d7",
        "training_corpora": ["LibriSpeech-960h (CC-BY-4.0)"],
        "eval_overlap": "none (train-clean-100/360 + train-other-500 disjoint from dev/test)",
        "honesty_label": "in-corpus",
        "license_code": "Apache-2.0",
        "license_weights": "Apache-2.0"
    });
    std::fs::write(
        out_dir.join("provenance.json"),
        serde_json::to_vec_pretty(&provenance).map_err(|e| CliError::Internal(e.to_string()))?,
    )?;
    rel_paths.push(String::from("provenance.json"));
    rel_paths.sort();

    let mut entries = Vec::with_capacity(rel_paths.len());
    for rel in rel_paths {
        let bytes = std::fs::read(out_dir.join(&rel))?;
        entries.push(BundleEntry {
            path: rel,
            bytes: bytes.len() as u64,
            sha256: sha256_hex(&bytes),
        });
    }

    let manifest = BundleManifest {
        schema: BUNDLE_SCHEMA.into(),
        bundle_id: bundle_id.clone(),
        candidate_id: format!("sherpa-onnx-streaming-zipformer-en-2023-06-26-{precision}"),
        precision: precision.into(),
        required_isa: vec!["avx2".into(), "fma".into()],
        packed_weight_bytes: weight_bytes,
        expanded_weight_bytes: weight_bytes,
        frontend: BundleFrontend {
            sample_rate_hz: 16_000,
            feature_dim: 80,
            kind: "logmel80".into(),
        },
        limits: BundleLimits {
            max_concurrent_streams: 128,
            max_service_rss_bytes: 1_500_000_000,
            supported_chunk_ms: vec![320],
        },
        normalizer_id: "ultra.en-norm.v1".into(),
        license_code: "Apache-2.0".into(),
        license_weights: "Apache-2.0".into(),
        provenance_summary: "LibriSpeech 960h only (CC-BY-4.0); Apache-2.0 weights".into(),
        entries,
    };
    std::fs::write(
        out_dir.join("manifest.json"),
        serde_json::to_vec_pretty(&manifest).map_err(|e| CliError::Internal(e.to_string()))?,
    )?;
    let verified = verify_bundle_dir(out_dir, None)?;
    println!(
        "{}",
        serde_json::to_string_pretty(&verified).map_err(|e| CliError::Internal(e.to_string()))?
    );
    Ok(())
}

pub(crate) fn cmd_bundle(args: &[String]) -> Result<(), CliError> {
    let Some((action, rest)) = args.split_first() else {
        return Err(CliError::Config(
            "expected `ultra bundle <verify|pack> ...`".into(),
        ));
    };
    if action == "pack" {
        let data = crate::jsr_data_root();
        let mut src = PathBuf::from(&data).join("models/zipformer-en-2023-06-26");
        let mut out: Option<PathBuf> = None;
        let mut prec = String::from("int8");
        let mut i = 0;
        while i < rest.len() {
            match rest[i].as_str() {
                "--source-dir" => {
                    i += 1;
                    src = PathBuf::from(
                        rest.get(i)
                            .ok_or_else(|| CliError::Config("--source-dir needs value".into()))?,
                    );
                }
                "--out" | "--bundle" => {
                    i += 1;
                    out =
                        Some(PathBuf::from(rest.get(i).ok_or_else(|| {
                            CliError::Config("--out needs value".into())
                        })?));
                }
                "--precision" => {
                    i += 1;
                    prec = rest
                        .get(i)
                        .cloned()
                        .ok_or_else(|| CliError::Config("--precision needs value".into()))?;
                }
                x => return Err(CliError::Config(format!("unknown option `{x}`"))),
            }
            i += 1;
        }
        let out_dir = out.ok_or_else(|| CliError::Config("missing --out".into()))?;
        return pack_zipformer_bundle(&src, &out_dir, &prec);
    }
    if action != "verify" {
        return Err(CliError::Config(format!(
            "unknown bundle action `{action}`"
        )));
    }
    let mut bundle_dir: Option<PathBuf> = None;
    let mut hw_path: Option<PathBuf> = None;
    let mut i = 0;
    while i < rest.len() {
        match rest[i].as_str() {
            "--bundle" => {
                i += 1;
                bundle_dir =
                    Some(PathBuf::from(rest.get(i).ok_or_else(|| {
                        CliError::Config("--bundle needs value".into())
                    })?));
            }
            "--hardware" => {
                i += 1;
                hw_path =
                    Some(PathBuf::from(rest.get(i).ok_or_else(|| {
                        CliError::Config("--hardware needs value".into())
                    })?));
            }
            x => return Err(CliError::Config(format!("unknown option `{x}`"))),
        }
        i += 1;
    }
    let bdir = bundle_dir.ok_or_else(|| CliError::Config("missing --bundle".into()))?;
    let hw_inv = match hw_path {
        Some(hp) => Some(HardwareInventory::load(&hp)?),
        None => None,
    };
    let hw_rec = hw_inv.as_ref().map(|inv| inv.machine(None)).transpose()?;
    let verified = verify_bundle_dir(&bdir, hw_rec)?;
    println!(
        "{}",
        serde_json::to_string_pretty(&verified).map_err(|e| CliError::Internal(e.to_string()))?
    );
    Ok(())
}

pub(crate) fn cmd_doctor(args: &[String]) -> Result<(), CliError> {
    let mut candidate = String::from("zipformer-int8");
    let mut hw_path: Option<PathBuf> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--candidate" => {
                i += 1;
                candidate = args
                    .get(i)
                    .cloned()
                    .ok_or_else(|| CliError::Config("--candidate needs value".into()))?;
            }
            "--hardware" => {
                i += 1;
                hw_path =
                    Some(PathBuf::from(args.get(i).ok_or_else(|| {
                        CliError::Config("--hardware needs value".into())
                    })?));
            }
            x => return Err(CliError::Config(format!("unknown option `{x}`"))),
        }
        i += 1;
    }
    let inv = match hw_path {
        Some(p) => HardwareInventory::load(&p)?,
        None => collect_hardware_inventory("H0")?,
    };
    let m = inv.machine(None)?;
    let backend = instantiate_backend(&candidate, 1)?;
    let caps = backend.capabilities();
    if !caps.cpu_only_verified {
        return Err(CliError::GateFailed(
            "GPU library detected in CPU-only process".into(),
        ));
    }
    let out = serde_json::json!({
        "status": "healthy",
        "machine": m.id,
        "cpu": m.cpu,
        "isa": m.isa,
        "candidate": candidate,
        "capabilities": caps,
    });
    println!(
        "{}",
        serde_json::to_string_pretty(&out).map_err(|e| CliError::Internal(e.to_string()))?
    );
    Ok(())
}
