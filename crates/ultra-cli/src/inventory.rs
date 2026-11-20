//! `inventory`, `candidate validate`, `bundle verify`, and `doctor` CLI commands.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use ultra_model::{parse_candidate_docs, HardwareInventory, MachineRecord, HARDWARE_SCHEMA};

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
