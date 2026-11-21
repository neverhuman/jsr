//! `create_new` artifact writers and per-candidate `SuiteJobManifest` (`ultra.suite-job-manifest/v1`)
//! for `ultra bench-suite`.

use std::collections::BTreeSet;
use std::fs::{File, OpenOptions};
use std::io::{ErrorKind, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};
use ultra_core::hash::sha256_reader;
use ultra_core::receipt::Receipt;
use ultra_eval::score::ScoreRecord;

use crate::CliError;

pub const SUITE_JOB_MANIFEST_SCHEMA: &str = "ultra.suite-job-manifest/v1";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuiteJobManifest {
    pub schema: String,
    pub candidate: String,
    pub loaded_identity: String,
    pub driver: String,
    pub cost_scope: String,
    pub build_source_commit: String,
    pub launch_checkout_commit: String,
    pub source_commit: String,
    pub executable_path: String,
    pub executable_sha256: String,
    pub split: String,
    pub npz_path: String,
    pub npz_sha256: String,
    pub precision_requested: Option<String>,
    pub threads: i32,
    pub cpu_affinity_list: String,
    pub observer_cpus: Option<String>,
    pub observer_service_disjoint: Option<bool>,
    pub loadavg_before: [f64; 3],
    pub cli_args: Vec<String>,
}

pub(crate) struct SuiteHostSnapshot {
    pub executable_path: String,
    pub executable_sha256: String,
    pub npz_sha256: String,
    pub cpu_affinity_list: String,
    pub observer_cpus: Option<String>,
    pub observer_service_disjoint: Option<bool>,
}

impl SuiteHostSnapshot {
    pub fn capture(npz_path: &Path, observer_cpus: Option<&str>) -> Result<Self, CliError> {
        let exe = std::env::current_exe()?;
        let executable_sha256 = sha256_file(&exe)?;
        let npz_sha256 = sha256_file(npz_path)?;
        let cpu_affinity_list = read_cpu_affinity_list();
        let observer_service_disjoint = match observer_cpus {
            Some(obs) => {
                validate_observer_cpus_disjoint(&cpu_affinity_list, obs)?;
                Some(true)
            }
            None => None,
        };
        Ok(Self {
            executable_path: exe.to_string_lossy().into_owned(),
            executable_sha256,
            npz_sha256,
            cpu_affinity_list,
            observer_cpus: observer_cpus.map(str::to_owned),
            observer_service_disjoint,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn build_job_manifest(
        &self,
        candidate: &str,
        loaded_identity: &str,
        driver: &str,
        cost_scope: &str,
        source_commit: &str,
        split: &str,
        npz_path: &Path,
        precision_requested: Option<&str>,
        threads: i32,
        loadavg_before: [f64; 3],
        cli_args: &[String],
    ) -> SuiteJobManifest {
        SuiteJobManifest {
            schema: SUITE_JOB_MANIFEST_SCHEMA.into(),
            candidate: candidate.to_owned(),
            loaded_identity: loaded_identity.to_owned(),
            driver: driver.to_owned(),
            cost_scope: cost_scope.to_owned(),
            build_source_commit: option_env!("ULTRA_BUILD_GIT_COMMIT")
                .unwrap_or("unknown")
                .to_owned(),
            launch_checkout_commit: source_commit.to_owned(),
            source_commit: source_commit.to_owned(),
            executable_path: self.executable_path.clone(),
            executable_sha256: self.executable_sha256.clone(),
            split: split.to_owned(),
            npz_path: npz_path.to_string_lossy().into_owned(),
            npz_sha256: self.npz_sha256.clone(),
            precision_requested: precision_requested.map(str::to_owned),
            threads,
            cpu_affinity_list: self.cpu_affinity_list.clone(),
            observer_cpus: self.observer_cpus.clone(),
            observer_service_disjoint: self.observer_service_disjoint,
            loadavg_before,
            cli_args: cli_args.to_vec(),
        }
    }
}

/// Parse Linux CPU list syntax (`"0-3,64-67"` or `"52"`) into logical CPU IDs.
pub fn parse_cpuset(spec: &str) -> Result<BTreeSet<u32>, CliError> {
    let mut out = BTreeSet::new();
    for part in spec.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        if let Some((a, b)) = part.split_once('-') {
            let lo: u32 = a
                .trim()
                .parse()
                .map_err(|_| CliError::Config(format!("invalid CPU range `{part}`")))?;
            let hi: u32 = b
                .trim()
                .parse()
                .map_err(|_| CliError::Config(format!("invalid CPU range `{part}`")))?;
            if lo > hi || hi > 4096 {
                return Err(CliError::Config(format!("invalid CPU range `{part}`")));
            }
            out.extend(lo..=hi);
        } else {
            let c: u32 = part
                .parse()
                .map_err(|_| CliError::Config(format!("invalid CPU id `{part}`")))?;
            if c > 4096 {
                return Err(CliError::Config(format!("invalid CPU id `{part}`")));
            }
            out.insert(c);
        }
    }
    if out.is_empty() {
        return Err(CliError::Config("empty CPU set".into()));
    }
    Ok(out)
}

/// Refuse any logical-CPU, SMT-sibling, or L3-CCX overlap between service and observer cpusets.
pub fn validate_observer_cpus_disjoint(
    service_spec: &str,
    observer_spec: &str,
) -> Result<(), CliError> {
    let svc = parse_cpuset(service_spec)?;
    let obs = parse_cpuset(observer_spec)?;
    let mut expanded_svc = svc.clone();
    for &c in &svc {
        for rel in [
            format!("/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list"),
            format!("/sys/devices/system/cpu/cpu{c}/cache/index3/shared_cpu_list"),
        ] {
            if let Ok(s) = std::fs::read_to_string(rel) {
                if let Ok(sibs) = parse_cpuset(s.trim()) {
                    expanded_svc.extend(sibs);
                }
            }
        }
    }
    let overlap: Vec<u32> = obs.intersection(&expanded_svc).copied().collect();
    if !overlap.is_empty() {
        return Err(CliError::Config(format!(
            "--observer-cpus `{observer_spec}` overlaps service cpuset/SMT/L3 `{service_spec}` on CPUs {overlap:?}"
        )));
    }
    Ok(())
}

pub(crate) fn sha256_file(path: &Path) -> Result<String, CliError> {
    let mut fh = File::open(path)?;
    Ok(sha256_reader(&mut fh)?)
}

fn open_create_new(path: &Path) -> Result<File, CliError> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    match OpenOptions::new().write(true).create_new(true).open(path) {
        Ok(f) => Ok(f),
        Err(e) if e.kind() == ErrorKind::AlreadyExists => Err(CliError::Config(format!(
            "refusing to overwrite existing benchmark artifact `{}` (pass a fresh --out-dir)",
            path.display()
        ))),
        Err(e) => Err(CliError::Internal(e.to_string())),
    }
}

pub(crate) fn write_bytes_create_new(path: &Path, bytes: &[u8]) -> Result<(), CliError> {
    let mut fh = open_create_new(path)?;
    fh.write_all(bytes)?;
    Ok(())
}

pub(crate) fn write_json_create_new<T: Serialize>(path: &Path, val: &T) -> Result<(), CliError> {
    let bytes = serde_json::to_vec_pretty(val).map_err(|e| CliError::Internal(e.to_string()))?;
    write_bytes_create_new(path, &bytes)
}

pub(crate) fn write_receipt_create_new(path: &Path, rec: &Receipt) -> Result<(), CliError> {
    write_json_create_new(path, rec)
}

pub(crate) fn write_scores_create_new(path: &Path, rows: &[ScoreRecord]) -> Result<(), CliError> {
    let mut fh = open_create_new(path)?;
    for r in rows {
        let line = serde_json::to_string(r).map_err(|e| CliError::Internal(e.to_string()))?;
        writeln!(fh, "{line}")?;
    }
    Ok(())
}

pub(crate) fn sanitize_slug(s: &str) -> String {
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

pub(crate) fn current_loadavg_3() -> [f64; 3] {
    let Ok(raw) = std::fs::read_to_string("/proc/loadavg") else {
        return [0.0, 0.0, 0.0];
    };
    let mut it = raw.split_whitespace().filter_map(|p| p.parse::<f64>().ok());
    [
        it.next().unwrap_or(0.0),
        it.next().unwrap_or(0.0),
        it.next().unwrap_or(0.0),
    ]
}

pub(crate) fn read_cpu_affinity_list() -> String {
    let Ok(status) = std::fs::read_to_string("/proc/self/status") else {
        return String::from("unknown");
    };
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("Cpus_allowed_list:") {
            let trimmed = rest.trim();
            if !trimmed.is_empty() {
                return trimmed.to_owned();
            }
        }
    }
    String::from("unknown")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_new_writers_and_observer_cpuset_check_enforce_safety() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let p = tmp.path().join("frontier.json");
        write_bytes_create_new(&p, b"{}").expect("first write ok");
        let err = write_bytes_create_new(&p, b"{}").expect_err("second write must fail");
        assert_eq!(err.exit_code(), crate::EXIT_CONFIG);

        assert!(validate_observer_cpus_disjoint("52", "52").is_err());
        assert!(validate_observer_cpus_disjoint("24-27", "26-29").is_err());
        assert!(validate_observer_cpus_disjoint("52", "48-51").is_ok());
    }
}
