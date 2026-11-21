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
    pub build_tree_dirty: bool,
    pub launch_checkout_commit: String,
    pub source_commit: String,
    pub executable_path: String,
    pub executable_sha256: String,
    pub split: String,
    pub npz_path: String,
    pub npz_sha256: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub panel_manifest_sha256: Option<String>,
    pub grid_ms: u32,
    pub max_chunks: usize,
    pub precision_requested: Option<String>,
    pub threads: i32,
    pub cpu_affinity_list: String,
    pub cpu_governor: String,
    pub cpu_boost: String,
    pub observer_cpus: Option<String>,
    pub observer_service_disjoint: Option<bool>,
    pub observer_topology_verified: Option<bool>,
    pub loadavg_before: [f64; 3],
    pub cli_args: Vec<String>,
}

pub(crate) struct SuiteHostSnapshot {
    pub executable_path: String,
    pub executable_sha256: String,
    pub npz_sha256: String,
    pub cpu_affinity_list: String,
    pub cpu_governor: String,
    pub cpu_boost: String,
    pub observer_cpus: Option<String>,
    pub observer_service_disjoint: Option<bool>,
    pub observer_topology_verified: Option<bool>,
}

fn read_sysfs_trimmed(path: &str) -> Option<String> {
    std::fs::read_to_string(path)
        .ok()
        .map(|s| s.trim().to_owned())
        .filter(|s| !s.is_empty())
}

fn read_cpu_governor_and_boost() -> (String, String) {
    let gov = read_sysfs_trimmed("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
        .unwrap_or_else(|| "unknown".into());
    let boost = if let Some(b) = read_sysfs_trimmed("/sys/devices/system/cpu/cpufreq/boost") {
        format!("cpufreq_boost={b}")
    } else if let Some(nt) = read_sysfs_trimmed("/sys/devices/system/cpu/intel_pstate/no_turbo") {
        format!("intel_no_turbo={nt}")
    } else {
        "unknown".into()
    };
    (gov, boost)
}

impl SuiteHostSnapshot {
    pub fn capture(npz_path: &Path, observer_cpus: Option<&str>) -> Result<Self, CliError> {
        let exe = std::env::current_exe()?;
        let executable_sha256 = sha256_file(&exe)?;
        let npz_sha256 = sha256_file(npz_path)?;
        let cpu_affinity_list = read_cpu_affinity_list();
        let (cpu_governor, cpu_boost) = read_cpu_governor_and_boost();
        let (observer_service_disjoint, observer_topology_verified) = match observer_cpus {
            Some(obs) => {
                let topo_ok = validate_observer_cpus_disjoint(&cpu_affinity_list, obs, false)?;
                (Some(true), Some(topo_ok))
            }
            None => (None, None),
        };
        Ok(Self {
            executable_path: exe.to_string_lossy().into_owned(),
            executable_sha256,
            npz_sha256,
            cpu_affinity_list,
            cpu_governor,
            cpu_boost,
            observer_cpus: observer_cpus.map(str::to_owned),
            observer_service_disjoint,
            observer_topology_verified,
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
        panel_manifest_sha256: Option<&str>,
        grid_ms: u32,
        max_chunks: usize,
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
            build_tree_dirty: option_env!("ULTRA_BUILD_GIT_DIRTY") == Some("true"),
            launch_checkout_commit: source_commit.to_owned(),
            source_commit: source_commit.to_owned(),
            executable_path: self.executable_path.clone(),
            executable_sha256: self.executable_sha256.clone(),
            split: split.to_owned(),
            npz_path: npz_path.to_string_lossy().into_owned(),
            npz_sha256: self.npz_sha256.clone(),
            panel_manifest_sha256: panel_manifest_sha256.map(str::to_owned),
            grid_ms,
            max_chunks,
            precision_requested: precision_requested.map(str::to_owned),
            threads,
            cpu_affinity_list: self.cpu_affinity_list.clone(),
            cpu_governor: self.cpu_governor.clone(),
            cpu_boost: self.cpu_boost.clone(),
            observer_cpus: self.observer_cpus.clone(),
            observer_service_disjoint: self.observer_service_disjoint,
            observer_topology_verified: self.observer_topology_verified,
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
/// When `fail_closed_topology` is `true`, unreadable sysfs topology/L3 files return `Err`.
pub fn validate_observer_cpus_disjoint(
    service_spec: &str,
    observer_spec: &str,
    fail_closed_topology: bool,
) -> Result<bool, CliError> {
    let svc = parse_cpuset(service_spec)?;
    let obs = parse_cpuset(observer_spec)?;
    let mut expanded_svc = svc.clone();
    let mut all_topo_ok = true;
    for &c in &svc {
        for rel in [
            format!("/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list"),
            format!("/sys/devices/system/cpu/cpu{c}/cache/index3/shared_cpu_list"),
        ] {
            match std::fs::read_to_string(&rel) {
                Ok(s) => expanded_svc.extend(parse_cpuset(s.trim())?),
                Err(_) if fail_closed_topology => {
                    return Err(CliError::Config(format!(
                        "missing CPU/L3 topology sysfs `{rel}` required for qualifying observer isolation"
                    )))
                }
                Err(_) => all_topo_ok = false,
            }
        }
    }
    let overlap: Vec<u32> = obs.intersection(&expanded_svc).copied().collect();
    if !overlap.is_empty() {
        return Err(CliError::Config(format!(
            "--observer-cpus `{observer_spec}` overlaps service cpuset/SMT/L3 `{service_spec}` on CPUs {overlap:?}"
        )));
    }
    Ok(all_topo_ok)
}

/// Verify that `tag` exists in Git and its target commit is an ancestor of the binary's baked-in
/// `build_commit` (NOT run-time HEAD alone; re-review `B4`/`N3`), and verify that `build_commit`
/// matches the run-time checkout HEAD.
pub fn verify_freeze_tag_ancestor(tag: &str, build_commit: &str) -> Result<String, CliError> {
    let rev = std::process::Command::new("git")
        .args([
            "rev-parse",
            "--verify",
            &format!("refs/tags/{tag}^{{commit}}"),
        ])
        .output()
        .map_err(|e| {
            CliError::Config(format!("git rev-parse for freeze tag `{tag}` failed: {e}"))
        })?;
    if !rev.status.success() {
        return Err(CliError::Config(format!(
            "--qualified requires freeze tag `{tag}` to exist in git"
        )));
    }
    let tag_commit = String::from_utf8_lossy(&rev.stdout).trim().to_owned();
    let head = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_owned())
        .unwrap_or_default();
    if !head.is_empty() && !head.starts_with(build_commit) && !build_commit.starts_with(&head) {
        return Err(CliError::Config(format!(
            "--qualified binary build_commit `{build_commit}` does not match checkout HEAD `{head}` (B4)"
        )));
    }
    let anc = std::process::Command::new("git")
        .args(["merge-base", "--is-ancestor", &tag_commit, build_commit])
        .status()
        .map_err(|e| CliError::Config(format!("git merge-base check failed: {e}")))?;
    if !anc.success() {
        return Err(CliError::Config(format!(
            "--qualified requires freeze tag `{tag}` ({tag_commit}) to be an ancestor of binary build_commit `{build_commit}` (B4)"
        )));
    }
    Ok(tag_commit)
}

/// Verify pinned manifest SHA-256 entries in `docs/ultra/CANDIDATES.lock` (`B4`).
pub fn verify_candidates_lock_pins(lock_path: &Path) -> Result<usize, CliError> {
    let raw = std::fs::read_to_string(lock_path).map_err(|e| {
        CliError::Config(format!(
            "--qualified requires `{}` to verify pinned candidate identities: {e}",
            lock_path.display()
        ))
    })?;
    let mut cur_manifest: Option<String> = None;
    let mut verified = 0usize;
    for line in raw.lines() {
        let t = line.trim();
        if let Some(m) = t.strip_prefix("manifest:") {
            cur_manifest = Some(m.split('#').next().unwrap_or("").trim().to_owned());
        } else if let Some(s) = t.strip_prefix("manifest_sha256:") {
            let want = s.split('#').next().unwrap_or("").trim();
            if let Some(mp) = cur_manifest.take() {
                if !mp.is_empty() && want.len() == 64 {
                    let got = sha256_file(Path::new(&mp))?;
                    if !got.eq_ignore_ascii_case(want) {
                        return Err(CliError::Config(format!(
                            "CANDIDATES.lock pin mismatch for `{mp}`: expected {want}, got {got}"
                        )));
                    }
                    verified += 1;
                }
            }
        }
    }
    if verified == 0 {
        return Err(CliError::Config(format!(
            "no pinned manifest_sha256 entries verified in `{}`",
            lock_path.display()
        )));
    }
    Ok(verified)
}

/// Fail-closed pre-flight validation for `--qualified` runs (`PLAN_TO_V1` G6 / `FREEZE.md` §4):
/// requires disjoint CCX observer (sysfs verified), `BENCH_LEASE.host`, clean build provenance,
/// `CANDIDATES.lock` pin integrity, and `bench-freeze-v1` tag as an ancestor of `build_commit`.
pub fn validate_qualified_preflight(
    service_cpus: &str,
    observer_cpus: Option<&str>,
    lease_path: Option<&Path>,
    build_commit: &str,
    build_dirty: bool,
    freeze_tag: &str,
) -> Result<String, CliError> {
    let Some(obs) = observer_cpus else {
        return Err(CliError::Config(
            "--qualified requires explicit --observer-cpus on a separate CCX from the service cpuset"
                .into(),
        ));
    };
    validate_observer_cpus_disjoint(service_cpus, obs, true)?;
    let Some(lp) = lease_path else {
        return Err(CliError::Config(
            "--qualified requires --lease .../BENCH_LEASE.host".into(),
        ));
    };
    if lp.file_name().and_then(|n| n.to_str()) != Some("BENCH_LEASE.host") {
        return Err(CliError::Config(format!(
            "--qualified requires a host lease named `BENCH_LEASE.host` (got `{}`)",
            lp.display()
        )));
    }
    if build_dirty || build_commit == "unknown" || build_commit.is_empty() {
        return Err(CliError::Config(format!(
            "--qualified requires a clean committed release build (commit={build_commit}, dirty={build_dirty})"
        )));
    }
    for (k, _) in std::env::vars() {
        if k.starts_with("ULTRA_ENGINE_") {
            return Err(CliError::Config(format!(
                "--qualified refuses unrecorded engine environment override `{k}` (FREEZE.md §3)"
            )));
        }
    }
    verify_candidates_lock_pins(Path::new("docs/ultra/CANDIDATES.lock"))?;
    verify_freeze_tag_ancestor(freeze_tag, build_commit)
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
    write_jsonl_create_new(path, rows)
}

pub(crate) fn write_jsonl_create_new<T: Serialize>(
    path: &Path,
    rows: &[T],
) -> Result<(), CliError> {
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

        assert!(validate_observer_cpus_disjoint("52", "52", false).is_err());
        assert!(validate_observer_cpus_disjoint("24-27", "26-29", false).is_err());
        assert!(validate_observer_cpus_disjoint("52", "48-51", false).is_ok());
        assert!(validate_observer_cpus_disjoint("4095", "48", true).is_err());

        // Qualifying pre-flight refuses missing observer, non-host lease, dirty build, missing tag.
        assert!(
            validate_qualified_preflight("52", None, None, "abc", false, "bench-freeze-v1")
                .is_err()
        );
        assert!(validate_qualified_preflight(
            "52",
            Some("48"),
            Some(Path::new("/tmp/BENCH_LEASE.runtime")),
            "abc",
            false,
            "bench-freeze-v1"
        )
        .is_err());
        assert!(validate_qualified_preflight(
            "52",
            Some("48"),
            Some(Path::new("/tmp/BENCH_LEASE.host")),
            "abc",
            true,
            "bench-freeze-v1"
        )
        .is_err());
        assert!(validate_qualified_preflight(
            "52",
            Some("48"),
            Some(Path::new("/tmp/BENCH_LEASE.host")),
            "abc",
            false,
            "nonexistent-freeze-tag-v999"
        )
        .is_err());
    }
}
