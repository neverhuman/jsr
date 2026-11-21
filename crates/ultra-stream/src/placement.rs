//! Fail-closed worker L3-domain and SMT-sibling verification (`FREEZE.md §3` / `S4`) and
//! periodic fleet sustained-run series records (`S5`).

use std::collections::BTreeSet;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use ultra_eval::procfs::TreeSampler;

use crate::api::StreamError;
use crate::contrast::current_affinity;
use crate::ort_r1_ws::thread_cpu_ns;

/// Periodic 10-second resource & backlog sample recorded during sustained fleet runs (`FREEZE §4.5`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FleetSeriesPoint {
    pub elapsed_s: f64,
    /// Instantaneous process-tree RSS in bytes at sample time (`S5`).
    pub rss_bytes: u64,
    /// Instantaneous process-tree PSS in bytes at sample time (`S5`).
    pub pss_bytes: u64,
    /// Instantaneous thread count at sample time (`S5`).
    pub threads: u64,
    /// Maximum worker lateness/backlog observed during the preceding 10 s interval in ms (`S5`).
    pub max_worker_backlog_ms: Option<f64>,
}

pub(crate) struct ObserverRunSummary {
    pub peak_rss_bytes: u64,
    pub max_threads: u64,
    pub observer_cpu_s: f64,
    pub effective_cpus: Vec<usize>,
    pub series: Vec<FleetSeriesPoint>,
}

/// Shared observer thread body pinned to `obs_pin` (disjoint CCX) that samples `/proc` RSS/PSS/
/// threads every 25 ms and emits 10-second sustained-run series points with current RSS/PSS and
/// real max worker backlog (`S5`).
pub(crate) fn run_fleet_observer(
    pid: i32,
    obs_pin: &[usize],
    stop: &Arc<AtomicBool>,
    max_backlog_us: &Arc<AtomicU64>,
) -> Result<ObserverRunSummary, StreamError> {
    let effective_cpus = pin_current_thread_verified(obs_pin)?;
    let t0_ns = thread_cpu_ns(true);
    let obs_t0 = Instant::now();
    let mut next_series_s = 10.0f64;
    let mut series = Vec::new();
    let mut sampler = TreeSampler::new(pid);
    sampler.sample();
    while !stop.load(Ordering::Relaxed) {
        std::thread::sleep(Duration::from_millis(25));
        let cur = sampler.sample();
        let el = obs_t0.elapsed().as_secs_f64();
        if el >= next_series_s {
            let backlog_us = max_backlog_us.swap(0, Ordering::Relaxed);
            series.push(FleetSeriesPoint {
                elapsed_s: (el * 1000.0).round() / 1000.0,
                rss_bytes: cur.rss_bytes,
                pss_bytes: cur.pss_bytes,
                threads: cur.threads,
                max_worker_backlog_ms: Some(((backlog_us as f64) / 1000.0 * 100.0).round() / 100.0),
            });
            while next_series_s <= el {
                next_series_s += 10.0;
            }
        }
    }
    sampler.sample();
    let observer_cpu_s = (thread_cpu_ns(true).saturating_sub(t0_ns)) as f64 * 1e-9;
    let sum = sampler.summary();
    Ok(ObserverRunSummary {
        peak_rss_bytes: sum.peak_rss_bytes,
        max_threads: sum.max_threads,
        observer_cpu_s,
        effective_cpus,
        series,
    })
}

/// Assign **exactly 1 physical core per worker** (`FREEZE.md §3` / `B1` / `S4`) and verify L3/SMT
/// topology fail-closed (even when `observer_cpus` is `None`). Refuses `phys_aff.len() < workers`
/// (no silent unpinned fallback) and, when `require_exact_count` is true, refuses excess CPUs.
pub fn assign_worker_cpus(
    workers: usize,
    avail_affinity: &[usize],
    observer_cpus: Option<&[usize]>,
    require_exact_count: bool,
) -> Result<Vec<Vec<usize>>, StreamError> {
    if workers == 0 {
        return Err(StreamError::Protocol("workers must be >= 1".into()));
    }
    let phys_aff = physical_first_cpus(avail_affinity);
    if phys_aff.len() < workers {
        return Err(StreamError::Protocol(format!(
            "insufficient physical cores in process affinity for {workers} workers (only {} physical cores available from {avail_affinity:?}; S4 forbids unpinned fallback)",
            phys_aff.len()
        )));
    }
    if require_exact_count && phys_aff.len() != workers {
        return Err(StreamError::Protocol(format!(
            "qualified placement requires exact 1:1 physical cores ({workers} workers vs {} physical cores in affinity {avail_affinity:?}; B1)",
            phys_aff.len()
        )));
    }
    let worker_cpus: Vec<Vec<usize>> = phys_aff
        .into_iter()
        .take(workers)
        .map(|c| vec![c])
        .collect();
    verify_worker_l3_and_smt(&worker_cpus, observer_cpus, true)?;
    Ok(worker_cpus)
}

/// Pin the current OS thread to `cpus` and verify via `sched_getaffinity` readback that the
/// effective mask matches the requested set. Fails closed on invalid CPUs or OS refusal.
pub fn pin_current_thread_verified(cpus: &[usize]) -> Result<Vec<usize>, StreamError> {
    if cpus.is_empty() {
        return Ok(current_affinity());
    }
    let mut want = cpus.to_vec();
    want.sort_unstable();
    want.dedup();
    let mut set = nix::sched::CpuSet::new();
    for &c in &want {
        set.set(c).map_err(|e| {
            StreamError::Protocol(format!("invalid CPU id {c} in worker affinity mask: {e}"))
        })?;
    }
    nix::sched::sched_setaffinity(nix::unistd::Pid::from_raw(0), &set)
        .map_err(|e| StreamError::Runtime(format!("sched_setaffinity({want:?}) failed: {e}")))?;
    let eff = current_affinity();
    if eff != want {
        return Err(StreamError::Runtime(format!(
            "effective thread affinity {eff:?} does not match requested {want:?}"
        )));
    }
    Ok(eff)
}

fn parse_sysfs_cpu_list(raw: &str) -> Option<Vec<usize>> {
    let mut out = Vec::new();
    for part in raw.trim().split(',').filter(|s| !s.is_empty()) {
        if let Some((a, b)) = part.split_once('-') {
            let lo: usize = a.trim().parse().ok()?;
            let hi: usize = b.trim().parse().ok()?;
            if lo > hi {
                return None;
            }
            out.extend(lo..=hi);
        } else {
            out.push(part.trim().parse().ok()?);
        }
    }
    (!out.is_empty()).then_some(out)
}

/// Select one logical CPU per distinct physical core from `aff` using Linux sysfs
/// `topology/thread_siblings_list` (no hardcoded `< 64` assumption).
pub fn physical_first_cpus(aff: &[usize]) -> Vec<usize> {
    let mut sorted = aff.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    let mut seen_sibs = BTreeSet::new();
    let mut phys = Vec::with_capacity(sorted.len());
    for &cpu in &sorted {
        if seen_sibs.contains(&cpu) {
            continue;
        }
        phys.push(cpu);
        let path = format!("/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list");
        if let Ok(raw) = std::fs::read_to_string(&path) {
            if let Some(sibs) = parse_sysfs_cpu_list(&raw) {
                for s in sibs {
                    seen_sibs.insert(s);
                }
            }
        }
    }
    if phys.is_empty() {
        sorted
    } else {
        phys
    }
}

fn read_cpu_set(path: &str) -> Result<BTreeSet<usize>, StreamError> {
    let raw = std::fs::read_to_string(path).map_err(|e| {
        StreamError::Protocol(format!("cannot verify CPU topology via `{path}`: {e}"))
    })?;
    let list = parse_sysfs_cpu_list(&raw)
        .ok_or_else(|| StreamError::Protocol(format!("malformed CPU list in `{path}`")))?;
    Ok(list.into_iter().collect())
}

/// Verify fleet worker and observer CPU placement against Linux sysfs L3 and SMT topology (`S4`):
/// - Every worker's CPUs are online and share a single L3 cache domain (`cache/index3/shared_cpu_list`).
/// - No two workers share a physical core / SMT sibling (`topology/thread_siblings_list`),
///   preventing logical SMT CPUs (`>= 64` on 64-core Zen 2) from silently colliding with physical cores.
/// - When `observer_cpus` is non-empty, observer CPUs are strictly disjoint from every worker's
///   L3 domain and SMT sibling set.
pub fn verify_worker_l3_and_smt(
    worker_cpus: &[Vec<usize>],
    observer_cpus: Option<&[usize]>,
    fail_closed_topology: bool,
) -> Result<(), StreamError> {
    let online = match read_cpu_set("/sys/devices/system/cpu/online") {
        Ok(s) => s,
        Err(_) if !fail_closed_topology => return Ok(()),
        Err(e) => return Err(e),
    };
    let mut claimed_smt_domains: BTreeSet<usize> = BTreeSet::new();
    let mut all_worker_l3_and_smt: BTreeSet<usize> = BTreeSet::new();

    for (w, cpus) in worker_cpus.iter().enumerate() {
        if cpus.is_empty() {
            continue;
        }
        let mut worker_l3: Option<BTreeSet<usize>> = None;
        for &c in cpus {
            if !online.contains(&c) {
                return Err(StreamError::Protocol(format!(
                    "worker {w} CPU {c} is offline or nonexistent"
                )));
            }
            let sib_path = format!("/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list");
            let l3_path = format!("/sys/devices/system/cpu/cpu{c}/cache/index3/shared_cpu_list");
            let sibs = match read_cpu_set(&sib_path) {
                Ok(s) => s,
                Err(_) if !fail_closed_topology => continue,
                Err(e) => return Err(e),
            };
            let l3 = match read_cpu_set(&l3_path) {
                Ok(s) => s,
                Err(_) if !fail_closed_topology => continue,
                Err(e) => return Err(e),
            };
            if let Some(ref prev_l3) = worker_l3 {
                if prev_l3 != &l3 {
                    return Err(StreamError::Protocol(format!(
                        "worker {w} CPUs {cpus:?} span multiple L3 domains"
                    )));
                }
            } else {
                worker_l3 = Some(l3.clone());
            }
            if sibs.iter().any(|s| claimed_smt_domains.contains(s)) {
                return Err(StreamError::Protocol(format!(
                    "worker {w} CPU {c} collides on an SMT sibling with another worker (siblings {sibs:?})"
                )));
            }
            claimed_smt_domains.extend(&sibs);
            all_worker_l3_and_smt.extend(&sibs);
            all_worker_l3_and_smt.extend(&l3);
        }
    }

    if let Some(obs) = observer_cpus.filter(|o| !o.is_empty()) {
        for &c in obs {
            if !online.contains(&c) {
                return Err(StreamError::Protocol(format!(
                    "observer CPU {c} is offline or nonexistent"
                )));
            }
            if all_worker_l3_and_smt.contains(&c) {
                return Err(StreamError::Protocol(format!(
                    "observer CPU {c} overlaps a worker CPU, SMT sibling, or L3 cache domain"
                )));
            }
        }
    }
    Ok(())
}
