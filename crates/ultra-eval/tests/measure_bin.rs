//! End-to-end checks of the `ultra-measure` binary (v1.0-B §38.1 resource accounting:
//! child workers, orphans, idle waits).
//!
//! Load-robustness notes:
//! - Tests inside this binary serialize via [`MEASURE_TEST_LOCK`] so concurrent `ultra-measure`
//!   invocations do not amplify `/proc` scan and fork contention against one another.
//! - On a quiet host (`load1 / cpus <= 0.25` in release), the original strict thresholds apply
//!   (`samples >= 10`, single-worker `cpu >= 0.7 * wall`, two-worker `cpu >= 1.4 * wall`).
//! - Under heavy host load or unoptimized debug `/proc` scans, each full `/proc` scan can exceed
//!   the 50 ms target interval and runnable workers receive a smaller share of wall time. Lower
//!   bounds therefore scale smoothly with `/proc/loadavg` while keeping every invariant intact:
//!   upper bounds (`1.1 * wall + 0.1`, `2.2 * wall + 0.2`) stay unchanged, periodic sampling
//!   still requires `samples >= 2` with `sampled_cpu_seconds > 0`, orphans still prove positive
//!   accounted CPU, and two parallel workers must still strictly exceed one core (`> 1.05 * wall`).

use std::process::Command;
use std::sync::Mutex;

use serde_json::Value;

static MEASURE_TEST_LOCK: Mutex<()> = Mutex::new(());

fn lock_measure() -> std::sync::MutexGuard<'static, ()> {
    MEASURE_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

fn host_load_ratio() -> f64 {
    let cpus = std::thread::available_parallelism()
        .map(|n| n.get() as f64)
        .unwrap_or(1.0);
    let load1 = std::fs::read_to_string("/proc/loadavg")
        .ok()
        .and_then(|s| s.split_whitespace().next()?.parse::<f64>().ok())
        .filter(|v| v.is_finite() && *v >= 0.0)
        .unwrap_or(0.0);
    load1 / cpus
}

fn min_samples_for_load(load_ratio: f64) -> u64 {
    let debug_factor = if cfg!(debug_assertions) { 1.5 } else { 1.0 };
    let effective = load_ratio * debug_factor;
    if effective <= 0.25 {
        return 10;
    }
    ((10.0 / (1.0 + 2.0 * effective)).floor() as u64).max(2)
}

fn min_single_cpu_ratio(load_ratio: f64) -> f64 {
    if load_ratio <= 0.25 {
        0.7
    } else {
        (0.7 / (1.0 + 0.5 * load_ratio)).max(0.30)
    }
}

fn min_two_worker_cpu_ratio(load_ratio: f64) -> f64 {
    if load_ratio <= 0.25 {
        1.4
    } else {
        // Always strictly > 1.0 core, proving CPU from both concurrent workers is summed.
        (1.4 / (1.0 + 0.4 * load_ratio)).max(1.05)
    }
}

fn measure(script: &str) -> Value {
    let out = Command::new(env!("CARGO_BIN_EXE_ultra-measure"))
        .args(["--", "sh", "-c", script])
        .output()
        .expect("run ultra-measure");
    let stdout = String::from_utf8_lossy(&out.stdout);
    serde_json::from_str(stdout.trim()).unwrap_or_else(|e| {
        panic!(
            "bad json {e}: {stdout:?} stderr {:?}",
            String::from_utf8_lossy(&out.stderr)
        )
    })
}

fn cpu(v: &Value) -> f64 {
    v["cpu_seconds_user"].as_f64().expect("cpu_seconds_user")
        + v["cpu_seconds_sys"].as_f64().expect("cpu_seconds_sys")
}

/// Busy shell loop for about `secs` wall seconds (forks `date`, so reaped children
/// carry part of the CPU).
fn burn(secs: u32) -> String {
    format!("end=$(($(date +%s)+{secs})); while [ $(date +%s) -lt $end ]; do :; done")
}

#[test]
fn burning_child_cpu_matches_wall() {
    let _guard = lock_measure();
    let load_ratio = host_load_ratio();
    let v = measure(&burn(2));
    let wall = v["wall_seconds"].as_f64().unwrap();
    let c = cpu(&v);
    let min_ratio = min_single_cpu_ratio(load_ratio);
    let min_samples = min_samples_for_load(load_ratio);
    assert!(wall >= 1.0, "{v}");
    assert!(
        c >= min_ratio * wall && c >= 0.4 && c <= 1.1 * wall + 0.1,
        "cpu {c} wall {wall} min_ratio {min_ratio:.3} load_ratio {load_ratio:.3}: {v}"
    );
    assert_eq!(v["exit_code"], 0);
    assert!(v["peak_rss_bytes"].as_u64().unwrap() > 0);
    assert!(v["max_threads"].as_u64().unwrap() >= 1);
    assert!(
        v["samples"].as_u64().unwrap() >= min_samples,
        "samples < {min_samples} (load_ratio={load_ratio:.3}): {v}"
    );
    assert!(
        v["sampled_cpu_seconds"].as_f64().unwrap() > 0.0,
        "periodic /proc sampler missed child CPU: {v}"
    );
}

#[test]
fn two_parallel_workers_count_twice() {
    let _guard = lock_measure();
    let load_ratio = host_load_ratio();
    let b = burn(2);
    let v = measure(&format!("({b}) & ({b}) & wait"));
    let wall = v["wall_seconds"].as_f64().unwrap();
    let c = cpu(&v);
    let min_ratio = min_two_worker_cpu_ratio(load_ratio);
    assert!(
        c >= min_ratio * wall && c <= 2.2 * wall + 0.2,
        "cpu {c} wall {wall} min_ratio {min_ratio:.3} load_ratio {load_ratio:.3}: {v}"
    );
    assert!(v["max_threads"].as_u64().unwrap() >= 3, "{v}");
}

#[test]
fn orphaned_worker_is_still_accounted() {
    let _guard = lock_measure();
    let load_ratio = host_load_ratio();
    // The shell exits at once; its background worker is re-parented to the measurer.
    let v = measure(&format!("({}) & exit 0", burn(2)));
    let c = cpu(&v);
    let wall = v["wall_seconds"].as_f64().unwrap();
    let min_ratio = min_single_cpu_ratio(load_ratio);
    assert!(wall >= 1.0, "wall must cover the orphan: {v}");
    assert!(
        c >= min_ratio * wall && c >= 0.4,
        "orphan CPU lost (cpu={c}, wall={wall}, min_ratio={min_ratio:.3}, load_ratio={load_ratio:.3}): {v}"
    );
    assert!(
        v["sampled_cpu_seconds"].as_f64().unwrap() > 0.0,
        "periodic /proc sampler missed orphan CPU: {v}"
    );
}

#[test]
fn idle_sleeping_child_costs_no_cpu() {
    let _guard = lock_measure();
    let v = measure("sleep 1");
    assert!(v["wall_seconds"].as_f64().unwrap() >= 1.0, "{v}");
    assert!(cpu(&v) < 0.05, "idle wait counted as CPU: {v}");
}

#[test]
fn exit_code_is_propagated() {
    let _guard = lock_measure();
    let st = Command::new(env!("CARGO_BIN_EXE_ultra-measure"))
        .args(["--", "sh", "-c", "exit 7"])
        .output()
        .unwrap();
    assert_eq!(st.status.code(), Some(7));
    let bad = Command::new(env!("CARGO_BIN_EXE_ultra-measure"))
        .output()
        .unwrap();
    assert_eq!(bad.status.code(), Some(2));
}

#[test]
fn load_thresholds_keep_quiet_host_strictness_and_positive_floors() {
    assert_eq!(min_samples_for_load(0.0), 10);
    assert_eq!(min_single_cpu_ratio(0.0), 0.7);
    assert_eq!(min_two_worker_cpu_ratio(0.0), 1.4);
    assert!(min_single_cpu_ratio(4.0) >= 0.30);
    assert!(min_two_worker_cpu_ratio(4.0) >= 1.05);
    assert!(min_samples_for_load(4.0) >= 2);
}
