//! End-to-end checks of the `ultra-measure` binary (v1.0-B §38.1 resource accounting:
//! child workers, orphans, idle waits).

use std::process::Command;

use serde_json::Value;

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
    let v = measure(&burn(2));
    let wall = v["wall_seconds"].as_f64().unwrap();
    let c = cpu(&v);
    assert!(wall >= 1.0, "{v}");
    assert!(
        c >= 0.7 * wall && c <= 1.1 * wall + 0.1,
        "cpu {c} wall {wall}: {v}"
    );
    assert_eq!(v["exit_code"], 0);
    assert!(v["peak_rss_bytes"].as_u64().unwrap() > 0);
    assert!(v["max_threads"].as_u64().unwrap() >= 1);
    assert!(v["samples"].as_u64().unwrap() >= 10);
}

#[test]
fn two_parallel_workers_count_twice() {
    let b = burn(2);
    let v = measure(&format!("({b}) & ({b}) & wait"));
    let wall = v["wall_seconds"].as_f64().unwrap();
    let c = cpu(&v);
    assert!(
        c >= 1.4 * wall && c <= 2.2 * wall + 0.2,
        "cpu {c} wall {wall}: {v}"
    );
    assert!(v["max_threads"].as_u64().unwrap() >= 3, "{v}");
}

#[test]
fn orphaned_worker_is_still_accounted() {
    // The shell exits at once; its background worker is re-parented to the measurer.
    let v = measure(&format!("({}) & exit 0", burn(2)));
    let c = cpu(&v);
    let wall = v["wall_seconds"].as_f64().unwrap();
    assert!(wall >= 1.0, "wall must cover the orphan: {v}");
    assert!(c >= 0.7 * wall, "orphan CPU lost: {v}");
}

#[test]
fn idle_sleeping_child_costs_no_cpu() {
    let v = measure("sleep 1");
    assert!(v["wall_seconds"].as_f64().unwrap() >= 1.0, "{v}");
    assert!(cpu(&v) < 0.05, "idle wait counted as CPU: {v}");
}

#[test]
fn exit_code_is_propagated() {
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
