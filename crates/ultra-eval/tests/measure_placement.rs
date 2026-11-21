//! Isolated-process checks: affinity, descendant inheritance, refusal, legacy JSON.

use std::process::Command;
use std::time::Duration;

use ultra_eval::affinity::{parse_cpus, PlacementSpec};
use ultra_eval::procfs::Measurement;

fn current_list() -> String {
    std::fs::read_to_string("/proc/thread-self/status")
        .expect("read this thread's Linux status")
        .lines()
        .find_map(|line| line.strip_prefix("Cpus_allowed_list:").map(str::trim))
        .expect("Linux status includes affinity")
        .to_owned()
}

fn pair() -> Option<(String, String)> {
    let allowed = parse_cpus(&current_list()).expect("parse caller affinity");
    for service in &allowed {
        for observer in &allowed {
            let spec = PlacementSpec {
                service_cpus: service.to_string(),
                observer_cpus: observer.to_string(),
            };
            if spec.validate().is_ok() {
                return Some((spec.service_cpus, spec.observer_cpus));
            }
        }
    }
    eprintln!("SKIPPED placement fixture: caller affinity lacks two supported L3 domains");
    None
}

#[test]
#[allow(
    clippy::zombie_processes,
    reason = "The isolated fixture deliberately leaves a descendant for ultra-measure to adopt and reap"
)]
fn placement_child_fixture() {
    let Ok(mode) = std::env::var("ULTRA_PLACEMENT_FIXTURE") else {
        return;
    };
    if mode == "restore" {
        let before = current_list();
        let placement = PlacementSpec {
            service_cpus: std::env::var("ULTRA_SERVICE_CPU").unwrap(),
            observer_cpus: std::env::var("ULTRA_OBSERVER_CPU").unwrap(),
        };
        let measured = ultra_eval::procfs::measure_command_with_placement(
            "true",
            &[],
            Duration::from_millis(20),
            Some(&placement),
        )
        .unwrap();
        assert!(measured.placement.is_some());
        assert_eq!(current_list(), before, "restore after successful drain");
        assert!(ultra_eval::procfs::measure_command_with_placement(
            "/nonexistent-ultra-placement-fixture",
            &[],
            Duration::from_millis(20),
            Some(&placement),
        )
        .is_err());
        assert_eq!(current_list(), before, "restore after failed spawn");
        return;
    }
    eprintln!("PLACEMENT_FIXTURE:{mode}:{}", current_list());
    if mode == "child" {
        // Deliberate orphan: the measurer must drain this inherited-affinity descendant.
        let _child = Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "placement_child_fixture", "--nocapture"])
            .env("ULTRA_PLACEMENT_FIXTURE", "grandchild")
            .spawn()
            .unwrap();
    } else {
        std::thread::sleep(Duration::from_millis(150));
    }
}

#[test]
fn library_restores_affinity_on_success_and_spawn_error() {
    let Some((service, observer)) = pair() else {
        return;
    };
    let output = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "placement_child_fixture", "--nocapture"])
        .env("ULTRA_PLACEMENT_FIXTURE", "restore")
        .env("ULTRA_SERVICE_CPU", service)
        .env("ULTRA_OBSERVER_CPU", observer)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn observer_and_service_are_pinned_and_descendants_drain() {
    let Some((service, observer)) = pair() else {
        return;
    };
    let output = Command::new(env!("CARGO_BIN_EXE_ultra-measure"))
        .args([
            "--interval-ms",
            "20",
            "--service-cpus",
            &service,
            "--observer-cpus",
            &observer,
            "--",
        ])
        .arg(std::env::current_exe().unwrap())
        .args(["--exact", "placement_child_fixture", "--nocapture"])
        .env("ULTRA_PLACEMENT_FIXTURE", "child")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let measured: Measurement = serde_json::from_slice(&output.stdout).unwrap();
    let report = measured.placement.unwrap();
    let expected: Vec<_> = parse_cpus(&service).unwrap().into_iter().collect();
    assert_eq!(report.effective_service_cpus_at_spawn, expected);
    assert_eq!(
        report.effective_observer_cpus,
        parse_cpus(&observer)
            .unwrap()
            .into_iter()
            .collect::<Vec<_>>()
    );
    assert!(report.separate_l3_and_smt);
    let stderr = String::from_utf8_lossy(&output.stderr);
    for mode in ["child", "grandchild"] {
        assert!(
            stderr.contains(&format!("PLACEMENT_FIXTURE:{mode}:{service}")),
            "{stderr}"
        );
    }
    assert!(measured.wall_seconds >= 0.15, "orphan drain lost");
    assert!(measured.max_processes.unwrap() >= 1);
    assert!(measured.observer_cpu_seconds.unwrap().is_finite());
    assert_eq!(measured.interval_ms, Some(20));
}

#[test]
fn bad_placement_fails_before_command_execution() {
    let Some((service, observer)) = pair() else {
        return;
    };
    for (svc, obs) in [(&service[..], &service[..]), ("999999", &observer)] {
        let output = Command::new(env!("CARGO_BIN_EXE_ultra-measure"))
            .args([
                "--service-cpus",
                svc,
                "--observer-cpus",
                obs,
                "--",
                "sh",
                "-c",
                "printf SHOULD_NOT_RUN",
            ])
            .output()
            .unwrap();
        assert_eq!(output.status.code(), Some(3));
        assert!(output.stdout.is_empty());
        assert!(!String::from_utf8_lossy(&output.stderr).contains("SHOULD_NOT_RUN"));
    }
    let output = Command::new(env!("CARGO_BIN_EXE_ultra-measure"))
        .args(["--service-cpus", &service, "--", "true"])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2));
}

#[test]
fn legacy_measurements_remain_readable_and_exit_status_is_preserved() {
    let output = Command::new(env!("CARGO_BIN_EXE_ultra-measure"))
        .args(["--", "sh", "-c", "exit 7"])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(7));
    let mut value: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    for key in [
        "placement",
        "observer_cpu_seconds",
        "max_processes",
        "interval_ms",
    ] {
        value.as_object_mut().unwrap().remove(key);
    }
    let old: Measurement = serde_json::from_value(value).unwrap();
    assert!(old.placement.is_none() && old.observer_cpu_seconds.is_none());
    assert!(old.max_processes.is_none() && old.interval_ms.is_none());
    assert_eq!(old.exit_code, Some(7));
}
