//! End-to-end check of the `ultra-score` binary on two paired systems.

use std::process::Command;

fn write(name: &str, lines: &[String]) -> String {
    let p = format!("{}/{name}", env!("CARGO_TARGET_TMPDIR"));
    std::fs::write(&p, lines.join("\n")).expect("write fixture");
    p
}

fn line(id: usize, ds: &str, r: &str, h: &str) -> String {
    serde_json::json!({"id": id.to_string(), "speaker": format!("spk{}", id % 12),
                       "dataset": ds, "ref": r, "hyp": h, "duration_s": 3.0})
    .to_string()
}

#[test]
fn two_systems_json_and_markdown() {
    let r = "the quick brown fox jumps over the lazy dog";
    let (mut a, mut b) = (Vec::new(), Vec::new());
    for i in 0..48 {
        let ds = if i % 2 == 0 { "clean" } else { "other" };
        a.push(line(i, ds, r, "the quick brown fox jumped over a lazy dog"));
        b.push(line(i, ds, r, "the quick brown fox jumps over a lazy dog"));
    }
    a.push(line(100, "clean", "", "uh"));
    b.push(line(100, "clean", "", ""));
    let pa = write("score_a.jsonl", &a);
    let pb = write("score_b.jsonl", &b);
    let md = format!("{}/score.md", env!("CARGO_TARGET_TMPDIR"));
    let out = Command::new(env!("CARGO_BIN_EXE_ultra-score"))
        .args([
            &pa,
            &pb,
            "--replicates",
            "5000",
            "--seed",
            "7",
            "--md-out",
            &md,
        ])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(v["schema"], "ultra.score-report/v1");
    assert_eq!(
        v["system_a"]["datasets"]["clean"]["counts"]["substitutions"],
        48
    );
    assert_eq!(
        v["system_a"]["datasets"]["clean"]["silence"]["inserted_words"],
        1
    );
    assert_eq!(
        v["system_a"]["datasets"]["clean"]["silence"]["insertions_per_hour"],
        1200.0
    );
    let d = v["comparison"]["macro_delta"]["delta"].as_f64().unwrap();
    assert!((d + 1.0 / 9.0).abs() < 1e-12, "{d}");
    assert!(v["comparison"]["macro_delta"]["upper"].as_f64().unwrap() < 0.0);
    let table = std::fs::read_to_string(&md).unwrap();
    assert!(table.contains("| B | clean | 11.11 |"), "{table}");
    assert!(
        table.contains("Delta macro WER (B - A): -11.111 points"),
        "{table}"
    );
}

#[test]
fn usage_errors() {
    let out = Command::new(env!("CARGO_BIN_EXE_ultra-score"))
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(2));
    let out = Command::new(env!("CARGO_BIN_EXE_ultra-score"))
        .args(["/nonexistent/x.jsonl"])
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(1));
}
