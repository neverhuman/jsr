//! End-to-end check of the `ultra-score` binary on two paired systems.

use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};

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

#[test]
fn candidate_edits_are_flushed_with_canonical_ties_and_normalization() {
    let mut child = Command::new(env!("CARGO_BIN_EXE_ultra-score"))
        .arg("--candidate-edits")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let mut input = child.stdin.take().unwrap();
    let mut output = BufReader::new(child.stdout.take().unwrap());
    writeln!(
        input,
        "{}",
        serde_json::json!({"ref":"a b", "hyps":["b a", "a b"]})
    )
    .unwrap();
    let mut line = String::new();
    output.read_line(&mut line).unwrap();
    let counts: Vec<ultra_eval::wer::EditCounts> = serde_json::from_str(&line).unwrap();
    assert_eq!(
        (
            counts[0].substitutions,
            counts[0].deletions,
            counts[0].insertions
        ),
        (2, 0, 0)
    );
    assert_eq!(counts[1].errors(), 0);
    writeln!(input, "{}", serde_json::json!({"ref":"ΟΣ", "hyps":["οσ"]})).unwrap();
    line.clear();
    output.read_line(&mut line).unwrap();
    let counts: Vec<ultra_eval::wer::EditCounts> = serde_json::from_str(&line).unwrap();
    assert_eq!(counts[0].errors(), 0);
    drop(input);
    assert!(child.wait().unwrap().success());
}

#[test]
fn alignment_ledger_matches_aggregate_and_preserves_evidence() {
    let input = write(
        "ledger_score.jsonl",
        &[serde_json::json!({
            "id": "clip",
            "dataset": "dev-other",
            "speaker": "person",
            "ref": "Go, go in.",
            "hyp": "go and",
            "timing_clock": "audio",
            "condition": "quiet onset",
            "reference_timing": [
                {"w": "Go,", "start_s": 0.1, "end_s": 0.2},
                {"w": "go", "start_s": 0.2, "end_s": 0.3},
                {"w": "in.", "start_s": 0.3, "end_s": 0.4}
            ],
            "ref_words": [
                {"time_s": [0.1, 0.2], "token_span": [0, 1]},
                {"time_s": [0.2, 0.3], "token_span": [1, 2]},
                {"function_word": true, "train_frequency": 42, "phones": "IH N"}
            ],
            "hyp_words": [{}, {"function_word": true, "phones": "AE N D"}],
            "candidates": ["go and", "go go in"]
        })
        .to_string()],
    );
    let better = write(
        "ledger_score_b.jsonl",
        &[serde_json::json!({
            "id": "clip", "dataset": "dev-other", "speaker": "person",
            "ref": "Go, go in.", "hyp": "go go in"
        })
        .to_string()],
    );
    let receipt = write(
        "ledger_identity.json",
        &[serde_json::json!({
            "audio_manifest_sha256": "a".repeat(64),
            "artifact_sha256": "b".repeat(64),
            "runtime_sha256": "c".repeat(64),
            "decoder": "beam 4",
            "chunk_policy": "fixed 160 ms"
        })
        .to_string()],
    );
    let ledger = format!("{}/ledger_output.jsonl", env!("CARGO_TARGET_TMPDIR"));
    let _ = std::fs::remove_file(&ledger);
    let output = Command::new(env!("CARGO_BIN_EXE_ultra-score"))
        .args([
            &input,
            &better,
            "--alignment-ledger",
            &ledger,
            "--receipt-a",
            &receipt,
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let score: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    let rows: Vec<serde_json::Value> = std::fs::read_to_string(&ledger)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(rows[0]["run"]["decoder"], "beam 4");
    assert_eq!(rows[1]["op"], "del");
    assert_eq!(rows[1]["reference"]["index"], 0);
    assert_eq!(rows[1]["first_correct_candidate_rank"], 2);
    assert_eq!(rows[2]["op"], "match");
    assert_eq!(
        rows[2]["reference"]["evidence"]["time_s"],
        serde_json::json!([0.2, 0.3])
    );
    assert_eq!(rows[3]["op"], "sub");
    assert_eq!(rows[3]["reference"]["word"], "in");
    assert_eq!(rows[3]["hypothesis"]["word"], "and");
    assert_eq!(
        rows[4]["canonical"],
        score["system_a"]["datasets"]["dev-other"]["counts"]
    );
    assert!(rows
        .iter()
        .any(|row| row["dimension"] == "phonetic_confusion"
            && row["bucket"] == "IH N → AE N D"
            && row["counts"]["substitutions"] == 1));
    assert!(rows.iter().any(|row| row["kind"] == "paired_transition"
        && row["op_a"] == "del"
        && row["op_b"] == "match"
        && row["ref_words"] == 1));
}
