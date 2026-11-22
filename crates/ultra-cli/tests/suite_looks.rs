//! Integration tests for S6 sealed-read guard (`PUBLIC-TEST` + cross-route `ood-sealed-read`
//! keyed by canonical OOD set, matching `python/ultra/ultra/ood.py` and `score-ood`).

use std::path::Path;
use std::process::Command;

use ultra_cli::suite_looks::{
    canonical_ood_set, guard_and_log_in_repo, guard_and_log_in_repo_for_partition,
    record_aborted_public_test_reads,
};
use ultra_core::receipt::Partition;

fn git_in(repo: &Path, args: &[&str]) -> String {
    let o = Command::new("git")
        .arg("-C")
        .arg(repo)
        .args(args)
        .output()
        .expect("git");
    assert!(o.status.success(), "git {args:?} failed");
    String::from_utf8_lossy(&o.stdout).trim().to_owned()
}

fn scratch_repo() -> (tempfile::TempDir, String) {
    let tmp = tempfile::tempdir().expect("tempdir");
    git_in(tmp.path(), &["init", "-q"]);
    std::fs::write(tmp.path().join("README"), "x").expect("write");
    git_in(tmp.path(), &["add", "README"]);
    git_in(
        tmp.path(),
        &[
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@t",
            "commit",
            "-q",
            "-m",
            "init",
        ],
    );
    let commit = git_in(tmp.path(), &["rev-parse", "HEAD"]);
    (tmp, commit)
}

#[test]
fn public_test_guard_enforces_tag_clean_build_canonical_id_and_abort_log() {
    let (tmp, commit) = scratch_repo();
    let cands = vec![("nemotron-560:gguf=deadbeef".into(), "nemotron-560".into())];

    // S6(1): refused before bench-freeze-v1 tag exists, even in exploratory mode.
    assert!(guard_and_log_in_repo(
        tmp.path(),
        "look-001",
        "test-clean",
        &cands,
        "bench-freeze-v1",
        &commit,
        false,
        "exploratory",
    )
    .is_err());

    git_in(tmp.path(), &["tag", "bench-freeze-v1"]);

    // Dirty build still refused.
    assert!(guard_and_log_in_repo(
        tmp.path(),
        "look-001",
        "test-clean",
        &cands,
        "bench-freeze-v1",
        &commit,
        true,
        "qualified",
    )
    .is_err());

    // Historical reconstruction line must NOT block the first post-freeze read.
    let looks = tmp.path().join("docs/ultra/LOOKS.jsonl");
    std::fs::create_dir_all(looks.parent().expect("parent")).expect("mkdir");
    std::fs::write(
        &looks,
        "{\"kind\":\"public-test-read\",\"historical_reconstruction\":true,\"split\":\"test-clean\",\"candidate\":\"nemotron-560:gguf=deadbeef\",\"freeze_tag\":\"bench-freeze-v1\"}\n",
    )
    .expect("seed");

    let logged = guard_and_log_in_repo(
        tmp.path(),
        "look-001",
        "test-clean",
        &cands,
        "bench-freeze-v1",
        &commit,
        false,
        "qualified",
    )
    .expect("first post-freeze read passes");

    // S6(3): different CLI alias resolving to the same canonical_id MUST be refused.
    let alias = vec![(
        "nemotron-560:gguf=deadbeef".into(),
        "nemotron_en_q8_live_560".into(),
    )];
    let err = guard_and_log_in_repo(
        tmp.path(),
        "look-002",
        "test-clean",
        &alias,
        "bench-freeze-v1",
        &commit,
        false,
        "qualified",
    )
    .expect_err("alias second read must fail");
    assert!(err.to_string().contains("second PUBLIC-TEST read"));

    // S6(4): abort audit record appends cleanly.
    record_aborted_public_test_reads(&logged, "simulated downstream error");
    let text = std::fs::read_to_string(&looks).expect("read");
    assert!(text.contains("public-test-read-aborted"));
    assert!(text.contains("simulated downstream error"));
}

#[test]
fn domain_sealed_ood_uses_unified_ood_sealed_read_keyed_by_set_across_routes() {
    assert_eq!(canonical_ood_set("ami-ihm"), "ami");
    assert_eq!(canonical_ood_set("AMI_test"), "ami");
    assert_eq!(canonical_ood_set("voxpopuli-en"), "voxpopuli");

    let (tmp, commit) = scratch_repo();
    git_in(tmp.path(), &["tag", "bench-freeze-v1"]);
    let looks = tmp.path().join("docs/ultra/LOOKS.jsonl");
    std::fs::create_dir_all(looks.parent().expect("parent")).expect("mkdir");

    // Route 1 (Python ood.py / score-ood) already read `voxpopuli` without `freeze_tag`/`candidate`.
    std::fs::write(
        &looks,
        format!(
            "{{\"ts\":\"2026-11-22T00:00:00Z\",\"kind\":\"ood-sealed-read\",\"look_id\":\"ood-py-1\",\"set\":\"voxpopuli\",\"user\":\"ubuntu\",\"head\":\"{commit}\",\"freeze_tag_commit\":\"{commit}\",\"sealed_sha256\":\"00\"}}\n"
        ),
    )
    .expect("seed ood.py look");

    let cand_a = vec![("r1-dense:int8=aaaa".into(), "r1-dense".into())];
    let cand_b = vec![("r1-mr:int8=bbbb".into(), "r1-mr".into())];

    // Classic bench-suite MUST refuse reading `voxpopuli` (or alias `voxpopuli-en`) after ood.py/score-ood!
    for alias in ["voxpopuli", "voxpopuli-en"] {
        let err = guard_and_log_in_repo_for_partition(
            tmp.path(),
            Partition::DomainSealed,
            "look-classic-vp",
            alias,
            &cand_a,
            "bench-freeze-v1",
            &commit,
            false,
            "qualified",
        )
        .expect_err("second OOD read after ood.py/score-ood must be refused");
        assert!(err.to_string().contains("second PRIVATE-OOD read"));
        assert!(err.to_string().contains("voxpopuli"));
    }

    // First read of `ami` via classic bench-suite (using alias `ami-ihm`) logs `kind=ood-sealed-read` + `set=ami`.
    let ood_logged = guard_and_log_in_repo_for_partition(
        tmp.path(),
        Partition::DomainSealed,
        "look-ood-ami-1",
        "ami-ihm",
        &cand_a,
        "bench-freeze-v1",
        &commit,
        false,
        "qualified",
    )
    .expect("first AMI read passes");
    assert_eq!(ood_logged.board, "PRIVATE-OOD");
    assert_eq!(ood_logged.kind, "ood-sealed-read");
    assert_eq!(ood_logged.ood_set.as_deref(), Some("ami"));

    // Verify the exact JSON keys expected by python/ultra/ultra/ood.py (`kind` + `set`) are written.
    let lines: Vec<serde_json::Value> = std::fs::read_to_string(&looks)
        .expect("read looks")
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).expect("valid json"))
        .collect();
    let last = lines.last().expect("appended line");
    assert_eq!(
        last.get("kind").and_then(|v| v.as_str()),
        Some("ood-sealed-read")
    );
    assert_eq!(last.get("set").and_then(|v| v.as_str()), Some("ami"));
    assert_eq!(
        last.get("look_id").and_then(|v| v.as_str()),
        Some("look-ood-ami-1")
    );

    // Second read of `ami` (different candidate, different split alias, different look_id) MUST be refused.
    let err = guard_and_log_in_repo_for_partition(
        tmp.path(),
        Partition::DomainSealed,
        "look-ood-ami-2",
        "ami",
        &cand_b,
        "bench-freeze-v1",
        &commit,
        false,
        "qualified",
    )
    .expect_err("second AMI read must be refused across aliases/candidates");
    assert!(err.to_string().contains("second PRIVATE-OOD read"));
    assert!(err.to_string().contains("ami"));
}
