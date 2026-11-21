//! Integration tests for pass-3 `B4` loaded-artifact `CANDIDATES.lock` pin verification and
//! `N4` upfront Nemotron + `--panel` refusal.

use std::fs::File;

use ultra_cli::run_cli_args;
use ultra_cli::suite_pins::{
    verify_loaded_candidates_against_lock, CandidateKind, ResolvedCandidate,
};
use ultra_core::hash::sha256_reader;

fn sha_file(p: &std::path::Path) -> String {
    let mut f = File::open(p).expect("open");
    sha256_reader(&mut f).expect("sha")
}

#[test]
fn loaded_candidate_pin_verification_checks_actual_artifact_hashes() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let arch = tmp.path().join("arch.json");
    let w_int8 = tmp.path().join("weights.int8.safetensors");
    std::fs::write(&arch, b"{\"schema\":\"ultra.r1-weights/v1\"}").expect("arch");
    std::fs::write(&w_int8, b"int8-payload").expect("weights");
    let arch_sha = sha_file(&arch);
    let w_sha = sha_file(&w_int8);

    // Also test YAML manifest + external model artifact SHA-256 verification.
    let enc = tmp.path().join("enc.int8.onnx");
    let tok = tmp.path().join("tokens.txt");
    std::fs::write(&enc, b"enc-bytes").expect("enc");
    std::fs::write(&tok, b"tok-bytes").expect("tok");
    let enc_sha = sha_file(&enc);
    let tok_sha = sha_file(&tok);
    let yaml_path = tmp.path().join("zipformer.yaml");
    std::fs::write(
        &yaml_path,
        format!(
            "---\nschema: ultra.candidate.v1\ncandidate_id: zipformer_en_int8_320\nmodel:\n  artifact: enc.int8.onnx\n  artifact_sha256: {enc_sha}\n  tokenizer: tokens.txt\n  tokenizer_sha256: {tok_sha}\n"
        ),
    )
    .expect("yaml");
    let yaml_sha = sha_file(&yaml_path);

    let lock = tmp.path().join("CANDIDATES.lock");
    std::fs::write(
        &lock,
        format!(
            "candidates:\n  - candidate_id: r1-dense\n    status: PINNED_PENDING_FREEZE\n    arch_sha256: {arch_sha}\n    weights_int8_sha256: {w_sha}\n  - candidate_id: zipformer\n    manifest: zipformer.yaml\n    manifest_sha256: {yaml_sha}\n    status: PINNED_PENDING_FREEZE\n"
        ),
    )
    .expect("lock");

    let ok_eng = ResolvedCandidate {
        raw_arg: "r1-dense".into(),
        canonical_id: format!(
            "engine:int8:arch={}:weights={}",
            &arch_sha[..16],
            &w_sha[..16]
        ),
        kind: CandidateKind::Engine {
            precision: "int8".into(),
            prepared_int8: true,
        },
        artifacts: vec![
            ("arch.json".into(), arch.clone(), arch_sha.clone()),
            (
                "weights.int8.safetensors".into(),
                w_int8.clone(),
                w_sha.clone(),
            ),
        ],
    };
    let ok_zip = ResolvedCandidate {
        raw_arg: "zipformer-int8".into(),
        canonical_id: "sherpa-zipformer:int8".into(),
        kind: CandidateKind::Zipformer {
            precision: "int8".into(),
        },
        artifacts: vec![
            ("enc.int8.onnx".into(), enc, enc_sha),
            ("tokens.txt".into(), tok, tok_sha),
        ],
    };
    verify_loaded_candidates_against_lock(&lock, tmp.path(), &[ok_eng.clone(), ok_zip])
        .expect("pinned engine + yaml-backed sherpa pass");

    // Runtime weight quantization (prepared_int8 = false) is refused in --qualified.
    let mut unprep = ok_eng.clone();
    unprep.kind = CandidateKind::Engine {
        precision: "int8".into(),
        prepared_int8: false,
    };
    assert!(verify_loaded_candidates_against_lock(&lock, tmp.path(), &[unprep]).is_err());

    // Tampered loaded weights SHA is refused even though CANDIDATES.lock itself is unchanged.
    let mut tampered = ok_eng;
    tampered.artifacts[1].2 = "0".repeat(64);
    assert!(verify_loaded_candidates_against_lock(&lock, tmp.path(), &[tampered]).is_err());

    // RESEARCH_UNQUALIFIED alone must be refused in --qualified.
    let lock_unq = tmp.path().join("CANDIDATES_unq.lock");
    std::fs::write(
        &lock_unq,
        format!(
            "candidates:\n  - candidate_id: r1-dense\n    status: RESEARCH_UNQUALIFIED\n    arch_sha256: {arch_sha}\n    weights_int8_sha256: {w_sha}\n"
        ),
    )
    .expect("lock2");
    assert!(verify_loaded_candidates_against_lock(
        &lock_unq,
        tmp.path(),
        &[ResolvedCandidate {
            raw_arg: "r1-dense".into(),
            canonical_id: "id".into(),
            kind: CandidateKind::Engine {
                precision: "int8".into(),
                prepared_int8: true,
            },
            artifacts: vec![
                ("arch.json".into(), arch, arch_sha),
                ("weights.int8.safetensors".into(), w_int8, w_sha),
            ],
        }]
    )
    .is_err());
}

#[test]
fn nemotron_with_panel_is_refused_upfront_before_any_work() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let out = tmp.path().join("should_not_exist");
    let args: Vec<String> = [
        "bench-suite",
        "--candidates",
        "zipformer-int8,nemotron-560",
        "--panel",
        "docs/ultra/data/dev_selection_panel.json",
        "--out-dir",
        out.to_str().expect("utf8"),
    ]
    .iter()
    .map(|s| (*s).to_owned())
    .collect();
    let err = run_cli_args(&args).expect_err("nemotron + --panel must fail upfront");
    assert!(
        err.to_string().contains("Nemotron") && err.to_string().contains("--panel"),
        "unexpected error: {err}"
    );
    assert!(
        !out.exists(),
        "upfront refusal must not create out-dir or run earlier candidates"
    );
}

#[test]
fn legacy_bench_and_qualified_g8_rejection_cases_fail_closed() {
    use ultra_cli::suite_abab::validate_qualified_ccost_arms;
    use ultra_cli::suite_ladder::{validate_qualified_ladder_preset, LadderSpec};

    let sv = |xs: &[&str]| -> Vec<String> { xs.iter().map(|s| (*s).to_owned()).collect() };

    // S6: legacy `ultra bench` refuses test-* and sealed npz basenames under dev-* alias.
    assert!(run_cli_args(&sv(&["bench", "--split", "test-clean"])).is_err());
    assert!(run_cli_args(&sv(&[
        "bench",
        "--split",
        "dev-clean",
        "--npz",
        "/nonexistent/test-clean.npz"
    ]))
    .is_err());
    assert!(run_cli_args(&sv(&[
        "bench-suite",
        "--candidates",
        "zipformer-int8",
        "--split",
        "dev-clean",
        "--npz",
        "/nonexistent/test-clean.npz",
        "--out-dir",
        "/tmp/never-created"
    ]))
    .is_err());

    // G8: NaN / Infinity durations refused explicitly.
    let freeze = LadderSpec::freeze_preset();
    assert!(validate_qualified_ladder_preset(&freeze, 3, f64::NAN, 300.0).is_err());
    assert!(validate_qualified_ladder_preset(&freeze, 3, 120.0, f64::INFINITY).is_err());

    // G8: --qualified locks grid_ms=320, max_chunks=2, threads=1, and per-CCX workers=4.
    let tmp = tempfile::tempdir().expect("tempdir");
    let fake_npz = tmp.path().join("dev-clean.npz");
    std::fs::write(&fake_npz, b"stub").expect("stub");
    let npz_s = fake_npz.to_str().expect("utf8");
    let out_s = tmp.path().join("out").to_str().expect("utf8").to_owned();
    assert!(run_cli_args(&sv(&[
        "bench-suite",
        "--qualified",
        "--candidates",
        "zipformer-int8",
        "--split",
        "dev-clean",
        "--npz",
        npz_s,
        "--density-workers",
        "2",
        "--out-dir",
        &out_s
    ]))
    .is_err());
    assert!(run_cli_args(&sv(&[
        "bench-suite",
        "--qualified",
        "--candidates",
        "zipformer-int8",
        "--split",
        "dev-clean",
        "--npz",
        npz_s,
        "--grid-ms",
        "160",
        "--out-dir",
        &out_s
    ]))
    .is_err());
    assert!(run_cli_args(&sv(&[
        "bench-suite",
        "--qualified",
        "--candidates",
        "zipformer-int8",
        "--split",
        "dev-clean",
        "--npz",
        npz_s,
        "--threads",
        "2",
        "--out-dir",
        &out_s
    ]))
    .is_err());

    // N10 / 04d09f0: --qualified C-cost enforces exact 3-arm [R0, R1-dense, R1-MR] + 8 blocks.
    assert_eq!(
        validate_qualified_ccost_arms(&["zipformer-int8", "r1-dense", "r1-mr"], 8, None)
            .expect("valid 3-arm"),
        0
    );
    assert!(validate_qualified_ccost_arms(&["zipformer-int8", "r1-dense"], 8, None).is_err());
    assert!(
        validate_qualified_ccost_arms(&["r1-dense", "zipformer-int8", "r1-mr"], 8, None).is_err()
    );
    assert!(
        validate_qualified_ccost_arms(&["zipformer-int8", "r1-dense", "r1-mr"], 5, None).is_err()
    );
}
