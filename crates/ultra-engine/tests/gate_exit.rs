//! `ultra-engine-gate` exit status: 0 PASS, 1 FAIL (verdict still written), 2 error.
#![cfg(feature = "wer")]

use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::json;

/// An engine_eval-shaped run: 2 splits x 25 speakers x 2 utterances, 20 words each.
fn run(dir: &Path, name: &str, precision: &str, panel: &str, extra_errors: u64) -> PathBuf {
    let mut rows = Vec::new();
    for split in ["dev-clean", "dev-other"] {
        for spk in 0..25i64 {
            for u in 0..2 {
                rows.push(json!({"id": format!("{split}-{spk}-{u}"), "split": split,
                    "speaker": spk, "errors": 2 + if u == 0 { extra_errors } else { 0 },
                    "ref_words": 20, "hyp": "", "tokens": []}));
            }
        }
    }
    let v = json!({"panel_manifest_sha256": panel, "arch_checkpoint_sha256": "c",
        "precision": precision, "executable_sha256": "e", "int8_prepared_sha256": "p",
        "host": "h", "isa": "i", "rows": rows});
    let p = dir.join(format!("{name}.json"));
    std::fs::write(&p, v.to_string()).expect("write run");
    p
}

fn gate(fp32: &Path, int8: &Path) -> Option<i32> {
    Command::new(env!("CARGO_BIN_EXE_ultra-engine-gate"))
        .args([
            "--fp32".as_ref(),
            fp32.as_os_str(),
            "--int8".as_ref(),
            int8.as_os_str(),
        ])
        .args(["--replicates", "5000"])
        .output()
        .expect("run ultra-engine-gate")
        .status
        .code()
}

#[test]
fn exit_status_follows_the_verdict() {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("gate-exit-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("tmp dir");
    let a = run(&dir, "fp32", "fp32", "panel", 0);
    let same = run(&dir, "int8-same", "int8", "panel", 0);
    let worse = run(&dir, "int8-worse", "int8", "panel", 3);
    let other = run(&dir, "int8-other-panel", "int8", "other", 0);
    assert_eq!(gate(&a, &same), Some(0), "identical errors must PASS");
    assert_eq!(gate(&a, &worse), Some(1), "+7.5 pp must FAIL with exit 1");
    assert_eq!(gate(&a, &other), Some(2), "different panels are an error");
    std::fs::remove_dir_all(&dir).ok();
}
