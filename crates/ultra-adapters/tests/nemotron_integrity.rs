//! Integrity, missing/extra output rejection, and grandchild `ESRCH` orphan-reaping tests
//! for `ultra_adapters::nemotron`.

use std::os::unix::fs::PermissionsExt;
use std::path::PathBuf;
use std::time::Duration;

use ultra_adapters::{run_nemotron_batch, NemotronChunkMode, NemotronConfig};
use ultra_core::receipt::{HonestyLabel, Partition};

fn write_executable(path: &std::path::Path, body: &str) {
    std::fs::write(path, body).expect("write script");
    let mut perms = std::fs::metadata(path).expect("meta").permissions();
    perms.set_mode(0o755);
    std::fs::set_permissions(path, perms).expect("chmod +x");
}

#[test]
fn nemotron_adapter_enforces_exact_outputs_and_reaps_grandchild_orphans() {
    let npz_path = std::env::var("JSR_DATA")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/mnt/xbabe3-external/jsr"))
        .join("cache/librispeech_pcm/dev-clean.npz");
    if !npz_path.exists() {
        eprintln!("SKIPPED nemotron_integrity: dev-clean.npz missing");
        return;
    }

    let tmp = tempfile::tempdir().expect("tempdir");
    let fake_model = tmp.path().join("fake.gguf");
    std::fs::write(&fake_model, b"gguf-placeholder").expect("write model");

    // 1. Valid mock nemo-speech script writing exact u00000.txt & u00001.txt + JSON stdout.
    let ok_bin = tmp.path().join("mock_ok.sh");
    write_executable(
        &ok_bin,
        r#"#!/usr/bin/env bash
set -euo pipefail
out_dir=""
while [[ $# -gt 0 ]]; do
  if [[ "$1" == "--save" ]]; then out_dir="$2"; shift 2; else shift; fi
done
sleep 0.04
printf "mister quilter is the apostle\n" > "$out_dir/u00000.txt"
printf "nor is mister quilter's manner\n" > "$out_dir/u00001.txt"
printf '{"load_ms":12.5,"runs":[{"metrics":{"chunk_latency_ms":{"p50":40.0,"p95":55.0,"p99":60.0}}}]}\n'
"#,
    );

    let mut cfg = NemotronConfig {
        binary_path: ok_bin,
        model_path: fake_model.clone(),
        mode: NemotronChunkMode::Live160,
        threads: 1,
        concurrency: 1,
        timeout: Duration::from_secs(5),
        npz_path: npz_path.clone(),
        split_name: "dev-clean".into(),
        partition: Partition::Dev,
        offset: 0,
        limit: Some(2),
        source_commit: "test".into(),
        hardware_id: "h0".into(),
        completion_status: "exploratory".into(),
    };

    let res = run_nemotron_batch(&cfg).expect("valid mock nemo-speech");
    assert_eq!(res.utterances, 2);
    assert_eq!(res.receipt.label, HonestyLabel::UnknownOverlap);
    assert_eq!(res.receipt.cap_hits, None);
    assert_eq!(res.receipt.completion_status, "exploratory");
    assert!(res.receipt.validate_as_evidence().is_err());
    assert_eq!(res.scored_rows.len(), 2);

    // 2. Missing output file (exits 0 with only u00000.txt) MUST fail integrity check.
    let missing_bin = tmp.path().join("mock_missing.sh");
    write_executable(
        &missing_bin,
        r#"#!/usr/bin/env bash
set -euo pipefail
out_dir=""
while [[ $# -gt 0 ]]; do
  if [[ "$1" == "--save" ]]; then out_dir="$2"; shift 2; else shift; fi
done
printf "only first\n" > "$out_dir/u00000.txt"
printf '{"load_ms":1.0,"runs":[{"metrics":{"chunk_latency_ms":{"p50":1.0,"p95":1.0,"p99":1.0}}}]}\n'
"#,
    );
    cfg.binary_path = missing_bin;
    assert!(run_nemotron_batch(&cfg).is_err());

    // 3. Hung script spawning a grandchild `sleep 60` MUST time out and kill the grandchild (ESRCH).
    let gc_pid_file = tmp.path().join("grandchild.pid");
    let hung_bin = tmp.path().join("mock_hung.sh");
    write_executable(
        &hung_bin,
        &format!(
            "#!/usr/bin/env bash\nsleep 60 &\necho $! > {}\nwait\n",
            gc_pid_file.display()
        ),
    );
    cfg.binary_path = hung_bin;
    cfg.timeout = Duration::from_millis(200);
    assert!(run_nemotron_batch(&cfg).is_err());

    let gc_pid: i32 = std::fs::read_to_string(&gc_pid_file)
        .expect("read grandchild.pid")
        .trim()
        .parse()
        .expect("parse pid");
    // SAFETY: Signal 0 checks whether `gc_pid` exists without delivering a signal.
    let rc = unsafe { libc::kill(gc_pid, 0) };
    assert_eq!(rc, -1);
    assert_eq!(
        std::io::Error::last_os_error().raw_os_error(),
        Some(libc::ESRCH),
        "grandchild {gc_pid} survived timeout kill!"
    );
}
