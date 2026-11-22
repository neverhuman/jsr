//! Real child-process faults must fail the bridge, never synthesize an empty output.
use std::os::unix::fs::PermissionsExt;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use ultra_bridge::protocol::{Packet, Worker};
use ultra_bridge::records::{Job, Plan, SCHEMA};

static NEXT: AtomicU64 = AtomicU64::new(0);
/// Serialize script writes against concurrent `fork`/`exec` calls in other test threads so a
/// sibling thread's `fork()` never inherits an open writable script fd and triggers `ETXTBSY`.
static SCRIPT_EXEC_LOCK: Mutex<()> = Mutex::new(());

fn write_exec_script(path: &std::path::Path, content: &str) {
    let _g = SCRIPT_EXEC_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, content).expect("fixture script tmp");
    std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(0o700))
        .expect("fixture executable");
    std::fs::rename(&tmp, path).expect("fixture script rename");
}

fn load_worker(plan: &Plan) -> Result<Worker, String> {
    let _g = SCRIPT_EXEC_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    Worker::load(plan).map_err(|e| e.to_string())
}

struct Fixture(PathBuf);
impl Fixture {
    fn new(body: &str) -> Self {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/bridge-tests")
            .join(format!(
                "{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
        std::fs::create_dir_all(&root).expect("fixture directory");
        write_exec_script(&root.join("worker.sh"), &format!("#!/bin/sh\n{body}\n"));
        Self(root)
    }
    fn plan(&self) -> Plan {
        let path = self.0.join("unused");
        Plan {
            schema: SCHEMA.into(),
            run_id: "current".into(),
            job: Job {
                inputs: path.clone(),
                bundle: path.clone(),
                engine: "r1-bridge".into(),
                precision: "fp32".into(),
                out: path.clone(),
                streams: 1,
                packet_ms: 20,
                threads: 1,
                paced: false,
                guarded: false,
                python: self.0.join("worker.sh"),
                worker: path.clone(),
                measure: path,
                timeout_s: 1,
                host_window: "fault fixture".into(),
                service_cpus: None,
                observer_cpus: None,
            },
            workload_sha256: "a".repeat(64),
            bundle_manifest_sha256: "a".repeat(64),
            executable_sha256: "a".repeat(64),
            worker_sha256: "a".repeat(64),
            consumer_sha256: "a".repeat(64),
            source_commit: "a".repeat(40),
            source_dirty: false,
            host: serde_json::Value::Null,
        }
    }
    fn body(&self, text: &str) {
        write_exec_script(&self.0.join("worker.sh"), &format!("#!/bin/sh\n{text}\n"));
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).expect("remove fixture");
    }
}

#[test]
fn crashed_and_malformed_workers_are_errors() {
    for body in ["exit 23", "printf 'not-json\\n'", "printf '{\"run_id\":\"stale\",\"worker_id\":\"w0\",\"seq\":0,\"kind\":\"init\",\"rows\":[],\"info\":null,\"encoder_calls\":0}\\n'"] {
        let fixture = Fixture::new(body);
        assert!(load_worker(&fixture.plan()).is_err());
    }
}

#[test]
fn hung_worker_and_its_child_are_killed_and_reaped_at_timeout() {
    let fixture = Fixture::new("");
    let pidfile = fixture.0.join("grandchild.pid");
    fixture.body(&format!(
        "sleep 30 &\necho $! > '{}'\n{}",
        pidfile.display(),
        script("wait")
    ));
    // PID publication precedes the valid init response. Returning from load is
    // the handshake: the descendant is known before the step timeout can fire.
    let mut worker = load_worker(&fixture.plan()).expect("fixture init after PID publication");
    assert!(pidfile.is_file(), "init must follow descendant publication");
    let start = Instant::now();
    assert!(worker.step(&[packet()]).is_err());
    drop(worker); // Timeout returns an error; RAII owns process-group drain.
    assert!(start.elapsed() < Duration::from_secs(5));
    assert_gone(&pidfile);
}

#[test]
fn hung_initialization_obeys_its_timeout() {
    let fixture = Fixture::new("exec sleep 30");
    let start = Instant::now();
    assert!(load_worker(&fixture.plan()).is_err());
    assert!(start.elapsed() < Duration::from_secs(5));
}

fn assert_gone(path: &std::path::Path) {
    let pid: i32 = std::fs::read_to_string(path)
        .expect("grandchild pid")
        .trim()
        .parse()
        .expect("pid");
    // A SIGKILLed, orphaned grandchild stays a zombie until init reaps it, and
    // kill(pid, 0) still succeeds on a zombie. Poll (bounded) instead of
    // checking once; the process must still disappear well within the bound.
    let deadline = Instant::now() + Duration::from_secs(3);
    loop {
        // SAFETY: signal 0 only checks existence; no process is modified.
        let rc = unsafe { libc::kill(pid, 0) };
        let errno = std::io::Error::last_os_error().raw_os_error();
        if rc == -1 && errno == Some(libc::ESRCH) {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "grandchild {pid} still exists 3 s after drain (rc={rc}, errno={errno:?})"
        );
        std::thread::sleep(Duration::from_millis(20));
    }
}

fn ready() -> String {
    serde_json::json!({"run_id":"current", "worker_id":"w0", "seq":0, "kind":"init",
        "rows":[], "encoder_calls":0, "model_batch_ms":[], "info":{"manifest_sha256":"a".repeat(64),
        "providers":[["CPUExecutionProvider"],["CPUExecutionProvider"],["CPUExecutionProvider"]]}})
    .to_string()
}

fn batch(rows: serde_json::Value) -> String {
    serde_json::json!({"run_id":"current", "worker_id":"w0", "seq":1, "kind":"batch",
        "rows":rows, "encoder_calls":0, "model_batch_ms":[]})
    .to_string()
}

fn script(after_init: &str) -> String {
    format!(
        r#"
read_frame() {{
    n=$(dd bs=4 count=1 iflag=fullblock status=none | od -An -tu4)
    h=$(dd bs=1 count="$n" iflag=fullblock status=none)
    samples=$(printf '%s' "$h" | jq '[.items[].samples] | add // 0')
    dd bs=1 count=$((samples * 2)) iflag=fullblock status=none >/dev/null
}}
read_frame
printf '%s\n' '{}'
read_frame
{after_init}
"#,
        ready()
    )
}

fn packet() -> Packet {
    Packet {
        id: "u0".into(),
        offset: 0,
        pcm: vec![0; 320],
        start: true,
        finish: true,
    }
}

#[test]
fn crash_eof_oversized_and_wrong_or_absent_rows_fail_during_step() {
    let wrong = batch(
        serde_json::json!([{"id":"wrong", "accepted_samples":320,"processed_samples":320,
        "terminal":true,"drained":true,"hyp":"","forced_advances":0,"pcm_sha256":"a".repeat(64)}]),
    );
    for fault in [
        "exit 23".into(),
        "exit 0".into(),
        "head -c 4194305 /dev/zero | tr '\\000' x".into(),
        format!("printf '%s\\n' '{}'", batch(serde_json::json!([]))),
        format!("printf '%s\\n' '{wrong}'"),
    ] {
        let fixture = Fixture::new(&script(&fault));
        let mut worker = load_worker(&fixture.plan()).expect("valid init control");
        assert!(worker.step(&[packet()]).is_err());
    }
}

#[test]
fn step_timeout_drains_actual_grandchild_and_nonzero_shutdown_is_failure() {
    let fixture = Fixture::new("");
    let pidfile = fixture.0.join("grandchild.pid");
    fixture.body(&format!(
        "sleep 30 &\necho $! > '{}'\n{}",
        pidfile.display(),
        script("wait")
    ));
    let mut worker = load_worker(&fixture.plan()).expect("valid init control");
    assert!(worker.step(&[packet()]).is_err());
    drop(worker);
    assert_gone(&pidfile);

    let row = serde_json::json!([{"id":"u0", "accepted_samples":320,"processed_samples":320,
        "terminal":true,"drained":true,"hyp":"","forced_advances":0,"pcm_sha256":"a".repeat(64)}]);
    let shutdown = serde_json::json!({"run_id":"current", "worker_id":"w0", "seq":2,"kind":"shutdown",
        "rows":[],"encoder_calls":0,"model_batch_ms":[]});
    fixture.body(&script(&format!(
        "printf '%s\\n' '{}'\nread_frame\nprintf '%s\\n' '{shutdown}'\nexit 7",
        batch(row)
    )));
    let mut worker = load_worker(&fixture.plan()).expect("valid init control");
    worker
        .step(&[packet()])
        .expect("valid explicit empty terminal control");
    assert!(worker.shutdown().is_err());
}
