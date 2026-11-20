//! Bounded, sequenced IPC. A timeout kills/reaps the worker and invalidates the run.

use std::io::{BufRead, BufReader, Read, Write};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc;
use std::thread::JoinHandle;
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::records::{Observation, Plan};
use crate::{require, Result};

pub struct Packet {
    pub id: String,
    pub offset: u64,
    pub pcm: Vec<i16>,
    pub start: bool,
    pub finish: bool,
}

#[derive(Serialize)]
struct Item<'a> {
    id: &'a str,
    offset: u64,
    samples: usize,
    start: bool,
    finish: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Reply {
    run_id: String,
    worker_id: String,
    seq: u64,
    kind: String,
    rows: Vec<Observation>,
    info: Option<serde_json::Value>,
    encoder_calls: u64,
    model_batch_ms: Vec<f64>,
}

type Exchange = (Vec<u8>, mpsc::Sender<std::result::Result<Vec<u8>, String>>);

pub struct Worker {
    child: Child,
    requests: Option<mpsc::Sender<Exchange>>,
    io_thread: Option<JoinHandle<()>>,
    run_id: String,
    seq: u64,
    timeout: Duration,
    pub info: serde_json::Value,
    pub wakeups: u64,
    pub model_batch_ms: Vec<f64>,
    reaped: bool,
}

impl Worker {
    pub fn load(plan: &Plan) -> Result<Self> {
        use std::os::unix::process::CommandExt;
        // Descendants killed on failure must be reaped here, while the outer
        // ultra-measure still accounts for the complete service's resource use.
        // SAFETY: Linux subreaper flag changes only this process's child adoption.
        require(
            unsafe { libc::prctl(libc::PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) } == 0,
            "cannot enable worker orphan drain",
        )?;
        let j = &plan.job;
        let mut command = Command::new(&j.python);
        command
            .args(["-I", "-u"])
            .arg(&j.worker)
            .arg("--bundle")
            .arg(&j.bundle)
            .arg("--precision")
            .arg(&j.precision)
            .arg("--threads")
            .arg(j.threads.to_string())
            .arg("--run-id")
            .arg(&plan.run_id)
            .arg("--manifest-sha256")
            .arg(&plan.bundle_manifest_sha256)
            .env("CUDA_VISIBLE_DEVICES", "")
            .env("OMP_NUM_THREADS", "1")
            .env("OPENBLAS_NUM_THREADS", "1")
            .env("MKL_NUM_THREADS", "1")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .process_group(0);
        let mut child = command.spawn()?;
        let mut input = child.stdin.take().ok_or("worker stdin missing")?;
        let output = child.stdout.take().ok_or("worker stdout missing")?;
        let (tx, rx) = mpsc::channel::<Exchange>();
        let io_thread = std::thread::spawn(move || {
            let mut output = BufReader::new(output);
            for (request, reply) in rx {
                let result = (|| -> std::io::Result<Vec<u8>> {
                    input.write_all(&request)?;
                    input.flush()?;
                    let mut line = Vec::new();
                    output
                        .by_ref()
                        .take(4 * 1024 * 1024)
                        .read_until(b'\n', &mut line)?;
                    if line.last() != Some(&b'\n') {
                        return Err(std::io::Error::other("worker EOF/oversized output"));
                    }
                    Ok(line)
                })()
                .map_err(|e| e.to_string());
                let failed = result.is_err();
                let _ = reply.send(result);
                if failed {
                    break;
                }
            }
        });
        let mut worker = Self {
            child,
            requests: Some(tx),
            io_thread: Some(io_thread),
            run_id: plan.run_id.clone(),
            seq: 0,
            timeout: Duration::from_secs(j.timeout_s),
            info: serde_json::Value::Null,
            wakeups: 0,
            model_batch_ms: Vec::new(),
            reaped: false,
        };
        let reply = worker.exchange("init", &[])?;
        require(reply.rows.is_empty(), "unexpected init output")?;
        worker.info = reply.info.ok_or("missing effective worker settings")?;
        require(
            worker.info["manifest_sha256"] == plan.bundle_manifest_sha256,
            "wrong loaded bundle",
        )?;
        require(
            worker.info["providers"]
                == serde_json::json!([
                    ["CPUExecutionProvider"],
                    ["CPUExecutionProvider"],
                    ["CPUExecutionProvider"]
                ]),
            "CPU-only sessions not verified",
        )?;
        Ok(worker)
    }

    fn exchange(&mut self, kind: &str, packets: &[Packet]) -> Result<Reply> {
        let items: Vec<_> = packets
            .iter()
            .map(|p| Item {
                id: &p.id,
                offset: p.offset,
                samples: p.pcm.len(),
                start: p.start,
                finish: p.finish,
            })
            .collect();
        let header = serde_json::to_vec(&serde_json::json!({"run_id": self.run_id,
            "worker_id": "w0", "seq": self.seq, "kind": kind, "items": items}))?;
        require(header.len() <= 65536, "oversized request header")?;
        let mut bytes = (header.len() as u32).to_le_bytes().to_vec();
        bytes.extend_from_slice(&header);
        for p in packets {
            for s in &p.pcm {
                bytes.extend_from_slice(&s.to_le_bytes());
            }
        }
        let (tx, rx) = mpsc::channel();
        self.requests
            .as_ref()
            .ok_or("worker closed")?
            .send((bytes, tx))?;
        let bytes = rx
            .recv_timeout(self.timeout)
            .map_err(|_| "worker timeout/disconnected")?
            .map_err(std::io::Error::other)?;
        let reply: Reply = serde_json::from_slice(&bytes)?;
        require(
            reply.seq == self.seq
                && reply.run_id == self.run_id
                && reply.worker_id == "w0"
                && reply.kind == kind,
            "stale/wrong-config worker reply",
        )?;
        require(
            reply.encoder_calls >= self.wakeups,
            "encoder counter regressed",
        )?;
        require(
            reply.model_batch_ms.len() as u64 == reply.encoder_calls - self.wakeups
                && reply
                    .model_batch_ms
                    .iter()
                    .all(|v| v.is_finite() && *v >= 0.0),
            "invalid model-batch measurements",
        )?;
        self.model_batch_ms.extend(&reply.model_batch_ms);
        self.wakeups = reply.encoder_calls;
        self.seq += 1;
        Ok(reply)
    }

    pub fn step(&mut self, packets: &[Packet]) -> Result<Vec<Observation>> {
        let reply = self.exchange("batch", packets)?;
        require(reply.info.is_none(), "unexpected configuration change")?;
        crate::service::validate_batch(packets, &reply.rows)?;
        Ok(reply.rows)
    }

    pub fn shutdown(&mut self) -> Result<()> {
        let reply = self.exchange("shutdown", &[])?;
        require(
            reply.rows.is_empty() && reply.info.is_none(),
            "unexpected shutdown output",
        )?;
        // Closing stdin also permits clean EOF handling in a worker that is winding down.
        self.requests.take();
        let until = std::time::Instant::now() + self.timeout;
        loop {
            if let Some(status) = self.child.try_wait()? {
                self.reaped = true;
                require(status.success(), "worker failed after final output")?;
                let mut child_status = 0;
                // SAFETY: checks only adopted children in this worker's group.
                let orphan = unsafe {
                    libc::waitpid(-(self.child.id() as i32), &mut child_status, libc::WNOHANG)
                };
                require(
                    orphan < 0
                        && std::io::Error::last_os_error().raw_os_error() == Some(libc::ECHILD),
                    "worker left unexpected descendants",
                )?;
                break;
            }
            require(std::time::Instant::now() < until, "worker shutdown timeout")?;
            std::thread::sleep(Duration::from_millis(10));
        }
        Ok(())
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        // Dedicated process group includes any accidental descendants. The outer
        // ultra-measure subreaper still waits for orphan drain before issuing a receipt.
        let pid = self.child.id() as i32;
        let drain_until = std::time::Instant::now() + Duration::from_secs(2);
        // SAFETY: negative pid targets only the group created for this worker above.
        if !self.reaped {
            // SAFETY: this is the process group assigned at worker creation.
            unsafe {
                libc::kill(-pid, libc::SIGKILL);
            }
            while std::time::Instant::now() < drain_until {
                match self.child.try_wait() {
                    Ok(Some(_)) | Err(_) => break,
                    Ok(None) => std::thread::sleep(Duration::from_millis(1)),
                }
            }
        }
        // Reap only this dedicated group's adopted descendants, never another worker.
        loop {
            let mut status = 0;
            // SAFETY: status is writable; -pid selects children in this worker's group.
            let result = unsafe { libc::waitpid(-pid, &mut status, libc::WNOHANG) };
            if result == 0 {
                // A live child in this group proves that its group identity is still ours.
                // SAFETY: waitpid just established an owned child in this dedicated group.
                unsafe {
                    libc::kill(-pid, libc::SIGKILL);
                }
                if std::time::Instant::now() >= drain_until {
                    break;
                }
                std::thread::sleep(Duration::from_millis(1));
                continue;
            }
            if result < 0 {
                if std::io::Error::last_os_error().raw_os_error() == Some(libc::EINTR) {
                    continue;
                }
                break;
            }
        }
        self.requests.take();
        if let Some(thread) = self.io_thread.take() {
            while !thread.is_finished() && std::time::Instant::now() < drain_until {
                std::thread::sleep(Duration::from_millis(1));
            }
            if thread.is_finished() {
                let _ = thread.join();
            }
        }
    }
}
