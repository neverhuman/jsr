//! Process-tree CPU measurement, `/proc/<pid>/maps` `libggml-cpu` & GPU inspection, and
//! `-pgid` `ESRCH` drain helpers for `ultra_adapters::nemotron`.

use std::collections::BTreeSet;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::Path;
use std::time::{Duration, Instant};

use nix::sys::resource::{getrusage, UsageWho};
use ultra_core::hash::sha256_reader;

use crate::AdapterError;

pub(crate) fn self_and_child_cpu_s() -> Result<(f64, f64), AdapterError> {
    let to_s = |u: UsageWho| -> Result<f64, AdapterError> {
        let r = getrusage(u).map_err(|e| AdapterError::Process(format!("getrusage: {e}")))?;
        let ut = r.user_time();
        let st = r.system_time();
        Ok((ut.tv_sec() as f64 + ut.tv_usec() as f64 * 1e-6)
            + (st.tv_sec() as f64 + st.tv_usec() as f64 * 1e-6))
    };
    Ok((
        to_s(UsageWho::RUSAGE_SELF)?,
        to_s(UsageWho::RUSAGE_CHILDREN)?,
    ))
}

pub(crate) fn whole_tree_cpu_s() -> Result<f64, AdapterError> {
    let (s, c) = self_and_child_cpu_s()?;
    Ok(s + c)
}

pub(crate) fn hash_file_streaming(path: &Path) -> Result<String, AdapterError> {
    let mut fh = File::open(path)?;
    Ok(sha256_reader(&mut fh)?)
}

/// Inspect `/proc/<pid>/maps`: rejects any GPU shared object and records whether `libggml-cpu`
/// was positively mapped alongside all `.so` library paths.
pub(crate) fn inspect_pid_maps_cpu(
    pid: u32,
    libs_out: &mut BTreeSet<String>,
) -> Option<Result<bool, String>> {
    let maps = std::fs::read_to_string(format!("/proc/{pid}/maps")).ok()?;
    let mut saw_ggml_cpu = false;
    for line in maps.lines() {
        let lower = line.to_ascii_lowercase();
        if lower.contains("libcuda")
            || lower.contains("libcudart")
            || lower.contains("libnvidia-compute")
            || lower.contains("libggml-cuda")
            || lower.contains("libggml-vulkan")
        {
            return Some(Err(format!("GPU library mapped in child {pid}: {line}")));
        }
        if lower.contains("libggml-cpu") {
            saw_ggml_cpu = true;
        }
        if let Some(p) = line.split_whitespace().nth(5) {
            if p.starts_with('/') && (p.ends_with(".so") || p.contains(".so.")) {
                libs_out.insert(p.to_owned());
            }
        }
    }
    Some(Ok(saw_ggml_cpu))
}

pub(crate) fn kill_and_verify_pgid_gone(pgid: i32) -> Result<(), AdapterError> {
    // SAFETY: Negative `pgid` targets the dedicated child process group created by `process_group(0)`.
    unsafe {
        let _ = libc::kill(-pgid, libc::SIGKILL);
    }
    let deadline = Instant::now() + Duration::from_secs(3);
    loop {
        // SAFETY: Signal `0` probes process-group existence without sending a signal.
        let rc = unsafe { libc::kill(-pgid, 0) };
        if rc == -1 && std::io::Error::last_os_error().raw_os_error() == Some(libc::ESRCH) {
            return Ok(());
        }
        if Instant::now() >= deadline {
            return Err(AdapterError::Process(format!(
                "orphan processes still alive in process group {pgid} after SIGKILL"
            )));
        }
        std::thread::sleep(Duration::from_millis(10));
    }
}

pub(crate) fn write_create_new(path: &Path, bytes: &[u8]) -> Result<(), AdapterError> {
    let mut fh = OpenOptions::new().write(true).create_new(true).open(path)?;
    fh.write_all(bytes)?;
    Ok(())
}
