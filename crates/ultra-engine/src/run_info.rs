//! Identities and knobs every eval/density output records, the CPU clock the binaries use,
//! and the qualifying-mode guards (no runtime weight quantisation, no environment knobs).

use serde_json::{json, Value};

use crate::{Model, Precision};

/// Git commit the binary was built from (`build.rs`), and whether tracked files differed.
pub const GIT_COMMIT: &str = env!("ULTRA_ENGINE_GIT_COMMIT");
pub const GIT_DIRTY: &str = env!("ULTRA_ENGINE_GIT_DIRTY");

/// SHA-256 of the running executable.
pub fn exe_sha256() -> String {
    std::env::current_exe()
        .and_then(std::fs::read)
        .map(|b| ultra_core::receipt::sha256_hex(&b))
        .unwrap_or_else(|_| "unavailable".into())
}

/// CPUs this process may run on (`sched_getaffinity`), or `None` where unsupported.
pub fn effective_affinity() -> Option<Vec<usize>> {
    #[cfg(target_os = "linux")]
    {
        use nix::sched::{sched_getaffinity, CpuSet};
        let set = sched_getaffinity(nix::unistd::Pid::from_raw(0)).ok()?;
        Some(
            (0..CpuSet::count())
                .filter(|&c| set.is_set(c).unwrap_or(false))
                .collect(),
        )
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

/// Peak resident set size of this process in MiB (`getrusage` maxrss).
pub fn peak_rss_mib() -> Option<f64> {
    let u = nix::sys::resource::getrusage(nix::sys::resource::UsageWho::RUSAGE_SELF).ok()?;
    let v = u.max_rss() as f64;
    // Linux reports KiB, macOS bytes.
    Some(if cfg!(target_os = "macos") {
        v / (1024.0 * 1024.0)
    } else {
        v / 1024.0
    })
}

/// This thread's CPU run time in seconds. Linux: calling thread's scheduler run time
/// (`/proc/thread-self/schedstat`), falling back to `getrusage(RUSAGE_THREAD)` and then
/// `RUSAGE_SELF`. Elsewhere (macOS): process user + system time.
pub fn thread_cpu_s() -> f64 {
    if let Some(v) = std::fs::read_to_string("/proc/thread-self/schedstat")
        .ok()
        .and_then(|s| {
            s.split_whitespace()
                .next()
                .and_then(|v| v.parse::<f64>().ok())
        })
    {
        return v * 1e-9;
    }
    let tv_s = |u: nix::sys::resource::Usage| {
        let t = |tv: nix::sys::time::TimeVal| tv.tv_sec() as f64 + tv.tv_usec() as f64 * 1e-6;
        t(u.user_time()) + t(u.system_time())
    };
    #[cfg(target_os = "linux")]
    if let Ok(u) = nix::sys::resource::getrusage(nix::sys::resource::UsageWho::RUSAGE_THREAD) {
        return tv_s(u);
    }
    nix::sys::resource::getrusage(nix::sys::resource::UsageWho::RUSAGE_SELF).map_or(f64::NAN, tv_s)
}

/// Non-negative finite thread CPU time in seconds (safe fallback to `0.0` when unavailable).
pub fn thread_cpu_s_or_zero() -> f64 {
    let v = thread_cpu_s();
    if v.is_finite() && v >= 0.0 {
        v
    } else {
        0.0
    }
}

/// Any `ULTRA_ENGINE_*` environment variable. The binaries take every knob as a recorded
/// flag; a leftover variable from an old script is refused rather than silently ignored.
pub fn refuse_engine_env() -> Result<(), String> {
    let set: Vec<String> = std::env::vars_os()
        .filter_map(|(k, _)| k.into_string().ok())
        .filter(|k| k.starts_with("ULTRA_ENGINE_"))
        .collect();
    if set.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "environment knobs are not accepted (use the recorded flags): {}",
            set.join(", ")
        ))
    }
}

/// Qualifying mode: every int8 weight must come from a prepared artifact. Refuses runtime
/// weight quantisation (an fp32 directory run as int8, `--int8-stem`, or an int8-trained
/// stem packed at load from fp32 weights).
pub fn check_qualifying(
    model: &Model,
    precision: Precision,
    int8_stem: bool,
) -> Result<(), String> {
    let prepared = model.int8_prepared.is_some();
    if int8_stem {
        return Err("qualifying mode: --int8-stem quantises weights at run time".into());
    }
    if precision == Precision::Int8 && !prepared {
        return Err(
            "qualifying mode: int8 needs a prepared-INT8 directory (ultra-engine-prepare); \
             runtime weight quantisation is refused"
                .into(),
        );
    }
    if model.arch.qat_stem_int8 && !prepared {
        return Err(
            "qualifying mode: the int8-trained stem would be packed at load; use the \
             prepared-INT8 directory"
                .into(),
        );
    }
    Ok(())
}

/// The identity and knob block shared by `ultra-engine-eval` and `ultra-engine-density`.
pub struct Knobs<'a> {
    pub mode: &'a str,
    pub precision: Precision,
    pub isa: &'a str,
    pub max_chunks: usize,
    pub act7: bool,
    pub gemm_threads: usize,
    pub int8_fp32_kinds: &'a [String],
    pub int8_stem_kinds: &'a [String],
    pub int8_stem_layers_quantised: usize,
    /// Decoder beam width (1 = greedy).
    pub beam: usize,
}

pub fn identity(model: &Model, weights: &std::path::Path, k: &Knobs<'_>) -> Value {
    let qat_stem: Vec<&str> = if model.arch.qat_stem_int8 {
        crate::model::QAT_STEM_KINDS.to_vec()
    } else {
        Vec::new()
    };
    let host = nix::unistd::gethostname()
        .ok()
        .and_then(|h| h.into_string().ok());
    json!({
        "host": host,
        "mode": k.mode,
        "exploratory": k.mode != "qualifying",
        "executable_sha256": exe_sha256(),
        "git_commit": GIT_COMMIT,
        "git_dirty": GIT_DIRTY,
        "isa": k.isa,
        "effective_affinity": effective_affinity(),
        "weights": weights,
        "arch_checkpoint_sha256": model.arch.checkpoint_sha256,
        "weights_sha256": model.arch.weights_sha256,
        "int8_prepared_sha256": model.int8_prepared.as_ref().map(|p| p.sha256.clone()),
        "precision": k.precision.as_str(),
        "max_chunks": k.max_chunks,
        "act7": k.act7,
        "gemm_threads": k.gemm_threads,
        "int8_fp32_kinds": k.int8_fp32_kinds,
        "int8_stem_kinds": k.int8_stem_kinds,
        "int8_stem_layers_quantised": k.int8_stem_layers_quantised,
        "beam": k.beam,
        "qat_int8_stem_kinds": qat_stem,
    })
}
