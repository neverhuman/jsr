//! Library backing the `ultra` CLI (spec v1.0-B §35, P08, P09).
//!
//! Deterministic exit codes:
//! - `0`: success
//! - `1`: internal / runtime failure
//! - `2`: invalid configuration or CLI usage
//! - `3`: unavailable backend or missing model/hardware artifact
//! - `4`: failed quality or noninferiority gate
//! - `5`: resource limit / backpressure failure

pub mod bench_cmd;
pub mod inventory;
pub mod runner;

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::os::unix::io::AsRawFd;
use std::path::Path;
use std::process::Command;

use ultra_core::repair::RepairHint;
use ultra_eval::score::ScoreRecord;
use ultra_model::ModelError;
use ultra_stream::StreamError;

pub use bench_cmd::{compare_score_files, CompareVerdictReport};
pub use inventory::collect_hardware_inventory;
pub use runner::{instantiate_backend, resolve_offline_config, transcribe_audio_file};

pub const EXIT_OK: i32 = 0;
pub const EXIT_INTERNAL: i32 = 1;
pub const EXIT_CONFIG: i32 = 2;
pub const EXIT_UNAVAILABLE: i32 = 3;
pub const EXIT_GATE_FAILED: i32 = 4;
pub const EXIT_RESOURCE: i32 = 5;

#[derive(Debug, thiserror::Error)]
pub enum CliError {
    #[error("invalid usage or config: {0}")]
    Config(String),
    #[error("unavailable backend or artifact: {0}")]
    Unavailable(String),
    #[error("quality gate failed: {0}")]
    GateFailed(String),
    #[error("resource limit exceeded: {0}")]
    Resource(String),
    #[error("internal failure: {0}")]
    Internal(String),
}

impl CliError {
    pub fn exit_code(&self) -> i32 {
        match self {
            Self::Config(_) => EXIT_CONFIG,
            Self::Unavailable(_) => EXIT_UNAVAILABLE,
            Self::GateFailed(_) => EXIT_GATE_FAILED,
            Self::Resource(_) => EXIT_RESOURCE,
            Self::Internal(_) => EXIT_INTERNAL,
        }
    }

    pub fn repair(&self) -> RepairHint {
        match self {
            Self::Config(_) => RepairHint {
                purpose: "validate CLI flags, candidate YAML manifests, and bundle schemas",
                reason: "invalid CLI argument, unpinned candidate placeholder, or hash mismatch",
                common_fixes: &[
                    "run ultra --help or ultra candidate validate --config PATH --verify-files",
                ],
                docs_url: "docs/errors.md#cli-config",
                repair_hint: "cargo test -p ultra-cli",
            },
            Self::Unavailable(_) => RepairHint {
                purpose: "fail closed with exit code 3 when model, cache, or ISA is missing",
                reason: "requested bundle/split artifact is absent or requires unsupported ISA",
                common_fixes: &[
                    "run ultra doctor --candidate zipformer-int8 --hardware docs/ultra/HARDWARE.json",
                ],
                docs_url: "docs/errors.md#cli-unavailable",
                repair_hint: "cargo test -p ultra-model",
            },
            Self::GateFailed(_) => RepairHint {
                purpose: "enforce noninferiority margin and CPU-only process verification",
                reason: "paired bootstrap upper bound exceeded margin or GPU library was mapped",
                common_fixes: &[
                    "inspect compare verdict JSON and /proc/self/maps CPU-only check",
                ],
                docs_url: "docs/errors.md#cli-gate",
                repair_hint: "cargo test -p ultra-cli",
            },
            Self::Resource(_) => RepairHint {
                purpose: "serialize timing benchmarks via advisory flock leases",
                reason: "flock acquisition or worker resource cap failed",
                common_fixes: &[
                    "coordinate quiet host window in AGENT_CHAT.md before acquiring BENCH_LEASE.host",
                ],
                docs_url: "docs/testing.md",
                repair_hint: "cargo test -p ultra-cli",
            },
            Self::Internal(_) => RepairHint {
                purpose: "report unexpected I/O, serialisation, or receipt validation errors",
                reason: "filesystem write or receipt evidence check failed",
                common_fixes: &[
                    "check free disk space on /mnt/xbabe3-external and receipt fields",
                ],
                docs_url: "docs/errors.md#receipt",
                repair_hint: "just fast",
            },
        }
    }
}

impl From<std::io::Error> for CliError {
    fn from(e: std::io::Error) -> Self {
        Self::Internal(e.to_string())
    }
}

impl From<ModelError> for CliError {
    fn from(e: ModelError) -> Self {
        match e {
            ModelError::MissingArtifact(_) | ModelError::UnsupportedIsa { .. } => {
                Self::Unavailable(e.to_string())
            }
            _ => Self::Config(e.to_string()),
        }
    }
}

impl From<StreamError> for CliError {
    fn from(e: StreamError) -> Self {
        match e {
            StreamError::BackendUnavailable(_) => Self::Unavailable(e.to_string()),
            StreamError::ResourceLimit(_) => Self::Resource(e.to_string()),
            StreamError::Protocol(_) => Self::Config(e.to_string()),
            _ => Self::Internal(e.to_string()),
        }
    }
}

/// Exclusive `flock` lease guard for serialized H0 benchmark timing.
pub struct BenchLease {
    _file: File,
}

impl BenchLease {
    pub fn acquire(path: &Path) -> Result<Self, CliError> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(false)
            .open(path)?;
        // SAFETY: Valid file descriptor owned by `file`; LOCK_EX acquires exclusive advisory lock.
        let rc = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX) };
        if rc != 0 {
            return Err(CliError::Resource(format!(
                "flock({}) failed",
                path.display()
            )));
        }
        Ok(Self { _file: file })
    }
}

pub fn git_short_commit() -> String {
    let Ok(out) = Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
    else {
        return String::from("unknown");
    };
    if !out.status.success() {
        return String::from("unknown");
    }
    let s = String::from_utf8_lossy(&out.stdout).trim().to_owned();
    if s.is_empty() {
        String::from("unknown")
    } else {
        s
    }
}

pub fn jsr_data_root() -> String {
    match std::env::var("JSR_DATA") {
        Ok(v) if !v.trim().is_empty() => v,
        _ => String::from("/mnt/xbabe3-external/jsr"),
    }
}

pub fn write_score_records_jsonl(path: &Path, rows: &[ScoreRecord]) -> Result<(), CliError> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut f = File::create(path)?;
    for r in rows {
        let line = serde_json::to_string(r).map_err(|e| CliError::Internal(e.to_string()))?;
        writeln!(f, "{line}")?;
    }
    Ok(())
}

pub fn usage_text() -> &'static str {
    "usage: ultra <command> [options]\n\
     commands:\n\
       inventory [--id H0] [--out PATH]\n\
       candidate validate --config PATH [--verify-files]\n\
       bundle verify --bundle DIR [--hardware PATH]\n\
       bundle pack [--source-dir DIR] --out DIR [--precision int8|fp32]\n\
       doctor --candidate ID [--hardware PATH]\n\
       transcribe --candidate ID --audio PATH [--threads N] [--events JSONL] [--receipt JSON]\n\
       stream --candidate ID --input PATH [--profile live|efficient|accurate] [--policy dictate|commit]\n\
              [--packet-ms N] [--pace F] [--threads N] [--events JSONL] [--receipt JSON]\n\
       bench --candidate ID [--split SPLIT | --npz PATH] [--limit N] [--offset N] [--duration-s S]\n\
             [--streams N] [--decode single|batch] [--grid-ms G] [--precision int8|fp32] [--paced]\n\
             [--threads N] [--lease PATH] [--out JSONL] [--scores-out JSONL] [--receipt-out JSON]\n\
       score --predictions JSONL [--md-out PATH] [--out PATH]\n\
       compare --baseline JSONL --candidate JSONL [--alpha A] [--margin-wer-pp M] [--out PATH]"
}

pub fn run_cli_args(args: &[String]) -> Result<(), CliError> {
    let Some((sub, rest)) = args.split_first() else {
        return Err(CliError::Config(usage_text().into()));
    };
    match sub.as_str() {
        "-h" | "--help" | "help" => {
            eprintln!("{}", usage_text());
            Ok(())
        }
        "inventory" => inventory::cmd_inventory(rest),
        "candidate" => inventory::cmd_candidate(rest),
        "bundle" => inventory::cmd_bundle(rest),
        "doctor" => inventory::cmd_doctor(rest),
        "transcribe" => runner::cmd_transcribe(rest, false),
        "stream" => runner::cmd_transcribe(rest, true),
        "sp-a" | "bench" => bench_cmd::cmd_bench(rest),
        "score" => bench_cmd::cmd_score(rest),
        "compare" => bench_cmd::cmd_compare(rest),
        other => Err(CliError::Config(format!(
            "unknown subcommand `{other}`\n{}",
            usage_text()
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ultra_frontend::write_wav_pcm16;
    use ultra_stream::{OperatingProfile, OutputPolicy};

    #[test]
    fn cli_inventory_transcribe_and_compare_work_end_to_end() {
        let inv = collect_hardware_inventory("H0").expect("inventory");
        inv.validate().expect("valid hardware schema");
        assert!(!inv.machines[0].isa.is_empty());

        let tmp = tempfile::tempdir().expect("tempdir");
        let wav = tmp.path().join("silent.wav");
        write_wav_pcm16(&wav, 16_000, &[0i16; 16_000]).expect("wav");

        let (txt, evs, rec) = transcribe_audio_file(
            "fake",
            &wav,
            OperatingProfile::Live,
            OutputPolicy::EditableDictation,
            20,
            0.0,
            1,
        )
        .expect("transcribe fake");
        assert!(txt.is_empty());
        assert!(!evs.is_empty());
        rec.validate_as_evidence().expect("valid receipt");

        let base_j = tmp.path().join("a.jsonl");
        let cand_j = tmp.path().join("b.jsonl");
        let rows: Vec<ScoreRecord> = (0..12)
            .map(|k| ScoreRecord {
                id: format!("u{k}"),
                speaker: format!("spk{}", k % 4),
                dataset: "dev-clean".into(),
                reference: "the quick brown fox jumps".into(),
                hyp: "the quick brown fox jumps".into(),
                duration_s: Some(2.0),
            })
            .collect();
        write_score_records_jsonl(&base_j, &rows).expect("write a");
        write_score_records_jsonl(&cand_j, &rows).expect("write b");
        let v = compare_score_files(&base_j, &cand_j, 0.05, Some(5_000), 0.10).expect("compare");
        assert_eq!(v.verdict, "PASS");
    }
}
