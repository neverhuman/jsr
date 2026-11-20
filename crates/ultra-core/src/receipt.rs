//! Receipts (`ultra.receipt/v1`, Appendix D). A receipt with a null measurement or a
//! `schema_example` status is refused as evidence.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::CoreError;

/// Labels every reported number carries (§11.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum HonestyLabel {
    Fair,
    InCorpus,
    Recall,
    /// Pretrained system whose evaluation-data overlap is not established.
    UnknownOverlap,
}

/// Partitions (§7.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING-KEBAB-CASE")]
pub enum Partition {
    Train,
    Calibration,
    Dev,
    Confirm,
    PublicTest,
    DomainSealed,
    AdaptTest,
    Parity,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ChunkMs {
    pub p50: Option<f64>,
    pub p95: Option<f64>,
    pub p99: Option<f64>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Hashes {
    pub events: String,
    pub preview: String,
    pub commit: String,
    #[serde(rename = "final")]
    pub final_: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Receipt {
    pub schema: String,
    pub bundle: String,
    pub source_commit: String,
    pub hardware: String,
    pub target: String,
    pub configured_threads: u32,
    pub observed_threads: Option<u32>,
    pub cpu_only_verified: bool,
    pub audio_seconds: f64,
    pub process_tree_cpu_seconds: Option<f64>,
    pub peak_rss_bytes: Option<u64>,
    pub events: Option<u64>,
    pub wakeups: Option<u64>,
    pub cap_hits: Option<u64>,
    pub chunk_ms: ChunkMs,
    pub status_events: Vec<String>,
    pub gaps: Vec<(u64, u64)>,
    pub hashes: Hashes,
    pub label: HonestyLabel,
    pub partition: Partition,
    pub completion_status: String,
}

pub const RECEIPT_SCHEMA: &str = "ultra.receipt/v1";

impl Receipt {
    /// Refuse a receipt as evidence if any measurement is missing (Appendix D).
    pub fn validate_as_evidence(&self) -> Result<(), CoreError> {
        let refuse = |m: &'static str| Err(CoreError::Receipt(m));
        if self.schema != RECEIPT_SCHEMA {
            return refuse("wrong schema");
        }
        if self.completion_status != "complete" {
            return refuse("completion_status is not complete");
        }
        if !self.cpu_only_verified {
            return refuse("cpu_only_verified is false");
        }
        let identities = [
            &self.bundle,
            &self.source_commit,
            &self.hardware,
            &self.target,
        ];
        if identities.iter().any(|s| s.trim().is_empty()) {
            return refuse("empty identity (bundle, source_commit, hardware or target)");
        }
        let h = &self.hashes;
        if [&h.events, &h.preview, &h.commit, &h.final_]
            .iter()
            .any(|x| !is_sha256_hex(x))
        {
            return refuse("event hashes must be 64-char lower-case hex SHA-256");
        }
        if self.configured_threads == 0 || self.observed_threads.is_none_or(|t| t == 0) {
            return refuse("thread counts must be measured and positive");
        }
        let counts = [self.events, self.wakeups, self.cap_hits];
        if counts.iter().any(Option::is_none) || self.peak_rss_bytes.is_none_or(|b| b == 0) {
            return refuse("null where a measurement belongs");
        }
        let finite_nonneg = |x: Option<f64>| x.is_some_and(|v| v.is_finite() && v >= 0.0);
        let c = &self.chunk_ms;
        if !finite_nonneg(self.process_tree_cpu_seconds)
            || ![c.p50, c.p95, c.p99].into_iter().all(finite_nonneg)
        {
            return refuse("CPU and chunk timings must be measured, finite and non-negative");
        }
        let (p50, p95, p99) = (
            c.p50.unwrap_or(0.0),
            c.p95.unwrap_or(0.0),
            c.p99.unwrap_or(0.0),
        );
        if !(p50 <= p95 && p95 <= p99) {
            return refuse("chunk percentiles out of order");
        }
        if !self.audio_seconds.is_finite() || self.audio_seconds <= 0.0 {
            return refuse("no audio, or non-finite audio duration");
        }
        if self.gaps.iter().any(|(s, e)| s >= e) {
            return refuse("empty or reversed gap range");
        }
        Ok(())
    }

    /// Process-tree CPU seconds per audio second.
    pub fn cpu_cost(&self) -> Option<f64> {
        self.process_tree_cpu_seconds
            .map(|c| c / self.audio_seconds)
    }
}

/// Is `s` a 64-character lower-case hex string?
pub fn is_sha256_hex(s: &str) -> bool {
    s.len() == 64
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// Lower-case hex SHA-256.
pub fn sha256_hex(bytes: &[u8]) -> String {
    let d = Sha256::digest(bytes);
    d.iter().map(|b| format!("{b:02x}")).collect()
}
