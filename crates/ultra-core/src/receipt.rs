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
        if self.schema != RECEIPT_SCHEMA {
            return Err(CoreError::Receipt("wrong schema"));
        }
        if self.completion_status != "complete" {
            return Err(CoreError::Receipt("completion_status is not complete"));
        }
        if !self.cpu_only_verified {
            return Err(CoreError::Receipt("cpu_only_verified is false"));
        }
        let measured = [
            self.observed_threads.is_some(),
            self.process_tree_cpu_seconds.is_some(),
            self.peak_rss_bytes.is_some(),
            self.chunk_ms.p50.is_some(),
            self.chunk_ms.p95.is_some(),
            self.chunk_ms.p99.is_some(),
        ];
        if measured.contains(&false) {
            return Err(CoreError::Receipt("null where a measurement belongs"));
        }
        if self.audio_seconds <= 0.0 {
            return Err(CoreError::Receipt("no audio"));
        }
        Ok(())
    }

    /// Process-tree CPU seconds per audio second.
    pub fn cpu_cost(&self) -> Option<f64> {
        self.process_tree_cpu_seconds
            .map(|c| c / self.audio_seconds)
    }
}

/// Lower-case hex SHA-256.
pub fn sha256_hex(bytes: &[u8]) -> String {
    let d = Sha256::digest(bytes);
    d.iter().map(|b| format!("{b:02x}")).collect()
}
