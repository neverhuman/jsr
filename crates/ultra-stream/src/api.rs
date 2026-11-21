//! Logical Rust streaming boundary (spec v1.0-B §9, §13, §33.1).

use std::sync::Arc;
use std::time::Instant;

use serde::{Deserialize, Serialize};
use ultra_core::events::TranscriptEvent;
use ultra_core::repair::RepairHint;

/// Monotonic observation clock anchored at stream creation (or injected for deterministic tests).
#[derive(Clone)]
pub struct StreamClock {
    inner: ClockKind,
}

#[derive(Clone)]
enum ClockKind {
    Real(Instant),
    Custom(Arc<dyn Fn() -> u64 + Send + Sync>),
}

impl std::fmt::Debug for StreamClock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.inner {
            ClockKind::Real(_) => f.write_str("StreamClock::Real"),
            ClockKind::Custom(_) => f.write_str("StreamClock::Custom"),
        }
    }
}

impl Default for StreamClock {
    fn default() -> Self {
        Self::real()
    }
}

impl StreamClock {
    /// Real monotonic wall clock anchored at the instant this [`StreamClock`] is constructed.
    pub fn real() -> Self {
        Self {
            inner: ClockKind::Real(Instant::now()),
        }
    }

    /// Deterministic test clock returning nanoseconds since stream start.
    pub fn custom<F>(f: F) -> Self
    where
        F: Fn() -> u64 + Send + Sync + 'static,
    {
        Self {
            inner: ClockKind::Custom(Arc::new(f)),
        }
    }

    /// Observed monotonic elapsed nanoseconds (`u64::MAX`-saturated).
    pub fn now_ns(&self) -> u64 {
        match &self.inner {
            ClockKind::Real(t0) => u64::try_from(t0.elapsed().as_nanos()).unwrap_or(u64::MAX),
            ClockKind::Custom(f) => f(),
        }
    }
}

/// Output policy axis (orthogonal to resource profile, spec v1.0-B §13).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutputPolicy {
    EditableDictation,
    Commit,
}

impl OutputPolicy {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::EditableDictation => "dictate",
            Self::Commit => "commit",
        }
    }
}

/// Product operating profile (spec v1.0-B §9).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OperatingProfile {
    Live,
    Efficient,
    Accurate,
}

impl OperatingProfile {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Live => "live",
            Self::Efficient => "efficient",
            Self::Accurate => "accurate",
        }
    }

    /// Primary `CQ(D)` deadline in milliseconds (§12.6).
    pub fn primary_cq_deadline_ms(self) -> u32 {
        match self {
            Self::Live => 750,
            Self::Efficient => 2_000,
            Self::Accurate => 1_500,
        }
    }

    /// Initial finalization p95 deadline after speech end in milliseconds (§9).
    pub fn finalization_deadline_ms(self) -> u64 {
        match self {
            Self::Live => 800,
            Self::Efficient => 1_500,
            Self::Accurate => 1_500,
        }
    }

    /// Initial CPU-seconds per audio-second target (§9).
    pub fn cpu_budget_per_audio_s(self) -> f64 {
        match self {
            Self::Live => 1.0,
            Self::Efficient => 0.25,
            Self::Accurate => 1.0,
        }
    }
}

/// Per-stream configuration bound at `Backend::create_stream`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StreamConfig {
    pub stream_id: String,
    pub model_config_id: String,
    pub policy: OutputPolicy,
    pub profile: OperatingProfile,
    /// Maximum unprocessed audio retained before returning `Backpressure` (samples @ 16 kHz).
    pub max_buffered_samples: usize,
    /// Soft segment cap in 16 kHz samples (default `20 * 16_000` = 20 s, §13.3).
    pub segment_soft_cap_samples: u64,
    /// Hard segment cap in 16 kHz samples (default `30 * 16_000` = 30 s, §13.3).
    pub segment_hard_cap_samples: u64,
    /// Total causal finalization budget from observable speech stop (ms, §18.6, §38.3).
    pub final_budget_ms: u64,
    /// Endpoint hangover duration that consumes from `final_budget_ms` (ms).
    pub endpoint_hangover_ms: u64,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            stream_id: "stream-0".into(),
            model_config_id: "default-model-v1".into(),
            policy: OutputPolicy::EditableDictation,
            profile: OperatingProfile::Live,
            max_buffered_samples: 16_000 * 10, // 10 s queue bound
            segment_soft_cap_samples: 16_000 * 20,
            segment_hard_cap_samples: 16_000 * 30,
            final_budget_ms: 800,
            endpoint_hangover_ms: 300,
        }
    }
}

/// Declared backend capabilities (§33.1, §33.4).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackendCaps {
    pub name: String,
    pub supports_batch_step: bool,
    pub supports_partial_push: bool,
    pub supports_commit_policy: bool,
    pub chunk_samples: usize,
    pub right_context_samples: usize,
    pub cpu_only_verified: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PushStatus {
    AcceptedAll,
    Backpressure { accepted_samples: usize },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PushReport {
    pub start_sample: u64,
    pub accepted_samples: usize,
    pub status: PushStatus,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TerminalOutcome {
    Finished,
    Cancelled,
    Failed(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PollResult {
    Pending,
    Event(Box<TranscriptEvent>),
    Ended(TerminalOutcome),
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct StreamMetrics {
    pub accepted_samples: u64,
    pub processed_samples: u64,
    pub wakeups: u64,
    pub events_emitted: u64,
    pub cap_hits: u64,
    pub refiner_attempts: u64,
    pub refiner_accepts: u64,
    pub refiner_rejects: u64,
    pub refiner_charged_cpu_s: f64,
    pub retained_bytes: u64,
}

/// Errors from the streaming runtime and native engine wrappers.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum StreamError {
    #[error(transparent)]
    Core(#[from] ultra_core::CoreError),
    #[error(transparent)]
    Frontend(#[from] ultra_frontend::FrontendError),
    #[error(transparent)]
    Gp(#[from] ultra_gp::GpError),
    #[error(transparent)]
    Engine(#[from] ultra_engine::EngineError),
    #[error(transparent)]
    Scoring(#[from] ultra_eval::EvalError),
    #[error("protocol error: {0}")]
    Protocol(String),
    #[error("resource limit exceeded: {0}")]
    ResourceLimit(String),
    #[error("backend unavailable: {0}")]
    BackendUnavailable(String),
    #[error("runtime failure: {0}")]
    Runtime(String),
}

impl StreamError {
    /// Structured repair guidance for agents and operators (`docs/errors.md`).
    pub fn repair(&self) -> RepairHint {
        match self {
            Self::Core(e) => e.repair(),
            Self::Gp(e) => e.repair(),
            Self::Engine(e) => e.repair(),
            Self::Frontend(_) => RepairHint {
                purpose: "enforce 16 kHz mono PCM16 / NPZ audio-only boundary (spec §7, §34)",
                reason: "WAV header, resampler input, or NPZ archive failed validation",
                common_fixes: &[
                    "pass mono 16 kHz or 48 kHz PCM16 WAV input",
                    "verify the split .npz cache exists under $JSR_DATA/cache/librispeech_pcm",
                ],
                docs_url: "docs/errors.md#frontend",
                repair_hint: "cargo test -p ultra-frontend",
            },
            Self::Scoring(_) => RepairHint {
                purpose: "compute paired speaker-cluster bootstrap WER and delay metrics",
                reason: "scoring input rows or dataset weights violated ultra-eval preconditions",
                common_fixes: &[
                    "ensure ScoreRecord JSONL has matching utterance ids and non-empty datasets",
                ],
                docs_url: "docs/errors.md#scoring",
                repair_hint: "cargo test -p ultra-eval",
            },
            Self::Protocol(_) => RepairHint {
                purpose: "preserve ordered causal stream delivery and non-overlapping PCM pushes",
                reason: "caller pushed overlapping samples, reused event_sequence, or zero config",
                common_fixes: &[
                    "push monotonic start_sample offsets (gaps emit Body::Gap automatically)",
                    "keep streams, packet_ms, and max_batch strictly positive",
                ],
                docs_url: "docs/errors.md#protocol",
                repair_hint: "cargo test -p ultra-stream --test contract_probes",
            },
            Self::ResourceLimit(_) => RepairHint {
                purpose: "enforce bounded per-stream queues and uninterruptible worker slots",
                reason: "active + draining cancelled workers reached max_workers or buffer cap",
                common_fixes: &[
                    "wait for in-flight cancelled native calls to reap before spawning new workers",
                ],
                docs_url: "docs/errors.md#resourcelimit",
                repair_hint: "cargo test -p ultra-stream --test contract_probes",
            },
            Self::BackendUnavailable(_) => RepairHint {
                purpose: "verify CPU-only native runtime and pinned ONNX artifacts before run",
                reason: "libsherpa-onnx-c-api.so or a required model file was not found",
                common_fixes: &[
                    "set SHERPA_ONNX_LIB_DIR or run just venv so libsherpa-onnx-c-api.so is present",
                    "run ultra bundle verify --bundle DIR",
                ],
                docs_url: "docs/errors.md#backendunavailable",
                repair_hint: "cargo test -p ultra-stream --test sherpa_abi",
            },
            Self::Runtime(_) => RepairHint {
                purpose: "surface native C-API / OS resource measurement failures explicitly",
                reason: "native recognizer returned NULL, worker panicked, or getrusage failed",
                common_fixes: &[
                    "run ultra doctor --candidate zipformer-int8 to verify native engine health",
                ],
                docs_url: "docs/errors.md#runtime",
                repair_hint: "cargo test -p ultra-stream",
            },
        }
    }
}

/// Recognition backend factory (`Send + Sync`, shares immutable weights across streams).
pub trait Backend: Send + Sync {
    fn capabilities(&self) -> BackendCaps;
    fn create_stream(&self, cfg: &StreamConfig) -> Result<Box<dyn AsrStream>, StreamError>;
    fn create_stream_with_clock(
        &self,
        cfg: &StreamConfig,
        _clock: StreamClock,
    ) -> Result<Box<dyn AsrStream>, StreamError> {
        self.create_stream(cfg)
    }
}

/// Per-stream stateful recogniser (`Send`, owned by one coordinator).
pub trait AsrStream: Send {
    fn push_pcm(&mut self, start_sample: u64, pcm: &[i16]) -> Result<PushReport, StreamError>;
    fn poll_event(&mut self) -> Result<PollResult, StreamError>;
    fn finish_input(&mut self) -> Result<(), StreamError>;
    fn cancel(&mut self) -> Result<(), StreamError>;
    fn metrics(&self) -> StreamMetrics;
}
