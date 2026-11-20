//! ULTRA streaming runtime, coordinator, finite-batch scheduler, and native C-ABI engine
//! drivers (spec v1.0-B P02, P03, P05, §13–15, §33, §38.1, §38.3).

pub mod api;
pub mod contrast;
pub mod coordinator;
pub mod delivery;
pub mod fake;
pub mod fleet_inproc;
pub mod grid_sim;
pub mod scheduler;
pub mod sherpa_c_offline;
pub mod sherpa_c_types;
pub mod sherpa_ffi;
pub mod sherpa_offline;
pub mod sherpa_online;

pub use api::{
    AsrStream, Backend, BackendCaps, OperatingProfile, OutputPolicy, PollResult, PushReport,
    PushStatus, StreamConfig, StreamError, StreamMetrics, TerminalOutcome,
};
pub use contrast::{run_offline_contrast, ContrastConfig, ContrastResult};
pub use coordinator::{DeliveryFilter, RefinerTicket, StreamCoordinator};
pub use fake::{FakeBackend, FakeBackendConfig, FakeWord, UninterruptibleWorkerSim};
pub use fleet_inproc::{run_sp_a_shared_workers, SharedFleetResult, WorkerSummary};
pub use scheduler::{
    default_npz_for_split, run_sp_a_zipformer, simulate_grid, sustains_lateness, DecodeMode,
    LatenessQuantiles, SimClock, SimOutcome, SimParams, SimSlotBank, SpAConfig, SpAResult,
    WerSummary, DEFAULT_LATENESS_BOUND_MS,
};
pub use sherpa_ffi::SherpaLib;
pub use sherpa_offline::{OfflineModelKind, SherpaOfflineBackend, SherpaOfflineConfig};
pub use sherpa_online::{RawOnlineStream, SherpaZipformerBackend, SherpaZipformerConfig};
