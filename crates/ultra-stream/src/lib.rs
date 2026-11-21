//! ULTRA streaming runtime, coordinator, finite-batch scheduler, and native C-ABI engine
//! drivers (spec v1.0-B P02, P03, P05, §13–15, §33, §38.1, §38.3).

pub mod api;
pub mod contrast;
pub mod coordinator;
pub mod delivery;
pub mod fake;
pub mod fleet_inproc;
pub mod grid_sim;
pub mod native_engine;
pub mod native_engine_bank;
pub mod native_engine_fleet;
pub mod ort_c_api;
pub mod ort_ffi;
pub mod ort_r1;
pub mod ort_r1_bank;
pub mod ort_r1_fleet;
mod ort_r1_greedy;
pub mod ort_r1_stream;
mod ort_r1_ws;
pub mod placement;
pub mod scheduler;
pub mod sherpa_c_offline;
pub mod sherpa_c_types;
pub mod sherpa_ffi;
pub mod sherpa_offline;
pub mod sherpa_online;
mod x86_avx2_finite;

pub use api::{
    AsrStream, Backend, BackendCaps, OperatingProfile, OutputPolicy, PollResult, PushReport,
    PushStatus, StreamClock, StreamConfig, StreamError, StreamMetrics, TerminalOutcome,
};
pub use contrast::{run_offline_contrast, ContrastConfig, ContrastResult};
pub use coordinator::{DeliveryFilter, RefinerTicket, StreamCoordinator};
pub use fake::{FakeBackend, FakeBackendConfig, FakeWord, UninterruptibleWorkerSim};
pub use fleet_inproc::{
    run_sp_a_shared_workers, run_sp_a_shared_workers_with_observer, SharedFleetResult,
    WorkerSummary,
};
pub use grid_sim::SustainedSample;
pub use native_engine::NativeEngineBackend;
pub use native_engine_bank::{
    run_engine_packet_streaming_with_tokens, run_sp_a_native_engine, EnginePacketStreamingRun,
    EngineSpAConfig, NativeEngineSlotBank,
};
pub use native_engine_fleet::run_sp_a_native_engine_shared_workers;
pub use ort_c_api::{OrtElemType, OrtEnv, OrtRunScratch, OrtSession};
pub use ort_r1::{BatchWorkspace, OrtTransducerBackend, R1PhaseCounters};
pub use ort_r1_bank::{
    run_r1_packet_streaming, run_r1_packet_streaming_with_tokens, run_sp_a_ort_r1, OrtR1SlotBank,
    R1PacketStreamingRun, R1SpAConfig,
};
pub use ort_r1_fleet::{run_sp_a_ort_r1_shared_workers, R1FleetResult};
pub use ort_r1_stream::OrtR1RawStream;
pub use placement::{
    physical_first_cpus, pin_current_thread_verified, verify_worker_l3_and_smt, FleetSeriesPoint,
};
pub use scheduler::{
    default_npz_for_split, run_sp_a_zipformer, simulate_grid, sustains_lateness, DecodeMode,
    LatenessQuantiles, SimClock, SimOutcome, SimParams, SimSlotBank, SpAConfig, SpAResult,
    WerSummary, DEFAULT_LATENESS_BOUND_MS,
};
pub use sherpa_ffi::SherpaLib;
pub use sherpa_offline::{OfflineModelKind, SherpaOfflineBackend, SherpaOfflineConfig};
pub use sherpa_online::{RawOnlineStream, SherpaZipformerBackend, SherpaZipformerConfig};
