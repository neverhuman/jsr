//! `ultra-alwayson`: H47 5-state sleep ladder, 3 s PCM16 ring buffer, one-pass batched wake
//! replay, and bit-exact isolated-segment verification for ULTRA always-on ASR.
#![forbid(unsafe_code)]

pub mod backend;
pub mod error;
pub mod mock;
pub mod ring;
pub mod session;
pub mod types;

pub use backend::{AsrBackend, DecodeStep, EngineBackend};
pub use error::AlwaysOnError;
pub use mock::{synthetic_stream_pcm, MockBackend};
pub use ring::{PcmRing, SampleCounters, DEFAULT_RING_SAMPLES, SAMPLE_RATE_HZ};
pub use session::AlwaysOnSession;
pub use types::{
    AlwaysOnConfig, ContinuityGap, LadderState, Segment, SegmentCause, SessionEvent, SessionStats,
};
