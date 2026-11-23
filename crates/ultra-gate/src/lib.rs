//! `ultra-gate`: pure-integer Sentinel-0 wake gate and Linux per-thread `uclamp` pacing.
//!
//! - [`sentinel0`]: `#![forbid(unsafe_code)]`, zero-float pure-integer two-speed wake gate on
//!   160-sample (10 ms) 16 kHz mono PCM16 hops, bit-identical across ISAs.
//! - [`uclamp`]: opt-in Linux per-thread utilisation clamp (`sched_setattr` +
//!   `SCHED_FLAG_UTIL_CLAMP_MAX`), safe no-op on non-Linux targets.

pub mod sentinel0;
pub mod uclamp;

pub use sentinel0::{ilog2_q8, GateConfig, HopOut, Sentinel0, HOP, KS, LOG2_LUT};
pub use uclamp::{
    get_thread_uclamp_max, parse_uclamp_max, set_thread_uclamp_max, UclampError, UCLAMP_MAX_LIMIT,
};
