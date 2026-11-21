//! Reusable contiguous scratch workspace (`BatchWorkspace`), `CLOCK_THREAD_CPUTIME_ID` phase
//! telemetry (`R1PhaseCounters`), and AVX2 finite-output check for [`crate::ort_r1::OrtTransducerBackend`].

use serde::{Deserialize, Serialize};
use ultra_gp::GpError;

use crate::api::StreamError;
use crate::ort_c_api::{OrtRunScratch, OrtTensorIn, OrtTensorOut};
use crate::x86_avx2_finite::all_finite_f32;

#[derive(Debug, Clone)]
pub(crate) enum StateBuf {
    F32(Vec<f32>),
    I64(Vec<i64>),
}

impl StateBuf {
    pub(crate) fn retained_bytes(&self) -> usize {
        match self {
            Self::F32(v) => v.capacity() * 4,
            Self::I64(v) => v.capacity() * 8,
        }
    }
}

/// Optional per-phase thread-CPU counters (`CLOCK_THREAD_CPUTIME_ID`, nanoseconds) for `R1` hot-spot
/// attribution without `/proc` or `perf_event` kernel privileges.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct R1PhaseCounters {
    pub featurize_cpu_ns: u64,
    pub cache_pack_cpu_ns: u64,
    pub encoder_cpu_ns: u64,
    pub finite_validation_cpu_ns: u64,
    pub greedy_cpu_ns: u64,
    pub decoder_cpu_ns: u64,
    pub joiner_cpu_ns: u64,
    pub scatter_cpu_ns: u64,
    pub compatible_groups: u64,
    pub encoder_ort_calls: u64,
    pub decoder_cache_hits: u64,
    pub decoder_ort_calls: u64,
    pub joiner_ort_calls: u64,
    pub zero_copy_single_steps: u64,
    pub batched_steps: u64,
}

#[inline]
pub(crate) fn thread_cpu_ns(enabled: bool) -> u64 {
    if !enabled {
        return 0;
    }
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: `ts` is a valid stack `timespec` initialized above; `CLOCK_THREAD_CPUTIME_ID` is POSIX.
    let rc = unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut ts) };
    if rc == 0 {
        (ts.tv_sec.max(0) as u64)
            .saturating_mul(1_000_000_000)
            .saturating_add(ts.tv_nsec.max(0) as u64)
    } else {
        0
    }
}

#[derive(Default)]
pub(crate) struct BatchWorkspaceInner {
    pub(crate) feats_in: Vec<f32>,
    pub(crate) t0_in: Vec<i64>,
    pub(crate) state_ins: Vec<StateBuf>,
    pub(crate) state_outs: Vec<StateBuf>,
    pub(crate) state_shapes: Vec<Vec<i64>>,
    pub(crate) enc_out: Vec<f32>,
    pub(crate) dec_ctx: Vec<i64>,
    pub(crate) dec_out: Vec<f32>,
    pub(crate) joi_enc: Vec<f32>,
    pub(crate) joi_dec: Vec<f32>,
    pub(crate) joi_logits: Vec<f32>,
    pub(crate) ort_scratch: OrtRunScratch,
    pub(crate) ort_ins_cap: usize,
    pub(crate) profiling_enabled: bool,
    pub(crate) phases: R1PhaseCounters,
}

/// Pointer-sized (`8 B`) handle to reusable contiguous scratch buffers eliminating per-chunk heap
/// allocations across streams while keeping caller structs compact on the stack.
#[derive(Default)]
pub struct BatchWorkspace(pub(crate) Box<BatchWorkspaceInner>);

impl BatchWorkspace {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_profiling(enabled: bool) -> Self {
        Self(Box::new(BatchWorkspaceInner {
            profiling_enabled: enabled,
            ..BatchWorkspaceInner::default()
        }))
    }

    pub fn set_profiling(&mut self, enabled: bool) {
        self.0.profiling_enabled = enabled;
    }

    pub fn phase_counters(&self) -> &R1PhaseCounters {
        &self.0.phases
    }

    /// Total heap bytes retained across workspace scratch buffers, ORT handles, and shape vectors.
    pub fn retained_bytes(&self) -> usize {
        let inner = &*self.0;
        let states_b: usize = inner
            .state_ins
            .iter()
            .chain(&inner.state_outs)
            .map(StateBuf::retained_bytes)
            .sum();
        let shapes_b: usize = inner.state_shapes.capacity() * std::mem::size_of::<Vec<i64>>()
            + inner
                .state_shapes
                .iter()
                .map(|v| v.capacity() * 8)
                .sum::<usize>()
            + (inner.state_ins.capacity() + inner.state_outs.capacity())
                * std::mem::size_of::<StateBuf>();
        std::mem::size_of::<BatchWorkspaceInner>()
            + states_b
            + shapes_b
            + inner.ort_scratch.retained_bytes()
            + inner.ort_ins_cap
                * (std::mem::size_of::<OrtTensorIn<'static>>()
                    + std::mem::size_of::<OrtTensorOut<'static>>())
            + (inner.feats_in.capacity()
                + inner.enc_out.capacity()
                + inner.dec_out.capacity()
                + inner.joi_enc.capacity()
                + inner.joi_dec.capacity()
                + inner.joi_logits.capacity())
                * 4
            + (inner.t0_in.capacity() + inner.dec_ctx.capacity()) * 8
    }
}

pub(crate) fn ensure_all_finite(slice: &[f32], label: &'static str) -> Result<(), StreamError> {
    if all_finite_f32(slice) {
        Ok(())
    } else {
        Err(GpError::Output(format!("non-finite {label} tensor entry")).into())
    }
}
