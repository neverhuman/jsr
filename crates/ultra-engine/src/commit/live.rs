//! Helpers supporting non-blocking live-endpointer commit pass workers.

use std::time::Instant;

use super::{ctc_head, drain_stream, CommitConfig, CommitPassSession};
use crate::{Engine, EngineError, Workspace};

/// Snapshotted live-segment payload handed to an async endpoint commit worker.
#[derive(Debug)]
pub struct CommitSnapshot {
    pub provisional: String,
    pub pcm: Vec<i16>,
    pub pcm_truncated: bool,
    pub start_sample: u64,
    pub end_sample: u64,
    pub forced_advances: u64,
}

/// Honest separated wall-clock and thread-CPU timing for a single commit pass (`F09`).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct CommitTiming {
    pub wall_s: f64,
    pub cpu_s: f64,
}

impl CommitTiming {
    #[inline]
    pub fn wall_ms(self) -> f64 {
        self.wall_s * 1000.0
    }

    #[inline]
    pub fn cpu_ms(self) -> f64 {
        self.cpu_s * 1000.0
    }
}

/// Run a standalone endpoint commit pass over buffered PCM using a cloned [`Engine`]
/// (sharing `Arc<Model>` and `Arc<CommitLm>` with the live stream) and a worker-local [`Workspace`],
/// returning both wall-clock and actual calling-thread CPU time.
pub fn run_commit_pcm_timed(
    commit_engine: &Engine,
    cfg: CommitConfig,
    pcm: &[i16],
    ws: &mut Workspace,
) -> Result<(String, usize, CommitTiming), EngineError> {
    let cpu0 = crate::run_info::thread_cpu_s_or_zero();
    let t0 = Instant::now();
    let (raw, idx, _) =
        ctc_head::decode_commit_segment(commit_engine, pcm, cfg, ws, None, None, None, None)?;
    let wall_s = t0.elapsed().as_secs_f64();
    let cpu1 = crate::run_info::thread_cpu_s_or_zero();
    let cpu_s = (cpu1 - cpu0).max(0.0);
    Ok((raw, idx, CommitTiming { wall_s, cpu_s }))
}

/// Run a standalone endpoint commit pass over buffered PCM, returning `(text, hyp_idx, wall_ms)`.
pub fn run_commit_pcm(
    commit_engine: &Engine,
    cfg: CommitConfig,
    pcm: &[i16],
    ws: &mut Workspace,
) -> Result<(String, usize, f64), EngineError> {
    let (raw, idx, timing) = run_commit_pcm_timed(commit_engine, cfg, pcm, ws)?;
    Ok((raw, idx, timing.wall_ms()))
}

impl CommitPassSession {
    /// Flush provisional text, take bounded PCM for an async commit worker, and immediately reset
    /// the live provisional stream so subsequent audio ingestion never stalls.
    pub fn take_commit_snapshot(&mut self) -> Result<CommitSnapshot, EngineError> {
        let (start_sample, end_sample) = self.segment_sample_range();
        let (pcm_truncated, forced_advances) =
            (self.seg.pcm_truncated, self.stream.forced_advances());
        self.stream.input_finished();
        drain_stream(&self.provisional_engine, &mut self.stream, &mut self.ws)?;
        let provisional = self.provisional_engine.text(&self.stream);
        let pcm = std::mem::take(&mut self.seg.pcm_buf);
        self.reset_segment();
        Ok(CommitSnapshot {
            provisional,
            pcm,
            pcm_truncated,
            start_sample,
            end_sample,
            forced_advances,
        })
    }
}
