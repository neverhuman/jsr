//! Bounded 1-in-flight background commit worker and pending-job settlement helpers.

use std::sync::mpsc::{self, Receiver, SyncSender, TryRecvError, TrySendError};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use ultra_engine::commit::{run_commit_pcm, CommitConfig};
use ultra_engine::{Engine, EngineError, Workspace};

use crate::api::{StreamClock, StreamError};
use crate::commit_queue::{FinalizeInput, LiveCommitConfig, LiveCommitState};

pub(crate) struct CommitJobResult {
    pub id: u64,
    pub res: Result<(String, f64), EngineError>,
}

pub(crate) struct PendingCommit {
    pub id: u64,
    pub trigger: &'static str,
    pub seg_start: u64,
    pub seg_end: u64,
    pub provisional: String,
    pub speech_end_ns: Option<u64>,
    pub deadline_ns: u64,
}

struct CommitJob {
    id: u64,
    pcm: Vec<i16>,
}

/// Persistent single-worker bounded queue (capacity 1, at most one commit pass in flight).
pub(crate) struct CommitWorker {
    req_tx: Option<SyncSender<CommitJob>>,
    res_rx: Receiver<CommitJobResult>,
    handle: Option<JoinHandle<()>>,
    busy: bool,
}

impl CommitWorker {
    pub fn spawn(commit_engine: Engine, cfg: CommitConfig) -> Self {
        let (req_tx, req_rx) = mpsc::sync_channel::<CommitJob>(1);
        let (res_tx, res_rx) = mpsc::channel::<CommitJobResult>();
        let handle = thread::Builder::new()
            .name("ultra-commit-worker".into())
            .spawn(move || {
                let mut ws = Workspace::default();
                while let Ok(job) = req_rx.recv() {
                    let res = run_commit_pcm(&commit_engine, cfg, &job.pcm, &mut ws)
                        .map(|(txt, _, ms)| (txt, ms));
                    if res_tx.send(CommitJobResult { id: job.id, res }).is_err() {
                        break;
                    }
                }
            })
            .expect("spawn ultra-commit-worker");
        Self {
            req_tx: Some(req_tx),
            res_rx,
            handle: Some(handle),
            busy: false,
        }
    }

    pub fn is_busy(&self) -> bool {
        self.busy
    }

    pub fn try_submit(&mut self, id: u64, pcm: Vec<i16>) -> bool {
        if self.busy {
            return false;
        }
        let Some(tx) = self.req_tx.as_ref() else {
            return false;
        };
        match tx.try_send(CommitJob { id, pcm }) {
            Ok(()) => {
                self.busy = true;
                true
            }
            Err(TrySendError::Full(_) | TrySendError::Disconnected(_)) => false,
        }
    }

    pub fn try_poll(&mut self) -> Option<CommitJobResult> {
        match self.res_rx.try_recv() {
            Ok(r) => {
                self.busy = false;
                Some(r)
            }
            Err(TryRecvError::Empty | TryRecvError::Disconnected) => None,
        }
    }

    pub fn wait_timeout(&mut self, dur: Duration) -> Option<CommitJobResult> {
        if !self.busy {
            return self.try_poll();
        }
        match self.res_rx.recv_timeout(dur) {
            Ok(r) => {
                self.busy = false;
                Some(r)
            }
            Err(_) => None,
        }
    }

    pub fn drain_stale(&mut self, st: &mut LiveCommitState) {
        while let Some(r) = self.try_poll() {
            if let Ok((_, cw_ms)) = r.res {
                st.metrics.refiner_charged_cpu_s += cw_ms / 1000.0;
            }
            st.metrics.refiner_rejects += 1;
        }
    }

    /// Poll or bounded-wait for `pending` to finish before `deadline_ns`; invokes `on_finalize`
    /// once the active pending job completes or hits its deadline fallback.
    pub fn settle<F>(
        &mut self,
        pending: &mut Option<PendingCommit>,
        cfg: &LiveCommitConfig,
        clock: &StreamClock,
        st: &mut LiveCommitState,
        wait_until_deadline: bool,
        mut on_finalize: F,
    ) -> Result<(), StreamError>
    where
        F: FnMut(&mut LiveCommitState, FinalizeInput<'_>) -> Result<(), StreamError>,
    {
        let mut handle_res = |r: CommitJobResult,
                              pend: &mut Option<PendingCommit>,
                              st: &mut LiveCommitState|
         -> Result<(), StreamError> {
            if pend.as_ref().is_some_and(|p| p.id == r.id) {
                let p = pend.take().expect("pending");
                let (raw, cw_ms) = r.res.map_err(|e| {
                    st.failed = Some(e.to_string());
                    StreamError::from(e)
                })?;
                return on_finalize(
                    st,
                    FinalizeInput {
                        trigger: p.trigger,
                        s0: p.seg_start,
                        s1: p.seg_end,
                        provisional: &p.provisional,
                        raw_committed: &raw,
                        commit_wall_ms: cw_ms,
                        speech_end_ns: p.speech_end_ns,
                        pcm_truncated: false,
                        fallback_reason: "none",
                    },
                );
            }
            if let Ok((_, cw_ms)) = r.res {
                st.metrics.refiner_charged_cpu_s += cw_ms / 1000.0;
            }
            st.metrics.refiner_rejects += 1;
            Ok(())
        };

        while let Some(r) = self.try_poll() {
            handle_res(r, pending, st)?;
        }
        if wait_until_deadline {
            while let Some(p) = pending.as_ref() {
                let now_ns = clock.now_ns();
                if now_ns >= p.deadline_ns {
                    break;
                }
                let cap_ns = u64::from(cfg.deadline_ms.max(800)) * 1_000_000;
                let dur = Duration::from_nanos((p.deadline_ns - now_ns).min(cap_ns));
                match self.wait_timeout(dur) {
                    Some(r) => handle_res(r, pending, st)?,
                    None => break,
                }
            }
        }
        if pending
            .as_ref()
            .is_some_and(|p| wait_until_deadline || clock.now_ns() >= p.deadline_ns)
        {
            let p = pending.take().expect("pending");
            on_finalize(
                st,
                FinalizeInput {
                    trigger: p.trigger,
                    s0: p.seg_start,
                    s1: p.seg_end,
                    provisional: &p.provisional,
                    raw_committed: &p.provisional,
                    commit_wall_ms: 0.0,
                    speech_end_ns: p.speech_end_ns,
                    pcm_truncated: false,
                    fallback_reason: "deadline",
                },
            )?;
        }
        Ok(())
    }
}

impl Drop for CommitWorker {
    fn drop(&mut self) {
        self.req_tx.take();
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Arc;

    #[test]
    fn queued_late_result_after_expiry_is_discarded_and_charged_without_corrupting_next_segment() {
        let (res_tx, res_rx) = mpsc::channel::<CommitJobResult>();
        let mut worker = CommitWorker {
            req_tx: None,
            res_rx,
            handle: None,
            busy: true,
        };
        let now_ns = Arc::new(AtomicU64::new(1_000_000_000));
        let clk_ref = Arc::clone(&now_ns);
        let clock = StreamClock::custom(move || clk_ref.load(Ordering::SeqCst));
        let cfg = LiveCommitConfig {
            deadline_ms: 800,
            ..LiveCommitConfig::default()
        };
        let mut st = LiveCommitState::default();
        let mut out: Vec<(u64, u64, String, &'static str)> = Vec::new();
        let mk_pend = |id, s0, s1, prov: &str, t0| PendingCommit {
            id,
            trigger: "hangover",
            seg_start: s0,
            seg_end: s1,
            provisional: prov.into(),
            speech_end_ns: Some(t0),
            deadline_ns: cfg.deadline_ns(Some(t0), t0),
        };
        // Segment 0 expires past deadline -> immediate greedy fallback.
        let mut p0 = Some(mk_pend(0, 0, 16_000, "seg0 greedy", 1_000_000_000));
        now_ns.store(1_850_000_000, Ordering::SeqCst);
        worker
            .settle(&mut p0, &cfg, &clock, &mut st, false, |_, inp| {
                out.push((
                    inp.s0,
                    inp.s1,
                    inp.raw_committed.to_owned(),
                    inp.fallback_reason,
                ));
                Ok(())
            })
            .expect("settle expiry");
        assert!(p0.is_none() && out == vec![(0, 16_000, "seg0 greedy".into(), "deadline")]);

        // Job 0 finishes LATE (queued in res_rx after expiry) ahead of segment 1 (id=1).
        res_tx
            .send(CommitJobResult {
                id: 0,
                res: Ok(("STALE LATE".into(), 125.0)),
            })
            .expect("tx0");
        res_tx
            .send(CommitJobResult {
                id: 1,
                res: Ok(("seg1 ok".into(), 95.0)),
            })
            .expect("tx1");
        let mut p1 = Some(mk_pend(1, 16_000, 32_000, "seg1 greedy", 2_000_000_000));
        now_ns.store(2_100_000_000, Ordering::SeqCst);
        worker
            .settle(&mut p1, &cfg, &clock, &mut st, false, |_, inp| {
                out.push((
                    inp.s0,
                    inp.s1,
                    inp.raw_committed.to_owned(),
                    inp.fallback_reason,
                ));
                Ok(())
            })
            .expect("settle next");
        assert!(p1.is_none());
        assert_eq!(
            out,
            vec![
                (0, 16_000, "seg0 greedy".into(), "deadline"),
                (16_000, 32_000, "seg1 ok".into(), "none")
            ]
        );
        assert_eq!(st.metrics.refiner_rejects, 1);
        assert!((st.metrics.refiner_charged_cpu_s - 0.125).abs() < 1e-6);
    }
}
