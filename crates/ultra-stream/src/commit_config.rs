//! Bounded 1-in-flight background commit worker and pending-job settlement helpers.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, SyncSender, TryRecvError, TrySendError};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

use ultra_engine::commit::{run_commit_pcm_timed, CommitConfig, CommitTiming};
use ultra_engine::{Engine, EngineError, Workspace};

use crate::api::{StreamClock, StreamError};
use crate::commit_queue::{FinalizeInput, LiveCommitConfig, LiveCommitState};

pub(crate) struct CommitJobResult {
    pub id: u64,
    pub res: Result<(String, CommitTiming), EngineError>,
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
    worker_cpu_ns: Arc<AtomicU64>,
    worker_completed: Arc<AtomicU64>,
    accounted_cpu_ns: u64,
    accounted_completed: u64,
}

impl CommitWorker {
    pub fn spawn(commit_engine: Engine, cfg: CommitConfig) -> Self {
        let (req_tx, req_rx) = mpsc::sync_channel::<CommitJob>(1);
        let (res_tx, res_rx) = mpsc::channel::<CommitJobResult>();
        let worker_cpu_ns = Arc::new(AtomicU64::new(0));
        let worker_completed = Arc::new(AtomicU64::new(0));
        let cpu_acc = Arc::clone(&worker_cpu_ns);
        let done_acc = Arc::clone(&worker_completed);
        let handle = thread::Builder::new()
            .name("ultra-commit-worker".into())
            .spawn(move || {
                let mut ws = Workspace::default();
                while let Ok(job) = req_rx.recv() {
                    let res = run_commit_pcm_timed(&commit_engine, cfg, &job.pcm, &mut ws)
                        .map(|(txt, _, timing)| (txt, timing));
                    if let Ok((_, timing)) = &res {
                        let ns = (timing.cpu_s.max(0.0) * 1e9).round() as u64;
                        cpu_acc.fetch_add(ns, Ordering::Relaxed);
                        done_acc.fetch_add(1, Ordering::Relaxed);
                    }
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
            worker_cpu_ns,
            worker_completed,
            accounted_cpu_ns: 0,
            accounted_completed: 0,
        }
    }

    /// Thread CPU seconds burned by cancelled/timed-out worker jobs that finished asynchronously
    /// and have not yet been reaped through `settle` or `drain_stale`.
    pub fn unreaped_worker_cpu_s(&self) -> f64 {
        let total = self.worker_cpu_ns.load(Ordering::Relaxed);
        total.saturating_sub(self.accounted_cpu_ns) as f64 * 1e-9
    }

    pub fn unreaped_worker_jobs(&self) -> u64 {
        let total = self.worker_completed.load(Ordering::Relaxed);
        total.saturating_sub(self.accounted_completed)
    }

    fn note_accounted(&mut self, timing: CommitTiming) {
        let ns = (timing.cpu_s.max(0.0) * 1e9).round() as u64;
        self.accounted_cpu_ns = self.accounted_cpu_ns.saturating_add(ns);
        self.accounted_completed = self.accounted_completed.saturating_add(1);
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
            if let Ok((_, timing)) = r.res {
                self.note_accounted(timing);
                st.metrics.refiner_charged_cpu_s += timing.cpu_s;
                st.stale_worker_cpu_s += timing.cpu_s;
            }
            st.stale_worker_jobs += 1;
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
                              st: &mut LiveCommitState,
                              accounted_cpu: &mut u64,
                              accounted_done: &mut u64|
         -> Result<(), StreamError> {
            if pend.as_ref().is_some_and(|p| p.id == r.id) {
                let p = pend.take().expect("pending");
                let (raw, timing) = r.res.map_err(|e| {
                    st.failed = Some(e.to_string());
                    StreamError::from(e)
                })?;
                let ns = (timing.cpu_s.max(0.0) * 1e9).round() as u64;
                *accounted_cpu = accounted_cpu.saturating_add(ns);
                *accounted_done = accounted_done.saturating_add(1);
                return on_finalize(
                    st,
                    FinalizeInput {
                        trigger: p.trigger,
                        s0: p.seg_start,
                        s1: p.seg_end,
                        provisional: &p.provisional,
                        raw_committed: &raw,
                        commit_timing: timing,
                        speech_end_ns: p.speech_end_ns,
                        pcm_truncated: false,
                        fallback_reason: "none",
                    },
                );
            }
            if let Ok((_, timing)) = r.res {
                let ns = (timing.cpu_s.max(0.0) * 1e9).round() as u64;
                *accounted_cpu = accounted_cpu.saturating_add(ns);
                *accounted_done = accounted_done.saturating_add(1);
                st.metrics.refiner_charged_cpu_s += timing.cpu_s;
                st.stale_worker_cpu_s += timing.cpu_s;
            }
            st.stale_worker_jobs += 1;
            st.metrics.refiner_rejects += 1;
            Ok(())
        };

        while let Some(r) = self.try_poll() {
            handle_res(
                r,
                pending,
                st,
                &mut self.accounted_cpu_ns,
                &mut self.accounted_completed,
            )?;
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
                    Some(r) => handle_res(
                        r,
                        pending,
                        st,
                        &mut self.accounted_cpu_ns,
                        &mut self.accounted_completed,
                    )?,
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
                    commit_timing: CommitTiming::default(),
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
        let worker_cpu_ns = Arc::new(AtomicU64::new(125_000_000 + 40_000_000));
        let worker_completed = Arc::new(AtomicU64::new(2));
        let mut worker = CommitWorker {
            req_tx: None,
            res_rx,
            handle: None,
            busy: true,
            worker_cpu_ns: Arc::clone(&worker_cpu_ns),
            worker_completed: Arc::clone(&worker_completed),
            accounted_cpu_ns: 0,
            accounted_completed: 0,
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

        // Job 0 finishes LATE with wall_s=0.300 != cpu_s=0.125 (F09: charge CPU, not wall!).
        res_tx
            .send(CommitJobResult {
                id: 0,
                res: Ok((
                    "STALE LATE".into(),
                    CommitTiming {
                        wall_s: 0.300,
                        cpu_s: 0.125,
                    },
                )),
            })
            .expect("tx0");
        res_tx
            .send(CommitJobResult {
                id: 1,
                res: Ok((
                    "seg1 ok".into(),
                    CommitTiming {
                        wall_s: 0.095,
                        cpu_s: 0.040,
                    },
                )),
            })
            .expect("tx1");
        let mut p1 = Some(mk_pend(1, 16_000, 32_000, "seg1 greedy", 2_000_000_000));
        now_ns.store(2_100_000_000, Ordering::SeqCst);
        worker
            .settle(&mut p1, &cfg, &clock, &mut st, false, |_, inp| {
                assert!((inp.commit_timing.wall_s - 0.095).abs() < 1e-9);
                assert!((inp.commit_timing.cpu_s - 0.040).abs() < 1e-9);
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
        assert_eq!(st.stale_worker_jobs, 1);
        assert!((st.stale_worker_cpu_s - 0.125).abs() < 1e-6);
        assert!((st.metrics.refiner_charged_cpu_s - 0.125).abs() < 1e-6);
        assert!(worker.unreaped_worker_cpu_s().abs() < 1e-9);

        // Simulate a cancelled in-flight worker job completing asynchronously after cancel():
        // unreaped worker CPU must remain observable without double-counting prior jobs.
        worker_cpu_ns.fetch_add(60_000_000, Ordering::Relaxed);
        worker_completed.fetch_add(1, Ordering::Relaxed);
        assert_eq!(worker.unreaped_worker_jobs(), 1);
        assert!((worker.unreaped_worker_cpu_s() - 0.060).abs() < 1e-6);
    }
}
