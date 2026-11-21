//! Optional intra-op split for single-stream latency: one int8 GEMM is divided by weight
//! panels across `threads` persistent workers (the caller computes the first share). Each
//! output value comes from the same tile code as the 1-thread GEMM, so results are
//! bit-identical; only wall time per chunk changes (total CPU rises by the hand-off cost).
//! Density runs keep the default single thread per worker.
//!
//! Safe Rust only: a job owns copies of the (small) quantised activations and an
//! `Arc<Model>`; workers return their column blocks, which the caller scatters.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;

use crate::kernels::{self, Isa, NR};
use crate::model::Model;

struct Job {
    model: Arc<Model>,
    layer: usize,
    a: Vec<i8>,
    sa: Vec<f32>,
    ca: Vec<f32>,
    rows: usize,
    a7: bool,
    isa: Isa,
    ranges: Vec<(usize, usize)>,
}

#[derive(Default)]
struct State {
    epoch: u64,
    job: Option<Arc<Job>>,
    results: Vec<Option<Vec<f32>>>,
    quit: bool,
}

struct Shared {
    state: Mutex<State>,
    work: Condvar,
    /// Published epoch and outstanding shares, polled lock-free (hand-off is per GEMM, so
    /// futex wake-ups would cost more than the work; workers spin briefly, then park).
    epoch: AtomicU64,
    pending: AtomicUsize,
}

/// Spin iterations before a worker parks on the condition variable (~0.1-0.5 ms).
const SPIN: u32 = 200_000;

/// Persistent GEMM helper threads (`threads - 1` of them).
pub struct GemmPool {
    threads: usize,
    shared: Arc<Shared>,
    handles: Vec<JoinHandle<()>>,
}

impl std::fmt::Debug for GemmPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GemmPool")
            .field("threads", &self.threads)
            .finish()
    }
}

fn compute(job: &Job, idx: usize) -> Vec<f32> {
    let lin = job
        .model
        .linear_at(job.layer)
        .expect("layer id from Model::linear_id");
    let p = lin.q.as_ref().expect("int8 layer");
    let (p0, p1) = job.ranges[idx];
    let width = (p1 * NR).min(p.n()) - p0 * NR;
    let mut out = vec![0.0f32; job.rows * width];
    kernels::gemm_i8_panels(
        job.isa, p, &job.a, &job.sa, &job.ca, job.rows, p0, p1, &mut out, job.a7,
    );
    out
}

impl GemmPool {
    /// A pool for `threads` (2..=16) participants: the caller plus `threads - 1` workers.
    pub fn new(threads: usize) -> Self {
        let threads = threads.clamp(2, 16);
        let shared = Arc::new(Shared {
            state: Mutex::new(State::default()),
            work: Condvar::new(),
            epoch: AtomicU64::new(0),
            pending: AtomicUsize::new(0),
        });
        let handles = (1..threads)
            .map(|idx| {
                let sh = shared.clone();
                std::thread::spawn(move || {
                    let mut seen = 0u64;
                    loop {
                        let mut spins = 0u32;
                        while sh.epoch.load(Ordering::Acquire) == seen && spins < SPIN {
                            std::hint::spin_loop();
                            spins += 1;
                        }
                        let job = {
                            let mut st = sh.state.lock().unwrap_or_else(|e| e.into_inner());
                            while !st.quit && st.epoch == seen {
                                st = sh.work.wait(st).unwrap_or_else(|e| e.into_inner());
                            }
                            if st.quit {
                                return;
                            }
                            seen = st.epoch;
                            st.job.clone()
                        };
                        let Some(job) = job else { continue };
                        if idx >= job.ranges.len() {
                            continue;
                        }
                        let r = compute(&job, idx);
                        sh.state.lock().unwrap_or_else(|e| e.into_inner()).results[idx] = Some(r);
                        sh.pending.fetch_sub(1, Ordering::AcqRel);
                    }
                })
            })
            .collect();
        Self {
            threads,
            shared,
            handles,
        }
    }

    pub fn threads(&self) -> usize {
        self.threads
    }

    /// `out[r][j]` for all columns of int8 layer `layer` of `model`, from quantised rows
    /// `a` (stride `kp`), splitting the panels across the pool.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_i8(
        &self,
        isa: Isa,
        model: &Arc<Model>,
        layer: usize,
        a: &[i8],
        sa: &[f32],
        ca: &[f32],
        rows: usize,
        a7: bool,
        out: &mut [f32],
    ) {
        let lin = model.linear_at(layer).expect("layer id");
        let p = lin.q.as_ref().expect("int8 layer");
        let (n, panels) = (p.n(), p.panels());
        let parts = self.threads.min(panels);
        let ranges: Vec<(usize, usize)> = (0..parts)
            .map(|i| (i * panels / parts, (i + 1) * panels / parts))
            .collect();
        let job = Arc::new(Job {
            model: model.clone(),
            layer,
            a: a[..rows * p.kp()].to_vec(),
            sa: sa[..rows].to_vec(),
            ca: ca[..rows].to_vec(),
            rows,
            a7,
            isa,
            ranges: ranges.clone(),
        });
        {
            let mut st = self.shared.state.lock().unwrap_or_else(|e| e.into_inner());
            st.results = vec![None; self.threads];
            st.job = Some(job.clone());
            st.epoch += 1;
            self.shared.pending.store(parts - 1, Ordering::Release);
            self.shared.epoch.store(st.epoch, Ordering::Release);
        }
        self.shared.work.notify_all();
        let mine = compute(&job, 0);
        while self.shared.pending.load(Ordering::Acquire) > 0 {
            std::hint::spin_loop();
        }
        let mut st = self.shared.state.lock().unwrap_or_else(|e| e.into_inner());
        st.job = None;
        let mut results = std::mem::take(&mut st.results);
        drop(st);
        results[0] = Some(mine);
        for (idx, &(p0, p1)) in ranges.iter().enumerate() {
            let block = results[idx].take().expect("every share computed");
            let width = (p1 * NR).min(n) - p0 * NR;
            for r in 0..rows {
                out[r * n + p0 * NR..r * n + p0 * NR + width]
                    .copy_from_slice(&block[r * width..(r + 1) * width]);
            }
        }
    }
}

impl Drop for GemmPool {
    fn drop(&mut self) {
        {
            let mut st = self.shared.state.lock().unwrap_or_else(|e| e.into_inner());
            st.quit = true;
        }
        self.shared.work.notify_all();
        for h in self.handles.drain(..) {
            let _ = h.join();
        }
    }
}
