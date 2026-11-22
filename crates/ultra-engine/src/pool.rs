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
    model: Option<Arc<Model>>,
    layer: usize,
    a: Vec<i8>,
    sa: Vec<f32>,
    ca: Vec<f32>,
    rows: usize,
    a7: bool,
    isa: Isa,
    ranges: Vec<(usize, usize)>,
}

impl Default for Job {
    fn default() -> Self {
        Self {
            model: None,
            layer: 0,
            a: Vec::new(),
            sa: Vec::new(),
            ca: Vec::new(),
            rows: 0,
            a7: false,
            isa: Isa::Scalar,
            ranges: Vec::new(),
        }
    }
}

#[derive(Default)]
struct State {
    epoch: u64,
    job: Option<Arc<Job>>,
    quit: bool,
}

struct Shared {
    state: Mutex<State>,
    /// Per-thread reusable output buffers (index 0 = caller, 1..threads = workers).
    /// Each slot is only locked by its owning thread while running and by the caller after
    /// `pending == 0`, avoiding per-GEMM heap allocations and cross-worker lock contention.
    worker_out: Vec<Mutex<Vec<f32>>>,
    /// Recycled `Arc<Job>` kept between calls so `a`/`sa`/`ca`/`ranges` reuse capacity in-place.
    recycle: Mutex<Option<Arc<Job>>>,
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

fn compute_into(job: &Job, idx: usize, out: &mut Vec<f32>) {
    let model = job.model.as_ref().expect("job model populated");
    let lin = model
        .linear_at(job.layer)
        .expect("layer id from Model::linear_id");
    let p = lin.q.as_ref().expect("int8 layer");
    let (p0, p1) = job.ranges[idx];
    let width = (p1 * NR).min(p.n()) - p0 * NR;
    let need = job.rows * width;
    if out.len() < need {
        out.resize(need, 0.0);
    }
    kernels::gemm_i8_panels(
        job.isa,
        p,
        &job.a,
        &job.sa,
        &job.ca,
        job.rows,
        p0,
        p1,
        &mut out[..need],
        job.a7,
    );
}

impl GemmPool {
    /// A pool for `threads` (2..=16) participants: the caller plus `threads - 1` workers.
    pub fn new(threads: usize) -> Self {
        let threads = threads.clamp(2, 16);
        let shared = Arc::new(Shared {
            state: Mutex::new(State::default()),
            worker_out: (0..threads).map(|_| Mutex::new(Vec::new())).collect(),
            recycle: Mutex::new(Some(Arc::new(Job::default()))),
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
                        let participated = idx < job.ranges.len();
                        if participated {
                            let mut buf =
                                sh.worker_out[idx].lock().unwrap_or_else(|e| e.into_inner());
                            compute_into(&job, idx, &mut buf);
                        }
                        // Drop our Arc<Job> reference BEFORE decrementing `pending` so the
                        // caller can reclaim exclusive ownership via `Arc::get_mut` with 0 allocs.
                        drop(job);
                        if participated {
                            sh.pending.fetch_sub(1, Ordering::AcqRel);
                        }
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

        let mut job_arc = self
            .shared
            .recycle
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take()
            .unwrap_or_else(|| Arc::new(Job::default()));
        let job_mut = if let Some(m) = Arc::get_mut(&mut job_arc) {
            m
        } else {
            job_arc = Arc::new(Job::default());
            Arc::get_mut(&mut job_arc).expect("fresh Arc is unique")
        };
        job_mut.model = Some(model.clone());
        job_mut.layer = layer;
        job_mut.a.clear();
        job_mut.a.extend_from_slice(&a[..rows * p.kp()]);
        job_mut.sa.clear();
        job_mut.sa.extend_from_slice(&sa[..rows]);
        job_mut.ca.clear();
        job_mut.ca.extend_from_slice(&ca[..rows]);
        job_mut.rows = rows;
        job_mut.a7 = a7;
        job_mut.isa = isa;
        job_mut.ranges.clear();
        for i in 0..parts {
            job_mut
                .ranges
                .push((i * panels / parts, (i + 1) * panels / parts));
        }

        {
            let mut st = self.shared.state.lock().unwrap_or_else(|e| e.into_inner());
            st.job = Some(job_arc.clone());
            st.epoch += 1;
            self.shared.pending.store(parts - 1, Ordering::Release);
            self.shared.epoch.store(st.epoch, Ordering::Release);
        }
        self.shared.work.notify_all();
        {
            let mut my_buf = self.shared.worker_out[0]
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            compute_into(&job_arc, 0, &mut my_buf);
        }
        while self.shared.pending.load(Ordering::Acquire) > 0 {
            std::hint::spin_loop();
        }
        {
            let mut st = self.shared.state.lock().unwrap_or_else(|e| e.into_inner());
            st.job = None;
        }
        for (idx, &(p0, p1)) in job_arc.ranges.iter().enumerate() {
            let buf = self.shared.worker_out[idx]
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            let width = (p1 * NR).min(n) - p0 * NR;
            for r in 0..rows {
                out[r * n + p0 * NR..r * n + p0 * NR + width]
                    .copy_from_slice(&buf[r * width..(r + 1) * width]);
            }
        }
        *self
            .shared
            .recycle
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = Some(job_arc);
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
