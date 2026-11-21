//! Opt-in per-phase wall-clock accounting (`Workspace::set_profiling`). On a pinned,
//! otherwise idle core wall time per phase approximates CPU time; perf_event is not used.

use std::time::Instant;

/// Phases of one chunk step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    Stem1 = 0,
    Quant,
    GemmF32,
    GemmI8,
    Norm,
    Act,
    Attention,
    Conv,
    Greedy,
}

pub const PHASE_NAMES: [&str; 9] = [
    "stem_layer1",
    "quantize",
    "gemm_f32",
    "gemm_i8",
    "layer_norm",
    "activation_residual",
    "attention",
    "depthwise_conv",
    "greedy_other",
];

#[derive(Debug, Clone, Default)]
pub struct Prof {
    pub on: bool,
    pub ns: [u64; 9],
}

impl Prof {
    #[inline]
    pub fn start(&self) -> Option<Instant> {
        self.on.then(Instant::now)
    }

    #[inline]
    pub fn stop(&mut self, p: Phase, t: Option<Instant>) {
        if let Some(t) = t {
            self.ns[p as usize] += u64::try_from(t.elapsed().as_nanos()).unwrap_or(u64::MAX);
        }
    }

    pub fn add(&mut self, o: &Prof) {
        for (a, b) in self.ns.iter_mut().zip(&o.ns) {
            *a += b;
        }
    }

    /// `(name, seconds)` per phase.
    pub fn seconds(&self) -> Vec<(&'static str, f64)> {
        PHASE_NAMES
            .iter()
            .zip(&self.ns)
            .map(|(&n, &v)| (n, v as f64 * 1e-9))
            .collect()
    }
}
