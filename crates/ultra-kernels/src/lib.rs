//! ULTRA GEMV kernels (spec §5.4, cadence §5.3).
//!
//! Every kernel computes a multi-row GEMV over `F` frames per weight pass:
//! `out[r * F + f] = sum_j w[r][j] * x[f][j]` in int32, for a `rows x din` weight matrix
//! and `F` A7 activation vectors of width `din`. Results are bit-identical to the scalar
//! reference in [`scalar`], which wraps `ultra_core::{numeric, ternary}`.
//!
//! Weight formats: [`I8Matrix`] (int8, rule N1), [`T1Matrix`] (2-bit packed ternary, the
//! layout of the spec's `wstream.c`) and [`TLut2Matrix`] (pairs of trits as 4-bit table
//! indices, row-interleaved for `vpshufb`). Backends are picked by runtime CPU detection
//! ([`Backend::detect`]); `unsafe` lives only in the `x86` modules and is documented there.

mod frames;
mod matrix;
mod par;
pub mod scalar;
mod tlut2;
#[cfg(target_arch = "x86_64")]
mod x86_avx2;
#[cfg(target_arch = "x86_64")]
mod x86_avx512;

pub use frames::Frames;
pub use matrix::{I8Matrix, I8View, T1Matrix, T1View};
pub use par::gemv_par;
pub use tlut2::{TLut2Matrix, TLut2View, TLUT2_BLOCK};

use ultra_core::CoreError;

/// Errors from kernel construction and dispatch.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum KernelError {
    #[error(transparent)]
    Core(#[from] CoreError),
    #[error("backend {0:?} is not available on this CPU")]
    Unavailable(Backend),
    #[error("shape mismatch: {0}")]
    Shape(&'static str),
    #[error("a worker thread panicked")]
    WorkerPanicked,
}

/// A kernel implementation family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Backend {
    /// Portable loops over the ultra-core reference functions.
    Scalar,
    /// x86 AVX2: `vpmaddubsw` + `vpmaddwd` (int8, T1) and `vpshufb` (T-LUT2).
    Avx2,
    /// x86 AVX-512 F/BW/VBMI/VNNI port of Appendix A (`vpdpbusd`). UNTESTED: no host with
    /// these features has run it; T-LUT2 under this backend uses the AVX2 kernel.
    Avx512Vnni,
}

impl Backend {
    /// Every backend, in preference order (best first).
    pub const ALL: [Backend; 3] = [Backend::Avx512Vnni, Backend::Avx2, Backend::Scalar];

    /// Whether this CPU can run the backend.
    pub fn is_available(self) -> bool {
        match self {
            Backend::Scalar => true,
            #[cfg(target_arch = "x86_64")]
            Backend::Avx2 => std::arch::is_x86_feature_detected!("avx2"),
            #[cfg(target_arch = "x86_64")]
            Backend::Avx512Vnni => {
                std::arch::is_x86_feature_detected!("avx2")
                    && std::arch::is_x86_feature_detected!("avx512f")
                    && std::arch::is_x86_feature_detected!("avx512bw")
                    && std::arch::is_x86_feature_detected!("avx512vbmi")
                    && std::arch::is_x86_feature_detected!("avx512vnni")
            }
            #[cfg(not(target_arch = "x86_64"))]
            _ => false,
        }
    }

    /// The best available backend.
    pub fn detect() -> Backend {
        Backend::ALL
            .into_iter()
            .find(|b| b.is_available())
            .unwrap_or(Backend::Scalar)
    }

    /// All backends this CPU can run.
    pub fn available() -> Vec<Backend> {
        Backend::ALL
            .into_iter()
            .filter(|b| b.is_available())
            .collect()
    }

    /// Short lower-case name used in benchmark output.
    pub fn name(self) -> &'static str {
        match self {
            Backend::Scalar => "scalar",
            Backend::Avx2 => "avx2",
            Backend::Avx512Vnni => "avx512vnni",
        }
    }

    fn require(self) -> Result<(), KernelError> {
        if self.is_available() {
            Ok(())
        } else {
            Err(KernelError::Unavailable(self))
        }
    }
}

/// A weight matrix view that a kernel can multiply by a batch of frames.
pub trait Gemv: Copy + Send + Sync {
    /// Output rows.
    fn rows(&self) -> usize;
    /// Input width (weights per row).
    fn din(&self) -> usize;
    /// Row-partition granularity: `slice_rows` boundaries must be multiples of this.
    fn row_align(&self) -> usize;
    /// Rows `start..end` as a view; `start` (and `end`, unless it equals `rows`) must be
    /// multiples of [`Gemv::row_align`].
    fn slice_rows(&self, start: usize, end: usize) -> Result<Self, KernelError>;
    /// `out[r * F + f] = sum_j w[r][j] * x[f][j]`; `out.len()` must be `rows * F`.
    fn gemv(&self, backend: Backend, x: &Frames<'_>, out: &mut [i32]) -> Result<(), KernelError>;
}

/// Shared argument check for every `gemv` entry point.
fn check_call(
    backend: Backend,
    rows: usize,
    din: usize,
    x: &Frames<'_>,
    out: &[i32],
) -> Result<(), KernelError> {
    backend.require()?;
    if x.din() != din {
        return Err(KernelError::Shape("frame width != matrix din"));
    }
    if out.len() != rows * x.count() {
        return Err(KernelError::Shape("out.len() != rows * frames"));
    }
    Ok(())
}

/// Shared row-range check for `slice_rows`.
fn check_slice(start: usize, end: usize, rows: usize, align: usize) -> Result<(), KernelError> {
    let aligned = |v: usize| v.is_multiple_of(align) || v == rows;
    if start > end || end > rows || !start.is_multiple_of(align) || !aligned(end) {
        return Err(KernelError::Shape("row slice out of range or misaligned"));
    }
    Ok(())
}
