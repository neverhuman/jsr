//! Int8 weights (per-output-channel scales) and per-row dynamic activation quantisation.

use crate::kernels::Isa;

/// Int8 packed weight matrix (filled in by stage E2).
#[derive(Debug, Clone, PartialEq)]
pub struct PackedI8 {
    pub n: usize,
    pub k: usize,
}

impl PackedI8 {
    pub fn from_packed_f32(f: &crate::kernels::PackedF32) -> Self {
        Self { n: f.n, k: f.k }
    }
}

/// Reusable quantisation buffers.
#[derive(Debug, Default)]
pub struct QuantScratch {
    pub a: Vec<i8>,
    pub scale: Vec<f32>,
}

pub fn linear_i8(
    _isa: Isa,
    p: &PackedI8,
    _x: &[f32],
    _rows: usize,
    _out: &mut [f32],
    _q: &mut QuantScratch,
) {
    unreachable!(
        "PackedI8 {}x{} is never constructed before stage E2",
        p.n, p.k
    )
}
