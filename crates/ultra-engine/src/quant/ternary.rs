//! Ternary (R4) layers on AVX2: the same trit codes as ultra-kernels T-LUT2 weights.

use super::{quantize_rows, PackedI8, QuantScratch, KG};
use crate::kernels::{Isa, NR};

/// A ternary layer's codes as ultra-kernels T-LUT2 weights (2 bits per weight), used on
/// AVX2 hosts; the scale / wsum / bias epilogue stays the `PackedI8`'s, and the integer sums
/// are exact, so outputs equal the int8-code path bit for bit.
#[derive(Debug, Clone)]
pub struct PackedT {
    pub(crate) tl: ultra_kernels::TLut2Matrix,
}

impl PackedT {
    /// `Some` on AVX2 hosts (where T-LUT2 has a SIMD kernel), after checking every code is a
    /// trit; `None` elsewhere (the int8-code path is used, e.g. NEON sdot/smmla).
    pub fn from_ternary_codes(p: &PackedI8) -> Result<Option<Self>, crate::EngineError> {
        if p.data.iter().any(|&c| !(-1..=1).contains(&c)) {
            return Err(crate::EngineError::Format(
                "ternary artifact holds a non-trit code".into(),
            ));
        }
        if !ultra_kernels::Backend::Avx2.is_available() {
            return Ok(None);
        }
        let mut trits = vec![0i8; p.n * p.k];
        for row in 0..p.n {
            let (pp, j) = (row / NR, row % NR);
            for kk in 0..p.k {
                trits[row * p.k + kk] =
                    p.data[((pp * (p.kp / KG) + kk / KG) * NR + j) * KG + kk % KG];
            }
        }
        let tl = ultra_kernels::TLut2Matrix::from_trits(p.n, p.k, &trits)
            .map_err(|e| crate::EngineError::Shape(format!("ternary layer: {e}")))?;
        Ok(Some(Self { tl }))
    }
}

/// Ternary linear on T-LUT2: same quantisation and epilogue as [`linear_i8`]
/// (`f32(acc) * (s_a * s_w) + (c * wsum + b)`, fused), integer sums from T-LUT2.
pub fn linear_t(
    isa: Isa,
    p: &PackedI8,
    t: &PackedT,
    x: &[f32],
    rows: usize,
    out: &mut [f32],
    q: &mut QuantScratch,
) {
    use crate::prof::Phase;
    let (n, k) = (p.n, p.k);
    assert!(rows.checked_mul(k).is_some_and(|v| x.len() >= v) && out.len() >= rows * n);
    let tq = q.prof.start();
    let act7 = std::mem::replace(&mut q.act7, false);
    quantize_rows(isa, x, rows, k, k, q);
    q.act7 = act7;
    q.prof.stop(Phase::Quant, tq);
    let tg = q.prof.start();
    if q.acc.len() < rows * n {
        q.acc.resize(rows * n, 0);
    }
    let backend = if matches!(isa, Isa::Avx2Fma | Isa::Avx512Vnni) {
        ultra_kernels::Backend::Avx2
    } else {
        ultra_kernels::Backend::Scalar
    };
    t.tl.view()
        .gemv_signed(backend, &q.a[..rows * k], &mut q.acc[..rows * n])
        .expect("T-LUT2 shapes are fixed at load");
    for r in 0..rows {
        let (sa, ca) = (q.scale[r], q.center[r]);
        for c in 0..n {
            let base = ca.mul_add(p.wsum[c], p.bias[c]);
            out[r * n + c] = (q.acc[c * rows + r] as f32).mul_add(sa * p.scale[c], base);
        }
    }
    q.prof.stop(Phase::GemmI8, tg);
}
