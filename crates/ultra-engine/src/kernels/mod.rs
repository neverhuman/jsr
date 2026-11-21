//! Kernels: portable scalar references and runtime-dispatched x86 AVX2/FMA versions.
//!
//! Every SIMD kernel has a scalar twin here with the same accumulation order, and a test
//! that checks them against each other (bit-exact where the arithmetic allows it: the fp32
//! GEMM reference uses `f32::mul_add`, so it matches the FMA kernel bit for bit).
//! `unsafe` lives only in [`x86_avx2`] and is reached only through the safe dispatchers in
//! this module after [`Isa::is_available`] has been checked at construction time.

pub mod scalar;
#[cfg(target_arch = "x86_64")]
mod x86_avx2;

/// Output columns per packed weight panel (two 8-lane AVX registers).
pub const NR: usize = 16;

/// An instruction-set family the engine can run on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Isa {
    /// Portable Rust, the reference for every kernel.
    Scalar,
    /// x86-64 AVX2 + FMA (Zen 2 and later; no AVX-512 or VNNI needed).
    Avx2Fma,
}

impl Isa {
    pub fn is_available(self) -> bool {
        match self {
            Isa::Scalar => true,
            #[cfg(target_arch = "x86_64")]
            Isa::Avx2Fma => {
                std::arch::is_x86_feature_detected!("avx2")
                    && std::arch::is_x86_feature_detected!("fma")
            }
            #[cfg(not(target_arch = "x86_64"))]
            Isa::Avx2Fma => false,
        }
    }

    /// The best available family.
    pub fn detect() -> Isa {
        if Isa::Avx2Fma.is_available() {
            Isa::Avx2Fma
        } else {
            Isa::Scalar
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Isa::Scalar => "scalar",
            Isa::Avx2Fma => "avx2fma",
        }
    }

    /// `self` if this CPU runs it, otherwise the scalar reference.
    pub fn or_scalar(self) -> Isa {
        if self.is_available() {
            self
        } else {
            Isa::Scalar
        }
    }
}

/// An fp32 weight matrix `W (n x k)` with bias, packed in panels of [`NR`] output columns:
/// `data[p][kk][j] = W[p * NR + j][kk]` (zero for padded columns), `bias` padded likewise.
#[derive(Debug, Clone, PartialEq)]
pub struct PackedF32 {
    pub(crate) n: usize,
    pub(crate) k: usize,
    pub(crate) data: Vec<f32>,
    pub(crate) bias: Vec<f32>,
}

impl PackedF32 {
    /// Pack row-major `w` (`n` rows of `k`) and an optional bias of length `n`. Fields are
    /// crate-private, so every `PackedF32` a kernel sees has `data.len() == panels * k * NR`
    /// (sizes checked here without overflow).
    pub fn pack(w: &[f32], n: usize, k: usize, bias: Option<&[f32]>) -> Self {
        assert!(n > 0 && k > 0, "PackedF32::pack empty matrix");
        assert_eq!(
            Some(w.len()),
            n.checked_mul(k),
            "PackedF32::pack weight length"
        );
        let panels = n.div_ceil(NR);
        let len = panels
            .checked_mul(k)
            .and_then(|v| v.checked_mul(NR))
            .expect("PackedF32::pack size overflow");
        let mut data = vec![0.0f32; len];
        for row in 0..n {
            let (p, j) = (row / NR, row % NR);
            for kk in 0..k {
                data[(p * k + kk) * NR + j] = w[row * k + kk];
            }
        }
        let mut b = vec![0.0f32; panels * NR];
        if let Some(bias) = bias {
            assert_eq!(bias.len(), n, "PackedF32::pack bias length");
            b[..n].copy_from_slice(bias);
        }
        Self {
            n,
            k,
            data,
            bias: b,
        }
    }

    pub fn n(&self) -> usize {
        self.n
    }

    pub fn k(&self) -> usize {
        self.k
    }

    pub fn panels(&self) -> usize {
        self.n.div_ceil(NR)
    }

    pub fn panel(&self, p: usize) -> &[f32] {
        &self.data[p * self.k * NR..(p + 1) * self.k * NR]
    }
}

/// `out[r][j] = bias[j] + sum_kk x[r][kk] * W[j][kk]` for `rows` rows (`x` stride `k`,
/// `out` stride `n`).
pub fn gemm_f32(isa: Isa, w: &PackedF32, x: &[f32], rows: usize, out: &mut [f32]) {
    assert!(
        x.len() >= rows * w.k && out.len() >= rows * w.n,
        "gemm_f32 lengths"
    );
    match isa {
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2Fma if isa.is_available() => x86_avx2::gemm_f32(w, x, rows, out),
        _ => scalar::gemm_f32(w, x, rows, out),
    }
}

/// LayerNorm over each `d`-wide row of `x` into `out` (eps 1e-5, biased variance).
pub fn layer_norm(isa: Isa, x: &[f32], d: usize, g: &[f32], b: &[f32], out: &mut [f32]) {
    assert!(
        d > 0 && x.len() == out.len() && x.len().is_multiple_of(d) && g.len() == d && b.len() == d
    );
    match isa {
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2Fma if isa.is_available() => x86_avx2::layer_norm(x, d, g, b, out),
        _ => scalar::layer_norm(x, d, g, b, out),
    }
}

/// `x = x * sigmoid(x)` in place.
pub fn silu_inplace(isa: Isa, x: &mut [f32]) {
    match isa {
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2Fma if isa.is_available() => x86_avx2::silu_inplace(x),
        _ => scalar::silu_inplace(x),
    }
}

/// GLU over rows of width `2 d`: `out[r][c] = a[r][c] * sigmoid(a[r][d + c])`.
pub fn glu(isa: Isa, a: &[f32], d: usize, out: &mut [f32]) {
    assert!(d > 0 && a.len() == 2 * out.len() && out.len().is_multiple_of(d));
    match isa {
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2Fma if isa.is_available() => x86_avx2::glu(a, d, out),
        _ => scalar::glu(a, d, out),
    }
}

/// `sum_i a[i] * b[i]`.
pub fn dot(isa: Isa, a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    match isa {
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2Fma if isa.is_available() => x86_avx2::dot(a, b),
        _ => scalar::dot(a, b),
    }
}

/// `y += alpha * x`.
pub fn axpy(isa: Isa, alpha: f32, x: &[f32], y: &mut [f32]) {
    assert_eq!(x.len(), y.len());
    match isa {
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2Fma if isa.is_available() => x86_avx2::axpy(alpha, x, y),
        _ => scalar::axpy(alpha, x, y),
    }
}

/// `y += alpha * x` element-wise (residual add with a scale).
pub fn add_scaled(isa: Isa, y: &mut [f32], x: &[f32], alpha: f32) {
    axpy(isa, alpha, x, y);
}

/// Softmax attention of one head: queries `q[i * q_stride ..][..dh]` for `i < n`; keys and
/// values are the rows (width `dh`) of two contiguous segments each (a ring buffer's window);
/// writes `out[i * o_stride ..][..dh] = softmax(q k^T * scale) v`. `scores` holds >= nk floats.
#[allow(clippy::too_many_arguments)]
pub fn attend_head(
    isa: Isa,
    q: &[f32],
    q_stride: usize,
    n: usize,
    dh: usize,
    keys: [&[f32]; 2],
    vals: [&[f32]; 2],
    scale: f32,
    scores: &mut [f32],
    out: &mut [f32],
    o_stride: usize,
) {
    assert!(dh > 0, "attend_head dh");
    let nk = (keys[0].len() + keys[1].len()) / dh;
    assert!(n > 0 && nk > 0 && keys[0].len() == vals[0].len() && keys[1].len() == vals[1].len());
    assert!(keys.iter().all(|k| k.len().is_multiple_of(dh)) && scores.len() >= nk);
    assert!(q.len() >= (n - 1) * q_stride + dh && out.len() >= (n - 1) * o_stride + dh);
    match isa {
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2Fma if isa.is_available() && dh.is_multiple_of(8) && dh <= 128 => {
            x86_avx2::attend_head(q, q_stride, n, dh, keys, vals, scale, scores, out, o_stride)
        }
        _ => scalar::attend_head(q, q_stride, n, dh, keys, vals, scale, scores, out, o_stride),
    }
}

/// Per-row centred symmetric int8 quantisation: with `c = (max + min) / 2` and
/// `h = (max - min) / 2`, writes `round_half_even((x - c) * 127 / h)` into `out` and returns
/// `(scale = h / 127, centre = c)`, so `x ~ scale * q + c`. A constant row gives zero codes,
/// scale 0 and centre equal to the constant.
pub fn quantize_row(isa: Isa, x: &[f32], out: &mut [i8]) -> (f32, f32) {
    assert_eq!(x.len(), out.len());
    match isa {
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2Fma if isa.is_available() => x86_avx2::quantize_row(x, out),
        _ => scalar::quantize_row(x, out),
    }
}

/// Int8 GEMM with the dequantising epilogue: `out[r][j] = f32(sum_k a[r][k] w_q[j][k]) *
/// (sa[r] * s_w[j]) + (ca[r] * wsum[j] + b[j])`, where `wsum[j]` is the sum of row `j`'s
/// dequantised weights (the centre's contribution); `a` has row stride `w.kp`, `out` `w.n`.
pub fn gemm_i8(
    isa: Isa,
    w: &crate::quant::PackedI8,
    a: &[i8],
    sa: &[f32],
    ca: &[f32],
    rows: usize,
    out: &mut [f32],
) {
    assert!(
        rows.checked_mul(w.kp).is_some_and(|v| a.len() >= v)
            && rows.checked_mul(w.n).is_some_and(|v| out.len() >= v)
            && sa.len() >= rows
            && ca.len() >= rows,
        "gemm_i8 lengths"
    );
    match isa {
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2Fma if isa.is_available() => x86_avx2::gemm_i8(w, a, sa, ca, rows, out),
        _ => scalar::gemm_i8(w, a, sa, ca, rows, out),
    }
}

#[cfg(test)]
pub(crate) mod tests;
