//! Scalar-vs-SIMD agreement for every kernel (skipped on CPUs without AVX2+FMA).

use super::*;

pub(crate) struct Rng(u64);

impl Rng {
    pub(crate) fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    pub(crate) fn next_u32(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 32) as u32
    }
    /// Uniform in [-scale, scale).
    pub(crate) fn f32(&mut self, scale: f32) -> f32 {
        (self.next_u32() as f32 / 4_294_967_296.0 * 2.0 - 1.0) * scale
    }
    pub(crate) fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| self.f32(scale)).collect()
    }
}

fn simd() -> Option<Isa> {
    Isa::Avx2Fma.is_available().then_some(Isa::Avx2Fma)
}

#[test]
fn gemm_f32_simd_is_bit_exact_with_scalar() {
    let Some(isa) = simd() else { return };
    let mut rng = Rng::new(1);
    for &(rows, n, k) in &[
        (1, 16, 8),
        (7, 33, 19),
        (13, 320, 384),
        (6, 4097, 5),
        (12, 15, 64),
    ] {
        let w = rng.vec(n * k, 1.0);
        let b = rng.vec(n, 1.0);
        let x = rng.vec(rows * k, 2.0);
        let p = PackedF32::pack(&w, n, k, Some(&b));
        let mut o1 = vec![0.0; rows * n];
        let mut o2 = vec![1.0; rows * n];
        gemm_f32(Isa::Scalar, &p, &x, rows, &mut o1);
        gemm_f32(isa, &p, &x, rows, &mut o2);
        assert_eq!(o1, o2, "rows {rows} n {n} k {k}");
        // And the scalar reference is a plain matmul.
        for r in 0..rows {
            for j in 0..n {
                let mut acc = f64::from(b[j]);
                for kk in 0..k {
                    acc += f64::from(x[r * k + kk]) * f64::from(w[j * k + kk]);
                }
                assert!((acc - f64::from(o1[r * n + j])).abs() < 1e-4 * (1.0 + acc.abs()));
            }
        }
    }
}

#[test]
fn exp_and_activations_bit_exact_and_accurate() {
    let mut rng = Rng::new(2);
    let mut x = rng.vec(4099, 30.0);
    x.extend_from_slice(&[0.0, -0.0, 88.0, -88.0, 100.0, -100.0, 1e-8, -87.5]);
    for &v in &x {
        let e = scalar::exp_approx(v);
        let r = f64::from(v.clamp(scalar::EXP_LO, scalar::EXP_HI)).exp();
        if r > 1e-37 {
            assert!(
                ((f64::from(e) - r) / r).abs() < 4e-7,
                "exp({v}) = {e} vs {r}"
            );
        }
        let s = scalar::silu(v);
        let rs = f64::from(v) / (1.0 + (-f64::from(v)).exp());
        assert!(
            (f64::from(s) - rs).abs() <= 1e-6 * (1.0 + rs.abs()),
            "silu({v})"
        );
    }
    let Some(isa) = simd() else { return };
    let mut a = x.clone();
    let mut b = x.clone();
    silu_inplace(Isa::Scalar, &mut a);
    silu_inplace(isa, &mut b);
    assert_eq!(a, b);
    let d = 384;
    let g = rng.vec(4 * 2 * d, 12.0);
    let mut o1 = vec![0.0; 4 * d];
    let mut o2 = vec![0.0; 4 * d];
    glu(Isa::Scalar, &g, d, &mut o1);
    glu(isa, &g, d, &mut o2);
    assert_eq!(o1, o2);
}

#[test]
fn reductions_agree_within_tolerance() {
    let Some(isa) = simd() else { return };
    let mut rng = Rng::new(3);
    let d = 384;
    let x: Vec<f32> = (0..5 * d).map(|i| rng.f32(3.0) + (i % 7) as f32).collect();
    let g = rng.vec(d, 1.0);
    let b = rng.vec(d, 1.0);
    let mut o1 = vec![0.0; x.len()];
    let mut o2 = vec![0.0; x.len()];
    layer_norm(Isa::Scalar, &x, d, &g, &b, &mut o1);
    layer_norm(isa, &x, d, &g, &b, &mut o2);
    for (u, v) in o1.iter().zip(&o2) {
        assert!((u - v).abs() < 2e-5, "{u} {v}");
    }
    for n in [1, 8, 64, 67, 132] {
        let a = rng.vec(n, 2.0);
        let c = rng.vec(n, 2.0);
        let (u, v) = (dot(Isa::Scalar, &a, &c), dot(isa, &a, &c));
        assert!((u - v).abs() < 1e-4 * (1.0 + u.abs()));
        let mut y1 = rng.vec(n, 1.0);
        let mut y2 = y1.clone();
        axpy(Isa::Scalar, 0.37, &a, &mut y1);
        axpy(isa, 0.37, &a, &mut y2);
        for (p, q) in y1.iter().zip(&y2) {
            assert!((p - q).abs() < 1e-6);
        }
    }
}
