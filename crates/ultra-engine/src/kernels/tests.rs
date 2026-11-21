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
fn exp_endpoints_and_non_finite_inputs() {
    // Upper endpoint: finite, within the stated error.
    let e = scalar::exp_approx(scalar::EXP_HI);
    let r = f64::from(scalar::EXP_HI).exp();
    assert!(e.is_finite() && ((f64::from(e) - r) / r).abs() < 4e-7);
    // Lower endpoint: results near 2^-126 lose the relative bound (the exponent field
    // flushes subnormals); documented exception, sigmoid/SiLU only need ~0 there.
    for v in [scalar::EXP_LO, f32::NEG_INFINITY, -1e30] {
        let e = scalar::exp_approx(v);
        assert!((0.0..1e-37).contains(&e), "exp({v}) = {e}");
    }
    assert!(scalar::exp_approx(f32::INFINITY).is_finite());
    assert!((0.0..1e-37).contains(&scalar::sigmoid(-200.0)));
    assert!(scalar::silu(-200.0).abs() < 1e-34);
    assert!((scalar::sigmoid(200.0) - 1.0).abs() < 1e-7);
    // NaN never turns into a confident value in the scalar reference.
    assert!(scalar::silu(f32::NAN).is_nan());
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

#[test]
fn quantize_row_simd_is_bit_exact_with_scalar() {
    let Some(isa) = simd() else { return };
    let mut rng = Rng::new(4);
    for n in [1, 7, 31, 32, 33, 384, 1536, 1155] {
        for scale in [1e-3, 1.0, 50.0] {
            let mut x = rng.vec(n, scale);
            if n > 3 {
                // Exact half steps and the extremes.
                let mx = x.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
                x[1] = mx * 0.5 / 127.0;
                x[2] = -mx;
            }
            let (mut a, mut b) = (vec![0i8; n], vec![0i8; n]);
            let s1 = quantize_row(Isa::Scalar, &x, &mut a);
            let s2 = quantize_row(isa, &x, &mut b);
            assert_eq!(
                (s1.0.to_bits(), s1.1.to_bits(), &a),
                (s2.0.to_bits(), s2.1.to_bits(), &b),
                "n {n}"
            );
            assert!(a.iter().all(|&q| q != i8::MIN));
            // Reconstruction within half a step.
            for (&v, &q) in x.iter().zip(&a) {
                assert!(
                    (f32::from(q).mul_add(s1.0, s1.1) - v).abs() <= 0.5001 * s1.0 + 1e-6 * v.abs()
                );
            }
        }
    }
    let mut z = vec![1i8; 40];
    assert_eq!(quantize_row(isa, &[0.0; 40], &mut z), (0.0, 0.0));
    assert!(z.iter().all(|&q| q == 0));
    assert_eq!(quantize_row(isa, &[-2.5; 40], &mut z), (0.0, -2.5));
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        for pos in [0usize, 5, 39] {
            let mut x = vec![0.5f32; 40];
            x[pos] = bad;
            for i in [Isa::Scalar, isa] {
                let (s, c) = quantize_row(i, &x, &mut z);
                assert!(s.is_nan() && c.is_nan(), "{bad} at {pos}");
            }
        }
    }
    // A subnormal range would overflow 127 / half: treated as a constant row in both paths.
    let tiny: Vec<f32> = (0..40)
        .map(|i| if i % 2 == 0 { 1e-44 } else { -1e-44 })
        .collect();
    for i in [Isa::Scalar, isa] {
        let (s, c) = quantize_row(i, &tiny, &mut z);
        assert!(
            s == 0.0 && c.abs() < 1e-37 && z.iter().all(|&q| q == 0),
            "{s} {c}"
        );
    }
    let mut one = [0i8; 1];
    assert_eq!(quantize_row(Isa::Scalar, &[3.0], &mut one), (0.0, 3.0));
}

#[test]
fn gemm_i8_simd_is_bit_exact_including_extremes() {
    use crate::quant::PackedI8;
    let Some(isa) = simd() else { return };
    let mut rng = Rng::new(5);
    for &(rows, n, k) in &[
        (1, 16, 4),
        (5, 33, 30),
        (9, 320, 384),
        (4, 4097, 8),
        (7, 17, 1536),
    ] {
        let w = rng.vec(n * k, 1.0);
        let b = rng.vec(n, 1.0);
        let mut p = PackedI8::quantize(&w, n, k, Some(&b));
        // Adversarial: every weight +-127 with alternating signs and activations +-127 of the
        // same sign pattern -> every pair sums to +-32258 (the vpmaddubsw maximum).
        if n == 17 {
            for (i, v) in p.data.iter_mut().enumerate() {
                *v = if i % 2 == 0 { 127 } else { -127 };
            }
        }
        let mut a = vec![0i8; rows * p.kp];
        for (i, v) in a.iter_mut().enumerate() {
            *v = if n == 17 {
                if i % 2 == 0 {
                    127
                } else {
                    -127
                }
            } else {
                (rng.next_u32() % 255) as i8
            };
            // Full i8 range including -128 (|a| = 128 still never saturates: 2*128*127 < 2^15).
        }
        for r in 0..rows {
            a[r * p.kp + k..(r + 1) * p.kp].fill(0);
        }
        let sa = rng.vec(rows, 0.1);
        let ca = rng.vec(rows, 1.0);
        let mut o1 = vec![0.0; rows * n];
        let mut o2 = vec![1.0; rows * n];
        gemm_i8(Isa::Scalar, &p, &a, &sa, &ca, rows, &mut o1);
        gemm_i8(isa, &p, &a, &sa, &ca, rows, &mut o2);
        assert_eq!(o1, o2, "rows {rows} n {n} k {k}");
        if n == 17 {
            let full = (k as f32) * 127.0 * 127.0;
            let base = ca[0].mul_add(p.wsum[0], p.bias[0]);
            assert_eq!(o1[0], full.mul_add(sa[0] * p.scale[0], base));
        }
    }
}

#[test]
fn attend_head_simd_matches_scalar() {
    let Some(isa) = simd() else { return };
    let mut rng = Rng::new(6);
    // (dh, queries, ring, window start, window length): wrapped and unwrapped windows.
    for &(dh, n, ring, start, nk) in &[
        (64usize, 4usize, 136usize, 100usize, 132usize),
        (64, 1, 136, 0, 3),
        (40, 3, 24, 15, 9),
        (128, 2, 72, 60, 17),
        (64, 5, 144, 7, 137),
    ] {
        let q = rng.vec(n * 3 * dh, 2.0);
        let kt = rng.vec(dh * ring, 2.0);
        let vring = rng.vec(ring * dh, 1.0);
        let first = nk.min(ring - start);
        let segs = [(start, first), (0, nk - first)];
        let vals = [
            &vring[start * dh..(start + first) * dh],
            &vring[..(nk - first) * dh],
        ];
        let scale = 1.0 / (dh as f32).sqrt();
        let (mut s1, mut s2) = (vec![0.0; n * nk], vec![0.0; n * nk]);
        let mut o1 = vec![0.0; n * 2 * dh];
        let mut o2 = vec![0.0; n * 2 * dh];
        // Scores alone are bit-exact (same FMA chain).
        scalar::attention_scores(&q, 3 * dh, n, dh, &kt, ring, segs, scale, &mut s1);
        #[cfg(target_arch = "x86_64")]
        if dh % 8 == 0 {
            x86_avx2::attention_scores(&q, 3 * dh, n, dh, &kt, ring, segs, scale, &mut s2);
            assert_eq!(s1, s2, "scores dh {dh}");
        }
        attend_head(
            Isa::Scalar,
            &q,
            3 * dh,
            n,
            dh,
            &kt,
            ring,
            segs,
            vals,
            scale,
            &mut s1,
            &mut o1,
            2 * dh,
        );
        attend_head(
            isa,
            &q,
            3 * dh,
            n,
            dh,
            &kt,
            ring,
            segs,
            vals,
            scale,
            &mut s2,
            &mut o2,
            2 * dh,
        );
        for (a, b) in o1.iter().zip(&o2) {
            assert!((a - b).abs() < 1e-5, "dh {dh}: {a} {b}");
        }
        // And against a direct f64 softmax-attention.
        for i in 0..n {
            let keys: Vec<usize> = (0..nk).map(|j| (start + j) % ring).collect();
            let sc: Vec<f64> = keys
                .iter()
                .map(|&c| {
                    (0..dh)
                        .map(|e| f64::from(q[i * 3 * dh + e]) * f64::from(kt[kt_index(c, e, dh)]))
                        .sum::<f64>()
                        * f64::from(scale)
                })
                .collect();
            let mx = sc.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let w: Vec<f64> = sc.iter().map(|s| (s - mx).exp()).collect();
            let z: f64 = w.iter().sum();
            for e in 0..dh {
                let r: f64 = keys
                    .iter()
                    .zip(&w)
                    .map(|(&c, p)| p / z * f64::from(vring[c * dh + e]))
                    .sum();
                assert!(
                    (r - f64::from(o1[i * 2 * dh + e])).abs() < 1e-4,
                    "ref dh {dh}"
                );
            }
        }
    }
}
