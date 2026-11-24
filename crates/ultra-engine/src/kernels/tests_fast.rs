//! 0-ULP bit-exact parity tests for Zen 2 speed-bank kernels (`int8_fast`, `ops_fast`, 4-query
//! tiled attention) against their scalar references.

use super::tests::Rng;
use super::*;
use crate::quant::PackedI8;

#[test]
fn quantize_and_gemm_with_abs_are_bit_exact_with_scalar() {
    let isa = Isa::detect();
    let mut rng = Rng::new(71);
    for &(rows, n, k) in &[
        (1, 16, 4),
        (1, 501, 320),
        (2, 65, 384),
        (3, 33, 30),
        (4, 384, 384),
        (4, 1536, 384),
        (7, 17, 64),
        (16, 384, 384),
    ] {
        let w = rng.vec(n * k, 1.0);
        let b = rng.vec(n, 0.5);
        let x = rng.vec(rows * k, 3.0);
        let p = PackedI8::quantize(&w, n, k, Some(&b));
        let mut a = vec![0i8; rows * p.kp];
        let mut a_abs = vec![0u8; rows * p.kp];
        let mut sa = vec![0.0f32; rows];
        let mut ca = vec![0.0f32; rows];
        for r in 0..rows {
            let xr = &x[r * k..(r + 1) * k];
            let mut a_ref = vec![0i8; k];
            let (sr, cr) = quantize_row(Isa::Scalar, xr, &mut a_ref);
            let (sg, cg) = quantize_row_with_abs(
                isa,
                xr,
                &mut a[r * p.kp..r * p.kp + k],
                &mut a_abs[r * p.kp..r * p.kp + k],
                127,
            );
            assert_eq!((sr.to_bits(), cr.to_bits()), (sg.to_bits(), cg.to_bits()));
            assert_eq!(&a[r * p.kp..r * p.kp + k], &a_ref[..]);
            for (&qv, &av) in a_ref.iter().zip(&a_abs[r * p.kp..r * p.kp + k]) {
                assert_eq!(av, qv.unsigned_abs());
            }
            sa[r] = sg;
            ca[r] = cg;
        }
        let mut o_ref = vec![0.0f32; rows * n];
        let mut o_got = vec![0.0f32; rows * n];
        gemm_i8(Isa::Scalar, &p, &a, &sa, &ca, rows, &mut o_ref);
        gemm_i8_with_abs(isa, &p, &a, &a_abs, &sa, &ca, rows, &mut o_got);
        for (i, (&u, &v)) in o_ref.iter().zip(&o_got).enumerate() {
            assert_eq!(
                u.to_bits(),
                v.to_bits(),
                "mismatch rows={rows} n={n} k={k} idx={i}: {u} vs {v}"
            );
        }
    }
}

#[test]
fn rope_dwconv_and_stem_c1_are_bit_exact_with_scalar() {
    let isa = Isa::detect();
    let mut rng = Rng::new(73);
    // 1. RoPE across dh = 16, 32, 64, 72
    for dh in [16usize, 32, 64, 72] {
        let mut v_ref = rng.vec(dh, 4.0);
        let mut v_got = v_ref.clone();
        let cs = rng.vec(dh, 1.0);
        rope_pair(Isa::Scalar, &mut v_ref, &cs);
        rope_pair(isa, &mut v_got, &cs);
        for (i, (&a, &b)) in v_ref.iter().zip(&v_got).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "rope dh={dh} i={i}: {a} vs {b}");
        }
    }
    // 2. Depthwise conv across d = 72, 384 and kernel = 15
    for d in [72usize, 384] {
        let kernel = 15;
        let km1 = kernel - 1;
        let bias = rng.vec(d, 0.5);
        let hist = rng.vec(km1 * d, 2.0);
        let cur = rng.vec((km1 + 2) * d, 2.0);
        let dw_w = rng.vec(kernel * d, 0.5);
        for fi in [0, 3, km1, km1 + 1] {
            let current = &cur[..(fi + 1) * d];
            let mut o_ref = vec![0.0f32; d];
            let mut o_got = vec![0.0f32; d];
            depthwise_conv_frame(
                Isa::Scalar,
                &mut o_ref,
                &bias,
                &hist,
                current,
                &dw_w,
                fi,
                kernel,
                d,
            );
            depthwise_conv_frame(isa, &mut o_got, &bias, &hist, current, &dw_w, fi, kernel, d);
            for (i, (&a, &b)) in o_ref.iter().zip(&o_got).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "dw d={d} fi={fi} i={i}: {a} vs {b}"
                );
            }
        }
    }
    // 3. Stem layer-1 conv + SiLU across ch = 128, 70
    for ch in [70usize, 128] {
        let bias = rng.vec(ch, 0.5);
        let w = rng.vec(9 * ch, 0.5);
        let mut x9 = [0.0f32; 9];
        for v in &mut x9 {
            *v = rng.f32(3.0);
        }
        let mut o_ref = vec![0.0f32; ch];
        let mut o_got = vec![0.0f32; ch];
        stem_c1_pos(Isa::Scalar, &mut o_ref, &bias, &w, x9, ch);
        stem_c1_pos(isa, &mut o_got, &bias, &w, x9, ch);
        for (i, (&a, &b)) in o_ref.iter().zip(&o_got).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "c1 ch={ch} i={i}: {a} vs {b}");
        }
    }
}
