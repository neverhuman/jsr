//! 0-ULP bit-exact parity tests for Zen 2 speed-bank kernels (`int8_fast`, `ops_fast`, 4-query
//! tiled attention) against their scalar references.

use super::tests::Rng;
use super::*;
use crate::quant::PackedI8;

/// Return the host SIMD ISA to compare against scalar, or `None` on scalar-only hosts so tests
/// never silently compare `Isa::Scalar` against itself.
fn simd_isa_or_skip() -> Option<Isa> {
    if Isa::Avx2Fma.is_available() {
        Some(Isa::Avx2Fma)
    } else if Isa::Neon.is_available() {
        Some(Isa::Neon)
    } else {
        None
    }
}

#[test]
fn quantize_and_gemm_with_abs_are_bit_exact_with_scalar() {
    let Some(isa) = simd_isa_or_skip() else {
        return;
    };
    assert_ne!(isa, Isa::Scalar);
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
fn gemm_i8_with_abs_exhaustive_shapes_and_i16_overflow_boundaries() {
    if !Isa::Avx2Fma.is_available() {
        return;
    }
    let isa = Isa::Avx2Fma;
    let mut rng = Rng::new(139);

    // Tight-slice exhaustive parity across odd, aligned, narrow, and Conformer shapes.
    for &rows in &[1usize, 2, 3, 4, 5, 7, 8, 16] {
        for &n in &[1usize, 2, 3, 4, 5, 7, 8, 16, 64, 384, 501] {
            for &k in &[4usize, 12, 16, 20, 32, 64, 96, 128] {
                let w = rng.vec(n * k, 1.0);
                let b = rng.vec(n, 0.5);
                let p = PackedI8::quantize(&w, n, k, Some(&b));
                let mut a = vec![0i8; rows * p.kp];
                let mut a_abs = vec![0u8; rows * p.kp];
                let mut sa = vec![0.0f32; rows];
                let mut ca = vec![0.0f32; rows];
                let x = rng.vec(rows * k, 3.0);
                for r in 0..rows {
                    let xr = &x[r * k..(r + 1) * k];
                    let (sg, cg) = quantize_row_with_abs(
                        isa,
                        xr,
                        &mut a[r * p.kp..r * p.kp + k],
                        &mut a_abs[r * p.kp..r * p.kp + k],
                        127,
                    );
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
                        "shape mismatch rows={rows} n={n} k={k} idx={i}: {u} vs {v}"
                    );
                }
            }
        }
    }

    // Large Conformer dimensions (M=4 and M=16).
    for &(rows, n, k) in &[
        (4usize, 384usize, 384usize),
        (4, 1536, 384),
        (4, 384, 768),
        (4, 384, 1536),
        (4, 1536, 1536),
        (16, 1536, 384),
        (16, 384, 1536),
    ] {
        let w = rng.vec(n * k, 1.0);
        let b = rng.vec(n, 0.5);
        let p = PackedI8::quantize(&w, n, k, Some(&b));
        let mut a = vec![0i8; rows * p.kp];
        let mut a_abs = vec![0u8; rows * p.kp];
        let mut sa = vec![0.0f32; rows];
        let mut ca = vec![0.0f32; rows];
        let x = rng.vec(rows * k, 3.0);
        for r in 0..rows {
            let (sg, cg) = quantize_row_with_abs(
                isa,
                &x[r * k..(r + 1) * k],
                &mut a[r * p.kp..r * p.kp + k],
                &mut a_abs[r * p.kp..r * p.kp + k],
                127,
            );
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
                "conformer mismatch rows={rows} n={n} k={k} idx={i}: {u} vs {v}"
            );
        }
    }

    // Extreme i8::MIN / i8::MAX and exact i16 overflow-boundary stress tests (sum = 258 vs 259).
    for &(rows, n, k) in &[
        (4usize, 17usize, 64usize),
        (4, 33, 384),
        (3, 19, 48),
        (5, 32, 36),
        (8, 16, 128),
    ] {
        let w = rng.vec(n * k, 1.0);
        let b = rng.vec(n, 0.5);
        let mut p = PackedI8::quantize(&w, n, k, Some(&b));
        // Adversarial weights: every code is +127 or -127 so every product hits worst-case |w|=127.
        for (i, v) in p.data.iter_mut().enumerate() {
            *v = if i % 2 == 0 { 127 } else { -127 };
        }
        p.rebuild_mm();
        let sa = rng.vec(rows, 0.1);
        let ca = rng.vec(rows, 1.0);

        // Patterns:
        // 0: all +127/-127 (SINGLE fallback, maximal single-group sum 32258)
        // 1: all -128 (i8::MIN, |a|=128, single-group sum 32512)
        // 2: exact QUAD boundary sum = 258 (8 values summing to 258 -> 258*127 = 32766)
        // 3: QUAD overflow +1 = 259 (must drop to PAIR; pair sums 130 & 129 <= 258)
        // 4: exact PAIR boundary sum = 258 (4 values summing to 258)
        // 5: PAIR overflow +1 = 259 (must drop to SINGLE)
        // 6: mixed per-row modes (row 0 QUAD, row 1 PAIR, row 2 SINGLE, row 3 all -128)
        for pat in 0..7 {
            let mut a = vec![0i8; rows * p.kp];
            for r in 0..rows {
                let eff_pat = if pat == 6 {
                    match r % 4 {
                        0 => 2,
                        1 => 4,
                        2 => 5,
                        _ => 1,
                    }
                } else {
                    pat
                };
                for kk in 0..p.kp {
                    let sign: i8 = if (kk % 2) == 0 { 1 } else { -1 };
                    let v: i8 = match eff_pat {
                        0 => sign * 127,
                        1 => i8::MIN,
                        2 => {
                            // 8 values per even/odd lane across 16 elements: 6*32 + 2*33 = 258
                            let mag = if kk % 16 >= 12 { 33 } else { 32 };
                            sign * mag
                        }
                        3 => {
                            // 5*32 + 3*33 = 259 (exceeds QUAD=258, safe for PAIR: 129 & 130)
                            let mag = if kk % 16 >= 10 { 33 } else { 32 };
                            sign * mag
                        }
                        4 => {
                            // 4 values per pair: 2*64 + 2*65 = 258 (exact PAIR boundary)
                            let mag = if kk % 8 >= 4 { 65 } else { 64 };
                            sign * mag
                        }
                        _ => {
                            // 1*64 + 3*65 = 259 (exceeds PAIR=258 -> SINGLE)
                            let mag = if kk % 8 >= 2 { 65 } else { 64 };
                            sign * mag
                        }
                    };
                    a[r * p.kp + kk] = v;
                }
            }
            let a_abs: Vec<u8> = a.iter().map(|v| v.unsigned_abs()).collect();
            let mut o_ref = vec![0.0f32; rows * n];
            let mut o_got = vec![0.0f32; rows * n];
            gemm_i8(Isa::Scalar, &p, &a, &sa, &ca, rows, &mut o_ref);
            gemm_i8_with_abs(isa, &p, &a, &a_abs, &sa, &ca, rows, &mut o_got);
            for (i, (&u, &v)) in o_ref.iter().zip(&o_got).enumerate() {
                assert_eq!(
                    u.to_bits(),
                    v.to_bits(),
                    "extreme pat={pat} rows={rows} n={n} k={k} idx={i}: {u} vs {v}"
                );
            }
        }

        // Skewed unsigned-offset paths (U128Pos: mean <= -48, U127Neg: mean >= 48) including
        // exact QUAD/PAIR/SINGLE/SPLIT overflow outliers (+127 and -128 in the same row).
        for skewed_mode in 0..4 {
            let mut a = vec![0i8; rows * p.kp];
            for r in 0..rows {
                for kk in 0..p.kp {
                    let v: i8 = match skewed_mode {
                        // U128Pos all QUAD-safe (`u = a + 128` in 0..=20)
                        0 => -128 + ((kk * 7 + r * 3) % 21) as i8,
                        // U128Pos with heavy positive outliers (+127 -> u=255 -> SPLIT)
                        1 => {
                            if kk % 16 == 0 || kk % 16 == 1 {
                                127
                            } else if kk % 16 == 4 {
                                90
                            } else {
                                -120 + ((kk + r) % 15) as i8
                            }
                        }
                        // U127Neg all QUAD-safe (`u = 127 - a` in 0..=20)
                        2 => 127 - ((kk * 5 + r * 3) % 21) as i8,
                        // U127Neg with heavy negative outliers (i8::MIN -> u=255 -> SPLIT)
                        _ => {
                            if kk % 16 == 2 || kk % 16 == 3 {
                                i8::MIN
                            } else if kk % 16 == 8 {
                                -90
                            } else {
                                118 - ((kk + r) % 15) as i8
                            }
                        }
                    };
                    a[r * p.kp + kk] = v;
                }
            }
            let a_abs: Vec<u8> = a.iter().map(|v| v.unsigned_abs()).collect();
            let mut o_ref = vec![0.0f32; rows * n];
            let mut o_got = vec![0.0f32; rows * n];
            gemm_i8(Isa::Scalar, &p, &a, &sa, &ca, rows, &mut o_ref);
            gemm_i8_with_abs(isa, &p, &a, &a_abs, &sa, &ca, rows, &mut o_got);
            for (i, (&u, &v)) in o_ref.iter().zip(&o_got).enumerate() {
                assert_eq!(
                    u.to_bits(),
                    v.to_bits(),
                    "skewed={skewed_mode} rows={rows} n={n} k={k} idx={i}: {u} vs {v}"
                );
            }
        }
    }
}

#[test]
fn gemm_i8_with_abs_u_mode_plain_cold_path_and_tails_are_bit_exact() {
    if !Isa::Avx2Fma.is_available() {
        return;
    }
    let isa = Isa::Avx2Fma;
    let mut rng = Rng::new(141);

    let check_u_case = |rows: usize, n: usize, k: usize, a: &[i8], label: &str, rng: &mut Rng| {
        let w = rng.vec(n * k, 1.0);
        let b = rng.vec(n, 0.5);
        let mut p = PackedI8::quantize(&w, n, k, Some(&b));
        // Adversarial +-127 weights so every i16 lane product hits worst-case |w| = 127.
        for (i, v) in p.data.iter_mut().enumerate() {
            *v = if i % 2 == 0 { 127 } else { -127 };
        }
        p.rebuild_mm();
        let sa = rng.vec(rows, 0.1);
        let ca = rng.vec(rows, 1.0);
        let a_abs: Vec<u8> = a.iter().map(|v| v.unsigned_abs()).collect();
        let mut o_ref = vec![0.0f32; rows * n];
        let mut o_got = vec![0.0f32; rows * n];
        gemm_i8(Isa::Scalar, &p, a, &sa, &ca, rows, &mut o_ref);
        gemm_i8_with_abs(isa, &p, a, &a_abs, &sa, &ca, rows, &mut o_got);
        for (i, (&u, &v)) in o_ref.iter().zip(&o_got).enumerate() {
            assert_eq!(
                u.to_bits(),
                v.to_bits(),
                "{label} rows={rows} n={n} k={k} idx={i}: {u} vs {v}"
            );
        }
    };

    // 1. U-mode plain cold-path (`label 3:`): each group sum <= 258 (split = false), but cross-group
    //    sum = 130 + 129 = 259 > 258 (also exact boundary 258 + 1 = 259), for both U128Pos and
    //    U127Neg across pair (`rows=4`) and quad (`rows=5`) modes.
    for &(rows, n, k) in &[(4usize, 17usize, 64usize), (5, 33, 64)] {
        for neg in [false, true] {
            let mut u = vec![0u8; rows * k];
            for r in 0..rows {
                let row = &mut u[r * k..(r + 1) * k];
                // Pair/quad 0 even lane: e0 = 130 <= 258, e1 = 129 <= 258 -> e0 + e1 = 259 > 258.
                row[0] = 130;
                row[4] = 129;
                // Pair/quad 1 odd lane (kk=16..24): o0 = 258 (129+129) <= 258, o1 = 1 -> 259 > 258.
                row[18] = 129;
                row[19] = 129;
                row[22] = 1;
                // Keep remaining elements small (0..=8) so mean stays well beyond +-48 skew gate.
                for (kk, slot) in row.iter_mut().enumerate().skip(24) {
                    *slot = ((kk * 3 + r) % 9) as u8;
                }
            }
            let a: Vec<i8> = u
                .iter()
                .map(|&uv| {
                    if !neg {
                        (uv as i16 - 128) as i8
                    } else {
                        (127 - uv as i16) as i8
                    }
                })
                .collect();
            // Sanity check skewness reaches U128Pos (mean <= -48) / U127Neg (mean >= 48).
            let thresh = (k as i32) * 48;
            for r in 0..rows {
                let s: i32 = a[r * k..(r + 1) * k].iter().map(|&v| i32::from(v)).sum();
                if !neg {
                    assert!(s <= -thresh, "expected U128Pos skew: sum={s}");
                } else {
                    assert!(s >= thresh, "expected U127Neg skew: sum={s}");
                }
            }
            check_u_case(
                rows,
                n,
                k,
                &a,
                if neg {
                    "u127neg_plain_cold"
                } else {
                    "u128pos_plain_cold"
                },
                &mut rng,
            );
        }
    }

    // 2. U-quad tails of 1/2/3 groups (`rows=5, k=36/40/44`) and odd group counts for U-pair
    //    (`rows=4, k=36/68`), explicitly reaching U128Pos and U127Neg with both plain-cold and
    //    tail-outlier activations.
    for &(rows, n, k) in &[
        (5usize, 17usize, 36usize),
        (5, 33, 40),
        (5, 501, 44),
        (4, 17, 36),
        (4, 33, 68),
    ] {
        for neg in [false, true] {
            let mut u = vec![0u8; rows * k];
            for r in 0..rows {
                let row = &mut u[r * k..(r + 1) * k];
                // Main body: plain-cold trigger in first pair/quad + small skewed body.
                row[0] = 130;
                row[4] = 129;
                for (kk, slot) in row.iter_mut().enumerate().skip(8) {
                    *slot = ((kk * 5 + r * 3) % 13) as u8;
                }
                // Exercise exact tail group accumulation with a large u value in the tail.
                row[k - 1] = 255;
            }
            let a: Vec<i8> = u
                .iter()
                .map(|&uv| {
                    if !neg {
                        (uv as i16 - 128) as i8
                    } else {
                        (127 - uv as i16) as i8
                    }
                })
                .collect();
            let thresh = (k as i32) * 48;
            for r in 0..rows {
                let s: i32 = a[r * k..(r + 1) * k].iter().map(|&v| i32::from(v)).sum();
                if !neg {
                    assert!(s <= -thresh, "expected U128Pos skew: sum={s}");
                } else {
                    assert!(s >= thresh, "expected U127Neg skew: sum={s}");
                }
            }
            check_u_case(
                rows,
                n,
                k,
                &a,
                if neg { "u127neg_tail" } else { "u128pos_tail" },
                &mut rng,
            );
        }
    }
}

#[test]
fn rope_dwconv_and_stem_c1_are_bit_exact_with_scalar() {
    let Some(isa) = simd_isa_or_skip() else {
        return;
    };
    assert_ne!(isa, Isa::Scalar);
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
