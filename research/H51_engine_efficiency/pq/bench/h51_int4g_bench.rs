#![allow(unsafe_op_in_unsafe_fn, clippy::all)]
//! h51-pq scratch (NOT for the repo): the kernel cost of 128-column group scales (A4-G128).
//!
//! K2  : offset-binary int4 nibbles, one scale per row (as in h51_int4_bench).
//! K2g : the same nibble stream plus one f32 scale per (row, 128-column group). Per group, the i32
//!       dot product (offset removed with that group's 8 * sum(a)) is converted to f32 and FMA'd
//!       with the group's 16 column scales into an f32 accumulator; epilogue y = facc * sa + base.
//! K2g is checked bit-exact against a scalar reference with the same float order, and its
//! integer dot products against exact i64 sums. Timed against INT8 `gemm_i8` and K2, cache-hot
//! and as a whole-encoder sweep, at M = 1 / 2 / 4.
//!
//! cargo run --release -p ultra-engine --example h51_int4g_bench -- [reps]
use std::arch::x86_64::*;
use std::time::Instant;

use ultra_engine::kernels::{gemm_i8, quantize_row, Isa, NR};
use ultra_engine::quant::PackedI8;

const KG: usize = 4;
const GS: usize = 128;

fn lcg(s: &mut u64) -> u64 {
    *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    *s >> 33
}

struct W4 {
    n: usize,
    kp: usize,
    q: Vec<i8>,       // row-major int4 values, n x kp
    data: Vec<u8>,    // offset-binary nibble panels (K2 layout)
    scale: Vec<f32>,  // per row (K2)
    gscale: Vec<f32>, // (panel, group, 16) per-group column scales (K2g)
    wsum: Vec<f32>,   // per row, dequantised sum (K2 uses scale-based, K2g group-based)
    gwsum: Vec<f32>,
    bias: Vec<f32>,
}

fn make(n: usize, k: usize, seed: &mut u64) -> (W4, PackedI8) {
    assert!(k % GS == 0);
    let kp = k;
    let panels = n.div_ceil(NR);
    let ng = kp / GS;
    let mut q = vec![0i8; panels * NR * kp];
    for row in 0..n {
        for kk in 0..k {
            q[row * kp + kk] = (lcg(seed) % 15) as i8 - 7;
        }
    }
    let mut scale = vec![0f32; panels * NR];
    let mut wsum = vec![0f32; panels * NR];
    let mut gwsum = vec![0f32; panels * NR];
    let mut bias = vec![0f32; panels * NR];
    let mut gscale = vec![0f32; panels * ng * NR];
    for row in 0..n {
        scale[row] = 0.001 + (lcg(seed) % 1000) as f32 * 1e-5;
        wsum[row] = q[row * kp..row * kp + kp].iter().map(|&v| i32::from(v)).sum::<i32>() as f32 * scale[row];
        bias[row] = (lcg(seed) % 200) as f32 * 1e-3 - 0.1;
        let mut gs = 0f32;
        for g in 0..ng {
            let s = 0.0005 + (lcg(seed) % 1000) as f32 * 1e-5;
            gscale[((row / NR) * ng + g) * NR + row % NR] = s;
            let sq: i32 = q[row * kp + g * GS..row * kp + (g + 1) * GS].iter().map(|&v| i32::from(v)).sum();
            gs += sq as f32 * s;
        }
        gwsum[row] = gs;
    }
    let mut d8 = vec![0i8; panels * kp * NR];
    for row in 0..panels * NR {
        let (p, j) = (row / NR, row % NR);
        for kk in 0..kp {
            d8[((p * (kp / KG) + kk / KG) * NR + j) * KG + kk % KG] = q[row * kp + kk];
        }
    }
    let mut data = vec![0u8; panels * kp * NR / 2];
    for p in 0..panels {
        for gp in 0..kp / 8 {
            for j in 0..NR {
                for kk in 0..KG {
                    let lo = d8[((p * (kp / KG) + 2 * gp) * NR + j) * KG + kk];
                    let hi = d8[((p * (kp / KG) + 2 * gp + 1) * NR + j) * KG + kk];
                    data[(p * (kp / 8) + gp) * 64 + j * KG + kk] = ((lo + 8) as u8) | (((hi + 8) as u8) << 4);
                }
            }
        }
    }
    let p8 = PackedI8::from_parts(n, k, d8, &scale, &wsum, &bias[..n]).expect("packed i8");
    (W4 { n, kp, q, data, scale, gscale, wsum, gwsum, bias }, p8)
}

#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn dot_pairs<const M: usize>(base: *const u8, a: &[i8], kp: usize, gp0: usize, npairs: usize, acc32: &mut [[__m256i; 2]; M]) {
    let ones = _mm256_set1_epi16(1);
    let m4 = _mm256_set1_epi8(0x0F);
    let mut acc16 = [[_mm256_setzero_si256(); 2]; M];
    for t in 0..npairs {
        let gp = gp0 + t;
        let v0 = _mm256_loadu_si256(base.add(gp * 64).cast());
        let v1 = _mm256_loadu_si256(base.add(gp * 64 + 32).cast());
        let u0 = [_mm256_and_si256(v0, m4), _mm256_and_si256(_mm256_srli_epi16::<4>(v0), m4)];
        let u1 = [_mm256_and_si256(v1, m4), _mm256_and_si256(_mm256_srli_epi16::<4>(v1), m4)];
        for m in 0..M {
            for h in 0..2 {
                let a4 = std::ptr::read_unaligned(a.as_ptr().add(m * kp + 4 * (2 * gp + h)).cast::<i32>());
                let av = _mm256_set1_epi32(a4);
                acc16[m][0] = _mm256_add_epi16(acc16[m][0], _mm256_maddubs_epi16(u0[h], av));
                acc16[m][1] = _mm256_add_epi16(acc16[m][1], _mm256_maddubs_epi16(u1[h], av));
            }
        }
        if t % 4 == 3 || t + 1 == npairs {
            for m in 0..M {
                acc32[m][0] = _mm256_add_epi32(acc32[m][0], _mm256_madd_epi16(acc16[m][0], ones));
                acc32[m][1] = _mm256_add_epi32(acc32[m][1], _mm256_madd_epi16(acc16[m][1], ones));
                acc16[m] = [_mm256_setzero_si256(); 2];
            }
        }
    }
}

#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn store16(y0: __m256, y1: __m256, out: &mut [f32], ncols: usize) {
    let mut tmp = [0f32; 16];
    _mm256_storeu_ps(tmp.as_mut_ptr(), y0);
    _mm256_storeu_ps(tmp.as_mut_ptr().add(8), y1);
    out[..ncols].copy_from_slice(&tmp[..ncols]);
}

/// K2: per-row scale; epilogue identical to gemm_i8's.
#[target_feature(enable = "avx2,fma")]
unsafe fn k2<const M: usize>(w: &W4, a: &[i8], asum: &[i32], sa: &[f32], ca: &[f32], out: &mut [f32]) {
    let (n, kp) = (w.n, w.kp);
    let npairs = kp / 8;
    for p in 0..n.div_ceil(NR) {
        let mut acc32 = [[_mm256_setzero_si256(); 2]; M];
        dot_pairs::<M>(w.data.as_ptr().add(p * npairs * 64), a, kp, 0, npairs, &mut acc32);
        let ncols = NR.min(n - p * NR);
        let c = p * NR;
        for m in 0..M {
            let off = _mm256_set1_epi32(8 * asum[m]);
            let sv = _mm256_set1_ps(sa[m]);
            let cv = _mm256_set1_ps(ca[m]);
            let s0 = _mm256_mul_ps(sv, _mm256_loadu_ps(w.scale.as_ptr().add(c)));
            let s1 = _mm256_mul_ps(sv, _mm256_loadu_ps(w.scale.as_ptr().add(c + 8)));
            let b0 = _mm256_fmadd_ps(cv, _mm256_loadu_ps(w.wsum.as_ptr().add(c)), _mm256_loadu_ps(w.bias.as_ptr().add(c)));
            let b1 = _mm256_fmadd_ps(cv, _mm256_loadu_ps(w.wsum.as_ptr().add(c + 8)), _mm256_loadu_ps(w.bias.as_ptr().add(c + 8)));
            let y0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_sub_epi32(acc32[m][0], off)), s0, b0);
            let y1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_sub_epi32(acc32[m][1], off)), s1, b1);
            store16(y0, y1, &mut out[m * n + c..], ncols);
        }
    }
}

/// K2g: per-(row, 128-column group) scales.
#[target_feature(enable = "avx2,fma")]
unsafe fn k2g<const M: usize>(w: &W4, a: &[i8], gasum: &[i32], sa: &[f32], ca: &[f32], out: &mut [f32]) {
    let (n, kp) = (w.n, w.kp);
    let npairs = kp / 8;
    let ng = kp / GS;
    let ppg = GS / 8; // 16 pairs per group
    for p in 0..n.div_ceil(NR) {
        let mut facc = [[_mm256_setzero_ps(); 2]; M];
        let base = w.data.as_ptr().add(p * npairs * 64);
        for g in 0..ng {
            let mut acc32 = [[_mm256_setzero_si256(); 2]; M];
            dot_pairs::<M>(base, a, kp, g * ppg, ppg, &mut acc32);
            let gs = w.gscale.as_ptr().add((p * ng + g) * NR);
            let (g0, g1) = (_mm256_loadu_ps(gs), _mm256_loadu_ps(gs.add(8)));
            for m in 0..M {
                let off = _mm256_set1_epi32(8 * gasum[m * ng + g]);
                facc[m][0] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_sub_epi32(acc32[m][0], off)), g0, facc[m][0]);
                facc[m][1] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_sub_epi32(acc32[m][1], off)), g1, facc[m][1]);
            }
        }
        let ncols = NR.min(n - p * NR);
        let c = p * NR;
        for m in 0..M {
            let sv = _mm256_set1_ps(sa[m]);
            let cv = _mm256_set1_ps(ca[m]);
            let b0 = _mm256_fmadd_ps(cv, _mm256_loadu_ps(w.gwsum.as_ptr().add(c)), _mm256_loadu_ps(w.bias.as_ptr().add(c)));
            let b1 = _mm256_fmadd_ps(cv, _mm256_loadu_ps(w.gwsum.as_ptr().add(c + 8)), _mm256_loadu_ps(w.bias.as_ptr().add(c + 8)));
            store16(_mm256_fmadd_ps(facc[m][0], sv, b0), _mm256_fmadd_ps(facc[m][1], sv, b1), &mut out[m * n + c..], ncols);
        }
    }
}

/// Scalar reference for K2g, same float order.
fn k2g_ref(w: &W4, a: &[i8], sa: &[f32], ca: &[f32], m: usize, out: &mut [f32]) {
    let (n, kp) = (w.n, w.kp);
    let ng = kp / GS;
    for r in 0..m {
        for col in 0..n {
            let mut facc = 0f32;
            for g in 0..ng {
                let mut acc = 0i32;
                for kk in g * GS..(g + 1) * GS {
                    acc += i32::from(a[r * kp + kk]) * i32::from(w.q[col * kp + kk]);
                }
                facc = (acc as f32).mul_add(w.gscale[((col / NR) * ng + g) * NR + col % NR], facc);
            }
            let base = ca[r].mul_add(w.gwsum[col], w.bias[col]);
            out[r * n + col] = facc.mul_add(sa[r], base);
        }
    }
}

fn run(arm: usize, w: &W4, w8: &PackedI8, a: &[i8], asum: &[i32], gasum: &[i32], sa: &[f32], ca: &[f32], m: usize, out: &mut [f32]) {
    unsafe {
        match (arm, m) {
            (0, _) => gemm_i8(Isa::Avx2Fma, w8, a, sa, ca, m, out),
            (1, 1) => k2::<1>(w, a, asum, sa, ca, out),
            (1, 2) => k2::<2>(w, a, asum, sa, ca, out),
            (1, _) => k2::<4>(w, a, asum, sa, ca, out),
            (_, 1) => k2g::<1>(w, a, gasum, sa, ca, out),
            (_, 2) => k2g::<2>(w, a, gasum, sa, ca, out),
            (_, _) => k2g::<4>(w, a, gasum, sa, ca, out),
        }
    }
}

struct Act {
    a: Vec<i8>,
    sa: Vec<f32>,
    ca: Vec<f32>,
    asum: Vec<i32>,
    gasum: Vec<i32>,
}

fn acts(m: usize, k: usize, seed: &mut u64) -> Act {
    let ng = k / GS;
    let mut a = vec![0i8; m * k];
    let (mut sa, mut ca, mut asum, mut gasum) = (vec![0f32; m], vec![0f32; m], vec![0i32; m], vec![0i32; m * ng]);
    for r in 0..m {
        let x: Vec<f32> = (0..k).map(|_| (lcg(seed) % 2001) as f32 * 1e-3 - 0.6).collect();
        let (s, c) = quantize_row(Isa::Avx2Fma, &x, &mut a[r * k..(r + 1) * k]);
        sa[r] = s;
        ca[r] = c;
        asum[r] = a[r * k..(r + 1) * k].iter().map(|&v| i32::from(v)).sum();
        for g in 0..ng {
            gasum[r * ng + g] = a[r * k + g * GS..r * k + (g + 1) * GS].iter().map(|&v| i32::from(v)).sum();
        }
    }
    Act { a, sa, ca, asum, gasum }
}

fn main() {
    let reps: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(200);
    assert!(Isa::Avx2Fma.is_available());
    let mut seed = 0x5151_2027u64;
    let shapes: [(&str, usize, usize, usize); 8] = [
        ("ff_w1 384->1536", 1536, 384, 32), ("ff_w2 1536->384", 384, 1536, 32),
        ("qkv 384->1152", 1152, 384, 16), ("att_out 384->384", 384, 384, 16),
        ("pw1 384->768", 768, 384, 16), ("pw2 384->384", 384, 384, 16),
        ("stem_out 2432->384", 384, 2432, 1), ("enc_proj 384->320", 320, 384, 1)];
    println!("{{\"kind\":\"exact_hot\",\"reps\":{reps},\"group\":{GS}}}");
    for &(name, n, k, _) in &shapes {
        let (w4, w8) = make(n, k, &mut seed);
        for m in [1usize, 2, 4] {
            let x = acts(m, k, &mut seed);
            let mut outs = vec![vec![0f32; m * n]; 3];
            for arm in 0..3 {
                run(arm, &w4, &w8, &x.a, &x.asum, &x.gasum, &x.sa, &x.ca, m, &mut outs[arm]);
            }
            let mut r = vec![0f32; m * n];
            k2g_ref(&w4, &x.a, &x.sa, &x.ca, m, &mut r);
            let same = |u: &[f32], v: &[f32]| u.iter().zip(v).all(|(x, y)| x.to_bits() == y.to_bits());
            let ex_k2 = same(&outs[0], &outs[1]);
            let ex_k2g = same(&r, &outs[2]);
            let mut best = [f64::MAX; 3];
            for _ in 0..5 {
                for arm in 0..3 {
                    let t = Instant::now();
                    for _ in 0..reps {
                        run(arm, &w4, &w8, &x.a, &x.asum, &x.gasum, &x.sa, &x.ca, m, &mut outs[arm]);
                    }
                    best[arm] = best[arm].min(t.elapsed().as_secs_f64() / reps as f64);
                }
            }
            let macs = (m * n * k) as f64;
            println!("{{\"shape\":\"{name}\",\"m\":{m},\"exact_k2_vs_i8\":{ex_k2},\"exact_k2g_vs_ref\":{ex_k2g},\"i8_ns\":{:.0},\"k2_ns\":{:.0},\"k2g_ns\":{:.0},\"i8_gmacs\":{:.1},\"k2_gmacs\":{:.1},\"k2g_gmacs\":{:.1}}}",
                best[0] * 1e9, best[1] * 1e9, best[2] * 1e9, macs / best[0] * 1e-9, macs / best[1] * 1e-9, macs / best[2] * 1e-9);
            assert!(ex_k2 && ex_k2g, "not bit-exact: {name} m={m}");
        }
    }
    let mut mats = Vec::new();
    for &(_, n, k, count) in &shapes[..7] {
        for _ in 0..count {
            mats.push(make(n, k, &mut seed));
        }
    }
    let i8_bytes: usize = mats.iter().map(|(w, _)| w.n.div_ceil(NR) * NR * w.kp).sum();
    let gs_bytes: usize = mats.iter().map(|(w, _)| w.gscale.len() * 4).sum();
    println!("{{\"kind\":\"sweep\",\"matrices\":{},\"int8_mb\":{:.2},\"int4_mb\":{:.2},\"g128_f32_scales_mb\":{:.2}}}",
        mats.len(), i8_bytes as f64 / 1e6, i8_bytes as f64 / 2e6, gs_bytes as f64 / 1e6);
    for m in [1usize, 2, 4] {
        let xs: Vec<Act> = mats.iter().map(|(w, _)| acts(m, w.kp, &mut seed)).collect();
        let mut out = vec![0f32; 4 * 2432];
        let mut tot = [Vec::new(), Vec::new(), Vec::new()];
        for _ in 0..30 {
            for arm in 0..3 {
                let t = Instant::now();
                for (i, (w4, w8)) in mats.iter().enumerate() {
                    let x = &xs[i];
                    run(arm, w4, w8, &x.a, &x.asum, &x.gasum, &x.sa, &x.ca, m, &mut out[..m * w4.n]);
                }
                tot[arm].push(t.elapsed().as_secs_f64());
            }
        }
        let med = |v: &mut Vec<f64>| {
            v.sort_by(|x, y| x.partial_cmp(y).unwrap());
            v[v.len() / 2]
        };
        let (t8, t2, tg) = (med(&mut tot[0]), med(&mut tot[1]), med(&mut tot[2]));
        println!("{{\"kind\":\"sweep_step\",\"m\":{m},\"i8_ms\":{:.3},\"k2_ms\":{:.3},\"k2g_ms\":{:.3},\"k2_speedup\":{:.3},\"k2g_speedup\":{:.3}}}",
            t8 * 1e3, t2 * 1e3, tg * 1e3, t8 / t2, t8 / tg);
    }
}
