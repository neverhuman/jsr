#![allow(unsafe_op_in_unsafe_fn, clippy::all)]
//! h51-pq scratch (NOT for the repo): int4-weight x int8-activation small-M GEMM on AVX2 vs the
//! engine's INT8 `gemm_i8`, on A1-49000's real encoder shapes.
//!
//! K1: two's-complement nibbles, unpacked to int8, then the unchanged INT8 tile path.
//! K2: offset-binary nibbles (w + 8, 1..=15) as the unsigned `vpmaddubsw` operand against the
//!     broadcast signed activation codes; i16 sums over 8 K-groups (8 * 2 * 15 * 127 = 30,480 <
//!     32,767) before one `vpmaddwd`; the offset is removed in the epilogue as 8 * sum(a).
//! Both are checked bit-exact against `gemm_i8` (AVX2 and the portable scalar reference) on the
//! same int4-valued weights, then timed:
//! cache-hot (one matrix) and a whole-encoder sweep (129 distinct matrices, DRAM-streamed).
//!
//! cargo run --release -p ultra-engine --example h51_int4_bench -- [reps]
use std::arch::x86_64::*;
use std::time::Instant;

use ultra_engine::kernels::{gemm_i8, quantize_row, Isa, NR};
use ultra_engine::quant::PackedI8;

const KG: usize = 4;

fn lcg(s: &mut u64) -> u64 {
    *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    *s >> 33
}

/// int4 weights in panels of 16 columns; each 64-byte block holds two K-groups:
/// low nibble = group 2*gp, high nibble = group 2*gp + 1, byte layout of an INT8 group block.
struct PackedI4 {
    n: usize,
    kp: usize,
    data_tc: Vec<u8>,  // two's complement nibbles (K1)
    data_ob: Vec<u8>,  // offset-binary nibbles (K2)
    scale: Vec<f32>,
    wsum: Vec<f32>,
    bias: Vec<f32>,
}

fn make(n: usize, k: usize, seed: &mut u64) -> (PackedI4, PackedI8) {
    let kp = k.div_ceil(8) * 8;
    let panels = n.div_ceil(NR);
    let mut q = vec![0i8; n * kp];
    for row in 0..n {
        for kk in 0..k {
            q[row * kp + kk] = (lcg(seed) % 15) as i8 - 7;
        }
    }
    let mut scale = vec![0f32; panels * NR];
    let mut wsum = vec![0f32; panels * NR];
    let mut bias = vec![0f32; n];
    for row in 0..n {
        scale[row] = 0.001 + (lcg(seed) % 1000) as f32 * 1e-5;
        wsum[row] = q[row * kp..row * kp + kp].iter().map(|&v| i32::from(v)).sum::<i32>() as f32 * scale[row];
        bias[row] = (lcg(seed) % 200) as f32 * 1e-3 - 0.1;
    }
    // INT8 packing of the same values: data[((p * (kp/4) + g) * 16 + j) * 4 + kk % 4]
    let mut d8 = vec![0i8; panels * kp * NR];
    for row in 0..n {
        let (p, j) = (row / NR, row % NR);
        for kk in 0..kp {
            d8[((p * (kp / KG) + kk / KG) * NR + j) * KG + kk % KG] = q[row * kp + kk];
        }
    }
    let mut d_tc = vec![0u8; panels * kp * NR / 2];
    let mut d_ob = vec![0u8; panels * kp * NR / 2];
    for p in 0..panels {
        for gp in 0..kp / 8 {
            for j in 0..NR {
                for kk in 0..KG {
                    let lo = d8[((p * (kp / KG) + 2 * gp) * NR + j) * KG + kk];
                    let hi = d8[((p * (kp / KG) + 2 * gp + 1) * NR + j) * KG + kk];
                    let o = (p * (kp / 8) + gp) * 64 + j * KG + kk;
                    d_tc[o] = ((lo as u8) & 0x0F) | (((hi as u8) & 0x0F) << 4);
                    d_ob[o] = ((lo + 8) as u8) | (((hi + 8) as u8) << 4);
                }
            }
        }
    }
    let p8 = PackedI8::from_parts(n, k, d8, &scale, &wsum, &bias).expect("packed i8");
    let mut b = vec![0f32; panels * NR];
    b[..n].copy_from_slice(&bias);
    (PackedI4 { n, kp, data_tc: d_tc, data_ob: d_ob, scale, wsum, bias: b }, p8)
}

#[target_feature(enable = "avx2,fma")]
unsafe fn epilogue(acc: [__m256i; 2], sa: f32, ca: f32, sw: &[f32], ws: &[f32], bi: &[f32], out: &mut [f32], ncols: usize) {
    let (s0, s1) = (_mm256_loadu_ps(sw.as_ptr()), _mm256_loadu_ps(sw.as_ptr().add(8)));
    let (b0, b1) = (_mm256_loadu_ps(bi.as_ptr()), _mm256_loadu_ps(bi.as_ptr().add(8)));
    let (u0, u1) = (_mm256_loadu_ps(ws.as_ptr()), _mm256_loadu_ps(ws.as_ptr().add(8)));
    let sv = _mm256_set1_ps(sa);
    let cv = _mm256_set1_ps(ca);
    let base0 = _mm256_fmadd_ps(cv, u0, b0);
    let base1 = _mm256_fmadd_ps(cv, u1, b1);
    let y0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(acc[0]), _mm256_mul_ps(sv, s0), base0);
    let y1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(acc[1]), _mm256_mul_ps(sv, s1), base1);
    let mut tmp = [0f32; 16];
    _mm256_storeu_ps(tmp.as_mut_ptr(), y0);
    _mm256_storeu_ps(tmp.as_mut_ptr().add(8), y1);
    out[..ncols].copy_from_slice(&tmp[..ncols]);
}

/// K1: unpack two's-complement nibbles, then exactly the INT8 tile's arithmetic.
#[target_feature(enable = "avx2,fma")]
unsafe fn k1<const M: usize>(w: &PackedI4, a: &[i8], sa: &[f32], ca: &[f32], out: &mut [f32]) {
    let (n, kp) = (w.n, w.kp);
    let ones = _mm256_set1_epi16(1);
    let m4 = _mm256_set1_epi8(0x0F);
    let m8 = _mm256_set1_epi8(0x08);
    for p in 0..n.div_ceil(NR) {
        let mut acc = [[_mm256_setzero_si256(); 2]; M];
        let base = w.data_tc.as_ptr().add(p * (kp / 8) * 64);
        for gp in 0..kp / 8 {
            let v0 = _mm256_loadu_si256(base.add(gp * 64).cast());
            let v1 = _mm256_loadu_si256(base.add(gp * 64 + 32).cast());
            let w0 = [
                _mm256_sub_epi8(_mm256_xor_si256(_mm256_and_si256(v0, m4), m8), m8),
                _mm256_sub_epi8(_mm256_xor_si256(_mm256_and_si256(_mm256_srli_epi16::<4>(v0), m4), m8), m8),
            ];
            let w1 = [
                _mm256_sub_epi8(_mm256_xor_si256(_mm256_and_si256(v1, m4), m8), m8),
                _mm256_sub_epi8(_mm256_xor_si256(_mm256_and_si256(_mm256_srli_epi16::<4>(v1), m4), m8), m8),
            ];
            for (m, acc_m) in acc.iter_mut().enumerate() {
                for h in 0..2 {
                    let a4 = std::ptr::read_unaligned(a.as_ptr().add(m * kp + 4 * (2 * gp + h)).cast::<i32>());
                    let av = _mm256_set1_epi32(a4);
                    let aa = _mm256_abs_epi8(av);
                    let p0 = _mm256_maddubs_epi16(aa, _mm256_sign_epi8(w0[h], av));
                    let p1 = _mm256_maddubs_epi16(aa, _mm256_sign_epi8(w1[h], av));
                    acc_m[0] = _mm256_add_epi32(acc_m[0], _mm256_madd_epi16(p0, ones));
                    acc_m[1] = _mm256_add_epi32(acc_m[1], _mm256_madd_epi16(p1, ones));
                }
            }
        }
        let ncols = NR.min(n - p * NR);
        for m in 0..M {
            epilogue(acc[m], sa[m], ca[m], &w.scale[p * NR..], &w.wsum[p * NR..], &w.bias[p * NR..], &mut out[m * n + p * NR..], ncols);
        }
    }
}

/// K2: offset-binary nibbles as the unsigned operand, i16 sums over 8 groups, offset in epilogue.
#[target_feature(enable = "avx2,fma")]
unsafe fn k2<const M: usize>(w: &PackedI4, a: &[i8], asum: &[i32], sa: &[f32], ca: &[f32], out: &mut [f32]) {
    let (n, kp) = (w.n, w.kp);
    let ones = _mm256_set1_epi16(1);
    let m4 = _mm256_set1_epi8(0x0F);
    let npairs = kp / 8;
    for p in 0..n.div_ceil(NR) {
        let mut acc32 = [[_mm256_setzero_si256(); 2]; M];
        let mut acc16 = [[_mm256_setzero_si256(); 2]; M];
        let base = w.data_ob.as_ptr().add(p * npairs * 64);
        for gp in 0..npairs {
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
            if gp % 4 == 3 || gp + 1 == npairs {
                // 4 pairs = 8 groups per flush: <= 8 * 3,810 = 30,480 per i16 lane
                for m in 0..M {
                    acc32[m][0] = _mm256_add_epi32(acc32[m][0], _mm256_madd_epi16(acc16[m][0], ones));
                    acc32[m][1] = _mm256_add_epi32(acc32[m][1], _mm256_madd_epi16(acc16[m][1], ones));
                    acc16[m] = [_mm256_setzero_si256(); 2];
                }
            }
        }
        let ncols = NR.min(n - p * NR);
        for m in 0..M {
            let off = _mm256_set1_epi32(8 * asum[m]);
            let acc = [_mm256_sub_epi32(acc32[m][0], off), _mm256_sub_epi32(acc32[m][1], off)];
            epilogue(acc, sa[m], ca[m], &w.scale[p * NR..], &w.wsum[p * NR..], &w.bias[p * NR..], &mut out[m * n + p * NR..], ncols);
        }
    }
}

fn run_k1(w: &PackedI4, a: &[i8], sa: &[f32], ca: &[f32], m: usize, out: &mut [f32]) {
    unsafe {
        match m {
            1 => k1::<1>(w, a, sa, ca, out),
            2 => k1::<2>(w, a, sa, ca, out),
            _ => k1::<4>(w, a, sa, ca, out),
        }
    }
}

fn run_k2(w: &PackedI4, a: &[i8], asum: &[i32], sa: &[f32], ca: &[f32], m: usize, out: &mut [f32]) {
    unsafe {
        match m {
            1 => k2::<1>(w, a, asum, sa, ca, out),
            2 => k2::<2>(w, a, asum, sa, ca, out),
            _ => k2::<4>(w, a, asum, sa, ca, out),
        }
    }
}

fn acts(m: usize, k: usize, kp: usize, seed: &mut u64) -> (Vec<i8>, Vec<f32>, Vec<f32>, Vec<i32>) {
    let mut a = vec![0i8; m * kp];
    let (mut sa, mut ca, mut asum) = (vec![0f32; m], vec![0f32; m], vec![0i32; m]);
    for r in 0..m {
        let x: Vec<f32> = (0..k).map(|_| (lcg(seed) % 2001) as f32 * 1e-3 - 0.6).collect();
        let (s, c) = quantize_row(Isa::Avx2Fma, &x, &mut a[r * kp..r * kp + k]);
        sa[r] = s;
        ca[r] = c;
        asum[r] = a[r * kp..(r + 1) * kp].iter().map(|&v| i32::from(v)).sum();
    }
    (a, sa, ca, asum)
}

fn main() {
    let reps: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(200);
    assert!(Isa::Avx2Fma.is_available());
    let mut seed = 0x5151_2026u64;
    // A1-49000 encoder GEMMs: (name, n, k, count per step)
    let shapes: [(&str, usize, usize, usize); 8] = [
        ("ff_w1 384->1536", 1536, 384, 32), ("ff_w2 1536->384", 384, 1536, 32),
        ("qkv 384->1152", 1152, 384, 16), ("att_out 384->384", 384, 384, 16),
        ("pw1 384->768", 768, 384, 16), ("pw2 384->384", 384, 384, 16),
        ("stem_out 2432->384", 384, 2432, 1), ("enc_proj 384->320", 320, 384, 1)];
    // 1. exactness + hot timing per shape
    println!("{{\"kind\":\"exact_hot\",\"reps\":{reps}}}");
    for &(name, n, k, _) in &shapes {
        let (w4, w8) = make(n, k, &mut seed);
        for m in [1usize, 2, 4] {
            let (a, sa, ca, asum) = acts(m, k, w4.kp, &mut seed);
            let (mut o8, mut o1, mut o2) = (vec![0f32; m * n], vec![0f32; m * n], vec![0f32; m * n]);
            gemm_i8(Isa::Avx2Fma, &w8, &a, &sa, &ca, m, &mut o8);
            run_k1(&w4, &a, &sa, &ca, m, &mut o1);
            run_k2(&w4, &a, &asum, &sa, &ca, m, &mut o2);
            // scalar reference: the engine's portable gemm_i8 (same fma epilogue, exact i32 dot)
            let mut os = vec![0f32; m * n];
            gemm_i8(Isa::Scalar, &w8, &a, &sa, &ca, m, &mut os);
            let same = |u: &[f32], v: &[f32]| u.iter().zip(v).all(|(x, y)| x.to_bits() == y.to_bits());
            let ex8s = same(&o8, &os);
            let ex1 = same(&o8, &o1) && same(&os, &o1);
            let ex2 = same(&o8, &o2) && same(&os, &o2);
            let mut best = [f64::MAX; 3];
            for _ in 0..5 {
                let t = Instant::now();
                for _ in 0..reps { gemm_i8(Isa::Avx2Fma, &w8, &a, &sa, &ca, m, &mut o8); }
                best[0] = best[0].min(t.elapsed().as_secs_f64() / reps as f64);
                let t = Instant::now();
                for _ in 0..reps { run_k1(&w4, &a, &sa, &ca, m, &mut o1); }
                best[1] = best[1].min(t.elapsed().as_secs_f64() / reps as f64);
                let t = Instant::now();
                for _ in 0..reps { run_k2(&w4, &a, &asum, &sa, &ca, m, &mut o2); }
                best[2] = best[2].min(t.elapsed().as_secs_f64() / reps as f64);
            }
            let macs = (m * n * k) as f64;
            println!("{{\"shape\":\"{name}\",\"m\":{m},\"exact_i8_avx2_vs_scalar\":{ex8s},\"exact_k1\":{ex1},\"exact_k2\":{ex2},\"i8_ns\":{:.0},\"k1_ns\":{:.0},\"k2_ns\":{:.0},\"i8_gmacs\":{:.1},\"k1_gmacs\":{:.1},\"k2_gmacs\":{:.1}}}",
                best[0] * 1e9, best[1] * 1e9, best[2] * 1e9, macs / best[0] * 1e-9, macs / best[1] * 1e-9, macs / best[2] * 1e-9);
            assert!(ex8s && ex1 && ex2, "not bit-exact: {name} m={m}");
        }
    }
    // 2. whole-encoder sweep: 129 distinct matrices per step (DRAM-streamed), interleaved arms
    let mut mats = Vec::new();
    for &(_, n, k, count) in &shapes[..7] {
        for _ in 0..count { mats.push(make(n, k, &mut seed)); }
    }
    let i8_bytes: usize = mats.iter().map(|(w4, _)| w4.n.div_ceil(NR) * NR * w4.kp).sum();
    println!("{{\"kind\":\"sweep\",\"matrices\":{},\"int8_mb\":{:.1},\"int4_mb\":{:.1}}}", mats.len(), i8_bytes as f64 / 1e6, i8_bytes as f64 / 2e6);
    for m in [1usize, 2, 4] {
        let actv: Vec<_> = mats.iter().map(|(w4, _)| acts(m, w4.kp, w4.kp, &mut seed)).collect();
        let mut out = vec![0f32; 4 * 2432 * 4];
        let steps = 30usize;
        let mut tot = [Vec::new(), Vec::new(), Vec::new()];
        for _ in 0..steps {
            for arm in 0..3 {
                let t = Instant::now();
                for (i, (w4, w8)) in mats.iter().enumerate() {
                    let (a, sa, ca, asum) = &actv[i];
                    match arm {
                        0 => gemm_i8(Isa::Avx2Fma, w8, a, sa, ca, m, &mut out[..m * w4.n]),
                        1 => run_k1(w4, a, sa, ca, m, &mut out[..m * w4.n]),
                        _ => run_k2(w4, a, asum, sa, ca, m, &mut out[..m * w4.n]),
                    }
                }
                tot[arm].push(t.elapsed().as_secs_f64());
            }
        }
        let med = |v: &mut Vec<f64>| { v.sort_by(|x, y| x.partial_cmp(y).unwrap()); v[v.len() / 2] };
        let (t8, t1, t2) = (med(&mut tot[0]), med(&mut tot[1]), med(&mut tot[2]));
        println!("{{\"kind\":\"sweep_step\",\"m\":{m},\"i8_ms\":{:.3},\"k1_ms\":{:.3},\"k2_ms\":{:.3},\"k1_speedup\":{:.3},\"k2_speedup\":{:.3}}}",
            t8 * 1e3, t1 * 1e3, t2 * 1e3, t8 / t1, t8 / t2);
    }
}
