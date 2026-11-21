//! Micro-benchmark of the GEMM kernels (exploratory): hot (weights resident in L1/L2) versus
//! streaming (a working set far larger than L3), for a few row counts.
//!
//! `cargo run --release -p ultra-engine --example kernel_bench`

use std::time::Instant;

use ultra_engine::kernels::{gemm_f32, gemm_i8, quantize_row, Isa, PackedF32};
use ultra_engine::quant::PackedI8;

fn lcg(seed: &mut u64) -> f32 {
    *seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    ((*seed >> 40) as f32 / (1u64 << 24) as f32) - 0.5
}

fn main() {
    let isa = Isa::detect();
    let mut seed = 1u64;
    for &(n, k, copies, label) in &[
        (1536usize, 384usize, 1usize, "hot 1536x384"),
        (384, 1536, 1, "hot 384x1536"),
        (1536, 384, 96, "stream 96 x 1536x384 (54 MiB i8)"),
    ] {
        let w: Vec<f32> = (0..n * k).map(|_| lcg(&mut seed)).collect();
        let pf = PackedF32::pack(&w, n, k, None);
        let pq = PackedI8::quantize(&w, n, k, None);
        let fs: Vec<PackedF32> = (0..copies.min(24)).map(|_| pf.clone()).collect();
        let qs: Vec<PackedI8> = (0..copies).map(|_| pq.clone()).collect();
        for &rows in &[4usize, 16, 64] {
            let x: Vec<f32> = (0..rows * k).map(|_| lcg(&mut seed)).collect();
            let mut a = vec![0i8; rows * pq.kp()];
            let mut sa = vec![0.0f32; rows];
            let mut ca = vec![0.0f32; rows];
            for r in 0..rows {
                (sa[r], ca[r]) = quantize_row(
                    isa,
                    &x[r * k..(r + 1) * k],
                    &mut a[r * pq.kp()..r * pq.kp() + k],
                );
            }
            let mut out = vec![0.0f32; rows * n];
            let reps = (2_000_000_000 / (rows * n * k * copies)).max(3);
            let t = Instant::now();
            for _ in 0..reps {
                for q in &qs {
                    gemm_i8(isa, q, &a, &sa, &ca, rows, &mut out);
                }
            }
            let gi = (reps * copies * rows * n * k) as f64 / t.elapsed().as_secs_f64() / 1e9;
            let reps_f = (reps / 4).max(2);
            let t = Instant::now();
            for _ in 0..reps_f {
                for f in &fs {
                    gemm_f32(isa, f, &x, rows, &mut out);
                }
            }
            let gf = (reps_f * fs.len() * rows * n * k) as f64 / t.elapsed().as_secs_f64() / 1e9;
            println!("{label:36} rows {rows:3}: int8 {gi:6.1} GMAC/s   fp32 {gf:6.1} GMAC/s");
        }
    }
}
