//! H15 B2 micro-benchmark (exploratory): the signed int8 tile (K0), the `act7` tile, and the
//! experimental unsigned-offset 7-bit tile (U7) on the real dense-S layer shapes, hot (one
//! layer's weights resident) and streaming (all 16 blocks' weights, ~54 MiB). Checks U7 ==
//! act7 bit for bit on every shape before timing. Three alternating repeats.
//!
//! `taskset -c <core> cargo run --release -p ultra-engine --example u7_bench`

use std::time::Instant;

use ultra_engine::kernels::{colsum_i32, gemm_i8, gemm_i8_a7, gemm_i8_u7, quantize_row, Isa};
use ultra_engine::quant::PackedI8;

fn lcg(seed: &mut u64) -> f32 {
    *seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    ((*seed >> 40) as f32 / (1u64 << 24) as f32) - 0.5
}

/// (n, k) of one dense-S block's linear layers.
const SHAPES: [(usize, usize, &str); 6] = [
    (1536, 384, "ff_w1"),
    (384, 1536, "ff_w2"),
    (1152, 384, "qkv"),
    (384, 384, "att_out"),
    (768, 384, "pw1"),
    (384, 384, "pw2"),
];

struct Layer {
    p: PackedI8,
    cs: Vec<i32>,
}

fn main() {
    let isa = Isa::detect();
    println!("isa {isa:?}");
    let mut seed = 7u64;
    // One block = ff1 (w1, w2) + qkv + att_out + pw1 + pw2 + ff2 (w1, w2).
    let block: Vec<usize> = vec![0, 1, 2, 3, 4, 5, 0, 1];
    let layers: Vec<Layer> = SHAPES
        .iter()
        .map(|&(n, k, _)| {
            let w: Vec<f32> = (0..n * k).map(|_| lcg(&mut seed)).collect();
            let p = PackedI8::quantize(&w, n, k, None);
            let cs = colsum_i32(&p);
            Layer { p, cs }
        })
        .collect();
    for &rows in &[4usize, 16, 64] {
        // Per shape: codes for rows x kp.
        let mut inputs = Vec::new();
        for l in &layers {
            let (k, kp) = (l.p.k(), l.p.kp());
            let x: Vec<f32> = (0..rows * k).map(|_| lcg(&mut seed)).collect();
            let mut a = vec![0i8; rows * kp];
            let (mut sa, mut ca) = (vec![0.0f32; rows], vec![0.0f32; rows]);
            for r in 0..rows {
                (sa[r], ca[r]) =
                    quantize_row(isa, &x[r * k..(r + 1) * k], &mut a[r * kp..r * kp + k]);
            }
            let a7: Vec<i8> = a.iter().map(|&v| (v / 2).clamp(-63, 63)).collect();
            let u: Vec<u8> = a7.iter().map(|&v| (i16::from(v) + 64) as u8).collect();
            inputs.push((a, a7, u, sa, ca));
        }
        // Exactness first.
        for (l, (_, a7, u, sa, ca)) in layers.iter().zip(&inputs) {
            let n = l.p.n();
            let (mut o7, mut ou) = (vec![0.0; rows * n], vec![0.0; rows * n]);
            gemm_i8_a7(isa, &l.p, a7, sa, ca, rows, &mut o7);
            gemm_i8_u7(isa, &l.p, &l.cs, u, sa, ca, rows, &mut ou);
            assert_eq!(o7, ou, "U7 != act7 at rows {rows} n {n}");
        }
        for (mode, copies) in [("hot", 1usize), ("stream x16 blocks", 16)] {
            let macs: usize = block
                .iter()
                .map(|&i| rows * layers[i].p.n() * layers[i].p.k())
                .sum::<usize>()
                * copies;
            let reps = (3_000_000_000 / macs).max(2);
            let streams: Vec<Vec<PackedI8>> = layers
                .iter()
                .map(|l| (0..copies).map(|_| l.p.clone()).collect())
                .collect();
            let mut best = [f64::MAX; 3];
            for _rep in 0..3 {
                for (kind, slot) in best.iter_mut().enumerate() {
                    let t = Instant::now();
                    for _ in 0..reps {
                        for c in 0..copies {
                            for &i in &block {
                                let l = &layers[i];
                                let (a, a7, u, sa, ca) = &inputs[i];
                                let p = &streams[i][c];
                                let mut out = vec![0.0f32; rows * p.n()];
                                match kind {
                                    0 => gemm_i8(isa, p, a, sa, ca, rows, &mut out),
                                    1 => gemm_i8_a7(isa, p, a7, sa, ca, rows, &mut out),
                                    _ => gemm_i8_u7(isa, p, &l.cs, u, sa, ca, rows, &mut out),
                                }
                                std::hint::black_box(&out);
                            }
                        }
                    }
                    *slot = slot.min(t.elapsed().as_secs_f64() * 1e12 / (reps * macs) as f64);
                }
            }
            println!(
                "rows {rows:2} {mode:18}: ps/MAC K0 {:.2}  act7 {:.2}  U7 {:.2}   U7 vs K0 {:.2}x  vs act7 {:.2}x",
                best[0],
                best[1],
                best[2],
                best[0] / best[2],
                best[1] / best[2]
            );
        }
    }
}
