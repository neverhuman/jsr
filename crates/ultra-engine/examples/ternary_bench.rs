//! Exploratory: ternary layer cost with the engine's int8 GEMM (codes in {-1, 0, 1}) versus
//! ultra-kernels T-LUT2 (2-bit, two A7 passes for signed activations), hot and streaming.

use std::time::Instant;

use ultra_engine::kernels::{gemm_i8, quantize_row, Isa};
use ultra_engine::quant::PackedI8;
use ultra_kernels::{Backend, Frames, Gemv, TLut2Matrix};

fn main() {
    let isa = Isa::detect();
    let kb = Backend::detect();
    let mut seed = 1u64;
    let mut rnd = || {
        seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        (seed >> 33) as u32
    };
    for &(n, k, copies) in &[(1536usize, 384usize, 1usize), (1536, 384, 64)] {
        let trits: Vec<i8> = (0..n * k).map(|_| (rnd() % 3) as i8 - 1).collect();
        let wf: Vec<f32> = trits.iter().map(|&t| f32::from(t)).collect();
        let pq = PackedI8::quantize(&wf, n, k, None);
        let tl = TLut2Matrix::from_trits(n, k, &trits).expect("tlut2");
        let qs: Vec<PackedI8> = (0..copies).map(|_| pq.clone()).collect();
        let ts: Vec<TLut2Matrix> = (0..copies).map(|_| tl.clone()).collect();
        for &rows in &[4usize, 16, 64] {
            let x: Vec<f32> = (0..rows * k)
                .map(|_| (rnd() % 2000) as f32 / 1000.0 - 1.0)
                .collect();
            let mut a = vec![0i8; rows * pq.kp()];
            let (mut sa, mut ca) = (vec![0.0; rows], vec![0.0; rows]);
            for r in 0..rows {
                (sa[r], ca[r]) = quantize_row(
                    isa,
                    &x[r * k..(r + 1) * k],
                    &mut a[r * pq.kp()..r * pq.kp() + k],
                );
            }
            let pos: Vec<u8> = (0..rows)
                .flat_map(|r| {
                    a[r * pq.kp()..r * pq.kp() + k]
                        .iter()
                        .map(|&v| v.max(0) as u8)
                })
                .collect();
            let neg: Vec<u8> = (0..rows)
                .flat_map(|r| {
                    a[r * pq.kp()..r * pq.kp() + k]
                        .iter()
                        .map(|&v| (-(v.max(-127))).max(0) as u8)
                })
                .collect();
            let mut out = vec![0.0f32; rows * n];
            let reps = (400_000_000 / (rows * n * k * copies)).max(3);
            let t = Instant::now();
            for _ in 0..reps {
                for q in &qs {
                    gemm_i8(isa, q, &a, &sa, &ca, rows, &mut out);
                }
            }
            let gi = (reps * copies * rows * n * k) as f64 / t.elapsed().as_secs_f64() / 1e9;
            let (fp, fneg) = (
                Frames::new(&pos, k).expect("pos"),
                Frames::new(&neg, k).expect("neg"),
            );
            let (mut o1, mut o2) = (vec![0i32; rows * n], vec![0i32; rows * n]);
            let t = Instant::now();
            for _ in 0..reps {
                for m in &ts {
                    m.view().gemv(kb, &fp, &mut o1).expect("gemv");
                    m.view().gemv(kb, &fneg, &mut o2).expect("gemv");
                }
            }
            let gt = (reps * copies * rows * n * k) as f64 / t.elapsed().as_secs_f64() / 1e9;
            let sx: Vec<i8> = (0..rows)
                .flat_map(|r| a[r * pq.kp()..r * pq.kp() + k].to_vec())
                .collect();
            let t = Instant::now();
            for _ in 0..reps {
                for m in &ts {
                    m.view().gemv_signed(kb, &sx, &mut o1).expect("signed");
                }
            }
            let gs = (reps * copies * rows * n * k) as f64 / t.elapsed().as_secs_f64() / 1e9;
            println!("{n}x{k} x{copies} rows {rows:3}: engine int8 {gi:6.1}  tlut2 2-pass {gt:6.1}  tlut2 signed {gs:6.1} GMAC/s");
        }
    }
}
