//! A ternary layer's output for a given row must not depend on how many rows share the call
//! (batch invariance), with and without act7. T-LUT2 (<= 8 rows) quantises activations to 8
//! bits; with act7 the int8 tile uses 7 bits, so act7 must never route to T-LUT2.

use ultra_engine::encoder::{linear, Precision};
use ultra_engine::kernels::{Isa, PackedF32};
use ultra_engine::model::Linear;
use ultra_engine::quant::{PackedI8, PackedT, QuantScratch};

fn lcg(seed: &mut u64) -> f32 {
    *seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    ((*seed >> 40) as f32 / (1u64 << 24) as f32) - 0.5
}

#[test]
fn ternary_rows_are_batch_invariant_with_and_without_act7() {
    let (n, k) = (48, 96);
    let mut seed = 3u64;
    // Codes in {-1, 0, 1}: quantising weights already in {-s, 0, s} per row gives exact trits.
    let w: Vec<f32> = (0..n * k)
        .map(|i| {
            let s = 0.01 * (1 + i / k) as f32;
            match (lcg(&mut seed) * 3.0).round() as i32 {
                v if v < 0 => -s,
                0 => 0.0,
                _ => s,
            }
        })
        .collect();
    let b: Vec<f32> = (0..n).map(|_| lcg(&mut seed)).collect();
    // Quantising gives codes +-127 / 0; rebuild them as trits with scale s (exact).
    let q8 = PackedI8::quantize(&w, n, k, Some(&b));
    let (d, s8, _) = q8.parts();
    let trits: Vec<i8> = d.iter().map(|c| c.signum()).collect();
    let scale: Vec<f32> = s8.iter().map(|v| v * 127.0).collect();
    let mut wsum = vec![0.0f32; scale.len()];
    for (row, ws) in wsum.iter_mut().enumerate().take(n) {
        let tsum: i32 = (0..k)
            .map(|kk| (w[row * k + kk] / scale[row]).round() as i32)
            .sum();
        *ws = tsum as f32 * scale[row];
    }
    let q = PackedI8::from_parts(n, k, trits, &scale, &wsum, &b).expect("ternary parts");
    let Ok(Some(t)) = PackedT::from_ternary_codes(&q) else {
        eprintln!("SKIPPED: no T-LUT2 packing on this ISA");
        return;
    };
    let lin = Linear {
        f: PackedF32::pack(&w, n, k, Some(&b)),
        q: Some(q),
        t: Some(t),
        qat_i8: false,
    };
    let x: Vec<f32> = (0..16 * k).map(|_| lcg(&mut seed) * 4.0).collect();
    let isa = Isa::detect();
    for act7 in [false, true] {
        let mut outs = Vec::new();
        for rows in [1usize, 8, 9, 16] {
            let mut qs = QuantScratch {
                act7,
                ..QuantScratch::default()
            };
            let mut out = vec![0.0f32; rows * n];
            linear(
                isa,
                Precision::Int8,
                &lin,
                &x[..rows * k],
                rows,
                &mut out,
                &mut qs,
            );
            outs.push(out[..n].to_vec());
        }
        assert!(
            outs.windows(2).all(|w| w[0] == w[1]),
            "row 0 depends on the batch size (act7 = {act7})"
        );
    }
}
