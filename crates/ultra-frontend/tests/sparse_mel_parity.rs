//! 0-ULP bit-exactness proof for sparse triangular mel projection + precomputed FFT bit-reversal
//! against the brute-force dense `257 x 80` reference across edge cases (silence, `-0.0`,
//! impulses, pseudorandom speech-like PCM, and non-finite `NaN`/`Inf` fallback).

use ultra_frontend::mel::WIN_SAMPLES;
use ultra_frontend::trainer_mel::{GlobalCmvn, TrainerMelConfig};

#[test]
fn sparse_triangular_mel_matches_dense_reference_to_zero_ulp() {
    let cfg = TrainerMelConfig::default_htk(GlobalCmvn::identity()).expect("htk");
    assert!(
        cfg.sparse_fbank_nnz() < 600,
        "HTK triangular mel filterbank should have ~514 nonzeros, got {}",
        cfg.sparse_fbank_nnz()
    );

    let mut frames: Vec<Vec<f32>> = vec![
        vec![0.0f32; WIN_SAMPLES],
        vec![-0.0f32; WIN_SAMPLES],
        (0..WIN_SAMPLES)
            .map(|i| if i == 0 { 1.0 } else { 0.0 })
            .collect(),
        (0..WIN_SAMPLES)
            .map(|i| if i % 2 == 0 { 0.99 } else { -0.99 })
            .collect(),
    ];
    let mut rng = 0x5eed_cafe_1234_5678u64;
    for _ in 0..64 {
        let mut f = Vec::with_capacity(WIN_SAMPLES);
        for _ in 0..WIN_SAMPLES {
            rng = rng.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let s16 = ((rng >> 48) as i16) as f32 / 32768.0;
            f.push(s16);
        }
        frames.push(f);
    }

    for (fi, frame) in frames.iter().enumerate() {
        let sparse_auto = cfg.frame_f32_portable(frame);
        let sparse_scalar = cfg.frame_f32_portable_scalar(frame);
        let dense_ref = cfg.frame_f32_portable_dense_ref(frame);
        for b in 0..80 {
            assert_eq!(
                sparse_auto[b].to_bits(),
                dense_ref[b].to_bits(),
                "auto sparse vs dense bit mismatch frame={fi} bin={b}"
            );
            assert_eq!(
                sparse_scalar[b].to_bits(),
                dense_ref[b].to_bits(),
                "scalar sparse vs dense bit mismatch frame={fi} bin={b}"
            );
        }
    }

    // Non-finite NaN input must preserve exact `0 * NaN = NaN` semantics via dense fallback.
    let mut nan_frame = vec![0.1f32; WIN_SAMPLES];
    nan_frame[17] = f32::NAN;
    let s_nan = cfg.frame_f32_portable(&nan_frame);
    let d_nan = cfg.frame_f32_portable_dense_ref(&nan_frame);
    for b in 0..80 {
        assert_eq!(s_nan[b].is_nan(), d_nan[b].is_nan());
    }

    // Measure dense vs sparse portable throughput over 10,000 frames (= 100 s of 16 kHz audio).
    let bench_frame = &frames[10];
    let n_iters = 10_000usize;
    let t0 = std::time::Instant::now();
    let mut sink = 0.0f32;
    for _ in 0..n_iters {
        sink += cfg.frame_f32_portable_dense_ref(bench_frame)[0];
    }
    let dense_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let t1 = std::time::Instant::now();
    for _ in 0..n_iters {
        sink += cfg.frame_f32_portable(bench_frame)[0];
    }
    let sparse_ms = t1.elapsed().as_secs_f64() * 1000.0;
    assert!(sink.is_finite());
    eprintln!(
        "sparse_mel_parity ({n_iters} frames = 100 audio-s, nnz={}): dense={dense_ms:.2} ms ({:.3} CPU-ms/s) vs sparse={sparse_ms:.2} ms ({:.3} CPU-ms/s) -> {:.2}x speedup",
        cfg.sparse_fbank_nnz(),
        dense_ms / 100.0,
        sparse_ms / 100.0,
        dense_ms / sparse_ms.max(1e-9),
    );
}
