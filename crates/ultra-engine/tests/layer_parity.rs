//! E1 parity: every encoder layer of the fp32 engine against the trainer's own streaming
//! ChunkEncoder arithmetic (torch fp32), fed the same normalised features.
//!
//! Metric per layer and utterance: `max |engine - torch| / max |torch|` (normwise relative),
//! required <= 1e-4. The default test checks the first 8 chunks of 3 utterances; the
//! `--ignored` test checks every frame of all reference utterances (50 DEV).

mod common;

use std::sync::Arc;

use ultra_engine::safetensors::SafeTensors;
use ultra_engine::{Engine, Isa, Model, Precision, Workspace};
use ultra_frontend::MEL_BINS;

const TOL: f64 = 1e-4;

/// Returns the worst normwise relative error per layer over the checked utterances.
fn run(isa: Isa, utts: usize, max_chunks: Option<usize>) -> Option<Vec<f64>> {
    let (w, r) = (common::weights_dir(), common::reference_file());
    if !common::have(&[&w, &r]) {
        return None;
    }
    let model = Arc::new(Model::load(&w, true).expect("load weights"));
    let reference = SafeTensors::read(&r).expect("read reference");
    assert_eq!(
        reference.metadata.get("checkpoint_sha256"),
        Some(&model.arch.checkpoint_sha256),
        "reference and weights come from different checkpoints"
    );
    let count: usize = reference.metadata["count"].parse().expect("count");
    let engine = Engine::new(model.clone(), isa, Precision::F32).expect("engine");
    let n_layers = model.arch.n_blocks + 3;
    let mut worst = vec![0.0f64; n_layers];
    let mut ws = Workspace::default();
    for u in 0..utts.min(count) {
        let feats = &reference.tensors[&format!("u{u:03}.feats")];
        let mut rows: Vec<[f32; MEL_BINS]> = feats
            .data
            .chunks_exact(MEL_BINS)
            .map(|c| c.try_into().expect("80"))
            .collect();
        if let Some(c) = max_chunks {
            // Causality: the first c chunks depend only on features < 4 * 4c + 3.
            rows.truncate(rows.len().min(16 * c + 3));
        }
        let mut s = engine.create_stream();
        s.accept_features(&rows);
        if max_chunks.is_none() {
            s.input_finished();
        }
        let mut got: Vec<Vec<f32>> = vec![Vec::new(); n_layers];
        loop {
            let mut tap = |l: usize, v: &[f32], _w: usize| got[l].extend_from_slice(v);
            let done = engine
                .decode_tapped(&mut [&mut s], &mut ws, Some(&mut tap))
                .expect("decode");
            if done == 0 {
                break;
            }
        }
        for (l, g) in got.iter().enumerate() {
            let t = &reference.tensors[&format!("u{u:03}.layer{l:02}")];
            assert!(!g.is_empty() && g.len() <= t.data.len(), "layer {l} rows");
            if max_chunks.is_none() {
                assert_eq!(g.len(), t.data.len(), "layer {l} frame count");
            }
            let refv = &t.data[..g.len()];
            let mx = refv.iter().fold(0.0f64, |a, &v| a.max(f64::from(v).abs()));
            let err = g.iter().zip(refv).fold(0.0f64, |a, (&x, &y)| {
                a.max((f64::from(x) - f64::from(y)).abs())
            });
            worst[l] = worst[l].max(err / mx.max(1e-30));
        }
    }
    Some(worst)
}

fn check(worst: &[f64]) {
    for (l, &e) in worst.iter().enumerate() {
        eprintln!("layer {l:02}: max normwise relative error {e:.3e}");
    }
    let bad: Vec<_> = worst.iter().enumerate().filter(|(_, &e)| e > TOL).collect();
    assert!(bad.is_empty(), "layers over {TOL}: {bad:?}");
}

#[test]
fn fp32_layers_match_torch_prefix() {
    for isa in [Isa::Avx2Fma, Isa::Scalar] {
        if !isa.is_available() {
            continue;
        }
        if let Some(w) = run(isa, 3, Some(8)) {
            eprintln!("isa {}", isa.name());
            check(&w);
        }
    }
}

#[test]
#[ignore = "full 50-utterance DEV reference; run with --release --ignored"]
fn fp32_layers_match_torch_full() {
    if let Some(w) = run(Isa::detect(), usize::MAX, None) {
        check(&w);
    }
}
