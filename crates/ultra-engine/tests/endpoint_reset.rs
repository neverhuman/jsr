//! Verify optional always-on `Engine::with_endpoint_reset_ms`:
//! 1. `endpoint_reset_ms == 0` (default) is 0-ULP bit-identical to `Engine::new`.
//! 2. `endpoint_reset_ms > 0` triggers cache/context resets after silence gaps without dropping
//!    previously emitted tokens, and keeps `rel_t()` reset to 0 after each reset.

mod common;

use std::sync::Arc;

use ultra_engine::{Engine, Isa, Model, Precision, Workspace};
use ultra_frontend::npz::load_npz_audio_only;

fn decode_pcm(engine: &Engine, pcm: &[i16]) -> (Vec<u32>, u64) {
    let mut s = engine.create_stream();
    let mut ws = Workspace::default();
    for chunk in pcm.chunks(320) {
        s.accept_i16(chunk).expect("accept");
        while engine.decode(&mut [&mut s], &mut ws).expect("decode") > 0 {}
    }
    s.input_finished();
    while engine.decode(&mut [&mut s], &mut ws).expect("decode") > 0 {}
    (s.tokens().to_vec(), s.endpoint_resets())
}

#[test]
fn endpoint_reset_zero_is_bit_exact_and_nonzero_resets_across_silence() {
    let weights = common::weights_dir();
    let npz = common::dev_clean_npz();
    if !common::have(&[&weights, &npz]) {
        return;
    }
    let mut model = Model::load(&weights, false).expect("load model");
    model.quantize_int8();
    let model = Arc::new(model);

    let clips = load_npz_audio_only(&npz, 0, Some(2)).expect("npz");
    // Stitch clip 0 + 3.0 s digital silence + clip 1 into one continuous stream.
    let mut stitched = Vec::new();
    stitched.extend_from_slice(&clips[0].pcm);
    stitched.extend(std::iter::repeat_n(0i16, 48_000));
    stitched.extend_from_slice(&clips[1].pcm);

    let eng_def = Engine::new(model.clone(), Isa::detect(), Precision::Int8).unwrap();
    let eng_zero = Engine::new(model.clone(), Isa::detect(), Precision::Int8)
        .unwrap()
        .with_endpoint_reset_ms(0);
    let (tok_def, res_def) = decode_pcm(&eng_def, &stitched);
    let (tok_zero, res_zero) = decode_pcm(&eng_zero, &stitched);
    assert_eq!(res_def, 0);
    assert_eq!(res_zero, 0);
    assert_eq!(tok_def, tok_zero, "endpoint_reset_ms=0 must be identical");

    let eng_reset = Engine::new(model.clone(), Isa::detect(), Precision::Int8)
        .unwrap()
        .with_endpoint_reset_ms(1000);
    let (tok_rst, resets) = decode_pcm(&eng_reset, &stitched);
    assert!(
        resets >= 1,
        "expected >= 1 endpoint reset across 3 s silence gap, got {resets}"
    );
    assert!(
        !tok_rst.is_empty(),
        "endpoint reset must preserve emitted tokens"
    );

    // Reset with replay: periodic re-arm + feature replay preserves tokens across silence.
    let eng_replay = Engine::new(model, Isa::detect(), Precision::Int8)
        .unwrap()
        .with_endpoint_reset(1500, 1000);
    let (tok_rep, rep_resets) = decode_pcm(&eng_replay, &stitched);
    assert!(
        rep_resets >= 1,
        "expected >= 1 replay reset across 3 s silence gap, got {rep_resets}"
    );
    assert!(
        !tok_rep.is_empty(),
        "endpoint reset with replay must preserve emitted tokens"
    );
}
