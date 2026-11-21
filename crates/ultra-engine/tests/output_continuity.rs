//! Output continuity: kernel and layout changes must not move a single bit. The fixture
//! stores, per precision (fp32, int8) and DEV utterance, the token ids and the SHA-256 of
//! every encoder-projection value the engine produced, from a build whose 400-DEV outputs
//! were verified identical to the previous release. A mismatch means a numerical change:
//! that is a new candidate needing the full quality gate, not a performance change.
//!
//! Regenerate only for an intended numerical change:
//! `ULTRA_ENGINE_WRITE_CONTINUITY=1 cargo test -p ultra-engine --test output_continuity`.

mod common;

use std::path::Path;
use std::sync::Arc;

use ultra_core::receipt::sha256_hex;
use ultra_engine::{Engine, Isa, Model, Precision, Workspace};
use ultra_frontend::npz::load_npz_audio_only;

const UTTS: usize = 4;

fn run(engine: &Engine, pcm: &[i16]) -> (Vec<u32>, String) {
    let mut ws = Workspace::default();
    let mut s = engine.create_stream();
    s.accept_i16(pcm).expect("accept");
    s.input_finished();
    let last = engine.model.arch.n_blocks + 2;
    let mut bytes: Vec<u8> = Vec::new();
    loop {
        let mut tap = |l: usize, v: &[f32], _w: usize| {
            if l == last {
                bytes.extend(v.iter().flat_map(|x| x.to_bits().to_le_bytes()));
            }
        };
        if engine
            .decode_tapped(&mut [&mut s], &mut ws, Some(&mut tap))
            .expect("decode")
            == 0
        {
            break;
        }
    }
    (s.tokens().to_vec(), sha256_hex(&bytes))
}

#[test]
fn fp32_and_int8_outputs_match_the_stored_fixture() {
    // From PCM, so this includes ultra-frontend's log-mel, whose x86 (AVX2 kernels, glibc)
    // and aarch64 (scalar, Apple libm) features differ in the last bits; the engine itself is
    // checked across ISAs by `encoder_outputs_from_reference_features_match_the_fixture`.
    if !cfg!(target_arch = "x86_64") {
        eprintln!("SKIPPED: PCM fixture recorded with the x86 frontend");
        return;
    }
    let (w, npz) = (common::weights_dir(), common::dev_clean_npz());
    if !common::have(&[&w, &npz]) {
        return;
    }
    let fixture =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/output_continuity.json");
    let mut model = Model::load(&w, true).expect("load");
    model.quantize_int8();
    let model = Arc::new(model);
    let utts = load_npz_audio_only(&npz, 0, Some(UTTS)).expect("npz");
    let mut entries = Vec::new();
    for prec in [Precision::F32, Precision::Int8] {
        let engine = Engine::new(model.clone(), Isa::detect(), prec).expect("engine");
        for (i, u) in utts.iter().enumerate() {
            let (tokens, enc) = run(&engine, &u.pcm);
            entries.push(serde_json::json!({"precision": prec.as_str(), "utt": i,
                "tokens": tokens, "enc_proj_sha256": enc}));
        }
    }
    // The 4-thread panel split must reproduce the single-thread int8 entries exactly.
    let pooled = Engine::new(model.clone(), Isa::detect(), Precision::Int8)
        .expect("engine")
        .with_gemm_threads(4);
    for (i, u) in utts.iter().enumerate() {
        let (tokens, enc) = run(&pooled, &u.pcm);
        let single = &entries[UTTS + i];
        assert_eq!(
            single["tokens"],
            serde_json::json!(tokens),
            "pool tokens utt {i}"
        );
        assert_eq!(
            single["enc_proj_sha256"],
            serde_json::json!(enc),
            "pool outputs utt {i}"
        );
    }
    let got = serde_json::json!({"weights_sha256": model.arch.weights_sha256, "entries": entries});
    if std::env::var("ULTRA_ENGINE_WRITE_CONTINUITY").as_deref() == Ok("1") {
        std::fs::write(&fixture, serde_json::to_string_pretty(&got).expect("json")).expect("write");
        return;
    }
    let want: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&fixture).expect("fixture")).expect("parse");
    if want["weights_sha256"] != got["weights_sha256"] {
        eprintln!("SKIPPED: fixture was recorded for other weights");
        return;
    }
    assert_eq!(
        got, want,
        "engine outputs moved: a numerical change needs the full quality gate"
    );
}

/// Engine-only continuity, identical on every ISA: fp32 and int8 encoder outputs (SHA-256)
/// and tokens when fed the torch reference features (no frontend).
#[test]
fn encoder_outputs_from_reference_features_match_the_fixture() {
    use ultra_engine::safetensors::SafeTensors;
    use ultra_frontend::MEL_BINS;
    let (w, r) = (common::weights_dir(), common::reference_file());
    if !common::have(&[&w, &r]) {
        return;
    }
    let fixture =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/encoder_continuity.json");
    let mut model = Model::load(&w, true).expect("load");
    model.quantize_int8();
    let model = Arc::new(model);
    let reference = SafeTensors::read(&r).expect("reference");
    let mut entries = Vec::new();
    for prec in [Precision::F32, Precision::Int8] {
        let engine = Engine::new(model.clone(), Isa::detect(), prec).expect("engine");
        for u in 0..3 {
            let feats: Vec<[f32; MEL_BINS]> = reference.tensors[&format!("u{u:03}.feats")]
                .data
                .chunks_exact(MEL_BINS)
                .map(|c| c.try_into().expect("80"))
                .collect();
            let mut ws = Workspace::default();
            let mut s = engine.create_stream();
            s.accept_features(&feats).expect("features");
            s.input_finished();
            let last = engine.model.arch.n_blocks + 2;
            let mut bytes: Vec<u8> = Vec::new();
            loop {
                let mut tap = |l: usize, v: &[f32], _w: usize| {
                    if l == last {
                        bytes.extend(v.iter().flat_map(|x| x.to_bits().to_le_bytes()));
                    }
                };
                let n = engine
                    .decode_tapped(&mut [&mut s], &mut ws, Some(&mut tap))
                    .expect("decode");
                if n == 0 {
                    break;
                }
            }
            entries.push(serde_json::json!({"precision": prec.as_str(), "utt": u,
                "tokens": s.tokens(), "enc_proj_sha256": sha256_hex(&bytes)}));
        }
    }
    let got = serde_json::json!({"weights_sha256": model.arch.weights_sha256,
        "reference_checkpoint_sha256": reference.metadata.get("checkpoint_sha256"),
        "entries": entries});
    if std::env::var("ULTRA_ENGINE_WRITE_CONTINUITY").as_deref() == Ok("1") {
        std::fs::write(&fixture, serde_json::to_string_pretty(&got).expect("json")).expect("write");
        return;
    }
    let want: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&fixture).expect("fixture")).expect("parse");
    if want["weights_sha256"] != got["weights_sha256"] {
        eprintln!("SKIPPED: fixture was recorded for other weights");
        return;
    }
    assert_eq!(got, want, "engine outputs moved (or differ across ISAs)");
}
