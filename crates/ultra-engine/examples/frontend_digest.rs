//! Cross-host digests that separate the frontend from the encoder: the log-mel features of
//! DEV utterance 0, and the fp32/int8 encoder outputs when fed the torch reference features
//! (no frontend). `cargo run --release --example frontend_digest` (uses JSR_DATA paths).

use std::sync::Arc;

use ultra_core::receipt::sha256_hex;
use ultra_engine::safetensors::SafeTensors;
use ultra_engine::{Engine, Isa, Model, Precision, Workspace};
use ultra_frontend::npz::load_npz_audio_only;
use ultra_frontend::{TrainerLogMel, MEL_BINS};

fn main() {
    let root = std::path::PathBuf::from(
        std::env::var("JSR_DATA").unwrap_or_else(|_| "/mnt/xbabe3-external/jsr".into()),
    );
    let w = root.join("runs/engine/weights-r1-d1-ckpt7800-v3");
    let mut model = Model::load(&w, true).expect("load");
    model.quantize_int8();
    let model = Arc::new(model);
    let utt = &load_npz_audio_only(
        &root.join("cache/librispeech_pcm/dev-clean.npz"),
        0,
        Some(1),
    )
    .expect("npz")[0];
    let mut mel = TrainerLogMel::new(model.mel_cfg.clone());
    let rows = mel.push_i16(&utt.pcm);
    let b: Vec<u8> = rows
        .iter()
        .flatten()
        .flat_map(|x| x.to_bits().to_le_bytes())
        .collect();
    println!(
        "frontend features {} rows {}",
        &sha256_hex(&b)[..16],
        rows.len()
    );
    let reference =
        SafeTensors::read(&root.join("runs/engine/ref-d1-dev50-v1/reference.safetensors"))
            .expect("ref");
    let feats: Vec<[f32; MEL_BINS]> = reference.tensors["u000.feats"]
        .data
        .chunks_exact(MEL_BINS)
        .map(|c| c.try_into().expect("80"))
        .collect();
    for prec in [Precision::F32, Precision::Int8] {
        let engine = Engine::new(model.clone(), Isa::detect(), prec).expect("engine");
        let mut s = engine.create_stream();
        s.accept_features(&feats).expect("feats");
        s.input_finished();
        let mut ws = Workspace::default();
        let last = model.arch.n_blocks + 2;
        let mut bytes = Vec::new();
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
        println!(
            "{} encoder-from-reference-features {} tokens {:?}",
            prec.as_str(),
            &sha256_hex(&bytes)[..16],
            &s.tokens()[..8]
        );
    }
}
