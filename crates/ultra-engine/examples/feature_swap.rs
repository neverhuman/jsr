//! Cross-host attribution: `dump <utt> <file>` writes this host's log-mel features of DEV
//! utterance `utt`; `run <file> <weights>` decodes such a file with int8 weights (fp32 if
//! the directory is fp32) and prints the tokens. Feeding one host's features to the other
//! separates frontend differences from engine differences. Uses JSR_DATA paths.

use std::sync::Arc;

use ultra_engine::{Engine, Isa, Model, Precision, Workspace};
use ultra_frontend::npz::load_npz_audio_only;
use ultra_frontend::{TrainerLogMel, MEL_BINS};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let root = std::path::PathBuf::from(
        std::env::var("JSR_DATA").unwrap_or_else(|_| "/mnt/xbabe3-external/jsr".into()),
    );
    match args.first().map(String::as_str) {
        Some("dump") => {
            let utt: usize = args[1].parse().expect("utt");
            let m = Model::load(&root.join("runs/engine/weights-r1-d1-ckpt7800-v3"), true)
                .expect("load");
            let u = &load_npz_audio_only(
                &root.join("cache/librispeech_pcm/dev-clean.npz"),
                utt,
                Some(1),
            )
            .expect("npz")[0];
            let rows = TrainerLogMel::new(m.mel_cfg.clone()).push_i16(&u.pcm);
            let b: Vec<u8> = rows
                .iter()
                .flatten()
                .flat_map(|x| x.to_le_bytes())
                .collect();
            std::fs::write(&args[2], b).expect("write");
        }
        Some("run") => {
            let b = std::fs::read(&args[1]).expect("read");
            let v: Vec<f32> = b
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let rows: Vec<[f32; MEL_BINS]> = v
                .chunks_exact(MEL_BINS)
                .map(|c| c.try_into().expect("80"))
                .collect();
            let mut m = Model::load(std::path::Path::new(&args[2]), true).expect("load");
            let prec = if m.int8_prepared.is_some() {
                Precision::Int8
            } else {
                Precision::F32
            };
            if prec == Precision::F32 && args.get(3).map(String::as_str) == Some("int8") {
                m.quantize_int8();
            }
            let prec = if args.get(3).map(String::as_str) == Some("int8") {
                Precision::Int8
            } else {
                prec
            };
            let e = Engine::new(Arc::new(m), Isa::detect(), prec).expect("engine");
            let mut s = e.create_stream();
            s.accept_features(&rows).expect("features");
            s.input_finished();
            let mut ws = Workspace::default();
            while e.decode(&mut [&mut s], &mut ws).expect("decode") > 0 {}
            println!("{} {:?}", Isa::detect().name(), s.tokens());
        }
        _ => eprintln!("usage: feature_swap dump <utt> <file> | run <file> <weights> [int8]"),
    }
}
