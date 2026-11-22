//! B6 decoder-only identity check: replay encoder projections written by the trainer's
//! torch encoder (a safetensors file of `[T, joiner]` f32 tensors, one per utterance id)
//! through the engine's beam search, and print `id<TAB>tokens` per utterance, to compare
//! with `train/beam.py` on the same projections (no encoder or frontend in the loop).
//!
//! `beam_replay <fp32 weights dir> <enc_p.safetensors> <beam> [int8]`

use ultra_engine::beam::{beam_chunk, BeamState};
use ultra_engine::decoder::DecWorkspace;
use ultra_engine::safetensors::SafeTensors;
use ultra_engine::{Isa, Model, Precision};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    assert!(
        args.len() >= 3,
        "usage: beam_replay <weights> <enc_p.safetensors> <beam> [int8]"
    );
    let mut model = Model::load(std::path::Path::new(&args[0]), true).expect("load weights");
    let prec = if args.get(3).map(String::as_str) == Some("int8") {
        model.quantize_int8();
        Precision::Int8
    } else {
        Precision::F32
    };
    let beam: usize = args[2].parse().expect("beam");
    let st = SafeTensors::read(std::path::Path::new(&args[1])).expect("read enc_p");
    let jn = model.arch.joiner;
    let mut ws = DecWorkspace::default();
    let mut ids: Vec<&String> = st.tensors.keys().collect();
    ids.sort();
    for id in ids {
        let t = &st.tensors[id];
        assert_eq!(t.data.len() % jn, 0, "{id}: not [T, {jn}]");
        let n = t.data.len() / jn;
        let mut b = BeamState::new(&model);
        beam_chunk(
            &model,
            Isa::detect(),
            prec,
            beam,
            &mut [&mut b],
            &[n],
            &t.data,
            &mut ws,
        )
        .expect("beam");
        let toks: Vec<String> = b.best().iter().map(u32::to_string).collect();
        println!("{id}\t{}", toks.join(","));
    }
}
