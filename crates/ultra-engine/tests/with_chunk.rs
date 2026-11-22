//! Verify `Engine::with_chunk(usize)`: validation, bit-identical equivalence at the exported
//! chunk, and batch-invariant multi-chunk decoding at `chunk = 8` (320 ms) and `16` (640 ms).

mod common;

use std::sync::Arc;

use ultra_engine::{Engine, EngineStream, Isa, Model, Precision, Workspace};
use ultra_frontend::npz::load_npz_audio_only;

fn decode_batch(engine: &Engine, clips: &[Vec<i16>], batch: usize) -> Vec<(Vec<f32>, Vec<u32>)> {
    let mut ws = Workspace::default();
    let mut streams: Vec<EngineStream> = clips.iter().map(|_| engine.create_stream()).collect();
    let mut enc: Vec<Vec<f32>> = vec![Vec::new(); clips.len()];
    let mut cur = vec![0usize; clips.len()];
    let packet = 320usize;
    loop {
        let mut progressed = false;
        for (i, s) in streams.iter_mut().enumerate() {
            if cur[i] < clips[i].len() {
                let end = (cur[i] + packet).min(clips[i].len());
                s.accept_i16(&clips[i][cur[i]..end]).expect("accept");
                cur[i] = end;
                if end == clips[i].len() {
                    s.input_finished();
                }
                progressed = true;
            }
        }
        loop {
            let ready: Vec<usize> = (0..streams.len())
                .filter(|&k| engine.is_ready(&streams[k]))
                .collect();
            if ready.is_empty() {
                break;
            }
            for group in ready.chunks(batch) {
                let ns: Vec<usize> = group
                    .iter()
                    .map(|&k| {
                        let (t0, t1) = engine.next_chunk(&streams[k]).expect("ready");
                        t1 - t0
                    })
                    .collect();
                let mut rows: Vec<f32> = Vec::new();
                let width = engine.model.arch.joiner;
                let last = engine.model.arch.n_blocks + 2;
                let mut tap = |l: usize, v: &[f32], _w: usize| {
                    if l == last {
                        rows.extend_from_slice(v);
                    }
                };
                let mut refs: Vec<&mut EngineStream> = streams
                    .iter_mut()
                    .enumerate()
                    .filter(|(k, _)| group.contains(k))
                    .map(|(_, s)| s)
                    .collect();
                engine
                    .decode_tapped(&mut refs, &mut ws, Some(&mut tap))
                    .expect("decode");
                let mut off = 0;
                for (&k, &nk) in group.iter().zip(&ns) {
                    enc[k].extend_from_slice(&rows[off * width..(off + nk) * width]);
                    off += nk;
                }
                progressed = true;
            }
        }
        if !progressed && cur.iter().zip(clips).all(|(c, p)| *c >= p.len()) {
            break;
        }
    }
    enc.into_iter()
        .zip(&streams)
        .map(|(e, s)| (e, s.tokens().to_vec()))
        .collect()
}

#[test]
fn with_chunk_validates_bounds_matches_default_bit_exact_and_supports_8_and_16() {
    let weights = common::weights_dir();
    let npz = common::dev_clean_npz();
    if !common::have(&[&weights, &npz]) {
        return;
    }
    let mut model = Model::load(&weights, false).expect("load model");
    model.quantize_int8();
    let model = Arc::new(model);
    let default_chunk = model.arch.chunk;

    // Bounds validation
    assert!(Engine::new(model.clone(), Isa::detect(), Precision::Int8)
        .unwrap()
        .with_chunk(0)
        .is_err());
    assert!(Engine::new(model.clone(), Isa::detect(), Precision::Int8)
        .unwrap()
        .with_chunk(65)
        .is_err());

    let clips: Vec<Vec<i16>> = load_npz_audio_only(&npz, 0, Some(4))
        .expect("npz")
        .into_iter()
        .map(|u| u.pcm)
        .collect();

    // 1) with_chunk(model.arch.chunk) is bit-identical to default Engine::new
    let eng_default = Engine::new(model.clone(), Isa::detect(), Precision::Int8).unwrap();
    let eng_same = Engine::new(model.clone(), Isa::detect(), Precision::Int8)
        .unwrap()
        .with_chunk(default_chunk)
        .unwrap();
    let tr_def = decode_batch(&eng_default, &clips, 1);
    let tr_same = decode_batch(&eng_same, &clips, 1);
    for (i, ((ed, td), (es, ts))) in tr_def.iter().zip(&tr_same).enumerate() {
        assert_eq!(td, ts, "clip {i}: tokens differ at default chunk");
        assert_eq!(
            ed.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            es.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "clip {i}: encoder output bits differ at default chunk"
        );
    }

    // 2) chunk = 8 (320 ms) and chunk = 16 (640 ms) decode cleanly & are batch-invariant (B=1 vs B=4)
    for &c in &[8usize, 16usize] {
        let eng = Engine::new(model.clone(), Isa::detect(), Precision::Int8)
            .unwrap()
            .with_chunk(c)
            .unwrap();
        assert_eq!(eng.chunk, c);
        let b1 = decode_batch(&eng, &clips, 1);
        let b4 = decode_batch(&eng, &clips, 4);
        for (i, ((e1, t1), (e4, t4))) in b1.iter().zip(&b4).enumerate() {
            assert!(!e1.is_empty(), "clip {i} chunk {c}: empty encoder output");
            assert_eq!(t1, t4, "clip {i} chunk {c}: batch-4 tokens differ from B=1");
            assert_eq!(
                e1.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                e4.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "clip {i} chunk {c}: batch-4 encoder bits differ from B=1"
            );
        }
    }
}
