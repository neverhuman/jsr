//! E2 acceptance: a stream's encoder outputs and tokens are bit-identical whatever batch it
//! shares (B = 1, 3, 5, 16, 32), with streams at mixed positions (staggered starts, different
//! lengths) and partial final chunks batched next to full chunks. Checked for int8 (per-row
//! dynamic quantisation) and fp32.

mod common;

use std::sync::Arc;

use ultra_engine::{Engine, EngineStream, Isa, Model, Precision, Workspace};
use ultra_frontend::npz::load_npz_audio_only;

const STREAMS: usize = 32;

/// Per stream: every encoder-projection row it produced (in order) and its tokens.
type Trace = Vec<(Vec<f32>, Vec<u32>)>;

/// Run all clips with at most `batch` streams decoded together; stream `i` joins at round
/// `i % 7` (staggered positions). Returns each stream's trace.
fn run(engine: &Engine, clips: &[Vec<i16>], batch: usize, packet: usize) -> Trace {
    let mut ws = Workspace::default();
    let mut streams: Vec<EngineStream> = clips.iter().map(|_| engine.create_stream()).collect();
    let mut cur = vec![0usize; clips.len()];
    let mut enc: Vec<Vec<f32>> = vec![Vec::new(); clips.len()];
    let mut round = 0usize;
    loop {
        let mut progressed = false;
        for (i, s) in streams.iter_mut().enumerate() {
            if round >= i % 7 && cur[i] < clips[i].len() {
                let end = (cur[i] + packet).min(clips[i].len());
                s.accept_i16(&clips[i][cur[i]..end]).expect("accept");
                cur[i] = end;
                if end == clips[i].len() {
                    s.input_finished();
                }
                progressed = true;
            }
        }
        // Decode the ready streams in groups of `batch`, rotating the start so groups mix.
        loop {
            let ready: Vec<usize> = (0..streams.len())
                .map(|k| (k + round) % streams.len())
                .filter(|&k| engine.is_ready(&streams[k]))
                .collect();
            if ready.is_empty() {
                break;
            }
            for group in ready.chunks(batch) {
                let mut order: Vec<usize> = group.to_vec();
                order.sort_unstable();
                let ns: Vec<usize> = order
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
                    .filter(|(k, _)| order.contains(k))
                    .map(|(_, s)| s)
                    .collect();
                let n = engine
                    .decode_tapped(&mut refs, &mut ws, Some(&mut tap))
                    .expect("decode");
                assert_eq!(n, order.len());
                let mut off = 0;
                for (&k, &nk) in order.iter().zip(&ns) {
                    enc[k].extend_from_slice(&rows[off * width..(off + nk) * width]);
                    off += nk;
                }
                progressed = true;
            }
        }
        round += 1;
        if !progressed && cur.iter().zip(clips).all(|(c, p)| *c >= p.len()) {
            break;
        }
    }
    enc.into_iter()
        .zip(&streams)
        .map(|(e, s)| (e, s.tokens().to_vec()))
        .collect()
}

fn check(prec: Precision) {
    check_beam(prec, 1);
}

/// `beam` 1 = greedy; > 1 = bounded beam search (tokens must also be batch-invariant).
fn check_beam(prec: Precision, beam: usize) {
    let (w, npz) = (common::weights_dir(), common::dev_clean_npz());
    if !common::have(&[&w, &npz]) {
        return;
    }
    let mut model = Model::load(&w, true).expect("load");
    if prec == Precision::Int8 {
        model.quantize_int8();
    }
    let engine = Engine::new(Arc::new(model), Isa::detect(), prec)
        .expect("engine")
        .with_beam(beam);
    let utts = load_npz_audio_only(&npz, 0, Some(STREAMS)).expect("npz");
    // 0.9 - 2.4 s clips with lengths that leave every partial-chunk size (1..=3 frames).
    let clips: Vec<Vec<i16>> = utts
        .iter()
        .enumerate()
        .map(|(i, u)| u.pcm[..(14_400 + 1_280 * i + 640 * (i % 4)).min(u.pcm.len())].to_vec())
        .collect();
    let base = run(&engine, &clips, 1, 960);
    assert!(base.iter().all(|(e, _)| !e.is_empty()));
    // Multi-chunk steps (1 s packets: several chunks ready at once) must not change a bit.
    let multi = engine.clone().with_max_chunks(4);
    let runs = [
        (&engine, 3, 960),
        (&engine, 5, 960),
        (&engine, 16, 960),
        (&engine, 32, 960),
        (&multi, 1, 16_000),
        (&multi, 5, 16_000),
        (&multi, 32, 5_120),
    ];
    for (eng, batch, packet) in runs {
        let got = run(eng, &clips, batch, packet);
        for (i, (g, b)) in got.iter().zip(&base).enumerate() {
            assert_eq!(
                g.1,
                b.1,
                "{} tokens differ: stream {i}, batch {batch}",
                prec.as_str()
            );
            assert!(
                g.0.len() == b.0.len()
                    && g.0
                        .iter()
                        .zip(&b.0)
                        .all(|(x, y)| x.to_bits() == y.to_bits()),
                "{} encoder rows differ: stream {i}, batch {batch}",
                prec.as_str()
            );
        }
    }
}

#[test]
fn int8_is_batch_invariant() {
    check(Precision::Int8);
}

#[test]
fn fp32_is_batch_invariant() {
    check(Precision::F32);
}

#[test]
fn int8_beam4_is_batch_invariant() {
    check_beam(Precision::Int8, 4);
}

#[test]
fn kv_int8_reduces_retained_bytes_over_3x_and_is_batch_invariant() {
    let (w, npz) = (common::weights_dir(), common::dev_clean_npz());
    if !common::have(&[&w, &npz]) {
        return;
    }
    let mut model = Model::load(&w, true).expect("load");
    model.quantize_int8();
    let model = Arc::new(model);
    let e_f32 = Engine::new(model.clone(), Isa::detect(), Precision::Int8).expect("engine");
    let e_kv8 = e_f32.clone().with_kv_int8(true);
    let s_f32 = e_f32.create_stream();
    let s_kv8 = e_kv8.create_stream();
    assert!(
        s_f32.enc.retained_bytes() > s_kv8.enc.retained_bytes() * 3,
        "expected >3x K/V memory reduction: fp32={} vs kv_int8={}",
        s_f32.enc.retained_bytes(),
        s_kv8.enc.retained_bytes()
    );
    let utts = load_npz_audio_only(&npz, 0, Some(4)).expect("npz");
    let clips: Vec<Vec<i16>> = utts
        .iter()
        .map(|u| u.pcm[..u.pcm.len().min(24_000)].to_vec())
        .collect();
    let b1 = run(&e_kv8, &clips, 1, 960);
    let b4 = run(&e_kv8, &clips, 4, 960);
    assert_eq!(b1, b4, "kv_int8 must be strictly batch-invariant");
}

#[test]
fn lifecycle_refuses_audio_after_finish() {
    let w = common::weights_dir();
    if !common::have(&[&w]) {
        return;
    }
    let engine = Engine::new(
        Arc::new(Model::load(&w, true).expect("load")),
        Isa::detect(),
        Precision::F32,
    )
    .expect("engine");
    let mut s = engine.create_stream();
    s.accept_i16(&[0i16; 1600])
        .expect("open stream accepts audio");
    s.input_finished();
    assert!(s.accept_i16(&[0i16; 160]).is_err());
    assert!(s.accept_f32(&[0.0; 160]).is_err());
}
