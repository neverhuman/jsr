//! `ultra-engine-density`: paced live-stream density of ONE engine worker (pin the process
//! to one core with `taskset`; one worker per CCX). `--streams N` streams replay DEV
//! utterances back to back in real time (staggered starts); every `--grid-ms` the worker
//! ingests the audio that has "arrived" and decodes every ready chunk (batched). Per chunk,
//! lateness = decode end - arrival of the last sample that chunk needs. Reports p50/p99/max
//! lateness, a no-growth check (second-half mean minus first-half mean), worker CPU per
//! audio second, and pass/fail against the A1 rule (p99 <= 800 ms, no growth). Exploratory.
//!
//! A DEVELOPMENT tool: evidence rows come from `ultra bench-suite` (one harness, one A1
//! predicate for every system). Every knob is a recorded flag (`--int8-fp32-kinds`,
//! `--int8-stem-kinds`); `ULTRA_ENGINE_*` environment variables are refused; `--qualifying`
//! refuses runtime weight quantisation.

use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde_json::json;
use ultra_engine::{run_info, Engine, EngineStream, Isa, Model, Precision, Workspace};
use ultra_frontend::npz::load_npz_audio_only;

struct Args {
    weights: PathBuf,
    npz: PathBuf,
    streams: usize,
    seconds: f64,
    grid_ms: u64,
    precision: Precision,
    max_chunks: usize,
    int8_stem: bool,
    int8_stem_kinds: Vec<String>,
    int8_skip: Vec<String>,
    qualifying: bool,
}

fn parse() -> Result<Args, String> {
    let mut a = Args {
        weights: PathBuf::new(),
        npz: PathBuf::new(),
        streams: 8,
        seconds: 60.0,
        grid_ms: 320,
        precision: Precision::Int8,
        max_chunks: 1,
        int8_stem: false,
        int8_stem_kinds: Vec::new(),
        int8_skip: Vec::new(),
        qualifying: false,
    };
    let mut it = std::env::args().skip(1);
    while let Some(k) = it.next() {
        let mut v = || it.next().ok_or_else(|| format!("{k} needs a value"));
        match k.as_str() {
            "--weights" => a.weights = v()?.into(),
            "--npz" => a.npz = v()?.into(),
            "--streams" => a.streams = v()?.parse().map_err(|e| format!("{e}"))?,
            "--seconds" => a.seconds = v()?.parse().map_err(|e| format!("{e}"))?,
            "--int8-stem" => a.int8_stem = true,
            "--max-chunks" => a.max_chunks = v()?.parse().map_err(|e| format!("{e}"))?,
            "--grid-ms" => a.grid_ms = v()?.parse().map_err(|e| format!("{e}"))?,
            "--precision" => a.precision = Precision::parse(&v()?).map_err(|e| e.to_string())?,
            "--int8-fp32-kinds" => a.int8_skip = kinds(&v()?),
            "--int8-stem-kinds" => {
                a.int8_stem_kinds = kinds(&v()?);
                a.int8_stem = true;
            }
            "--qualifying" => a.qualifying = true,
            o => return Err(format!("unknown argument {o}")),
        }
    }
    if a.streams == 0 || a.grid_ms == 0 || a.weights.as_os_str().is_empty() {
        return Err("--weights, positive --streams and --grid-ms are required".into());
    }
    if a.int8_stem && a.int8_stem_kinds.is_empty() {
        a.int8_stem_kinds = vec!["stem_c2".into(), "stem_out".into()];
    }
    Ok(a)
}

fn kinds(v: &str) -> Vec<String> {
    v.split(',')
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
        .collect()
}

struct Slot {
    utt: usize,
    stream: EngineStream,
    /// Wall-clock offset (s) at which this utterance's sample 0 arrives.
    start: f64,
    fed: usize,
}

/// Decode every ready step (batched) until none is ready. Records one lateness sample per
/// processed chunk: decode end minus the arrival of the last sample that chunk needs
/// (feature frame `4 t1 - 4` complete; only the stream's final, padded chunk is capped at the
/// true end of audio). Returns (decode calls, encoder frames processed).
fn decode_ready(
    engine: &Engine,
    slots: &mut [Slot],
    utts: &[ultra_frontend::npz::AudioUtterance],
    ws: &mut Workspace,
    t0: Instant,
    late: &mut Vec<(f64, f64)>,
) -> Result<(usize, usize), String> {
    let c = engine.model.arch.chunk;
    let (mut calls, mut frames) = (0usize, 0usize);
    loop {
        let mut arrivals: Vec<f64> = Vec::new();
        for s in slots.iter() {
            if let Some((a0, a1)) = engine.next_chunk(&s.stream) {
                frames += a1 - a0;
                let len = utts[s.utt].pcm.len();
                let mut e = a0;
                while e < a1 {
                    e = (e + c).min(a1);
                    let need = (4 * e - 4) * 160 + 400;
                    let need = if s.stream.is_finished() {
                        need.min(len)
                    } else {
                        need
                    };
                    arrivals.push(s.start + need as f64 / 16_000.0);
                }
            }
        }
        if arrivals.is_empty() {
            return Ok((calls, frames));
        }
        let mut refs: Vec<&mut EngineStream> = slots.iter_mut().map(|s| &mut s.stream).collect();
        engine.decode(&mut refs, ws).map_err(|e| e.to_string())?;
        calls += 1;
        let end = t0.elapsed().as_secs_f64();
        late.extend(arrivals.into_iter().map(|t| (t, end - t)));
    }
}

fn pct(v: &mut [f64], p: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(f64::total_cmp);
    v[((v.len() - 1) as f64 * p).round() as usize]
}

fn run(a: &Args) -> Result<serde_json::Value, String> {
    run_info::refuse_engine_env()?;
    let mut model = Model::load(&a.weights, true).map_err(|e| e.to_string())?;
    if a.qualifying {
        run_info::check_qualifying(&model, a.precision, a.int8_stem)?;
    }
    if model.int8_prepared.is_some() && !a.int8_skip.is_empty() {
        return Err("--int8-fp32-kinds conflicts with a prepared-INT8 directory".into());
    }
    if a.precision == Precision::Int8 && model.int8_prepared.is_none() {
        model.quantize_int8_where(|k, _| !a.int8_skip.iter().any(|s| s == k));
    }
    let stem_quantised = if a.int8_stem && a.precision == Precision::Int8 {
        let kinds: Vec<&str> = a.int8_stem_kinds.iter().map(String::as_str).collect();
        model.quantize_fp32_kinds_int8(&kinds)
    } else {
        0
    };
    let engine = Engine::new(Arc::new(model), Isa::detect(), a.precision)
        .map_err(|e| e.to_string())?
        .with_max_chunks(a.max_chunks);
    let utts = load_npz_audio_only(&a.npz, 0, Some(400)).map_err(|e| e.to_string())?;
    let grid = a.grid_ms as f64 / 1000.0;
    let mut slots: Vec<Slot> = (0..a.streams)
        .map(|i| Slot {
            utt: (i * 37) % utts.len(),
            stream: engine.create_stream(),
            start: grid * i as f64 / a.streams as f64,
            fed: 0,
        })
        .collect();
    let mut next_utt = a.streams * 37;
    let mut ws = Workspace::default();
    let mut late: Vec<(f64, f64)> = Vec::new();
    let (c0, t0) = (run_info::thread_cpu_s(), Instant::now());
    let mut audio_s = 0.0f64;
    let mut frames = 0usize;
    let mut tick = 1u64;
    let mut cutoff = false;
    loop {
        let target = Duration::from_secs_f64(grid * tick as f64);
        if let Some(d) = target.checked_sub(t0.elapsed()) {
            std::thread::sleep(d);
        }
        let mut now = t0.elapsed().as_secs_f64();
        if now > a.seconds {
            // Ingest exactly the audio that arrived by the cutoff, then drain it below.
            now = a.seconds;
            cutoff = true;
        }
        for s in &mut slots {
            let pcm = &utts[s.utt].pcm;
            let avail = (((now - s.start).max(0.0)) * 16_000.0) as usize;
            let upto = avail.min(pcm.len());
            if upto > s.fed {
                s.stream
                    .accept_i16(&pcm[s.fed..upto])
                    .map_err(|e| e.to_string())?;
                audio_s += (upto - s.fed) as f64 / 16_000.0;
                s.fed = upto;
                if upto == pcm.len() {
                    s.stream.input_finished();
                }
            }
        }
        if cutoff {
            break;
        }
        let (_, f) = decode_ready(&engine, &mut slots, &utts, &mut ws, t0, &mut late)?;
        frames += f;
        for s in &mut slots {
            if s.stream.is_finished() && !engine.is_ready(&s.stream) {
                // Next utterance starts right after this one ended (real-time continuity).
                let end = s.start + utts[s.utt].pcm.len() as f64 / 16_000.0;
                s.utt = next_utt % utts.len();
                next_utt += 1;
                s.stream = engine.create_stream();
                s.start = end;
                s.fed = 0;
            }
        }
        tick += 1;
    }
    // Drain: every chunk whose audio arrived by the cutoff is decoded and its lateness kept.
    let backlog_steps = slots.iter().filter(|s| engine.is_ready(&s.stream)).count();
    let n_before = late.len();
    let (_, f) = decode_ready(&engine, &mut slots, &utts, &mut ws, t0, &mut late)?;
    frames += f;
    let drained_chunks = late.len() - n_before;
    // Accepted audio not yet forming a complete chunk (never processed, never credited).
    let processed_s = frames as f64 * 0.04;
    let cpu = run_info::thread_cpu_s() - c0;
    let wall = t0.elapsed().as_secs_f64();
    let half = a.seconds / 2.0;
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len().max(1) as f64;
    let first: Vec<f64> = late
        .iter()
        .filter(|(t, _)| *t < half)
        .map(|x| x.1)
        .collect();
    let second: Vec<f64> = late
        .iter()
        .filter(|(t, _)| *t >= half)
        .map(|x| x.1)
        .collect();
    let growth = mean(&second) - mean(&first);
    let mut all: Vec<f64> = late.iter().map(|x| x.1).collect();
    let (p50, p99) = (pct(&mut all, 0.5), pct(&mut all, 0.99));
    let mx = all.last().copied().unwrap_or(f64::NAN);
    let pass = p99 <= 0.8 && growth <= 0.05;
    let fp32_kinds = match &engine.model.int8_prepared {
        Some(p) => p.fp32_kinds.clone(),
        None => a.int8_skip.clone(),
    };
    let knobs = run_info::Knobs {
        mode: if a.qualifying {
            "qualifying"
        } else {
            "exploratory"
        },
        precision: a.precision,
        isa: engine.isa.name(),
        max_chunks: a.max_chunks,
        act7: false,
        gemm_threads: 1,
        int8_fp32_kinds: &fp32_kinds,
        int8_stem_kinds: if a.int8_stem { &a.int8_stem_kinds } else { &[] },
        int8_stem_layers_quantised: stem_quantised,
    };
    let mut out = run_info::identity(&engine.model, &a.weights, &knobs);
    let extra = json!({
        "schema": "ultra.engine-density/v3", "tool": "development (not an evidence harness)",
        "npz": a.npz, "streams": a.streams, "grid_ms": a.grid_ms, "seconds": a.seconds,
        "cpu_scope": "decode thread only (schedstat), load excluded",
        "peak_rss_mib": run_info::peak_rss_mib(),
        "chunks": all.len(), "lateness_p50_s": p50, "lateness_p99_s": p99, "lateness_max_s": mx,
        "growth_s": growth, "a1_pass": pass,
        "worker_cpu_s": cpu, "wall_s": wall, "worker_util": cpu / wall,
        "workload": "duration-limited live replay: audio arriving after the cutoff is not \
                     accepted; everything accepted that forms a chunk is drained and scored",
        "accepted_audio_s": audio_s, "processed_audio_s": processed_s,
        "unprocessed_accepted_audio_s": (audio_s - processed_s).max(0.0),
        "backlog_steps_at_cutoff": backlog_steps, "drained_chunks": drained_chunks,
        "cpu_s_per_audio_s": cpu / processed_s,
    });
    if let (Some(m), Some(x)) = (out.as_object_mut(), extra.as_object()) {
        m.extend(x.clone());
    }
    Ok(out)
}

fn main() -> ExitCode {
    match parse().and_then(|a| run(&a)) {
        Ok(v) => {
            println!("{}", serde_json::to_string(&v).unwrap_or_default());
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("ultra-engine-density: {e}");
            ExitCode::FAILURE
        }
    }
}
