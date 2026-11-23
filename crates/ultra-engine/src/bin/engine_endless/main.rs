//! `ultra-engine-endless`: run a single continuous `EngineStream` over one or more sequential
//! 16 kHz mono PCM16 WAV files (e.g., the 27.6 h always-on endless stream `part-1..4.wav`),
//! emitting compact committed-word traces (`{"audio_s": t, "words": [...]}` matching
//! `score_alwayson.py`) and 1 Hz `/proc/self/status` `VmRSS`/`VmHWM` samples.
//!
//! ```text
//! ultra-engine-endless --weights <dir> --precision int8 \
//!     --wavs part-1.wav,part-2.wav,part-3.wav,part-4.wav \
//!     --trace-out trace.jsonl [--rss-out rss.jsonl] \
//!     [--beam 4 --beam-max-symbols 1 --lm M5.ulm --lm-weight 0.8 --lodr-weight 0.6 --blank-penalty 0] \
//!     [--endpoint-reset-ms 0] [--chunk 4] [--until-s 3600]
//! ```

mod cli;
mod committer;
mod wav;

use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

use serde_json::json;
use ultra_engine::{run_info, Engine, EngineStream, Model, Precision, Workspace};

use cli::Args;
use committer::{beam_lcp, WordCommitter};
use wav::open_wav_pcm16;

/// Immutable token prefix eligible for streaming word commits:
/// greedy mode uses `stream.tokens()` directly (byte-identical to legacy behavior),
/// while beam mode uses the longest common prefix across all active beam hypotheses.
fn active_prefix<'a>(engine: &Engine, stream: &'a EngineStream) -> &'a [u32] {
    if engine.beam > 1 {
        stream.dec.beam.as_ref().map_or(&[], |b| beam_lcp(&b.hyps))
    } else {
        stream.tokens()
    }
}

/// Read current `(VmRSS_kib, VmHWM_kib)` from `/proc/self/status` on Linux.
fn read_proc_rss_kib() -> (u64, u64) {
    let mut rss = 0u64;
    let mut hwm = 0u64;
    if let Ok(txt) = std::fs::read_to_string("/proc/self/status") {
        for line in txt.lines() {
            if let Some(rest) = line.strip_prefix("VmRSS:") {
                if let Some(tok) = rest.split_whitespace().next() {
                    rss = tok.parse().unwrap_or(0);
                }
            } else if let Some(rest) = line.strip_prefix("VmHWM:") {
                if let Some(tok) = rest.split_whitespace().next() {
                    hwm = tok.parse().unwrap_or(0);
                }
            }
        }
    }
    (rss, hwm)
}

fn run(a: &Args) -> Result<serde_json::Value, String> {
    if let Some(parent) = a.trace_out.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
    }
    let mut model = Model::load(&a.weights, true).map_err(|e| e.to_string())?;
    if a.precision == Precision::Int8 && model.int8_prepared.is_none() {
        model.quantize_int8();
    }
    let mut engine = Engine::new(Arc::new(model), a.isa, a.precision)
        .map_err(|e| e.to_string())?
        .with_beam(a.beam)
        .with_beam_max_symbols(a.beam_max_symbols)
        .with_kv_int8(a.kv_int8)
        .with_endpoint_reset(a.endpoint_reset_ms, a.endpoint_replay_ms);
    if let Some(c) = a.chunk {
        engine = engine.with_chunk(c).map_err(|e| e.to_string())?;
    }
    engine = cli::attach_lm(engine, a)?;

    let mut trace_w = BufWriter::new(File::create(&a.trace_out).map_err(|e| e.to_string())?);
    let mut rss_w = match &a.rss_out {
        Some(p) => {
            if let Some(parent) = p.parent() {
                if !parent.as_os_str().is_empty() {
                    std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
                }
            }
            Some(BufWriter::new(File::create(p).map_err(|e| e.to_string())?))
        }
        None => None,
    };

    let mut stream = engine.create_stream();
    let mut ws = Workspace::default();
    let mut committer = WordCommitter::new();
    let packet_samples = 16 * a.packet_ms;
    let mut raw_buf = vec![0u8; packet_samples * 2];
    let mut pcm_buf = vec![0i16; packet_samples];

    let max_samples: Option<u64> = a.until_s.map(|s| (s * 16_000.0).round() as u64);
    let mut total_samples: u64 = 0;

    let (c0, w0) = (run_info::thread_cpu_s(), Instant::now());
    let mut last_rss_wall = -1.0f64;
    let (rss0, _) = read_proc_rss_kib();
    let mut rss_peak_kib = rss0;

    'wav_loop: for wav_path in &a.wavs {
        let (mut reader, mut rem_bytes) = open_wav_pcm16(wav_path)?;
        while rem_bytes >= 2 {
            if let Some(cap) = max_samples {
                if total_samples >= cap {
                    break 'wav_loop;
                }
            }
            let want_samples = match max_samples {
                Some(cap) => (cap - total_samples).min(packet_samples as u64) as usize,
                None => packet_samples,
            };
            let want_bytes = (want_samples * 2).min(rem_bytes as usize) & !1;
            if want_bytes == 0 {
                break;
            }
            let mut got = 0usize;
            while got < want_bytes {
                let n = reader
                    .read(&mut raw_buf[got..want_bytes])
                    .map_err(|e| e.to_string())?;
                if n == 0 {
                    break;
                }
                got += n;
            }
            let got_samples = got / 2;
            if got_samples == 0 {
                break;
            }
            rem_bytes -= (got_samples * 2) as u64;
            for (dst, chunk) in pcm_buf[..got_samples]
                .iter_mut()
                .zip(raw_buf[..got_samples * 2].chunks_exact(2))
            {
                *dst = i16::from_le_bytes([chunk[0], chunk[1]]);
            }
            stream
                .accept_i16(&pcm_buf[..got_samples])
                .map_err(|e| e.to_string())?;
            total_samples += got_samples as u64;
            let audio_s = total_samples as f64 / 16_000.0;

            while engine
                .decode(&mut [&mut stream], &mut ws)
                .map_err(|e| e.to_string())?
                > 0
            {
                committer.update(
                    &engine.model.tokens,
                    active_prefix(&engine, &stream),
                    audio_s,
                    &mut trace_w,
                )?;
            }

            let wall_now = w0.elapsed().as_secs_f64();
            if wall_now - last_rss_wall >= 1.0 {
                last_rss_wall = wall_now;
                let (rss_kib, hwm_kib) = read_proc_rss_kib();
                rss_peak_kib = rss_peak_kib.max(rss_kib).max(hwm_kib);
                if let Some(rw) = rss_w.as_mut() {
                    writeln!(
                        rw,
                        "{}",
                        json!({
                            "wall_s": (wall_now * 100.0).round() / 100.0,
                            "audio_s": (audio_s * 100.0).round() / 100.0,
                            "rss_mb": (rss_kib as f64 / 1024.0 * 100.0).round() / 100.0,
                            "hwm_mb": (hwm_kib as f64 / 1024.0 * 100.0).round() / 100.0,
                            "resets": stream.endpoint_resets(),
                            "words": committer.total_words()
                        })
                    )
                    .map_err(|e| e.to_string())?;
                }
            }
        }
    }

    stream.input_finished();
    let final_audio_s = total_samples as f64 / 16_000.0;
    while engine
        .decode(&mut [&mut stream], &mut ws)
        .map_err(|e| e.to_string())?
        > 0
    {
        committer.update(
            &engine.model.tokens,
            active_prefix(&engine, &stream),
            final_audio_s,
            &mut trace_w,
        )?;
    }
    committer.finish(
        &engine.model.tokens,
        stream.tokens(),
        final_audio_s,
        &mut trace_w,
    )?;
    trace_w.flush().map_err(|e| e.to_string())?;
    if let Some(rw) = rss_w.as_mut() {
        rw.flush().map_err(|e| e.to_string())?;
    }

    let cpu_s = run_info::thread_cpu_s() - c0;
    let wall_s = w0.elapsed().as_secs_f64();
    let (final_rss, final_hwm) = read_proc_rss_kib();
    rss_peak_kib = rss_peak_kib.max(final_rss).max(final_hwm);
    let audio_h = final_audio_s / 3600.0;
    let rss_start_mb = rss0 as f64 / 1024.0;
    let rss_end_mb = final_rss as f64 / 1024.0;
    let rss_peak_mb = rss_peak_kib as f64 / 1024.0;

    Ok(json!({
        "audio_s": final_audio_s,
        "audio_h": audio_h,
        "cpu_s": cpu_s,
        "wall_s": wall_s,
        "cpu_s_per_audio_s": if final_audio_s > 0.0 { cpu_s / final_audio_s } else { 0.0 },
        "rtfx": if cpu_s > 0.0 { final_audio_s / cpu_s } else { 0.0 },
        "beam": engine.beam,
        "beam_max_symbols": engine.beam_max_symbols,
        "lm": a.lm_path,
        "lm_weight": a.lm_weight,
        "lodr_weight": a.lodr_weight,
        "blank_penalty": a.blank_penalty,
        "endpoint_reset_ms": a.endpoint_reset_ms,
        "endpoint_replay_ms": a.endpoint_replay_ms,
        "endpoint_resets": stream.endpoint_resets(),
        "words": committer.total_words(),
        "tokens": stream.tokens().len(),
        "rss_start_mb": rss_start_mb,
        "rss_end_mb": rss_end_mb,
        "rss_peak_mb": rss_peak_mb,
        "rss_delta_mb": rss_end_mb - rss_start_mb,
        "trace_out": a.trace_out,
        "rss_out": a.rss_out,
    }))
}

fn main() -> ExitCode {
    match cli::parse().and_then(|a| run(&a)) {
        Ok(s) => {
            println!("{}", serde_json::to_string_pretty(&s).unwrap_or_default());
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("ultra-engine-endless: {e}");
            ExitCode::FAILURE
        }
    }
}
