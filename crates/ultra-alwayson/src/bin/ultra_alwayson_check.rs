//! `ultra-alwayson-check`: CLI verifying sample conservation, sleep duty cycle, and
//! zero-mismatch isolated-segment identity (`--verify-identity`) for H47.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;

use serde_json::json;
use ultra_alwayson::{
    synthetic_stream_pcm, AlwaysOnConfig, AlwaysOnSession, AsrBackend, EngineBackend, MockBackend,
    SessionEvent,
};
use ultra_engine::{Engine, Isa, Model, Precision};
use ultra_frontend::wav::{read_wav_pcm, to_mono, SAMPLE_RATE_16K};
use ultra_gate::{parse_uclamp_max, set_thread_uclamp_max};

struct Cli {
    weights: Option<PathBuf>,
    wavs: Vec<PathBuf>,
    precision: Precision,
    chunk: Option<usize>,
    beam: usize,
    replay_ms: u32,
    idle_ms: u32,
    phase_ms: u32,
    packet_samples: usize,
    gate_enabled: bool,
    flush_on_sleep_close: bool,
    verify_identity: bool,
    uclamp_max: Option<u32>,
    ledger_out: Option<PathBuf>,
}

fn parse_cli() -> Result<Cli, String> {
    let mut it = std::env::args().skip(1);
    let mut c = Cli {
        weights: None,
        wavs: Vec::new(),
        precision: Precision::Int8,
        chunk: None,
        beam: 1,
        replay_ms: 500,
        idle_ms: 1500,
        phase_ms: 0,
        packet_samples: 320,
        gate_enabled: true,
        flush_on_sleep_close: false,
        verify_identity: true,
        uclamp_max: None,
        ledger_out: None,
    };
    while let Some(arg) = it.next() {
        let mut val = || it.next().ok_or_else(|| format!("{arg} requires a value"));
        match arg.as_str() {
            "--weights" => c.weights = Some(PathBuf::from(val()?)),
            "--wavs" => {
                c.wavs = val()?
                    .split(',')
                    .filter(|s| !s.is_empty())
                    .map(PathBuf::from)
                    .collect();
            }
            "--precision" => c.precision = Precision::parse(&val()?).map_err(|e| e.to_string())?,
            "--chunk" => c.chunk = Some(val()?.parse().map_err(|e| format!("--chunk: {e}"))?),
            "--beam" => c.beam = val()?.parse().map_err(|e| format!("--beam: {e}"))?,
            "--replay-ms" => {
                c.replay_ms = val()?.parse().map_err(|e| format!("--replay-ms: {e}"))?
            }
            "--idle-ms" => c.idle_ms = val()?.parse().map_err(|e| format!("--idle-ms: {e}"))?,
            "--phase-ms" => c.phase_ms = val()?.parse().map_err(|e| format!("--phase-ms: {e}"))?,
            "--packet-samples" => {
                c.packet_samples = val()?
                    .parse()
                    .map_err(|e| format!("--packet-samples: {e}"))?
            }
            "--no-gate" => c.gate_enabled = false,
            "--flush-on-sleep-close" => c.flush_on_sleep_close = true,
            "--verify-identity" => c.verify_identity = true,
            "--no-verify-identity" => c.verify_identity = false,
            "--uclamp-max" => {
                c.uclamp_max = Some(parse_uclamp_max(&val()?).map_err(|e| e.to_string())?)
            }
            "--ledger-out" => c.ledger_out = Some(PathBuf::from(val()?)),
            "-h" | "--help" => {
                return Err(
                    "usage: ultra-alwayson-check [--weights DIR --wavs a.wav,b.wav] \
                     [--precision int8|fp32] [--chunk N] [--beam K] [--replay-ms 500] \
                     [--idle-ms 1500] [--phase-ms 0] [--packet-samples 320] [--no-gate] \
                     [--flush-on-sleep-close] [--verify-identity] [--uclamp-max 0..1024] \
                     [--ledger-out path.jsonl]"
                        .into(),
                );
            }
            other => return Err(format!("unknown argument: {other}")),
        }
    }
    if c.packet_samples == 0 {
        return Err("--packet-samples must be > 0".into());
    }
    Ok(c)
}

fn load_wavs(paths: &[PathBuf]) -> Result<Vec<i16>, String> {
    let mut pcm = Vec::new();
    for p in paths {
        let bytes = std::fs::read(p).map_err(|e| format!("{}: {e}", p.display()))?;
        let wav = read_wav_pcm(&bytes).map_err(|e| format!("{}: {e}", p.display()))?;
        if wav.sample_rate_hz != SAMPLE_RATE_16K {
            return Err(format!(
                "{}: expected 16000 Hz PCM16, got {} Hz",
                p.display(),
                wav.sample_rate_hz
            ));
        }
        let mono = to_mono(wav.channels, &wav.samples).map_err(|e| e.to_string())?;
        pcm.extend_from_slice(&mono);
    }
    Ok(pcm)
}

fn run_check<B: AsrBackend>(
    backend: B,
    pcm: &[i16],
    cfg: AlwaysOnConfig,
    packet_samples: usize,
    verify_identity: bool,
    ledger_out: Option<&PathBuf>,
    mode: &str,
) -> Result<(), String> {
    let mut session = AlwaysOnSession::new(backend, cfg).map_err(|e| e.to_string())?;
    let mut noop = |_: SessionEvent<'_>| {};
    for chunk in pcm.chunks(packet_samples) {
        session
            .push_pcm(chunk, &mut noop)
            .map_err(|e| e.to_string())?;
    }
    session.finish(&mut noop).map_err(|e| e.to_string())?;

    let stats = session.stats();
    if stats.ring.samples_in != pcm.len() as u64 {
        return Err(format!(
            "sample conservation failed: samples_in={} != pcm_len={}",
            stats.ring.samples_in,
            pcm.len()
        ));
    }
    if stats.ring_underflows > 0 {
        return Err(format!(
            "ring underflow occurred {} time(s)",
            stats.ring_underflows
        ));
    }

    let mut mismatches = 0usize;
    let mut writer = match ledger_out {
        Some(p) => Some(BufWriter::new(
            File::create(p).map_err(|e| format!("{}: {e}", p.display()))?,
        )),
        None => None,
    };
    for (idx, seg) in session.ledger().iter().enumerate() {
        let (s0, e) = (seg.s0 as usize, seg.e as usize);
        if e > pcm.len() || s0 > e {
            return Err(format!(
                "segment {idx} out of bounds: [{s0}..{e}) vs {}",
                pcm.len()
            ));
        }
        if verify_identity {
            let iso = session
                .backend()
                .decode_isolated(&pcm[s0..e])
                .map_err(|err| format!("isolated decode failed for segment {idx}: {err}"))?;
            if iso != seg.tokens {
                mismatches += 1;
                eprintln!(
                    "MISMATCH seg={idx} s0={s0} e={e} frames={} session={:?} isolated={:?}",
                    seg.frames, seg.tokens, iso
                );
            }
        }
        if let Some(w) = writer.as_mut() {
            let row = json!({
                "segment": idx, "s0": seg.s0, "e": seg.e, "frames": seg.frames,
                "cause": format!("{:?}", seg.cause), "wake_decode_calls": seg.wake_decode_calls,
                "eof_flushed": seg.eof_flushed, "tokens": seg.tokens,
            });
            writeln!(w, "{row}").map_err(|e| e.to_string())?;
        }
    }

    let duty = if stats.total_samples_pushed > 0 {
        stats.recogniser_audio_samples as f64 / stats.total_samples_pushed as f64
    } else {
        0.0
    };
    let sleep_share = if stats.total_hops > 0 {
        stats.listening_hops as f64 / stats.total_hops as f64
    } else {
        0.0
    };
    println!(
        "{}",
        json!({
            "mode": mode, "gate_enabled": cfg.gate_enabled, "verify_identity": verify_identity,
            "samples_in": stats.ring.samples_in, "samples_replayed": stats.ring.samples_replayed,
            "samples_dropped": stats.ring.samples_dropped,
            "recogniser_audio_samples": stats.recogniser_audio_samples,
            "recogniser_duty": duty, "sleep_hop_share": sleep_share,
            "segments": session.ledger().len(), "wakes": stats.wake_count,
            "mismatches": mismatches, "conservation_ok": true,
        })
    );
    if mismatches > 0 {
        return Err(format!("{mismatches} segment identity mismatch(es)"));
    }
    Ok(())
}

fn main() -> ExitCode {
    let cli = match parse_cli() {
        Ok(c) => c,
        Err(msg) => {
            eprintln!("{msg}");
            return ExitCode::from(2);
        }
    };
    if let Some(u) = cli.uclamp_max {
        if let Err(err) = set_thread_uclamp_max(u) {
            eprintln!("uclamp note (--uclamp-max {u}): {err}; continuing unclamped");
        }
    }
    let cfg = AlwaysOnConfig {
        replay_ms: cli.replay_ms,
        idle_ms: cli.idle_ms,
        phase_ms: cli.phase_ms,
        gate_enabled: cli.gate_enabled,
        flush_on_sleep_close: cli.flush_on_sleep_close,
        ..AlwaysOnConfig::default()
    };

    let result = match (&cli.weights, cli.wavs.is_empty()) {
        (Some(weights), false) => (|| {
            let pcm = load_wavs(&cli.wavs)?;
            let model = Arc::new(
                Model::load(weights, cli.precision == Precision::Int8)
                    .map_err(|e| e.to_string())?,
            );
            let mut eng =
                Engine::new(model, Isa::detect(), cli.precision).map_err(|e| e.to_string())?;
            if let Some(ch) = cli.chunk {
                eng = eng.with_chunk(ch).map_err(|e| e.to_string())?;
            }
            eng = eng.with_beam(cli.beam);
            run_check(
                EngineBackend::new(eng),
                &pcm,
                cfg,
                cli.packet_samples,
                cli.verify_identity,
                cli.ledger_out.as_ref(),
                "engine",
            )
        })(),
        (None, true) => run_check(
            MockBackend::default(),
            &synthetic_stream_pcm(),
            cfg,
            cli.packet_samples,
            cli.verify_identity,
            cli.ledger_out.as_ref(),
            "synthetic-mock",
        ),
        _ => Err("provide both --weights and --wavs, or omit both for synthetic check".into()),
    };

    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("ultra-alwayson-check failed: {e}");
            ExitCode::FAILURE
        }
    }
}
