//! `ultra-engine-agc`: standalone streaming recogniser CLI exposing `--warm-agc` (default off).
//!
//! ```text
//! ultra-engine-agc --weights <dir> --wav <16k-mono.wav> \
//!     [--precision int8|fp32] [--chunk N] [--beam K] [--packet-ms 20] \
//!     [--warm-agc] [--preroll-ms 500] [--target-rms 0.08] \
//!     [--min-gain-db -12.0] [--max-gain-db 40.0] [--speech-floor-dbfs -65.0]
//! ```

use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

use serde_json::json;
use ultra_engine::{run_info, AgcStream, Engine, Isa, Model, Precision, WarmAgcConfig, Workspace};
use ultra_frontend::read_audio_16k_mono;

struct Args {
    weights: PathBuf,
    wav: PathBuf,
    precision: Precision,
    chunk: Option<usize>,
    beam: usize,
    packet_ms: usize,
    preroll_ms: usize,
    agc: WarmAgcConfig,
    isa: Isa,
    help: bool,
}

fn usage() -> &'static str {
    "usage: ultra-engine-agc --weights <dir> --wav <16k-mono.wav> \
     [--precision int8|fp32] [--chunk N] [--beam K] [--packet-ms 20] \
     [--warm-agc] [--preroll-ms 500] [--tau-s 5.0] [--target-rms 0.08] \
     [--min-gain-db -12.0] [--max-gain-db 40.0] [--speech-floor-dbfs -65.0]"
}

fn parse() -> Result<Args, String> {
    let mut it = std::env::args().skip(1);
    let mut a = Args {
        weights: PathBuf::new(),
        wav: PathBuf::new(),
        precision: Precision::Int8,
        chunk: None,
        beam: 1,
        packet_ms: 20,
        preroll_ms: 500,
        agc: WarmAgcConfig::default(),
        isa: Isa::detect(),
        help: false,
    };
    while let Some(k) = it.next() {
        let mut v = || it.next().ok_or_else(|| format!("{k} needs a value"));
        match k.as_str() {
            "-h" | "--help" => {
                a.help = true;
                return Ok(a);
            }
            "--weights" => a.weights = v()?.into(),
            "--wav" => a.wav = v()?.into(),
            "--precision" => a.precision = Precision::parse(&v()?).map_err(|e| e.to_string())?,
            "--chunk" => a.chunk = Some(v()?.parse().map_err(|e| format!("--chunk: {e}"))?),
            "--beam" => a.beam = v()?.parse().map_err(|e| format!("--beam: {e}"))?,
            "--packet-ms" => {
                a.packet_ms = v()?.parse().map_err(|e| format!("--packet-ms: {e}"))?;
            }
            "--preroll-ms" => {
                a.preroll_ms = v()?.parse().map_err(|e| format!("--preroll-ms: {e}"))?;
            }
            "--warm-agc" => a.agc.warm_agc = true,
            "--no-warm-agc" => a.agc.warm_agc = false,
            "--tau-s" => {
                let t: f32 = v()?.parse().map_err(|e| format!("--tau-s: {e}"))?;
                (a.agc.tau_s, a.agc.attack_tau_s) = (t, t);
            }
            "--target-rms" => {
                a.agc.target_rms = v()?.parse().map_err(|e| format!("--target-rms: {e}"))?;
            }
            "--min-gain-db" => {
                a.agc.min_gain_db = v()?.parse().map_err(|e| format!("--min-gain-db: {e}"))?;
            }
            "--max-gain-db" => {
                a.agc.max_gain_db = v()?.parse().map_err(|e| format!("--max-gain-db: {e}"))?;
            }
            "--speech-floor-dbfs" => {
                a.agc.speech_floor_dbfs = v()?
                    .parse()
                    .map_err(|e| format!("--speech-floor-dbfs: {e}"))?;
            }
            "--isa" => {
                a.isa = match v()?.as_str() {
                    "scalar" => Isa::Scalar,
                    "avx2fma" => Isa::Avx2Fma,
                    "avx512vnni" => Isa::Avx512Vnni,
                    "neon" => Isa::Neon,
                    o => return Err(format!("unknown isa {o}")),
                };
            }
            o => return Err(format!("unknown argument {o}")),
        }
    }
    if a.weights.as_os_str().is_empty() || a.wav.as_os_str().is_empty() {
        return Err(format!("--weights and --wav are required\n{}", usage()));
    }
    if a.packet_ms == 0 {
        return Err("--packet-ms must be > 0".into());
    }
    a.agc.validate().map_err(|e| e.to_string())?;
    Ok(a)
}

fn run(a: &Args) -> Result<serde_json::Value, String> {
    let wav = read_audio_16k_mono(&a.wav).map_err(|e| e.to_string())?;
    let mut model = Model::load(&a.weights, true).map_err(|e| e.to_string())?;
    if a.precision == Precision::Int8 && model.int8_prepared.is_none() {
        model.quantize_int8();
    }
    let mut engine = Engine::new(Arc::new(model), a.isa, a.precision)
        .map_err(|e| e.to_string())?
        .with_beam(a.beam);
    if let Some(c) = a.chunk {
        engine = engine.with_chunk(c).map_err(|e| e.to_string())?;
    }

    let mut stream = AgcStream::from_engine(&engine, a.agc).map_err(|e| e.to_string())?;
    let mut ws = Workspace::default();
    let packet_samples = 16 * a.packet_ms;
    let preroll_samples = (16 * a.preroll_ms).min(wav.len());

    let (c0, w0) = (run_info::thread_cpu_s(), Instant::now());
    if a.agc.warm_agc && preroll_samples > 0 {
        stream
            .agc_mut()
            .seed_from_preroll_i16(&wav[..preroll_samples]);
    }
    for pkt in wav.chunks(packet_samples) {
        stream.accept_i16(pkt).map_err(|e| e.to_string())?;
        stream
            .decode_ready(&engine, &mut ws)
            .map_err(|e| e.to_string())?;
    }
    stream.input_finished();
    stream
        .decode_ready(&engine, &mut ws)
        .map_err(|e| e.to_string())?;

    let cpu_s = run_info::thread_cpu_s() - c0;
    let wall_s = w0.elapsed().as_secs_f64();
    let audio_s = wav.len() as f64 / 16_000.0;
    let text = engine.text(stream.stream());
    Ok(json!({
        "warm_agc": a.agc.warm_agc,
        "tau_s": a.agc.tau_s,
        "preroll_ms": a.preroll_ms,
        "gain_db": stream.agc().gain_db(),
        "level_rms": stream.agc().level_rms(),
        "audio_s": audio_s,
        "cpu_s": cpu_s,
        "wall_s": wall_s,
        "rtfx": if cpu_s > 0.0 { audio_s / cpu_s } else { 0.0 },
        "tokens": stream.tokens().len(),
        "text": text,
    }))
}

fn main() -> ExitCode {
    match parse() {
        Ok(a) if a.help => {
            println!("{}", usage());
            ExitCode::SUCCESS
        }
        Ok(a) => match run(&a) {
            Ok(v) => {
                println!("{}", serde_json::to_string_pretty(&v).unwrap_or_default());
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("ultra-engine-agc: {e}");
                ExitCode::FAILURE
            }
        },
        Err(e) => {
            eprintln!("ultra-engine-agc: {e}");
            ExitCode::from(2)
        }
    }
}
