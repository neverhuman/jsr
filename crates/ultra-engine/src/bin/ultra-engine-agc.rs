//! `ultra-engine-agc`: streaming recogniser CLI exposing default-off `--warm-agc`.
//!
//! Supports both single-file `--wav` and multi-utterance `--manifest <jsonl>` (carrying
//! warm AGC state across utterances in one process while resetting decoder state per clip).

use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

use serde_json::{json, Value};
use ultra_engine::{run_info, AgcStream, Engine, Isa, Model, Precision, WarmAgcConfig, Workspace};
use ultra_frontend::read_audio_16k_mono;

struct Args {
    weights: PathBuf,
    wav: Option<PathBuf>,
    manifest: Option<PathBuf>,
    wav_root: Option<PathBuf>,
    out: Option<PathBuf>,
    precision: Precision,
    chunk: Option<usize>,
    beam: usize,
    packet_ms: usize,
    preroll_ms: usize,
    cold_per_clip: bool,
    agc: WarmAgcConfig,
    isa: Isa,
    help: bool,
}

fn usage() -> &'static str {
    "usage: ultra-engine-agc --weights <dir> (--wav <16k.wav> | --manifest <jsonl>) \
     [--wav-root <dir>] [--out <jsonl>] [--precision int8|fp32] [--chunk N] [--beam K] \
     [--packet-ms 20] [--warm-agc | --no-warm-agc] [--preroll-ms 0] [--cold-per-clip] \
     [--tau-s 5.0] [--target-rms 0.058884] [--min-gain-db -12.0] [--max-gain-db 40.0] \
     [--speech-floor-dbfs -65.0]"
}

fn parse() -> Result<Args, String> {
    let mut it = std::env::args().skip(1);
    let mut a = Args {
        weights: PathBuf::new(),
        wav: None,
        manifest: None,
        wav_root: None,
        out: None,
        precision: Precision::Int8,
        chunk: None,
        beam: 1,
        packet_ms: 20,
        preroll_ms: 0,
        cold_per_clip: false,
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
            "--wav" => a.wav = Some(v()?.into()),
            "--manifest" => a.manifest = Some(v()?.into()),
            "--wav-root" => a.wav_root = Some(v()?.into()),
            "--out" => a.out = Some(v()?.into()),
            "--precision" => a.precision = Precision::parse(&v()?).map_err(|e| e.to_string())?,
            "--chunk" => a.chunk = Some(v()?.parse().map_err(|e| format!("--chunk: {e}"))?),
            "--beam" => a.beam = v()?.parse().map_err(|e| format!("--beam: {e}"))?,
            "--packet-ms" => a.packet_ms = v()?.parse().map_err(|e| format!("--packet-ms: {e}"))?,
            "--preroll-ms" => {
                a.preroll_ms = v()?.parse().map_err(|e| format!("--preroll-ms: {e}"))?;
            }
            "--cold-per-clip" => a.cold_per_clip = true,
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
    if a.weights.as_os_str().is_empty() || a.wav.is_some() == a.manifest.is_some() {
        return Err(format!(
            "--weights and exactly one of --wav or --manifest are required\n{}",
            usage()
        ));
    }
    if a.packet_ms == 0 {
        return Err("--packet-ms must be > 0".into());
    }
    a.agc.validate().map_err(|e| e.to_string())?;
    Ok(a)
}

fn decode_clip(
    engine: &Engine,
    stream: &mut AgcStream,
    ws: &mut Workspace,
    wav_path: &Path,
    id: Option<&str>,
    a: &Args,
) -> Result<Value, String> {
    let wav = read_audio_16k_mono(wav_path).map_err(|e| format!("{}: {e}", wav_path.display()))?;
    let packet_samples = 16 * a.packet_ms;
    let preroll_samples = (16 * a.preroll_ms).min(wav.len());
    let (c0, w0) = (run_info::thread_cpu_s(), Instant::now());
    if a.agc.warm_agc && preroll_samples > 0 && !stream.agc().is_warm() {
        stream
            .agc_mut()
            .seed_from_preroll_i16(&wav[..preroll_samples]);
    }
    let gain_db_start = stream.agc().gain_db();
    for pkt in wav.chunks(packet_samples) {
        stream.accept_i16(pkt).map_err(|e| e.to_string())?;
        stream.decode_ready(engine, ws).map_err(|e| e.to_string())?;
    }
    stream.input_finished();
    stream.decode_ready(engine, ws).map_err(|e| e.to_string())?;
    let (cpu_s, wall_s) = (run_info::thread_cpu_s() - c0, w0.elapsed().as_secs_f64());
    let audio_s = wav.len() as f64 / 16_000.0;
    let mut row = json!({
        "warm_agc": a.agc.warm_agc,
        "tau_s": a.agc.tau_s,
        "preroll_ms": a.preroll_ms,
        "gain_db_start": gain_db_start,
        "gain_db": stream.agc().gain_db(),
        "level_rms": stream.agc().level_rms(),
        "audio_s": audio_s,
        "cpu_s": cpu_s,
        "wall_s": wall_s,
        "rtfx": if cpu_s > 0.0 { audio_s / cpu_s } else { 0.0 },
        "tokens": stream.tokens().len(),
        "text": engine.text(stream.stream()),
    });
    if let Some(clip_id) = id {
        row["id"] = json!(clip_id);
        row["wav"] = json!(wav_path);
    }
    Ok(row)
}

fn resolve_wav(raw: &str, root: Option<&Path>, manifest: &Path) -> PathBuf {
    let p = PathBuf::from(raw);
    if p.is_absolute() {
        return p;
    }
    if let Some(r) = root {
        return r.join(&p);
    }
    if p.exists() {
        return p;
    }
    manifest.parent().map_or(p.clone(), |d| d.join(&p))
}

fn run(a: &Args) -> Result<(), String> {
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

    if let Some(wav_path) = &a.wav {
        let v = decode_clip(&engine, &mut stream, &mut ws, wav_path, None, a)?;
        let s = serde_json::to_string_pretty(&v).map_err(|e| e.to_string())?;
        if let Some(out) = &a.out {
            std::fs::write(out, format!("{s}\n")).map_err(|e| e.to_string())?;
        } else {
            println!("{s}");
        }
        return Ok(());
    }

    let manifest = a.manifest.as_ref().expect("checked");
    let reader = BufReader::new(File::open(manifest).map_err(|e| e.to_string())?);
    let mut out_w: Box<dyn Write> = match &a.out {
        Some(p) => {
            if let Some(parent) = p.parent().filter(|d| !d.as_os_str().is_empty()) {
                std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
            }
            Box::new(BufWriter::new(File::create(p).map_err(|e| e.to_string())?))
        }
        None => Box::new(BufWriter::new(std::io::stdout().lock())),
    };
    for (idx, line) in reader.lines().enumerate() {
        let line = line.map_err(|e| e.to_string())?;
        if line.trim().is_empty() {
            continue;
        }
        let rec: Value =
            serde_json::from_str(&line).map_err(|e| format!("manifest line {}: {e}", idx + 1))?;
        let wav_s = rec["wav"]
            .as_str()
            .ok_or_else(|| format!("manifest line {} missing `wav`", idx + 1))?;
        let wav_path = resolve_wav(wav_s, a.wav_root.as_deref(), manifest);
        let id = rec["staged_id"]
            .as_str()
            .or_else(|| rec["id"].as_str())
            .or_else(|| rec["uid"].as_str())
            .map(str::to_owned)
            .unwrap_or_else(|| {
                wav_path
                    .file_stem()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_else(|| format!("u{idx:05}"))
            });
        if idx > 0 {
            if a.cold_per_clip {
                stream.hard_reset(&engine);
            } else {
                stream.reset_utterance(&engine);
            }
        }
        let row = decode_clip(&engine, &mut stream, &mut ws, &wav_path, Some(&id), a)?;
        writeln!(out_w, "{row}").map_err(|e| e.to_string())?;
    }
    out_w.flush().map_err(|e| e.to_string())
}

fn main() -> ExitCode {
    match parse() {
        Ok(a) if a.help => {
            println!("{}", usage());
            ExitCode::SUCCESS
        }
        Ok(a) => match run(&a) {
            Ok(()) => ExitCode::SUCCESS,
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
