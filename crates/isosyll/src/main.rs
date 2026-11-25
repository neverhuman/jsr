//! `isosyll`: pure-Rust streaming speech recognition with the v4.0.0 release decoders.

mod args;
mod decode;
mod model;
mod segment;
mod stream;

use std::io::{IsTerminal, Write};
use std::path::Path;
use std::process::ExitCode;
use std::time::{Duration, Instant};

use args::{Command, HelpTopic, Mode, StreamArgs, TranscribeArgs};
use decode::{FinalDecoder, LiveDecoder, PACKET_SAMPLES};
use model::{ModelDir, ResolveEnv};
use serde_json::json;
use stream::{SegEvent, Segmenter};
use ultra_frontend::read_audio_16k_mono;

const MAIN_HELP: &str = "\
isosyll - streaming speech recognition on one CPU core (pure Rust, INT8 safetensors model)

USAGE:
    isosyll transcribe <AUDIO>... [--mode final|live] [--json] [--model DIR] [--threads N] [--rms-norm]
    isosyll stream <AUDIO> [--model DIR] [--speed X]
    isosyll model verify [--model DIR]
    isosyll --version | -V
    isosyll --help | -h

Run `isosyll <command> --help` for details.

The model directory is taken from --model, then $ISOSYLL_MODEL, then
<exe dir>/../share/isosyll/model, then $HOME/.local/share/isosyll/model.
";

const TRANSCRIBE_HELP: &str = "\
isosyll transcribe - transcribe audio files

USAGE:
    isosyll transcribe <AUDIO>... [OPTIONS]

OPTIONS:
    --mode final|live   final (default): endpoint-committed text, highest accuracy.
                        live: low-latency streaming text.
    --json              one JSON object per file: file, text, mode, audio_s, wall_s
    --model DIR         model directory (see `isosyll --help` for the search order)
    --threads N         matrix-multiply threads (default 1, the measured setting)
    --rms-norm          final mode only: whole-segment RMS gain before the commit pass
    -h, --help          show this help

Output is one line of text per file, prefixed with `<path>TAB` when several files are given.
Input: WAV (16-bit PCM, any sample rate, mono or stereo; resampled to 16 kHz mono)
or raw 16 kHz 16-bit little-endian PCM (.pcm, .s16le).
Final mode on audio up to 40 s uses a single commit pass. Longer audio is cut at the quietest
point 20-30 s into each piece, each piece is committed separately, and the texts are joined with
a space. Live mode has no length limit.
";

const STREAM_HELP: &str = "\
isosyll stream - replay an audio file as a live stream

USAGE:
    isosyll stream <AUDIO> [OPTIONS]

OPTIONS:
    --model DIR   model directory
    --speed X     replay speed (default 1.0 = real time; 0 = as fast as possible)
    -h, --help    show this help

Live partial text updates on one line of stderr (only when stderr is a terminal).
Each final committed segment is printed as a line on stdout.
";

const MODEL_HELP: &str = "\
isosyll model - model bundle tools

USAGE:
    isosyll model verify [--model DIR]

`verify` checks the sha256 of every file listed in <model>/MANIFEST
(lines: \"<sha256>  <relpath>\"). Exit code 1 on any mismatch.
";

enum Failure {
    Runtime(String),
}

impl From<String> for Failure {
    fn from(s: String) -> Self {
        Failure::Runtime(s)
    }
}

fn resolve_model(flag: Option<&Path>) -> Result<ModelDir, String> {
    let dir = model::resolve(flag, &ResolveEnv::from_process())?;
    Ok(dir)
}

fn read_pcm(path: &Path) -> Result<Vec<i16>, String> {
    read_audio_16k_mono(path).map_err(|e| format!("{}: {e}", path.display()))
}

fn transcribe(a: &TranscribeArgs) -> Result<(), Failure> {
    let dir = resolve_model(a.model.as_deref())?;
    let lm = match a.mode {
        Mode::Final => dir.final_lm(),
        Mode::Live => dir.live_lm(),
    };
    model::require_file(&lm)?;
    enum Dec {
        Final(Box<FinalDecoder>),
        Live(Box<LiveDecoder>),
    }
    let mut dec = match a.mode {
        Mode::Final => Dec::Final(Box::new(FinalDecoder::new(&dir, a.threads, a.rms_norm)?)),
        Mode::Live => Dec::Live(Box::new(LiveDecoder::new(&dir, a.threads)?)),
    };
    let stdout = std::io::stdout();
    for f in &a.files {
        let pcm = read_pcm(f)?;
        let t0 = Instant::now();
        let text = match &mut dec {
            Dec::Final(d) => d.decode(&pcm)?,
            Dec::Live(d) => d.decode(&pcm)?,
        };
        let wall_s = t0.elapsed().as_secs_f64();
        let audio_s = pcm.len() as f64 / 16_000.0;
        let line = if a.json {
            json!({"file": f.display().to_string(), "text": text, "mode": a.mode.as_str(),
                   "audio_s": audio_s, "wall_s": wall_s})
            .to_string()
        } else if a.files.len() > 1 {
            format!("{}\t{text}", f.display())
        } else {
            text
        };
        let mut out = stdout.lock();
        writeln!(out, "{line}")
            .and_then(|()| out.flush())
            .map_err(|e| e.to_string())?;
    }
    Ok(())
}

fn stream(a: &StreamArgs) -> Result<(), Failure> {
    let dir = resolve_model(a.model.as_deref())?;
    model::require_file(&dir.live_lm())?;
    model::require_file(&dir.final_lm())?;
    let pcm = read_pcm(&a.file)?;
    let mut seg = Segmenter::new(&dir, 1, false)?;
    let tty = std::io::stderr().is_terminal();
    let start = Instant::now();
    let show = |evs: Vec<SegEvent>| -> Result<(), String> {
        for ev in evs {
            match ev {
                SegEvent::Partial(t) => {
                    if tty {
                        eprint!("\r\x1b[K{t}");
                        let _ = std::io::stderr().flush();
                    }
                }
                SegEvent::Segment(t) => {
                    if tty {
                        eprint!("\r\x1b[K");
                    }
                    let mut out = std::io::stdout().lock();
                    writeln!(out, "{t}")
                        .and_then(|()| out.flush())
                        .map_err(|e| e.to_string())?;
                }
            }
        }
        Ok(())
    };
    for (i, pkt) in pcm.chunks(PACKET_SAMPLES).enumerate() {
        if a.speed > 0.0 {
            // A packet is available once its last sample has been "spoken".
            let due = ((i * PACKET_SAMPLES + pkt.len()) as f64 / 16_000.0) / a.speed;
            let wait = due - start.elapsed().as_secs_f64();
            if wait > 0.0 {
                std::thread::sleep(Duration::try_from_secs_f64(wait).unwrap_or(Duration::ZERO));
            }
        }
        show(seg.push(pkt)?)?;
    }
    show(seg.finish()?)?;
    if tty {
        eprint!("\r\x1b[K");
    }
    Ok(())
}

fn verify(model_flag: Option<&Path>) -> Result<bool, Failure> {
    let dir = resolve_model(model_flag)?;
    let (ok, bad) = model::verify(&dir, |status, path| println!("{status}  {path}"))?;
    println!("{ok} ok, {bad} failed");
    Ok(bad == 0)
}

fn run(cmd: Command) -> Result<bool, Failure> {
    match cmd {
        Command::Help(t) => {
            print!(
                "{}",
                match t {
                    HelpTopic::Main => MAIN_HELP,
                    HelpTopic::Transcribe => TRANSCRIBE_HELP,
                    HelpTopic::Stream => STREAM_HELP,
                    HelpTopic::Model => MODEL_HELP,
                }
            );
            Ok(true)
        }
        Command::Version => {
            let dir = model::resolve(None, &ResolveEnv::from_process()).ok();
            println!("{}", model::version_line(dir.as_ref()));
            Ok(true)
        }
        Command::Transcribe(a) => transcribe(&a).map(|()| true),
        Command::Stream(a) => stream(&a).map(|()| true),
        Command::ModelVerify { model } => verify(model.as_deref()),
    }
}

fn main() -> ExitCode {
    let cmd = match args::parse(std::env::args().skip(1)) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error: {e}\nrun `isosyll --help` for usage");
            return ExitCode::from(2);
        }
    };
    match run(cmd) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::from(1),
        Err(Failure::Runtime(e)) => {
            eprintln!("error: {e}");
            ExitCode::from(1)
        }
    }
}
