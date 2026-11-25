//! Hand-written argument parsing for the `isosyll` CLI.

use std::path::PathBuf;

/// Decoding mode for `transcribe`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// Endpoint-committed final text (`ultra-engine-commit` release decoder).
    Final,
    /// Low-latency streaming text (`ultra-cold-wav` `r1-engine-int8` release decoder).
    Live,
}

impl Mode {
    pub fn as_str(self) -> &'static str {
        match self {
            Mode::Final => "final",
            Mode::Live => "live",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TranscribeArgs {
    pub files: Vec<PathBuf>,
    pub mode: Mode,
    pub json: bool,
    pub model: Option<PathBuf>,
    pub threads: usize,
    pub rms_norm: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StreamArgs {
    pub file: PathBuf,
    pub model: Option<PathBuf>,
    pub speed: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HelpTopic {
    Main,
    Transcribe,
    Stream,
    Model,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Command {
    Help(HelpTopic),
    Version,
    Transcribe(TranscribeArgs),
    Stream(StreamArgs),
    ModelVerify { model: Option<PathBuf> },
}

struct Cursor {
    items: Vec<String>,
    pos: usize,
}

impl Cursor {
    fn next(&mut self) -> Option<String> {
        let v = self.items.get(self.pos).cloned();
        self.pos += 1;
        v
    }
    fn value(&mut self, flag: &str) -> Result<String, String> {
        self.next().ok_or_else(|| format!("{flag} needs a value"))
    }
}

/// Split `--flag=value` into `("--flag", Some("value"))`.
fn split_eq(arg: &str) -> (String, Option<String>) {
    if arg.starts_with("--") {
        if let Some((k, v)) = arg.split_once('=') {
            return (k.to_owned(), Some(v.to_owned()));
        }
    }
    (arg.to_owned(), None)
}

fn is_help(a: &str) -> bool {
    a == "--help" || a == "-h"
}

/// Parse `argv[1..]`. Errors are usage errors (exit code 2).
pub fn parse<I: IntoIterator<Item = String>>(argv: I) -> Result<Command, String> {
    let mut c = Cursor {
        items: argv.into_iter().collect(),
        pos: 0,
    };
    let Some(first) = c.next() else {
        return Ok(Command::Help(HelpTopic::Main));
    };
    match first.as_str() {
        "--help" | "-h" | "help" => Ok(Command::Help(HelpTopic::Main)),
        "--version" | "-V" => Ok(Command::Version),
        "transcribe" => parse_transcribe(&mut c),
        "stream" => parse_stream(&mut c),
        "model" => parse_model(&mut c),
        other => Err(format!("unknown command '{other}'")),
    }
}

fn parse_transcribe(c: &mut Cursor) -> Result<Command, String> {
    let mut a = TranscribeArgs {
        files: Vec::new(),
        mode: Mode::Final,
        json: false,
        model: None,
        threads: 1,
        rms_norm: false,
    };
    let mut only_files = false;
    while let Some(raw) = c.next() {
        if only_files {
            a.files.push(PathBuf::from(raw));
            continue;
        }
        if is_help(&raw) {
            return Ok(Command::Help(HelpTopic::Transcribe));
        }
        let (flag, inline) = split_eq(&raw);
        let val = |c: &mut Cursor| match inline.clone() {
            Some(v) => Ok(v),
            None => c.value(&flag),
        };
        match flag.as_str() {
            "--" => only_files = true,
            "--mode" => {
                a.mode = match val(c)?.as_str() {
                    "final" => Mode::Final,
                    "live" => Mode::Live,
                    other => {
                        return Err(format!("--mode must be 'final' or 'live' (got '{other}')"))
                    }
                }
            }
            "--json" => a.json = true,
            "--rms-norm" => a.rms_norm = true,
            "--model" => a.model = Some(PathBuf::from(val(c)?)),
            "--threads" => {
                let v = val(c)?;
                a.threads = v
                    .parse::<usize>()
                    .map_err(|_| format!("--threads: invalid value '{v}'"))?;
                if a.threads == 0 {
                    return Err("--threads must be positive".into());
                }
            }
            f if f.starts_with('-') && f.len() > 1 => return Err(format!("unknown option '{f}'")),
            _ => a.files.push(PathBuf::from(raw)),
        }
    }
    if a.files.is_empty() {
        return Err("transcribe needs at least one audio file".into());
    }
    if a.rms_norm && a.mode == Mode::Live {
        return Err("--rms-norm applies to --mode final only".into());
    }
    Ok(Command::Transcribe(a))
}

fn parse_stream(c: &mut Cursor) -> Result<Command, String> {
    let (mut file, mut model, mut speed) = (None::<PathBuf>, None, 1.0f64);
    while let Some(raw) = c.next() {
        if is_help(&raw) {
            return Ok(Command::Help(HelpTopic::Stream));
        }
        let (flag, inline) = split_eq(&raw);
        let val = |c: &mut Cursor| match inline.clone() {
            Some(v) => Ok(v),
            None => c.value(&flag),
        };
        match flag.as_str() {
            "--model" => model = Some(PathBuf::from(val(c)?)),
            "--speed" => {
                let v = val(c)?;
                speed = v
                    .parse::<f64>()
                    .map_err(|_| format!("--speed: invalid value '{v}'"))?;
                if !speed.is_finite()
                    || speed < 0.0
                    || (speed > 0.0 && speed < 0.01)
                    || speed > 1000.0
                {
                    return Err(
                        "--speed must be 0 (as fast as possible) or between 0.01 and 1000".into(),
                    );
                }
            }
            f if f.starts_with('-') && f.len() > 1 => return Err(format!("unknown option '{f}'")),
            _ => {
                if file.replace(PathBuf::from(raw)).is_some() {
                    return Err("stream takes exactly one audio file".into());
                }
            }
        }
    }
    let file = file.ok_or("stream needs an audio file")?;
    Ok(Command::Stream(StreamArgs { file, model, speed }))
}

fn parse_model(c: &mut Cursor) -> Result<Command, String> {
    let Some(sub) = c.next() else {
        return Ok(Command::Help(HelpTopic::Model));
    };
    if is_help(&sub) {
        return Ok(Command::Help(HelpTopic::Model));
    }
    if sub != "verify" {
        return Err(format!("unknown model subcommand '{sub}'"));
    }
    let mut model = None;
    while let Some(raw) = c.next() {
        if is_help(&raw) {
            return Ok(Command::Help(HelpTopic::Model));
        }
        let (flag, inline) = split_eq(&raw);
        match flag.as_str() {
            "--model" => {
                let v = match inline {
                    Some(v) => v,
                    None => c.value("--model")?,
                };
                model = Some(PathBuf::from(v));
            }
            other => return Err(format!("unknown argument '{other}'")),
        }
    }
    Ok(Command::ModelVerify { model })
}

#[cfg(test)]
#[path = "args_tests.rs"]
mod tests;
