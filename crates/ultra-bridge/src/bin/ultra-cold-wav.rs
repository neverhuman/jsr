//! Reference-blind single-stream service for cold-cost measurement.

use std::fs::{self, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use serde_json::json;
use ultra_bridge::{records, require, Result};
use ultra_engine::Precision;
use ultra_frontend::read_wav_pcm;
use ultra_gp::R1Precision;
use ultra_stream::{
    Backend, NativeEngineBackend, OrtTransducerBackend, PollResult, PushStatus, SherpaLib,
    SherpaZipformerBackend, SherpaZipformerConfig, StreamConfig, TerminalOutcome,
};

const PACKET_SAMPLES: usize = 320;

fn load(kind: &str, bundle: &Path) -> Result<Box<dyn Backend>> {
    Ok(match kind {
        "zipformer-int8" => {
            let verified = ultra_model::bundle::verify_bundle_dir(bundle, None)?;
            require(
                verified.manifest.precision == "int8",
                "expected INT8 Zipformer",
            )?;
            for name in ["encoder", "decoder", "joiner"] {
                let path = format!("model/{name}-epoch-99-avg-1-chunk-16-left-128.int8.onnx");
                require(
                    verified
                        .manifest
                        .entries
                        .iter()
                        .any(|entry| entry.path == path),
                    format!("Zipformer bundle does not bind {path}"),
                )?;
            }
            require(
                verified
                    .manifest
                    .entries
                    .iter()
                    .any(|entry| entry.path == "model/tokens.txt"),
                "Zipformer bundle does not bind tokens.txt",
            )?;
            Box::new(SherpaZipformerBackend::new(
                SherpaLib::load_default()?,
                SherpaZipformerConfig {
                    model_dir: bundle.join("model"),
                    ..SherpaZipformerConfig::default()
                },
            )?)
        }
        "r1-engine-int8" => Box::new(NativeEngineBackend::load(bundle, Precision::Int8, 1, true)?),
        "r1-engine-fp32" => Box::new(NativeEngineBackend::load(bundle, Precision::F32, 1, true)?),
        "r1-ort-fp32" => Box::new(OrtTransducerBackend::load(
            bundle,
            R1Precision::Fp32,
            1,
            true,
        )?),
        _ => return Err(format!("unsupported cold service backend: {kind}").into()),
    })
}

fn inputs(dir: &Path) -> Result<Vec<(String, PathBuf)>> {
    let mut entries = Vec::new();
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let file_name = entry.file_name();
        let name = file_name.to_str().ok_or("WAV filename is not UTF-8")?;
        let id = name.strip_suffix(".wav").ok_or("expected .wav input")?;
        require(
            entry.file_type()?.is_file()
                && id.len() == 6
                && id.starts_with('u')
                && id.as_bytes()[1..].iter().all(u8::is_ascii_digit),
            format!("invalid staged input {name}"),
        )?;
        entries.push((id.to_owned(), entry.path()));
    }
    require(!entries.is_empty(), "staged WAV directory is empty")?;
    entries.sort_unstable_by(|a, b| a.0.cmp(&b.0));
    Ok(entries)
}

fn decode(
    backend: &dyn Backend,
    id: &str,
    input: &Path,
    mut trace: Option<&mut BufWriter<fs::File>>,
) -> Result<String> {
    use std::time::Instant;
    use ultra_core::events::Body;

    let wav = read_wav_pcm(&fs::read(input)?)?;
    require(
        wav.sample_rate_hz == 16_000 && wav.channels == 1 && !wav.samples.is_empty(),
        "expected non-empty mono 16 kHz PCM16",
    )?;
    let mut stream = backend.create_stream(&StreamConfig {
        stream_id: id.to_owned(),
        ..StreamConfig::default()
    })?;
    let mut offset = 0;
    let mut cum_wall_ms = 0.0f64;
    for packet in wav.samples.chunks(PACKET_SAMPLES) {
        let t0 = Instant::now();
        let accepted = stream.push_pcm(offset, packet)?;
        require(
            accepted.status == PushStatus::AcceptedAll,
            "service rejected staged audio",
        )?;
        offset += packet.len() as u64;
        let mut latest_partial: Option<String> = None;
        while let PollResult::Event(event) = stream.poll_event()? {
            match event.body {
                Body::Preview(update) | Body::Commit(update) => {
                    latest_partial = Some(update.snapshot_utf8);
                }
                _ => {}
            }
        }
        let dt_ms = t0.elapsed().as_secs_f64() * 1000.0;
        cum_wall_ms += dt_ms;
        if let Some(ref mut tw) = trace {
            if let Some(text) = latest_partial {
                records::json_line(
                    tw,
                    &json!({
                        "id": id,
                        "audio_s": offset as f64 / 16_000.0,
                        "chunk_wall_ms": dt_ms,
                        "cum_wall_ms": cum_wall_ms,
                        "text": text,
                        "final": false,
                    }),
                )?;
            }
        }
    }
    let t_fin = Instant::now();
    stream.finish_input()?;
    let mut final_text = None;
    loop {
        match stream.poll_event()? {
            PollResult::Event(event) => {
                if let Body::Final { update, processing } = event.body {
                    require(
                        processing.flushed
                            && processing.required_processed_through_sample == offset,
                        "missing EOF processing evidence",
                    )?;
                    require(
                        final_text.replace(update.snapshot_utf8).is_none(),
                        "multiple Final events",
                    )?;
                }
            }
            PollResult::Ended(TerminalOutcome::Finished) => break,
            status => return Err(format!("invalid stream drain: {status:?}").into()),
        }
    }
    let fin_ms = t_fin.elapsed().as_secs_f64() * 1000.0;
    cum_wall_ms += fin_ms;
    let metrics = stream.metrics();
    require(
        metrics.accepted_samples == offset && metrics.processed_samples == offset,
        "stream did not process all staged audio",
    )?;
    let hyp = final_text.ok_or_else(|| "missing Final event".to_string())?;
    if let Some(ref mut tw) = trace {
        records::json_line(
            tw,
            &json!({
                "id": id,
                "audio_s": offset as f64 / 16_000.0,
                "chunk_wall_ms": fin_ms,
                "cum_wall_ms": cum_wall_ms,
                "text": hyp,
                "final": true,
            }),
        )?;
    }
    Ok(hyp)
}

fn run() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let (kind, bundle, wav_dir, output, trace_path) =
        match args.as_slice() {
            [kind, bundle, wav_dir, output] => (kind, bundle, wav_dir, output, None),
            [kind, bundle, wav_dir, output, trace] => (kind, bundle, wav_dir, output, Some(trace)),
            _ => return Err(
                "usage: ultra-cold-wav BACKEND MODEL_DIR WAV_DIR PREDICTIONS.jsonl [TRACE.jsonl]"
                    .into(),
            ),
        };
    let inputs = inputs(Path::new(wav_dir))?;
    let backend = load(kind, Path::new(bundle))?;
    let caps = backend.capabilities();
    require(
        caps.cpu_only_verified,
        "recognizer failed CPU-only verification",
    )?;
    eprintln!(
        "{}",
        json!({"capabilities": caps, "runtime_libraries": records::mapped_runtime_libraries()?})
    );
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)?;
    let mut out = BufWriter::new(file);
    let mut trace = match trace_path {
        Some(p) => Some(BufWriter::new(
            OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open(p)?,
        )),
        None => None,
    };
    for (id, path) in inputs {
        let hyp = decode(&*backend, &id, &path, trace.as_mut())?;
        records::json_line(&mut out, &json!({"id": id, "hyp": hyp}))?;
    }
    out.flush()?;
    if let Some(ref mut tw) = trace {
        tw.flush()?;
    }
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("ultra-cold-wav: {error}");
        std::process::exit(1);
    }
}
