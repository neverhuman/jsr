//! Reference-blind streaming service for cold-cost measurement (1 or N shared-weights workers).

use std::fs::{self, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde_json::json;
use ultra_bridge::{records, require, Result};
use ultra_engine::{LmConfig, NgramLm, Precision};
use ultra_frontend::read_wav_pcm;
use ultra_gp::R1Precision;
use ultra_stream::{
    Backend, NativeEngineBackend, NativeEngineLoadConfig, OrtTransducerBackend, PollResult,
    PushStatus, SherpaLib, SherpaZipformerBackend, SherpaZipformerConfig, StreamConfig,
    TerminalOutcome,
};

const PACKET_SAMPLES: usize = 320;

#[derive(Debug, PartialEq)]
struct CliArgs {
    kind: String,
    bundle: PathBuf,
    wav_dir: PathBuf,
    output: PathBuf,
    trace_path: Option<PathBuf>,
    workers: usize,
    chunk: Option<usize>,
    beam: usize,
    beam_max_symbols: usize,
    beam_margin: f32,
    lm_path: Option<PathBuf>,
    lm_weight: f32,
    lodr_weight: f32,
    blank_penalty: f32,
}

fn usage() -> &'static str {
    "usage: ultra-cold-wav [--workers N] [--chunk C] [--beam B] [--beam-max-symbols S] [--beam-margin M] [--lm PATH --lm-weight W --lodr-weight R --blank-penalty P] BACKEND MODEL_DIR WAV_DIR PREDICTIONS.jsonl [TRACE.jsonl]"
}

fn parse_args(args: &[String]) -> Result<CliArgs> {
    let mut workers: usize = 1;
    let mut chunk: Option<usize> = None;
    let mut beam: usize = 1;
    let mut beam_max_symbols: usize = 1;
    let mut beam_margin: f32 = 0.0;
    let mut lm_path: Option<PathBuf> = None;
    let mut lm_weight: f32 = 0.0;
    let mut lodr_weight: f32 = 0.0;
    let mut blank_penalty: f32 = 0.0;
    let mut positional: Vec<String> = Vec::new();
    let mut i = 0;
    while i < args.len() {
        let arg = &args[i];
        let next_val = |idx: usize, name: &str| -> Result<&str> {
            args.get(idx + 1)
                .map(String::as_str)
                .ok_or_else(|| format!("missing value for {name}; {}", usage()).into())
        };
        if arg == "--workers" {
            let val = next_val(i, "--workers")?;
            workers = val
                .parse::<usize>()
                .map_err(|_| format!("invalid --workers value: {val}"))?;
            i += 2;
        } else if let Some(val) = arg.strip_prefix("--workers=") {
            workers = val
                .parse::<usize>()
                .map_err(|_| format!("invalid --workers value: {val}"))?;
            i += 1;
        } else if arg == "--chunk" {
            let val = next_val(i, "--chunk")?;
            chunk = Some(
                val.parse::<usize>()
                    .map_err(|_| format!("invalid --chunk value: {val}"))?,
            );
            i += 2;
        } else if arg == "--beam" {
            let val = next_val(i, "--beam")?;
            beam = val
                .parse::<usize>()
                .map_err(|_| format!("invalid --beam value: {val}"))?;
            i += 2;
        } else if arg == "--beam-max-symbols" {
            let val = next_val(i, "--beam-max-symbols")?;
            beam_max_symbols = val
                .parse::<usize>()
                .map_err(|_| format!("invalid --beam-max-symbols value: {val}"))?;
            i += 2;
        } else if arg == "--beam-margin" {
            let val = next_val(i, "--beam-margin")?;
            beam_margin = val
                .parse::<f32>()
                .map_err(|_| format!("invalid --beam-margin value: {val}"))?;
            i += 2;
        } else if arg == "--lm" {
            lm_path = Some(PathBuf::from(next_val(i, "--lm")?));
            i += 2;
        } else if arg == "--lm-weight" {
            let val = next_val(i, "--lm-weight")?;
            lm_weight = val
                .parse::<f32>()
                .map_err(|_| format!("invalid --lm-weight value: {val}"))?;
            i += 2;
        } else if arg == "--lodr-weight" {
            let val = next_val(i, "--lodr-weight")?;
            lodr_weight = val
                .parse::<f32>()
                .map_err(|_| format!("invalid --lodr-weight value: {val}"))?;
            i += 2;
        } else if arg == "--blank-penalty" {
            let val = next_val(i, "--blank-penalty")?;
            blank_penalty = val
                .parse::<f32>()
                .map_err(|_| format!("invalid --blank-penalty value: {val}"))?;
            i += 2;
        } else if arg.starts_with("--") {
            return Err(format!("unknown option: {arg}; {}", usage()).into());
        } else {
            positional.push(arg.clone());
            i += 1;
        }
    }
    require(workers >= 1, "--workers must be >= 1")?;
    require(beam >= 1, "--beam must be >= 1")?;
    require(beam_max_symbols >= 1, "--beam-max-symbols must be >= 1")?;
    require(
        beam_margin.is_finite() && beam_margin >= 0.0,
        "--beam-margin must be finite and >= 0",
    )?;
    require(
        lm_weight.is_finite() && lm_weight >= 0.0,
        "--lm-weight must be finite and >= 0",
    )?;
    require(
        lodr_weight.is_finite() && lodr_weight >= 0.0,
        "--lodr-weight must be finite and >= 0",
    )?;
    require(blank_penalty.is_finite(), "--blank-penalty must be finite")?;
    let (kind, bundle, wav_dir, output, trace_path) = match positional.as_slice() {
        [kind, bundle, wav_dir, output] => (kind, bundle, wav_dir, output, None),
        [kind, bundle, wav_dir, output, trace] => (kind, bundle, wav_dir, output, Some(trace)),
        _ => return Err(usage().into()),
    };
    require(
        !(workers > 1 && trace_path.is_some()),
        "TRACE.jsonl is only supported with --workers 1",
    )?;
    let has_native_decode_opts = chunk.is_some()
        || beam > 1
        || beam_max_symbols > 1
        || beam_margin != 0.0
        || lm_path.is_some()
        || lm_weight != 0.0
        || lodr_weight != 0.0
        || blank_penalty != 0.0;
    let is_native = matches!(kind.as_str(), "r1-engine-int8" | "r1-engine-fp32");
    require(
        is_native || !has_native_decode_opts,
        "--chunk/--beam/--lm options are only supported with r1-engine-int8 and r1-engine-fp32",
    )?;
    if lm_path.is_some() {
        require(lm_weight > 0.0, "--lm requires --lm-weight > 0")?;
    } else {
        require(
            lm_weight == 0.0 && lodr_weight == 0.0 && blank_penalty == 0.0,
            "--lm-weight/--lodr-weight/--blank-penalty require --lm",
        )?;
    }
    Ok(CliArgs {
        kind: kind.clone(),
        bundle: PathBuf::from(bundle),
        wav_dir: PathBuf::from(wav_dir),
        output: PathBuf::from(output),
        trace_path: trace_path.map(PathBuf::from),
        workers,
        chunk,
        beam,
        beam_max_symbols,
        beam_margin,
        lm_path,
        lm_weight,
        lodr_weight,
        blank_penalty,
    })
}

fn load_native(cli: &CliArgs, precision: Precision) -> Result<Arc<dyn Backend>> {
    let lm = match &cli.lm_path {
        Some(path) => {
            let ngram = Arc::new(NgramLm::load(path)?);
            let cfg = LmConfig {
                lm_weight: cli.lm_weight,
                lodr_weight: cli.lodr_weight,
                blank_penalty: cli.blank_penalty,
            };
            Some((ngram, cfg))
        }
        None => None,
    };
    let cfg = NativeEngineLoadConfig {
        max_chunks: 1,
        chunk: cli.chunk,
        beam: cli.beam,
        beam_max_symbols: cli.beam_max_symbols,
        beam_margin: cli.beam_margin,
        lm,
        live_only: true,
    };
    Ok(Arc::new(NativeEngineBackend::load_with_config(
        &cli.bundle,
        precision,
        1,
        true,
        &cfg,
    )?))
}

fn load(cli: &CliArgs) -> Result<Arc<dyn Backend>> {
    let bundle = &cli.bundle;
    Ok(match cli.kind.as_str() {
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
            Arc::new(SherpaZipformerBackend::new(
                SherpaLib::load_default()?,
                SherpaZipformerConfig {
                    model_dir: bundle.join("model"),
                    ..SherpaZipformerConfig::default()
                },
            )?)
        }
        "r1-engine-int8" => load_native(cli, Precision::Int8)?,
        "r1-engine-fp32" => load_native(cli, Precision::F32)?,
        "r1-ort-fp32" => Arc::new(OrtTransducerBackend::load(
            bundle,
            R1Precision::Fp32,
            1,
            true,
        )?),
        other => return Err(format!("unsupported cold service backend: {other}").into()),
    })
}

fn inputs(dir: &Path) -> Result<Vec<(String, PathBuf, u64)>> {
    let mut entries = Vec::new();
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let file_name = entry.file_name();
        let name = file_name.to_str().ok_or("WAV filename is not UTF-8")?;
        let id = name.strip_suffix(".wav").ok_or("expected .wav input")?;
        let meta = entry.metadata()?;
        require(
            meta.is_file()
                && id.len() == 6
                && id.starts_with('u')
                && id.as_bytes()[1..].iter().all(u8::is_ascii_digit),
            format!("invalid staged input {name}"),
        )?;
        entries.push((id.to_owned(), entry.path(), meta.len()));
    }
    require(!entries.is_empty(), "staged WAV directory is empty")?;
    entries.sort_unstable_by(|a, b| a.0.cmp(&b.0));
    Ok(entries)
}

/// Partition WAV inputs into duration-balanced shards (LPT bin packing by PCM16 WAV file size).
fn shard_inputs(
    mut entries: Vec<(String, PathBuf, u64)>,
    workers: usize,
) -> Vec<Vec<(String, PathBuf)>> {
    let n = workers.min(entries.len()).max(1);
    entries.sort_unstable_by(|a, b| b.2.cmp(&a.2).then_with(|| a.0.cmp(&b.0)));
    let mut shards: Vec<Vec<(String, PathBuf)>> = vec![Vec::new(); n];
    let mut totals = vec![0u64; n];
    for (id, path, bytes) in entries {
        let best = totals
            .iter()
            .enumerate()
            .min_by_key(|&(_, &sum)| sum)
            .map(|(idx, _)| idx)
            .unwrap_or(0);
        totals[best] = totals[best].saturating_add(bytes);
        shards[best].push((id, path));
    }
    shards
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

fn decode_parallel(
    backend: &Arc<dyn Backend>,
    entries: Vec<(String, PathBuf, u64)>,
    workers: usize,
) -> Result<Vec<(String, String)>> {
    let shards = shard_inputs(entries, workers);
    let mut results: Vec<(String, String)> = Vec::new();
    std::thread::scope(|scope| -> Result<()> {
        let mut handles = Vec::with_capacity(shards.len());
        for shard in shards {
            let b = Arc::clone(backend);
            handles.push(scope.spawn(move || -> Result<Vec<(String, String)>> {
                let mut local = Vec::with_capacity(shard.len());
                for (id, path) in shard {
                    let hyp = decode(&*b, &id, &path, None)?;
                    local.push((id, hyp));
                }
                Ok(local)
            }));
        }
        for handle in handles {
            let chunk = handle
                .join()
                .map_err(|_| "ultra-cold-wav worker thread panicked")??;
            results.extend(chunk);
        }
        Ok(())
    })?;
    results.sort_unstable_by(|a, b| a.0.cmp(&b.0));
    Ok(results)
}

fn run() -> Result<()> {
    let raw_args: Vec<_> = std::env::args().skip(1).collect();
    let cli = parse_args(&raw_args)?;
    let entries = inputs(&cli.wav_dir)?;
    let backend = load(&cli)?;
    let caps = backend.capabilities();
    require(
        caps.cpu_only_verified,
        "recognizer failed CPU-only verification",
    )?;
    eprintln!(
        "{}",
        json!({
            "capabilities": caps,
            "workers": cli.workers,
            "runtime_libraries": records::mapped_runtime_libraries()?
        })
    );
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&cli.output)?;
    let mut out = BufWriter::new(file);
    if cli.workers == 1 {
        let mut trace = match cli.trace_path {
            Some(p) => Some(BufWriter::new(
                OpenOptions::new()
                    .write(true)
                    .create(true)
                    .truncate(true)
                    .open(p)?,
            )),
            None => None,
        };
        for (id, path, _) in entries {
            let hyp = decode(&*backend, &id, &path, trace.as_mut())?;
            records::json_line(&mut out, &json!({"id": id, "hyp": hyp}))?;
        }
        out.flush()?;
        if let Some(ref mut tw) = trace {
            tw.flush()?;
        }
    } else {
        let rows = decode_parallel(&backend, entries, cli.workers)?;
        for (id, hyp) in rows {
            records::json_line(&mut out, &json!({"id": id, "hyp": hyp}))?;
        }
        out.flush()?;
    }
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("ultra-cold-wav: {error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ultra_stream::{FakeBackend, FakeBackendConfig};

    fn write_wav_16k(path: &Path, samples: usize) {
        let data_bytes = (samples * 2) as u32;
        let mut buf = Vec::with_capacity(44 + data_bytes as usize);
        buf.extend_from_slice(b"RIFF");
        buf.extend_from_slice(&(36 + data_bytes).to_le_bytes());
        buf.extend_from_slice(b"WAVEfmt ");
        buf.extend_from_slice(&16u32.to_le_bytes());
        buf.extend_from_slice(&1u16.to_le_bytes()); // PCM
        buf.extend_from_slice(&1u16.to_le_bytes()); // mono
        buf.extend_from_slice(&16_000u32.to_le_bytes());
        buf.extend_from_slice(&32_000u32.to_le_bytes());
        buf.extend_from_slice(&2u16.to_le_bytes());
        buf.extend_from_slice(&16u16.to_le_bytes());
        buf.extend_from_slice(b"data");
        buf.extend_from_slice(&data_bytes.to_le_bytes());
        buf.resize(44 + data_bytes as usize, 0);
        fs::write(path, buf).unwrap();
    }

    #[test]
    fn parse_args_handles_workers_flag_and_defaults() {
        let s = |v: &[&str]| v.iter().map(|x| (*x).to_string()).collect::<Vec<_>>();
        let c4 = parse_args(&s(&["r1-engine-int8", "m", "w", "out.jsonl"])).unwrap();
        assert_eq!(c4.workers, 1);
        assert!(c4.trace_path.is_none());

        let c5 = parse_args(&s(&["r1-engine-int8", "m", "w", "out.jsonl", "tr.jsonl"])).unwrap();
        assert_eq!(c5.workers, 1);
        assert_eq!(c5.trace_path.as_deref(), Some(Path::new("tr.jsonl")));

        let cw = parse_args(&s(&[
            "r1-engine-int8",
            "m",
            "w",
            "out.jsonl",
            "--workers",
            "8",
        ]))
        .unwrap();
        assert_eq!(cw.workers, 8);

        let cweq = parse_args(&s(&[
            "--workers=4",
            "r1-engine-int8",
            "m",
            "w",
            "out.jsonl",
        ]))
        .unwrap();
        assert_eq!(cweq.workers, 4);

        let cbeam = parse_args(&s(&[
            "--chunk",
            "4",
            "--beam",
            "4",
            "--beam-max-symbols",
            "1",
            "--beam-margin",
            "3.0",
            "--lm",
            "lm.ulm",
            "--lm-weight",
            "0.7",
            "--lodr-weight",
            "0.5",
            "r1-engine-int8",
            "m",
            "w",
            "out.jsonl",
        ]))
        .unwrap();
        assert_eq!(cbeam.chunk, Some(4));
        assert_eq!(cbeam.beam, 4);
        assert_eq!(cbeam.beam_max_symbols, 1);
        assert!((cbeam.beam_margin - 3.0).abs() < 1e-6);
        assert_eq!(cbeam.lm_path.as_deref(), Some(Path::new("lm.ulm")));
        assert!((cbeam.lm_weight - 0.7).abs() < 1e-6);
        assert!((cbeam.lodr_weight - 0.5).abs() < 1e-6);

        assert!(parse_args(&s(&["--workers", "0", "r1-engine-int8", "m", "w", "o"])).is_err());
        assert!(parse_args(&s(&[
            "--workers",
            "2",
            "r1-engine-int8",
            "m",
            "w",
            "o",
            "tr"
        ]))
        .is_err());
        // Beam/LM flags must fail closed on non-native backends.
        assert!(parse_args(&s(&["--beam", "4", "zipformer-int8", "m", "w", "o"])).is_err());
        assert!(parse_args(&s(&[
            "--lm",
            "lm.ulm",
            "--lm-weight",
            "0.7",
            "r1-ort-fp32",
            "m",
            "w",
            "o"
        ]))
        .is_err());
        // --lm with non-positive --lm-weight or orphan LM weights must fail closed.
        assert!(parse_args(&s(&["--lm", "lm.ulm", "r1-engine-int8", "m", "w", "o"])).is_err());
        assert!(parse_args(&s(&[
            "--lm",
            "lm.ulm",
            "--lm-weight",
            "0.0",
            "r1-engine-int8",
            "m",
            "w",
            "o"
        ]))
        .is_err());
        assert!(parse_args(&s(&["--lm-weight", "0.7", "r1-engine-int8", "m", "w", "o"])).is_err());
    }

    #[test]
    fn shard_inputs_balances_durations_deterministically() {
        let items = vec![
            ("u00000".into(), PathBuf::from("u00000.wav"), 100),
            ("u00001".into(), PathBuf::from("u00001.wav"), 500),
            ("u00002".into(), PathBuf::from("u00002.wav"), 400),
            ("u00003".into(), PathBuf::from("u00003.wav"), 300),
            ("u00004".into(), PathBuf::from("u00004.wav"), 200),
            ("u00005".into(), PathBuf::from("u00005.wav"), 100),
        ];
        let shards = shard_inputs(items, 2);
        assert_eq!(shards.len(), 2);
        assert_eq!(shards[0].len() + shards[1].len(), 6);
        let sums: Vec<u64> = shards
            .iter()
            .map(|sh| {
                sh.iter()
                    .map(|(id, _)| match id.as_str() {
                        "u00001" => 500,
                        "u00002" => 400,
                        "u00003" => 300,
                        "u00004" => 200,
                        _ => 100,
                    })
                    .sum()
            })
            .collect();
        assert_eq!(sums, vec![800, 800]);
    }

    #[test]
    fn decode_parallel_matches_single_worker_byte_for_byte() {
        let dir = std::env::temp_dir().join(format!("ultra-cold-wav-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        for (i, samples) in [1600usize, 3200, 4800, 2400, 4000, 1920]
            .into_iter()
            .enumerate()
        {
            write_wav_16k(&dir.join(format!("u{i:05}.wav")), samples);
        }
        let entries = inputs(&dir).unwrap();
        let backend: Arc<dyn Backend> = Arc::new(FakeBackend::new(FakeBackendConfig::default()));

        let mut seq = Vec::new();
        for (id, path, _) in &entries {
            seq.push((id.clone(), decode(&*backend, id, path, None).unwrap()));
        }
        let par = decode_parallel(&backend, entries, 4).unwrap();
        assert_eq!(seq, par);
        let _ = fs::remove_dir_all(&dir);
    }
}
