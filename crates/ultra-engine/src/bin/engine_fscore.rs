//! `ultra-engine-fscore`: CPU forced-alignment scorer for `(wav, candidates[])` manifests.
//!
//! ```text
//! ultra-engine-fscore --model <dir> --manifest <in.jsonl> --out <out.jsonl> \
//!     [--chunk 16] [--tail-pad-ms 320] [--lm <M5.ulm>] [--threads 8] [--precision int8]
//! ```

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use serde_json::{json, Value};
use ultra_engine::commit::SpmTokenizer;
use ultra_engine::fscore::{encode_pcm, score_tokens, tokenize_text, ForcedScore, FscoreWorkspace};
use ultra_engine::{Engine, Isa, Model, NgramLm, Precision};
use ultra_frontend::read_audio_16k_mono;

type RowResult = Result<(Value, usize, usize), String>;

struct Args {
    model: PathBuf,
    manifest: PathBuf,
    out: PathBuf,
    chunk: usize,
    tail_pad_ms: usize,
    lm_path: Option<PathBuf>,
    threads: usize,
    precision: Precision,
    isa: Isa,
    emit_token_ids: bool,
}

struct ManifestRow {
    uid: String,
    wav: PathBuf,
    candidates: Vec<String>,
    labels: Option<Vec<String>>,
}

fn parse_args<I: IntoIterator<Item = String>>(iter: I) -> Result<Args, String> {
    let mut it = iter.into_iter();
    let mut a = Args {
        model: PathBuf::new(),
        manifest: PathBuf::new(),
        out: PathBuf::new(),
        chunk: 16,
        tail_pad_ms: 320,
        lm_path: None,
        threads: 1,
        precision: Precision::Int8,
        isa: Isa::detect(),
        emit_token_ids: false,
    };
    while let Some(k) = it.next() {
        let mut v = || it.next().ok_or_else(|| format!("{k} needs a value"));
        match k.as_str() {
            "--model" | "--weights" => a.model = v()?.into(),
            "--manifest" => a.manifest = v()?.into(),
            "--out" => a.out = v()?.into(),
            "--chunk" => a.chunk = v()?.parse().map_err(|e| format!("{k}: {e}"))?,
            "--tail-pad-ms" => a.tail_pad_ms = v()?.parse().map_err(|e| format!("{k}: {e}"))?,
            "--lm" => a.lm_path = Some(v()?.into()),
            "--threads" => a.threads = v()?.parse().map_err(|e| format!("{k}: {e}"))?,
            "--precision" => a.precision = Precision::parse(&v()?).map_err(|e| e.to_string())?,
            "--isa" => {
                a.isa = match v()?.as_str() {
                    "scalar" => Isa::Scalar,
                    "avx2fma" => Isa::Avx2Fma,
                    "avx512vnni" => Isa::Avx512Vnni,
                    "neon" => Isa::Neon,
                    o => return Err(format!("unknown isa {o}")),
                }
            }
            "--emit-token-ids" => a.emit_token_ids = true,
            o => return Err(format!("unknown argument {o}")),
        }
    }
    if a.model.as_os_str().is_empty()
        || a.manifest.as_os_str().is_empty()
        || a.out.as_os_str().is_empty()
    {
        return Err("--model, --manifest, and --out are required".into());
    }
    if a.threads == 0 || a.chunk == 0 {
        return Err("--threads and --chunk must be >= 1".into());
    }
    Ok(a)
}

fn load_manifest(path: &Path) -> Result<Vec<ManifestRow>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let base = path.parent().unwrap_or_else(|| Path::new("."));
    let mut rows = Vec::new();
    for (idx, line) in text.lines().map(str::trim).enumerate() {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let v: Value = serde_json::from_str(line).map_err(|e| format!("line {}: {e}", idx + 1))?;
        let uid = v["uid"]
            .as_str()
            .or_else(|| v["id"].as_str())
            .ok_or_else(|| format!("line {}: missing uid/id", idx + 1))?
            .to_owned();
        let wav_s = v["wav"]
            .as_str()
            .or_else(|| v["audio_filepath"].as_str())
            .or_else(|| v["path"].as_str())
            .ok_or_else(|| format!("line {}: missing wav/path", idx + 1))?;
        let wav = if Path::new(wav_s).is_absolute() {
            PathBuf::from(wav_s)
        } else {
            base.join(wav_s)
        };
        let candidates = if let Some(arr) = v["candidates"].as_array() {
            arr.iter()
                .map(|c| {
                    c.as_str()
                        .map(str::to_owned)
                        .ok_or_else(|| format!("line {}: non-string candidate", idx + 1))
                })
                .collect::<Result<Vec<_>, _>>()?
        } else if let Some(t) = v["text"].as_str() {
            vec![t.to_owned()]
        } else {
            return Err(format!("line {}: missing candidates or text", idx + 1));
        };
        let labels = v["labels"].as_array().map(|arr| {
            arr.iter()
                .map(|x| x.as_str().unwrap_or("").to_owned())
                .collect()
        });
        rows.push(ManifestRow {
            uid,
            wav,
            candidates,
            labels,
        });
    }
    Ok(rows)
}

fn score_row(
    engine: &Engine,
    spm: &SpmTokenizer,
    lm: Option<&NgramLm>,
    row: &ManifestRow,
    tail_pad_samples: usize,
    emit_token_ids: bool,
    ws: &mut FscoreWorkspace,
) -> RowResult {
    let pcm = read_audio_16k_mono(&row.wav).map_err(|e| format!("{}: {e}", row.wav.display()))?;
    let enc = encode_pcm(engine, &pcm, tail_pad_samples, ws).map_err(|e| e.to_string())?;
    let blank_id = engine.model.arch.blank;
    let mut cache: HashMap<&str, (ForcedScore, Vec<u32>)> = HashMap::new();
    let mut scored = Vec::with_capacity(row.candidates.len());

    for (ci, text) in row.candidates.iter().enumerate() {
        if !cache.contains_key(text.as_str()) {
            let toks = tokenize_text(spm, text, blank_id).map_err(|e| e.to_string())?;
            let sc = score_tokens(engine, &enc, &toks, lm, ws).map_err(|e| e.to_string())?;
            cache.insert(text.as_str(), (sc, toks));
        }
        let (sc, toks) = &cache[text.as_str()];
        let mut obj = serde_json::Map::new();
        if let Some(l) = row.labels.as_ref().and_then(|ls| ls.get(ci)) {
            obj.insert("label".into(), json!(l));
        }
        obj.insert("text".into(), json!(text));
        obj.insert(
            "rnnt_full_logp".into(),
            json!(sc.rnnt_full_logp.is_finite().then_some(sc.rnnt_full_logp)),
        );
        obj.insert(
            "rnnt_sync_logp".into(),
            json!(sc.rnnt_sync_logp.is_finite().then_some(sc.rnnt_sync_logp)),
        );
        obj.insert("lm_logp".into(), json!(sc.lm_logp));
        obj.insert("lodr_logp".into(), json!(sc.lodr_logp));
        obj.insert("n_tokens".into(), json!(sc.n_tokens));
        obj.insert("n_frames".into(), json!(sc.n_frames));
        if emit_token_ids {
            obj.insert("token_ids".into(), json!(toks));
        }
        scored.push(Value::Object(obj));
    }

    let out = json!({
        "uid": row.uid,
        "wav": row.wav,
        "n_frames": enc.n_frames,
        "candidates": scored,
    });
    Ok((out, enc.n_frames, row.candidates.len()))
}

fn run(a: &Args) -> Result<Value, String> {
    let t0 = Instant::now();
    let rows = load_manifest(&a.manifest)?;
    let mut model = Model::load(&a.model, true).map_err(|e| e.to_string())?;
    if a.precision == Precision::Int8 && model.int8_prepared.is_none() {
        model.quantize_int8();
    }
    let spm =
        Arc::new(SpmTokenizer::from_file(&a.model.join("bpe.model")).map_err(|e| e.to_string())?);
    let lm = a
        .lm_path
        .as_ref()
        .map(|p| NgramLm::load(p).map(Arc::new).map_err(|e| e.to_string()))
        .transpose()?;
    if let Some(l) = &lm {
        if l.vocab != model.arch.vocab {
            return Err(format!(
                "LM vocab {} != model vocab {}",
                l.vocab, model.arch.vocab
            ));
        }
    }
    let engine = Arc::new(
        Engine::new(Arc::new(model), a.isa, a.precision)
            .map_err(|e| e.to_string())?
            .with_chunk(a.chunk)
            .map_err(|e| e.to_string())?,
    );
    let (tail_pad_samples, n_rows) = (16 * a.tail_pad_ms, rows.len());
    let n_workers = a.threads.min(n_rows.max(1));
    let next_idx = AtomicUsize::new(0);
    let results: Vec<Mutex<Option<RowResult>>> = (0..n_rows).map(|_| Mutex::new(None)).collect();

    std::thread::scope(|s| {
        for _ in 0..n_workers {
            let (eng, tok, lm_ref) = (Arc::clone(&engine), Arc::clone(&spm), lm.clone());
            let (rows_r, next_r, res_r, emit_ids) = (&rows, &next_idx, &results, a.emit_token_ids);
            s.spawn(move || {
                let mut ws = FscoreWorkspace::default();
                loop {
                    let i = next_r.fetch_add(1, Ordering::Relaxed);
                    if i >= rows_r.len() {
                        break;
                    }
                    let r = score_row(
                        &eng,
                        &tok,
                        lm_ref.as_deref(),
                        &rows_r[i],
                        tail_pad_samples,
                        emit_ids,
                        &mut ws,
                    );
                    if let Ok(mut slot) = res_r[i].lock() {
                        *slot = Some(r);
                    }
                }
            });
        }
    });

    if let Some(p) = a.out.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(p).map_err(|e| e.to_string())?;
    }
    let mut w = BufWriter::new(File::create(&a.out).map_err(|e| e.to_string())?);
    let (mut total_frames, mut total_cands) = (0usize, 0usize);
    for (i, slot) in results.into_iter().enumerate() {
        let item = slot
            .into_inner()
            .map_err(|_| format!("worker mutex poisoned at row {i}"))?
            .ok_or_else(|| format!("missing result for row {i}"))??;
        writeln!(w, "{}", item.0).map_err(|e| e.to_string())?;
        total_frames += item.1;
        total_cands += item.2;
    }
    w.flush().map_err(|e| e.to_string())?;

    Ok(json!({
        "utterances": n_rows, "candidates": total_cands, "total_frames": total_frames,
        "chunk": a.chunk, "tail_pad_ms": a.tail_pad_ms, "threads": n_workers,
        "precision": a.precision.as_str(), "isa": a.isa.name(), "lm": a.lm_path,
        "wall_s": t0.elapsed().as_secs_f64(), "out": a.out,
    }))
}

fn main() -> ExitCode {
    match parse_args(std::env::args().skip(1)).and_then(|a| run(&a)) {
        Ok(s) => {
            println!("{}", serde_json::to_string_pretty(&s).unwrap_or_default());
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("ultra-engine-fscore: {e}");
            ExitCode::FAILURE
        }
    }
}
