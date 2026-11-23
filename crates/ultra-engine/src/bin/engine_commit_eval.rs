//! `ultra-engine-commit`: H45 dual-pass endpoint commit evaluator CLI.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

use serde_json::{json, Value};
use ultra_engine::commit::{CommitConfig, CommitPassSession, DEFAULT_SAMPLE_RATE};
use ultra_engine::panel::Panel;
use ultra_engine::workload::Workload;
use ultra_engine::{Engine, Isa, Model, Precision};
use ultra_frontend::npz::{load_npz_audio_only, load_npz_references};
use ultra_frontend::read_audio_16k_mono;

struct Args {
    model: PathBuf,
    manifest: PathBuf,
    out: PathBuf,
    commit_chunk: usize,
    tail_pad_ms: usize,
    threads: usize,
    precision: Precision,
    limit: usize,
    cache_dir: Option<PathBuf>,
}

fn parse_args<I: IntoIterator<Item = String>>(args: I) -> Result<Args, String> {
    let mut it = args.into_iter();
    let mut a = Args {
        model: PathBuf::new(),
        manifest: PathBuf::new(),
        out: PathBuf::new(),
        commit_chunk: 16,
        tail_pad_ms: 320,
        threads: 1,
        precision: Precision::Int8,
        limit: 0,
        cache_dir: None,
    };
    while let Some(k) = it.next() {
        let mut v = || it.next().ok_or_else(|| format!("{k} needs a value"));
        match k.as_str() {
            "--model" | "--weights" => a.model = v()?.into(),
            "--manifest" => a.manifest = v()?.into(),
            "--out" => a.out = v()?.into(),
            "--commit-chunk" => a.commit_chunk = v()?.parse().map_err(|e| format!("{k}: {e}"))?,
            "--tail-pad-ms" => a.tail_pad_ms = v()?.parse().map_err(|e| format!("{k}: {e}"))?,
            "--threads" => a.threads = v()?.parse().map_err(|e| format!("{k}: {e}"))?,
            "--precision" => a.precision = Precision::parse(&v()?).map_err(|e| e.to_string())?,
            "--limit" => a.limit = v()?.parse().map_err(|e| format!("{k}: {e}"))?,
            "--cache-dir" => a.cache_dir = Some(v()?.into()),
            o => return Err(format!("unknown argument {o}")),
        }
    }
    if a.model.as_os_str().is_empty()
        || a.manifest.as_os_str().is_empty()
        || a.out.as_os_str().is_empty()
    {
        return Err("--model, --manifest, and --out are required".into());
    }
    if a.threads == 0 {
        return Err("--threads must be positive".into());
    }
    Ok(a)
}

fn load_manifest(a: &Args) -> Result<Vec<(String, Vec<i16>)>, String> {
    let path = &a.manifest;
    let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("");
    let lim = (a.limit > 0).then_some(a.limit);
    if ext.eq_ignore_ascii_case("npz") {
        let au = load_npz_audio_only(path, 0, lim).map_err(|e| e.to_string())?;
        let rf = load_npz_references(path, 0, lim).map_err(|e| e.to_string())?;
        return Ok(au
            .into_iter()
            .zip(rf)
            .map(|(x, r)| (r.uid, x.pcm))
            .collect());
    }
    if ext.eq_ignore_ascii_case("json") {
        let cache = a.cache_dir.clone().unwrap_or_else(|| {
            PathBuf::from(std::env::var("JSR_DATA").unwrap_or("/mnt/xbabe3-external/jsr".into()))
                .join("cache/librispeech_pcm")
        });
        let wl = Workload::panel(&Panel::read(path)?, &cache)?;
        let rf = wl.references()?;
        let take = lim.unwrap_or(wl.items.len()).min(wl.items.len());
        return Ok(wl
            .items
            .into_iter()
            .zip(rf)
            .take(take)
            .map(|(i, r)| (r.uid, i.pcm))
            .collect());
    }
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let base = path.parent().unwrap_or_else(|| Path::new("."));
    let mut out = Vec::new();
    for (idx, line) in text.lines().map(str::trim).enumerate() {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if lim.is_some_and(|n| out.len() >= n) {
            break;
        }
        let (uid, ap) = if line.starts_with('{') {
            let v: Value =
                serde_json::from_str(line).map_err(|e| format!("line {}: {e}", idx + 1))?;
            let uid = v["uid"]
                .as_str()
                .or_else(|| v["id"].as_str())
                .map(str::to_owned);
            let ap = v["wav"]
                .as_str()
                .or_else(|| v["audio_filepath"].as_str())
                .or_else(|| v["path"].as_str())
                .or_else(|| v["file"].as_str())
                .ok_or_else(|| format!("line {}: missing wav/path", idx + 1))?;
            (uid, ap.to_owned())
        } else if let Some((u, p)) = line.split_once('\t') {
            (Some(u.trim().to_owned()), p.trim().to_owned())
        } else {
            (None, line.to_owned())
        };
        let p = if Path::new(&ap).is_absolute() {
            PathBuf::from(&ap)
        } else {
            base.join(&ap)
        };
        let pcm = read_audio_16k_mono(&p).map_err(|e| format!("{}: {e}", p.display()))?;
        let id = uid.unwrap_or_else(|| {
            p.file_stem().map_or_else(
                || format!("utt_{:05}", out.len()),
                |s| s.to_string_lossy().into_owned(),
            )
        });
        out.push((id, pcm));
    }
    Ok(out)
}

fn pct(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        0.0
    } else {
        sorted[(((sorted.len() - 1) as f64 * p).round() as usize).min(sorted.len() - 1)]
    }
}

fn run(a: &Args) -> Result<Value, String> {
    let mut model = Model::load(&a.model, true).map_err(|e| e.to_string())?;
    if model.int8_prepared.is_none() && a.precision == Precision::Int8 {
        model.quantize_int8();
    }
    let isa = Isa::detect();
    let eng = Engine::new(Arc::new(model), isa, a.precision)
        .map_err(|e| e.to_string())?
        .with_gemm_threads(a.threads);
    let cfg = CommitConfig {
        commit_chunk: a.commit_chunk,
        tail_pad_samples: a.tail_pad_ms * DEFAULT_SAMPLE_RATE / 1000,
        ..CommitConfig::default()
    };
    let mut sess = CommitPassSession::new(eng, cfg).map_err(|e| e.to_string())?;
    let items = load_manifest(a)?;
    if items.is_empty() {
        return Err("empty manifest".into());
    }
    if let Some(p) = a.out.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(p).map_err(|e| e.to_string())?;
    }
    let mut out = BufWriter::new(File::create(&a.out).map_err(|e| e.to_string())?);
    let (mut tot_s, mut tot_prov, mut tot_commit) = (0.0, 0.0, 0.0);
    let (mut revised, mut cms) = (0usize, Vec::with_capacity(items.len()));
    for (uid, pcm) in &items {
        let t0 = Instant::now();
        for pkt in pcm.chunks(320) {
            sess.accept_i16(pkt).map_err(|e| e.to_string())?;
        }
        let acc_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let t1 = Instant::now();
        let r = sess.commit_endpoint().map_err(|e| e.to_string())?;
        let call_ms = t1.elapsed().as_secs_f64() * 1000.0;
        sess.reset_all();
        let prov_ms = acc_ms + (call_ms - r.commit_wall_ms).max(0.0);
        let audio_s = r.audio_ms / 1000.0;
        tot_s += audio_s;
        tot_prov += prov_ms;
        tot_commit += r.commit_wall_ms;
        revised += usize::from(r.revised);
        cms.push(r.commit_wall_ms);
        writeln!(out, "{}", json!({"uid": uid, "provisional": r.provisional, "committed": r.committed, "revised": r.revised, "audio_s": audio_s, "prov_wall_ms": prov_ms, "commit_wall_ms": r.commit_wall_ms}))
            .map_err(|e| e.to_string())?;
    }
    cms.sort_by(f64::total_cmp);
    let summary = json!({"summary": true, "utterances": items.len(), "revised_count": revised, "total_audio_s": tot_s, "total_prov_ms": tot_prov, "total_commit_ms": tot_commit, "revised_pct": 100.0 * revised as f64 / items.len() as f64, "p50_commit_ms": pct(&cms, 0.50), "p95_commit_ms": pct(&cms, 0.95), "commit_chunk": a.commit_chunk, "tail_pad_ms": a.tail_pad_ms, "threads": a.threads, "precision": format!("{:?}", a.precision), "isa": isa.name()});
    writeln!(out, "{summary}").map_err(|e| e.to_string())?;
    out.flush().map_err(|e| e.to_string())?;
    Ok(summary)
}

fn main() -> ExitCode {
    match parse_args(std::env::args().skip(1)).and_then(|a| run(&a)) {
        Ok(s) => {
            println!("{}", serde_json::to_string_pretty(&s).unwrap_or_default());
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("ultra-engine-commit: {e}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn commit_eval_cli_args_and_percentiles() {
        let s = "--model m --manifest in.jsonl --out o.jsonl --commit-chunk 16 --tail-pad-ms 320 --threads 2";
        let a = parse_args(s.split_whitespace().map(String::from)).expect("parse");
        assert_eq!((a.commit_chunk, a.tail_pad_ms, a.threads), (16, 320, 2));
        assert!(parse_args(["--model".into(), "m".into()]).is_err());
        let xs = [10.0, 20.0, 30.0, 40.0, 50.0];
        assert_eq!((pct(&xs, 0.50), pct(&xs, 0.95)), (30.0, 50.0));
    }
}
