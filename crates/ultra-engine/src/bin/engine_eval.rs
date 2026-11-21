//! `ultra-engine-eval`: stream DEV utterances through the engine in real-time-sized packets
//! (the `train.infer.run_streaming` schedule: `--streams` concurrent slots, `--packet-ms`
//! packets, every ready stream decoded once per round), then score WER and compare token
//! ids with a reference run (`ultra.b1-parity/v1` JSON, e.g. the FP32 bundle's 400 DEV).
//!
//! ```text
//! ultra-engine-eval --weights <dir> --npz dev-clean.npz --limit 400 --precision fp32 \
//!     --streams 16 --out <new run dir> [--reference r1_d1_400dev_fp32.json] [--isa scalar]
//! ```
//! Writes `<out>/engine_eval.json` (summary + per-utterance rows). CPU time is this thread's
//! scheduler run time (`/proc/thread-self/schedstat`); all speed numbers are exploratory.

use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

use serde_json::json;
use ultra_engine::{Engine, EngineStream, Isa, Model, Precision, Workspace};
#[cfg(feature = "wer")]
use ultra_eval::{normalise::normalise_words, wer::align};

/// (errors, ref words, substitutions, deletions, insertions) of one utterance.
#[cfg(feature = "wer")]
fn score(reference: &str, hyp: &str) -> Option<[u64; 5]> {
    let al = align(&normalise_words(reference), &normalise_words(hyp));
    let c = al.counts;
    Some([
        c.errors(),
        c.ref_words,
        c.substitutions,
        c.deletions,
        c.insertions,
    ])
}

/// Built without the `wer` feature: transcripts are written unscored.
#[cfg(not(feature = "wer"))]
fn score(_reference: &str, _hyp: &str) -> Option<[u64; 5]> {
    None
}
use ultra_frontend::npz::{load_npz_audio_only, load_npz_references};

struct Args {
    weights: PathBuf,
    npz: PathBuf,
    out: PathBuf,
    reference: Option<PathBuf>,
    limit: usize,
    streams: usize,
    packet_ms: usize,
    precision: Precision,
    max_chunks: usize,
    int8_stem: bool,
    act7: bool,
    gemm_threads: usize,
    isa: Isa,
    profile: bool,
}

fn parse() -> Result<Args, String> {
    let mut it = std::env::args().skip(1);
    let mut a = Args {
        weights: PathBuf::new(),
        npz: PathBuf::new(),
        out: PathBuf::new(),
        reference: None,
        limit: 400,
        streams: 16,
        packet_ms: 20,
        precision: Precision::F32,
        max_chunks: 1,
        int8_stem: false,
        act7: false,
        gemm_threads: 1,
        isa: Isa::detect(),
        profile: false,
    };
    while let Some(k) = it.next() {
        let mut v = || it.next().ok_or_else(|| format!("{k} needs a value"));
        match k.as_str() {
            "--weights" => a.weights = v()?.into(),
            "--npz" => a.npz = v()?.into(),
            "--out" => a.out = v()?.into(),
            "--profile" => a.profile = true,
            "--int8-stem" => a.int8_stem = true,
            "--act7" => a.act7 = true,
            "--gemm-threads" => {
                a.gemm_threads = v()?.parse().map_err(|e| format!("--gemm-threads: {e}"))?
            }
            "--max-chunks" => {
                a.max_chunks = v()?.parse().map_err(|e| format!("--max-chunks: {e}"))?
            }
            "--reference" => a.reference = Some(v()?.into()),
            "--limit" => a.limit = v()?.parse().map_err(|e| format!("--limit: {e}"))?,
            "--streams" => a.streams = v()?.parse().map_err(|e| format!("--streams: {e}"))?,
            "--packet-ms" => a.packet_ms = v()?.parse().map_err(|e| format!("--packet-ms: {e}"))?,
            "--precision" => a.precision = Precision::parse(&v()?).map_err(|e| e.to_string())?,
            "--isa" => {
                a.isa = match v()?.as_str() {
                    "scalar" => Isa::Scalar,
                    "avx2fma" => Isa::Avx2Fma,
                    "neon" => Isa::Neon,
                    o => return Err(format!("unknown isa {o}")),
                }
            }
            o => return Err(format!("unknown argument {o}")),
        }
    }
    if a.weights.as_os_str().is_empty()
        || a.npz.as_os_str().is_empty()
        || a.out.as_os_str().is_empty()
    {
        return Err("--weights, --npz and --out are required".into());
    }
    if a.streams == 0 || a.packet_ms == 0 {
        return Err("--streams and --packet-ms must be positive".into());
    }
    Ok(a)
}

/// This thread's CPU run time in seconds.
fn thread_cpu_s() -> f64 {
    // Linux: this thread's scheduler run time. Elsewhere (macOS): process user + system time
    // (the binaries decode on one thread; pool workers, if enabled, are then included).
    if let Some(v) = std::fs::read_to_string("/proc/thread-self/schedstat")
        .ok()
        .and_then(|s| {
            s.split_whitespace()
                .next()
                .and_then(|v| v.parse::<f64>().ok())
        })
    {
        return v * 1e-9;
    }
    nix::sys::resource::getrusage(nix::sys::resource::UsageWho::RUSAGE_SELF).map_or(f64::NAN, |u| {
        let t = |tv: nix::sys::time::TimeVal| tv.tv_sec() as f64 + tv.tv_usec() as f64 * 1e-6;
        t(u.user_time()) + t(u.system_time())
    })
}

/// Reference token ids keyed by utterance id (`rows[].id`, `rows[].py_tokens`).
fn reference_tokens(path: &PathBuf) -> Result<std::collections::HashMap<String, Vec<u32>>, String> {
    let v: serde_json::Value =
        serde_json::from_slice(&std::fs::read(path).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    let rows = v["rows"].as_array().ok_or("reference has no rows")?;
    let mut out = std::collections::HashMap::new();
    for r in rows {
        let id = r["id"]
            .as_str()
            .ok_or("reference row without id")?
            .to_owned();
        let toks = r["py_tokens"]
            .as_array()
            .ok_or("reference row without py_tokens")?
            .iter()
            .map(|t| {
                t.as_u64()
                    .and_then(|x| u32::try_from(x).ok())
                    .ok_or("bad token")
            })
            .collect::<Result<Vec<u32>, _>>()?;
        if out.insert(id.clone(), toks).is_some() {
            return Err(format!("duplicate reference id {id}"));
        }
    }
    Ok(out)
}

/// SHA-256 of the running executable (binds results to the exact binary).
fn exe_sha256() -> String {
    std::fs::read("/proc/self/exe")
        .map(|b| ultra_core::receipt::sha256_hex(&b))
        .unwrap_or_else(|_| "unavailable".into())
}

fn run(a: &Args) -> Result<serde_json::Value, String> {
    if a.out.join("engine_eval.json").exists() {
        return Err(format!(
            "{} already has engine_eval.json; use a new run dir",
            a.out.display()
        ));
    }
    std::fs::create_dir_all(&a.out).map_err(|e| e.to_string())?;
    let t_load = Instant::now();
    let mut model = Model::load(&a.weights, true).map_err(|e| e.to_string())?;
    // Ablation hook: ULTRA_ENGINE_INT8_SKIP=kind[,kind...] keeps those layer kinds fp32
    // (kinds: stem_c2 stem_out ff1_w1 ff1_w2 qkv att_out pw1 pw2 ff2_w1 ff2_w2 enc_proj joiner_out).
    let mut skip: Vec<String> = std::env::var("ULTRA_ENGINE_INT8_SKIP")
        .map(|v| {
            v.split(',')
                .filter(|s| !s.is_empty())
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default();
    if let Some(p) = &model.int8_prepared {
        // Prepared artifact: no weight quantisation here; its declared fp32 kinds rule.
        if !skip.is_empty() {
            return Err("ULTRA_ENGINE_INT8_SKIP conflicts with a prepared-INT8 directory".into());
        }
        skip = p.fp32_kinds.clone();
    } else if a.precision == Precision::Int8 {
        model.quantize_int8_where(|kind, _| !skip.iter().any(|s| s == kind));
    }
    // NEW numerical candidate (off by default): post-training int8 for a still-fp32 stem.
    let stem_quantised = if a.int8_stem && a.precision == Precision::Int8 {
        model.quantize_fp32_kinds_int8(&["stem_c2", "stem_out"])
    } else {
        0
    };
    let load_s = t_load.elapsed().as_secs_f64();
    let engine = Engine::new(Arc::new(model), a.isa, a.precision)
        .map_err(|e| e.to_string())?
        .with_max_chunks(a.max_chunks)
        .with_act7(a.act7)
        .with_gemm_threads(a.gemm_threads);
    let utts = load_npz_audio_only(&a.npz, 0, Some(a.limit)).map_err(|e| e.to_string())?;
    if utts.is_empty() {
        return Err("empty workload".into());
    }
    let packet = 16 * a.packet_ms;
    let mut queue: std::collections::VecDeque<usize> = (0..utts.len()).collect();
    let mut active: Vec<(usize, EngineStream, usize)> = Vec::new();
    let mut done: Vec<Option<(Vec<u32>, String, u64)>> = vec![None; utts.len()];
    let mut ws = Workspace::default();
    ws.set_profiling(a.profile);
    let (c0, w0) = (thread_cpu_s(), Instant::now());
    let mut batches = 0u64;
    let mut decode_wall = 0.0f64;
    let mut batch_rows = 0u64;
    while !queue.is_empty() || !active.is_empty() {
        while active.len() < a.streams {
            match queue.pop_front() {
                Some(i) => active.push((i, engine.create_stream(), 0)),
                None => break,
            }
        }
        for (i, s, cur) in active.iter_mut() {
            let pcm = &utts[*i].pcm;
            if *cur < pcm.len() {
                let end = (*cur + packet).min(pcm.len());
                s.accept_i16(&pcm[*cur..end]).map_err(|e| e.to_string())?;
                *cur = end;
                if end >= pcm.len() {
                    s.input_finished();
                }
            }
        }
        loop {
            let mut refs_mut: Vec<&mut EngineStream> =
                active.iter_mut().map(|(_, s, _)| s).collect();
            let td = Instant::now();
            let n = engine
                .decode(&mut refs_mut, &mut ws)
                .map_err(|e| e.to_string())?;
            decode_wall += td.elapsed().as_secs_f64();
            if n == 0 {
                break;
            }
            batches += 1;
            batch_rows += n as u64;
        }
        let mut k = 0;
        while k < active.len() {
            let (i, s, cur) = &active[k];
            if *cur >= utts[*i].pcm.len() && !engine.is_ready(s) {
                done[*i] = Some((s.tokens().to_vec(), engine.text(s), s.forced_advances()));
                active.swap_remove(k);
            } else {
                k += 1;
            }
        }
    }
    let cpu_s = thread_cpu_s() - c0;
    let wall_s = w0.elapsed().as_secs_f64();
    let audio_s: f64 = utts.iter().map(|u| u.seconds()).sum();
    if !cpu_s.is_finite() || cpu_s <= 0.0 || audio_s <= 0.0 {
        return Err(format!("unusable CPU clock ({cpu_s}) or audio ({audio_s})"));
    }
    // References are read only after inference and timing have finished.
    let refs = load_npz_references(&a.npz, 0, Some(a.limit)).map_err(|e| e.to_string())?;
    let ref_toks = a.reference.as_ref().map(reference_tokens).transpose()?;
    let (mut subs, mut dels, mut ins) = (0u64, 0u64, 0u64);
    let (mut errs, mut words, mut same_tok, mut forced) = (0u64, 0u64, 0usize, 0u64);
    let mut rows = Vec::new();
    for (i, d) in done.into_iter().enumerate() {
        let (toks, text, f) = d.ok_or("an utterance never finished")?;
        if let Some([e, w, s, d, n]) = score(&refs[i].text, &text) {
            errs += e;
            words += w;
            subs += s;
            dels += d;
            ins += n;
        }
        forced += f;
        let eq = ref_toks
            .as_ref()
            .map(|r| r.get(&refs[i].uid) == Some(&toks));
        if eq == Some(true) {
            same_tok += 1;
        }
        rows.push(
            json!({"index": i, "id": refs[i].uid, "hyp": text, "tokens": toks,
                         "equal_reference_tokens": eq}),
        );
    }
    let wer = (words > 0).then(|| 100.0 * errs as f64 / words as f64);
    let summary = json!({
        "schema": "ultra.engine-eval/v1",
        "exploratory": true,
        "weights": a.weights, "arch_checkpoint_sha256": engine.model.arch.checkpoint_sha256,
        "weights_sha256": engine.model.arch.weights_sha256,
        "npz": a.npz, "utterances": utts.len(), "audio_s": audio_s,
        "precision": a.precision.as_str(), "isa": a.isa.name(), "int8_fp32_kinds": skip,
        "int8_prepared_sha256": engine.model.int8_prepared.as_ref().map(|p| p.sha256.clone()),
        "streams": a.streams, "max_chunks": a.max_chunks, "act7": a.act7, "int8_stem_layers_quantised": stem_quantised, "gemm_threads": a.gemm_threads, "packet_ms": a.packet_ms,
        "wer_pct": wer, "ref_words": words, "errors": errs, "forced_advances": forced,
        "reference": a.reference, "identical_reference_tokens": ref_toks.as_ref().map(|_| same_tok),
        "cpu_s": cpu_s, "wall_s": wall_s, "decode_wall_s": decode_wall,
        "decode_wall_s_per_audio_s": decode_wall / audio_s, "cpu_s_per_audio_s": cpu_s / audio_s,
        "load_s": load_s, "decode_batches": batches,
        "mean_streams_per_batch": batch_rows as f64 / batches.max(1) as f64,
        "substitutions": subs, "deletions": dels, "insertions": ins,
        "executable_sha256": exe_sha256(),
    });
    let mut summary = summary;
    if a.profile {
        let (e, d) = ws.profile();
        let ph = |p: &ultra_engine::prof::Prof| {
            p.seconds()
                .into_iter()
                .filter(|(_, v)| *v > 0.0)
                .map(|(k, v)| (k.to_string(), json!(v / audio_s)))
                .collect::<serde_json::Map<_, _>>()
        };
        summary["phase_s_per_audio_s"] = json!({"encoder": ph(&e), "decoder": ph(&d)});
    }
    let mut full = summary.clone();
    full["rows"] = serde_json::Value::Array(rows);
    let text = serde_json::to_string_pretty(&full).map_err(|e| e.to_string())?;
    std::fs::write(a.out.join("engine_eval.json"), text).map_err(|e| e.to_string())?;
    Ok(summary)
}

fn main() -> ExitCode {
    match parse().and_then(|a| run(&a)) {
        Ok(s) => {
            println!("{}", serde_json::to_string_pretty(&s).unwrap_or_default());
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("ultra-engine-eval: {e}");
            ExitCode::FAILURE
        }
    }
}
