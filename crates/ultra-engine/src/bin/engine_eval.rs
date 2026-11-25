//! `ultra-engine-eval`: stream DEV utterances through the engine in real-time-sized packets
//! (the `train.infer.run_streaming` schedule: `--streams` concurrent slots, `--packet-ms`
//! packets, every ready stream decoded once per round), then score WER and compare token
//! ids with a reference run (`ultra.b1-parity/v1` JSON, e.g. the FP32 bundle's 400 DEV).
//!
//! ```text
//! ultra-engine-eval --weights <dir> --npz dev-clean.npz --limit 400 --precision fp32 \
//!     --streams 16 --out <new run dir> [--reference r1_d1_400dev_fp32.json] [--isa scalar]
//! ultra-engine-eval --weights <dir> --panel dev_selection_panel.json [--cache-dir <dir>] ...
//!     [--int8-fp32-kinds k,..] [--int8-stem-kinds stem_c2,stem_out] [--qualifying]
//! ```
//! `--panel` runs a verified DEV selection panel (manifest and shard hashes checked; rows
//! carry split and speaker for the cluster bootstrap). `--qualifying` refuses runtime weight
//! quantisation (int8 needs a prepared directory) and requires a panel. Every knob is a
//! flag recorded in the output; `ULTRA_ENGINE_*` environment variables are refused.
//! `test-*` (PUBLIC-TEST) splits are refused unless `--sealed-read <look-id>` is passed and the
//! `bench-freeze-v1` tag is an ancestor of the (clean) build commit; the read is appended to
//! `docs/ultra/LOOKS.jsonl` first (`ultra_engine::sealed`; `--repo`, `--looks` override).
//! Writes `<out>/engine_eval.json` (summary + per-utterance rows). CPU time is this thread's
//! scheduler run time (`/proc/thread-self/schedstat`); all speed numbers are exploratory.

use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

use serde_json::json;
use ultra_engine::panel::Panel;
use ultra_engine::workload::Workload;
use ultra_engine::{run_info, sealed, Engine, EngineStream, Isa, Model, Precision, Workspace};
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

struct Args {
    weights: PathBuf,
    npz: PathBuf,
    panel: Option<PathBuf>,
    cache_dir: PathBuf,
    qualifying: bool,
    int8_fp32_kinds: Vec<String>,
    int8_stem_kinds: Vec<String>,
    sealed_read: Option<String>,
    beam: usize,
    beam_max_symbols: usize,
    beam_margin: f32,
    lm_path: Option<PathBuf>,
    lm_weight: f32,
    lodr_weight: f32,
    blank_penalty: f32,
    kv_int8: bool,
    fold_prenorm: bool,
    repo: Option<PathBuf>,
    looks: Option<PathBuf>,
    out: PathBuf,
    reference: Option<PathBuf>,
    limit: usize,
    streams: usize,
    packet_ms: usize,
    precision: Precision,
    chunk: Option<usize>,
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
        panel: None,
        cache_dir: std::env::var_os("JSR_DATA")
            .map_or_else(|| PathBuf::from("/mnt/xbabe3-external/jsr"), PathBuf::from)
            .join("cache/librispeech_pcm"),
        qualifying: false,
        int8_fp32_kinds: Vec::new(),
        int8_stem_kinds: Vec::new(),
        sealed_read: None,
        beam: 1,
        beam_max_symbols: 1,
        beam_margin: 0.0,
        lm_path: None,
        lm_weight: 0.0,
        lodr_weight: 0.0,
        blank_penalty: 0.0,
        kv_int8: false,
        fold_prenorm: false,
        repo: None,
        looks: None,
        out: PathBuf::new(),
        reference: None,
        limit: 400,
        streams: 16,
        packet_ms: 20,
        precision: Precision::F32,
        chunk: None,
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
            "--panel" => a.panel = Some(v()?.into()),
            "--cache-dir" => a.cache_dir = v()?.into(),
            "--qualifying" => a.qualifying = true,
            "--sealed-read" => a.sealed_read = Some(v()?),
            "--beam" => a.beam = v()?.parse().map_err(|e| format!("--beam: {e}"))?,
            "--beam-max-symbols" => {
                a.beam_max_symbols = v()?
                    .parse()
                    .map_err(|e| format!("--beam-max-symbols: {e}"))?
            }
            "--beam-margin" => {
                a.beam_margin = v()?.parse().map_err(|e| format!("--beam-margin: {e}"))?
            }
            "--lm" => a.lm_path = Some(v()?.into()),
            "--lm-weight" => a.lm_weight = v()?.parse().map_err(|e| format!("--lm-weight: {e}"))?,
            "--lodr-weight" => {
                a.lodr_weight = v()?.parse().map_err(|e| format!("--lodr-weight: {e}"))?
            }
            "--blank-penalty" => {
                a.blank_penalty = v()?.parse().map_err(|e| format!("--blank-penalty: {e}"))?
            }
            "--kv-int8" => a.kv_int8 = true,
            "--fold-prenorm" => a.fold_prenorm = true,
            "--repo" => a.repo = Some(v()?.into()),
            "--looks" => a.looks = Some(v()?.into()),
            "--int8-fp32-kinds" => a.int8_fp32_kinds = kinds(&v()?),
            "--int8-stem-kinds" => {
                a.int8_stem_kinds = kinds(&v()?);
                a.int8_stem = true;
            }
            "--act7" => a.act7 = true,
            "--gemm-threads" => {
                a.gemm_threads = v()?.parse().map_err(|e| format!("--gemm-threads: {e}"))?
            }
            "--chunk" => {
                a.chunk = Some(v()?.parse().map_err(|e| format!("--chunk: {e}"))?);
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
                    "avx512vnni" => Isa::Avx512Vnni,
                    "neon" => Isa::Neon,
                    o => return Err(format!("unknown isa {o}")),
                }
            }
            o => return Err(format!("unknown argument {o}")),
        }
    }
    if a.weights.as_os_str().is_empty()
        || (a.npz.as_os_str().is_empty() == a.panel.is_none())
        || a.out.as_os_str().is_empty()
    {
        return Err("--weights, --out and exactly one of --npz / --panel are required".into());
    }
    if a.qualifying && a.panel.is_none() {
        return Err("--qualifying requires --panel".into());
    }
    if a.int8_stem && a.int8_stem_kinds.is_empty() {
        a.int8_stem_kinds = vec!["stem_c2".into(), "stem_out".into()];
    }
    if a.streams == 0 || a.packet_ms == 0 {
        return Err("--streams and --packet-ms must be positive".into());
    }
    Ok(a)
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

fn kinds(v: &str) -> Vec<String> {
    v.split(',')
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
        .collect()
}

/// Repository for the freeze-tag check: `--repo` when given, otherwise the checkout the
/// binary was built from.
fn freeze_repo(a: &Args) -> Option<PathBuf> {
    match &a.repo {
        Some(r) => Some(r.clone()),
        None => sealed::repo_root(std::path::Path::new(env!("CARGO_MANIFEST_DIR"))),
    }
}

fn run(a: &Args) -> Result<serde_json::Value, String> {
    if a.out.join("engine_eval.json").exists() {
        return Err(format!(
            "{} already has engine_eval.json; use a new run dir",
            a.out.display()
        ));
    }
    run_info::refuse_engine_env()?;
    std::fs::create_dir_all(&a.out).map_err(|e| e.to_string())?;
    let t_load = Instant::now();
    let mut model = Model::load(&a.weights, true).map_err(|e| e.to_string())?;
    if a.qualifying {
        run_info::check_qualifying(&model, a.precision, a.int8_stem)?;
    }
    // --int8-fp32-kinds keeps those layer kinds fp32 in the int8 engine (ablations; kinds:
    // stem_c2 stem_out ff1_w1 ff1_w2 qkv att_out pw1 pw2 ff2_w1 ff2_w2 pair restore enc_proj
    // joiner_out).
    let mut skip = a.int8_fp32_kinds.clone();
    if let Some(p) = &model.int8_prepared {
        // Prepared artifact: no weight quantisation here; its declared fp32 kinds rule.
        if !skip.is_empty() {
            return Err("--int8-fp32-kinds conflicts with a prepared-INT8 directory".into());
        }
        skip = p.fp32_kinds.clone();
    } else if a.precision == Precision::Int8 {
        model.quantize_int8_where(|kind, _| !skip.iter().any(|s| s == kind));
    }
    // NEW numerical candidate (off by default): post-training int8 for a still-fp32 stem.
    let stem_quantised = if a.int8_stem && a.precision == Precision::Int8 {
        let kinds: Vec<&str> = a.int8_stem_kinds.iter().map(String::as_str).collect();
        model.quantize_fp32_kinds_int8(&kinds)
    } else {
        0
    };
    if a.fold_prenorm {
        model.fold_prenorm();
    }
    let load_s = t_load.elapsed().as_secs_f64();
    let mut engine = Engine::new(Arc::new(model), a.isa, a.precision)
        .map_err(|e| e.to_string())?
        .with_max_chunks(a.max_chunks)
        .with_act7(a.act7)
        .with_gemm_threads(a.gemm_threads)
        .with_beam(a.beam)
        .with_beam_max_symbols(a.beam_max_symbols)
        .with_beam_margin(a.beam_margin)
        .with_kv_int8(a.kv_int8);
    if let Some(c) = a.chunk {
        engine = engine.with_chunk(c).map_err(|e| e.to_string())?;
    }
    if let Some(lm_path) = &a.lm_path {
        let lm = Arc::new(ultra_engine::NgramLm::load(lm_path).map_err(|e| e.to_string())?);
        let cfg = ultra_engine::LmConfig {
            lm_weight: a.lm_weight,
            lodr_weight: a.lodr_weight,
            blank_penalty: a.blank_penalty,
        };
        engine = engine.with_lm(lm, cfg).map_err(|e| e.to_string())?;
    }
    // Sealed PUBLIC-TEST guard (test-* splits), decided and logged before any audio is read.
    let panel = a.panel.as_ref().map(|p| Panel::read(p)).transpose()?;
    let splits: Vec<String> = match &panel {
        Some(p) => p.splits.iter().map(|s| s.split.clone()).collect(),
        None => vec![a
            .npz
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default()],
    };
    let repo = freeze_repo(a);
    let looks_read = match &repo {
        Some(repo) => {
            let looks = match &a.looks {
                Some(l) => l.clone(),
                None => repo.join("docs/ultra/LOOKS.jsonl"),
            };
            let candidate = format!(
                "{}+{}",
                engine.model.arch.checkpoint_sha256,
                engine
                    .model
                    .int8_prepared
                    .as_ref()
                    .map_or("fp32", |p| p.sha256.as_str())
            );
            sealed::guard(
                &splits,
                &sealed::SealedRead {
                    repo,
                    looks: &looks,
                    look_id: a.sealed_read.as_deref(),
                    candidate: &candidate,
                    build_commit: run_info::GIT_COMMIT,
                    build_dirty: run_info::GIT_DIRTY,
                },
            )?
        }
        None if splits.iter().any(|s| sealed::is_public_test(s)) => {
            return Err("PUBLIC-TEST split: no git repository to check the freeze tag".into())
        }
        None => Vec::new(),
    };
    let wl = match &panel {
        Some(p) => Workload::panel(p, &a.cache_dir)?,
        None => Workload::first(&a.npz, a.limit)?,
    };
    let utts = &wl.items;
    if utts.is_empty() {
        return Err("empty workload".into());
    }
    let packet = 16 * a.packet_ms;
    let mut queue: std::collections::VecDeque<usize> = (0..utts.len()).collect();
    let mut active: Vec<(usize, EngineStream, usize)> = Vec::new();
    let mut done: Vec<Option<(Vec<u32>, String, u64)>> = vec![None; utts.len()];
    let mut ws = Workspace::default();
    ws.set_profiling(a.profile);
    let (c0, w0) = (run_info::thread_cpu_s(), Instant::now());
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
    let cpu_s = run_info::thread_cpu_s() - c0;
    let wall_s = w0.elapsed().as_secs_f64();
    let audio_s: f64 = utts.iter().map(|u| u.pcm.len() as f64 / 16_000.0).sum();
    if !cpu_s.is_finite() || cpu_s <= 0.0 || audio_s <= 0.0 {
        return Err(format!("unusable CPU clock ({cpu_s}) or audio ({audio_s})"));
    }
    // References are read only after inference and timing have finished.
    let refs = wl.references()?;
    let ref_toks = a.reference.as_ref().map(reference_tokens).transpose()?;
    let (mut subs, mut dels, mut ins) = (0u64, 0u64, 0u64);
    let (mut errs, mut words, mut same_tok, mut forced) = (0u64, 0u64, 0usize, 0u64);
    let mut rows = Vec::new();
    let mut per_split: std::collections::BTreeMap<String, (u64, u64, usize)> = Default::default();
    for (i, d) in done.into_iter().enumerate() {
        let (toks, text, f) = d.ok_or("an utterance never finished")?;
        let sc = score(&refs[i].text, &text);
        if let Some([e, w, s, d, n]) = sc {
            errs += e;
            words += w;
            subs += s;
            dels += d;
            ins += n;
            let ps = per_split.entry(refs[i].split.clone()).or_default();
            *ps = (ps.0 + e, ps.1 + w, ps.2 + 1);
        }
        forced += f;
        let eq = ref_toks
            .as_ref()
            .map(|r| r.get(&refs[i].uid) == Some(&toks));
        if eq == Some(true) {
            same_tok += 1;
        }
        rows.push(
            json!({"index": i, "id": refs[i].uid, "split": refs[i].split,
                   "speaker": refs[i].speaker, "errors": sc.map(|v| v[0]),
                   "ref_words": sc.map(|v| v[1]), "hyp": text, "tokens": toks,
                   "equal_reference_tokens": eq}),
        );
    }
    let wer = (words > 0).then(|| 100.0 * errs as f64 / words as f64);
    let knobs = run_info::Knobs {
        mode: if a.qualifying {
            "qualifying"
        } else {
            "exploratory"
        },
        precision: a.precision,
        isa: a.isa.name(),
        max_chunks: a.max_chunks,
        act7: a.act7,
        gemm_threads: a.gemm_threads,
        int8_fp32_kinds: &skip,
        int8_stem_kinds: if a.int8_stem { &a.int8_stem_kinds } else { &[] },
        int8_stem_layers_quantised: stem_quantised,
        beam: engine.beam,
    };
    let mut summary = run_info::identity(&engine.model, &a.weights, &knobs);
    let split_wer: serde_json::Map<String, serde_json::Value> = per_split
        .into_iter()
        .map(|(k, (e, w, n))| {
            let wer = (w > 0).then(|| 100.0 * e as f64 / w as f64);
            (
                k,
                json!({"utterances": n, "errors": e, "ref_words": w, "wer_pct": wer}),
            )
        })
        .collect();
    let eff_cfg = engine.effective_decode_config();
    let extra = json!({
        "schema": "ultra.engine-eval/v2",
        "effective_decode_config": eff_cfg,
        "decode_policy_sha256": eff_cfg.policy_sha256(),
        "npz": wl.sources.iter().map(|s| &s.npz).collect::<Vec<_>>(),
        "panel": a.panel, "panel_manifest_sha256": wl.panel_manifest_sha256,
        "utterances": utts.len(), "audio_s": audio_s,
        "streams": a.streams, "packet_ms": a.packet_ms,
        "wer_pct": wer, "ref_words": words, "errors": errs, "per_split": split_wer,
        "forced_advances": forced,
        "reference": a.reference, "identical_reference_tokens": ref_toks.as_ref().map(|_| same_tok),
        "cpu_s": cpu_s, "wall_s": wall_s, "decode_wall_s": decode_wall,
        "decode_wall_s_per_audio_s": decode_wall / audio_s, "cpu_s_per_audio_s": cpu_s / audio_s,
        "cpu_scope": "decode thread only (schedstat), load excluded",
        "load_s": load_s, "decode_batches": batches,
        "mean_streams_per_batch": batch_rows as f64 / batches.max(1) as f64,
        "substitutions": subs, "deletions": dels, "insertions": ins,
        "peak_rss_mib": run_info::peak_rss_mib(),
        "lm": a.lm_path, "lm_weight": a.lm_weight, "lodr_weight": a.lodr_weight,
        "blank_penalty": a.blank_penalty,
        "sealed_reads": looks_read,
    });
    if let (Some(m), Some(x)) = (summary.as_object_mut(), extra.as_object()) {
        m.extend(x.clone());
    }
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
