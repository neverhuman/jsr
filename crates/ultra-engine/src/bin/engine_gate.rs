//! `ultra-engine-gate`: the INT8 quality gate between two `ultra-engine-eval` runs of the
//! same checkpoint on the same DEV selection panel (A = FP32 control, B = INT8 candidate).
//!
//! ```text
//! ultra-engine-gate --fp32 <run>/engine_eval.json --int8 <run>/engine_eval.json --out gate.json
//!     [--max-delta-pp 0.10] [--max-upper-pp 0.50] [--replicates 10000] [--seed 20261121]
//! ```
//! Pass = point delta (pooled WER, B - A) <= `--max-delta-pp` AND the one-sided 95% upper
//! bound of the paired cluster bootstrap (`ultra_eval::bootstrap`, strata = split,
//! clusters = speaker) <= `--max-upper-pp`. Refuses runs on different panels, checkpoints,
//! utterance sets, or runs without a panel. `--cache-dir <dir>` scores runs written without
//! WER (built without the `wer` feature, e.g. on macOS) from `<dir>/<split>.npz` with the
//! same normaliser and aligner as `ultra-engine-eval`. Exit status: 0 PASS, 1 FAIL, 2 error.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::ExitCode;

use serde_json::{json, Value};
use ultra_eval::bootstrap::{paired_cluster_bootstrap, BootstrapConfig, PairedUnit};

/// Strata with fewer speakers than this are reported as limited.
const MIN_CLUSTERS: u64 = 20;

struct Args {
    fp32: PathBuf,
    int8: PathBuf,
    out: Option<PathBuf>,
    cache_dir: Option<PathBuf>,
    max_delta_pp: f64,
    max_upper_pp: f64,
    replicates: usize,
    seed: u64,
}

fn parse() -> Result<Args, String> {
    let mut a = Args {
        fp32: PathBuf::new(),
        int8: PathBuf::new(),
        out: None,
        cache_dir: None,
        max_delta_pp: 0.10,
        max_upper_pp: 0.50,
        replicates: 10_000,
        seed: 20_261_121,
    };
    let mut it = std::env::args().skip(1);
    while let Some(k) = it.next() {
        let mut v = || it.next().ok_or_else(|| format!("{k} needs a value"));
        let num = |s: String| s.parse::<f64>().map_err(|e| format!("{k}: {e}"));
        match k.as_str() {
            "--fp32" => a.fp32 = v()?.into(),
            "--int8" => a.int8 = v()?.into(),
            "--out" => a.out = Some(v()?.into()),
            "--cache-dir" => a.cache_dir = Some(v()?.into()),
            "--max-delta-pp" => a.max_delta_pp = num(v()?)?,
            "--max-upper-pp" => a.max_upper_pp = num(v()?)?,
            "--replicates" => a.replicates = v()?.parse().map_err(|e| format!("{k}: {e}"))?,
            "--seed" => a.seed = v()?.parse().map_err(|e| format!("{k}: {e}"))?,
            o => return Err(format!("unknown argument {o}")),
        }
    }
    if a.fp32.as_os_str().is_empty() || a.int8.as_os_str().is_empty() {
        return Err("--fp32 and --int8 are required".into());
    }
    Ok(a)
}

fn read(p: &PathBuf) -> Result<Value, String> {
    serde_json::from_slice(&std::fs::read(p).map_err(|e| format!("{}: {e}", p.display()))?)
        .map_err(|e| format!("{}: {e}", p.display()))
}

/// (split, speaker, errors, ref words) of one utterance.
type Row = (String, i64, u64, u64);

/// Rows per utterance id.
fn rows(v: &Value) -> Result<BTreeMap<String, Row>, String> {
    let mut out = BTreeMap::new();
    for r in v["rows"].as_array().ok_or("run has no rows")? {
        let id = r["id"].as_str().ok_or("row without id")?.to_owned();
        let split = r["split"].as_str().ok_or("row without split (old run?)")?;
        let spk = r["speaker"].as_i64().ok_or("row without speaker")?;
        let e = r["errors"]
            .as_u64()
            .ok_or("row without errors (unscored run?)")?;
        let w = r["ref_words"].as_u64().ok_or("row without ref_words")?;
        if out
            .insert(id.clone(), (split.to_owned(), spk, e, w))
            .is_some()
        {
            return Err(format!("duplicate row {id}"));
        }
    }
    Ok(out)
}

/// Fill `errors` / `ref_words` of rows written unscored (returns whether any row was).
fn rescore(v: &mut Value, cache: Option<&PathBuf>) -> Result<bool, String> {
    use ultra_eval::{normalise::normalise_words, wer::align};
    let rows = v["rows"].as_array_mut().ok_or("run has no rows")?;
    if rows.iter().all(|r| r["errors"].is_u64()) {
        return Ok(false);
    }
    let cache = cache.ok_or("unscored run: pass --cache-dir to score it here")?;
    let mut refs: BTreeMap<String, BTreeMap<String, String>> = BTreeMap::new();
    for r in rows.iter_mut() {
        let split = r["split"].as_str().ok_or("row without split")?.to_owned();
        if !refs.contains_key(&split) {
            let p = cache.join(format!("{split}.npz"));
            let m = ultra_frontend::npz::load_npz_references(&p, 0, None)
                .map_err(|e| format!("{}: {e}", p.display()))?
                .into_iter()
                .map(|u| (u.uid, u.text))
                .collect();
            refs.insert(split.clone(), m);
        }
        let id = r["id"].as_str().ok_or("row without id")?;
        let text = refs[&split]
            .get(id)
            .ok_or_else(|| format!("{id} not in {split}"))?;
        let al = align(
            &normalise_words(text),
            &normalise_words(r["hyp"].as_str().ok_or("row without hyp")?),
        );
        r["errors"] = json!(al.counts.errors());
        r["ref_words"] = json!(al.counts.ref_words);
    }
    Ok(true)
}

fn run(a: &Args) -> Result<Value, String> {
    let (mut fa, mut fb) = (read(&a.fp32)?, read(&a.int8)?);
    let rescored = [
        rescore(&mut fa, a.cache_dir.as_ref())?,
        rescore(&mut fb, a.cache_dir.as_ref())?,
    ];
    let panel = fa["panel_manifest_sha256"]
        .as_str()
        .ok_or("the FP32 run has no panel (run ultra-engine-eval --panel)")?;
    if fb["panel_manifest_sha256"].as_str() != Some(panel) {
        return Err("the runs use different panels".into());
    }
    if fa["arch_checkpoint_sha256"] != fb["arch_checkpoint_sha256"] {
        return Err("the runs come from different checkpoints".into());
    }
    if fa["precision"] != "fp32" || fb["precision"] != "int8" {
        return Err("expected --fp32 an fp32 run and --int8 an int8 run".into());
    }
    let (ra, rb) = (rows(&fa)?, rows(&fb)?);
    if ra.keys().ne(rb.keys()) {
        return Err("the runs cover different utterances".into());
    }
    let mut units = Vec::new();
    for (id, (split, spk, ea, w)) in &ra {
        let (sb, kb, eb, wb) = &rb[id];
        if sb != split || kb != spk || wb != w {
            return Err(format!("{id}: split/speaker/reference differ between runs"));
        }
        units.push(PairedUnit {
            dataset: split.clone(),
            cluster: spk.to_string(),
            num_a: *ea,
            num_b: *eb,
            den: *w,
        });
    }
    let cfg = BootstrapConfig {
        replicates: a.replicates,
        seed: a.seed,
        alpha: 0.05,
    };
    let b = paired_cluster_bootstrap(&units, None, &cfg).map_err(|e| e.to_string())?;
    let (delta_pp, upper_pp) = (100.0 * b.delta, 100.0 * b.upper);
    let limited = b.strata.iter().any(|s| s.clusters < MIN_CLUSTERS);
    let pass = delta_pp <= a.max_delta_pp && upper_pp <= a.max_upper_pp;
    Ok(json!({
        "schema": "ultra.engine-int8-gate/v1",
        "fp32_run": a.fp32, "int8_run": a.int8,
        "fp32_executable_sha256": fa["executable_sha256"], "int8_executable_sha256": fb["executable_sha256"],
        "int8_prepared_sha256": fb["int8_prepared_sha256"],
        "hosts": [fa["host"].clone(), fb["host"].clone()], "isas": [fa["isa"].clone(), fb["isa"].clone()],
        "rescored_here": rescored,
        "panel_manifest_sha256": panel, "checkpoint_sha256": fa["arch_checkpoint_sha256"],
        "utterances": units.len(),
        "fp32_wer_pct": 100.0 * b.observed_a, "int8_wer_pct": 100.0 * b.observed_b,
        "delta_pp": delta_pp, "upper95_pp": upper_pp, "lower95_pp": 100.0 * b.lower,
        "tail_at_least_max_delta": b.tail_at_least(a.max_delta_pp / 100.0),
        "rule": format!("delta <= {} pp AND one-sided 95% upper <= {} pp", a.max_delta_pp, a.max_upper_pp),
        "replicates": b.replicates, "seed": b.seed,
        "strata": b.strata, "limited": limited || b.limited,
        "pass": pass,
    }))
}

fn main() -> ExitCode {
    match parse().and_then(|a| {
        let v = run(&a)?;
        if let Some(o) = &a.out {
            if o.exists() {
                return Err(format!("{} exists; never overwrite", o.display()));
            }
            let text = serde_json::to_string_pretty(&v).map_err(|e| e.to_string())?;
            std::fs::write(o, text).map_err(|e| e.to_string())?;
        }
        Ok(v)
    }) {
        // Exit 0 only when the gate passes: 1 = FAIL (the JSON is still printed and written),
        // 2 = any error (no verdict).
        Ok(v) => {
            println!("{}", serde_json::to_string_pretty(&v).unwrap_or_default());
            if v["pass"] == true {
                ExitCode::SUCCESS
            } else {
                eprintln!("ultra-engine-gate: FAIL");
                ExitCode::from(1)
            }
        }
        Err(e) => {
            eprintln!("ultra-engine-gate: {e}");
            ExitCode::from(2)
        }
    }
}
