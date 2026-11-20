//! `ultra-score A.jsonl [B.jsonl] [--weights ds=w,...] [--replicates N] [--alpha A]
//!              [--seed S] [--md-out PATH]`
//!
//! Each JSONL line is `{id, speaker, dataset, ref, hyp[, duration_s]}`. Prints the JSON
//! report (`ultra.score-report/v1`) on stdout and a short markdown table on stderr
//! (or to `--md-out`). With B, B is paired with A by id and a paired speaker-cluster
//! bootstrap of B - A is reported (macro, pooled and per dataset). Default weights are
//! equal over datasets with reference words; default replicates are
//! `max(10000, ceil(200/alpha))`; default alpha 0.05 (one-sided, unadjusted: pass the
//! family-adjusted alpha yourself).

use ultra_eval::bootstrap::{recommended_replicates, BootstrapConfig};
use ultra_eval::corpus::MacroWeights;
use ultra_eval::score::{markdown, parse_jsonl, scorable_datasets, score_report};
use ultra_eval::EvalError;

struct Args {
    files: Vec<String>,
    weights: Option<String>,
    replicates: Option<usize>,
    alpha: f64,
    seed: u64,
    md_out: Option<String>,
}

fn parse_args() -> Result<Args, String> {
    let mut a = Args {
        files: Vec::new(),
        weights: None,
        replicates: None,
        alpha: 0.05,
        seed: 20_260_101,
        md_out: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(x) = it.next() {
        let mut val = |name: &str| it.next().ok_or(format!("{name} needs a value"));
        match x.as_str() {
            "--weights" => a.weights = Some(val("--weights")?),
            "--replicates" => {
                a.replicates = Some(val("--replicates")?.parse().map_err(|e| format!("{e}"))?)
            }
            "--alpha" => a.alpha = val("--alpha")?.parse().map_err(|e| format!("{e}"))?,
            "--seed" => a.seed = val("--seed")?.parse().map_err(|e| format!("{e}"))?,
            "--md-out" => a.md_out = Some(val("--md-out")?),
            f if !f.starts_with("--") => a.files.push(f.to_owned()),
            other => return Err(format!("unknown option {other}")),
        }
    }
    if a.files.is_empty() || a.files.len() > 2 {
        return Err("give one or two JSONL files".into());
    }
    Ok(a)
}

fn parse_weights(spec: &str) -> Result<MacroWeights, EvalError> {
    let mut map = std::collections::BTreeMap::new();
    for part in spec.split(',').filter(|p| !p.is_empty()) {
        let (k, v) = part
            .split_once('=')
            .ok_or_else(|| EvalError::Weights(format!("bad entry {part}")))?;
        let w: f64 = v
            .parse()
            .map_err(|_| EvalError::Weights(format!("bad weight {v}")))?;
        map.insert(k.to_owned(), w);
    }
    MacroWeights::new(map)
}

fn run(a: &Args) -> Result<(), EvalError> {
    let read = |p: &str| -> Result<_, EvalError> {
        let text = std::fs::read_to_string(p).map_err(|e| EvalError::Io(format!("{p}: {e}")))?;
        parse_jsonl(&text)
    };
    let ra = read(&a.files[0])?;
    let rb = a.files.get(1).map(|p| read(p)).transpose()?;
    let weights = match &a.weights {
        Some(s) => parse_weights(s)?,
        None => MacroWeights::equal(scorable_datasets(&ra))?,
    };
    let cfg = BootstrapConfig {
        replicates: a
            .replicates
            .unwrap_or_else(|| recommended_replicates(a.alpha)),
        seed: a.seed,
        alpha: a.alpha,
    };
    let report = score_report(&ra, rb.as_deref(), &weights, &cfg)?;
    let json = serde_json::to_string_pretty(&report).map_err(|e| EvalError::Io(e.to_string()))?;
    println!("{json}");
    let md = markdown(&report);
    match &a.md_out {
        Some(p) => std::fs::write(p, md)?,
        None => eprint!("{md}"),
    }
    Ok(())
}

fn main() {
    let args = parse_args().unwrap_or_else(|e| {
        eprintln!("ultra-score: {e}");
        eprintln!(
            "usage: ultra-score A.jsonl [B.jsonl] [--weights ds=w,...] [--replicates N] \
             [--alpha A] [--seed S] [--md-out PATH]"
        );
        std::process::exit(2);
    });
    if let Err(e) = run(&args) {
        eprintln!("ultra-score: {e}");
        std::process::exit(1);
    }
}
