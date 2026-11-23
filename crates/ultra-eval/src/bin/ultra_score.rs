//! `ultra-score A.jsonl [B.jsonl] [--weights ds=w,...] [--replicates N] [--alpha A]
//!              [--seed S] [--md-out PATH] [--alignment-ledger PATH]
//!              [--receipt-a PATH] [--receipt-b PATH]`
//!
//! Each JSONL line is `{id, speaker, dataset, ref, hyp[, duration_s]}`. Prints the JSON
//! report (`ultra.score-report/v1`) on stdout and a short markdown table on stderr
//! (or to `--md-out`). With B, B is paired with A by id and a paired speaker-cluster
//! bootstrap of B - A is reported (macro, pooled and per dataset). Default weights are
//! equal over datasets with reference words; default replicates are
//! `max(10000, ceil(200/alpha))`; default alpha 0.05 (one-sided, unadjusted: pass the
//! family-adjusted alpha yourself).

use std::fs::{self, File};
use std::io::{BufRead, BufWriter, Write};

use serde::Deserialize;
use ultra_core::hash::sha256_reader;
use ultra_core::receipt::sha256_hex;
use ultra_eval::alignment_ledger::{
    write_paired_transitions, write_system, LedgerRecords, RunIdentity,
};
use ultra_eval::bootstrap::{recommended_replicates, BootstrapConfig};
use ultra_eval::corpus::MacroWeights;
use ultra_eval::normalise::normalise_words;
use ultra_eval::score::{markdown, parse_jsonl, scorable_datasets, score_report};
use ultra_eval::wer::align;
use ultra_eval::EvalError;

struct Args {
    files: Vec<String>,
    weights: Option<String>,
    replicates: Option<usize>,
    alpha: f64,
    seed: u64,
    md_out: Option<String>,
    ledger: Option<String>,
    receipts: [Option<String>; 2],
}

fn parse_args() -> Result<Args, String> {
    let mut a = Args {
        files: Vec::new(),
        weights: None,
        replicates: None,
        alpha: 0.05,
        seed: 20_260_101,
        md_out: None,
        ledger: None,
        receipts: [None, None],
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
            "--alignment-ledger" => a.ledger = Some(val("--alignment-ledger")?),
            "--receipt-a" => a.receipts[0] = Some(val("--receipt-a")?),
            "--receipt-b" => a.receipts[1] = Some(val("--receipt-b")?),
            f if !f.starts_with("--") => a.files.push(f.to_owned()),
            other => return Err(format!("unknown option {other}")),
        }
    }
    if a.files.is_empty() || a.files.len() > 2 {
        return Err("give one or two JSONL files".into());
    }
    if a.receipts.iter().any(Option::is_some) && a.ledger.is_none() {
        return Err("alignment receipts require --alignment-ledger".into());
    }
    if a.receipts[1].is_some() && a.files.len() < 2 {
        return Err("--receipt-b requires a system B score input".into());
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

fn write_ledger(path: &str, args: &Args, inputs: &[String]) -> Result<(), EvalError> {
    let records = inputs
        .iter()
        .map(|input| LedgerRecords::parse(input))
        .collect::<Result<Vec<_>, _>>()?;
    let mut receipts = Vec::new();
    for path in &args.receipts {
        let receipt = if let Some(path) = path {
            let raw = fs::read_to_string(path)?;
            let receipt: RunIdentity =
                serde_json::from_str(&raw).map_err(|e| EvalError::Input(format!("{path}: {e}")))?;
            receipt.validate()?;
            Some(receipt)
        } else {
            None
        };
        receipts.push(receipt);
    }
    if let [Some(a), Some(b)] = receipts.as_slice() {
        if a.audio_manifest_sha256 != b.audio_manifest_sha256 {
            return Err(EvalError::Input(
                "paired audio manifest hashes differ".into(),
            ));
        }
    }
    let scorer = sha256_reader(&mut File::open(std::env::current_exe()?)?)?;
    let mut out =
        BufWriter::new(File::create_new(path).map_err(|e| EvalError::Io(format!("{path}: {e}")))?);
    for (index, system_records) in records.iter().enumerate() {
        write_system(
            &mut out,
            system_records,
            ["A", "B"][index],
            receipts[index].as_ref(),
            &sha256_hex(inputs[index].as_bytes()),
            &scorer,
        )?;
    }
    if let [a, b] = records.as_slice() {
        write_paired_transitions(&mut out, a, b)?;
    }
    out.flush()?;
    Ok(())
}

fn run(a: &Args) -> Result<(), EvalError> {
    let inputs = a
        .files
        .iter()
        .map(|p| fs::read_to_string(p).map_err(|e| EvalError::Io(format!("{p}: {e}"))))
        .collect::<Result<Vec<_>, _>>()?;
    let ra = parse_jsonl(&inputs[0])?;
    let rb = inputs.get(1).map(|p| parse_jsonl(p)).transpose()?;
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
    if let Some(path) = &a.ledger {
        if a.md_out
            .as_ref()
            .is_some_and(|md| std::path::absolute(md).ok() == std::path::absolute(path).ok())
        {
            return Err(EvalError::Input(
                "ledger and markdown output paths coincide".into(),
            ));
        }
        write_ledger(path, a, &inputs)?;
    }
    let json = serde_json::to_string_pretty(&report).map_err(|e| EvalError::Io(e.to_string()))?;
    println!("{json}");
    let md = markdown(&report);
    match &a.md_out {
        Some(p) => fs::write(p, md)?,
        None => eprint!("{md}"),
    }
    Ok(())
}

/// One request batches candidates for the same reference. The training process
/// can keep one scorer open and receive an ordered response after each line.
#[derive(Deserialize)]
struct CandidateRequest {
    #[serde(rename = "ref")]
    reference: String,
    hyps: Vec<String>,
}

fn candidate_edits(input: impl BufRead, mut output: impl Write) -> Result<(), EvalError> {
    for (index, line) in input.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let request: CandidateRequest = serde_json::from_str(&line)
            .map_err(|e| EvalError::Input(format!("candidate request line {}: {e}", index + 1)))?;
        let reference = normalise_words(&request.reference);
        let counts: Vec<_> = request
            .hyps
            .iter()
            .map(|hyp| align(&reference, &normalise_words(hyp)).counts)
            .collect();
        serde_json::to_writer(&mut output, &counts).map_err(|e| EvalError::Io(e.to_string()))?;
        output.write_all(b"\n")?;
        output.flush()?;
    }
    Ok(())
}

fn main() {
    if std::env::args().nth(1).as_deref() == Some("--candidate-edits") {
        if std::env::args().len() != 2 {
            eprintln!("usage: ultra-score --candidate-edits");
            std::process::exit(2);
        }
        if let Err(e) = candidate_edits(std::io::stdin().lock(), std::io::stdout().lock()) {
            eprintln!("ultra-score: {e}");
            std::process::exit(1);
        }
        return;
    }
    let args = parse_args().unwrap_or_else(|e| {
        eprintln!("ultra-score: {e}");
        eprintln!(
            "usage: ultra-score A.jsonl [B.jsonl] [--weights ds=w,...] [--replicates N] \
             [--alpha A] [--seed S] [--md-out PATH] [--alignment-ledger PATH] \
             [--receipt-a PATH] [--receipt-b PATH]\n       ultra-score --candidate-edits"
        );
        std::process::exit(2);
    });
    if let Err(e) = run(&args) {
        eprintln!("ultra-score: {e}");
        std::process::exit(1);
    }
}
