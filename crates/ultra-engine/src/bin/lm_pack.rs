//! `ultra-lm-pack`: pack token-level KenLM `.arpa` (+ optional LODR bigram `.arpa`) into a
//! SHA-256 verified `ultra.lm-ngram/v1` pure-Rust reverse-trie binary.
//!
//! ```text
//! ultra-lm-pack --arpa model.arpa [--lodr lodr_2gram.arpa] --vocab 501 --out model.ulm [--fp32]
//! ```

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use ultra_engine::NgramLm;

fn main() -> ExitCode {
    if let Err(e) = run() {
        eprintln!("ultra-lm-pack: {e}");
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}

fn run() -> Result<(), String> {
    let mut it = std::env::args().skip(1);
    let mut arpa: Option<PathBuf> = None;
    let mut lodr: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    let mut vocab: usize = 501;
    let mut quantize = true;

    while let Some(k) = it.next() {
        let mut v = || it.next().ok_or_else(|| format!("{k} needs a value"));
        match k.as_str() {
            "--arpa" => arpa = Some(v()?.into()),
            "--lodr" => lodr = Some(v()?.into()),
            "--out" => out = Some(v()?.into()),
            "--vocab" => vocab = v()?.parse().map_err(|e| format!("--vocab: {e}"))?,
            "--fp32" => quantize = false,
            "--q8" => quantize = true,
            o => return Err(format!("unknown argument {o}")),
        }
    }
    let arpa = arpa.ok_or("--arpa is required")?;
    let out = out.ok_or("--out is required")?;

    let t0 = Instant::now();
    let lm = NgramLm::from_arpa_files(&arpa, lodr.as_deref(), vocab, quantize)
        .map_err(|e| e.to_string())?;
    lm.save(&out).map_err(|e| e.to_string())?;
    let file_bytes = std::fs::metadata(&out).map(|m| m.len()).unwrap_or(0);
    println!(
        "packed ultra.lm-ngram/v1: order={} vocab={} quantized={} lodr={} max_logp={:.5} mem={:.2}MB file={:.2}MB elapsed={:.2}s -> {}",
        lm.order,
        lm.vocab,
        lm.quantized,
        lm.has_lodr(),
        lm.max_lm_logp(),
        lm.memory_bytes() as f64 / 1e6,
        file_bytes as f64 / 1e6,
        t0.elapsed().as_secs_f64(),
        out.display()
    );
    Ok(())
}
