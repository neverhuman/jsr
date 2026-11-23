//! `ultra-lm-pack`: pack token-level KenLM `.arpa` (+ optional LODR bigram `.arpa`) into a
//! SHA-256 verified `ultra.lm-ngram/v2` pure-Rust bit-packed reverse-trie binary, or repack an
//! existing `v1`/`v2` `.ulm` binary directly with `--repack`.
//!
//! ```text
//! ultra-lm-pack --arpa model.arpa [--lodr lodr_2gram.arpa] --vocab 501 --out model.ulm [--fp32]
//! ultra-lm-pack --repack old_v1.ulm --out new_v2.ulm
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
    let mut repack: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    let mut vocab: usize = 501;
    let mut quantize = true;
    let mut include_eos = false;

    while let Some(k) = it.next() {
        let mut v = || it.next().ok_or_else(|| format!("{k} needs a value"));
        match k.as_str() {
            "--arpa" => arpa = Some(v()?.into()),
            "--lodr" => lodr = Some(v()?.into()),
            "--repack" => repack = Some(v()?.into()),
            "--out" => out = Some(v()?.into()),
            "--vocab" => vocab = v()?.parse().map_err(|e| format!("--vocab: {e}"))?,
            "--fp32" => quantize = false,
            "--q8" => quantize = true,
            "--include-eos" => include_eos = true,
            o => return Err(format!("unknown argument {o}")),
        }
    }
    let out = out.ok_or("--out is required")?;

    let t0 = Instant::now();
    let lm = match (repack, arpa) {
        (Some(src), None) => NgramLm::load(&src).map_err(|e| e.to_string())?,
        (None, Some(arpa)) => {
            NgramLm::from_arpa_files_with_eos(&arpa, lodr.as_deref(), vocab, quantize, include_eos)
                .map_err(|e| e.to_string())?
        }
        (Some(_), Some(_)) => return Err("pass either --arpa or --repack, not both".into()),
        (None, None) => return Err("--arpa or --repack is required".into()),
    };
    lm.save(&out).map_err(|e| e.to_string())?;
    let file_bytes = std::fs::metadata(&out).map(|m| m.len()).unwrap_or(0);
    println!(
        "packed ultra.lm-ngram/v2: order={} vocab={} quantized={} lodr={} eos={} max_logp={:.5} mem={:.2}MB file={:.2}MB elapsed={:.2}s -> {}",
        lm.order,
        lm.vocab,
        lm.quantized,
        lm.has_lodr(),
        lm.has_eos(),
        lm.max_lm_logp(),
        lm.memory_bytes() as f64 / 1e6,
        file_bytes as f64 / 1e6,
        t0.elapsed().as_secs_f64(),
        out.display()
    );
    Ok(())
}
