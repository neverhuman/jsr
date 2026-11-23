//! Fast N-best JSONL rescorer benchmark & parity driver for `ultra.tf-rescore/v1` (D4).

use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;

use serde_json::{json, Value};
use ultra_engine::{Isa, TfRescorer, TfWorkspace};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let (mut nbest, mut l2r_p, mut r2l_p, mut out_p) = (None, None, None, None);
    let (mut limit, mut act7) = (0usize, false);
    while let Some(k) = args.next() {
        let v = args.next().ok_or_else(|| format!("{k} needs a value"))?;
        match k.as_str() {
            "--nbest" => nbest = Some(PathBuf::from(v)),
            "--l2r" => l2r_p = Some(PathBuf::from(v)),
            "--r2l" => r2l_p = Some(PathBuf::from(v)),
            "--out" => out_p = Some(PathBuf::from(v)),
            "--limit" => limit = v.parse()?,
            "--act7" => act7 = v != "0" && v != "false",
            o => return Err(format!("unknown arg {o}").into()),
        }
    }
    let nbest = nbest.ok_or("--nbest required")?;
    let out_p = out_p.ok_or("--out required")?;
    let l2r = l2r_p.as_deref().map(TfRescorer::load).transpose()?;
    let r2l = r2l_p.as_deref().map(TfRescorer::load).transpose()?;
    let isa = Isa::detect();
    let mut ws = TfWorkspace::default();
    ws.set_act7(act7);
    let mut out = BufWriter::new(File::create(&out_p)?);
    let (mut n_utts, mut n_cands) = (0usize, 0usize);
    let (mut l2r_ns, mut r2l_ns) = (0u128, 0u128);

    for line in BufReader::new(File::open(&nbest)?).lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let rec: Value = serde_json::from_str(&line)?;
        let uid = rec["uid"].as_str().ok_or("missing uid")?;
        let cands = rec["cands"].as_array().ok_or("missing cands")?;
        let mut tok_vecs: Vec<Vec<u32>> = Vec::with_capacity(cands.len());
        for c in cands {
            let arr = c["tokens"].as_array().ok_or("missing tokens")?;
            tok_vecs.push(arr.iter().map(|x| x.as_u64().unwrap_or(0) as u32).collect());
        }
        let slices: Vec<&[u32]> = tok_vecs.iter().map(Vec::as_slice).collect();
        let l2r_sc = match &l2r {
            Some(m) => {
                let t0 = Instant::now();
                let s = m.score_batch(isa, &slices, &mut ws)?;
                l2r_ns += t0.elapsed().as_nanos();
                Some(s)
            }
            None => None,
        };
        let r2l_sc = match &r2l {
            Some(m) => {
                let t0 = Instant::now();
                let s = m.score_batch(isa, &slices, &mut ws)?;
                r2l_ns += t0.elapsed().as_nanos();
                Some(s)
            }
            None => None,
        };
        let l2r_tot: Vec<f32> = l2r_sc
            .as_ref()
            .map(|v| v.iter().map(|s| s.total_logp).collect())
            .unwrap_or_default();
        let l2r_tok: Vec<f32> = l2r_sc
            .as_ref()
            .map(|v| v.iter().map(|s| s.tok_logp).collect())
            .unwrap_or_default();
        let r2l_tot: Vec<f32> = r2l_sc
            .as_ref()
            .map(|v| v.iter().map(|s| s.total_logp).collect())
            .unwrap_or_default();
        let r2l_tok: Vec<f32> = r2l_sc
            .as_ref()
            .map(|v| v.iter().map(|s| s.tok_logp).collect())
            .unwrap_or_default();
        writeln!(
            out,
            "{}",
            json!({"uid": uid, "l2r_tot": l2r_tot, "l2r_tok": l2r_tok, "r2l_tot": r2l_tot, "r2l_tok": r2l_tok})
        )?;
        n_utts += 1;
        n_cands += cands.len();
        if limit > 0 && n_utts >= limit {
            break;
        }
    }
    out.flush()?;
    let l2r_ms = (l2r_ns as f64 / 1e6) / (n_utts.max(1) as f64);
    let r2l_ms = (r2l_ns as f64 / 1e6) / (n_utts.max(1) as f64);
    println!(
        "{}",
        json!({
            "isa": isa.name(),
            "utts": n_utts,
            "cands": n_cands,
            "l2r_ms_per_utt": l2r_ms,
            "r2l_ms_per_utt": r2l_ms,
            "total_ms_per_utt": l2r_ms + r2l_ms,
        })
    );
    Ok(())
}
