//! `costfit`: fit `ps per weight per pass = a + c F` per (kernel, size, threads) from a
//! `wstream` CSV, print the fit table as CSV, the W01 verdict, and optionally write
//! `ultra.costlaw/v1` records (spec Appendix D) as a JSON array.
//!
//! Usage: `costfit <wstream.csv> [--json out.json] [--machine H0] [--cpu "..."]
//!         [--protocol "..."]`

use serde_json::json;
use std::collections::BTreeMap;
use std::process::ExitCode;
use ultra_bench::fit::{fit_line, residency, w01_verdict, Series};

struct Opts {
    csv: String,
    json: Option<String>,
    machine: String,
    cpu: String,
    protocol: String,
}

fn parse() -> Result<Opts, String> {
    let mut it = std::env::args().skip(1);
    let mut o = Opts {
        csv: it
            .next()
            .ok_or("usage: costfit <wstream.csv> [--json path]")?,
        json: None,
        machine: "H0".into(),
        cpu: "unknown".into(),
        protocol: "best-of-3".into(),
    };
    while let Some(k) = it.next() {
        let v = it.next().ok_or_else(|| format!("{k} needs a value"))?;
        match k.as_str() {
            "--json" => o.json = Some(v),
            "--machine" => o.machine = v,
            "--cpu" => o.cpu = v,
            "--protocol" => o.protocol = v,
            _ => return Err(format!("unknown argument {k}")),
        }
    }
    Ok(o)
}

type Key = (String, u64, usize, String); // kernel, weights (for ordering), threads, size
/// Per series: weight-image bytes and (F, ps per weight) points.
type Groups = BTreeMap<Key, (u64, Vec<(f64, f64)>)>;

/// Group CSV rows into (F, ps per weight) series keyed by kernel/size/threads.
fn load(text: &str) -> Result<Groups, String> {
    let mut lines = text
        .lines()
        .filter(|l| !l.starts_with('#') && !l.is_empty());
    let header: Vec<&str> = lines.next().ok_or("empty csv")?.split(',').collect();
    let col = |name: &str| {
        header
            .iter()
            .position(|h| *h == name)
            .ok_or(format!("missing column {name}"))
    };
    let (ck, cw, cf, ct, cb, cm) = (
        col("kernel")?,
        col("weights")?,
        col("frames")?,
        col("threads")?,
        col("weight_bytes")?,
        col("ms_per_pass")?,
    );
    let mut groups = Groups::new();
    for l in lines {
        let v: Vec<&str> = l.split(',').collect();
        let bad = || format!("bad row {l:?}");
        let size = v.get(cw).ok_or_else(bad)?.to_string();
        let weights = ultra_bench::parse_size(&size).ok_or_else(bad)? as u64;
        let num = |i: usize| v.get(i).and_then(|s| s.parse::<f64>().ok()).ok_or_else(bad);
        let ps_per_weight = num(cm)? * 1e9 / weights as f64;
        let key = (
            v.get(ck).ok_or_else(bad)?.to_string(),
            weights,
            num(ct)? as usize,
            size,
        );
        let entry = groups.entry(key).or_insert((num(cb)? as u64, Vec::new()));
        entry.1.push((num(cf)?, ps_per_weight));
    }
    Ok(groups)
}

fn run(o: &Opts) -> Result<(), String> {
    let text = std::fs::read_to_string(&o.csv).map_err(|e| format!("{}: {e}", o.csv))?;
    let mut series = Vec::new();
    println!("kernel,size,threads,weight_bytes,residency,a_ps_per_weight,c_ps_per_mac,asymptote_GMAC_s,r2,max_rel_err,points");
    for ((kernel, _, threads, size), (bytes, pts)) in load(&text)? {
        let Some(fit) = fit_line(&pts) else { continue };
        println!(
            "{kernel},{size},{threads},{bytes},{},{:.2},{:.2},{:.1},{:.4},{:.3},{}",
            residency(bytes),
            fit.a,
            fit.c,
            1e3 / fit.c,
            fit.r2,
            fit.max_rel_err,
            pts.len()
        );
        series.push(Series {
            kernel,
            size,
            threads,
            weight_bytes: bytes,
            fit,
        });
    }
    let (verdict, lines) = w01_verdict(&series);
    eprintln!("W01 verdict: {verdict}");
    for l in &lines {
        eprintln!("  {l}");
    }
    if let Some(path) = &o.json {
        let records: Vec<_> = series
            .iter()
            .map(|s| {
                json!({
                    "schema": "ultra.costlaw/v1",
                    "machine": o.machine,
                    "cpu": o.cpu,
                    "kernel": s.kernel,
                    "threads": s.threads,
                    "residency": residency(s.weight_bytes),
                    "a_ps_per_weight": (s.fit.a * 100.0).round() / 100.0,
                    "c_ps_per_mac": (s.fit.c * 100.0).round() / 100.0,
                    "r2": (s.fit.r2 * 1e4).round() / 1e4,
                    "protocol": o.protocol,
                    "size": s.size,
                    "weight_bytes": s.weight_bytes,
                    "max_rel_err": (s.fit.max_rel_err * 1e3).round() / 1e3,
                    "label": "[H0-MEASURED]",
                })
            })
            .collect();
        let doc = serde_json::to_string_pretty(&records).map_err(|e| e.to_string())?;
        std::fs::write(path, doc + "\n").map_err(|e| format!("{path}: {e}"))?;
    }
    Ok(())
}

fn main() -> ExitCode {
    match parse().and_then(|o| run(&o)) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("costfit: {e}");
            ExitCode::from(1)
        }
    }
}
