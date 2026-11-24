//! `ultra-engine-prepare`: write a prepared-INT8 weight directory (immutable, hashed) from
//! an `ultra.r1-weights/v1` fp32 directory, so int8 inference loads packed codes and scales
//! instead of quantising weights at start-up.
//!
//! ```text
//! ultra-engine-prepare --weights <fp32 dir> --out <fresh dir> [--int8-fp32-kinds joiner_out,...]
//! ```

use std::path::PathBuf;
use std::process::ExitCode;

use ultra_engine::Model;

fn run() -> Result<String, String> {
    let mut it = std::env::args().skip(1);
    let (mut w, mut out, mut skip) = (None, None, Vec::<String>::new());
    while let Some(k) = it.next() {
        let mut v = || it.next().ok_or_else(|| format!("{k} needs a value"));
        match k.as_str() {
            "--weights" => w = Some(PathBuf::from(v()?)),
            "--out" => out = Some(PathBuf::from(v()?)),
            "--int8-fp32-kinds" => {
                skip = v()?
                    .split(',')
                    .filter(|s| !s.is_empty())
                    .map(str::to_owned)
                    .collect()
            }
            o => return Err(format!("unknown argument {o}")),
        }
    }
    let (w, out) = (w.ok_or("--weights required")?, out.ok_or("--out required")?);
    let mut m = Model::load(&w, true).map_err(|e| e.to_string())?;
    if let Some(prep) = &m.int8_prepared {
        if prep.self_contained {
            return Err("input is already a self-contained slim INT8 directory".into());
        }
        if !skip.is_empty() && skip != prep.fp32_kinds {
            return Err("--int8-fp32-kinds conflicts with input prepared-INT8 directory".into());
        }
        skip = prep.fp32_kinds.clone();
    } else {
        m.quantize_int8_where(|k, _| !skip.iter().any(|s| s == k));
    }
    let sha = m
        .write_prepared_int8(&out, &skip)
        .map_err(|e| e.to_string())?;
    // Read back through the strict loader before reporting success.
    let back = Model::load(&out, true).map_err(|e| format!("verification load: {e}"))?;
    if back.int8_prepared.as_ref().map(|p| &p.sha256) != Some(&sha) {
        return Err("verification load did not use the written artifact".into());
    }
    Ok(serde_json::json!({
        "out": out,
        "weights_int8_sha256": sha,
        "fp32_kinds": skip,
        "self_contained": true,
        "source_weights_sha256": back.arch.weights_sha256
    })
    .to_string())
}

fn main() -> ExitCode {
    match run() {
        Ok(s) => {
            println!("{s}");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("ultra-engine-prepare: {e}");
            ExitCode::FAILURE
        }
    }
}
