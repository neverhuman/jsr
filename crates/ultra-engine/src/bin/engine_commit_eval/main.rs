//! `ultra-engine-commit`: H45 / H55 dual-pass endpoint commit evaluator CLI.

mod cli;

use std::fs::File;
use std::io::{BufWriter, Write};
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

use cli::{load_manifest, parse_args, Args};
use serde_json::{json, Value};
use ultra_engine::commit::{
    CommitConfig, CommitLmConfig, CommitPassSession, CommitTfRescoreConfig, DEFAULT_SAMPLE_RATE,
};
use ultra_engine::{Engine, Isa, Model, Precision};

fn pct(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        0.0
    } else {
        sorted[(((sorted.len() - 1) as f64 * p).round() as usize).min(sorted.len() - 1)]
    }
}

fn run(a: &Args) -> Result<Value, String> {
    let mut model = Model::load(&a.model, true).map_err(|e| e.to_string())?;
    if a.ctc_weight > 0.0 && model.ctc_head.is_none() {
        return Err("--ctc-weight > 0 requires ctc_head weights in weights.safetensors".into());
    }
    if model.int8_prepared.is_none() && a.precision == Precision::Int8 {
        model.quantize_int8();
    }
    let isa = Isa::detect();
    let eng = Engine::new(Arc::new(model), isa, a.precision)
        .map_err(|e| e.to_string())?
        .with_gemm_threads(a.threads);
    let beam = a
        .commit_beam
        .unwrap_or(if a.commit_lm.is_some() { 4 } else { 1 });
    let max_pcm_samples = (a.commit_max_audio_s * DEFAULT_SAMPLE_RATE as f64).round() as usize;
    let cfg = CommitConfig {
        commit_chunk: a.commit_chunk,
        tail_pad_samples: a.tail_pad_ms * DEFAULT_SAMPLE_RATE / 1000,
        max_pcm_samples,
        beam_size: beam,
        ctc_weight: a.ctc_weight,
        mbr_enabled: a.mbr,
        mbr_temp: a.mbr_temp,
    };
    let lm_cfg = a
        .commit_lm
        .as_ref()
        .map(|p| {
            CommitLmConfig::from_file(
                p,
                beam,
                a.lm_scale,
                a.lodr_scale,
                a.token_bonus,
                a.max_sym_per_frame,
            )
            .map_err(|e| e.to_string())
        })
        .transpose()?;
    let tf_cfg = CommitTfRescoreConfig::from_files(
        a.rescore_l2r.as_deref(),
        a.rescore_r2l.as_deref(),
        a.rescore_weight,
        a.rescore_m5_sub,
    )
    .map_err(|e| e.to_string())?;
    let mut sess = CommitPassSession::new(eng, cfg)
        .and_then(|s| s.with_commit_lm(lm_cfg))
        .and_then(|s| s.with_commit_tf_rescorer(tf_cfg))
        .map_err(|e| e.to_string())?;
    let items = load_manifest(a)?;
    if items.is_empty() {
        return Err("empty manifest".into());
    }
    if let Some(p) = a.out.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(p).map_err(|e| e.to_string())?;
    }
    let mut out = BufWriter::new(File::create(&a.out).map_err(|e| e.to_string())?);
    let (mut tot_s, mut tot_prov, mut tot_commit) = (0.0, 0.0, 0.0);
    let (mut revised, mut truncated, mut cms) = (0usize, 0usize, Vec::with_capacity(items.len()));
    for (uid, pcm) in &items {
        let t0 = Instant::now();
        for pkt in pcm.chunks(320) {
            sess.accept_i16(pkt).map_err(|e| e.to_string())?;
        }
        let acc_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let t1 = Instant::now();
        let r = sess.commit_endpoint().map_err(|e| e.to_string())?;
        let call_ms = t1.elapsed().as_secs_f64() * 1000.0;
        sess.reset_all();
        let prov_ms = acc_ms + (call_ms - r.commit_wall_ms).max(0.0);
        let audio_s = r.audio_ms / 1000.0;
        tot_s += audio_s;
        tot_prov += prov_ms;
        tot_commit += r.commit_wall_ms;
        revised += usize::from(r.revised);
        truncated += usize::from(r.pcm_truncated);
        cms.push(r.commit_wall_ms);
        writeln!(out, "{}", json!({"uid": uid, "provisional": r.provisional, "committed": r.committed, "revised": r.revised, "pcm_truncated": r.pcm_truncated, "audio_s": audio_s, "prov_wall_ms": prov_ms, "commit_wall_ms": r.commit_wall_ms}))
            .map_err(|e| e.to_string())?;
    }
    cms.sort_by(f64::total_cmp);
    let mut summary = json!({"summary": true, "utterances": items.len(), "revised_count": revised, "pcm_truncated_count": truncated, "total_audio_s": tot_s, "total_prov_ms": tot_prov, "total_commit_ms": tot_commit, "revised_pct": 100.0 * revised as f64 / items.len() as f64, "p50_commit_ms": pct(&cms, 0.50), "p95_commit_ms": pct(&cms, 0.95), "commit_chunk": a.commit_chunk, "tail_pad_ms": a.tail_pad_ms, "commit_max_audio_s": a.commit_max_audio_s, "threads": a.threads, "precision": format!("{:?}", a.precision), "isa": isa.name()});
    if let Some(map) = summary.as_object_mut() {
        if a.commit_lm.is_some() || a.commit_beam.is_some() {
            map.insert("commit_lm_path".into(), json!(a.commit_lm));
            map.insert("commit_beam".into(), json!(beam));
            map.insert("lm_scale".into(), json!(a.lm_scale));
            map.insert("lodr_scale".into(), json!(a.lodr_scale));
            map.insert("token_bonus".into(), json!(a.token_bonus));
        }
        if a.ctc_weight > 0.0 || a.mbr {
            map.insert("ctc_weight".into(), json!(a.ctc_weight));
            map.insert("mbr".into(), json!(a.mbr));
            map.insert("mbr_temp".into(), json!(a.mbr_temp));
        }
        if a.rescore_l2r.is_some() || a.rescore_r2l.is_some() {
            map.insert("rescore_l2r".into(), json!(a.rescore_l2r));
            map.insert("rescore_r2l".into(), json!(a.rescore_r2l));
            map.insert("rescore_weight".into(), json!(a.rescore_weight));
            map.insert("rescore_m5_sub".into(), json!(a.rescore_m5_sub));
        }
    }
    writeln!(out, "{summary}").map_err(|e| e.to_string())?;
    out.flush().map_err(|e| e.to_string())?;
    Ok(summary)
}

fn main() -> ExitCode {
    match parse_args(std::env::args().skip(1)).and_then(|a| run(&a)) {
        Ok(s) => {
            println!("{}", serde_json::to_string_pretty(&s).unwrap_or_default());
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("ultra-engine-commit: {e}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use ultra_engine::commit::{DEFAULT_RESCORE_M5_SUB, DEFAULT_RESCORE_WEIGHT};

    #[test]
    fn commit_eval_cli_args_and_percentiles() {
        let base = parse_args(
            "--model m --manifest in.jsonl --out o.jsonl"
                .split_whitespace()
                .map(String::from),
        )
        .expect("defaults");
        assert!((base.commit_max_audio_s - 40.0).abs() < 1e-9);
        assert_eq!(
            (base.ctc_weight, base.mbr, base.mbr_temp),
            (0.0, false, 1.0)
        );

        let s = "--model m --manifest in.jsonl --out o.jsonl --commit-chunk 16 --tail-pad-ms 320 --commit-max-audio-s 45.5 --threads 2 --commit-lm M5.ulm --commit-beam 32 --lm-scale 0.7 --lodr-scale 0.525 --token-bonus 0.1 --ctc-weight 0.2 --mbr --mbr-temp 1.5 --rescore-l2r l.utfr --rescore-r2l r.utfr --rescore-weight 0.35 --rescore-m5-sub 0.45";
        let a = parse_args(s.split_whitespace().map(String::from)).expect("parse");
        assert_eq!((a.commit_chunk, a.tail_pad_ms, a.threads), (16, 320, 2));
        assert!((a.commit_max_audio_s - 45.5).abs() < 1e-9);
        assert_eq!(a.commit_lm, Some(PathBuf::from("M5.ulm")));
        assert_eq!(
            (a.commit_beam, a.lm_scale, a.lodr_scale, a.token_bonus),
            (Some(32), 0.7, 0.525, 0.1)
        );
        assert_eq!((a.ctc_weight, a.mbr, a.mbr_temp), (0.2, true, 1.5));
        assert_eq!(
            (
                a.rescore_l2r,
                a.rescore_r2l,
                a.rescore_weight,
                a.rescore_m5_sub
            ),
            (
                Some(PathBuf::from("l.utfr")),
                Some(PathBuf::from("r.utfr")),
                0.35,
                0.45
            )
        );
        assert_eq!(
            (
                base.rescore_l2r,
                base.rescore_r2l,
                base.rescore_weight,
                base.rescore_m5_sub
            ),
            (None, None, DEFAULT_RESCORE_WEIGHT, DEFAULT_RESCORE_M5_SUB)
        );

        for bad in ["-0.1", "NaN", "inf"] {
            assert!(parse_args(
                format!("--model m --manifest in.jsonl --out o.jsonl --ctc-weight {bad}")
                    .split_whitespace()
                    .map(String::from)
            )
            .is_err());
        }
        for bad in ["0", "-1.0", "NaN", "inf"] {
            assert!(parse_args(
                format!("--model m --manifest in.jsonl --out o.jsonl --mbr-temp {bad}")
                    .split_whitespace()
                    .map(String::from)
            )
            .is_err());
        }
        let xs = [10.0, 20.0, 30.0, 40.0, 50.0];
        assert_eq!((pct(&xs, 0.50), pct(&xs, 0.95)), (30.0, 50.0));
    }
}
