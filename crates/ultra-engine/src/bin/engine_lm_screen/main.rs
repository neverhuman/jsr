//! `ultra-engine-lm-screen`: Wave-1B endpoint LM (`NgramLm`, `FfnnLm`, `M5+FFNN`),
//! Exact Stateless-Predictor ILME (`e_avg`), and Acoustic-Confidence Margin Gating evaluator.

mod io;

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

use serde_json::{json, Value};
use ultra_engine::commit::lm_gate::{
    decode_enc_gated, encode_pcm_frames, CommitTokenLm, InternalLm, LmGateConfig,
};
use ultra_engine::commit::DEFAULT_SAMPLE_RATE;
use ultra_engine::lm::{FfnnLm, NgramLm, StatelessIlme};
use ultra_engine::{Isa, Model, Precision};

struct Args {
    model: PathBuf,
    manifest: Option<PathBuf>,
    from_enc_cache: Option<PathBuf>,
    dump_enc_cache: Option<PathBuf>,
    dump_eavg: Option<PathBuf>,
    out: Option<PathBuf>,
    commit_chunk: usize,
    tail_pad_ms: usize,
    precision: Precision,
    limit: usize,
    commit_lm: Option<PathBuf>,
    ffnn_lm: Option<PathBuf>,
    interp_alpha: f32,
    commit_beam: usize,
    lm_scale: f32,
    lodr_scale: f32,
    ilme_eavg: Option<PathBuf>,
    ilme_zero: bool,
    ilme_scale: f32,
    token_bonus: f32,
    lm_conf_margin: f32,
    max_sym_per_frame: usize,
}

fn parse_args<I: IntoIterator<Item = String>>(args: I) -> Result<Args, String> {
    let mut it = args.into_iter();
    let mut a = Args {
        model: PathBuf::new(),
        manifest: None,
        from_enc_cache: None,
        dump_enc_cache: None,
        dump_eavg: None,
        out: None,
        commit_chunk: 16,
        tail_pad_ms: 320,
        precision: Precision::Int8,
        limit: 0,
        commit_lm: None,
        ffnn_lm: None,
        interp_alpha: 0.5,
        commit_beam: 16,
        lm_scale: 0.80,
        lodr_scale: 0.60,
        ilme_eavg: None,
        ilme_zero: false,
        ilme_scale: 0.0,
        token_bonus: 0.0,
        lm_conf_margin: 0.0,
        max_sym_per_frame: 1,
    };
    while let Some(k) = it.next() {
        let mut v = || it.next().ok_or_else(|| format!("{k} needs a value"));
        match k.as_str() {
            "--model" => a.model = v()?.into(),
            "--manifest" => a.manifest = Some(v()?.into()),
            "--from-enc-cache" => a.from_enc_cache = Some(v()?.into()),
            "--dump-enc-cache" => a.dump_enc_cache = Some(v()?.into()),
            "--dump-eavg" => a.dump_eavg = Some(v()?.into()),
            "--out" => a.out = Some(v()?.into()),
            "--commit-chunk" => a.commit_chunk = v()?.parse().map_err(|e| format!("{k}: {e}"))?,
            "--tail-pad-ms" => a.tail_pad_ms = v()?.parse().map_err(|e| format!("{k}: {e}"))?,
            "--precision" => a.precision = Precision::parse(&v()?).map_err(|e| e.to_string())?,
            "--limit" => a.limit = v()?.parse().map_err(|e| format!("{k}: {e}"))?,
            "--commit-lm" => a.commit_lm = Some(v()?.into()),
            "--ffnn-lm" => a.ffnn_lm = Some(v()?.into()),
            "--interp-alpha" => a.interp_alpha = v()?.parse().map_err(|e| format!("{k}: {e}"))?,
            "--commit-beam" => a.commit_beam = v()?.parse().map_err(|e| format!("{k}: {e}"))?,
            "--lm-scale" => a.lm_scale = v()?.parse().map_err(|e| format!("{k}: {e}"))?,
            "--lodr-scale" => a.lodr_scale = v()?.parse().map_err(|e| format!("{k}: {e}"))?,
            "--ilme-eavg" => a.ilme_eavg = Some(v()?.into()),
            "--ilme-zero" => a.ilme_zero = true,
            "--ilme-scale" => a.ilme_scale = v()?.parse().map_err(|e| format!("{k}: {e}"))?,
            "--token-bonus" => a.token_bonus = v()?.parse().map_err(|e| format!("{k}: {e}"))?,
            "--lm-conf-margin" => {
                a.lm_conf_margin = v()?.parse().map_err(|e| format!("{k}: {e}"))?
            }
            "--max-sym-per-frame" => {
                a.max_sym_per_frame = v()?.parse().map_err(|e| format!("{k}: {e}"))?
            }
            o => return Err(format!("unknown argument {o}")),
        }
    }
    if a.model.as_os_str().is_empty() {
        return Err("--model is required".into());
    }
    if a.manifest.is_none() && a.from_enc_cache.is_none() {
        return Err("--manifest or --from-enc-cache is required".into());
    }
    Ok(a)
}

fn run_decode(model: &Model, isa: Isa, a: &Args) -> Result<Value, String> {
    let out_path = a.out.as_ref().ok_or("--out is required for decode")?;
    let ng = a
        .commit_lm
        .as_ref()
        .map(|p| NgramLm::load(p).map(Arc::new))
        .transpose()
        .map_err(|e| e.to_string())?;
    let ff = a
        .ffnn_lm
        .as_ref()
        .map(|p| FfnnLm::load(p).map(Arc::new))
        .transpose()
        .map_err(|e| e.to_string())?;
    let token_lm = match (ng.clone(), ff) {
        (Some(n), Some(f)) => Some(CommitTokenLm::Interpolated {
            ngram: n,
            ffnn: f,
            alpha: a.interp_alpha,
        }),
        (Some(n), None) => Some(CommitTokenLm::Ngram(n)),
        (None, Some(f)) => Some(CommitTokenLm::Ffnn(f)),
        (None, None) => None,
    };
    let (internal_lm, ilm_scale) = if let Some(p) = &a.ilme_eavg {
        (
            InternalLm::StatelessIlme(Arc::new(
                StatelessIlme::from_eavg_file(model, p).map_err(|e| e.to_string())?,
            )),
            a.ilme_scale,
        )
    } else if a.ilme_zero {
        (
            InternalLm::StatelessIlme(Arc::new(
                StatelessIlme::from_zero_enc(model).map_err(|e| e.to_string())?,
            )),
            a.ilme_scale,
        )
    } else if let Some(n) = ng {
        if a.lodr_scale > 0.0 {
            (InternalLm::Lodr(n), a.lodr_scale)
        } else {
            (InternalLm::None, 0.0)
        }
    } else {
        (InternalLm::None, 0.0)
    };
    let gate = LmGateConfig {
        beam: a.commit_beam,
        lm_scale: a.lm_scale,
        ilm_scale,
        token_bonus: a.token_bonus,
        max_sym_per_frame: a.max_sym_per_frame,
        conf_margin: a.lm_conf_margin,
    };
    gate.validate().map_err(|e| e.to_string())?;
    if let Some(p) = out_path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(p).map_err(|e| e.to_string())?;
    }
    let mut out = BufWriter::new(File::create(out_path).map_err(|e| e.to_string())?);
    let (mut n_utts, mut tot_s, mut tot_ms) = (0usize, 0.0f64, 0.0f64);
    if let Some(cache) = &a.from_enc_cache {
        let mut r = BufReader::new(File::open(cache).map_err(|e| e.to_string())?);
        let (count, jn) = (
            io::read_u32(&mut r)? as usize,
            io::read_u32(&mut r)? as usize,
        );
        if jn != model.arch.joiner {
            return Err(format!(
                "cache joiner {jn} != model joiner {}",
                model.arch.joiner
            ));
        }
        let lim = if a.limit > 0 {
            a.limit.min(count)
        } else {
            count
        };
        for _ in 0..lim {
            let mut ub = vec![0u8; io::read_u32(&mut r)? as usize];
            r.read_exact(&mut ub).map_err(|e| e.to_string())?;
            let uid = String::from_utf8(ub).map_err(|e| e.to_string())?;
            let audio_s = f32::from_bits(io::read_u32(&mut r)?) as f64;
            let nf = io::read_u32(&mut r)? as usize;
            let mut raw = vec![0u8; nf * jn * 4];
            r.read_exact(&mut raw).map_err(|e| e.to_string())?;
            let enc: Vec<f32> = raw
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let t0 = Instant::now();
            let toks = decode_enc_gated(
                model,
                isa,
                a.precision,
                &enc,
                nf,
                token_lm.as_ref(),
                &internal_lm,
                &gate,
            )
            .map_err(|e| e.to_string())?;
            let ms = t0.elapsed().as_secs_f64() * 1000.0;
            let ids: Vec<i64> = toks.into_iter().map(i64::from).collect();
            writeln!(out, "{}", json!({"uid": uid, "committed": model.tokens.decode_ids(&ids), "audio_s": audio_s, "decode_wall_ms": ms})).map_err(|e| e.to_string())?;
            n_utts += 1;
            tot_s += audio_s;
            tot_ms += ms;
        }
    } else {
        let items = io::load_manifest(a.manifest.as_ref().ok_or("--manifest needed")?, a.limit)?;
        let pad = a.tail_pad_ms * DEFAULT_SAMPLE_RATE / 1000;
        for (uid, pcm) in &items {
            let t0 = Instant::now();
            let (enc, nf) = encode_pcm_frames(model, isa, a.precision, a.commit_chunk, pcm, pad)
                .map_err(|e| e.to_string())?;
            let toks = decode_enc_gated(
                model,
                isa,
                a.precision,
                &enc,
                nf,
                token_lm.as_ref(),
                &internal_lm,
                &gate,
            )
            .map_err(|e| e.to_string())?;
            let ms = t0.elapsed().as_secs_f64() * 1000.0;
            let ids: Vec<i64> = toks.into_iter().map(i64::from).collect();
            let audio_s = pcm.len() as f64 / DEFAULT_SAMPLE_RATE as f64;
            writeln!(out, "{}", json!({"uid": uid, "committed": model.tokens.decode_ids(&ids), "audio_s": audio_s, "decode_wall_ms": ms})).map_err(|e| e.to_string())?;
            n_utts += 1;
            tot_s += audio_s;
            tot_ms += ms;
        }
    }
    let summary = json!({"summary": true, "utterances": n_utts, "total_audio_s": tot_s, "total_decode_ms": tot_ms, "beam": a.commit_beam, "lm_scale": a.lm_scale, "ilm_scale": ilm_scale, "interp_alpha": a.interp_alpha, "lm_conf_margin": a.lm_conf_margin});
    writeln!(out, "{summary}").map_err(|e| e.to_string())?;
    out.flush().map_err(|e| e.to_string())?;
    Ok(summary)
}

fn main() -> ExitCode {
    let res = parse_args(std::env::args().skip(1)).and_then(|a| {
        let mut model = Model::load(&a.model, true).map_err(|e| e.to_string())?;
        if model.int8_prepared.is_none() && a.precision == Precision::Int8 {
            model.quantize_int8();
        }
        let isa = Isa::detect();
        if let Some(p) = &a.dump_eavg {
            io::run_dump_eavg(
                &model,
                isa,
                a.precision,
                a.commit_chunk,
                a.manifest.as_ref().ok_or("--manifest needed")?,
                a.limit,
                p,
            )
        } else if let Some(p) = &a.dump_enc_cache {
            io::dump_enc_cache(
                &model,
                isa,
                a.precision,
                a.commit_chunk,
                a.tail_pad_ms,
                a.manifest.as_ref().ok_or("--manifest needed")?,
                a.limit,
                p,
            )
        } else {
            run_decode(&model, isa, &a)
        }
    });
    match res {
        Ok(v) => {
            println!("{}", serde_json::to_string_pretty(&v).unwrap_or_default());
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("ultra-engine-lm-screen: {e}");
            ExitCode::FAILURE
        }
    }
}
