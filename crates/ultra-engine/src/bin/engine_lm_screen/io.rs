//! Manifest, encoder-cache, and `e_avg` I/O helpers for `ultra-engine-lm-screen`.

use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use serde_json::{json, Value};
use ultra_engine::commit::lm_gate::encode_pcm_frames;
use ultra_engine::commit::DEFAULT_SAMPLE_RATE;
use ultra_engine::decoder::{greedy_chunk, DecState, DecWorkspace};
use ultra_engine::lm::StatelessIlme;
use ultra_engine::{Isa, Model, Precision};
use ultra_frontend::npz::{load_npz_audio_only, load_npz_references};
use ultra_frontend::read_audio_16k_mono;

pub fn load_manifest(path: &Path, limit: usize) -> Result<Vec<(String, Vec<i16>)>, String> {
    let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("");
    let lim = (limit > 0).then_some(limit);
    if ext.eq_ignore_ascii_case("npz") {
        let au = load_npz_audio_only(path, 0, lim).map_err(|e| e.to_string())?;
        let rf = load_npz_references(path, 0, lim).map_err(|e| e.to_string())?;
        return Ok(au
            .into_iter()
            .zip(rf)
            .map(|(x, r)| (r.uid, x.pcm))
            .collect());
    }
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let base = path.parent().unwrap_or_else(|| Path::new("."));
    let mut out = Vec::new();
    for (idx, line) in text.lines().map(str::trim).enumerate() {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if lim.is_some_and(|n| out.len() >= n) {
            break;
        }
        let (uid, ap) = if line.starts_with('{') {
            let v: Value =
                serde_json::from_str(line).map_err(|e| format!("line {}: {e}", idx + 1))?;
            let uid = v["uid"]
                .as_str()
                .or_else(|| v["id"].as_str())
                .map(str::to_owned);
            let ap = v["wav"]
                .as_str()
                .or_else(|| v["path"].as_str())
                .ok_or("missing wav/path")?;
            (uid, ap.to_owned())
        } else if let Some((u, p)) = line.split_once('\t') {
            (Some(u.trim().to_owned()), p.trim().to_owned())
        } else {
            (None, line.to_owned())
        };
        let p = if Path::new(&ap).is_absolute() {
            PathBuf::from(&ap)
        } else {
            base.join(&ap)
        };
        let pcm = read_audio_16k_mono(&p).map_err(|e| format!("{}: {e}", p.display()))?;
        let id = uid.unwrap_or_else(|| format!("utt_{:05}", out.len()));
        out.push((id, pcm));
    }
    Ok(out)
}

pub fn read_u32<R: Read>(r: &mut R) -> Result<u32, String> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b).map_err(|e| e.to_string())?;
    Ok(u32::from_le_bytes(b))
}

#[allow(clippy::too_many_arguments)]
pub fn run_dump_eavg(
    model: &Model,
    isa: Isa,
    prec: Precision,
    commit_chunk: usize,
    manifest: &Path,
    limit: usize,
    out_path: &Path,
) -> Result<Value, String> {
    let items = load_manifest(manifest, limit)?;
    let jn = model.arch.joiner;
    let mut sum = vec![0.0f64; jn];
    let mut emissions = 0u64;
    let mut ws = DecWorkspace::default();
    let mut ckpts: Vec<(usize, Vec<f32>)> = Vec::new();
    for (idx, (_, pcm)) in items.iter().enumerate() {
        let (enc, nf) =
            encode_pcm_frames(model, isa, prec, commit_chunk, pcm, 0).map_err(|e| e.to_string())?;
        let mut st = DecState::new(model);
        for t in 0..nf {
            let prev = st.tokens.len();
            let frame = &enc[t * jn..(t + 1) * jn];
            greedy_chunk(model, isa, prec, &mut [&mut st], &[1], frame, &mut ws)
                .map_err(|e| e.to_string())?;
            let emitted = st.tokens.len() - prev;
            if emitted > 0 {
                let w = emitted as f64;
                for (s, &v) in sum.iter_mut().zip(frame) {
                    *s += w * f64::from(v);
                }
                emissions += emitted as u64;
            }
        }
        let n_done = idx + 1;
        if (n_done == 1000 || n_done == 2500 || n_done == items.len()) && emissions > 0 {
            ckpts.push((
                n_done,
                sum.iter().map(|&x| (x / emissions as f64) as f32).collect(),
            ));
        }
    }
    if emissions == 0 {
        return Err("zero non-blank emissions accumulated".into());
    }
    let e_avg: Vec<f32> = sum.iter().map(|&x| (x / emissions as f64) as f32).collect();
    let art = StatelessIlme::save_eavg_json(
        out_path,
        &e_avg,
        items.len(),
        emissions,
        &model.arch.weights_sha256,
    )
    .map_err(|e| e.to_string())?;
    let conv: Vec<Value> = ckpts
        .iter()
        .map(|(n, v)| {
            let (mut dot, mut na, mut nb, mut d2) = (0.0f64, 0.0, 0.0, 0.0);
            for (&a, &b) in v.iter().zip(&e_avg) {
                let (af, bf) = (f64::from(a), f64::from(b));
                dot += af * bf;
                na += af * af;
                nb += bf * bf;
                d2 += (af - bf) * (af - bf);
            }
            json!({"utts": n, "cos_vs_final": dot / (na.sqrt() * nb.sqrt()), "l2_vs_final": d2.sqrt()})
        })
        .collect();
    Ok(
        json!({"mode": "dump_eavg", "out": out_path, "utts": items.len(), "emissions": emissions, "e_avg_sha256": art.e_avg_sha256, "convergence": conv}),
    )
}

#[allow(clippy::too_many_arguments)]
pub fn dump_enc_cache(
    model: &Model,
    isa: Isa,
    prec: Precision,
    commit_chunk: usize,
    tail_pad_ms: usize,
    manifest: &Path,
    limit: usize,
    out_path: &Path,
) -> Result<Value, String> {
    let items = load_manifest(manifest, limit)?;
    if let Some(p) = out_path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(p).map_err(|e| e.to_string())?;
    }
    let mut w = BufWriter::new(File::create(out_path).map_err(|e| e.to_string())?);
    let pad = tail_pad_ms * DEFAULT_SAMPLE_RATE / 1000;
    w.write_all(&(items.len() as u32).to_le_bytes())
        .map_err(|e| e.to_string())?;
    w.write_all(&(model.arch.joiner as u32).to_le_bytes())
        .map_err(|e| e.to_string())?;
    for (uid, pcm) in &items {
        let (enc, nf) = encode_pcm_frames(model, isa, prec, commit_chunk, pcm, pad)
            .map_err(|e| e.to_string())?;
        let audio_s = pcm.len() as f32 / DEFAULT_SAMPLE_RATE as f32;
        let ub = uid.as_bytes();
        w.write_all(&(ub.len() as u32).to_le_bytes())
            .map_err(|e| e.to_string())?;
        w.write_all(ub).map_err(|e| e.to_string())?;
        w.write_all(&audio_s.to_le_bytes())
            .map_err(|e| e.to_string())?;
        w.write_all(&(nf as u32).to_le_bytes())
            .map_err(|e| e.to_string())?;
        for &v in &enc {
            w.write_all(&v.to_le_bytes()).map_err(|e| e.to_string())?;
        }
    }
    w.flush().map_err(|e| e.to_string())?;
    Ok(
        json!({"mode": "dump_enc_cache", "out": out_path, "utterances": items.len(), "commit_chunk": commit_chunk, "tail_pad_ms": tail_pad_ms}),
    )
}
