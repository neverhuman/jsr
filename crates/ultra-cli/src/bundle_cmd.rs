//! `ultra bundle pack` and `ultra bundle verify` CLI commands (`ultra.bundle/v1`).

use std::path::{Path, PathBuf};

use ultra_core::receipt::sha256_hex;
use ultra_model::{
    verify_bundle_dir, BundleEntry, BundleFrontend, BundleLimits, BundleManifest,
    HardwareInventory, BUNDLE_SCHEMA,
};

use crate::CliError;

fn pack_zipformer_bundle(
    source_dir: &Path,
    out_dir: &Path,
    precision: &str,
) -> Result<(), CliError> {
    let suffix = match precision {
        "int8" => ".int8.onnx",
        "fp32" => ".onnx",
        other => return Err(CliError::Config(format!("unsupported precision `{other}`"))),
    };
    let stem = "epoch-99-avg-1-chunk-16-left-128";
    let model_files = [
        format!("encoder-{stem}{suffix}"),
        format!("decoder-{stem}{suffix}"),
        format!("joiner-{stem}{suffix}"),
        String::from("tokens.txt"),
    ];
    let dst_model = out_dir.join("model");
    std::fs::create_dir_all(&dst_model)?;
    let mut weight_bytes = 0u64;
    let mut rel_paths = Vec::new();
    for name in &model_files {
        let src = source_dir.join(name);
        if !src.exists() {
            return Err(CliError::Unavailable(format!(
                "missing source file `{}`",
                src.display()
            )));
        }
        let dst = dst_model.join(name);
        std::fs::copy(&src, &dst)?;
        let len = std::fs::metadata(&dst)?.len();
        if name.ends_with(".onnx") {
            weight_bytes += len;
        }
        rel_paths.push(format!("model/{name}"));
    }

    let bundle_id = format!("r0-zipformer-{precision}");
    let card = format!(
        "# Model Card: `{bundle_id}`\n\n\
         - **Architecture**: Streaming Zipformer-Transducer (`chunk-16-left-128`, 320 ms chunk)\n\
         - **Precision**: `{precision}`\n\
         - **Training Corpus**: LibriSpeech 960h only (CC-BY-4.0); LibriSpeech dev/test held out (`in-corpus`, non-contaminated)\n\
         - **Licenses**: Code `Apache-2.0`, Weights `Apache-2.0` (`csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-26`)\n\
         - **Supported Domain**: English 16 kHz mono close-talk / read speech (`H0` x86_64 AVX2/FMA CPU)\n\
         - **Known Limitations**: Trained on read audiobooks; conversational far-field / telephony speech requires domain calibration.\n"
    );
    std::fs::write(out_dir.join("model_card.md"), card)?;
    rel_paths.push(String::from("model_card.md"));

    let provenance = serde_json::json!({
        "schema": "ultra.provenance/v1",
        "bundle_id": bundle_id,
        "upstream_repo": "csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-26",
        "upstream_revision": "4062f0f2dc9f9cbfcdcfbf2ad8fd0ce8b4cc89d7",
        "training_corpora": ["LibriSpeech-960h (CC-BY-4.0)"],
        "eval_overlap": "none (train-clean-100/360 + train-other-500 disjoint from dev/test)",
        "honesty_label": "in-corpus",
        "license_code": "Apache-2.0",
        "license_weights": "Apache-2.0"
    });
    std::fs::write(
        out_dir.join("provenance.json"),
        serde_json::to_vec_pretty(&provenance).map_err(|e| CliError::Internal(e.to_string()))?,
    )?;
    rel_paths.push(String::from("provenance.json"));
    rel_paths.sort();

    let mut entries = Vec::with_capacity(rel_paths.len());
    for rel in rel_paths {
        let bytes = std::fs::read(out_dir.join(&rel))?;
        entries.push(BundleEntry {
            path: rel,
            bytes: bytes.len() as u64,
            sha256: sha256_hex(&bytes),
        });
    }

    let manifest = BundleManifest {
        schema: BUNDLE_SCHEMA.into(),
        bundle_id: bundle_id.clone(),
        candidate_id: format!("sherpa-onnx-streaming-zipformer-en-2023-06-26-{precision}"),
        precision: precision.into(),
        required_isa: vec!["avx2".into(), "fma".into()],
        packed_weight_bytes: weight_bytes,
        expanded_weight_bytes: weight_bytes,
        frontend: BundleFrontend {
            sample_rate_hz: 16_000,
            feature_dim: 80,
            kind: "logmel80".into(),
        },
        limits: BundleLimits {
            max_concurrent_streams: 128,
            max_service_rss_bytes: 1_500_000_000,
            supported_chunk_ms: vec![320],
        },
        normalizer_id: "ultra.en-norm.v1".into(),
        license_code: "Apache-2.0".into(),
        license_weights: "Apache-2.0".into(),
        provenance_summary: "LibriSpeech 960h only (CC-BY-4.0); Apache-2.0 weights".into(),
        entries,
    };
    std::fs::write(
        out_dir.join("manifest.json"),
        serde_json::to_vec_pretty(&manifest).map_err(|e| CliError::Internal(e.to_string()))?,
    )?;
    let verified = verify_bundle_dir(out_dir, None)?;
    println!(
        "{}",
        serde_json::to_string_pretty(&verified).map_err(|e| CliError::Internal(e.to_string()))?
    );
    Ok(())
}

pub(crate) fn cmd_bundle(args: &[String]) -> Result<(), CliError> {
    let Some((action, rest)) = args.split_first() else {
        return Err(CliError::Config(
            "expected `ultra bundle <verify|pack> ...`".into(),
        ));
    };
    if action == "pack" {
        let data = crate::jsr_data_root();
        let mut src = PathBuf::from(&data).join("models/zipformer-en-2023-06-26");
        let mut out: Option<PathBuf> = None;
        let mut prec = String::from("int8");
        let mut i = 0;
        while i < rest.len() {
            match rest[i].as_str() {
                "--source-dir" => {
                    i += 1;
                    src = PathBuf::from(
                        rest.get(i)
                            .ok_or_else(|| CliError::Config("--source-dir needs value".into()))?,
                    );
                }
                "--out" | "--bundle" => {
                    i += 1;
                    out =
                        Some(PathBuf::from(rest.get(i).ok_or_else(|| {
                            CliError::Config("--out needs value".into())
                        })?));
                }
                "--precision" => {
                    i += 1;
                    prec = rest
                        .get(i)
                        .cloned()
                        .ok_or_else(|| CliError::Config("--precision needs value".into()))?;
                }
                x => return Err(CliError::Config(format!("unknown option `{x}`"))),
            }
            i += 1;
        }
        let out_dir = out.ok_or_else(|| CliError::Config("missing --out".into()))?;
        return pack_zipformer_bundle(&src, &out_dir, &prec);
    }
    if action != "verify" {
        return Err(CliError::Config(format!(
            "unknown bundle action `{action}`"
        )));
    }
    let mut bundle_dir: Option<PathBuf> = None;
    let mut hw_path: Option<PathBuf> = None;
    let mut i = 0;
    while i < rest.len() {
        match rest[i].as_str() {
            "--bundle" => {
                i += 1;
                bundle_dir =
                    Some(PathBuf::from(rest.get(i).ok_or_else(|| {
                        CliError::Config("--bundle needs value".into())
                    })?));
            }
            "--hardware" => {
                i += 1;
                hw_path =
                    Some(PathBuf::from(rest.get(i).ok_or_else(|| {
                        CliError::Config("--hardware needs value".into())
                    })?));
            }
            x => return Err(CliError::Config(format!("unknown option `{x}`"))),
        }
        i += 1;
    }
    let bdir = bundle_dir.ok_or_else(|| CliError::Config("missing --bundle".into()))?;
    let hw_inv = match hw_path {
        Some(hp) => Some(HardwareInventory::load(&hp)?),
        None => None,
    };
    let hw_rec = hw_inv.as_ref().map(|inv| inv.machine(None)).transpose()?;
    let verified = verify_bundle_dir(&bdir, hw_rec)?;
    println!(
        "{}",
        serde_json::to_string_pretty(&verified).map_err(|e| CliError::Internal(e.to_string()))?
    );
    Ok(())
}
