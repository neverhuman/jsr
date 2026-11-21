//! Admission for the existing engine bundle, including still-open upstream checks.

use std::collections::BTreeMap;
use std::path::{Component, Path};

use serde_json::Value;
use ultra_engine::{safetensors::SafeTensors, Model};

use crate::records::{hash_file, read_json};
use crate::{require, Result};

pub(crate) fn load(root: &Path, expected_manifest: &str) -> Result<Model> {
    let manifest = root.join("arch.json");
    require(
        hash_file(&manifest)? == expected_manifest,
        "fused manifest identity changed",
    )?;
    let arch: Value = read_json(&manifest)?;
    require(
        arch["schema"] == "ultra.r1-weights/v1",
        "unsupported fused bundle schema",
    )?;
    let checkpoint = arch["source"]["checkpoint_sha256"]
        .as_str()
        .ok_or("missing checkpoint identity")?;
    require(is_sha256(checkpoint), "invalid checkpoint identity")?;
    let files = arch["files"].as_object().ok_or("missing fused payloads")?;
    for name in ["weights", "tokens", "bpe", "mel_fbank"] {
        require(
            files.contains_key(name),
            format!("missing fused payload {name}"),
        )?;
    }
    let base = root.canonicalize()?;
    let mut resolved = BTreeMap::new();
    for (key, entry) in files {
        let name = entry["path"].as_str().ok_or("missing fused payload path")?;
        let expected = entry["sha256"]
            .as_str()
            .ok_or("missing fused payload hash")?;
        let mut components = Path::new(name).components();
        require(
            matches!(components.next(), Some(Component::Normal(_)))
                && components.next().is_none()
                && !name.contains('\\')
                && !name.starts_with('.')
                && is_sha256(expected),
            "invalid fused payload path/hash",
        )?;
        let path = base.join(name).canonicalize()?;
        require(
            path.parent() == Some(base.as_path()),
            "fused payload resolves outside bundle",
        )?;
        require(
            hash_file(&path)? == expected,
            format!("fused payload hash mismatch: {key}"),
        )?;
        resolved.insert(key.as_str(), path);
    }
    require(
        arch["source"]["tokenizer_sha256"] == files["bpe"]["sha256"],
        "fused tokenizer identity differs from BPE payload",
    )?;
    // Drop this validation copy before the model's packed allocations are created.
    {
        let st = SafeTensors::read(&resolved["weights"])?;
        require(
            st.metadata
                .get("checkpoint_sha256")
                .is_some_and(|v| v == checkpoint),
            "fused tensor checkpoint identity differs",
        )?;
        require(
            st.tensors
                .values()
                .all(|t| t.data.iter().all(|v| v.is_finite())),
            "nonfinite fused weight tensor",
        )?;
        for name in ["mean", "istd"] {
            let declared: Vec<f32> =
                serde_json::from_value(arch["features"]["cmvn"][name].clone())?;
            let stored = st.get(&format!("cmvn.{name}"), &[80])?;
            require(
                stored.data == declared,
                format!("fused CMVN {name} differs from weights"),
            )?;
        }
    }
    let model = Model::load(root, true)?;
    require(
        hash_file(&manifest)? == expected_manifest,
        "fused manifest changed during load",
    )?;
    Ok(model)
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
