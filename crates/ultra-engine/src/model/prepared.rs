//! Prepared INT8 weights: `weights.int8.safetensors` holds every quantised linear layer as
//! packed codes (`<name>.q`, I8, the kernel's panel layout), per-output-channel scales
//! (`<name>.scale`) and dequantised weight sums for the centring compensation
//! (`<name>.wsum`). `arch.json` declares it under `int8` (layout, source fp32 weights hash,
//! kinds kept fp32) and lists it in `files.weights_int8`. Loading it performs no weight
//! quantisation; only the per-row activation quantisation stays in timed inference.

use std::path::Path;

use ultra_core::receipt::sha256_hex;

use super::arch_json::RawInt8;
use super::Model;
use crate::quant::PackedI8;
use crate::safetensors::{LazyTensors, SafeTensors, Tensor, TensorI8};
use crate::EngineError;

/// Packing identity written into and required from every prepared artifact.
pub const INT8_LAYOUT: &str =
    "ultra-engine/packed-i8 v1: panels NR=16, K groups of 4, per-channel symmetric weights, wsum";

/// What a prepared load used.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedInt8 {
    pub sha256: String,
    pub fp32_kinds: Vec<String>,
}

fn name(kind: &str, blk: Option<usize>) -> String {
    blk.map_or_else(|| kind.to_owned(), |b| format!("blocks.{b}.{kind}"))
}

impl Model {
    pub(super) fn load_prepared_int8(
        &mut self,
        path: &Path,
        raw: &RawInt8,
    ) -> Result<(), EngineError> {
        if raw.layout != INT8_LAYOUT {
            return Err(EngineError::Unsupported(format!(
                "int8 layout {}",
                raw.layout
            )));
        }
        if raw.source_weights_sha256 != self.arch.weights_sha256 {
            return Err(EngineError::Integrity(
                "weights.int8.safetensors was prepared from other fp32 weights".into(),
            ));
        }
        let sha = super::arch_json::sha256_file(path)?;
        let st = LazyTensors::open(path)?;
        if st.metadata.get("source_weights_sha256") != Some(&self.arch.weights_sha256)
            || st.metadata.get("layout").map(String::as_str) != Some(INT8_LAYOUT)
        {
            return Err(EngineError::Integrity(
                "int8 artifact metadata mismatch".into(),
            ));
        }
        let fp32_kinds = raw.fp32_kinds.clone();
        let mut used = 0usize;
        for (kind, blk, lin) in self.linears_mut() {
            let nm = name(kind, blk);
            if fp32_kinds.iter().any(|k| k == kind) {
                if st.contains(&format!("{nm}.q")) {
                    return Err(EngineError::Format(format!(
                        "{nm} declared fp32 but has codes"
                    )));
                }
                lin.q = None;
                continue;
            }
            let (n, k) = (lin.n(), lin.k());
            let panels = n.div_ceil(crate::kernels::NR);
            let kp = k.div_ceil(crate::quant::KG) * crate::quant::KG;
            let q = st.i8(&format!("{nm}.q"), &[panels * kp * crate::kernels::NR])?;
            let scale = st.f32(&format!("{nm}.scale"), &[panels * crate::kernels::NR])?;
            let wsum = st.f32(&format!("{nm}.wsum"), &[panels * crate::kernels::NR])?;
            lin.q = Some(PackedI8::from_parts(
                n,
                k,
                q,
                &scale,
                &wsum,
                &lin.f.bias[..n],
            )?);
            used += 3;
        }
        if used != st.len() {
            return Err(EngineError::Format(
                "int8 artifact has undeclared tensors".into(),
            ));
        }
        self.int8_prepared = Some(PreparedInt8 {
            sha256: sha,
            fp32_kinds,
        });
        Ok(())
    }

    /// Write a prepared-INT8 directory to the fresh (absent or empty) directory `out`: the
    /// fp32 directory's files copied, `weights.int8.safetensors` from this model's current
    /// int8 packing (`quantize_int8_where`), and `arch.json` extended with `int8` and
    /// `files.weights_int8`. Returns the artifact's SHA-256.
    pub fn write_prepared_int8(
        &self,
        out: &Path,
        fp32_kinds: &[String],
    ) -> Result<String, EngineError> {
        if out.exists() && std::fs::read_dir(out)?.next().is_some() {
            return Err(EngineError::Format(format!(
                "{} is not empty",
                out.display()
            )));
        }
        let mut st = SafeTensors::default();
        st.metadata.insert("layout".into(), INT8_LAYOUT.into());
        st.metadata.insert(
            "source_weights_sha256".into(),
            self.arch.weights_sha256.clone(),
        );
        st.metadata.insert(
            "checkpoint_sha256".into(),
            self.arch.checkpoint_sha256.clone(),
        );
        let mut me = self.linears_snapshot();
        for (nm, kind, q) in me.drain(..) {
            match q {
                Some(p) if !fp32_kinds.iter().any(|k| k == &kind) => {
                    let (d, s, w) = p.parts();
                    st.i8_tensors.insert(
                        format!("{nm}.q"),
                        TensorI8 {
                            shape: vec![d.len()],
                            data: d.to_vec(),
                        },
                    );
                    st.tensors.insert(
                        format!("{nm}.scale"),
                        Tensor {
                            shape: vec![s.len()],
                            data: s.to_vec(),
                        },
                    );
                    st.tensors.insert(
                        format!("{nm}.wsum"),
                        Tensor {
                            shape: vec![w.len()],
                            data: w.to_vec(),
                        },
                    );
                }
                None if fp32_kinds.iter().any(|k| k == &kind) => {}
                _ => {
                    return Err(EngineError::Unsupported(format!(
                        "{nm}: int8 packing does not match the declared fp32 kinds"
                    )))
                }
            }
        }
        let bytes = st.to_bytes()?;
        let sha = sha256_hex(&bytes);
        std::fs::create_dir_all(out)?;
        let text = std::fs::read_to_string(self.root.join("arch.json"))?;
        let mut arch: serde_json::Value =
            serde_json::from_str(&text).map_err(|e| EngineError::Format(e.to_string()))?;
        let files = arch["files"].as_object().cloned().unwrap_or_default();
        for f in files.values() {
            let p = f["path"]
                .as_str()
                .ok_or_else(|| EngineError::Format("files.path".into()))?;
            std::fs::copy(self.root.join(p), out.join(p))?;
        }
        std::fs::write(out.join("weights.int8.safetensors"), &bytes)?;
        arch["files"]["weights_int8"] =
            serde_json::json!({"path": "weights.int8.safetensors", "sha256": sha});
        arch["int8"] = serde_json::json!({
            "layout": INT8_LAYOUT,
            "source_weights_sha256": self.arch.weights_sha256,
            "fp32_kinds": fp32_kinds,
            "weights": "per-output-channel symmetric int8 (|q| <= 127), scale = max|w| / 127",
            "activations": "per-row centred dynamic int8 at inference time (not stored)",
            "producer": "ultra-engine-prepare",
        });
        let pretty =
            serde_json::to_string_pretty(&arch).map_err(|e| EngineError::Format(e.to_string()))?;
        std::fs::write(out.join("arch.json"), pretty)?;
        Ok(sha)
    }

    /// `(name, kind, int8 packing)` of every quantisable layer (read-only view).
    fn linears_snapshot(&self) -> Vec<(String, String, Option<PackedI8>)> {
        let mut v = vec![
            (
                "stem_c2".to_owned(),
                "stem_c2".to_owned(),
                self.c2.q.clone(),
            ),
            (
                "stem_out".to_owned(),
                "stem_out".to_owned(),
                self.stem_out.q.clone(),
            ),
        ];
        for (i, b) in self.blocks.iter().enumerate() {
            for (kind, l) in [
                ("ff1_w1", &b.ff1_w1),
                ("ff1_w2", &b.ff1_w2),
                ("qkv", &b.qkv),
                ("att_out", &b.att_out),
                ("pw1", &b.pw1),
                ("pw2", &b.pw2),
                ("ff2_w1", &b.ff2_w1),
                ("ff2_w2", &b.ff2_w2),
            ] {
                v.push((name(kind, Some(i)), kind.to_owned(), l.q.clone()));
            }
        }
        v.push((
            "enc_proj".into(),
            "enc_proj".into(),
            self.enc_proj.q.clone(),
        ));
        v.push((
            "joiner_out".into(),
            "joiner_out".into(),
            self.joiner_out.q.clone(),
        ));
        v
    }
}
