//! Minimal, validating reader and writer for little-endian `F32` / `I8` safetensors files.
//!
//! Layout (safetensors spec): an 8-byte little-endian `u64` header length `n`, `n` bytes of
//! UTF-8 JSON mapping tensor names to `{dtype, shape, data_offsets: [begin, end]}` (offsets
//! relative to the end of the header) plus an optional `__metadata__` string map, then the raw
//! tensor bytes. Every entry is checked before any bytes are interpreted: dtype `F32`, the
//! shape's element count times 4 equals `end - begin`, offsets in bounds, no two tensors
//! overlapping. Values are decoded with `f32::from_le_bytes` into owned, aligned `Vec<f32>`
//! storage, so no pointer casts or alignment assumptions are involved.

use std::collections::BTreeMap;
use std::path::Path;

use serde::Deserialize;

use crate::EngineError;

/// One decoded tensor.
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

impl Tensor {
    pub fn numel(&self) -> usize {
        self.data.len()
    }
}

/// One decoded `I8` tensor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorI8 {
    pub shape: Vec<usize>,
    pub data: Vec<i8>,
}

/// All tensors of a file plus its `__metadata__` map.
#[derive(Debug, Clone, Default)]
pub struct SafeTensors {
    pub tensors: BTreeMap<String, Tensor>,
    pub i8_tensors: BTreeMap<String, TensorI8>,
    pub metadata: BTreeMap<String, String>,
}

#[derive(Deserialize)]
struct Entry {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
}

/// Upper bound on the JSON header (the spec's own limit is 100 MB).
const MAX_HEADER: usize = 100 << 20;

impl SafeTensors {
    pub fn read(path: &Path) -> Result<Self, EngineError> {
        let bytes = std::fs::read(path)?;
        Self::parse(&bytes)
    }

    pub fn parse(bytes: &[u8]) -> Result<Self, EngineError> {
        let fmt = |m: String| EngineError::Format(format!("safetensors: {m}"));
        let len_bytes: [u8; 8] = bytes
            .get(..8)
            .and_then(|s| s.try_into().ok())
            .ok_or_else(|| fmt("file shorter than the 8-byte header length".into()))?;
        let n = usize::try_from(u64::from_le_bytes(len_bytes))
            .map_err(|_| fmt("header length does not fit usize".into()))?;
        if n > MAX_HEADER || 8 + n > bytes.len() {
            return Err(fmt(format!("header length {n} out of range")));
        }
        let header: BTreeMap<String, serde_json::Value> = serde_json::from_slice(&bytes[8..8 + n])
            .map_err(|e| fmt(format!("header JSON: {e}")))?;
        let data = &bytes[8 + n..];
        let mut out = SafeTensors::default();
        let mut spans: Vec<(usize, usize, String)> = Vec::new();
        for (name, v) in header {
            if name == "__metadata__" {
                let m: BTreeMap<String, String> = serde_json::from_value(v)
                    .map_err(|e| fmt(format!("__metadata__ is not a string map: {e}")))?;
                out.metadata = m;
                continue;
            }
            let e: Entry =
                serde_json::from_value(v).map_err(|e| fmt(format!("entry {name}: {e}")))?;
            let width = match e.dtype.as_str() {
                "F32" => 4usize,
                "I8" => 1,
                d => return Err(fmt(format!("{name}: dtype {d} (only F32 and I8 are read)"))),
            };
            let [b, end] = e.data_offsets;
            let numel = e
                .shape
                .iter()
                .try_fold(1usize, |a, &d| a.checked_mul(d))
                .ok_or_else(|| fmt(format!("{name}: shape overflows")))?;
            let want = numel
                .checked_mul(width)
                .ok_or_else(|| fmt(format!("{name}: byte count overflows")))?;
            if b > end || end > data.len() || end - b != want {
                return Err(fmt(format!(
                    "{name}: offsets [{b}, {end}) do not hold {numel} {} within {} data bytes",
                    e.dtype,
                    data.len()
                )));
            }
            spans.push((b, end, name.clone()));
            if width == 1 {
                let data = data[b..end].iter().map(|&v| v as i8).collect();
                out.i8_tensors.insert(
                    name,
                    TensorI8 {
                        shape: e.shape,
                        data,
                    },
                );
                continue;
            }
            let tensor = Tensor {
                shape: e.shape,
                data: data[b..end]
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect(),
            };
            out.tensors.insert(name, tensor);
        }
        spans.sort();
        for w in spans.windows(2) {
            if w[1].0 < w[0].1 {
                return Err(fmt(format!("tensors {} and {} overlap", w[0].2, w[1].2)));
            }
        }
        Ok(out)
    }

    /// The tensor `name` with exactly `shape`.
    pub fn get(&self, name: &str, shape: &[usize]) -> Result<&Tensor, EngineError> {
        let t = self
            .tensors
            .get(name)
            .ok_or_else(|| EngineError::Format(format!("missing tensor {name}")))?;
        if t.shape != shape {
            return Err(EngineError::Shape(format!(
                "{name}: shape {:?}, expected {shape:?}",
                t.shape
            )));
        }
        Ok(t)
    }

    /// The `I8` tensor `name` with exactly `shape`.
    pub fn get_i8(&self, name: &str, shape: &[usize]) -> Result<&TensorI8, EngineError> {
        let t = self
            .i8_tensors
            .get(name)
            .ok_or_else(|| EngineError::Format(format!("missing I8 tensor {name}")))?;
        if t.shape != shape {
            return Err(EngineError::Shape(format!(
                "{name}: shape {:?}, expected {shape:?}",
                t.shape
            )));
        }
        Ok(t)
    }

    /// Serialise (`F32` and `I8` tensors in sorted name order, header padded to 8 bytes).
    pub fn to_bytes(&self) -> Result<Vec<u8>, EngineError> {
        let mut header = serde_json::Map::new();
        header.insert(
            "__metadata__".into(),
            serde_json::to_value(&self.metadata).map_err(|e| EngineError::Format(e.to_string()))?,
        );
        let mut names: Vec<(&String, bool)> = self.tensors.keys().map(|k| (k, false)).collect();
        names.extend(self.i8_tensors.keys().map(|k| (k, true)));
        names.sort();
        let mut body: Vec<u8> = Vec::new();
        for (name, is_i8) in names {
            let (dtype, start) = (if is_i8 { "I8" } else { "F32" }, body.len());
            let shape = if is_i8 {
                let t = &self.i8_tensors[name];
                body.extend(t.data.iter().map(|&v| v as u8));
                t.shape.clone()
            } else {
                let t = &self.tensors[name];
                for v in &t.data {
                    body.extend_from_slice(&v.to_le_bytes());
                }
                t.shape.clone()
            };
            header.insert(
                name.clone(),
                serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [start, body.len()]}),
            );
        }
        let mut h = serde_json::to_vec(&header).map_err(|e| EngineError::Format(e.to_string()))?;
        while !h.len().is_multiple_of(8) {
            h.push(b' ');
        }
        let mut out = (h.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(&h);
        out.extend_from_slice(&body);
        Ok(out)
    }

    /// Owned copy of `name` checked against `shape`.
    pub fn take(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>, EngineError> {
        Ok(self.get(name, shape)?.data.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build(header: &str, data: &[u8]) -> Vec<u8> {
        let mut h = header.as_bytes().to_vec();
        while !h.len().is_multiple_of(8) {
            h.push(b' ');
        }
        let mut out = (h.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(&h);
        out.extend_from_slice(data);
        out
    }

    #[test]
    fn reads_f32_tensors_and_metadata() {
        let data: Vec<u8> = [1.0f32, -2.5, 3.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let f = build(
            r#"{"__metadata__":{"k":"v"},"a":{"dtype":"F32","shape":[1,2],"data_offsets":[0,8]},"b":{"dtype":"F32","shape":[1],"data_offsets":[8,12]}}"#,
            &data,
        );
        let st = SafeTensors::parse(&f).expect("parse");
        assert_eq!(st.metadata["k"], "v");
        assert_eq!(st.get("a", &[1, 2]).expect("a").data, vec![1.0, -2.5]);
        assert_eq!(st.take("b", &[1]).expect("b"), vec![3.0]);
        assert!(st.get("a", &[2]).is_err());
        assert!(st.get("zz", &[1]).is_err());
    }

    #[test]
    fn i8_round_trip() {
        let mut st = SafeTensors::default();
        st.metadata.insert("m".into(), "1".into());
        st.tensors.insert(
            "f".into(),
            Tensor {
                shape: vec![2],
                data: vec![1.5, -2.0],
            },
        );
        st.i8_tensors.insert(
            "q".into(),
            TensorI8 {
                shape: vec![3],
                data: vec![-127, 0, 127],
            },
        );
        let back = SafeTensors::parse(&st.to_bytes().expect("bytes")).expect("parse");
        assert_eq!(back.tensors, st.tensors);
        assert_eq!(back.i8_tensors, st.i8_tensors);
        assert_eq!(back.metadata, st.metadata);
        assert!(back.get_i8("q", &[3]).is_ok() && back.get_i8("q", &[2]).is_err());
    }

    #[test]
    fn refuses_bad_offsets_dtype_and_overlap() {
        let d = [0u8; 8];
        for h in [
            r#"{"a":{"dtype":"F32","shape":[3],"data_offsets":[0,8]}}"#,
            r#"{"a":{"dtype":"F32","shape":[4],"data_offsets":[0,16]}}"#,
            r#"{"a":{"dtype":"F16","shape":[4],"data_offsets":[0,8]}}"#,
            r#"{"a":{"dtype":"I8","shape":[4],"data_offsets":[0,8]}}"#,
            r#"{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"b":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}}"#,
            r#"{"a":{"dtype":"F32","shape":[2],"data_offsets":[8,0]}}"#,
        ] {
            assert!(SafeTensors::parse(&build(h, &d)).is_err(), "{h}");
        }
        assert!(SafeTensors::parse(&[1, 0]).is_err());
        let mut huge = u64::MAX.to_le_bytes().to_vec();
        huge.extend_from_slice(b"{}");
        assert!(SafeTensors::parse(&huge).is_err());
    }
}
