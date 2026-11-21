//! The DEV selection panel (`ultra.dev-panel/v1`, `python/ultra/tools/dev_panel.py`): uid
//! lists per split with speakers, a manifest hash and per-split cache hashes, all verified
//! before a panel run.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde_json::Value;

pub const PANEL_SCHEMA: &str = "ultra.dev-panel/v1";

/// One split of the panel.
#[derive(Debug, Clone)]
pub struct PanelSplit {
    pub split: String,
    pub cache_sha256: String,
    /// Selected uids (sorted, as in the manifest).
    pub uids: Vec<String>,
    pub speaker_of: BTreeMap<String, i64>,
}

#[derive(Debug, Clone)]
pub struct Panel {
    pub manifest_sha256: String,
    pub splits: Vec<PanelSplit>,
}

impl Panel {
    /// Read and verify a panel manifest: schema, the manifest hash (recomputed exactly as
    /// `dev_panel.py` writes it) and per-split uid/speaker consistency.
    pub fn read(path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let mut v: Value = serde_json::from_str(&text).map_err(|e| format!("panel: {e}"))?;
        let obj = v.as_object_mut().ok_or("panel is not an object")?;
        let want = obj
            .remove("manifest_sha256")
            .and_then(|s| s.as_str().map(str::to_owned))
            .ok_or("panel has no manifest_sha256")?;
        let got = ultra_core::receipt::sha256_hex(py_dumps_indent1(&v)?.as_bytes());
        if got != want {
            return Err(format!("panel manifest_sha256 {want} != recomputed {got}"));
        }
        if v["schema"] != PANEL_SCHEMA {
            return Err(format!("panel schema {} != {PANEL_SCHEMA}", v["schema"]));
        }
        let mut splits = Vec::new();
        for s in v["splits"].as_array().ok_or("panel has no splits")? {
            let name = s["split"].as_str().ok_or("split without name")?;
            if name.is_empty() || name.contains(['/', '\\', '.']) {
                return Err(format!("bad split name {name}"));
            }
            let uids: Vec<String> = s["uids"]
                .as_array()
                .ok_or("split without uids")?
                .iter()
                .map(|u| u.as_str().map(str::to_owned).ok_or("non-string uid"))
                .collect::<Result<_, _>>()?;
            let speaker_of: BTreeMap<String, i64> = s["speaker_of"]
                .as_object()
                .ok_or("split without speaker_of")?
                .iter()
                .map(|(k, sp)| sp.as_i64().map(|x| (k.clone(), x)).ok_or("bad speaker"))
                .collect::<Result<_, _>>()?;
            let distinct: std::collections::BTreeSet<&String> = uids.iter().collect();
            if distinct.len() != uids.len()
                || uids.iter().any(|u| !speaker_of.contains_key(u))
                || speaker_of.len() != uids.len()
                || s["utterances"].as_u64() != Some(uids.len() as u64)
            {
                return Err(format!("panel split {name}: uids/speakers inconsistent"));
            }
            splits.push(PanelSplit {
                split: name.to_owned(),
                cache_sha256: s["cache_sha256"]
                    .as_str()
                    .ok_or("split without cache_sha256")?
                    .to_owned(),
                uids,
                speaker_of,
            });
        }
        if splits.is_empty() {
            return Err("panel has no splits".into());
        }
        Ok(Self {
            manifest_sha256: want,
            splits,
        })
    }

    /// `<cache_dir>/<split>.npz`, verified against the manifest's cache hash.
    pub fn verified_cache(&self, cache_dir: &Path, i: usize) -> Result<PathBuf, String> {
        let s = &self.splits[i];
        let p = cache_dir.join(format!("{}.npz", s.split));
        let got = sha256_file(&p)?;
        if got != s.cache_sha256 {
            return Err(format!(
                "{}: sha256 {got} != panel cache_sha256 {}",
                p.display(),
                s.cache_sha256
            ));
        }
        Ok(p)
    }
}

fn sha256_file(p: &Path) -> Result<String, String> {
    use sha2::{Digest, Sha256};
    use std::io::Read;
    let mut f = std::fs::File::open(p).map_err(|e| format!("{}: {e}", p.display()))?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf).map_err(|e| e.to_string())?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(h.finalize().iter().map(|b| format!("{b:02x}")).collect())
}

/// Python `json.dumps(v, indent=1, sort_keys=True)` (default `ensure_ascii`) for the value
/// kinds a panel holds: objects, arrays, strings, integers, booleans and null. Floats are
/// refused (Python's float repr is not reproduced).
pub fn py_dumps_indent1(v: &Value) -> Result<String, String> {
    let mut out = String::new();
    dump(v, 0, &mut out)?;
    Ok(out)
}

fn dump(v: &Value, depth: usize, out: &mut String) -> Result<(), String> {
    let pad = |d: usize, out: &mut String| {
        out.push('\n');
        out.push_str(&" ".repeat(d));
    };
    match v {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => {
            if n.is_f64() {
                return Err("panel hash: float values are not supported".into());
            }
            out.push_str(&n.to_string());
        }
        Value::String(s) => py_str(s, out),
        Value::Array(a) if a.is_empty() => out.push_str("[]"),
        Value::Array(a) => {
            out.push('[');
            for (i, x) in a.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                pad(depth + 1, out);
                dump(x, depth + 1, out)?;
            }
            pad(depth, out);
            out.push(']');
        }
        Value::Object(m) if m.is_empty() => out.push_str("{}"),
        Value::Object(m) => {
            let mut keys: Vec<&String> = m.keys().collect();
            keys.sort_by(|a, b| a.encode_utf16().cmp(b.encode_utf16()));
            out.push('{');
            for (i, k) in keys.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                pad(depth + 1, out);
                py_str(k, out);
                out.push_str(": ");
                dump(&m[*k], depth + 1, out)?;
            }
            pad(depth, out);
            out.push('}');
        }
    }
    Ok(())
}

fn py_str(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 || (c as u32) > 0x7e => {
                let mut b = [0u16; 2];
                for u in c.encode_utf16(&mut b) {
                    out.push_str(&format!("\\u{u:04x}"));
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_python_indent1_sort_keys() {
        // python3 -c 'import json;print(json.dumps({"b":[1,{"z":"é\n","a":[]}],"a":{}, \
        //   "c":True,"d":None,"e":"q\"\\"},indent=1,sort_keys=True))'
        let v: Value = serde_json::from_str(
            r#"{"b":[1,{"z":"é\n","a":[]}],"a":{},"c":true,"d":null,"e":"q\"\\"}"#,
        )
        .unwrap();
        let want = "{\n \"a\": {},\n \"b\": [\n  1,\n  {\n   \"a\": [],\n   \"z\": \"\\u00e9\\n\"\n  }\n ],\n \"c\": true,\n \"d\": null,\n \"e\": \"q\\\"\\\\\"\n}";
        assert_eq!(py_dumps_indent1(&v).unwrap(), want);
        assert!(py_dumps_indent1(&serde_json::json!({"x": 0.5})).is_err());
    }
}
