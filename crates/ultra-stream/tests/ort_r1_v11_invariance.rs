//! `ultra.r1-export/v1.1` consumer batch-invariance & Python parity verification across
//! `streams = 1, 3, 5, 16, 32` concurrent streams on `r1-d1-v1.1-dynrow` (`max_batch = 1`)
//! and `r1-d1-v1.1` (`static_qdq`).

use std::path::{Path, PathBuf};
use std::process::Command;

use ultra_frontend::npz::load_npz_audio_only;
use ultra_gp::R1Precision;
use ultra_stream::{run_r1_packet_streaming_with_tokens, OrtTransducerBackend};

fn data_root() -> PathBuf {
    std::env::var("JSR_DATA")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/mnt/xbabe3-external/jsr"))
}

fn run_py_single_stream(
    bundle_dir: &Path,
    npz_path: &Path,
    limit: usize,
) -> (Vec<String>, Vec<Vec<i64>>, u64) {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace root");
    let py_bin = Path::new("/home/ubuntu/jsr/.venv/bin/python");
    let script = r#"
import json, sys
import numpy as np
import train.bundle_infer as bi
from train.infer import run_streaming

bundle, npz_p, lim = sys.argv[1], sys.argv[2], int(sys.argv[3])
with open(f"{bundle}/manifest.json") as fh:
    hdr = json.load(fh)
if hasattr(bi, "SCHEMA") and hdr.get("schema") in ("ultra.r1-export/v1", "ultra.r1-export/v1.1"):
    bi.SCHEMA = hdr["schema"]
rec = bi.BundleRecognizer(bundle, "int8", threads=1, verify=True)
created = []
orig_create = rec.create_stream
def tracked_create():
    s = orig_create()
    created.append(s)
    return s
rec.create_stream = tracked_create
z = np.load(npz_p, allow_pickle=False)
pcm_all, offsets = z["pcm"], z["offsets"]
class U:
    def __init__(self, pcm): self.pcm = pcm
utts = [U(pcm_all[offsets[i]:offsets[i+1]]) for i in range(lim)]
hyps = run_streaming(rec, utts, streams=1, packet_ms=20)
print(json.dumps({"hyps": hyps, "tokens": [list(s.tokens) for s in created], "forced": rec.forced_advances}))
"#;
    let out = Command::new(py_bin)
        .current_dir(repo_root.join("python/ultra"))
        .arg("-c")
        .arg(script)
        .arg(bundle_dir)
        .arg(npz_path)
        .arg(limit.to_string())
        .output()
        .expect("spawn python bundle_infer");
    assert!(
        out.status.success(),
        "python bundle_infer failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let val: serde_json::Value = serde_json::from_slice(&out.stdout).expect("json");
    let hyps = val["hyps"]
        .as_array()
        .expect("hyps")
        .iter()
        .map(|v| v.as_str().expect("s").to_owned())
        .collect();
    let toks = val["tokens"]
        .as_array()
        .expect("tokens")
        .iter()
        .map(|a| {
            a.as_array()
                .expect("arr")
                .iter()
                .map(|x| x.as_i64().expect("i64"))
                .collect()
        })
        .collect();
    (hyps, toks, val["forced"].as_u64().expect("forced"))
}

#[test]
fn v11_int8_bundles_are_batch_invariant_across_1_3_5_16_32_streams_and_match_python() {
    let root = data_root();
    let npz_path = root.join("cache/librispeech_pcm/dev-clean.npz");
    if !npz_path.exists() {
        eprintln!("SKIPPED ort_r1_v11_invariance: dev-clean.npz missing");
        return;
    }
    let utts = load_npz_audio_only(&npz_path, 0, Some(8)).expect("load 8 dev-clean");

    for rel in ["bundles/r1-d1-v1.1-dynrow", "bundles/r1-d1-v1.1"] {
        let bdir = root.join(rel);
        if !bdir.join("manifest.json").exists() {
            continue;
        }
        let eng = OrtTransducerBackend::load(&bdir, R1Precision::Int8, 1, true)
            .expect("load v1.1 int8 bundle");
        assert!(eng.bundle.int8_consumer_batch_invariant());

        let base = run_r1_packet_streaming_with_tokens(&eng, &utts, 1, 20).expect("s=1");
        for &streams in &[3usize, 5, 16, 32] {
            let run = run_r1_packet_streaming_with_tokens(&eng, &utts, streams, 20)
                .expect("multi-stream v1.1 int8");
            assert_eq!(
                run.tokens, base.tokens,
                "{rel}: token drift between streams=1 and streams={streams}"
            );
            assert_eq!(
                run.hyps, base.hyps,
                "{rel}: transcript drift between streams=1 and streams={streams}"
            );
            assert_eq!(run.forced_advances, base.forced_advances);
        }

        if Path::new("/home/ubuntu/jsr/.venv/bin/python").exists() {
            let (py_hyps, py_toks, py_forced) = run_py_single_stream(&bdir, &npz_path, 8);
            let eq_toks = py_toks
                .iter()
                .zip(&base.tokens)
                .filter(|(a, b)| a == b)
                .count();
            let eq_hyps = py_hyps
                .iter()
                .zip(&base.hyps)
                .filter(|(a, b)| a == b)
                .count();
            eprintln!(
                "v1.1 [{rel}] 8-utt INT8 parity vs Python: eq_hyps={eq_hyps}/8, eq_toks={eq_toks}/8, forced={}/{py_forced}",
                base.forced_advances
            );
            assert_eq!(eq_hyps, 8, "{rel}: expected 8/8 identical INT8 transcripts");
            assert_eq!(
                eq_toks, 8,
                "{rel}: expected 8/8 identical INT8 token sequences"
            );
        }
    }
}
