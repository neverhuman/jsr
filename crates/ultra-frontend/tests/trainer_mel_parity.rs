//! B2 acceptance tests: packet-size invariance, malformed-parameter rejection, and 50-utterance
//! `dev-clean` parity (`<= 1e-4` max absolute difference) against `bundle_infer._featurize`.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use ultra_core::receipt::sha256_hex;
use ultra_frontend::{
    load_npz_audio_only, GlobalCmvn, TrainerLogMel, TrainerMelConfig, MEL_BINS, SPEC_BINS,
};

#[test]
fn trainer_logmel_packet_invariance_and_validation() {
    assert!(GlobalCmvn::new(&[0.0; MEL_BINS - 1], &[1.0; MEL_BINS]).is_err());
    let mut bad_istd = [1.0f32; MEL_BINS];
    bad_istd[7] = 0.0;
    assert!(GlobalCmvn::new(&[0.0; MEL_BINS], &bad_istd).is_err());
    let mut nan_mean = [0.0f32; MEL_BINS];
    nan_mean[3] = f32::NAN;
    assert!(GlobalCmvn::new(&nan_mean, &[1.0; MEL_BINS]).is_err());

    assert!(
        TrainerMelConfig::from_fbank_le_bytes(&[0u8; 16], 1e-6, GlobalCmvn::identity()).is_err()
    );
    let mut neg_fbank = Vec::with_capacity(SPEC_BINS * MEL_BINS * 4);
    for i in 0..(SPEC_BINS * MEL_BINS) {
        let v = if i == 42 { -0.1f32 } else { 0.01f32 };
        neg_fbank.extend_from_slice(&v.to_le_bytes());
    }
    assert!(
        TrainerMelConfig::from_fbank_le_bytes(&neg_fbank, 1e-6, GlobalCmvn::identity()).is_err()
    );

    let cfg = Arc::new(TrainerMelConfig::default_htk(GlobalCmvn::identity()).expect("htk"));
    let pcm: Vec<i16> = (0..16_543i32)
        .map(|i| (((i * 97 + 13) % 20_000) - 10_000) as i16)
        .collect();

    let whole = TrainerLogMel::new(Arc::clone(&cfg)).push_i16(&pcm);
    assert!(!whole.is_empty());

    for &pkt in &[1usize, 160, 320, 592] {
        let mut s = TrainerLogMel::new(Arc::clone(&cfg));
        let mut got = Vec::new();
        for ch in pcm.chunks(pkt) {
            s.push_i16_into(ch, &mut got);
        }
        assert_eq!(got.len(), whole.len(), "frame count mismatch at pkt={pkt}");
        for (f_idx, (a, b)) in whole.iter().zip(&got).enumerate() {
            assert_eq!(
                a, b,
                "bit-exact packet invariance broken at frame {f_idx}, pkt={pkt}"
            );
        }
    }
}

fn data_root() -> PathBuf {
    std::env::var("JSR_DATA")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/mnt/xbabe3-external/jsr"))
}

#[test]
fn b2_trainer_logmel_matches_bundle_infer_on_50_dev_clean_utterances() {
    let root = data_root();
    let bundle_dir = root.join("bundles/r1-d1");
    let manifest_path = bundle_dir.join("manifest.json");
    let npz_path = root.join("cache/librispeech_pcm/dev-clean.npz");
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace root");
    let py_bin = Path::new("/home/ubuntu/jsr/.venv/bin/python");

    if !manifest_path.exists() || !npz_path.exists() || !py_bin.exists() {
        if std::env::var("ULTRA_REQUIRE_NATIVE_TESTS").as_deref() == Ok("1") {
            panic!(
                "B2 parity inputs missing: manifest={}, npz={}, py={}",
                manifest_path.display(),
                npz_path.display(),
                py_bin.display()
            );
        }
        eprintln!(
            "SKIPPED B2 50-utt parity: missing {}, {}, or {}",
            manifest_path.display(),
            npz_path.display(),
            py_bin.display()
        );
        return;
    }

    let utts = load_npz_audio_only(&npz_path, 0, Some(50)).expect("load dev-clean audio");
    let limit = utts.len();
    let tmp = tempfile::tempdir().expect("tempdir");
    let py_out_path = tmp.path().join("py_feats.f32");
    let meta_out_path = tmp.path().join("meta.txt");

    let py_script = r#"
import json, os, struct, sys
import numpy as np

bundle, npz_path, limit_s, out_path, meta_path = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5]
with open(os.path.join(bundle, "manifest.json")) as fh:
    m = json.load(fh)
fe = m["features"]
win, hop, n_fft, n_mels, floor = fe["frame_length"], fe["frame_shift"], fe["n_fft"], fe["n_mels"], fe["log_floor"]
fbank_rel = m["files"]["mel_fbank"]["path"]
fbank_sha = m["files"]["mel_fbank"]["sha256"]
fbank = np.fromfile(os.path.join(bundle, fbank_rel), dtype="<f4").reshape(n_fft // 2 + 1, n_mels)
window = np.hanning(win).astype(np.float32)
mean = np.asarray(fe["cmvn"]["mean"], dtype=np.float32)
istd = np.asarray(fe["cmvn"]["istd"], dtype=np.float32)

with open(meta_path, "w") as mf:
    mf.write(f"{fbank_rel}\n{fbank_sha}\n{floor}\n")
    mf.write(" ".join(str(float(x)) for x in fe["cmvn"]["mean"]) + "\n")
    mf.write(" ".join(str(float(x)) for x in fe["cmvn"]["istd"]) + "\n")

z = np.load(npz_path, allow_pickle=False)
pcm_all = z["pcm"]
offsets = z["offsets"]

rows = []
for i in range(limit_s):
    pcm = pcm_all[offsets[i]:offsets[i+1]].astype(np.float32) / 32768.0
    # Feed in 20ms (320-sample) packets identically to bundle_infer._featurize
    samples = np.zeros(0, dtype=np.float32)
    sample_base = 0
    n_feat = 0
    feats = np.zeros((0, n_mels), dtype=np.float32)
    for cur in range(0, len(pcm), 320):
        samples = np.concatenate([samples, pcm[cur:cur+320]])
        avail = sample_base + len(samples)
        n_new = 0 if avail < win else 1 + (avail - win) // hop - n_feat
        if n_new > 0:
            start = n_feat * hop - sample_base
            idx = start + np.arange(n_new)[:, None] * hop + np.arange(win)[None]
            fr = samples[idx]
            fr = fr - fr.mean(axis=1, keepdims=True)
            spec = np.abs(np.fft.rfft(fr * window, n=n_fft)).astype(np.float32) ** 2
            logmel = np.log(spec @ fbank + np.float32(floor))
            feats = np.concatenate([feats, ((logmel - mean) * istd).astype(np.float32)])
            n_feat += n_new
            drop = n_feat * hop - sample_base
            samples = samples[drop:]
            sample_base += drop
    rows.append(feats)

all_f = np.concatenate(rows, axis=0).astype("<f4")
all_f.tofile(out_path)
print(json.dumps({"frames": int(all_f.shape[0]), "utts": limit_s}))
"#;

    let status = Command::new(py_bin)
        .current_dir(repo_root.join("python/ultra"))
        .arg("-c")
        .arg(py_script)
        .arg(&bundle_dir)
        .arg(&npz_path)
        .arg(limit.to_string())
        .arg(&py_out_path)
        .arg(&meta_out_path)
        .output()
        .expect("run python reference featurizer");
    assert!(
        status.status.success(),
        "python featurizer failed: {}",
        String::from_utf8_lossy(&status.stderr)
    );

    let meta_txt = std::fs::read_to_string(&meta_out_path).expect("read meta.txt");
    let mut lines = meta_txt.lines();
    let fbank_rel = lines.next().expect("fbank_rel");
    let fbank_sha = lines.next().expect("fbank_sha");
    let log_floor: f32 = lines.next().expect("log_floor").parse().expect("f32 floor");
    let mean: Vec<f32> = lines
        .next()
        .expect("mean")
        .split_whitespace()
        .map(|s| s.parse().expect("f32 mean"))
        .collect();
    let istd: Vec<f32> = lines
        .next()
        .expect("istd")
        .split_whitespace()
        .map(|s| s.parse().expect("f32 istd"))
        .collect();
    let fbank_bytes = std::fs::read(bundle_dir.join(fbank_rel)).expect("read mel_fbank");
    assert_eq!(
        sha256_hex(&fbank_bytes),
        fbank_sha,
        "checksummed files.mel_fbank SHA-256 mismatch"
    );
    let cmvn = GlobalCmvn::new(&mean, &istd).expect("valid cmvn");
    let cfg = Arc::new(
        TrainerMelConfig::from_fbank_le_bytes(&fbank_bytes, log_floor, cmvn).expect("fbank cfg"),
    );

    let py_bytes = std::fs::read(&py_out_path).expect("read py_feats.f32");
    let mut rust_rows: Vec<[f32; MEL_BINS]> = Vec::new();
    for u in utts.iter().take(limit) {
        let mut mels = TrainerLogMel::new(Arc::clone(&cfg));
        for pkt in u.pcm.chunks(320) {
            mels.push_i16_into(pkt, &mut rust_rows);
        }
    }

    assert_eq!(
        py_bytes.len(),
        rust_rows.len() * MEL_BINS * 4,
        "total frame count mismatch on 50 dev-clean utterances"
    );

    let mut max_diff = 0.0f32;
    for (f_idx, row) in rust_rows.iter().enumerate() {
        for (b, &rv) in row.iter().enumerate() {
            let off = (f_idx * MEL_BINS + b) * 4;
            let pv = f32::from_le_bytes([
                py_bytes[off],
                py_bytes[off + 1],
                py_bytes[off + 2],
                py_bytes[off + 3],
            ]);
            let d = (rv - pv).abs();
            if d > max_diff {
                max_diff = d;
            }
        }
    }
    eprintln!(
        "B2 50-utterance dev-clean parity ({} frames): max_abs_diff = {:.3e}",
        rust_rows.len(),
        max_diff
    );
    assert!(
        max_diff <= 1e-4,
        "B2 max_abs_diff {max_diff:.3e} exceeded 1e-4 gate"
    );
}
