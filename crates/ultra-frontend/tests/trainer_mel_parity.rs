//! B2 acceptance tests:
//! - Packet-size invariance (`1`, `160`, `320`, `592` samples) and malformed CMVN / fbank rejection.
//! - Direct parity against `train.bundle_infer.BundleRecognizer` / `BundleStream.accept_waveform`
//!   (no re-implemented Python math) on:
//!   1. Edge/synthetic fixtures (`< 400` samples -> 0 frames, exact `400` samples -> 1 frame,
//!      digital silence, 440 Hz sine tone, and full-scale `[-32768, 32767]` rail clipping).
//!   2. A stratified 50-utterance `dev-clean` panel spanning the full 2,703-utterance split.

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
    assert!(TrainerLogMel::new(Arc::clone(&cfg))
        .push_i16(&[0i16; 399])
        .is_empty());
    assert_eq!(
        TrainerLogMel::new(Arc::clone(&cfg))
            .push_i16(&[0i16; 400])
            .len(),
        1
    );

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

fn locate_r1_bundle(root: &Path) -> PathBuf {
    let reviewed = root.join("bundles/r1-d1-reviewed-20261120-v2");
    if reviewed.join("manifest.json").exists() {
        reviewed
    } else {
        root.join("bundles/r1-d1")
    }
}

#[test]
fn b2_trainer_logmel_matches_bundle_recognizer_on_synthetic_edges_and_stratified_50_utts() {
    let root = data_root();
    let bundle_dir = locate_r1_bundle(&root);
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
            "SKIPPED B2 parity: missing {}, {}, or {}",
            manifest_path.display(),
            npz_path.display(),
            py_bin.display()
        );
        return;
    }

    let all_utts = load_npz_audio_only(&npz_path, 0, None).expect("load dev-clean audio");
    let panel_len = 50.min(all_utts.len());
    let stride = (all_utts.len() / panel_len).max(1);
    let selected_indices: Vec<usize> = (0..panel_len).map(|i| i * stride).collect();

    // Synthetic edge fixtures: <400 samples (399), exact 400 samples, silence, 440 Hz tone, clipping.
    let sub400: Vec<i16> = vec![1234i16; 399];
    let exact400: Vec<i16> = (0..400i32).map(|n| (n * 50 - 10_000) as i16).collect();
    let silence: Vec<i16> = vec![0i16; 3_200];
    let tone440: Vec<i16> = (0..4_800)
        .map(|n| {
            let ph = 2.0 * std::f64::consts::PI * 440.0 * (n as f64) / 16_000.0;
            (ph.sin() * 24_000.0).round() as i16
        })
        .collect();
    let clipping: Vec<i16> = (0..1_600)
        .map(|n| if n % 2 == 0 { i16::MIN } else { i16::MAX })
        .collect();
    let synthetics: [&[i16]; 5] = [&sub400, &exact400, &silence, &tone440, &clipping];

    let tmp = tempfile::tempdir().expect("tempdir");
    let syn_in_path = tmp.path().join("syn_pcm.bin");
    let py_out_path = tmp.path().join("py_feats.f32");
    let meta_out_path = tmp.path().join("meta.txt");

    let mut syn_bytes = Vec::new();
    for s in synthetics {
        syn_bytes.extend_from_slice(&(s.len() as u32).to_le_bytes());
        for &v in s {
            syn_bytes.extend_from_slice(&v.to_le_bytes());
        }
    }
    std::fs::write(&syn_in_path, &syn_bytes).expect("write syn_pcm.bin");

    let idx_csv: String = selected_indices
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(",");

    let py_script = r#"
import json, os, struct, sys
import numpy as np
from train.bundle_infer import BundleRecognizer

bundle, npz_path, idx_csv, syn_path, out_path, meta_path = sys.argv[1:7]
rec = BundleRecognizer(bundle, precision="int8", threads=1, verify=True)
m = rec.m
fe = m["features"]
with open(meta_path, "w") as mf:
    mf.write(f"{m['files']['mel_fbank']['path']}\n{m['files']['mel_fbank']['sha256']}\n{fe['log_floor']}\n")
    mf.write(" ".join(str(float(x)) for x in fe["cmvn"]["mean"]) + "\n")
    mf.write(" ".join(str(float(x)) for x in fe["cmvn"]["istd"]) + "\n")

def run_one(pcm_i16):
    s = rec.create_stream()
    wav = pcm_i16.astype(np.float32) / 32768.0
    for cur in range(0, len(wav), 320):
        s.accept_waveform(16000, wav[cur:cur + 320])
    return s.feats

raw = open(syn_path, "rb").read()
pos = 0
syn_counts = []
rows = []
for _ in range(5):
    n = struct.unpack_from("<I", raw, pos)[0]
    pos += 4
    arr = np.frombuffer(raw, dtype="<i2", count=n, offset=pos)
    pos += n * 2
    f = run_one(arr)
    syn_counts.append(int(f.shape[0]))
    rows.append(f)

z = np.load(npz_path, allow_pickle=False)
pcm_all = z["pcm"]
offsets = z["offsets"]
for i in [int(x) for x in idx_csv.split(",") if x]:
    rows.append(run_one(pcm_all[offsets[i]:offsets[i+1]]))

all_f = np.concatenate(rows, axis=0).astype("<f4")
all_f.tofile(out_path)
with open(meta_path, "a") as mf:
    mf.write(" ".join(str(c) for c in syn_counts) + "\n")
"#;

    let status = Command::new(py_bin)
        .current_dir(repo_root.join("python/ultra"))
        .arg("-c")
        .arg(py_script)
        .arg(&bundle_dir)
        .arg(&npz_path)
        .arg(&idx_csv)
        .arg(&syn_in_path)
        .arg(&py_out_path)
        .arg(&meta_out_path)
        .output()
        .expect("run BundleRecognizer reference consumer");
    assert!(
        status.status.success(),
        "BundleRecognizer failed: {}",
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
    let py_syn_counts: Vec<usize> = lines
        .next()
        .expect("syn_counts")
        .split_whitespace()
        .map(|s| s.parse().expect("usize"))
        .collect();
    assert_eq!(py_syn_counts[0], 0);
    assert_eq!(py_syn_counts[1], 1);

    let fbank_bytes = std::fs::read(bundle_dir.join(fbank_rel)).expect("read mel_fbank");
    assert_eq!(sha256_hex(&fbank_bytes), fbank_sha);
    let cmvn = GlobalCmvn::new(&mean, &istd).expect("valid cmvn");
    let cfg = Arc::new(
        TrainerMelConfig::from_fbank_le_bytes(&fbank_bytes, log_floor, cmvn).expect("fbank cfg"),
    );

    let mut rust_rows: Vec<[f32; MEL_BINS]> = Vec::new();
    for (si, syn) in synthetics.iter().enumerate() {
        let before = rust_rows.len();
        let mut mels = TrainerLogMel::new(Arc::clone(&cfg));
        for pkt in syn.chunks(320) {
            mels.push_i16_into(pkt, &mut rust_rows);
        }
        assert_eq!(rust_rows.len() - before, py_syn_counts[si]);
    }
    for &idx in &selected_indices {
        let mut mels = TrainerLogMel::new(Arc::clone(&cfg));
        for pkt in all_utts[idx].pcm.chunks(320) {
            mels.push_i16_into(pkt, &mut rust_rows);
        }
    }

    let py_bytes = std::fs::read(&py_out_path).expect("read py_feats.f32");
    assert_eq!(py_bytes.len(), rust_rows.len() * MEL_BINS * 4);

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
            max_diff = max_diff.max((rv - pv).abs());
        }
    }
    eprintln!(
        "B2 BundleRecognizer parity (5 edge fixtures + 50 stratified dev-clean utts, {} frames): max_abs_diff = {:.3e}",
        rust_rows.len(),
        max_diff
    );
    assert!(
        max_diff <= 1e-4,
        "B2 max_abs_diff {max_diff:.3e} exceeded 1e-4 gate"
    );

    // G4: Verify cross-ISA portable math path matches BundleRecognizer within <= 1e-4 AND
    // matches `frame_f32_portable_scalar` to 0 ULP across every frame and bin.
    let cfg_portable = Arc::new((*cfg).clone().with_portable_math(true));
    let mut port_rows: Vec<[f32; MEL_BINS]> = Vec::with_capacity(rust_rows.len());
    for syn in synthetics {
        let mut mels = TrainerLogMel::new(Arc::clone(&cfg_portable));
        for pkt in syn.chunks(320) {
            mels.push_i16_into(pkt, &mut port_rows);
        }
    }
    for &idx in &selected_indices {
        let mut mels = TrainerLogMel::new(Arc::clone(&cfg_portable));
        for pkt in all_utts[idx].pcm.chunks(320) {
            mels.push_i16_into(pkt, &mut port_rows);
        }
    }
    assert_eq!(port_rows.len(), rust_rows.len());
    let mut port_max_diff = 0.0f32;
    for (f_idx, row) in port_rows.iter().enumerate() {
        for (b, &rv) in row.iter().enumerate() {
            let off = (f_idx * MEL_BINS + b) * 4;
            let pv = f32::from_le_bytes([
                py_bytes[off],
                py_bytes[off + 1],
                py_bytes[off + 2],
                py_bytes[off + 3],
            ]);
            port_max_diff = port_max_diff.max((rv - pv).abs());
        }
    }
    // Check 0-ULP bit-exactness between portable AVX2 and portable scalar on dev-clean utt 0.
    let u0_pcm = &all_utts[0].pcm;
    let u0_f32: Vec<f32> = u0_pcm.iter().map(|&s| f32::from(s) / 32768.0).collect();
    let mut u0_bytes = Vec::new();
    let mut off = 0usize;
    while off + 400 <= u0_f32.len() {
        let fr = &u0_f32[off..off + 400];
        let a = cfg_portable.frame_f32(fr);
        let b = cfg_portable.frame_f32_portable_scalar(fr);
        for (bin, (av, bv)) in a.iter().zip(&b).enumerate() {
            assert_eq!(
                av.to_bits(),
                bv.to_bits(),
                "G4 portable AVX2 vs scalar drift at off={off} bin={bin}"
            );
            u0_bytes.extend_from_slice(&av.to_bits().to_le_bytes());
        }
        off += 160;
    }
    eprintln!(
        "G4 portable BundleRecognizer parity ({} frames): max_abs_diff = {:.3e}, u0_portable_sha256 = {}",
        port_rows.len(),
        port_max_diff,
        sha256_hex(&u0_bytes)
    );
    assert!(
        port_max_diff <= 1e-4,
        "G4 portable max_abs_diff {port_max_diff:.3e} exceeded 1e-4 gate"
    );
}
