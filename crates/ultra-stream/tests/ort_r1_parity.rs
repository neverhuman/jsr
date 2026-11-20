//! B1 acceptance & parity tests for [`OrtTransducerBackend`]:
//! - Causal `StreamCoordinator` lifecycle, sub-chunk `processed_samples == 0`, and monotonic clock.
//! - Fast 16-utterance `dev-clean` `fp32` & `int8` parity against `python/ultra/train/bundle_infer.py`
//!   on every `cargo test`.
//! - Full 400-utterance `dev-clean` `fp32` & `int8` parity + single-core/batched release speed
//!   benchmark (`--ignored`, retaining per-utterance token IDs and every mismatch record for E1 review).

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use ultra_core::events::{Ledger, TranscriptEvent};
use ultra_eval::normalise::normalise_words;
use ultra_eval::wer::align;
use ultra_frontend::npz::{load_npz_audio_only, load_npz_references};
use ultra_gp::R1Precision;
use ultra_stream::{
    run_r1_packet_streaming_with_tokens, AsrStream, Backend, OrtTransducerBackend, OutputPolicy,
    PollResult, StreamClock, StreamConfig, StreamCoordinator,
};

fn data_root() -> PathBuf {
    std::env::var("JSR_DATA")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/mnt/xbabe3-external/jsr"))
}

fn collect_events(c: &mut StreamCoordinator) -> Vec<TranscriptEvent> {
    let mut out = Vec::new();
    while let PollResult::Event(ev) = c.poll_event().expect("poll") {
        out.push(*ev);
    }
    out
}

fn pooled_wer_pct(refs: &[String], hyps: &[String]) -> f64 {
    let mut errs = 0u64;
    let mut words = 0u64;
    for (r, h) in refs.iter().zip(hyps) {
        let rw = normalise_words(r);
        let hw = normalise_words(h);
        let al = align(&rw, &hw);
        errs += al.counts.errors();
        words += al.counts.ref_words;
    }
    if words == 0 {
        0.0
    } else {
        100.0 * (errs as f64) / (words as f64)
    }
}

struct PyBundleRun {
    hyps: Vec<String>,
    tokens: Vec<Vec<i64>>,
    forced: u64,
    cpu_s: f64,
}

fn run_python_bundle_hyps_and_tokens(
    bundle_dir: &Path,
    npz_path: &Path,
    precision: R1Precision,
    limit: usize,
) -> PyBundleRun {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace root");
    let py_bin = Path::new("/home/ubuntu/jsr/.venv/bin/python");
    let script = r#"
import json, resource, sys
import numpy as np
from train.bundle_infer import BundleRecognizer
from train.infer import run_streaming

bundle, npz_p, prec, lim = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
rec = BundleRecognizer(bundle, prec, threads=1, verify=True)
created = []
orig_create = rec.create_stream
def tracked_create():
    s = orig_create()
    created.append(s)
    return s
rec.create_stream = tracked_create

z = np.load(npz_p, allow_pickle=False)
pcm_all = z["pcm"]
offsets = z["offsets"]

class U:
    def __init__(self, pcm):
        self.pcm = pcm

utts = [U(pcm_all[offsets[i]:offsets[i+1]]) for i in range(lim)]
r0 = resource.getrusage(resource.RUSAGE_SELF)
hyps = run_streaming(rec, utts, streams=16, packet_ms=20)
r1 = resource.getrusage(resource.RUSAGE_SELF)
cpu_s = (r1.ru_utime + r1.ru_stime) - (r0.ru_utime + r0.ru_stime)
toks = [list(s.tokens) for s in created]
print(json.dumps({"hyps": hyps, "tokens": toks, "forced": rec.forced_advances, "cpu_s": cpu_s}))
"#;
    let out = Command::new(py_bin)
        .current_dir(repo_root.join("python/ultra"))
        .arg("-c")
        .arg(script)
        .arg(bundle_dir)
        .arg(npz_path)
        .arg(precision.as_str())
        .arg(limit.to_string())
        .output()
        .expect("spawn python bundle_infer");
    assert!(
        out.status.success(),
        "python bundle_infer failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let val: serde_json::Value = serde_json::from_slice(&out.stdout).expect("parse py json");
    let hyps = val["hyps"]
        .as_array()
        .expect("hyps array")
        .iter()
        .map(|v| v.as_str().expect("str").to_owned())
        .collect();
    let tokens = val["tokens"]
        .as_array()
        .expect("tokens array")
        .iter()
        .map(|arr| {
            arr.as_array()
                .expect("tok list")
                .iter()
                .map(|x| x.as_i64().expect("i64"))
                .collect()
        })
        .collect();
    let forced = val["forced"].as_u64().expect("forced");
    let cpu_s = val["cpu_s"].as_f64().expect("cpu_s");
    PyBundleRun {
        hyps,
        tokens,
        forced,
        cpu_s,
    }
}

fn locate_r1_bundle(root: &Path) -> PathBuf {
    let rev = root.join("bundles/r1-d1-reviewed-20261120-v2");
    if rev.join("manifest.json").exists() {
        rev
    } else {
        root.join("bundles/r1-d1")
    }
}

#[test]
fn ort_r1_stream_coordinator_subchunk_watermark_and_16_utt_parity() {
    let root = data_root();
    let bundle_dir = locate_r1_bundle(&root);
    let npz_path = root.join("cache/librispeech_pcm/dev-clean.npz");
    if !bundle_dir.join("manifest.json").exists()
        || !npz_path.exists()
        || !Path::new("/home/ubuntu/jsr/.venv/bin/python").exists()
    {
        eprintln!("SKIPPED ort_r1_parity: r1-d1 bundle or dev-clean.npz missing");
        return;
    }

    let engine = OrtTransducerBackend::load(&bundle_dir, R1Precision::Int8, 1, true)
        .expect("load OrtTransducerBackend int8");
    let utts = load_npz_audio_only(&npz_path, 0, Some(16)).expect("load 16 dev-clean");
    let refs = load_npz_references(&npz_path, 0, Some(16)).expect("load 16 refs");

    // 1. Verify StreamCoordinator + sub-chunk processed_samples == 0 + monotonic clock + causality.
    let cfg = StreamConfig {
        stream_id: "r1-u0".into(),
        policy: OutputPolicy::Commit,
        ..StreamConfig::default()
    };
    let mut st = engine.create_stream(&cfg).expect("create stream");
    st.push_pcm(0, &utts[0].pcm[..320]).expect("20ms subchunk");
    assert_eq!(st.metrics().accepted_samples, 320);
    assert_eq!(st.metrics().processed_samples, 0);

    let mut coord = StreamCoordinator::new(cfg.clone(), engine.create_stream(&cfg).expect("s2"));
    let mut ledger = Ledger::new();
    for (k, ch) in utts[0].pcm.chunks(320).enumerate() {
        coord.push_pcm((k * 320) as u64, ch).expect("push");
        for ev in collect_events(&mut coord) {
            assert!(ev.observed_monotonic_ns > 0);
            ledger.apply(&ev).expect("ledger apply");
        }
    }
    coord.finish_input().expect("finish");
    for ev in collect_events(&mut coord) {
        assert!(ev.observed_monotonic_ns > 0);
        ledger.apply(&ev).expect("ledger apply final");
    }

    let cut = (utts[0].pcm.len() / 2) & !319;
    if cut >= 8_000 {
        let mut alt = utts[0].pcm.clone();
        alt[cut..].fill(0);
        let run_half = |pcm: &[i16]| {
            let tick = Arc::new(AtomicU64::new(0));
            let tr = Arc::clone(&tick);
            let clk = StreamClock::custom(move || tr.fetch_add(1_000_000, Ordering::SeqCst) + 1);
            let mut c =
                StreamCoordinator::new(cfg.clone(), engine.create_stream_with_clock(&cfg, clk));
            let mut pre = Vec::new();
            for (k, ch) in pcm[..cut].chunks(320).enumerate() {
                c.push_pcm((k * 320) as u64, ch).expect("push pre");
                pre.extend(collect_events(&mut c));
            }
            for (k, ch) in pcm[cut..].chunks(320).enumerate() {
                c.push_pcm((cut + k * 320) as u64, ch).expect("push post");
                let _ = collect_events(&mut c);
            }
            pre
        };
        assert_eq!(run_half(&utts[0].pcm), run_half(&alt));
    }

    // 2. Fast 16-utterance fp32 AND int8 end-to-end streaming parity vs Python bundle_infer.py.
    let ref_texts: Vec<String> = refs.iter().map(|r| r.text.clone()).collect();
    for prec in [R1Precision::Fp32, R1Precision::Int8] {
        let eng = if prec == R1Precision::Int8 {
            Arc::clone(&engine)
        } else {
            OrtTransducerBackend::load(&bundle_dir, R1Precision::Fp32, 1, false)
                .expect("load OrtTransducerBackend fp32")
        };
        let py = run_python_bundle_hyps_and_tokens(&bundle_dir, &npz_path, prec, 16);
        let (py_hyps, py_toks, py_forced) = (py.hyps, py.tokens, py.forced);
        let rust =
            run_r1_packet_streaming_with_tokens(&eng, &utts, 16, 20).expect("rust r1 streaming");
        let (rust_hyps, rust_toks, rust_forced) = (rust.hyps, rust.tokens, rust.forced_advances);
        let py_wer = pooled_wer_pct(&ref_texts, &py_hyps);
        let rust_wer = pooled_wer_pct(&ref_texts, &rust_hyps);
        let identical_hyps = py_hyps
            .iter()
            .zip(&rust_hyps)
            .filter(|(a, b)| a == b)
            .count();
        let identical_toks = py_toks
            .iter()
            .zip(&rust_toks)
            .filter(|(a, b)| a == b)
            .count();
        eprintln!(
            "B1 16-utt [{}] parity: identical_hyps={identical_hyps}/16, identical_toks={identical_toks}/16, rust_wer={rust_wer:.3}%, py_wer={py_wer:.3}%, forced={rust_forced}/{py_forced}",
            prec.as_str()
        );
        assert!(
            (rust_wer - py_wer).abs() <= 0.1,
            "WER delta {:.4} > 0.1 for {}",
            (rust_wer - py_wer).abs(),
            prec.as_str()
        );
        if prec == R1Precision::Fp32 {
            assert_eq!(
                identical_hyps, 16,
                "expected 16/16 identical fp32 transcripts"
            );
            assert_eq!(
                identical_toks, 16,
                "expected 16/16 identical fp32 token sequences"
            );
        } else {
            assert!(
                identical_hyps >= 15,
                "expected >=15/16 identical int8 transcripts, got {identical_hyps}/16"
            );
        }
    }
}

#[test]
#[ignore = "400-utterance B1 FP32 & INT8 parity + release speed benchmark; run with --release -- --ignored"]
fn b1_full_400_dev_clean_parity_and_speed_report() {
    let root = data_root();
    let bundle_dir = locate_r1_bundle(&root);
    let npz_path = root.join("cache/librispeech_pcm/dev-clean.npz");
    let utts = load_npz_audio_only(&npz_path, 0, Some(400)).expect("400 dev-clean");
    let refs = load_npz_references(&npz_path, 0, Some(400)).expect("400 refs");
    let ref_texts: Vec<String> = refs.iter().map(|r| r.text.clone()).collect();
    let audio_s: f64 = utts.iter().map(|u| u.pcm.len() as f64 / 16_000.0).sum();

    let out_dir = root.join("runs/r1_rust_parity");
    std::fs::create_dir_all(&out_dir).expect("mkdir r1_rust_parity");

    for prec in [R1Precision::Fp32, R1Precision::Int8] {
        let engine =
            OrtTransducerBackend::load(&bundle_dir, prec, 1, true).expect("load R1 engine");
        let py = run_python_bundle_hyps_and_tokens(&bundle_dir, &npz_path, prec, 400);
        let (py_hyps, py_toks, py_forced, py_cpu_s) = (py.hyps, py.tokens, py.forced, py.cpu_s);

        let ru0 = nix::sys::resource::getrusage(nix::sys::resource::UsageWho::RUSAGE_SELF)
            .expect("rusage");
        let t0 = Instant::now();
        let rust =
            run_r1_packet_streaming_with_tokens(&engine, &utts, 16, 20).expect("rust 400 s=16");
        let (rust_hyps, rust_toks, rust_forced) = (rust.hyps, rust.tokens, rust.forced_advances);
        let wall_s16 = t0.elapsed().as_secs_f64();
        let ru1 = nix::sys::resource::getrusage(nix::sys::resource::UsageWho::RUSAGE_SELF)
            .expect("rusage");
        let cpu_f = |r: &nix::sys::resource::Usage| {
            let u = r.user_time();
            let s = r.system_time();
            (u.tv_sec() as f64 + u.tv_usec() as f64 * 1e-6)
                + (s.tv_sec() as f64 + s.tv_usec() as f64 * 1e-6)
        };
        let rust_cpu_s16 = cpu_f(&ru1) - cpu_f(&ru0);

        let py_wer = pooled_wer_pct(&ref_texts, &py_hyps);
        let rust_wer = pooled_wer_pct(&ref_texts, &rust_hyps);
        let identical = py_hyps
            .iter()
            .zip(&rust_hyps)
            .filter(|(a, b)| a == b)
            .count();
        let identical_toks = py_toks
            .iter()
            .zip(&rust_toks)
            .filter(|(a, b)| a == b)
            .count();
        let pct = 100.0 * (identical as f64) / 400.0;

        let mut mismatches = Vec::new();
        let mut rows = Vec::with_capacity(400);
        for i in 0..400 {
            let eq_txt = py_hyps[i] == rust_hyps[i];
            let eq_tok = py_toks[i] == rust_toks[i];
            if !eq_txt || !eq_tok {
                let first_diff_pos = py_toks[i]
                    .iter()
                    .zip(&rust_toks[i])
                    .position(|(a, b)| a != b)
                    .unwrap_or_else(|| py_toks[i].len().min(rust_toks[i].len()));
                mismatches.push(serde_json::json!({
                    "index": i,
                    "id": refs[i].uid,
                    "speaker": refs[i].speaker,
                    "precision": prec.as_str(),
                    "grouping": {"streams": 16, "packet_ms": 20, "chunk_frames": 4},
                    "first_diff_token_pos": first_diff_pos,
                    "py_tokens": py_toks[i],
                    "rust_tokens": rust_toks[i],
                    "py_hyp": py_hyps[i],
                    "rust_hyp": rust_hyps[i],
                }));
            }
            rows.push(serde_json::json!({
                "index": i,
                "id": refs[i].uid,
                "speaker": refs[i].speaker,
                "py_tokens": py_toks[i],
                "rust_tokens": rust_toks[i],
                "py_hyp": py_hyps[i],
                "rust_hyp": rust_hyps[i],
                "equal_text": eq_txt,
                "equal_tokens": eq_tok,
            }));
        }

        eprintln!(
            "B1 400-utt [{}] (S=16, 20ms, audio={audio_s:.1}s): identical_text={identical}/400 ({pct:.2}%), identical_toks={identical_toks}/400, mismatches={}, rust_wer={rust_wer:.3}% vs py_wer={py_wer:.3}%, rust_rtf={:.4} ({:.2}x RT, wall={wall_s16:.2}s) vs py_rtf={:.4} ({:.2}x RT), forced={rust_forced}/{py_forced}",
            prec.as_str(),
            mismatches.len(),
            rust_cpu_s16 / audio_s,
            audio_s / rust_cpu_s16,
            py_cpu_s / audio_s,
            audio_s / py_cpu_s,
        );

        let report = serde_json::json!({
            "schema": "ultra.b1-parity/v1",
            "bundle_dir": bundle_dir.to_string_lossy(),
            "bundle_schema": engine.bundle.schema,
            "int8_batch_invariant": engine.bundle.int8_batch_invariant,
            "bundle_manifest_sha256": engine.bundle.manifest_sha256,
            "ort_library_path": engine.ort.path.to_string_lossy(),
            "ort_library_sha256": engine.ort.sha256,
            "ort_version": engine.ort.version,
            "cpu_scope": "pure_engine_in_process_rusage_self",
            "precision": prec.as_str(),
            "split": "dev-clean",
            "utterances": 400,
            "audio_seconds": audio_s,
            "streams": 16,
            "packet_ms": 20,
            "threads": 1,
            "identical_hyps": identical,
            "identical_tokens": identical_toks,
            "identical_pct": pct,
            "mismatch_count": mismatches.len(),
            "rust_wer_pct": rust_wer,
            "python_wer_pct": py_wer,
            "wer_abs_delta_pct": (rust_wer - py_wer).abs(),
            "rust_forced_advances": rust_forced,
            "python_forced_advances": py_forced,
            "rust_cpu_s_per_audio_s": rust_cpu_s16 / audio_s,
            "python_cpu_s_per_audio_s": py_cpu_s / audio_s,
            "mismatches": mismatches,
            "rows": rows,
        });
        let out_file = out_dir.join(format!("r1_d1_400dev_{}.json", prec.as_str()));
        std::fs::write(&out_file, serde_json::to_vec_pretty(&report).expect("json"))
            .expect("write B1 parity report");

        if prec == R1Precision::Fp32 {
            assert!(
                (rust_wer - py_wer).abs() <= 0.1,
                "B1 FP32 WER delta {:.4} > 0.1",
                (rust_wer - py_wer).abs()
            );
            assert_eq!(identical, 400, "B1 FP32 expected 400/400 identical texts");
            assert_eq!(
                identical_toks, 400,
                "B1 FP32 expected 400/400 identical token sequences"
            );
        } else {
            // Dynamic per-tensor activation quantization (`DynamicQuantizeLinear` without axis)
            // shifts scales across co-batched streams; experimental until static QDQ export lands.
            assert!(
                (rust_wer - py_wer).abs() <= 0.25,
                "B1 experimental dynamic-INT8 WER delta {:.4} > 0.25",
                (rust_wer - py_wer).abs()
            );
            assert!(
                identical >= 350,
                "B1 experimental dynamic-INT8 identical {identical}/400; inspect {}",
                out_file.display()
            );
        }
    }
}
