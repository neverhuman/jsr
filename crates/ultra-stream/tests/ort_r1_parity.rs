//! B1 acceptance & parity tests for [`OrtTransducerBackend`]:
//! - Causal `StreamCoordinator` lifecycle, sub-chunk `processed_samples == 0`, and monotonic clock.
//! - Fast 16-utterance `dev-clean` `fp32` & `int8` parity against `python/ultra/train/bundle_infer.py`
//!   on every `cargo test`.
//! - Full 400-utterance `dev-clean` `fp32` & `int8` parity + single-core/batched release speed
//!   benchmark (`--ignored`, retaining per-utterance token IDs and every mismatch record for E1 review).

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use ultra_core::events::{Ledger, TranscriptEvent};
use ultra_eval::normalise::normalise_words;
use ultra_eval::wer::align;
use ultra_frontend::npz::{load_npz_audio_only, load_npz_references, AudioUtterance};
use ultra_frontend::{TrainerLogMel, TrainerMelConfig};
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
    utts: &[AudioUtterance],
    mel_cfg: &Arc<TrainerMelConfig>,
) -> PyBundleRun {
    let limit = utts.len();
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace root");
    let py_bin = Path::new("/home/ubuntu/jsr/.venv/bin/python");
    // G4 Option (a): feed Python's BundleRecognizer the exact same portable log-mel features that
    // Rust computes (like-for-like ORT/decoder parity; frontend parity vs torch is tested in B2).
    let mut feat_file = tempfile::NamedTempFile::new().expect("temp portable feats");
    let mut counts = Vec::with_capacity(limit);
    for u in utts {
        let frames = TrainerLogMel::new(Arc::clone(mel_cfg)).push_i16(&u.pcm);
        counts.push(frames.len().to_string());
        for fr in &frames {
            for &v in fr {
                feat_file
                    .write_all(&v.to_le_bytes())
                    .expect("write portable feat");
            }
        }
    }
    feat_file.flush().expect("flush portable feats");
    let counts_csv = counts.join(",");
    let script = r#"
import json, resource, sys
import numpy as np
import train.bundle_infer as bi
from train.infer import run_streaming

bundle, npz_p, prec, lim = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
feat_path, counts_csv = sys.argv[5], sys.argv[6]
counts = [int(x) for x in counts_csv.split(",") if x]
flat = np.fromfile(feat_path, dtype="<f4").reshape(-1, 80)
port_feats, off = [], 0
for c in counts:
    port_feats.append(flat[off:off+c])
    off += c

with open(f"{bundle}/manifest.json") as fh:
    hdr = json.load(fh)
if hasattr(bi, "SCHEMA") and hdr.get("schema") in ("ultra.r1-export/v1", "ultra.r1-export/v1.1"):
    bi.SCHEMA = hdr["schema"]
orig_json_load = bi.json.load
def compat_json_load(fh):
    m = orig_json_load(fh)
    i8 = m.get("int8") or {}
    if "joiner.int8" not in m.get("files", {}) and str(i8.get("joiner", "")).lower().startswith("fp32"):
        m["files"]["joiner.int8"] = m["files"]["joiner.fp32"]
    return m
bi.json.load = compat_json_load
rec = bi.BundleRecognizer(bundle, prec, threads=1, verify=True)
i8 = rec.m.get("int8") or {}
py_streams = int(i8.get("max_batch") or 16) if prec == "int8" else 16
created = []
orig_create = rec.create_stream
def tracked_create():
    s = orig_create()
    s._pf = port_feats[len(created)]
    created.append(s)
    return s
rec.create_stream = tracked_create
def portable_featurize(s):
    avail = s.sample_base + len(s.samples)
    n_new = 0 if avail < rec.win else 1 + (avail - rec.win) // rec.hop - s.n_feat
    if n_new <= 0:
        return
    s.feats = np.concatenate([s.feats, s._pf[s.n_feat:s.n_feat + n_new]])
    s.n_feat += n_new
    drop = s.n_feat * rec.hop - s.sample_base
    s.samples = s.samples[drop:]
    s.sample_base += drop
rec._featurize = portable_featurize

z = np.load(npz_p, allow_pickle=False)
pcm_all, offsets = z["pcm"], z["offsets"]
class U:
    def __init__(self, pcm): self.pcm = pcm
utts = [U(pcm_all[offsets[i]:offsets[i+1]]) for i in range(lim)]
r0 = resource.getrusage(resource.RUSAGE_SELF)
hyps = run_streaming(rec, utts, streams=py_streams, packet_ms=20)
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
        .arg(feat_file.path())
        .arg(&counts_csv)
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
    if let Ok(p) = std::env::var("ULTRA_B1_BUNDLE_DIR") {
        if !p.trim().is_empty() {
            return PathBuf::from(p);
        }
    }
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
    // Complete stream retained allocation includes ~6.36 MiB Dense S encoder caches.
    assert!(
        st.metrics().retained_bytes >= 6_000_000,
        "expected full stream state in retained_bytes, got {}",
        st.metrics().retained_bytes
    );

    // Late joiner/decoder failure must be terminal: retry of finish_input/push_pcm must fail and
    // NEVER emit a synthetic Final crediting failed work.
    let mut fail_coord = StreamCoordinator::new(
        cfg.clone(),
        engine.create_stream_with_late_failure(&cfg, StreamClock::real(), 1),
    );
    // Push 3,200 samples (200 ms -> chunk 0 succeeds); finish_input triggers chunk 1 failure.
    fail_coord
        .push_pcm(0, &utts[0].pcm[..3_200])
        .expect("chunk 0 ok");
    let pre_fail_evs = collect_events(&mut fail_coord);
    assert!(!pre_fail_evs
        .iter()
        .any(|e| matches!(e.body, ultra_core::events::Body::Final { .. })));
    assert!(
        fail_coord.finish_input().is_err(),
        "injected late failure must error"
    );
    assert!(
        fail_coord.finish_input().is_err(),
        "retry finish_input must error"
    );
    assert!(
        fail_coord.push_pcm(3_200, &[0i16; 320]).is_err(),
        "push after failure must error"
    );
    while let PollResult::Event(ev) = fail_coord.poll_event().expect("poll after fail") {
        assert!(
            !matches!(ev.body, ultra_core::events::Body::Final { .. }),
            "no Final after fail"
        );
    }

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
        let py = run_python_bundle_hyps_and_tokens(
            &bundle_dir,
            &npz_path,
            prec,
            &utts,
            &eng.bundle.mel_cfg,
        );
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

    let run_dir = match std::env::var("ULTRA_B1_PARITY_OUT_DIR") {
        Ok(p) if !p.is_empty() => PathBuf::from(p),
        _ => {
            let ms = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_millis());
            root.join("runs/r1_rust_parity")
                .join(format!("run-{}-{ms}", std::process::id()))
        }
    };
    std::fs::create_dir_all(&run_dir).expect("mkdir unique parity run_dir");

    let only_prec = std::env::var("ULTRA_B1_ONLY_PRECISION").ok();
    let mut fp32_rust_wer_opt: Option<f64> = None;
    for prec in [R1Precision::Fp32, R1Precision::Int8] {
        if let Some(ref want) = only_prec {
            if !want.trim().is_empty() && want != prec.as_str() {
                continue;
            }
        }
        if prec == R1Precision::Int8 && fp32_rust_wer_opt.is_none() {
            let eng_fp32 = OrtTransducerBackend::load(&bundle_dir, R1Precision::Fp32, 1, false)
                .expect("load FP32 baseline for INT8 WER delta");
            let fp32_run = run_r1_packet_streaming_with_tokens(&eng_fp32, &utts, 16, 20)
                .expect("rust 400 fp32 baseline");
            fp32_rust_wer_opt = Some(pooled_wer_pct(&ref_texts, &fp32_run.hyps));
        }
        let engine =
            OrtTransducerBackend::load(&bundle_dir, prec, 1, true).expect("load R1 engine");
        let py = run_python_bundle_hyps_and_tokens(
            &bundle_dir,
            &npz_path,
            prec,
            &utts,
            &engine.bundle.mel_cfg,
        );
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
        if prec == R1Precision::Fp32 {
            fp32_rust_wer_opt = Some(rust_wer);
        }
        let fp32_wer = fp32_rust_wer_opt.unwrap_or(rust_wer);
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

        let wer_delta_vs_py = (rust_wer - py_wer).abs();
        let wer_delta_vs_fp32 = (rust_wer - fp32_wer).abs();
        eprintln!(
            "B1 400-utt [{}] (S=16, 20ms, audio={audio_s:.1}s): identical_text={identical}/400 ({pct:.2}%), identical_toks={identical_toks}/400, mismatches={}, rust_wer={rust_wer:.3}% vs py_wer={py_wer:.3}% (fp32={fp32_wer:.3}%), rust_rtf={:.4} ({:.2}x RT, wall={wall_s16:.2}s) vs py_rtf={:.4} ({:.2}x RT), forced={rust_forced}/{py_forced}",
            prec.as_str(),
            mismatches.len(),
            rust_cpu_s16 / audio_s,
            audio_s / rust_cpu_s16,
            py_cpu_s / audio_s,
            audio_s / py_cpu_s,
        );

        let wer_gate_pass = wer_delta_vs_py <= 0.10;
        let wer_vs_fp32_gate_pass = wer_delta_vs_fp32 <= 0.10;
        let agreement_gate_pass = identical >= 392;
        let handoff_accepted = wer_gate_pass && agreement_gate_pass;
        let mut report = serde_json::json!({
            "schema": "ultra.b1-parity/v1",
            "bundle_dir": bundle_dir.to_string_lossy(),
            "bundle_schema": engine.bundle.schema,
            "int8_mode": engine.bundle.int8_mode,
            "int8_graph_batch_invariant": engine.bundle.int8_graph_batch_invariant,
            "int8_consumer_batch_invariant": engine.bundle.int8_consumer_batch_invariant(),
            "int8_batch_invariant": engine.bundle.int8_batch_invariant,
            "int8_max_batch_enc": engine.bundle.int8_max_batch_enc,
            "int8_max_batch_joi": engine.bundle.int8_max_batch_joi,
            "joiner_precision": engine.bundle.joiner_precision.as_str(),
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
            "wer_abs_delta_pct": wer_delta_vs_py,
            "wer_gate_0p10_pass": wer_gate_pass,
            "agreement_gate_98pct_pass": agreement_gate_pass,
            "parity_handoff_accepted": handoff_accepted,
            "experimental_health_screen_only": prec == R1Precision::Int8 && !engine.bundle.int8_consumer_batch_invariant(),
            "rust_forced_advances": rust_forced,
            "python_forced_advances": py_forced,
            "rust_cpu_s_per_audio_s": rust_cpu_s16 / audio_s,
            "python_cpu_s_per_audio_s": py_cpu_s / audio_s,
            "mismatches": mismatches,
            "rows": rows,
        });
        if let Some(obj) = report.as_object_mut() {
            obj.insert("rust_fp32_wer_pct".into(), serde_json::json!(fp32_wer));
            obj.insert(
                "wer_abs_delta_vs_fp32_pct".into(),
                serde_json::json!(wer_delta_vs_fp32),
            );
            obj.insert(
                "precision_quality_vs_fp32_0p10_pass".into(),
                serde_json::json!(wer_vs_fp32_gate_pass),
            );
        }
        let out_file = run_dir.join(format!("r1_d1_400dev_{}.json", prec.as_str()));
        {
            use std::io::Write;
            let mut fh = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&out_file)
                .expect("create_new B1 parity report");
            fh.write_all(&serde_json::to_vec_pretty(&report).expect("json"))
                .expect("write B1 parity report");
        }

        if prec == R1Precision::Fp32 {
            assert!(
                wer_gate_pass,
                "B1 FP32 WER delta {wer_delta_vs_py:.4} > 0.1"
            );
            assert_eq!(identical, 400, "B1 FP32 expected 400/400 identical texts");
            assert_eq!(
                identical_toks, 400,
                "B1 FP32 expected 400/400 identical tokens"
            );
        } else if engine.bundle.int8_batch_invariant {
            assert!(
                wer_gate_pass,
                "B1 v1.1 INT8 WER delta vs Python {wer_delta_vs_py:.4} > 0.10; inspect {}",
                out_file.display()
            );
            assert!(
                agreement_gate_pass,
                "B1 v1.1 INT8 identical {identical}/400 (< 392); inspect {}",
                out_file.display()
            );
        } else {
            assert!(
                wer_delta_vs_py <= 0.25,
                "B1 v1 dynamic-INT8 WER delta {wer_delta_vs_py:.4} > 0.25"
            );
            assert!(
                identical >= 350,
                "B1 v1 dynamic-INT8 {identical}/400; {}",
                out_file.display()
            );
        }
    }
}
