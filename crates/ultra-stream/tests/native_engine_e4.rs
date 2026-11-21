//! `E4` (`R-ACCEL-3`) integration tests for [`NativeEngineBackend`] (`ultra-engine` SafeTensors):
//! - Causal `StreamCoordinator` + `Ledger` lifecycle (`EditableDictation` & `Commit`), sub-chunk
//!   `processed_samples == 0`, and strict `threads == 1` enforcement.
//! - Exact `FP32` streaming token & transcript parity against [`OrtTransducerBackend`] (`8` utts).
//! - `32`-instance simultaneous mixed-length `INT8` batch invariance (`S = 1, 3, 5, 16, 32`).

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use ultra_core::events::{Ledger, TranscriptEvent};
use ultra_engine::Precision;
use ultra_frontend::npz::{load_npz_audio_only, AudioUtterance};
use ultra_gp::R1Precision;
use ultra_stream::{
    run_engine_packet_streaming_with_tokens, run_r1_packet_streaming_with_tokens, AsrStream,
    Backend, NativeEngineBackend, OrtTransducerBackend, OutputPolicy, PollResult, StreamClock,
    StreamConfig, StreamCoordinator,
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

#[test]
fn native_engine_e4_lifecycle_fp32_ort_parity_and_int8_32stream_invariance() {
    let require_native = std::env::var("ULTRA_REQUIRE_NATIVE_TESTS").as_deref() == Ok("1");
    let root = data_root();
    let weights_fp32 = root.join("runs/engine/weights-r1-d1-ckpt7800-v3");
    let npz_path = root.join("cache/librispeech_pcm/dev-clean.npz");
    if !weights_fp32.join("arch.json").exists() || !npz_path.exists() {
        assert!(
            !require_native,
            "ULTRA_REQUIRE_NATIVE_TESTS=1 but weights ({}) or dev-clean.npz ({}) missing",
            weights_fp32.display(),
            npz_path.display()
        );
        eprintln!("SKIPPED native_engine_e4: weights or dev-clean.npz missing");
        return;
    }

    // 1. Strict threads == 1 enforcement.
    assert!(
        NativeEngineBackend::load(&weights_fp32, Precision::F32, 2, false).is_err(),
        "NativeEngineBackend must reject threads != 1"
    );

    let utts = load_npz_audio_only(&npz_path, 0, Some(8)).expect("load 8 dev-clean");
    let fp32_eng = NativeEngineBackend::load(&weights_fp32, Precision::F32, 1, true)
        .expect("load NativeEngineBackend fp32");
    assert_eq!(fp32_eng.int8_source, "n/a");

    // 2. Causal StreamCoordinator + Ledger contract (EditableDictation & Commit) + causality check.
    for policy in [OutputPolicy::EditableDictation, OutputPolicy::Commit] {
        let cfg = StreamConfig {
            stream_id: format!("e4-{}", policy.as_str()),
            policy,
            ..StreamConfig::default()
        };
        let mut coord = StreamCoordinator::new(
            cfg.clone(),
            fp32_eng.create_stream(&cfg).expect("create stream"),
        );
        // Sub-chunk (320 samples = 20 ms < 160 ms chunk): processed_samples must stay 0.
        coord
            .push_pcm(0, &utts[0].pcm[..320])
            .expect("push 20ms packet");
        assert_eq!(coord.metrics().processed_samples, 0);

        let mut ledger = Ledger::new();
        for (k, ch) in utts[0].pcm[320..].chunks(320).enumerate() {
            coord
                .push_pcm(((k + 1) * 320) as u64, ch)
                .expect("push pcm");
            for ev in collect_events(&mut coord) {
                ledger.apply(&ev).expect("ledger apply");
            }
        }
        coord.finish_input().expect("finish");
        for ev in collect_events(&mut coord) {
            ledger.apply(&ev).expect("ledger apply final");
        }
        assert!(coord.metrics().retained_bytes > 0);
        assert!(
            coord
                .push_pcm(utts[0].pcm.len() as u64, &[0i16; 320])
                .is_err(),
            "push after finish_input must fail"
        );
    }

    // Future-audio causality probe.
    let cut = (utts[0].pcm.len() / 2) & !319;
    if cut >= 8_000 {
        let mut alt = utts[0].pcm.clone();
        alt[cut..].fill(0);
        let run_half = |pcm: &[i16]| {
            let tick = Arc::new(AtomicU64::new(0));
            let tr = Arc::clone(&tick);
            let clk = StreamClock::custom(move || tr.fetch_add(1_000_000, Ordering::SeqCst) + 1);
            let cfg = StreamConfig::default();
            let mut c =
                StreamCoordinator::new(cfg.clone(), fp32_eng.create_stream_with_clock(&cfg, clk));
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

    // 3. Exact 8-utterance FP32 token & transcript parity vs OrtTransducerBackend (same ckpt 7800).
    let ort_bdir = if root.join("bundles/r1-d1-v1.1b/manifest.json").exists() {
        root.join("bundles/r1-d1-v1.1b")
    } else {
        root.join("bundles/r1-d1-reviewed-20261120-v2")
    };
    if ort_bdir.join("manifest.json").exists() {
        let ort_fp32 = OrtTransducerBackend::load(&ort_bdir, R1Precision::Fp32, 1, false)
            .expect("load ORT fp32");
        let ort_run =
            run_r1_packet_streaming_with_tokens(&ort_fp32, &utts, 16, 20).expect("ort fp32 s=16");
        let eng_run = run_engine_packet_streaming_with_tokens(&fp32_eng, &utts, 16, 20)
            .expect("native engine fp32 s=16");
        assert_eq!(
            eng_run.tokens, ort_run.tokens,
            "FP32 token mismatch between NativeEngineBackend and OrtTransducerBackend"
        );
        assert_eq!(eng_run.hyps, ort_run.hyps);
        assert_eq!(eng_run.forced_advances, ort_run.forced_advances);
    }

    // 4. Unprepared FP32 directory must be refused by strict INT8 service loader; prepared
    //    `weights.int8.safetensors` directory loads with `int8_source == "prepared"` and is
    //    batch-invariant across 32 simultaneous mixed-length stream instances (S = 1, 3, 5, 16, 32).
    assert!(
        NativeEngineBackend::load(&weights_fp32, Precision::Int8, 1, false).is_err(),
        "NativeEngineBackend::load(Int8) must refuse unprepared FP32 directory"
    );
    let prep_dir = root.join("runs/engine/weights-r1-d1-ckpt7800-v3-int8prep-723da8f");
    let int8_eng = if prep_dir.join("arch.json").exists() {
        let b = NativeEngineBackend::load(&prep_dir, Precision::Int8, 1, true)
            .expect("load prepared NativeEngineBackend int8");
        assert_eq!(b.int8_source, "prepared");
        b
    } else {
        assert!(
            !require_native,
            "ULTRA_REQUIRE_NATIVE_TESTS=1 refuses diagnostic INT8 fallback; missing {}",
            prep_dir.display()
        );
        NativeEngineBackend::load_diagnostic_allow_unprepared_int8(
            &weights_fp32,
            Precision::Int8,
            1,
            true,
        )
        .expect("diagnostic int8 fallback")
    };
    let utts_32: Vec<AudioUtterance> = (0..32)
        .map(|i| {
            let src = &utts[i % utts.len()];
            let want = (18_000 + (i % 7) * 3_517).min(src.pcm.len());
            AudioUtterance {
                index: i,
                pcm: src.pcm[..want].to_vec(),
            }
        })
        .collect();
    let base = run_engine_packet_streaming_with_tokens(&int8_eng, &utts_32, 1, 20).expect("s=1");
    assert_eq!(base.max_active_streams, 1);
    for &streams in &[3usize, 5, 16, 32] {
        let run = run_engine_packet_streaming_with_tokens(&int8_eng, &utts_32, streams, 20)
            .expect("multi-stream native engine int8");
        assert_eq!(run.max_active_streams, streams);
        assert_eq!(
            run.tokens, base.tokens,
            "NativeEngineBackend INT8 token drift at streams={streams}"
        );
        assert_eq!(run.hyps, base.hyps);
    }
}

#[test]
fn verified_affinity_fails_closed_and_physical_first_deduplicates_smt() {
    use ultra_stream::{physical_first_cpus, pin_current_thread_verified};

    // Out-of-range CPU ID must fail closed instead of silently ignoring affinity errors.
    assert!(pin_current_thread_verified(&[999_999]).is_err());

    // Querying and re-pinning to the thread's own current affinity succeeds and matches readback.
    let cur = pin_current_thread_verified(&[]).expect("query current affinity");
    assert!(!cur.is_empty());
    let eff = pin_current_thread_verified(&cur).expect("re-pin current affinity");
    assert_eq!(eff, cur);

    // Physical-first filtering returns a non-empty subset without assuming IDs < 64.
    let phys = physical_first_cpus(&cur);
    assert!(!phys.is_empty());
    assert!(phys.len() <= cur.len());
    for &c in &phys {
        assert!(cur.contains(&c));
    }
}
