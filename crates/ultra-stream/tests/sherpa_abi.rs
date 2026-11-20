//! Live `c-api.h` struct layout equivalence and `0.wav` C-API transcription integration tests.
//!
//! Paths resolve via `ULTRA_SHERPA_HEADER`, `ULTRA_SHERPA_LIB`, `ULTRA_TEST_WAV`, and `JSR_DATA`
//! (falling back to ancestor `.venv/` discovery); missing optional artifacts emit an explicit
//! `SKIPPED` diagnostic unless `ULTRA_REQUIRE_NATIVE_TESTS=1` is set.

use std::mem::{offset_of, size_of};
use std::path::{Path, PathBuf};
use std::process::Command;

use ultra_core::events::TranscriptEvent;
use ultra_eval::pace::PacedReplay;
use ultra_eval::stream::{replay, TimedEvent};
use ultra_stream::sherpa_c_types::{
    SherpaOnnxOfflineModelConfig, SherpaOnnxOfflineRecognizerConfig,
    SherpaOnnxOfflineRecognizerResult, SherpaOnnxOnlineModelConfig,
    SherpaOnnxOnlineRecognizerConfig, SherpaOnnxOnlineRecognizerResult,
};
use ultra_stream::{AsrStream, Backend, OutputPolicy, PollResult, StreamConfig, StreamCoordinator};

fn collect_all_events(coord: &mut StreamCoordinator) -> Vec<TranscriptEvent> {
    let mut out = Vec::new();
    while let PollResult::Event(ev) = coord.poll_event().expect("poll") {
        out.push(*ev);
    }
    out
}

fn require_native() -> bool {
    matches!(
        std::env::var("ULTRA_REQUIRE_NATIVE_TESTS").as_deref(),
        Ok("1" | "true")
    )
}

fn locate_sherpa_header() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("ULTRA_SHERPA_HEADER") {
        let pb = PathBuf::from(p);
        if pb.exists() {
            return Some(pb);
        }
    }
    let rel = ".venv/lib/python3.12/site-packages/sherpa_onnx/include/sherpa-onnx/c-api/c-api.h";
    for anc in Path::new(env!("CARGO_MANIFEST_DIR")).ancestors().take(8) {
        let cand = anc.join(rel);
        if cand.exists() {
            return Some(cand);
        }
    }
    None
}

fn locate_test_wav() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("ULTRA_TEST_WAV") {
        let pb = PathBuf::from(p);
        if pb.exists() {
            return Some(pb);
        }
    }
    let data = match std::env::var("JSR_DATA") {
        Ok(v) if !v.trim().is_empty() => v,
        _ => String::from("/mnt/xbabe3-external/jsr"),
    };
    let p = Path::new(&data).join("models/zipformer-en-2023-06-26/test_wavs/0.wav");
    p.exists().then_some(p)
}

#[test]
fn c_abi_struct_layout_matches_sherpa_header_when_present() {
    let Some(hdr) = locate_sherpa_header() else {
        assert!(
            !require_native(),
            "c-api.h missing while ULTRA_REQUIRE_NATIVE_TESTS=1"
        );
        eprintln!(
            "SKIPPED c_abi_struct_layout_matches_sherpa_header_when_present: c-api.h not found (set ULTRA_SHERPA_HEADER or run `just venv`)"
        );
        return;
    };
    let cc = match std::env::var("CC") {
        Ok(v) if !v.trim().is_empty() => v,
        _ => String::from("cc"),
    };
    let tmp = tempfile::tempdir().expect("tempdir");
    let c_src = tmp.path().join("check_abi.c");
    let bin = tmp.path().join("check_abi");
    let code = format!(
        r#"#include <stdio.h>
#include <stddef.h>
#include "{}"
int main(void) {{
    printf("%zu %zu %zu %zu %zu %zu %zu %zu %zu %zu\n",
        sizeof(SherpaOnnxOnlineModelConfig),
        sizeof(SherpaOnnxOnlineRecognizerConfig),
        offsetof(SherpaOnnxOnlineRecognizerConfig, hr),
        sizeof(SherpaOnnxOnlineRecognizerResult),
        offsetof(SherpaOnnxOnlineRecognizerResult, json),
        sizeof(SherpaOnnxOfflineModelConfig),
        offsetof(SherpaOnnxOfflineModelConfig, moonshine),
        sizeof(SherpaOnnxOfflineRecognizerConfig),
        offsetof(SherpaOnnxOfflineRecognizerConfig, ctc_fst_decoder_config),
        sizeof(SherpaOnnxOfflineRecognizerResult)
    );
    return 0;
}}
"#,
        hdr.display()
    );
    std::fs::write(&c_src, code).expect("write c");
    let Ok(status) = Command::new(&cc)
        .args([c_src.as_os_str(), "-o".as_ref(), bin.as_os_str()])
        .status()
    else {
        assert!(
            !require_native(),
            "C compiler `{cc}` missing while ULTRA_REQUIRE_NATIVE_TESTS=1"
        );
        eprintln!("SKIPPED c_abi_struct_layout: C compiler `{cc}` not available");
        return;
    };
    assert!(status.success(), "cc failed");
    let out = Command::new(&bin).output().expect("run abi check");
    let nums: Vec<usize> = String::from_utf8_lossy(&out.stdout)
        .split_whitespace()
        .map(|s| s.parse::<usize>().expect("num"))
        .collect();
    let rust_nums = vec![
        size_of::<SherpaOnnxOnlineModelConfig>(),
        size_of::<SherpaOnnxOnlineRecognizerConfig>(),
        offset_of!(SherpaOnnxOnlineRecognizerConfig, hr),
        size_of::<SherpaOnnxOnlineRecognizerResult>(),
        offset_of!(SherpaOnnxOnlineRecognizerResult, json),
        size_of::<SherpaOnnxOfflineModelConfig>(),
        offset_of!(SherpaOnnxOfflineModelConfig, moonshine),
        size_of::<SherpaOnnxOfflineRecognizerConfig>(),
        offset_of!(SherpaOnnxOfflineRecognizerConfig, ctc_fst_decoder_config),
        size_of::<SherpaOnnxOfflineRecognizerResult>(),
    ];
    assert_eq!(rust_nums, nums, "Rust #[repr(C)] layout != c-api.h");
}

#[test]
fn real_sherpa_zipformer_c_api_transcribes_and_satisfies_causality_when_present() {
    let Some(wav) = locate_test_wav() else {
        assert!(
            !require_native(),
            "test wav missing while ULTRA_REQUIRE_NATIVE_TESTS=1"
        );
        eprintln!(
            "SKIPPED real_sherpa_zipformer_c_api_transcribes: test wav not found (set ULTRA_TEST_WAV or JSR_DATA)"
        );
        return;
    };
    let lib = match ultra_stream::SherpaLib::load_default() {
        Ok(l) => l,
        Err(e) => {
            assert!(
                !require_native(),
                "SherpaLib missing while ULTRA_REQUIRE_NATIVE_TESTS=1: {e}"
            );
            eprintln!(
                "SKIPPED real_sherpa_zipformer_c_api_transcribes: {e} (set ULTRA_SHERPA_LIB)"
            );
            return;
        }
    };
    let engine = ultra_stream::SherpaZipformerBackend::new(
        lib,
        ultra_stream::SherpaZipformerConfig::default(),
    )
    .expect("load zipformer int8 via C API");
    let pcm = ultra_frontend::read_audio_16k_mono(&wav).expect("read 0.wav");
    let cfg = StreamConfig {
        stream_id: "wav0".into(),
        policy: OutputPolicy::EditableDictation,
        ..StreamConfig::default()
    };
    let mut coord =
        StreamCoordinator::new(cfg.clone(), engine.create_stream(&cfg).expect("stream"));
    let mut timed = Vec::new();
    for pkt in PacedReplay::new(&pcm, 20, 0.0).expect("replay") {
        let t_s = (pkt.start_sample + pkt.pcm.len() as u64) as f64 / 16_000.0;
        coord.push_pcm(pkt.start_sample, pkt.pcm).expect("push");
        for ev in collect_all_events(&mut coord) {
            timed.push(TimedEvent { t_s, event: ev });
        }
    }
    coord.finish_input().expect("finish");
    let end_s = pcm.len() as f64 / 16_000.0;
    for ev in collect_all_events(&mut coord) {
        timed.push(TimedEvent {
            t_s: end_s,
            event: ev,
        });
    }
    let ss = replay(&timed).expect("ledger replay");
    assert!(!ss.final_text.is_empty());
    assert_eq!(ss.commit_revisions, 0);

    let cut = (pcm.len() / 2) & !319;
    if cut >= 16_000 {
        let mut altered = pcm.clone();
        altered[cut..].fill(0);
        let run_zip = |buf: &[i16]| {
            let mut c =
                StreamCoordinator::new(cfg.clone(), engine.create_stream(&cfg).expect("st"));
            let mut pre = Vec::new();
            for pkt in PacedReplay::new(&buf[..cut], 20, 0.0).expect("replay") {
                c.push_pcm(pkt.start_sample, pkt.pcm).expect("push");
                pre.extend(collect_all_events(&mut c));
            }
            for (k, ch) in buf[cut..].chunks(320).enumerate() {
                c.push_pcm((cut + k * 320) as u64, ch).expect("push tail");
                let _ = collect_all_events(&mut c);
            }
            pre
        };
        assert_eq!(run_zip(&pcm), run_zip(&altered));
    }
}
