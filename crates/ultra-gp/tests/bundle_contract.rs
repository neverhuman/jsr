//! Strict validation tests for `ultra.r1-export/v1` (`R1Bundle`, `TokenTable`, `FeatureRing`).

use std::path::PathBuf;

use ultra_frontend::MEL_BINS;
use ultra_gp::{
    checked_argmax_f32, chunk_bounds, FeatureRing, ModelFamily, R1Bundle, R1Precision, TokenTable,
};

#[test]
fn precision_and_token_table_reject_malformed_inputs() {
    assert_eq!(checked_argmax_f32(&[-1.0, 2.5, 2.5, 0.5]), Ok(1));
    assert!(checked_argmax_f32(&[]).is_err());
    assert!(checked_argmax_f32(&[0.0, f32::NAN, 1.0]).is_err());
    assert!(checked_argmax_f32(&[0.0, f32::INFINITY]).is_err());
    assert!(checked_argmax_f32(&[f32::NEG_INFINITY, 0.0]).is_err());

    assert!(R1Precision::parse("fp32").is_ok());
    assert!(R1Precision::parse("int8").is_ok());
    assert!(R1Precision::parse("auto").is_err());
    assert!(R1Precision::parse("fp16").is_err());
    assert!(ModelFamily::parse("dense").is_ok());
    assert!(ModelFamily::parse("multirate").is_ok());
    assert!(ModelFamily::parse("conformer").is_err());

    let ok_tokens = "<blk> 0\n\u{2581}hello 1\n\u{2581}world 2\n! 3\n";
    let tt = TokenTable::from_str_checked(ok_tokens, 4).expect("valid tokens");
    assert_eq!(tt.decode_ids(&[0, 1, 2, 3]), "hello world!");

    assert!(TokenTable::from_str_checked("unk 0\na 1\n", 2).is_err());
    assert!(TokenTable::from_str_checked("<blk> 0\na 2\n", 2).is_err());
    assert!(TokenTable::from_str_checked("<blk> 0\na 1\nb 1\n", 3).is_err());
}

#[test]
fn chunk_bounds_and_feature_ring_match_python_stem_window() {
    assert_eq!(chunk_bounds(0, 12, 4, false), None);
    assert_eq!(chunk_bounds(0, 13, 4, false), Some((0, 4)));
    assert_eq!(chunk_bounds(4, 28, 4, false), None);
    assert_eq!(chunk_bounds(4, 29, 4, false), Some((4, 8)));
    assert_eq!(chunk_bounds(4, 18, 4, true), Some((4, 5)));

    let mut ring = FeatureRing::new();
    let rows: Vec<[f32; MEL_BINS]> = (0..20)
        .map(|i| {
            let mut r = [0.0f32; MEL_BINS];
            r[0] = (i + 1) as f32;
            r
        })
        .collect();
    ring.push_rows(&rows);

    // Chunk 0: t0 = 0, n = 4 -> [4*0 - 6 ..= 4*4 - 4] = [-6 ..= 12] (19 rows: 6 zeros + rows 0..=12)
    let mut dst0 = vec![-1.0f32; 19 * MEL_BINS];
    ring.extract_chunk_into(0, 4, &mut dst0);
    for r in 0..6 {
        assert_eq!(dst0[r * MEL_BINS], 0.0);
    }
    for r in 6..19 {
        assert_eq!(dst0[r * MEL_BINS], (r - 6 + 1) as f32);
    }

    // Chunk 1: t0 = 4, n = 1 -> [10 ..= 16] (7 rows: rows 10..=16 -> values 11..=17)
    let mut dst1 = vec![-1.0f32; 7 * MEL_BINS];
    ring.extract_chunk_into(4, 1, &mut dst1);
    for r in 0..7 {
        assert_eq!(dst1[r * MEL_BINS], (11 + r) as f32);
    }
}

#[test]
fn real_r1_d1_bundle_loads_and_tampered_copy_is_rejected() {
    let root = std::env::var("JSR_DATA")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/mnt/xbabe3-external/jsr"));
    let bundle_dir = root.join("bundles/r1-d1");
    if !bundle_dir.join("manifest.json").exists() {
        eprintln!("SKIPPED R1Bundle::load on r1-d1: not present");
        return;
    }

    let b = R1Bundle::load(&bundle_dir, R1Precision::Int8, true).expect("load verified r1-d1");
    assert_eq!(b.family, ModelFamily::Dense);
    assert_eq!(b.streaming.chunk_frames, 4);
    assert_eq!(b.streaming.joiner_dim, 320);
    assert_eq!(b.decoder.vocab_size, 4097);
    assert_eq!(b.streaming.states.len(), 4);

    let v11_dir = root.join("bundles/r1-d1-v1.1");
    if v11_dir.join("manifest.json").exists() {
        let b11 = R1Bundle::load(&v11_dir, R1Precision::Int8, true).expect("load r1-d1-v1.1");
        assert_eq!(b11.schema, "ultra.r1-export/v1.1");
        assert!(b11.int8_batch_invariant);
    }

    // Build a minimal synthetic bundle in tempdir and verify tamper/path-traversal rejection.
    let tmp = tempfile::tempdir().expect("tempdir");
    let bad_manifest = serde_json::json!({
        "schema": "ultra.r1-export/v1",
        "model": {"family": "dense", "exporter_commit": "abc"},
        "files": {
            "tokens": {"path": "../escape.txt", "sha256": "0".repeat(64)}
        },
        "features": {
            "sample_rate": 16000, "frame_length": 400, "frame_shift": 160,
            "n_fft": 512, "n_mels": 80, "log_floor": 1e-6,
            "cmvn": {"mean": vec![0.0; 80], "istd": vec![1.0; 80]}
        },
        "streaming": {
            "chunk_frames": 4, "chunk_ms": 160, "left_context_frames": 128,
            "states": [], "encoder_output": {"name": "enc", "shape": ["B", "n", 320]}
        },
        "decoder": {"blank_id": 0, "vocab_size": 2, "ctx_init": [0, 0], "max_symbols": 8}
    });
    std::fs::write(
        tmp.path().join("manifest.json"),
        serde_json::to_vec(&bad_manifest).expect("json"),
    )
    .expect("write");
    assert!(R1Bundle::load(tmp.path(), R1Precision::Int8, true).is_err());
}
