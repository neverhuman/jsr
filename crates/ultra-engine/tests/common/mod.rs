//! Shared fixtures for the engine's data-backed tests (skipped when the data drive is absent).

#![allow(dead_code)]

use std::path::PathBuf;

pub fn data_root() -> PathBuf {
    std::env::var("JSR_DATA")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/mnt/xbabe3-external/jsr"))
}

/// The converted R1-D1 weights (override with `ULTRA_ENGINE_WEIGHTS`).
pub fn weights_dir() -> PathBuf {
    std::env::var("ULTRA_ENGINE_WEIGHTS")
        .map(PathBuf::from)
        .unwrap_or_else(|_| data_root().join("runs/engine/weights-r1-d1-ckpt7800-v2"))
}

/// Per-layer torch reference written by `r1_to_safetensors.py reference`.
pub fn reference_file() -> PathBuf {
    std::env::var("ULTRA_ENGINE_REFERENCE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| data_root().join("runs/engine/ref-d1-dev50-v1/reference.safetensors"))
}

pub fn dev_clean_npz() -> PathBuf {
    data_root().join("cache/librispeech_pcm/dev-clean.npz")
}

pub fn have(paths: &[&PathBuf]) -> bool {
    let ok = paths.iter().all(|p| p.exists());
    if !ok {
        eprintln!("SKIPPED: missing one of {paths:?}");
    }
    ok
}
