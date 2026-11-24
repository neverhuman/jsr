//! A prepared-INT8 directory reproduces the load-time packing exactly and is refused when
//! its artifact or declaration is tampered with.

mod common;

use std::path::{Path, PathBuf};

use ultra_engine::Model;

struct Dir(PathBuf);

impl Drop for Dir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn packings(m: &mut Model) -> Vec<Option<ultra_engine::quant::PackedI8>> {
    m.linears_mut()
        .into_iter()
        .map(|(_, _, l)| l.q.clone())
        .collect()
}

#[test]
fn prepared_matches_load_time_quantisation_and_rejects_tampering() {
    let w = common::weights_dir();
    if !common::have(&[&w]) {
        return;
    }
    let out = Dir(
        Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("prepared-{}", std::process::id()))
    );
    let _ = std::fs::remove_dir_all(&out.0);
    let skip = vec!["joiner_out".to_string()];
    let mut m = Model::load(&w, true).expect("load fp32");
    m.quantize_int8_where(|k, _| !skip.iter().any(|s| s == k));
    let sha = m.write_prepared_int8(&out.0, &skip).expect("write");
    assert!(
        m.write_prepared_int8(&out.0, &skip).is_err(),
        "refuses a non-empty directory"
    );
    let mut p = Model::load(&out.0, true).expect("load prepared");
    assert!(
        !out.0.join("weights.safetensors").exists(),
        "self-contained slim INT8 directory must not ship weights.safetensors"
    );
    assert_eq!(
        p.int8_prepared.as_ref().map(|x| x.sha256.clone()),
        Some(sha)
    );
    assert_eq!(
        p.int8_prepared.as_ref().map(|x| x.self_contained),
        Some(true)
    );
    assert_eq!(packings(&mut p), packings(&mut m), "bit-identical packing");
    // Quantised layers keep no fp32 weights; the kept-fp32 kind (joiner_out) does.
    assert!(p
        .linears_mut()
        .into_iter()
        .all(|(k, _, l)| l.has_fp32() == (k == "joiner_out")));
    let p = std::sync::Arc::new(p);
    assert!(ultra_engine::Engine::new(
        p.clone(),
        ultra_engine::Isa::detect(),
        ultra_engine::Precision::F32
    )
    .is_err());
    assert!(ultra_engine::Engine::new(
        p,
        ultra_engine::Isa::detect(),
        ultra_engine::Precision::Int8
    )
    .is_ok());
    // Tamper with one code byte: the arch.json hash check refuses it.
    let f = out.0.join("weights.int8.safetensors");
    let mut bytes = std::fs::read(&f).expect("read");
    let last = bytes.len() - 1;
    bytes[last] ^= 1;
    std::fs::write(&f, &bytes).expect("write");
    assert!(Model::load(&out.0, true).is_err());
}

/// Multirate models round-trip too (pair/restore included): random-init S_MR fixture.
#[test]
fn multirate_prepared_round_trip() {
    let w = common::data_root().join("runs/engine/smr-random-seed1/weights");
    if !common::have(&[&w]) {
        return;
    }
    let out =
        Dir(Path::new(env!("CARGO_TARGET_TMPDIR"))
            .join(format!("prepared-mr-{}", std::process::id())));
    let _ = std::fs::remove_dir_all(&out.0);
    let mut m = Model::load(&w, true).expect("load fp32");
    m.quantize_int8();
    m.write_prepared_int8(&out.0, &[]).expect("write");
    let mut p = Model::load(&out.0, true).expect("load prepared");
    assert!(
        p.pair.as_ref().is_some_and(|l| l.q.is_some())
            && p.restore.as_ref().is_some_and(|l| l.q.is_some())
    );
    assert_eq!(packings(&mut p), packings(&mut m), "bit-identical packing");
}
