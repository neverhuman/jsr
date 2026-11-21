//! The loader refuses tampered weight directories: a changed CMVN istd, a wrong hash for a
//! payload the engine never opens (bpe.model), and a declared file that is a symlink leaving
//! the directory. Fixtures hard-link (or copy) the real converted D1 files.

mod common;

use std::path::{Path, PathBuf};

use ultra_engine::{EngineError, Model};

/// A fixture directory removed on drop (the data drive is exFAT: no hard links, so files
/// are copied).
struct Fixture(PathBuf);

impl std::ops::Deref for Fixture {
    type Target = Path;
    fn deref(&self) -> &Path {
        &self.0
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn fixture(name: &str, edit: impl Fn(&mut serde_json::Value)) -> Option<Fixture> {
    let src = common::weights_dir();
    if !common::have(&[&src]) {
        return None;
    }
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("loader-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("mkdir");
    for f in [
        "weights.safetensors",
        "tokens.txt",
        "mel_fbank.f32",
        "bpe.model",
    ] {
        if std::fs::hard_link(src.join(f), dir.join(f)).is_err() {
            std::fs::copy(src.join(f), dir.join(f)).expect("copy");
        }
    }
    let mut arch: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(src.join("arch.json")).expect("arch"))
            .expect("json");
    edit(&mut arch);
    std::fs::write(dir.join("arch.json"), arch.to_string()).expect("write");
    Some(Fixture(dir))
}

#[test]
fn untouched_copy_loads() {
    if let Some(d) = fixture("ok", |_| {}) {
        Model::load(&d, true).expect("unmodified directory loads");
    }
}

#[test]
fn tampered_istd_is_refused() {
    if let Some(d) = fixture("istd", |a| a["features"]["cmvn"]["istd"][0] = 1.5.into()) {
        assert!(matches!(
            Model::load(&d, true),
            Err(EngineError::Integrity(_))
        ));
    }
}

#[test]
fn wrong_bpe_hash_is_refused() {
    if let Some(d) = fixture("bpe", |a| {
        a["files"]["bpe"]["sha256"] = "0".repeat(64).into()
    }) {
        assert!(matches!(
            Model::load(&d, true),
            Err(EngineError::Integrity(_))
        ));
    }
}

#[test]
fn missing_bpe_declaration_is_refused() {
    if let Some(d) = fixture("nobpe", |a| {
        a["files"].as_object_mut().expect("files").remove("bpe");
    }) {
        assert!(Model::load(&d, true).is_err());
    }
}

#[test]
fn escaping_symlink_is_refused() {
    if let Some(d) = fixture("link", |_| {}) {
        std::fs::remove_file(d.join("bpe.model")).expect("rm");
        if let Err(e) =
            std::os::unix::fs::symlink(common::weights_dir().join("bpe.model"), d.join("bpe.model"))
        {
            eprintln!("SKIPPED: this filesystem refuses symlinks ({e})");
            return;
        }
        assert!(matches!(
            Model::load(&d, false),
            Err(EngineError::Format(_))
        ));
    }
}
