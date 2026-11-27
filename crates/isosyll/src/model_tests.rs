//! Unit tests for `model.rs`.

use super::*;

fn touch_model(root: &Path) {
    std::fs::create_dir_all(root).unwrap();
    std::fs::write(root.join("arch.json"), "{}").unwrap();
}

#[test]
fn flag_is_exclusive() {
    let td = tempfile::tempdir().unwrap();
    let good = td.path().join("good");
    touch_model(&good);
    let env = ResolveEnv {
        model_env: Some(good),
        ..ResolveEnv::default()
    };
    let err = resolve(Some(&td.path().join("nope")), &env).unwrap_err();
    assert!(err.contains("nope") && err.contains("install"));
}

#[test]
fn order_env_then_exe_then_home() {
    let td = tempfile::tempdir().unwrap();
    let exe = td.path().join("prefix/bin/isosyll");
    let share = td.path().join("prefix/bin/../share/isosyll/model");
    let home = td.path().join("home");
    let home_model = home.join(".local/share/isosyll/model");
    let env = ResolveEnv {
        model_env: Some(td.path().join("envmodel")),
        exe: Some(exe),
        home: Some(home),
    };
    let c = candidates(None, &env);
    assert_eq!(c.len(), 3);
    assert_eq!(c[1], share);
    assert_eq!(c[2], home_model);
    // Only the home install exists: it is found.
    touch_model(&home_model);
    assert_eq!(resolve(None, &env).unwrap().root, home_model);
    // Creating the exe-relative install takes precedence over home.
    touch_model(&share);
    assert_eq!(resolve(None, &env).unwrap().root, share);
    // Env var wins over both.
    touch_model(&td.path().join("envmodel"));
    assert_eq!(
        resolve(None, &env).unwrap().root,
        td.path().join("envmodel")
    );
}

#[test]
fn not_found_names_all_paths() {
    let env = ResolveEnv {
        model_env: Some("/e/m".into()),
        exe: Some("/p/bin/isosyll".into()),
        home: Some("/h".into()),
    };
    let err = resolve(None, &env).unwrap_err();
    assert!(err.contains("/e/m") && err.contains("share/isosyll/model"));
    assert!(err.contains("/h/.local/share/isosyll/model"));
}

#[test]
fn manifest_parse_rejects_bad_lines() {
    let sha = "a".repeat(64);
    assert!(parse_manifest(&format!("{sha}  arch.json\n")).is_ok());
    assert!(parse_manifest(&format!("{sha}  ../x\n")).is_err());
    assert!(parse_manifest(&format!("{sha}  /abs\n")).is_err());
    assert!(parse_manifest("zz  arch.json\n").is_err());
    assert!(parse_manifest("# only comment\n").is_err());
}

#[test]
fn verify_detects_mismatch_and_version_line() {
    let td = tempfile::tempdir().unwrap();
    let root = td.path().to_path_buf();
    std::fs::write(root.join("arch.json"), "hello").unwrap();
    let good = hex(&Sha256::digest(b"hello"));
    std::fs::write(root.join("MANIFEST"), format!("{good}  arch.json\n")).unwrap();
    let m = ModelDir { root: root.clone() };
    assert_eq!(verify(&m, |_, _| {}).unwrap(), (1, 0));
    assert!(version_line(Some(&m)).starts_with("isosyll 4.2.1 ("));
    assert!(version_line(Some(&m)).contains(
        &hex(&Sha256::digest(
            std::fs::read(root.join("MANIFEST")).unwrap()
        ))[..12]
    ));
    std::fs::write(root.join("arch.json"), "tampered").unwrap();
    assert_eq!(verify(&m, |_, _| {}).unwrap(), (0, 1));
    assert!(version_line(None).ends_with("model not found"));
}

#[test]
fn final_pass_files_live_beside_the_live_bundle() {
    let m = ModelDir {
        root: PathBuf::from("/m"),
    };
    assert_eq!(m.final_acoustic(), Path::new("/m/final"));
    assert_eq!(
        m.final_files(),
        [
            "/m/final/arch.json",
            "/m/lm/final.ulm",
            "/m/lm/t2.utfr",
            "/m/final/content-fit.json",
            "/m/final/content-lexicon.json",
        ]
        .map(PathBuf::from)
    );
    assert!(INSTALL_HINT.contains("final/arch.json") && INSTALL_HINT.contains("t2.utfr"));
}

#[test]
fn live_endpoint_reranker_files_are_required_in_live_mode() {
    let m = ModelDir {
        root: PathBuf::from("/m"),
    };
    assert_eq!(m.live_content_fit(), Path::new("/m/live-content-fit.json"));
    assert_eq!(
        m.live_files(),
        [
            "/m/lm/live.ulm",
            "/m/lm/final.ulm",
            "/m/lm/t2.utfr",
            "/m/live-content-fit.json",
            "/m/final/content-lexicon.json",
        ]
        .map(PathBuf::from)
    );
}
