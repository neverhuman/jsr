use std::collections::BTreeSet;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use ultra_bridge::nemotron_service::collect_texts;

static NEXT: AtomicU64 = AtomicU64::new(0);
struct Fixture(PathBuf);
impl Fixture {
    fn new() -> Self {
        let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/bridge-tests")
            .join(format!(
                "files-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
        std::fs::create_dir_all(&p).expect("fixture directory");
        Self(p)
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).expect("cleanup");
    }
}

#[test]
fn missing_extra_malformed_and_symlinked_hypotheses_fail() {
    let f = Fixture::new();
    let expected = BTreeSet::from(["u00000".into()]);
    assert!(collect_texts(&f.0, &expected).is_err());
    std::fs::write(f.0.join("u00000.txt"), b"").expect("explicit empty result");
    let good =
        collect_texts(&f.0, &expected).expect("explicit empty result is distinct from absence");
    assert_eq!(good["u00000"], "");
    std::fs::write(f.0.join("unexpected.txt"), b"stale").expect("extra result");
    assert!(collect_texts(&f.0, &expected).is_err());
    std::fs::remove_file(f.0.join("unexpected.txt")).expect("remove extra");
    std::fs::write(f.0.join("u00000.txt"), [0xff]).expect("invalid UTF8");
    assert!(collect_texts(&f.0, &expected).is_err());
    std::fs::write(f.0.join("u00000.txt"), vec![b'a'; 4097]).expect("oversized result");
    assert!(collect_texts(&f.0, &expected).is_err());
    std::fs::remove_file(f.0.join("u00000.txt")).expect("remove bad result");
    std::fs::write(f.0.join("elsewhere"), b"text").expect("symlink target");
    std::os::unix::fs::symlink("elsewhere", f.0.join("u00000.txt")).expect("symlink");
    assert!(collect_texts(&f.0, &expected).is_err());
}
