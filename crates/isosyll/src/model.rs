//! Model directory resolution, `MANIFEST` verification and the version string.

use std::fs::File;
use std::io::Read;
use std::path::{Component, Path, PathBuf};

use sha2::{Digest, Sha256};

pub const INSTALL_HINT: &str = "install the isosyll model bundle so that <model>/arch.json, <model>/final/arch.json, <model>/lm/live.ulm, <model>/lm/final.ulm, <model>/lm/t2.utfr and <model>/MANIFEST exist (see the release notes for the install instructions), or pass --model DIR";

/// Environment inputs for model resolution (injectable for tests).
#[derive(Debug, Clone, Default)]
pub struct ResolveEnv {
    pub model_env: Option<PathBuf>,
    pub exe: Option<PathBuf>,
    pub home: Option<PathBuf>,
}

impl ResolveEnv {
    pub fn from_process() -> Self {
        let nonempty = |k: &str| {
            std::env::var_os(k)
                .filter(|v| !v.is_empty())
                .map(PathBuf::from)
        };
        Self {
            model_env: nonempty("ISOSYLL_MODEL"),
            exe: std::env::current_exe().ok(),
            home: nonempty("HOME"),
        }
    }
}

/// Candidate model directories in resolution order (an explicit `--model` is the only one).
pub fn candidates(flag: Option<&Path>, env: &ResolveEnv) -> Vec<PathBuf> {
    if let Some(f) = flag {
        return vec![f.to_path_buf()];
    }
    let mut v = Vec::new();
    if let Some(e) = &env.model_env {
        v.push(e.clone());
    }
    if let Some(dir) = env.exe.as_deref().and_then(Path::parent) {
        v.push(dir.join("..").join("share").join("isosyll").join("model"));
    }
    if let Some(h) = &env.home {
        v.push(h.join(".local/share/isosyll/model"));
    }
    v
}

/// A resolved model directory.
#[derive(Debug, Clone)]
pub struct ModelDir {
    pub root: PathBuf,
}

impl ModelDir {
    pub fn live_lm(&self) -> PathBuf {
        self.root.join("lm").join("live.ulm")
    }
    pub fn final_lm(&self) -> PathBuf {
        self.root.join("lm").join("final.ulm")
    }
    /// Final-pass acoustic bundle (v4.2.0: F3 soup), separate from the live bundle at the root.
    pub fn final_acoustic(&self) -> PathBuf {
        self.root.join("final")
    }
    /// L2R INT8 Transformer used by the final pass's reranker.
    pub fn final_rescorer(&self) -> PathBuf {
        self.root.join("lm").join("t2.utfr")
    }
    pub fn content_fit(&self) -> PathBuf {
        self.root.join("final").join("content-fit.json")
    }
    pub fn content_lexicon(&self) -> PathBuf {
        self.root.join("final").join("content-lexicon.json")
    }
    /// Every file `--mode final` reads besides the live bundle.
    pub fn final_files(&self) -> Vec<PathBuf> {
        vec![
            self.final_acoustic().join("arch.json"),
            self.final_lm(),
            self.final_rescorer(),
            self.content_fit(),
            self.content_lexicon(),
        ]
    }
    pub fn manifest(&self) -> PathBuf {
        self.root.join("MANIFEST")
    }
}

/// Pick the first candidate that contains `arch.json`.
pub fn resolve(flag: Option<&Path>, env: &ResolveEnv) -> Result<ModelDir, String> {
    let tried = candidates(flag, env);
    for c in &tried {
        if c.join("arch.json").is_file() {
            return Ok(ModelDir { root: c.clone() });
        }
    }
    let list = if tried.is_empty() {
        "(none: no --model, ISOSYLL_MODEL, executable path or HOME available)".to_owned()
    } else {
        tried
            .iter()
            .map(|p| p.display().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    };
    Err(format!(
        "model not found (looked for arch.json in: {list}); {INSTALL_HINT}"
    ))
}

/// Check that the LM files a decode needs exist, naming the missing path.
pub fn require_file(p: &Path) -> Result<(), String> {
    if p.is_file() {
        Ok(())
    } else {
        Err(format!(
            "missing model file {}; {INSTALL_HINT}",
            p.display()
        ))
    }
}

fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write;
    bytes.iter().fold(String::new(), |mut s, b| {
        let _ = write!(s, "{b:02x}");
        s
    })
}

pub fn sha256_file(path: &Path) -> std::io::Result<String> {
    let mut f = File::open(path)?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(hex(&h.finalize()))
}

/// `<first 12 hex of sha256(MANIFEST)>` or `not found`.
pub fn manifest_id(model: Option<&ModelDir>) -> String {
    model
        .and_then(|m| std::fs::read(m.manifest()).ok())
        .map(|b| hex(&Sha256::digest(&b))[..12].to_owned())
        .unwrap_or_else(|| "not found".to_owned())
}

pub fn version_line(model: Option<&ModelDir>) -> String {
    let sha = option_env!("ISOSYLL_BUILD_COMMIT")
        .filter(|s| !s.is_empty())
        .unwrap_or("unknown");
    format!(
        "isosyll {} ({sha}), model {}",
        env!("CARGO_PKG_VERSION"),
        manifest_id(model)
    )
}

/// One parsed `MANIFEST` line.
#[derive(Debug, PartialEq, Eq)]
pub struct ManifestEntry {
    pub sha256: String,
    pub path: String,
}

pub fn parse_manifest(text: &str) -> Result<Vec<ManifestEntry>, String> {
    let mut out = Vec::new();
    for (i, line) in text.lines().enumerate() {
        let line = line.trim_end();
        if line.trim().is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((sha, rel)) = line.split_once("  ") else {
            return Err(format!(
                "MANIFEST line {}: expected '<sha256>  <path>'",
                i + 1
            ));
        };
        let rel = rel.trim_start_matches(' ');
        if sha.len() != 64 || !sha.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err(format!("MANIFEST line {}: bad sha256", i + 1));
        }
        let p = Path::new(rel);
        if rel.is_empty()
            || p.is_absolute()
            || p.components().any(|c| !matches!(c, Component::Normal(_)))
        {
            return Err(format!("MANIFEST line {}: unsafe path '{rel}'", i + 1));
        }
        out.push(ManifestEntry {
            sha256: sha.to_ascii_lowercase(),
            path: rel.to_owned(),
        });
    }
    if out.is_empty() {
        return Err("MANIFEST lists no files".into());
    }
    Ok(out)
}

/// Verify every file in `MANIFEST`; returns `(ok, failed)` counts after calling `report`.
pub fn verify(
    model: &ModelDir,
    mut report: impl FnMut(&str, &str),
) -> Result<(usize, usize), String> {
    let mpath = model.manifest();
    let text = std::fs::read_to_string(&mpath)
        .map_err(|e| format!("{}: {e}; {INSTALL_HINT}", mpath.display()))?;
    let entries = parse_manifest(&text)?;
    let (mut ok, mut bad) = (0, 0);
    for e in entries {
        match sha256_file(&model.root.join(&e.path)) {
            Ok(h) if h == e.sha256 => {
                ok += 1;
                report("ok", &e.path);
            }
            Ok(_) => {
                bad += 1;
                report("MISMATCH", &e.path);
            }
            Err(err) => {
                bad += 1;
                report(&format!("UNREADABLE ({err})"), &e.path);
            }
        }
    }
    Ok((ok, bad))
}

#[cfg(test)]
#[path = "model_tests.rs"]
mod tests;
