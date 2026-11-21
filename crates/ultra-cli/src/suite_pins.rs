//! Canonical candidate identity resolution (`S6`) and `CANDIDATES.lock` loaded-artifact pin
//! verification (`B4`, `FREEZE.md §2/§3`).
//!
//! Resolves every requested candidate upfront so:
//! - CLI aliases (`nemotron-560` vs `nemotron_en_q8_live_560`, `zipformer-int8` vs
//!   `zipformer_en_int8_320`, etc.) map to the exact same canonical loaded identity for `LOOKS.jsonl`;
//! - `--qualified` verifies that EVERY requested candidate has a pinned (non-training,
//!   non-research-unqualified) entry in `docs/ultra/CANDIDATES.lock` AND that the SHA-256 of the
//!   **actually loaded** runtime artifacts matches the pinned hashes.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use ultra_adapters::{NemotronChunkMode, NemotronConfig};
use ultra_engine::Precision;
use ultra_stream::OfflineModelKind;

use crate::runner::{resolve_bundle_or_zipformer, resolve_offline_config, resolve_r1_bundle};
use crate::suite_engine::resolve_r1_weights_dir;
use crate::suite_manifest::sha256_file;
use crate::{jsr_data_root, CliError};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CandidateKind {
    Nemotron {
        mode_ms: u32,
    },
    Engine {
        precision: String,
        prepared_int8: bool,
    },
    OrtR1 {
        precision: String,
    },
    Zipformer {
        precision: String,
    },
    Offline {
        candidate_id: String,
    },
}

#[derive(Debug, Clone)]
pub struct ResolvedCandidate {
    pub raw_arg: String,
    pub canonical_id: String,
    pub kind: CandidateKind,
    /// `(role_or_filename, path_on_disk, actual_sha256)` for every artifact loaded by this candidate.
    pub artifacts: Vec<(String, PathBuf, String)>,
}

pub fn is_nemotron_candidate(raw: &str) -> bool {
    matches!(
        raw,
        "nemotron-160" | "nemotron_en_q8_live_160" | "nemotron-560" | "nemotron_en_q8_live_560"
    )
}

fn base_name(p: &Path, fallback: &str) -> String {
    p.file_name()
        .and_then(|s| s.to_str())
        .unwrap_or(fallback)
        .to_owned()
}

/// Resolve a CLI candidate string into its canonical loaded-artifact identity (`S6` / `B4`).
pub fn resolve_candidate_identity(
    raw_cand: &str,
    precision_override: Option<&str>,
    threads: i32,
) -> Result<ResolvedCandidate, CliError> {
    if is_nemotron_candidate(raw_cand) {
        let (mode, mode_ms) = if raw_cand.contains("160") {
            (NemotronChunkMode::Live160, 160u32)
        } else {
            (NemotronChunkMode::Live560, 560u32)
        };
        let ncfg = NemotronConfig::default_h0(mode, PathBuf::new(), String::new());
        let (gguf_sha, bin_sha) = (
            sha256_file(&ncfg.model_path)?,
            sha256_file(&ncfg.binary_path)?,
        );
        let fname = base_name(&ncfg.model_path, "nemotron.gguf");
        let bname = base_name(&ncfg.binary_path, "nemo-speech");
        return Ok(ResolvedCandidate {
            raw_arg: raw_cand.to_owned(),
            canonical_id: format!(
                "nemotron-{mode_ms}:bin={}:gguf={}",
                &bin_sha[..16],
                &gguf_sha[..16]
            ),
            kind: CandidateKind::Nemotron { mode_ms },
            artifacts: vec![
                (fname, ncfg.model_path, gguf_sha),
                (bname, ncfg.binary_path, bin_sha),
            ],
        });
    }

    if let Some(eng_res) = resolve_r1_weights_dir(raw_cand, precision_override) {
        let (weights_dir, prec) = eng_res?;
        let arch_path = weights_dir.join("arch.json");
        let arch_sha = sha256_file(&arch_path)?;
        let int8_path = weights_dir.join("weights.int8.safetensors");
        let prepared_int8 = int8_path.exists();
        let (w_name, w_path) = if matches!(prec, Precision::Int8) && prepared_int8 {
            ("weights.int8.safetensors".to_owned(), int8_path)
        } else {
            (
                "weights.safetensors".to_owned(),
                weights_dir.join("weights.safetensors"),
            )
        };
        let w_sha = sha256_file(&w_path)?;
        return Ok(ResolvedCandidate {
            raw_arg: raw_cand.to_owned(),
            canonical_id: format!(
                "engine:{}:arch={}:weights={}",
                prec.as_str(),
                &arch_sha[..16],
                &w_sha[..16]
            ),
            kind: CandidateKind::Engine {
                precision: prec.as_str().to_owned(),
                prepared_int8,
            },
            artifacts: vec![
                ("arch.json".into(), arch_path, arch_sha),
                (w_name, w_path, w_sha),
            ],
        });
    }

    if let Some(r1_res) = resolve_r1_bundle(raw_cand, precision_override) {
        let (bundle_dir, prec) = r1_res?;
        // Verify every declared file in manifest.json (ONNX models, tokenizer, mel_fbank) via R1Bundle.
        let _verified = ultra_gp::R1Bundle::load(&bundle_dir, prec, true)
            .map_err(|e| CliError::Config(format!("ORT R1 bundle verify failed: {e}")))?;
        let manifest_path = bundle_dir.join("manifest.json");
        let manifest_sha = sha256_file(&manifest_path)?;
        return Ok(ResolvedCandidate {
            raw_arg: raw_cand.to_owned(),
            canonical_id: format!("ort-r1:{}:manifest={}", prec.as_str(), &manifest_sha[..16]),
            kind: CandidateKind::OrtR1 {
                precision: prec.as_str().to_owned(),
            },
            artifacts: vec![("manifest.json".into(), manifest_path, manifest_sha)],
        });
    }

    if let Some(z_res) = resolve_bundle_or_zipformer(raw_cand, precision_override, threads, 0.8) {
        let (_bundle_id, zcfg) = z_res?;
        let suffix = if zcfg.precision == "fp32" {
            "onnx"
        } else {
            "int8.onnx"
        };
        let enc_name = format!("encoder-epoch-99-avg-1-chunk-16-left-128.{suffix}");
        let dec_name = format!("decoder-epoch-99-avg-1-chunk-16-left-128.{suffix}");
        let join_name = format!("joiner-epoch-99-avg-1-chunk-16-left-128.{suffix}");
        let tok_name = "tokens.txt".to_owned();
        let mut artifacts = Vec::with_capacity(4);
        for name in [&enc_name, &dec_name, &join_name, &tok_name] {
            let p = zcfg.model_dir.join(name);
            let sha = sha256_file(&p)?;
            artifacts.push((name.clone(), p, sha));
        }
        let canonical_id = format!(
            "sherpa-zipformer:{}:enc={}:dec={}:join={}:tok={}",
            zcfg.precision,
            &artifacts[0].2[..12],
            &artifacts[1].2[..12],
            &artifacts[2].2[..12],
            &artifacts[3].2[..12],
        );
        return Ok(ResolvedCandidate {
            raw_arg: raw_cand.to_owned(),
            canonical_id,
            kind: CandidateKind::Zipformer {
                precision: zcfg.precision,
            },
            artifacts,
        });
    }

    let models_root = PathBuf::from(jsr_data_root()).join("models");
    let off = resolve_offline_config(raw_cand, &models_root, threads)?;
    let paths: Vec<PathBuf> = match &off.kind {
        OfflineModelKind::MoonshineV2 {
            encoder,
            merged_decoder,
            tokens,
        } => {
            vec![encoder.clone(), merged_decoder.clone(), tokens.clone()]
        }
        OfflineModelKind::MoonshineV1 {
            preprocessor,
            encoder,
            uncached_decoder,
            cached_decoder,
            tokens,
        } => vec![
            preprocessor.clone(),
            encoder.clone(),
            uncached_decoder.clone(),
            cached_decoder.clone(),
            tokens.clone(),
        ],
        OfflineModelKind::Whisper {
            encoder,
            decoder,
            tokens,
        } => {
            vec![encoder.clone(), decoder.clone(), tokens.clone()]
        }
    };
    let mut artifacts = Vec::with_capacity(paths.len());
    for p in paths {
        let name = p
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("artifact")
            .to_owned();
        let sha = sha256_file(&p)?;
        artifacts.push((name, p, sha));
    }
    let canonical_id = format!(
        "sherpa-offline:{}:a0={}:a1={}:tok={}",
        off.candidate_id,
        &artifacts[0].2[..12],
        &artifacts[1].2[..12],
        &artifacts[2].2[..12],
    );
    Ok(ResolvedCandidate {
        raw_arg: raw_cand.to_owned(),
        canonical_id,
        kind: CandidateKind::Offline {
            candidate_id: off.candidate_id,
        },
        artifacts,
    })
}

#[derive(Debug, Default, Clone)]
struct LockEntry {
    candidate_id: String,
    status: String,
    manifest: Option<String>,
    manifest_sha256: Option<String>,
    inline_hashes: BTreeMap<String, String>,
}

fn strip_yaml_comment(s: &str) -> &str {
    s.split('#')
        .next()
        .unwrap_or("")
        .trim()
        .trim_matches('"')
        .trim_matches('\'')
}

fn insert_known_sha_key(map: &mut BTreeMap<String, String>, key: &str, val: &str) -> bool {
    if val.len() != 64 {
        return false;
    }
    let mapped = match key {
        "arch_sha256" => "arch.json",
        "weights_int8_sha256" => "weights.int8.safetensors",
        "weights_sha256" => "weights.safetensors",
        "binary_sha256" | "runtime_binary_sha256" => "nemo-speech",
        _ => return false,
    };
    map.insert(mapped.into(), val.to_owned());
    true
}

fn parse_candidates_lock(raw: &str) -> Vec<LockEntry> {
    let mut entries = Vec::new();
    let mut cur: Option<LockEntry> = None;
    for line in raw.lines() {
        let t = line.trim();
        if t.is_empty() || t.starts_with('#') {
            continue;
        }
        if let Some(rest) = t.strip_prefix("- candidate_id:") {
            if let Some(e) = cur.take() {
                entries.push(e);
            }
            cur = Some(LockEntry {
                candidate_id: strip_yaml_comment(rest).to_owned(),
                ..LockEntry::default()
            });
            continue;
        }
        if t.starts_with("- note:") {
            if let Some(e) = cur.take() {
                entries.push(e);
            }
            continue;
        }
        let Some(ref mut e) = cur else { continue };
        if let Some((k, v)) = t.split_once(':') {
            let (key, val) = (k.trim(), strip_yaml_comment(v));
            match key {
                "status" => e.status = val.to_owned(),
                "manifest" => e.manifest = Some(val.to_owned()),
                "manifest_sha256" => e.manifest_sha256 = Some(val.to_owned()),
                _ if insert_known_sha_key(&mut e.inline_hashes, key, val) => {}
                _ if key.ends_with("_sha256") && val.len() == 64 => {
                    e.inline_hashes.insert(key.to_owned(), val.to_owned());
                }
                _ => {}
            }
        }
    }
    if let Some(e) = cur {
        entries.push(e);
    }
    entries
}

/// Parse one or more `---`-separated candidate YAML manifests into per-doc artifact sha maps.
fn parse_manifest_yaml_hashes(raw: &str) -> Vec<BTreeMap<String, String>> {
    let mut docs = Vec::new();
    for doc in raw.split("\n---") {
        let mut map = BTreeMap::new();
        let (mut cur_artifact, mut cur_tokenizer, mut in_artifacts) = (None, None, false);
        for line in doc.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }
            if in_artifacts && (line.len() - line.trim_start().len()) <= 2 {
                in_artifacts = false;
            }
            if trimmed == "artifacts:" {
                in_artifacts = true;
                continue;
            }
            let Some((k, v)) = trimmed.split_once(':') else {
                continue;
            };
            let (key, val) = (k.trim(), strip_yaml_comment(v));
            if in_artifacts && val.len() == 64 {
                map.insert(key.to_owned(), val.to_owned());
                continue;
            }
            match key {
                "artifact" => cur_artifact = Some(val.to_owned()),
                "artifact_sha256" => {
                    if let Some(name) = cur_artifact.as_ref() {
                        map.insert(name.clone(), val.to_owned());
                    }
                }
                "tokenizer" => cur_tokenizer = Some(val.to_owned()),
                "tokenizer_sha256" => {
                    if let Some(name) = cur_tokenizer.as_ref() {
                        map.insert(name.clone(), val.to_owned());
                    }
                }
                "manifest_sha256" if val.len() == 64 => {
                    map.insert("manifest.json".into(), val.to_owned());
                }
                _ => {
                    insert_known_sha_key(&mut map, key, val);
                }
            }
        }
        if !map.is_empty() {
            docs.push(map);
        }
    }
    docs
}

fn entry_status_allowed_for_qualified(status: &str) -> bool {
    let up = status.trim().to_ascii_uppercase();
    !up.is_empty()
        && !up.starts_with("TRAINING")
        && up != "RESEARCH_UNQUALIFIED"
        && up != "UNPINNED"
}

type PinnedHashSet = (String, BTreeMap<String, String>);

fn build_pinned_hash_sets(
    lock_path: &Path,
    repo_root: &Path,
) -> Result<Vec<PinnedHashSet>, CliError> {
    let raw = std::fs::read_to_string(lock_path).map_err(|e| {
        CliError::Config(format!(
            "--qualified requires `{}` to verify pinned candidate identities: {e}",
            lock_path.display()
        ))
    })?;
    let entries = parse_candidates_lock(&raw);
    let mut out = Vec::new();
    for entry in entries {
        if !entry_status_allowed_for_qualified(&entry.status) {
            continue;
        }
        if let Some(ref mf) = entry.manifest {
            let mf_path = if Path::new(mf).is_absolute() {
                PathBuf::from(mf)
            } else {
                repo_root.join(mf)
            };
            let want = entry.manifest_sha256.as_deref().ok_or_else(|| {
                CliError::Config(format!(
                    "CANDIDATES.lock entry `{}` has manifest `{mf}` without manifest_sha256",
                    entry.candidate_id
                ))
            })?;
            let got = sha256_file(&mf_path)?;
            if !got.eq_ignore_ascii_case(want) {
                return Err(CliError::Config(format!(
                    "CANDIDATES.lock manifest pin mismatch for `{mf}`: expected {want}, got {got}"
                )));
            }
            let yaml_raw = std::fs::read_to_string(&mf_path)?;
            for mut doc_map in parse_manifest_yaml_hashes(&yaml_raw) {
                for (k, v) in &entry.inline_hashes {
                    doc_map.entry(k.clone()).or_insert_with(|| v.clone());
                }
                out.push((entry.candidate_id.clone(), doc_map));
            }
        } else {
            let mut map = entry.inline_hashes.clone();
            if let Some(ref ms) = entry.manifest_sha256 {
                if ms.len() == 64 {
                    map.insert("manifest.json".into(), ms.clone());
                }
            }
            if !map.is_empty() {
                out.push((entry.candidate_id.clone(), map));
            }
        }
    }
    Ok(out)
}

/// Verify (`B4`, `FREEZE.md §3`) that every requested candidate in `resolved`:
/// 1. If `ultra-engine` INT8, uses a prepared `weights.int8.safetensors` artifact (never runtime
///    weight quantization);
/// 2. Has a matching pinned (non-training, non-research-unqualified) entry in `CANDIDATES.lock`;
/// 3. Has EVERY actually loaded artifact SHA-256 equal to the pinned hash in `CANDIDATES.lock`
///    (and its referenced YAML manifest, if any).
pub fn verify_loaded_candidates_against_lock(
    lock_path: &Path,
    repo_root: &Path,
    resolved: &[ResolvedCandidate],
) -> Result<(), CliError> {
    let pinned_sets = build_pinned_hash_sets(lock_path, repo_root)?;
    if pinned_sets.is_empty() {
        return Err(CliError::Config(format!(
            "no pinned candidate artifact hashes found in `{}` (B4)",
            lock_path.display()
        )));
    }
    for cand in resolved {
        if let CandidateKind::Engine {
            precision,
            prepared_int8,
        } = &cand.kind
        {
            if precision == "int8" && !prepared_int8 {
                return Err(CliError::Config(format!(
                    "--qualified refuses runtime weight quantisation for engine candidate `{}` (missing weights.int8.safetensors; FREEZE.md §3)",
                    cand.raw_arg
                )));
            }
        }
        if cand.artifacts.is_empty() {
            return Err(CliError::Config(format!(
                "candidate `{}` resolved no verifiable artifacts (B4)",
                cand.raw_arg
            )));
        }
        let matched = pinned_sets.iter().any(|(_id, pin_map)| {
            cand.artifacts.iter().all(|(name, _path, actual_sha)| {
                pin_map
                    .get(name)
                    .is_some_and(|want| want.eq_ignore_ascii_case(actual_sha))
            })
        });
        if !matched {
            let summary: Vec<String> = cand
                .artifacts
                .iter()
                .map(|(n, _, s)| format!("{n}={}", &s[..12]))
                .collect();
            return Err(CliError::Config(format!(
                "--qualified candidate `{}` (canonical `{}`) is not pinned in `{}` or loaded artifact SHA-256 differs [{}] (B4)",
                cand.raw_arg,
                cand.canonical_id,
                lock_path.display(),
                summary.join(", ")
            )));
        }
    }
    Ok(())
}
