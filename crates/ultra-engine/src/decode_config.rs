//! Shared, serde-serializable [`EffectiveDecodeConfig`] capturing the actual resolved runtime
//! decode policy across `ultra-engine`, `ultra-stream`, and `ultra-cli` (`F01` / `F10`).

use std::fs::File;
use std::io::Read;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{Engine, Model, Precision};

/// Canonical runtime decode policy resolved from an instantiated [`Engine`].
///
/// Every entry point (`Engine`, `NativeEngineBackend`, `NativeEngineSlotBank`, fleet runners,
/// live commit streams, and CLI tools) reports this structure and its deterministic
/// [`Self::policy_sha256`] so no caller can silently drop LM weights, beam width, chunk overrides,
/// or endpoint settings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EffectiveDecodeConfig {
    pub beam_width: usize,
    pub beam_max_symbols: usize,
    pub beam_margin: f32,
    pub chunk_override: Option<usize>,
    pub lm_path: Option<String>,
    pub lm_sha256: Option<String>,
    pub lm_weight: f32,
    pub lodr_path: Option<String>,
    pub lodr_sha256: Option<String>,
    pub lodr_weight: f32,
    pub blank_penalty: f32,
    pub tokenizer_sha256: String,
    pub model_weights_sha256: String,
    pub numerical_policy: String,
    pub live_only: bool,
    pub endpoint_policy: String,
    pub context_mode: String,
}

#[inline]
fn norm_f32(v: f32) -> f32 {
    if v == 0.0 {
        0.0
    } else {
        v
    }
}

fn sha256_file_hex(path: &std::path::Path) -> Option<String> {
    let mut f = File::open(path).ok()?;
    let mut h = Sha256::new();
    let mut buf = [0u8; 65_536];
    loop {
        let n = f.read(&mut buf).ok()?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Some(format!("{:x}", h.finalize()))
}

fn resolve_tokenizer_sha256(model: &Model) -> String {
    if !model.root.as_os_str().is_empty() {
        let tok_path = model.root.join("tokens.txt");
        if let Some(hex) = sha256_file_hex(&tok_path) {
            return hex;
        }
    }
    let mut h = Sha256::new();
    h.update((model.arch.vocab as u64).to_le_bytes());
    for id in 0..model.arch.vocab {
        let piece = model.tokens.decode_ids(&[id as i64]);
        h.update((id as u32).to_le_bytes());
        h.update((piece.len() as u32).to_le_bytes());
        h.update(piece.as_bytes());
    }
    format!("{:x}", h.finalize())
}

fn resolve_model_weights_sha256(model: &Model, precision: Precision) -> String {
    if precision == Precision::Int8 {
        if let Some(prep) = &model.int8_prepared {
            if !prep.sha256.is_empty() {
                return prep.sha256.clone();
            }
        }
    }
    if !model.arch.weights_sha256.is_empty() {
        return model.arch.weights_sha256.clone();
    }
    if !model.arch.checkpoint_sha256.is_empty() {
        return model.arch.checkpoint_sha256.clone();
    }
    let mut h = Sha256::new();
    h.update((model.arch.d as u64).to_le_bytes());
    h.update((model.arch.n_blocks as u64).to_le_bytes());
    h.update((model.arch.vocab as u64).to_le_bytes());
    h.update((model.arch.joiner as u64).to_le_bytes());
    h.update((model.arch.chunk as u64).to_le_bytes());
    h.update(precision.as_str().as_bytes());
    format!("{:x}", h.finalize())
}

fn resolve_numerical_policy(engine: &Engine) -> String {
    let mut s = match engine.precision {
        Precision::F32 => String::from("exact_f32_tanh_f32_acc"),
        Precision::Int8 => {
            if engine.act7 {
                String::from("exact_f32_tanh_int32_acc_act7")
            } else {
                String::from("exact_f32_tanh_int32_acc")
            }
        }
    };
    if engine.kv_int8 {
        s.push_str("_kv_int8");
    }
    s
}

fn resolve_endpoint_policy(engine: &Engine) -> String {
    if engine.endpoint_reset_ms == 0 {
        String::from("disabled")
    } else if engine.endpoint_replay_ms > 0 {
        format!(
            "reset_{}ms_replay_{}ms",
            engine.endpoint_reset_ms, engine.endpoint_replay_ms
        )
    } else {
        format!("reset_{}ms_once", engine.endpoint_reset_ms)
    }
}

fn resolve_context_mode(engine: &Engine) -> String {
    let a = &engine.model.arch;
    let family = if a.mid > 0 { "multirate" } else { "dense" };
    let kv = if engine.kv_int8 { "kv_i8" } else { "kv_f32" };
    format!(
        "streaming_{family}_left{}_chunk{}_maxchunks{}_{kv}",
        a.left, engine.chunk, engine.max_chunks
    )
}

impl EffectiveDecodeConfig {
    /// Resolve the actual runtime decode policy from an instantiated [`Engine`].
    pub fn from_engine(engine: &Engine) -> Self {
        let beam_max_symbols = if engine.uses_beam_state() {
            engine.beam_max_symbols
        } else {
            engine.model.arch.max_symbols
        };
        let chunk_override = (engine.chunk != engine.model.arch.chunk).then_some(engine.chunk);
        let (lm_path, lm_sha256, lm_weight, lodr_path, lodr_sha256, lodr_weight, blank_penalty) =
            match &engine.lm {
                Some((lm, cfg)) => {
                    let lm_p = lm.source_path().map(str::to_owned);
                    let lodr_p = lm
                        .lodr_source_path()
                        .map(str::to_owned)
                        .or_else(|| lm.has_lodr().then(|| String::from("embedded")));
                    (
                        lm_p,
                        Some(lm.lm_sha256()),
                        norm_f32(cfg.lm_weight),
                        lodr_p,
                        lm.lodr_sha256(),
                        norm_f32(cfg.lodr_weight),
                        norm_f32(cfg.blank_penalty),
                    )
                }
                None => (None, None, 0.0, None, None, 0.0, 0.0),
            };
        Self {
            beam_width: engine.beam,
            beam_max_symbols,
            beam_margin: norm_f32(engine.beam_margin),
            chunk_override,
            lm_path,
            lm_sha256,
            lm_weight,
            lodr_path,
            lodr_sha256,
            lodr_weight,
            blank_penalty,
            tokenizer_sha256: resolve_tokenizer_sha256(&engine.model),
            model_weights_sha256: resolve_model_weights_sha256(&engine.model, engine.precision),
            numerical_policy: resolve_numerical_policy(engine),
            live_only: engine.live_only,
            endpoint_policy: resolve_endpoint_policy(engine),
            context_mode: resolve_context_mode(engine),
        }
    }

    /// Return a copy with an explicit endpoint policy label (e.g. live-commit hangover policy).
    pub fn with_endpoint_policy(mut self, policy: impl Into<String>) -> Self {
        self.endpoint_policy = policy.into();
        self
    }

    /// Return a canonical normalized copy (`-0.0 -> 0.0`) suitable for deterministic hashing.
    fn normalized(&self) -> Self {
        Self {
            beam_margin: norm_f32(self.beam_margin),
            lm_weight: norm_f32(self.lm_weight),
            lodr_weight: norm_f32(self.lodr_weight),
            blank_penalty: norm_f32(self.blank_penalty),
            ..self.clone()
        }
    }

    /// Deterministic compact canonical JSON encoding of this decode policy.
    pub fn canonical_json(&self) -> String {
        serde_json::to_string(&self.normalized())
            .expect("EffectiveDecodeConfig serialization is infallible")
    }

    /// Deterministic lowercase hex SHA-256 digest of [`Self::canonical_json`].
    pub fn policy_sha256(&self) -> String {
        let json = self.canonical_json();
        format!("{:x}", Sha256::digest(json.as_bytes()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_cfg() -> EffectiveDecodeConfig {
        EffectiveDecodeConfig {
            beam_width: 4,
            beam_max_symbols: 2,
            beam_margin: 0.0,
            chunk_override: Some(4),
            lm_path: Some("m5.ulm".into()),
            lm_sha256: Some("a".repeat(64)),
            lm_weight: 0.8,
            lodr_path: Some("embedded".into()),
            lodr_sha256: Some("b".repeat(64)),
            lodr_weight: 0.6,
            blank_penalty: 0.0,
            tokenizer_sha256: "c".repeat(64),
            model_weights_sha256: "d".repeat(64),
            numerical_policy: "exact_f32_tanh_int32_acc".into(),
            live_only: true,
            endpoint_policy: "disabled".into(),
            context_mode: "streaming_dense_left16_chunk4_maxchunks1_kv_f32".into(),
        }
    }

    #[test]
    fn policy_sha256_is_deterministic_normalized_and_sensitive() {
        let a = sample_cfg();
        let mut b = sample_cfg();
        b.beam_margin = -0.0;
        b.blank_penalty = -0.0;
        assert_eq!(a.canonical_json(), b.canonical_json());
        assert_eq!(a.policy_sha256(), b.policy_sha256());
        assert_eq!(a.policy_sha256().len(), 64);

        let rt: EffectiveDecodeConfig =
            serde_json::from_str(&a.canonical_json()).expect("roundtrip");
        assert_eq!(rt, a);

        let mut c = sample_cfg();
        c.lm_weight = 0.75;
        assert_ne!(a.policy_sha256(), c.policy_sha256());

        let mut d = sample_cfg();
        d.beam_width = 1;
        assert_ne!(a.policy_sha256(), d.policy_sha256());

        let mut e = sample_cfg();
        e.live_only = false;
        assert_ne!(a.policy_sha256(), e.policy_sha256());
    }
}
