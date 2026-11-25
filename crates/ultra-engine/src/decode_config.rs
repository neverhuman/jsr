//! Shared, serde-serializable [`EffectiveDecodeConfig`] capturing the actual resolved runtime
//! decode policy across `ultra-engine`, `ultra-stream`, and `ultra-cli` (`F01` / `F10`).

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::commit::SessionLexicalConfig;
use crate::{Engine, Model, Precision};

const SAMPLES_PER_ENC_FRAME: usize = 640;

/// Structured session-scoped lexical bias policy captured in [`EffectiveDecodeConfig`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SessionBiasPolicy {
    pub enabled: bool,
    pub min_occurrences: u32,
    pub max_entries: usize,
    pub bias_bonus: f32,
    pub min_word_chars: usize,
    pub min_bpe_pieces: usize,
}

impl Default for SessionBiasPolicy {
    fn default() -> Self {
        Self::from_config(SessionLexicalConfig::default())
    }
}

impl SessionBiasPolicy {
    pub fn from_config(cfg: SessionLexicalConfig) -> Self {
        Self {
            enabled: cfg.enabled,
            min_occurrences: cfg.min_occurrences,
            max_entries: cfg.max_entries,
            bias_bonus: norm_f32(cfg.bias_bonus),
            min_word_chars: cfg.min_word_chars,
            min_bpe_pieces: cfg.min_bpe_pieces,
        }
    }

    fn normalized(self) -> Self {
        Self {
            bias_bonus: norm_f32(self.bias_bonus),
            ..self
        }
    }
}

/// Canonical runtime decode policy resolved from an instantiated [`Engine`].
///
/// Every entry point (`Engine`, `NativeEngineBackend`, `NativeEngineSlotBank`, fleet runners,
/// live commit streams, and CLI tools) reports this structure and its deterministic
/// [`Self::policy_sha256`] so no caller can silently drop LM weights, beam width, chunk overrides,
/// frontend/tokenizer assets, tail-pad settings, endpoint policy, or session-bias settings.
///
/// Host-specific display paths (`lm_path`, `lodr_path`) are serialized in JSON receipts for
/// human diagnostics but **excluded** from [`Self::canonical_json`] and [`Self::policy_sha256`].
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
    pub bpe_sha256: Option<String>,
    pub frontend_fbank_sha256: String,
    pub frontend_policy: String,
    pub model_weights_sha256: String,
    pub numerical_policy: String,
    pub live_only: bool,
    pub endpoint_policy: String,
    pub tail_pad_samples: usize,
    pub tail_pad_frames: usize,
    pub tail_pad_policy: String,
    pub session_bias_policy: SessionBiasPolicy,
    pub context_mode: String,
    /// Z1 commit-pass RMS normalisation. Absent from the canonical policy when off, so Z1-off
    /// hashes are unchanged from releases that predate the flag.
    #[serde(default)]
    pub commit_rms_norm: bool,
}

/// Path-free canonical projection hashed by [`EffectiveDecodeConfig::policy_sha256`].
/// Excludes host-specific filesystem display paths (`lm_path`, `lodr_path`).
#[derive(Serialize)]
struct CanonicalDecodePolicy<'a> {
    beam_width: usize,
    beam_max_symbols: usize,
    beam_margin: f32,
    chunk_override: Option<usize>,
    lm_sha256: Option<&'a str>,
    lm_weight: f32,
    lodr_sha256: Option<&'a str>,
    lodr_weight: f32,
    blank_penalty: f32,
    tokenizer_sha256: &'a str,
    bpe_sha256: Option<&'a str>,
    frontend_fbank_sha256: &'a str,
    frontend_policy: &'a str,
    model_weights_sha256: &'a str,
    numerical_policy: &'a str,
    live_only: bool,
    endpoint_policy: &'a str,
    tail_pad_samples: usize,
    tail_pad_frames: usize,
    tail_pad_policy: &'a str,
    session_bias_policy: SessionBiasPolicy,
    context_mode: &'a str,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    commit_rms_norm: bool,
}

#[inline]
fn norm_f32(v: f32) -> f32 {
    if v == 0.0 {
        0.0
    } else {
        v
    }
}

fn resolve_tokenizer_sha256(model: &Model) -> String {
    if let Some(hex) = &model.assets.tokens_sha256 {
        if !hex.is_empty() {
            return hex.clone();
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

fn resolve_frontend_fbank_sha256(model: &Model) -> String {
    if let Some(hex) = &model.assets.mel_fbank_sha256 {
        if !hex.is_empty() {
            return hex.clone();
        }
    }
    let mut h = Sha256::new();
    for &w in model.mel_cfg.fbank() {
        h.update(w.to_le_bytes());
    }
    format!("{:x}", h.finalize())
}

fn resolve_frontend_policy(model: &Model) -> String {
    let mut h = Sha256::new();
    for &m in model.mel_cfg.cmvn.mean() {
        h.update(norm_f32(m).to_le_bytes());
    }
    for &s in model.mel_cfg.cmvn.istd() {
        h.update(norm_f32(s).to_le_bytes());
    }
    let cmvn_sha256 = format!("{:x}", h.finalize());
    format!(
        "trainer_log_mel_sr16000_mels{}_win400_hop160_fft512_hann_dc1_dither0_preemph0_logfloor{:e}_portable{}_cmvn{}",
        model.arch.n_mels,
        norm_f32(model.mel_cfg.log_floor),
        u8::from(model.mel_cfg.is_portable_math()),
        cmvn_sha256
    )
}

fn resolve_tail_pad(samples: usize) -> (usize, String) {
    let frames = samples / SAMPLES_PER_ENC_FRAME;
    let policy = if samples == 0 {
        String::from("eof_flush_no_pad")
    } else {
        format!("eof_zero_pcm_{samples}samples_{frames}frames")
    };
    (frames, policy)
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
        let (tail_pad_frames, tail_pad_policy) = resolve_tail_pad(engine.tail_pad_samples);
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
            bpe_sha256: engine.model.assets.bpe_sha256.clone(),
            frontend_fbank_sha256: resolve_frontend_fbank_sha256(&engine.model),
            frontend_policy: resolve_frontend_policy(&engine.model),
            model_weights_sha256: resolve_model_weights_sha256(&engine.model, engine.precision),
            numerical_policy: resolve_numerical_policy(engine),
            live_only: engine.live_only,
            endpoint_policy: resolve_endpoint_policy(engine),
            tail_pad_samples: engine.tail_pad_samples,
            tail_pad_frames,
            tail_pad_policy,
            session_bias_policy: SessionBiasPolicy::default(),
            context_mode: resolve_context_mode(engine),
            commit_rms_norm: false,
        }
    }

    /// Return a copy with an explicit endpoint policy label (e.g. live-commit hangover policy).
    pub fn with_endpoint_policy(mut self, policy: impl Into<String>) -> Self {
        self.endpoint_policy = policy.into();
        self
    }

    /// Return a copy with an explicit EOF tail-pad sample count (e.g. commit pass `tail_pad_samples`).
    pub fn with_tail_pad_samples(mut self, samples: usize) -> Self {
        let (frames, policy) = resolve_tail_pad(samples);
        self.tail_pad_samples = samples;
        self.tail_pad_frames = frames;
        self.tail_pad_policy = policy;
        self
    }

    /// Return a copy recording whether the commit pass applies Z1 RMS normalisation.
    pub fn with_commit_rms_norm(mut self, on: bool) -> Self {
        self.commit_rms_norm = on;
        self
    }

    /// Return a copy with an explicit [`SessionBiasPolicy`].
    pub fn with_session_bias_policy(mut self, policy: SessionBiasPolicy) -> Self {
        self.session_bias_policy = policy.normalized();
        self
    }

    /// Return a copy reflecting an active [`SessionLexicalConfig`].
    pub fn with_session_bias_config(self, cfg: SessionLexicalConfig) -> Self {
        self.with_session_bias_policy(SessionBiasPolicy::from_config(cfg))
    }

    /// Deterministic compact canonical JSON encoding of the path-free decode policy
    /// (excludes diagnostic display paths `lm_path` and `lodr_path`).
    pub fn canonical_json(&self) -> String {
        let canonical = CanonicalDecodePolicy {
            beam_width: self.beam_width,
            beam_max_symbols: self.beam_max_symbols,
            beam_margin: norm_f32(self.beam_margin),
            chunk_override: self.chunk_override,
            lm_sha256: self.lm_sha256.as_deref(),
            lm_weight: norm_f32(self.lm_weight),
            lodr_sha256: self.lodr_sha256.as_deref(),
            lodr_weight: norm_f32(self.lodr_weight),
            blank_penalty: norm_f32(self.blank_penalty),
            tokenizer_sha256: &self.tokenizer_sha256,
            bpe_sha256: self.bpe_sha256.as_deref(),
            frontend_fbank_sha256: &self.frontend_fbank_sha256,
            frontend_policy: &self.frontend_policy,
            model_weights_sha256: &self.model_weights_sha256,
            numerical_policy: &self.numerical_policy,
            live_only: self.live_only,
            endpoint_policy: &self.endpoint_policy,
            tail_pad_samples: self.tail_pad_samples,
            tail_pad_frames: self.tail_pad_frames,
            tail_pad_policy: &self.tail_pad_policy,
            session_bias_policy: self.session_bias_policy.normalized(),
            context_mode: &self.context_mode,
            commit_rms_norm: self.commit_rms_norm,
        };
        serde_json::to_string(&canonical)
            .expect("CanonicalDecodePolicy serialization is infallible")
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
            lm_path: Some("/staging/dir_a/m5.ulm".into()),
            lm_sha256: Some("a".repeat(64)),
            lm_weight: 0.8,
            lodr_path: Some("/staging/dir_a/lodr.arpa".into()),
            lodr_sha256: Some("b".repeat(64)),
            lodr_weight: 0.6,
            blank_penalty: 0.0,
            tokenizer_sha256: "c".repeat(64),
            bpe_sha256: Some("e".repeat(64)),
            frontend_fbank_sha256: "f".repeat(64),
            frontend_policy: "trainer_log_mel_sr16000_mels80_win400_hop160_fft512".into(),
            model_weights_sha256: "d".repeat(64),
            numerical_policy: "exact_f32_tanh_int32_acc".into(),
            live_only: true,
            endpoint_policy: "disabled".into(),
            tail_pad_samples: 0,
            tail_pad_frames: 0,
            tail_pad_policy: "eof_flush_no_pad".into(),
            session_bias_policy: SessionBiasPolicy::default(),
            context_mode: "streaming_dense_left16_chunk4_maxchunks1_kv_f32".into(),
            commit_rms_norm: false,
        }
    }

    #[test]
    fn policy_sha256_is_deterministic_path_independent_and_sensitive() {
        let a = sample_cfg();
        let mut b = sample_cfg();
        b.beam_margin = -0.0;
        b.blank_penalty = -0.0;
        b.session_bias_policy.bias_bonus = -0.0;
        // Changing host display paths must NOT change canonical_json or policy_sha256.
        b.lm_path = Some("/other/host/dir_b/m5.ulm".into());
        b.lodr_path = Some("embedded".into());
        assert_eq!(a.canonical_json(), b.canonical_json());
        assert_eq!(a.policy_sha256(), b.policy_sha256());
        assert_eq!(a.policy_sha256().len(), 64);
        assert!(!a.canonical_json().contains("lm_path"));
        assert!(!a.canonical_json().contains("lodr_path"));

        // Full serde serialization retains diagnostic display paths and round-trips cleanly.
        let full_json = serde_json::to_string(&a).expect("serialize");
        assert!(full_json.contains("/staging/dir_a/m5.ulm"));
        let rt: EffectiveDecodeConfig = serde_json::from_str(&full_json).expect("roundtrip");
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

        let mut f = sample_cfg();
        f.frontend_fbank_sha256 = "0".repeat(64);
        assert_ne!(a.policy_sha256(), f.policy_sha256());

        let g = sample_cfg().with_tail_pad_samples(5_120);
        assert_eq!(g.tail_pad_frames, 8);
        assert_ne!(a.policy_sha256(), g.policy_sha256());

        // Z1: on differs from off; off is byte-identical to a pre-Z1 canonical policy.
        let z = sample_cfg().with_commit_rms_norm(true);
        assert_ne!(a.policy_sha256(), z.policy_sha256());
        assert!(!a.canonical_json().contains("commit_rms_norm"));
        assert!(z.canonical_json().contains("\"commit_rms_norm\":true"));
        assert_eq!(
            a.policy_sha256(),
            z.with_commit_rms_norm(false).policy_sha256()
        );

        let h = sample_cfg().with_session_bias_config(SessionLexicalConfig {
            enabled: true,
            bias_bonus: 0.35,
            ..SessionLexicalConfig::default()
        });
        assert_ne!(a.policy_sha256(), h.policy_sha256());
    }
}
