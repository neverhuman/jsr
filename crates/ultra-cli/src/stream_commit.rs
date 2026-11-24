//! Opt-in `--commit-pass` and `--session-lexical-bias` CLI flags for `ultra stream` (Wave 1D).

use std::path::PathBuf;
use std::sync::Arc;

use ultra_engine::commit::{
    CommitConfig, CommitLmConfig, SessionLexicalConfig, DEFAULT_MAX_PCM_SAMPLES,
    DEFAULT_SAMPLE_RATE,
};
use ultra_engine::{LmConfig, NgramLm};
use ultra_stream::{LiveCommitConfig, NativeEngineLoadConfig};

use crate::CliError;

/// Parsed opt-in live decode, commit-pass, and session lexical bias CLI options for `ultra stream`.
#[derive(Debug, Clone)]
pub struct StreamCommitCliOptions {
    pub beam: usize,
    pub beam_max_symbols: usize,
    pub beam_margin: f32,
    pub chunk: Option<usize>,
    pub lm_path: Option<PathBuf>,
    pub lm_weight: f32,
    pub lodr_weight: f32,
    pub blank_penalty: f32,
    pub enabled: bool,
    pub hangover_ms: u32,
    pub preroll_ms: u32,
    pub deadline_ms: u32,
    pub commit_chunk: usize,
    pub tail_pad_ms: usize,
    pub commit_lm: Option<PathBuf>,
    pub commit_beam: Option<usize>,
    pub lm_scale: f32,
    pub lodr_scale: f32,
    pub max_audio_s: f64,
    pub session_lexical_bias: bool,
    pub session_bias_bonus: f32,
}

impl Default for StreamCommitCliOptions {
    fn default() -> Self {
        Self {
            beam: 1,
            beam_max_symbols: 1,
            beam_margin: 0.0,
            chunk: None,
            lm_path: None,
            lm_weight: 0.0,
            lodr_weight: 0.0,
            blank_penalty: 0.0,
            enabled: false,
            hangover_ms: 300,
            preroll_ms: 300,
            deadline_ms: 800,
            commit_chunk: 16,
            tail_pad_ms: 320,
            commit_lm: None,
            commit_beam: None,
            lm_scale: 0.80,
            lodr_scale: 0.60,
            max_audio_s: DEFAULT_MAX_PCM_SAMPLES as f64 / DEFAULT_SAMPLE_RATE as f64,
            session_lexical_bias: false,
            session_bias_bonus: 0.35,
        }
    }
}

impl StreamCommitCliOptions {
    /// Attempt to parse one decode (`--beam`, `--lm`, `--chunk`), `--commit-*`, or `--session-*`
    /// flag at `args[*i]`. Returns `Ok(true)` if consumed, `Ok(false)` if unrecognized.
    pub fn try_parse_flag(&mut self, args: &[String], i: &mut usize) -> Result<bool, CliError> {
        let flag = args[*i].as_str();
        let next = |idx: &mut usize, name: &str| -> Result<&str, CliError> {
            *idx += 1;
            args.get(*idx)
                .map(String::as_str)
                .ok_or_else(|| CliError::Config(format!("missing value for {name}")))
        };
        let p_err = |f: &str| CliError::Config(format!("invalid {f}"));
        match flag {
            "--beam" => self.beam = next(i, flag)?.parse().map_err(|_| p_err(flag))?,
            "--beam-max-symbols" => {
                self.beam_max_symbols = next(i, flag)?.parse().map_err(|_| p_err(flag))?;
            }
            "--beam-margin" => {
                self.beam_margin = next(i, flag)?.parse().map_err(|_| p_err(flag))?;
            }
            "--chunk" => {
                self.chunk = Some(next(i, flag)?.parse().map_err(|_| p_err(flag))?);
            }
            "--lm" => self.lm_path = Some(PathBuf::from(next(i, flag)?)),
            "--lm-weight" => self.lm_weight = next(i, flag)?.parse().map_err(|_| p_err(flag))?,
            "--lodr-weight" => {
                self.lodr_weight = next(i, flag)?.parse().map_err(|_| p_err(flag))?;
            }
            "--blank-penalty" => {
                self.blank_penalty = next(i, flag)?.parse().map_err(|_| p_err(flag))?;
            }
            "--commit-pass" => self.enabled = true,
            "--commit-hangover-ms" => {
                self.hangover_ms = next(i, flag)?.parse().map_err(|_| p_err(flag))?;
            }
            "--commit-preroll-ms" => {
                self.preroll_ms = next(i, flag)?.parse().map_err(|_| p_err(flag))?;
            }
            "--commit-deadline-ms" => {
                self.deadline_ms = next(i, flag)?.parse().map_err(|_| p_err(flag))?;
            }
            "--commit-chunk" => {
                self.commit_chunk = next(i, flag)?.parse().map_err(|_| p_err(flag))?;
            }
            "--commit-tail-pad-ms" => {
                self.tail_pad_ms = next(i, flag)?.parse().map_err(|_| p_err(flag))?;
            }
            "--commit-lm" => self.commit_lm = Some(PathBuf::from(next(i, flag)?)),
            "--commit-beam" => {
                self.commit_beam = Some(next(i, flag)?.parse().map_err(|_| p_err(flag))?);
            }
            "--commit-lm-scale" => {
                self.lm_scale = next(i, flag)?.parse().map_err(|_| p_err(flag))?;
            }
            "--commit-lodr-scale" => {
                self.lodr_scale = next(i, flag)?.parse().map_err(|_| p_err(flag))?;
            }
            "--commit-max-audio-s" | "--commit-max-segment-s" => {
                self.max_audio_s = next(i, flag)?.parse().map_err(|_| p_err(flag))?;
            }
            "--session-lexical-bias" => self.session_lexical_bias = true,
            "--session-bias-bonus" => {
                self.session_bias_bonus = next(i, flag)?.parse().map_err(|_| p_err(flag))?;
            }
            _ => return Ok(false),
        }
        Ok(true)
    }

    /// Resolve a shared [`NativeEngineLoadConfig`] so CLI streaming never silently drops beam/LM.
    pub fn to_native_load_config(&self) -> Result<NativeEngineLoadConfig, CliError> {
        if !(1..=64).contains(&self.beam) {
            return Err(CliError::Config(format!(
                "--beam must be in 1..=64 (got {})",
                self.beam
            )));
        }
        if !(1..=8).contains(&self.beam_max_symbols) {
            return Err(CliError::Config(format!(
                "--beam-max-symbols must be in 1..=8 (got {})",
                self.beam_max_symbols
            )));
        }
        if !self.beam_margin.is_finite() || self.beam_margin < 0.0 {
            return Err(CliError::Config(
                "--beam-margin must be finite and >= 0".into(),
            ));
        }
        let lm_cfg = LmConfig {
            lm_weight: self.lm_weight,
            lodr_weight: self.lodr_weight,
            blank_penalty: self.blank_penalty,
        };
        lm_cfg
            .validate()
            .map_err(|e| CliError::Config(e.to_string()))?;
        let lm = match &self.lm_path {
            Some(p) => {
                let loaded = NgramLm::load(p)
                    .map_err(|e| CliError::Unavailable(format!("--lm {}: {e}", p.display())))?;
                Some((Arc::new(loaded), lm_cfg))
            }
            None => {
                if self.lm_weight != 0.0 || self.lodr_weight != 0.0 || self.blank_penalty != 0.0 {
                    return Err(CliError::Config(
                        "--lm-weight, --lodr-weight, and --blank-penalty require --lm".into(),
                    ));
                }
                None
            }
        };
        Ok(NativeEngineLoadConfig {
            max_chunks: 1,
            chunk: self.chunk,
            beam: self.beam,
            beam_max_symbols: self.beam_max_symbols,
            beam_margin: self.beam_margin,
            lm,
            ..NativeEngineLoadConfig::default()
        })
    }

    pub fn to_live_commit_config(&self) -> Result<LiveCommitConfig, CliError> {
        if !self.max_audio_s.is_finite() || self.max_audio_s <= 0.0 || self.max_audio_s > 3600.0 {
            return Err(CliError::Config(
                "--commit-max-audio-s must be finite in (0, 3600]".into(),
            ));
        }
        let beam = self
            .commit_beam
            .unwrap_or(if self.commit_lm.is_some() { 16 } else { 1 });
        let max_pcm_samples = (self.max_audio_s * DEFAULT_SAMPLE_RATE as f64).round() as usize;
        let commit_lm = self
            .commit_lm
            .as_ref()
            .map(|p| {
                CommitLmConfig::from_file(p, beam, self.lm_scale, self.lodr_scale, 0.0, 1)
                    .map_err(|e| CliError::Unavailable(format!("--commit-lm: {e}")))
            })
            .transpose()?;
        let session_bias = SessionLexicalConfig {
            enabled: self.session_lexical_bias,
            bias_bonus: if self.session_lexical_bias {
                self.session_bias_bonus
            } else {
                0.0
            },
            ..SessionLexicalConfig::default()
        };
        let cfg = LiveCommitConfig {
            commit: CommitConfig {
                commit_chunk: self.commit_chunk,
                tail_pad_samples: self.tail_pad_ms * DEFAULT_SAMPLE_RATE / 1000,
                max_pcm_samples,
                beam_size: beam,
                ctc_weight: 0.0,
                mbr_enabled: false,
                mbr_temp: 1.0,
            },
            hangover_ms: self.hangover_ms,
            preroll_ms: self.preroll_ms,
            deadline_ms: self.deadline_ms,
            commit_lm,
            session_bias,
        };
        cfg.validate().map_err(CliError::from)?;
        Ok(cfg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn commit_cli_options_default_off_and_parse_release_flags() {
        let def = StreamCommitCliOptions::default();
        assert!(!def.enabled && !def.session_lexical_bias);
        assert_eq!(
            (
                def.hangover_ms,
                def.preroll_ms,
                def.commit_chunk,
                def.tail_pad_ms
            ),
            (300, 300, 16, 320)
        );
        assert!((def.max_audio_s - 40.0).abs() < 1e-9);

        let args: Vec<String> = "--commit-pass --commit-hangover-ms 800 --commit-preroll-ms 450 --commit-chunk 16 --commit-tail-pad-ms 320 --commit-beam 16 --commit-lm-scale 0.80 --commit-lodr-scale 0.60 --commit-max-audio-s 40.0 --session-lexical-bias --session-bias-bonus 0.4"
            .split_whitespace()
            .map(String::from)
            .collect();
        let mut opts = StreamCommitCliOptions::default();
        let mut i = 0;
        while i < args.len() {
            assert!(opts.try_parse_flag(&args, &mut i).expect("parse"));
            i += 1;
        }
        assert!(opts.enabled && opts.session_lexical_bias);
        let live = opts.to_live_commit_config().expect("live cfg");
        assert_eq!(
            (
                live.hangover_ms,
                live.preroll_ms,
                live.commit.commit_chunk,
                live.commit.beam_size
            ),
            (800, 450, 16, 16)
        );
        assert_eq!(live.commit.max_pcm_samples, 640_000);
        assert!((live.session_bias.bias_bonus - 0.4).abs() < 1e-6);
    }
}
