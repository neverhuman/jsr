//! Pure-Rust frozen `C4` evaluation slices (`docs/ultra/SLICES.md`, `python/ultra/ultra/slices.py`)
//! and `10-minute` non-speech silence probe (`corpora/probes/silence_noise_10min.wav`) rescored
//! exclusively through `ultra_eval` (`ultra-score` truth).

use std::path::Path;

use serde::{Deserialize, Serialize};
use ultra_eval::corpus::{pool, SilenceScore};
use ultra_eval::score::{score_records, ScoreRecord};
use ultra_eval::stream::{replay, TimedEvent};
use ultra_frontend::read_audio_16k_mono;
use ultra_stream::{
    AsrStream, Backend, OperatingProfile, OutputPolicy, PollResult, StreamClock, StreamConfig,
    StreamCoordinator,
};

use crate::suite_manifest::{sha256_file, write_scores_create_new};
use crate::CliError;

/// Frozen `long_form` duration threshold (`duration_s > 20.0`, `docs/ultra/SLICES.md`).
pub const LONG_FORM_S: f64 = 20.0;

/// Frozen `fast_speech` DEV p90 words-per-second threshold (`>= 3.5682`, `n = 5567`).
pub const FAST_SPEECH_WPS: f64 = 3.5682;

/// Frozen canonical SHA-256 of `corpora/probes/silence_noise_10min.wav`.
pub const SILENCE_PROBE_SHA256: &str =
    "8e43130d3f52446f000a3d76659e2ee53accd30d4e6cf9d0928ff7c5439dd666";

/// Pooled `ultra-score` summary over a frozen `C4` slice subset.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SliceMetric {
    pub utterances: usize,
    pub audio_seconds: f64,
    pub ref_words: u64,
    pub errors: u64,
    pub substitutions: u64,
    pub deletions: u64,
    pub insertions: u64,
    pub wer_pct: Option<f64>,
}

/// Frozen `C4` slice evaluations (`long_form` and `fast_speech`) for a candidate run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SliceSummaries {
    pub long_form_threshold_s: f64,
    pub fast_speech_threshold_wps: f64,
    pub long_form: SliceMetric,
    pub fast_speech: SliceMetric,
}

/// Non-speech silence probe report (`inserted_words_per_hour` & `false_episodes_per_hour`, never WER).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SilenceProbeReport {
    pub probe_path: String,
    pub probe_sha256: String,
    pub canonical_probe_verified: bool,
    pub episodes: u64,
    pub audio_seconds: f64,
    pub inserted_words: u64,
    pub inserted_words_per_hour: f64,
    pub false_episodes: u64,
    pub false_episodes_per_hour: f64,
    pub scores_file: Option<String>,
}

fn summarize_subset(subset: &[ScoreRecord]) -> SliceMetric {
    let audio_seconds: f64 = subset
        .iter()
        .filter_map(|r| r.duration_s)
        .filter(|d| d.is_finite() && *d >= 0.0)
        .sum();
    if subset.is_empty() {
        return SliceMetric {
            utterances: 0,
            audio_seconds: 0.0,
            ref_words: 0,
            errors: 0,
            substitutions: 0,
            deletions: 0,
            insertions: 0,
            wer_pct: None,
        };
    }
    let scored = score_records(subset);
    let mut ref_words = 0u64;
    let mut substitutions = 0u64;
    let mut deletions = 0u64;
    let mut insertions = 0u64;
    for u in &scored {
        if u.counts.ref_words > 0 {
            ref_words += u.counts.ref_words;
            substitutions += u.counts.substitutions;
            deletions += u.counts.deletions;
            insertions += u.counts.insertions;
        }
    }
    let errors = substitutions + deletions + insertions;
    let wer_pct = (ref_words > 0)
        .then(|| ((100.0 * (errors as f64) / (ref_words as f64)) * 1000.0).round() / 1000.0);
    SliceMetric {
        utterances: subset.len(),
        audio_seconds: (audio_seconds * 1000.0).round() / 1000.0,
        ref_words,
        errors,
        substitutions,
        deletions,
        insertions,
        wer_pct,
    }
}

/// Compute frozen `C4` `long_form` (`duration_s > 20.0`) and `fast_speech`
/// (`reference_words / duration_s >= 3.5682`) summaries matching `python/ultra/ultra/slices.py`.
pub fn compute_slice_summaries(records: &[ScoreRecord]) -> SliceSummaries {
    let mut long_form = Vec::new();
    let mut fast_speech = Vec::new();
    for r in records {
        let Some(sec) = r.duration_s.filter(|d| d.is_finite() && *d > 0.0) else {
            continue;
        };
        if sec > LONG_FORM_S {
            long_form.push(r.clone());
        }
        let raw_words = r.reference.split_whitespace().count() as f64;
        if (raw_words / sec.max(1e-9)) >= FAST_SPEECH_WPS {
            fast_speech.push(r.clone());
        }
    }
    SliceSummaries {
        long_form_threshold_s: LONG_FORM_S,
        fast_speech_threshold_wps: FAST_SPEECH_WPS,
        long_form: summarize_subset(&long_form),
        fast_speech: summarize_subset(&fast_speech),
    }
}

/// Run the `10-minute` (`30 s` live episode windows) non-speech silence probe on `backend` and
/// score via `ultra_eval::corpus::pool` (`SilenceScore`, zero WER denominator).
pub fn run_silence_probe_backend(
    backend: &dyn Backend,
    wav_path: &Path,
    require_canonical_sha: bool,
    scores_out: Option<&Path>,
) -> Result<SilenceProbeReport, CliError> {
    if !wav_path.is_file() {
        return Err(CliError::Unavailable(format!(
            "silence probe WAV `{}` not found",
            wav_path.display()
        )));
    }
    let probe_sha256 = sha256_file(wav_path)?;
    let canonical_probe_verified = probe_sha256 == SILENCE_PROBE_SHA256;
    if require_canonical_sha && !canonical_probe_verified {
        return Err(CliError::Config(format!(
            "silence probe `{}` sha256 {probe_sha256} != frozen {SILENCE_PROBE_SHA256}",
            wav_path.display()
        )));
    }
    let pcm = read_audio_16k_mono(wav_path)
        .map_err(|e| CliError::Config(format!("{}: {e}", wav_path.display())))?;
    if pcm.is_empty() {
        return Err(CliError::Config("empty silence probe WAV".into()));
    }

    let episode_samples = 30 * 16_000usize;
    let model_id = backend.capabilities().name;
    let mut records = Vec::new();

    for (ep_idx, ep_pcm) in pcm.chunks(episode_samples).enumerate() {
        let cfg = StreamConfig {
            stream_id: format!("sil-{ep_idx:04}"),
            model_config_id: model_id.clone(),
            policy: OutputPolicy::EditableDictation,
            profile: OperatingProfile::Efficient,
            segment_soft_cap_samples: (ep_pcm.len() as u64) + 16_000,
            segment_hard_cap_samples: (ep_pcm.len() as u64) + 32_000,
            ..StreamConfig::default()
        };
        let clk = StreamClock::real();
        let inner = backend.create_stream_with_clock(&cfg, clk.clone())?;
        let mut coord = StreamCoordinator::with_clock(cfg, inner, clk);
        let mut timed = Vec::new();
        for (k, pkt) in ep_pcm.chunks(320).enumerate() {
            coord.push_pcm((k * 320) as u64, pkt)?;
            while let PollResult::Event(ev) = coord.poll_event()? {
                timed.push(TimedEvent {
                    t_s: (ev.observed_monotonic_ns as f64) * 1e-9,
                    event: *ev,
                });
            }
        }
        coord.finish_input()?;
        while let PollResult::Event(ev) = coord.poll_event()? {
            timed.push(TimedEvent {
                t_s: (ev.observed_monotonic_ns as f64) * 1e-9,
                event: *ev,
            });
        }
        let sc = replay(&timed).map_err(|e| CliError::Internal(e.to_string()))?;
        records.push(ScoreRecord {
            id: format!("silence-ep-{ep_idx:04}"),
            speaker: "silence_probe".into(),
            dataset: "silence_probe".into(),
            reference: String::new(),
            hyp: sc.final_text,
            duration_s: Some((ep_pcm.len() as f64) / 16_000.0),
        });
    }

    let scores_file = match scores_out {
        Some(p) => {
            write_scores_create_new(p, &records)?;
            Some(p.to_string_lossy().into_owned())
        }
        None => None,
    };

    let utt_res = score_records(&records);
    let sil: SilenceScore = pool(&utt_res)
        .remove("silence_probe")
        .map(|d| d.silence)
        .unwrap_or(SilenceScore {
            utterances: 0,
            inserted_words: 0,
            false_transcript_utterances: 0,
            seconds: Some(0.0),
            insertions_per_hour: Some(0.0),
            false_transcripts_per_hour: Some(0.0),
        });
    let audio_seconds = sil.seconds.unwrap_or(0.0);
    Ok(SilenceProbeReport {
        probe_path: wav_path.to_string_lossy().into_owned(),
        probe_sha256,
        canonical_probe_verified,
        episodes: sil.utterances,
        audio_seconds: (audio_seconds * 1000.0).round() / 1000.0,
        inserted_words: sil.inserted_words,
        inserted_words_per_hour: (sil.insertions_per_hour.unwrap_or(0.0) * 1000.0).round() / 1000.0,
        false_episodes: sil.false_transcript_utterances,
        false_episodes_per_hour: (sil.false_transcripts_per_hour.unwrap_or(0.0) * 1000.0).round()
            / 1000.0,
        scores_file,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ultra_frontend::write_wav_pcm16;
    use ultra_stream::{FakeBackend, FakeBackendConfig};

    #[test]
    fn c4_slice_selection_and_silence_probe_match_frozen_rules() {
        let mk = |id: &str, spk: &str, r: &str, h: &str, d: f64| ScoreRecord {
            id: id.into(),
            speaker: spk.into(),
            dataset: "dev-clean".into(),
            reference: r.into(),
            hyp: h.into(),
            duration_s: Some(d),
        };
        let recs = vec![
            mk("u1-short-slow", "s1", "hello world", "hello world", 2.0),
            mk(
                "u2-long",
                "s1",
                "one two three four",
                "one two three five",
                20.5,
            ),
            mk("u3-fast", "s2", "a b c d", "a b c d", 1.0),
        ];
        let s = compute_slice_summaries(&recs);
        assert_eq!(s.long_form.utterances, 1);
        assert_eq!(s.long_form.wer_pct, Some(25.0));
        assert_eq!(s.fast_speech.utterances, 1);
        assert_eq!(s.fast_speech.wer_pct, Some(0.0));

        let tmp = tempfile::tempdir().expect("tempdir");
        let wav = tmp.path().join("sil.wav");
        write_wav_pcm16(&wav, 16_000, &[0i16; 16_000]).expect("write wav");
        let fake = FakeBackend::new(FakeBackendConfig {
            words: Vec::new(),
            chunk_samples: 320,
            ..FakeBackendConfig::default()
        });
        assert!(run_silence_probe_backend(&fake, &wav, true, None).is_err());
        let rep = run_silence_probe_backend(&fake, &wav, false, None).expect("probe ok");
        assert!(!rep.canonical_probe_verified);
        assert_eq!(rep.inserted_words, 0);
        assert_eq!(rep.false_episodes, 0);
    }
}
