//! CLI argument parsing and manifest loading for `ultra-engine-commit`.

use std::path::{Path, PathBuf};

use serde_json::Value;
use ultra_engine::commit::{
    DEFAULT_MAX_PCM_SAMPLES, DEFAULT_RESCORE_M5_SUB, DEFAULT_RESCORE_WEIGHT, DEFAULT_SAMPLE_RATE,
};
use ultra_engine::panel::Panel;
use ultra_engine::workload::Workload;
use ultra_engine::Precision;
use ultra_frontend::npz::{load_npz_audio_only, load_npz_references};
use ultra_frontend::read_audio_16k_mono;

pub(super) struct Args {
    pub(super) model: PathBuf,
    pub(super) manifest: PathBuf,
    pub(super) out: PathBuf,
    pub(super) commit_chunk: usize,
    pub(super) tail_pad_ms: usize,
    pub(super) commit_max_audio_s: f64,
    pub(super) threads: usize,
    pub(super) precision: Precision,
    pub(super) limit: usize,
    pub(super) cache_dir: Option<PathBuf>,
    pub(super) commit_lm: Option<PathBuf>,
    pub(super) commit_beam: Option<usize>,
    pub(super) lm_scale: f32,
    pub(super) lodr_scale: f32,
    pub(super) token_bonus: f32,
    pub(super) max_sym_per_frame: usize,
    pub(super) ctc_weight: f32,
    pub(super) mbr: bool,
    pub(super) mbr_temp: f32,
    pub(super) rescore_l2r: Option<PathBuf>,
    pub(super) rescore_r2l: Option<PathBuf>,
    pub(super) rescore_weight: f32,
    pub(super) rescore_m5_sub: f32,
    pub(super) rescore_r2l_top_k: Option<usize>,
    pub(super) rescore_skip_margin: Option<f32>,
    pub(super) rescore_act7: bool,
    /// Z1: whole-utterance RMS gain before the commit pass. Off by default.
    pub(super) rms_norm: bool,
}

pub(super) fn parse_args<I: IntoIterator<Item = String>>(args: I) -> Result<Args, String> {
    let mut it = args.into_iter();
    let mut a = Args {
        model: PathBuf::new(),
        manifest: PathBuf::new(),
        out: PathBuf::new(),
        commit_chunk: 16,
        tail_pad_ms: 320,
        commit_max_audio_s: DEFAULT_MAX_PCM_SAMPLES as f64 / DEFAULT_SAMPLE_RATE as f64,
        threads: 1,
        precision: Precision::Int8,
        limit: 0,
        cache_dir: None,
        commit_lm: None,
        commit_beam: None,
        lm_scale: 0.8,
        lodr_scale: 0.6,
        token_bonus: 0.0,
        max_sym_per_frame: 1,
        ctc_weight: 0.0,
        mbr: false,
        mbr_temp: 1.0,
        rescore_l2r: None,
        rescore_r2l: None,
        rescore_weight: DEFAULT_RESCORE_WEIGHT,
        rescore_m5_sub: DEFAULT_RESCORE_M5_SUB,
        rescore_r2l_top_k: None,
        rescore_skip_margin: None,
        rescore_act7: false,
        rms_norm: false,
    };
    while let Some(k) = it.next() {
        let mut v = || it.next().ok_or_else(|| format!("{k} needs a value"));
        match k.as_str() {
            "--model" | "--weights" => a.model = v()?.into(),
            "--manifest" => a.manifest = v()?.into(),
            "--out" => a.out = v()?.into(),
            "--commit-chunk" => a.commit_chunk = v()?.parse().map_err(|e| format!("{k}: {e}"))?,
            "--tail-pad-ms" => a.tail_pad_ms = v()?.parse().map_err(|e| format!("{k}: {e}"))?,
            "--commit-max-audio-s" | "--commit-max-segment-s" => {
                a.commit_max_audio_s = v()?.parse().map_err(|e| format!("{k}: {e}"))?
            }
            "--threads" => a.threads = v()?.parse().map_err(|e| format!("{k}: {e}"))?,
            "--precision" => a.precision = Precision::parse(&v()?).map_err(|e| e.to_string())?,
            "--limit" => a.limit = v()?.parse().map_err(|e| format!("{k}: {e}"))?,
            "--cache-dir" => a.cache_dir = Some(v()?.into()),
            "--commit-lm" => a.commit_lm = Some(v()?.into()),
            "--commit-beam" => a.commit_beam = Some(v()?.parse().map_err(|e| format!("{k}: {e}"))?),
            "--lm-scale" => a.lm_scale = v()?.parse().map_err(|e| format!("{k}: {e}"))?,
            "--lodr-scale" => a.lodr_scale = v()?.parse().map_err(|e| format!("{k}: {e}"))?,
            "--token-bonus" => a.token_bonus = v()?.parse().map_err(|e| format!("{k}: {e}"))?,
            "--max-sym-per-frame" => {
                a.max_sym_per_frame = v()?.parse().map_err(|e| format!("{k}: {e}"))?
            }
            "--ctc-weight" => a.ctc_weight = v()?.parse().map_err(|e| format!("{k}: {e}"))?,
            // Z1: whole-utterance RMS gain before the commit re-encode (default off). Only
            // `committed` is normalised; `provisional`/`revised` stay on the raw audio.
            "--mbr" => a.mbr = true,
            "--rms-norm" => a.rms_norm = true,
            "--mbr-temp" => a.mbr_temp = v()?.parse().map_err(|e| format!("{k}: {e}"))?,
            "--rescore-l2r" => a.rescore_l2r = Some(v()?.into()),
            "--rescore-r2l" => a.rescore_r2l = Some(v()?.into()),
            "--rescore-weight" => {
                a.rescore_weight = v()?.parse().map_err(|e| format!("{k}: {e}"))?
            }
            "--rescore-skip-margin" => {
                a.rescore_skip_margin = Some(v()?.parse().map_err(|e| format!("{k}: {e}"))?)
            }
            "--rescore-act7" => a.rescore_act7 = true,
            "--rescore-r2l-top-k" => {
                a.rescore_r2l_top_k = Some(v()?.parse().map_err(|e| format!("{k}: {e}"))?)
            }
            "--rescore-m5-sub" => {
                a.rescore_m5_sub = v()?.parse().map_err(|e| format!("{k}: {e}"))?
            }
            o => return Err(format!("unknown argument {o}")),
        }
    }
    if a.model.as_os_str().is_empty()
        || a.manifest.as_os_str().is_empty()
        || a.out.as_os_str().is_empty()
    {
        return Err("--model, --manifest, and --out are required".into());
    }
    if a.threads == 0 {
        return Err("--threads must be positive".into());
    }
    if !a.commit_max_audio_s.is_finite()
        || a.commit_max_audio_s <= 0.0
        || (a.commit_max_audio_s * DEFAULT_SAMPLE_RATE as f64).round() < 1.0
        || a.commit_max_audio_s > 3600.0
    {
        return Err("--commit-max-audio-s must be finite and in (0, 3600]".into());
    }
    if !a.ctc_weight.is_finite() || a.ctc_weight < 0.0 {
        return Err("--ctc-weight must be finite and >= 0".into());
    }
    if !a.mbr_temp.is_finite() || a.mbr_temp <= 0.0 {
        return Err("--mbr-temp must be finite and > 0".into());
    }
    Ok(a)
}

pub(super) enum ManifestStream {
    Paths(std::vec::IntoIter<(String, PathBuf)>),
    Memory(std::vec::IntoIter<(String, Vec<i16>)>),
}

impl ManifestStream {
    pub(super) fn len(&self) -> usize {
        match self {
            Self::Paths(it) => it.len(),
            Self::Memory(it) => it.len(),
        }
    }

    pub(super) fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub(super) fn next_item(&mut self) -> Option<Result<(String, Vec<i16>), String>> {
        match self {
            Self::Paths(it) => it.next().map(|(id, p)| {
                read_audio_16k_mono(&p)
                    .map(|pcm| (id, pcm))
                    .map_err(|e| format!("{}: {e}", p.display()))
            }),
            Self::Memory(it) => it.next().map(Ok),
        }
    }
}

pub(super) fn open_manifest(a: &Args) -> Result<ManifestStream, String> {
    let path = &a.manifest;
    let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("");
    let lim = (a.limit > 0).then_some(a.limit);
    if ext.eq_ignore_ascii_case("npz") {
        let au = load_npz_audio_only(path, 0, lim).map_err(|e| e.to_string())?;
        let rf = load_npz_references(path, 0, lim).map_err(|e| e.to_string())?;
        let items: Vec<(String, Vec<i16>)> = au
            .into_iter()
            .zip(rf)
            .map(|(x, r)| (r.uid, x.pcm))
            .collect();
        return Ok(ManifestStream::Memory(items.into_iter()));
    }
    if ext.eq_ignore_ascii_case("json") {
        let cache = a.cache_dir.clone().unwrap_or_else(|| {
            PathBuf::from(std::env::var("JSR_DATA").unwrap_or("/mnt/xbabe3-external/jsr".into()))
                .join("cache/librispeech_pcm")
        });
        let wl = Workload::panel(&Panel::read(path)?, &cache)?;
        let rf = wl.references()?;
        let take = lim.unwrap_or(wl.items.len()).min(wl.items.len());
        let items: Vec<(String, Vec<i16>)> = wl
            .items
            .into_iter()
            .zip(rf)
            .take(take)
            .map(|(i, r)| (r.uid, i.pcm))
            .collect();
        return Ok(ManifestStream::Memory(items.into_iter()));
    }
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let base = path.parent().unwrap_or_else(|| Path::new("."));
    let mut entries: Vec<(String, PathBuf)> = Vec::new();
    for (idx, line) in text.lines().map(str::trim).enumerate() {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if lim.is_some_and(|n| entries.len() >= n) {
            break;
        }
        let (uid, ap) = if line.starts_with('{') {
            let v: Value =
                serde_json::from_str(line).map_err(|e| format!("line {}: {e}", idx + 1))?;
            let uid = v["uid"]
                .as_str()
                .or_else(|| v["id"].as_str())
                .map(str::to_owned);
            let ap = v["wav"]
                .as_str()
                .or_else(|| v["audio_filepath"].as_str())
                .or_else(|| v["path"].as_str())
                .or_else(|| v["file"].as_str())
                .ok_or_else(|| format!("line {}: missing wav/path", idx + 1))?;
            (uid, ap.to_owned())
        } else if let Some((u, p)) = line.split_once('\t') {
            (Some(u.trim().to_owned()), p.trim().to_owned())
        } else {
            (None, line.to_owned())
        };
        let p = if Path::new(&ap).is_absolute() {
            PathBuf::from(&ap)
        } else {
            base.join(&ap)
        };
        let id = uid.unwrap_or_else(|| {
            p.file_stem().map_or_else(
                || format!("utt_{:05}", entries.len()),
                |s| s.to_string_lossy().into_owned(),
            )
        });
        entries.push((id, p));
    }
    Ok(ManifestStream::Paths(entries.into_iter()))
}
