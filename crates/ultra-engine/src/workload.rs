//! Evaluation workloads: the first `limit` utterances of one npz shard, or a verified DEV
//! selection panel over several shards. Audio is read reference-blind; transcripts are read
//! only by [`Workload::references`], after inference.

use std::path::{Path, PathBuf};

use ultra_frontend::npz::{load_npz_audio_only, load_npz_references};

use crate::panel::Panel;

/// One utterance to decode.
pub struct Item {
    /// Index into `Workload::sources`.
    pub source: usize,
    /// Index inside that shard.
    pub index: usize,
    pub pcm: Vec<i16>,
}

/// A shard the workload reads, with its split name.
pub struct Source {
    pub split: String,
    pub npz: PathBuf,
}

pub struct Workload {
    pub sources: Vec<Source>,
    pub items: Vec<Item>,
    pub panel_manifest_sha256: Option<String>,
}

/// Reference fields of one item, in workload order.
pub struct ItemRef {
    pub split: String,
    pub uid: String,
    pub speaker: i64,
    pub text: String,
}

impl Workload {
    /// The first `limit` utterances of `npz`.
    pub fn first(npz: &Path, limit: usize) -> Result<Self, String> {
        let split = npz
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        let items = load_npz_audio_only(npz, 0, Some(limit))
            .map_err(|e| e.to_string())?
            .into_iter()
            .map(|u| Item {
                source: 0,
                index: u.index,
                pcm: u.pcm,
            })
            .collect();
        Ok(Self {
            sources: vec![Source {
                split,
                npz: npz.to_owned(),
            }],
            items,
            panel_manifest_sha256: None,
        })
    }

    /// Every utterance of a verified panel, shards at `<cache_dir>/<split>.npz` (each hash
    /// checked). Uids are mapped to shard indices through the shard's uid list; the
    /// transcripts parsed alongside are dropped here and re-read after inference.
    pub fn panel(panel: &Panel, cache_dir: &Path) -> Result<Self, String> {
        let (mut sources, mut items) = (Vec::new(), Vec::new());
        for (si, s) in panel.splits.iter().enumerate() {
            let npz = panel.verified_cache(cache_dir, si)?;
            let index_of: std::collections::HashMap<String, (usize, i64)> =
                load_npz_references(&npz, 0, None)
                    .map_err(|e| e.to_string())?
                    .into_iter()
                    .map(|r| (r.uid, (r.index, r.speaker)))
                    .collect();
            for uid in &s.uids {
                let &(index, spk) = index_of
                    .get(uid)
                    .ok_or_else(|| format!("panel uid {uid} not in {}", npz.display()))?;
                if Some(&spk) != s.speaker_of.get(uid) {
                    return Err(format!("panel speaker of {uid} differs from the shard"));
                }
                let mut u = load_npz_audio_only(&npz, index, Some(1)).map_err(|e| e.to_string())?;
                let u = u.pop().ok_or_else(|| format!("{uid}: no audio"))?;
                items.push(Item {
                    source: si,
                    index,
                    pcm: u.pcm,
                });
            }
            sources.push(Source {
                split: s.split.clone(),
                npz,
            });
        }
        Ok(Self {
            sources,
            items,
            panel_manifest_sha256: Some(panel.manifest_sha256.clone()),
        })
    }

    /// Transcripts, uids and speakers of every item (call after inference and timing).
    pub fn references(&self) -> Result<Vec<ItemRef>, String> {
        let mut per_source = Vec::new();
        for s in &self.sources {
            let refs = load_npz_references(&s.npz, 0, None).map_err(|e| e.to_string())?;
            per_source.push(refs);
        }
        self.items
            .iter()
            .map(|it| {
                let r = per_source[it.source]
                    .get(it.index)
                    .filter(|r| r.index == it.index)
                    .ok_or_else(|| format!("no reference for index {}", it.index))?;
                Ok(ItemRef {
                    split: self.sources[it.source].split.clone(),
                    uid: r.uid.clone(),
                    speaker: r.speaker,
                    text: r.text.clone(),
                })
            })
            .collect()
    }
}
