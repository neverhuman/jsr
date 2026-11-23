//! Separate CTC vocabulary binding for H45 CTC rescoring.

use std::collections::BTreeMap;
use std::path::Path;

use super::spm::SpmTokenizer;
use crate::EngineError;

#[derive(Debug, Clone, PartialEq)]
enum VocabBackend {
    Spm(SpmTokenizer),
    ExactPieces {
        uses_sp_marker: bool,
        pieces: BTreeMap<String, usize>,
        max_piece_bytes: usize,
    },
}

/// Separate CTC vocabulary used to retokenize candidate text before CTC forward scoring.
#[derive(Debug, Clone, PartialEq)]
pub struct CtcVocab {
    blank_id: usize,
    vocab_size: usize,
    backend: VocabBackend,
}

impl CtcVocab {
    /// Bind the checkpoint's exact SentencePiece `.model` (`ModelProto`) for CTC rescoring.
    pub fn from_spm_model_file(path: &Path, blank_id: usize) -> Result<Self, EngineError> {
        Self::from_spm(SpmTokenizer::from_file(path)?, blank_id)
    }

    pub fn from_spm_model_bytes(bytes: &[u8], blank_id: usize) -> Result<Self, EngineError> {
        Self::from_spm(SpmTokenizer::from_model_bytes(bytes)?, blank_id)
    }

    pub fn from_bpe_pieces(
        blank_id: usize,
        pieces: Vec<(String, f32)>,
    ) -> Result<Self, EngineError> {
        Self::from_spm(SpmTokenizer::from_bpe_pieces(pieces)?, blank_id)
    }

    fn from_spm(spm: SpmTokenizer, blank_id: usize) -> Result<Self, EngineError> {
        let vocab_size = spm.num_pieces() + 1;
        if blank_id >= vocab_size {
            return Err(EngineError::Format(format!(
                "invalid CTC blank_id {blank_id} for {vocab_size} outputs"
            )));
        }
        Ok(Self {
            blank_id,
            vocab_size,
            backend: VocabBackend::Spm(spm),
        })
    }

    /// Simple exact-piece / char-level CTC vocabulary for synthetic fixtures.
    pub fn new(blank_id: usize, tokens: Vec<String>) -> Result<Self, EngineError> {
        let vocab_size = tokens.len();
        if vocab_size == 0 || blank_id >= vocab_size {
            return Err(EngineError::Format(format!(
                "invalid CTC vocab: blank_id {blank_id} for {vocab_size} entries"
            )));
        }
        let (mut pieces, mut max_piece_bytes, mut uses_sp_marker) =
            (BTreeMap::new(), 0usize, false);
        for (id, piece) in tokens.into_iter().enumerate() {
            if id == blank_id || piece.is_empty() {
                continue;
            }
            uses_sp_marker |= piece.contains('\u{2581}');
            max_piece_bytes = max_piece_bytes.max(piece.len());
            if pieces.insert(piece, id).is_some() {
                return Err(EngineError::Format(format!(
                    "duplicate non-blank CTC piece at id {id}"
                )));
            }
        }
        Ok(Self {
            blank_id,
            vocab_size,
            backend: VocabBackend::ExactPieces {
                uses_sp_marker,
                pieces,
                max_piece_bytes,
            },
        })
    }

    pub fn blank_id(&self) -> usize {
        self.blank_id
    }
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }
    pub fn is_exact_spm(&self) -> bool {
        matches!(self.backend, VocabBackend::Spm(_))
    }
    pub fn model_sha256(&self) -> Option<&str> {
        match &self.backend {
            VocabBackend::Spm(s) if !s.sha256().is_empty() => Some(s.sha256()),
            _ => None,
        }
    }

    pub fn retokenize_pieces(&self, text: &str) -> Option<Vec<&str>> {
        match &self.backend {
            VocabBackend::Spm(spm) => spm.encode_pieces(text),
            VocabBackend::ExactPieces { .. } => None,
        }
    }

    pub fn retokenize_text(&self, text: &str) -> Option<Vec<usize>> {
        match &self.backend {
            VocabBackend::Spm(spm) => {
                let b = self.blank_id;
                let ids = spm.encode_ids(text)?;
                Some(
                    ids.into_iter()
                        .map(|id| if id < b { id } else { id + 1 })
                        .collect(),
                )
            }
            VocabBackend::ExactPieces {
                uses_sp_marker,
                pieces,
                max_piece_bytes,
            } => {
                let norm: Vec<&str> = text.split_whitespace().collect();
                if norm.is_empty() {
                    return Some(Vec::new());
                }
                let encoded = if *uses_sp_marker {
                    format!("\u{2581}{}", norm.join("\u{2581}"))
                } else {
                    norm.join(" ")
                };
                let (mut out, mut pos) = (Vec::new(), 0usize);
                while pos < encoded.len() {
                    let mut end = (pos + *max_piece_bytes).min(encoded.len());
                    let mut matched = None;
                    while end > pos {
                        if encoded.is_char_boundary(end) {
                            if let Some(&id) = pieces.get(&encoded[pos..end]) {
                                matched = Some((id, end));
                                break;
                            }
                        }
                        end -= 1;
                    }
                    let (id, next_pos) = matched?;
                    out.push(id);
                    pos = next_pos;
                }
                Some(out)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn commit_ctc_bpe_merge_order_matches_spm_not_longest_piece() {
        let raw: &[(&str, f32)] = &[
            ("<unk>", 0.0),
            ("\u{2581}t", 0.0),
            ("\u{2581}f", -17.0),
            ("\u{2581}th", -28.0),
            ("re", -8.0),
            ("at", -18.0),
            ("ed", -19.0),
            ("en", -21.0),
            ("ve", -55.0),
            ("red", -114.0),
            ("\u{2581}fe", -171.0),
            ("reat", -231.0),
            ("ver", -240.0),
            ("ate", -250.0),
            ("ned", -310.0),
            ("\u{2581}thr", -400.0),
            ("ened", -475.0),
            ("\u{2581}", -995.0),
            ("a", -996.0),
            ("d", -997.0),
            ("e", -998.0),
            ("f", -999.0),
            ("h", -1000.0),
            ("n", -1001.0),
            ("r", -1002.0),
            ("t", -1003.0),
            ("v", -1004.0),
        ];
        let pieces = raw.iter().map(|&(p, s)| (p.to_owned(), s)).collect();
        let vocab = CtcVocab::from_bpe_pieces(0, pieces).expect("bpe");
        assert!(vocab.is_exact_spm());
        assert_eq!(
            vocab.retokenize_pieces("fevered").expect("fev"),
            vec!["\u{2581}fe", "ve", "red"]
        );
        assert_eq!(
            vocab.retokenize_pieces("threatened").expect("thr"),
            vec!["\u{2581}th", "reat", "ened"]
        );
    }
}
