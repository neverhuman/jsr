//! Zero-dependency pure-Rust SentencePiece BPE and Unigram encoder for H45 CTC rescoring.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BinaryHeap};
use std::path::Path;

use ultra_core::receipt::sha256_hex;

use super::spm_proto::decode_model_proto;
use crate::EngineError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpmModelType {
    Unigram,
    Bpe,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CaseFold {
    None,
    Lower,
    Upper,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SpmTokenizer {
    model_type: SpmModelType,
    sha256: String,
    pieces_by_id: Vec<String>,
    piece_to_id_score: BTreeMap<String, (usize, f32)>,
    max_piece_bytes: usize,
    add_dummy_prefix: bool,
    escape_whitespaces: bool,
    fold_case: CaseFold,
}

#[derive(Clone, Copy)]
struct BpeCand {
    score: f32,
    left: usize,
    right: usize,
    merged_len: usize,
}

impl PartialEq for BpeCand {
    fn eq(&self, other: &Self) -> bool {
        self.score.to_bits() == other.score.to_bits() && self.left == other.left
    }
}
impl Eq for BpeCand {}
impl PartialOrd for BpeCand {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for BpeCand {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .total_cmp(&other.score)
            .then_with(|| other.left.cmp(&self.left))
    }
}

impl SpmTokenizer {
    pub fn from_file(path: &Path) -> Result<Self, EngineError> {
        Self::from_model_bytes(&std::fs::read(path)?)
    }

    pub fn from_model_bytes(bytes: &[u8]) -> Result<Self, EngineError> {
        let raw = decode_model_proto(bytes)?;
        let model_type = match raw.model_type_id {
            1 => SpmModelType::Unigram,
            2 => SpmModelType::Bpe,
            other => {
                return Err(EngineError::Unsupported(format!(
                    "unsupported SentencePiece model_type {other}"
                )))
            }
        };
        Self::from_parsed(
            model_type,
            sha256_hex(bytes),
            raw.pieces,
            raw.add_dummy_prefix,
            raw.escape_whitespaces,
        )
    }

    pub fn from_bpe_pieces(pieces: Vec<(String, f32)>) -> Result<Self, EngineError> {
        let raw = pieces.into_iter().map(|(p, s)| (p, s, 1u32)).collect();
        Self::from_parsed(SpmModelType::Bpe, String::new(), raw, true, true)
    }

    fn from_parsed(
        model_type: SpmModelType,
        sha256: String,
        raw: Vec<(String, f32, u32)>,
        add_dummy_prefix: bool,
        escape_whitespaces: bool,
    ) -> Result<Self, EngineError> {
        if raw.is_empty() {
            return Err(EngineError::Format(
                "SentencePiece model has no pieces".into(),
            ));
        }
        let (mut pieces_by_id, mut map) = (Vec::with_capacity(raw.len()), BTreeMap::new());
        let (mut max_piece_bytes, mut upper, mut lower) = (0usize, 0usize, 0usize);
        for (id, (piece, score, ptype)) in raw.into_iter().enumerate() {
            if ptype == 1 || ptype == 4 {
                upper += piece.chars().filter(|c| c.is_uppercase()).count();
                lower += piece.chars().filter(|c| c.is_lowercase()).count();
                max_piece_bytes = max_piece_bytes.max(piece.len());
                map.entry(piece.clone()).or_insert((id, score));
            }
            pieces_by_id.push(piece);
        }
        let fold_case = if upper > lower {
            CaseFold::Upper
        } else if lower > upper {
            CaseFold::Lower
        } else {
            CaseFold::None
        };
        Ok(Self {
            model_type,
            sha256,
            pieces_by_id,
            piece_to_id_score: map,
            max_piece_bytes,
            add_dummy_prefix,
            escape_whitespaces,
            fold_case,
        })
    }

    pub fn sha256(&self) -> &str {
        &self.sha256
    }
    pub fn num_pieces(&self) -> usize {
        self.pieces_by_id.len()
    }
    pub fn model_type(&self) -> SpmModelType {
        self.model_type
    }
    pub fn piece(&self, id: usize) -> Option<&str> {
        self.pieces_by_id.get(id).map(String::as_str)
    }

    fn normalize(&self, text: &str) -> String {
        let folded = match self.fold_case {
            CaseFold::Lower => text.to_lowercase(),
            CaseFold::Upper => text.to_uppercase(),
            CaseFold::None => text.to_owned(),
        };
        let words: Vec<&str> = folded.split_whitespace().collect();
        if words.is_empty() {
            return String::new();
        }
        let sep = if self.escape_whitespaces {
            "\u{2581}"
        } else {
            " "
        };
        if self.add_dummy_prefix {
            format!("{sep}{}", words.join(sep))
        } else {
            words.join(sep)
        }
    }

    pub fn encode_ids(&self, text: &str) -> Option<Vec<usize>> {
        let norm = self.normalize(text);
        if norm.is_empty() {
            return Some(Vec::new());
        }
        match self.model_type {
            SpmModelType::Bpe => self.encode_bpe(&norm),
            SpmModelType::Unigram => self.encode_unigram(&norm),
        }
    }

    pub fn encode_pieces(&self, text: &str) -> Option<Vec<&str>> {
        self.encode_ids(text)?
            .into_iter()
            .map(|id| self.piece(id))
            .collect()
    }

    fn encode_bpe(&self, norm: &str) -> Option<Vec<usize>> {
        let mut syms: Vec<String> = norm.chars().map(|c| c.to_string()).collect();
        let n = syms.len();
        let mut prev: Vec<Option<usize>> = (0..n).map(|i| i.checked_sub(1)).collect();
        let mut next: Vec<Option<usize>> = (0..n).map(|i| (i + 1 < n).then_some(i + 1)).collect();
        let mut alive = vec![true; n];
        let mut heap = BinaryHeap::new();
        let mut buf = String::new();
        let mut push = |l: usize, r: usize, syms: &[String], h: &mut BinaryHeap<BpeCand>| {
            buf.clear();
            buf.push_str(&syms[l]);
            buf.push_str(&syms[r]);
            if let Some(&(_, score)) = self.piece_to_id_score.get(buf.as_str()) {
                h.push(BpeCand {
                    score,
                    left: l,
                    right: r,
                    merged_len: buf.len(),
                });
            }
        };
        for i in 0..n.saturating_sub(1) {
            push(i, i + 1, &syms, &mut heap);
        }
        while let Some(c) = heap.pop() {
            if !alive[c.left]
                || !alive[c.right]
                || next[c.left] != Some(c.right)
                || syms[c.left].len() + syms[c.right].len() != c.merged_len
            {
                continue;
            }
            let rhs = std::mem::take(&mut syms[c.right]);
            syms[c.left].push_str(&rhs);
            alive[c.right] = false;
            let nr = next[c.right];
            next[c.left] = nr;
            if let Some(nri) = nr {
                prev[nri] = Some(c.left);
            }
            if let Some(pl) = prev[c.left] {
                push(pl, c.left, &syms, &mut heap);
            }
            if let Some(nri) = nr {
                push(c.left, nri, &syms, &mut heap);
            }
        }
        let (mut out, mut cur) = (Vec::new(), Some(0usize));
        while let Some(i) = cur {
            if alive[i] {
                out.push(self.piece_to_id_score.get(syms[i].as_str())?.0);
            }
            cur = next[i];
        }
        Some(out)
    }

    fn encode_unigram(&self, norm: &str) -> Option<Vec<usize>> {
        let n = norm.len();
        let (mut best, mut back) = (
            vec![f64::NEG_INFINITY; n + 1],
            vec![(0usize, 0usize); n + 1],
        );
        best[0] = 0.0;
        for (i, _) in norm.char_indices() {
            if !best[i].is_finite() {
                continue;
            }
            for end in (i + 1)..=(i + self.max_piece_bytes).min(n) {
                if norm.is_char_boundary(end) {
                    if let Some(&(id, sc)) = self.piece_to_id_score.get(&norm[i..end]) {
                        let cand = best[i] + f64::from(sc);
                        if cand > best[end] {
                            best[end] = cand;
                            back[end] = (i, id);
                        }
                    }
                }
            }
        }
        if !best[n].is_finite() {
            return None;
        }
        let (mut ids, mut cur) = (Vec::new(), n);
        while cur > 0 {
            let (p, id) = back[cur];
            ids.push(id);
            cur = p;
        }
        ids.reverse();
        Some(ids)
    }
}
