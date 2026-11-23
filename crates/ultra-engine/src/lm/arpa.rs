//! ARPA parsing, reverse-trie CSR construction, suffix closure, LODR bigram materialisation,
//! and 8-bit quantile quantisation (`-q 8 -b 8` density).

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use super::bitvec::BitVec;
use super::{LevelVals, NgramLm, TrieLevel, MAX_ORDER};
use crate::EngineError;

const LN_10: f32 = std::f32::consts::LN_10;

pub(super) fn from_arpa_str_with_eos(
    lm_arpa: &str,
    lodr_arpa: Option<&str>,
    vocab: usize,
    quantize_bits8: bool,
    include_eos: bool,
) -> Result<NgramLm, EngineError> {
    let lm_parsed = parse_arpa(lm_arpa.as_bytes(), vocab, include_eos)?;
    let lodr_table = match lodr_arpa {
        Some(s) => build_lodr_dense(&parse_arpa(s.as_bytes(), vocab, false)?, vocab),
        None => Vec::new(),
    };
    build_from_parsed(lm_parsed, lodr_table, vocab, quantize_bits8, include_eos)
}

pub(super) fn from_arpa_files_with_eos(
    lm_path: &Path,
    lodr_path: Option<&Path>,
    vocab: usize,
    quantize_bits8: bool,
    include_eos: bool,
) -> Result<NgramLm, EngineError> {
    let lm_parsed = parse_arpa(BufReader::new(File::open(lm_path)?), vocab, include_eos)?;
    let lodr_table = match lodr_path {
        Some(p) => build_lodr_dense(
            &parse_arpa(BufReader::new(File::open(p)?), vocab, false)?,
            vocab,
        ),
        None => Vec::new(),
    };
    build_from_parsed(lm_parsed, lodr_table, vocab, quantize_bits8, include_eos)
}

struct ParsedArpa {
    order: usize,
    uni_prob: Vec<f32>,
    uni_bow: Vec<f32>,
    seen: Vec<bool>,
    ngrams: Vec<Vec<([u16; MAX_ORDER], f32, f32)>>,
}

fn parse_arpa_token(tok: &str, vocab: usize, include_eos: bool) -> Option<u16> {
    if tok == "<unk>" || tok == "<UNK>" {
        return Some(0);
    }
    if tok == "<s>" {
        return Some(vocab as u16);
    }
    if tok == "</s>" {
        return include_eos.then_some((vocab + 1) as u16);
    }
    let id: usize = tok.parse().ok()?;
    (id >= 1 && id < vocab).then_some(id as u16)
}

fn parse_arpa<R: BufRead>(
    mut reader: R,
    vocab: usize,
    include_eos: bool,
) -> Result<ParsedArpa, EngineError> {
    let n_uni = vocab + 1 + usize::from(include_eos);
    let mut uni_prob = vec![-20.0f32; n_uni];
    let mut uni_bow = vec![0.0f32; n_uni];
    let mut seen = vec![false; n_uni];
    let mut ngrams: Vec<Vec<([u16; MAX_ORDER], f32, f32)>> = vec![Vec::new(); MAX_ORDER + 1];

    let mut section = 0usize;
    let mut max_order = 0usize;
    let mut line = String::new();
    while reader.read_line(&mut line)? > 0 {
        let s = line.trim();
        if s.is_empty() {
            line.clear();
            continue;
        }
        if s == "\\end\\" {
            break;
        }
        if s.starts_with('\\') && s.ends_with("-grams:") {
            let num_str = &s[1..s.len() - "-grams:".len()];
            section = num_str
                .parse::<usize>()
                .map_err(|e| EngineError::Format(format!("bad ARPA section {s}: {e}")))?;
            if section == 0 || section > MAX_ORDER {
                return Err(EngineError::Format(format!(
                    "unsupported ARPA order {section} (max {MAX_ORDER})"
                )));
            }
            max_order = max_order.max(section);
            line.clear();
            continue;
        }
        if section == 0 {
            line.clear();
            continue;
        }
        let mut parts = s.split_whitespace();
        let Some(p_str) = parts.next() else {
            line.clear();
            continue;
        };
        let prob = p_str
            .parse::<f32>()
            .map_err(|e| EngineError::Format(format!("bad ARPA prob {p_str}: {e}")))?
            * LN_10;

        let mut key = [0u16; MAX_ORDER];
        let mut skip = false;
        for slot in key.iter_mut().take(section) {
            let Some(tok) = parts.next() else {
                return Err(EngineError::Format(format!(
                    "short ARPA {section}-gram line: {s}"
                )));
            };
            match parse_arpa_token(tok, vocab, include_eos) {
                Some(id) => *slot = id,
                None => {
                    skip = true;
                    break;
                }
            }
        }
        if skip {
            line.clear();
            continue;
        }
        let bow = match parts.next() {
            Some(b_str) => {
                b_str
                    .parse::<f32>()
                    .map_err(|e| EngineError::Format(format!("bad ARPA bow {b_str}: {e}")))?
                    * LN_10
            }
            None => 0.0,
        };
        if section == 1 {
            let id = usize::from(key[0]);
            uni_prob[id] = prob;
            uni_bow[id] = bow;
            seen[id] = true;
        } else {
            ngrams[section].push((key, prob, bow));
        }
        line.clear();
    }
    if max_order == 0 {
        return Err(EngineError::Format("empty ARPA file (no n-grams)".into()));
    }
    Ok(ParsedArpa {
        order: max_order,
        uni_prob,
        uni_bow,
        seen,
        ngrams,
    })
}

fn build_from_parsed(
    mut parsed: ParsedArpa,
    lodr_bigram: Vec<f32>,
    vocab: usize,
    quantize_bits8: bool,
    include_eos: bool,
) -> Result<NgramLm, EngineError> {
    if vocab == 0 || vocab >= usize::from(u16::MAX) - 1 {
        return Err(EngineError::Format(format!(
            "vocab must be in 1..65534 (got {vocab})"
        )));
    }
    let order = parsed.order;
    if order == 0 || order > MAX_ORDER {
        return Err(EngineError::Format(format!(
            "ARPA order must be in 1..={MAX_ORDER} (got {order})"
        )));
    }
    ensure_suffix_closure(&mut parsed);

    let n_uni = vocab + 1 + usize::from(include_eos);
    let unk_prob = parsed.uni_prob[0];
    let mut uni_prob = vec![unk_prob; n_uni];
    let mut uni_bow = vec![0.0f32; n_uni];
    for i in 0..n_uni {
        if parsed.seen[i] {
            uni_prob[i] = parsed.uni_prob[i];
            uni_bow[i] = parsed.uni_bow[i];
        }
    }
    let mut max_lm_logp = f32::NEG_INFINITY;
    for (i, &p) in uni_prob.iter().enumerate() {
        if i < vocab && p.is_finite() {
            max_lm_logp = max_lm_logp.max(p);
        }
    }
    if order == 1 {
        return Ok(NgramLm {
            vocab,
            order,
            quantized: quantize_bits8,
            has_eos: include_eos,
            uni_prob,
            uni_bow,
            seen: parsed.seen,
            uni_child_off: Vec::new(),
            levels: Vec::new(),
            lodr_bigram,
            max_lm_logp: if max_lm_logp.is_finite() {
                max_lm_logp
            } else {
                0.0
            },
        });
    }

    let mut rev_levels: Vec<Vec<([u16; MAX_ORDER], f32, f32)>> = Vec::with_capacity(order - 1);
    for o in 2..=order {
        let mut entries = std::mem::take(&mut parsed.ngrams[o]);
        for (k, _, _) in &mut entries {
            k[..o].reverse();
        }
        entries.sort_unstable_by(|a, b| a.0[..o].cmp(&b.0[..o]));
        entries.dedup_by(|a, b| a.0[..o] == b.0[..o]);
        for &(k, p, _) in &entries {
            if usize::from(k[0]) < vocab && p.is_finite() {
                max_lm_logp = max_lm_logp.max(p);
            }
        }
        rev_levels.push(entries);
    }

    let mut uni_child_off = vec![0u32; n_uni + 1];
    let mut cur = 0usize;
    for (u, slot) in uni_child_off.iter_mut().enumerate().take(n_uni) {
        *slot = cur as u32;
        while cur < rev_levels[0].len() && usize::from(rev_levels[0][cur].0[0]) == u {
            cur += 1;
        }
    }
    uni_child_off[n_uni] = cur as u32;

    let mut levels = Vec::with_capacity(order - 1);
    for lv_idx in 0..(order - 1) {
        let o = lv_idx + 2;
        let cur_entries = &rev_levels[lv_idx];
        let is_leaf = lv_idx + 1 == order - 1;
        let mut tok = Vec::with_capacity(cur_entries.len());
        let mut probs = Vec::with_capacity(cur_entries.len());
        let mut bows = Vec::with_capacity(if is_leaf { 0 } else { cur_entries.len() });
        for &(key, p, b) in cur_entries {
            tok.push(key[o - 1]);
            probs.push(p);
            if !is_leaf {
                bows.push(b);
            }
        }
        let child_off = if is_leaf {
            Vec::new()
        } else {
            let next_entries = &rev_levels[lv_idx + 1];
            let mut offs = Vec::with_capacity(cur_entries.len() + 1);
            let mut nc = 0usize;
            for parent in cur_entries {
                offs.push(nc as u32);
                while nc < next_entries.len() && next_entries[nc].0[..o] == parent.0[..o] {
                    nc += 1;
                }
            }
            offs.push(nc as u32);
            if nc != next_entries.len() {
                return Err(EngineError::Format(format!(
                    "reverse-trie suffix closure violated at order {}: linked {} of {}",
                    o + 1,
                    nc,
                    next_entries.len()
                )));
            }
            offs
        };
        let vals = if quantize_bits8 {
            let (prob_lut, q_prob) = quantize_q8(&probs);
            for (idx, &q) in q_prob.iter().enumerate() {
                if usize::from(cur_entries[idx].0[0]) < vocab {
                    max_lm_logp = max_lm_logp.max(prob_lut[usize::from(q)]);
                }
            }
            let (bow_lut, q_bow) = if is_leaf {
                ([0.0f32; 256], Vec::new())
            } else {
                quantize_q8(&bows)
            };
            LevelVals::Q8 {
                prob_lut: Box::new(prob_lut),
                bow_lut: Box::new(bow_lut),
                q_prob,
                q_bow,
            }
        } else {
            LevelVals::F32 {
                prob: probs,
                bow: bows,
            }
        };
        let max_tok = (vocab + usize::from(include_eos)) as u32;
        let tok_bits = BitVec::bits_needed(max_tok);
        let tok_bv = BitVec::from_u16_slice(&tok, tok_bits);
        let child_off_bv = if child_off.is_empty() {
            BitVec::empty()
        } else {
            let max_off = *child_off.last().unwrap_or(&0);
            BitVec::from_u32_slice(&child_off, BitVec::bits_needed(max_off))
        };
        levels.push(TrieLevel {
            tok: tok_bv,
            vals,
            child_off: child_off_bv,
        });
    }
    Ok(NgramLm {
        vocab,
        order,
        quantized: quantize_bits8,
        has_eos: include_eos,
        uni_prob,
        uni_bow,
        seen: parsed.seen,
        uni_child_off,
        levels,
        lodr_bigram,
        max_lm_logp: if max_lm_logp.is_finite() {
            max_lm_logp
        } else {
            0.0
        },
    })
}

fn ensure_suffix_closure(parsed: &mut ParsedArpa) {
    if parsed.order < 3 {
        return;
    }
    for o in (3..=parsed.order).rev() {
        let mut existing: HashMap<Vec<u16>, (f32, f32)> =
            HashMap::with_capacity(parsed.ngrams[o - 1].len());
        for &(k, p, b) in &parsed.ngrams[o - 1] {
            existing.insert(k[..o - 1].to_vec(), (p, b));
        }
        let mut missing: Vec<Vec<u16>> = Vec::new();
        for &(k, _, _) in &parsed.ngrams[o] {
            if !existing.contains_key(&k[1..o]) {
                missing.push(k[1..o].to_vec());
            }
        }
        if missing.is_empty() {
            continue;
        }
        missing.sort();
        missing.dedup();
        let mut lower: Vec<HashMap<Vec<u16>, (f32, f32)>> = vec![HashMap::new(); o - 1];
        for lo in 2..(o - 1) {
            for &(k, p, b) in &parsed.ngrams[lo] {
                lower[lo].insert(k[..lo].to_vec(), (p, b));
            }
        }
        for suf in missing {
            let p = backoff_eval(&suf, &parsed.uni_prob, &parsed.uni_bow, &lower);
            let mut key = [0u16; MAX_ORDER];
            key[..o - 1].copy_from_slice(&suf);
            parsed.ngrams[o - 1].push((key, p, 0.0));
        }
    }
}

fn backoff_eval(
    seq: &[u16],
    uni_prob: &[f32],
    uni_bow: &[f32],
    lower: &[HashMap<Vec<u16>, (f32, f32)>],
) -> f32 {
    if seq.len() == 1 {
        return uni_prob[usize::from(seq[0])];
    }
    if seq.len() < lower.len() {
        if let Some(&(p, _)) = lower[seq.len()].get(seq) {
            return p;
        }
    }
    let ctx = &seq[..seq.len() - 1];
    let bow = if ctx.len() == 1 {
        uni_bow[usize::from(ctx[0])]
    } else if ctx.len() < lower.len() {
        lower[ctx.len()].get(ctx).map_or(0.0, |&(_, b)| b)
    } else {
        0.0
    };
    bow + backoff_eval(&seq[1..], uni_prob, uni_bow, lower)
}

fn build_lodr_dense(parsed: &ParsedArpa, vocab: usize) -> Vec<f32> {
    let n_uni = vocab + 1;
    let unk_p = parsed.uni_prob[0];
    let mut table = vec![0.0f32; n_uni * vocab];
    for prev in 0..n_uni {
        let bow = if parsed.seen[prev] {
            parsed.uni_bow[prev]
        } else {
            0.0
        };
        for k in 0..vocab {
            let uni = if parsed.seen[k] {
                parsed.uni_prob[k]
            } else {
                unk_p
            };
            table[prev * vocab + k] = bow + uni;
        }
    }
    if parsed.order >= 2 {
        for &(key, prob, _) in &parsed.ngrams[2] {
            let (prev, k) = (usize::from(key[0]), usize::from(key[1]));
            if prev < n_uni && k < vocab {
                table[prev * vocab + k] = prob;
            }
        }
    }
    table
}

fn quantize_q8(values: &[f32]) -> ([f32; 256], Vec<u8>) {
    let mut lut = [0.0f32; 256];
    if values.is_empty() {
        return (lut, Vec::new());
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable_by(f32::total_cmp);
    let n = sorted.len();
    for (b, slot) in lut.iter_mut().enumerate() {
        let lo = b * n / 256;
        let hi = ((b + 1) * n / 256).max(lo + 1).min(n);
        *slot = sorted[(lo + hi - 1) / 2];
    }
    let mut q = Vec::with_capacity(n);
    for &v in values {
        let pos = lut.partition_point(|&c| c < v);
        let idx = if pos == 0 {
            0
        } else if pos >= 256 {
            255
        } else if (v - lut[pos - 1]).abs() <= (lut[pos] - v).abs() {
            pos - 1
        } else {
            pos
        };
        q.push(idx as u8);
    }
    (lut, q)
}
