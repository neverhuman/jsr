//! SHA-256 verified binary serialisation for `ultra.lm-ngram/v2` (bit-packed trie)
//! with full backward-compatible loading of `ultra.lm-ngram/v1`.

use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::path::Path;

use sha2::{Digest, Sha256};

use super::bitvec::BitVec;
use super::{LevelVals, NgramLm, TrieLevel, MAX_ORDER};
use crate::EngineError;

const MAGIC: &[u8; 8] = b"ULTRLM01";
const FLAG_QUANTIZED: u32 = 1 << 0;
const FLAG_HAS_LODR: u32 = 1 << 1;
const FLAG_HAS_EOS: u32 = 1 << 2;

pub(super) fn save(lm: &NgramLm, path: &Path) -> Result<(), EngineError> {
    save_version(lm, path, 2)
}

pub(super) fn save_v1(lm: &NgramLm, path: &Path) -> Result<(), EngineError> {
    save_version(lm, path, 1)
}

fn save_version(lm: &NgramLm, path: &Path, version: u32) -> Result<(), EngineError> {
    let mut payload = Vec::new();
    payload.extend_from_slice(MAGIC);
    write_u32(&mut payload, version);
    write_u32(&mut payload, lm.vocab as u32);
    write_u32(&mut payload, lm.order as u32);
    let flags: u32 = (u32::from(lm.quantized) * FLAG_QUANTIZED)
        | (u32::from(!lm.lodr_bigram.is_empty()) * FLAG_HAS_LODR)
        | (u32::from(lm.has_eos) * FLAG_HAS_EOS);
    write_u32(&mut payload, flags);
    write_f32(&mut payload, lm.max_lm_logp);

    for &v in &lm.uni_prob {
        write_f32(&mut payload, v);
    }
    for &v in &lm.uni_bow {
        write_f32(&mut payload, v);
    }
    for &s in &lm.seen {
        payload.push(u8::from(s));
    }
    write_u32(&mut payload, lm.uni_child_off.len() as u32);
    for &o in &lm.uni_child_off {
        write_u32(&mut payload, o);
    }
    for lv in &lm.levels {
        write_u32(&mut payload, lv.len() as u32);
        if version == 1 {
            for i in 0..lv.len() {
                write_u16(&mut payload, lv.tok.get_u16(i));
            }
        } else {
            write_bitvec(&mut payload, &lv.tok);
        }
        match &lv.vals {
            LevelVals::Q8 {
                prob_lut,
                bow_lut,
                q_prob,
                q_bow,
            } => {
                for &x in prob_lut.iter() {
                    write_f32(&mut payload, x);
                }
                for &x in bow_lut.iter() {
                    write_f32(&mut payload, x);
                }
                payload.extend_from_slice(q_prob);
                write_u32(&mut payload, q_bow.len() as u32);
                payload.extend_from_slice(q_bow);
            }
            LevelVals::F32 { prob, bow } => {
                for &x in prob {
                    write_f32(&mut payload, x);
                }
                write_u32(&mut payload, bow.len() as u32);
                for &x in bow {
                    write_f32(&mut payload, x);
                }
            }
        }
        write_u32(&mut payload, lv.child_off.len() as u32);
        if version == 1 {
            for i in 0..lv.child_off.len() {
                write_u32(&mut payload, lv.child_off.get_u32(i));
            }
        } else {
            write_bitvec(&mut payload, &lv.child_off);
        }
    }
    for &x in &lm.lodr_bigram {
        write_f32(&mut payload, x);
    }

    let digest = Sha256::digest(&payload);
    let mut w = BufWriter::new(File::create(path)?);
    w.write_all(&payload)?;
    w.write_all(&digest)?;
    w.flush()?;
    Ok(())
}

pub(super) fn load(path: &Path) -> Result<NgramLm, EngineError> {
    let mut raw = Vec::new();
    File::open(path)?.read_to_end(&mut raw)?;
    if raw.len() < 32 + 28 {
        return Err(EngineError::Format("LM file truncated".into()));
    }
    let (payload, tail) = raw.split_at(raw.len() - 32);
    if tail != Sha256::digest(payload).as_slice() {
        return Err(EngineError::Integrity(
            "LM file SHA-256 verification failed".into(),
        ));
    }

    let mut cur = Reader::new(payload);
    if cur.read_bytes(8)? != MAGIC {
        return Err(EngineError::Format("bad LM magic header".into()));
    }
    let version = cur.read_u32()?;
    if version != 1 && version != 2 {
        return Err(EngineError::Unsupported(format!(
            "unsupported LM binary version {version}"
        )));
    }
    let vocab = cur.read_u32()? as usize;
    let order = cur.read_u32()? as usize;
    if vocab == 0 || vocab >= usize::from(u16::MAX) || order == 0 || order > MAX_ORDER {
        return Err(EngineError::Format(format!(
            "invalid LM header vocab={vocab} order={order}"
        )));
    }
    let flags = cur.read_u32()?;
    let quantized = (flags & FLAG_QUANTIZED) != 0;
    let has_lodr = (flags & FLAG_HAS_LODR) != 0;
    let has_eos = (flags & FLAG_HAS_EOS) != 0;
    if (flags & !(FLAG_QUANTIZED | FLAG_HAS_LODR | FLAG_HAS_EOS)) != 0 {
        return Err(EngineError::Unsupported(format!(
            "unsupported LM flags {flags:#x}"
        )));
    }
    let max_lm_logp = cur.read_f32()?;
    if !max_lm_logp.is_finite() {
        return Err(EngineError::Format("non-finite max_lm_logp".into()));
    }

    let n_uni = vocab + 1 + usize::from(has_eos);
    let uni_prob = cur.read_f32_vec(n_uni)?;
    let uni_bow = cur.read_f32_vec(n_uni)?;
    let seen: Vec<bool> = cur.read_bytes(n_uni)?.iter().map(|&b| b != 0).collect();

    let uco_len = cur.read_u32()? as usize;
    if order > 1 && uco_len != n_uni + 1 {
        return Err(EngineError::Format(format!(
            "unexpected uni_child_off len {uco_len}"
        )));
    }
    let uni_child_off = cur.read_u32_vec(uco_len)?;

    let tok_bits_default = BitVec::bits_needed((vocab + usize::from(has_eos)) as u32);
    let mut levels = Vec::with_capacity(order.saturating_sub(1));
    for lv_idx in 0..order.saturating_sub(1) {
        let n = cur.read_u32()? as usize;
        let tok = if version == 1 {
            let v = cur.read_u16_vec(n)?;
            BitVec::from_u16_slice(&v, tok_bits_default)
        } else {
            read_bitvec(&mut cur, n)?
        };
        let vals = if quantized {
            let mut prob_lut = Box::new([0.0f32; 256]);
            let mut bow_lut = Box::new([0.0f32; 256]);
            prob_lut.copy_from_slice(&cur.read_f32_vec(256)?);
            bow_lut.copy_from_slice(&cur.read_f32_vec(256)?);
            let q_prob = cur.read_bytes(n)?.to_vec();
            let qb_len = cur.read_u32()? as usize;
            let q_bow = cur.read_bytes(qb_len)?.to_vec();
            LevelVals::Q8 {
                prob_lut,
                bow_lut,
                q_prob,
                q_bow,
            }
        } else {
            let prob = cur.read_f32_vec(n)?;
            let b_len = cur.read_u32()? as usize;
            let bow = cur.read_f32_vec(b_len)?;
            LevelVals::F32 { prob, bow }
        };
        let co_len = cur.read_u32()? as usize;
        let is_leaf = lv_idx + 1 == order.saturating_sub(1);
        if !is_leaf && co_len != n + 1 {
            return Err(EngineError::Format(format!(
                "level {} child_off len mismatch ({co_len} != {})",
                lv_idx + 2,
                n + 1
            )));
        }
        let child_off = if version == 1 {
            let v = cur.read_u32_vec(co_len)?;
            if v.is_empty() {
                BitVec::empty()
            } else {
                let max_off = *v.last().unwrap_or(&0);
                BitVec::from_u32_slice(&v, BitVec::bits_needed(max_off))
            }
        } else {
            read_bitvec(&mut cur, co_len)?
        };
        levels.push(TrieLevel {
            tok,
            vals,
            child_off,
        });
    }

    let lodr_bigram = if has_lodr {
        cur.read_f32_vec((vocab + 1) * vocab)?
    } else {
        Vec::new()
    };
    if cur.remaining() != 0 {
        return Err(EngineError::Format(format!(
            "trailing {} unexpected bytes in LM file",
            cur.remaining()
        )));
    }
    if order > 1 {
        validate_csr_slice(&uni_child_off, levels[0].len() as u32)?;
        for i in 0..levels.len().saturating_sub(1) {
            validate_csr_bitvec(&levels[i].child_off, levels[i + 1].len() as u32)?;
        }
    }
    Ok(NgramLm {
        vocab,
        order,
        quantized,
        has_eos,
        uni_prob,
        uni_bow,
        seen,
        uni_child_off,
        levels,
        lodr_bigram,
        max_lm_logp,
    })
}

fn validate_csr_slice(offs: &[u32], child_len: u32) -> Result<(), EngineError> {
    if offs.is_empty() || *offs.last().unwrap_or(&0) != child_len {
        return Err(EngineError::Format("CSR offset sentinel mismatch".into()));
    }
    for w in offs.windows(2) {
        if w[0] > w[1] {
            return Err(EngineError::Format("non-monotone CSR offsets".into()));
        }
    }
    Ok(())
}

fn validate_csr_bitvec(offs: &BitVec, child_len: u32) -> Result<(), EngineError> {
    if offs.is_empty() || offs.get_u32(offs.len() - 1) != child_len {
        return Err(EngineError::Format("CSR offset sentinel mismatch".into()));
    }
    let mut prev = offs.get_u32(0);
    for i in 1..offs.len() {
        let cur = offs.get_u32(i);
        if prev > cur {
            return Err(EngineError::Format("non-monotone CSR offsets".into()));
        }
        prev = cur;
    }
    Ok(())
}

fn write_bitvec(out: &mut Vec<u8>, bv: &BitVec) {
    out.push(bv.bits);
    let nw = bv.disk_words();
    write_u32(out, nw as u32);
    for &w in &bv.words[..nw] {
        out.extend_from_slice(&w.to_le_bytes());
    }
}

fn read_bitvec(cur: &mut Reader<'_>, len: usize) -> Result<BitVec, EngineError> {
    let bits = cur.read_bytes(1)?[0];
    let nw = cur.read_u32()? as usize;
    let words = cur.read_u64_vec(nw)?;
    BitVec::from_raw(bits, len, words).ok_or_else(|| {
        EngineError::Format(format!(
            "corrupt bit-packed vector bits={bits} len={len} words={nw}"
        ))
    })
}

fn write_u16(out: &mut Vec<u8>, v: u16) {
    out.extend_from_slice(&v.to_le_bytes());
}
fn write_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}
fn write_f32(out: &mut Vec<u8>, v: f32) {
    out.extend_from_slice(&v.to_le_bytes());
}

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }
    fn remaining(&self) -> usize {
        self.buf.len().saturating_sub(self.pos)
    }
    fn read_bytes(&mut self, n: usize) -> Result<&'a [u8], EngineError> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or_else(|| EngineError::Format("LM file offset overflow".into()))?;
        if end > self.buf.len() {
            return Err(EngineError::Format("unexpected EOF reading LM file".into()));
        }
        let s = &self.buf[self.pos..end];
        self.pos = end;
        Ok(s)
    }
    fn read_u32(&mut self) -> Result<u32, EngineError> {
        let b = self.read_bytes(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
    fn read_f32(&mut self) -> Result<f32, EngineError> {
        let b = self.read_bytes(4)?;
        Ok(f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
    fn read_u16_vec(&mut self, n: usize) -> Result<Vec<u16>, EngineError> {
        let b = self.read_bytes(n * 2)?;
        Ok(b.chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect())
    }
    fn read_u32_vec(&mut self, n: usize) -> Result<Vec<u32>, EngineError> {
        let b = self.read_bytes(n * 4)?;
        Ok(b.chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }
    fn read_u64_vec(&mut self, n: usize) -> Result<Vec<u64>, EngineError> {
        let b = self.read_bytes(n * 8)?;
        Ok(b.chunks_exact(8)
            .map(|c| u64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
            .collect())
    }
    fn read_f32_vec(&mut self, n: usize) -> Result<Vec<f32>, EngineError> {
        let b = self.read_bytes(n * 4)?;
        Ok(b.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }
}
