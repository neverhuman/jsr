//! Prefix/suffix trie builder and row-major INT8 packing helpers for `ultra.tf-rescore/v1`.

use crate::kernels::NR;
use crate::quant::{PackedI8, KG};
use crate::EngineError;

use super::tf_rescore_io::TfDirection;

/// Reusable linked-list prefix/suffix trie over a batch of N-best token sequences.
#[derive(Debug, Default)]
pub(super) struct TrieBuf {
    pub(super) tok: Vec<u16>,
    pub(super) pos: Vec<u16>,
    pub(super) parent: Vec<i32>,
    first_child: Vec<i32>,
    next_sib: Vec<i32>,
    pub(super) steps: Vec<(u32, u16)>,
    pub(super) ranges: Vec<(usize, usize)>,
}

impl TrieBuf {
    pub(super) fn build(
        &mut self,
        seqs: &[&[u32]],
        max_toks: usize,
        vocab: usize,
        dir: TfDirection,
    ) {
        self.tok.clear();
        self.pos.clear();
        self.parent.clear();
        self.first_child.clear();
        self.next_sib.clear();
        self.steps.clear();
        self.ranges.clear();
        self.tok.push(0);
        self.pos.push(0);
        self.parent.push(-1);
        self.first_child.push(-1);
        self.next_sib.push(-1);
        for seq in seqs {
            let (start, mut cur, n) = (self.steps.len(), 0usize, seq.len().min(max_toks));
            for i in 0..n {
                let raw = match dir {
                    TfDirection::L2r => seq[i],
                    TfDirection::R2l => seq[seq.len() - 1 - i],
                } as usize;
                let tok = if raw < vocab { raw as u16 } else { 0 };
                self.steps.push((cur as u32, tok));
                let mut child = self.first_child[cur];
                let mut found = None;
                while child >= 0 {
                    let cu = child as usize;
                    if self.tok[cu] == tok {
                        found = Some(cu);
                        break;
                    }
                    child = self.next_sib[cu];
                }
                cur = found.unwrap_or_else(|| {
                    let nu = self.tok.len();
                    self.tok.push(tok);
                    self.pos.push(self.pos[cur] + 1);
                    self.parent.push(cur as i32);
                    self.first_child.push(-1);
                    self.next_sib.push(self.first_child[cur]);
                    self.first_child[cur] = nu as i32;
                    nu
                });
            }
            self.steps.push((cur as u32, 0));
            self.ranges.push((start, self.steps.len()));
        }
    }
}

pub(super) fn mul2(a: usize, b: usize) -> Result<usize, EngineError> {
    a.checked_mul(b)
        .ok_or_else(|| EngineError::Format("UTFR size overflow".into()))
}

pub(super) fn push_f32(out: &mut Vec<u8>, xs: &[f32]) {
    xs.iter()
        .for_each(|x| out.extend_from_slice(&x.to_le_bytes()));
}

pub(super) fn read_slice<'a>(
    buf: &'a [u8],
    pos: &mut usize,
    n: usize,
) -> Result<&'a [u8], EngineError> {
    let end = pos
        .checked_add(n)
        .ok_or_else(|| EngineError::Format("UTFR offset overflow".into()))?;
    if end > buf.len() {
        return Err(EngineError::Format("unexpected EOF in UTFR".into()));
    }
    let s = &buf[*pos..end];
    *pos = end;
    Ok(s)
}

pub(super) fn read_f32(buf: &[u8], pos: &mut usize, n: usize) -> Result<Vec<f32>, EngineError> {
    let raw = read_slice(buf, pos, mul2(n, 4)?)?;
    raw.chunks_exact(4)
        .map(|c| {
            let v = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            v.is_finite()
                .then_some(v)
                .ok_or_else(|| EngineError::Format("non-finite f32 in UTFR".into()))
        })
        .collect()
}

pub(super) fn read_packed_i8(
    buf: &[u8],
    pos: &mut usize,
    n: usize,
    k: usize,
) -> Result<PackedI8, EngineError> {
    let q: Vec<i8> = read_slice(buf, pos, mul2(n, k)?)?
        .iter()
        .map(|&b| b as i8)
        .collect();
    let scale = read_f32(buf, pos, n)?;
    if q.contains(&i8::MIN) || scale.iter().any(|&s| s < 0.0) {
        return Err(EngineError::Format(
            "UTFR invalid int8 code or scale".into(),
        ));
    }
    let (kp, panels) = (k.div_ceil(KG) * KG, n.div_ceil(NR));
    let mut data = vec![0i8; mul2(mul2(panels, kp)?, NR)?];
    let (mut sc, mut wsum) = (vec![0.0f32; panels * NR], vec![0.0f32; panels * NR]);
    for row in 0..n {
        sc[row] = scale[row];
        let (p, j, mut qsum) = (row / NR, row % NR, 0i64);
        for kk in 0..k {
            let v = q[row * k + kk];
            data[((p * (kp / KG) + kk / KG) * NR + j) * KG + kk % KG] = v;
            qsum += i64::from(v);
        }
        wsum[row] = (qsum as f32) * scale[row];
    }
    PackedI8::from_parts(n, k, data, &sc, &wsum, &vec![0.0f32; n])
}

pub(super) fn push_packed_i8(out: &mut Vec<u8>, p: &PackedI8) {
    let ((data, scale, _), n, k, kp) = (p.parts(), p.n(), p.k(), p.kp());
    for row in 0..n {
        let (pn, j) = (row / NR, row % NR);
        for kk in 0..k {
            out.push(data[((pn * (kp / KG) + kk / KG) * NR + j) * KG + kk % KG] as u8);
        }
    }
    push_f32(out, &scale[..n]);
}
