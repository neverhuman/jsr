//! SHA-256 verified binary I/O for `ultra.lm-ffnn/v1` and cross-ISA / Python parity tests.

use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::path::Path;

use sha2::{Digest, Sha256};

use super::ffnn::{FfnnActivation, FfnnLm};
use super::traits::TokenLm;
use crate::EngineError;

pub const FFNN_MAGIC: &[u8; 8] = b"ULTRFF01";
pub const FFNN_VERSION: u32 = 1;
const FLAG_TIED: u32 = 1 << 0;
const FLAG_SILU: u32 = 1 << 1;
const FLAG_KNOWN_MASK: u32 = FLAG_TIED | FLAG_SILU;

pub(super) fn to_bytes(lm: &FfnnLm) -> Vec<u8> {
    let mut buf = Vec::with_capacity(lm.size_bytes() + 64);
    buf.extend_from_slice(FFNN_MAGIC);
    let flags = u32::from(lm.tied) | (u32::from(lm.act == FfnnActivation::Silu) << 1);
    for v in [
        FFNN_VERSION,
        lm.vocab as u32,
        lm.ctx as u32,
        lm.d_e as u32,
        lm.d_h as u32,
        flags,
    ] {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    buf.extend(lm.emb_q.iter().map(|&x| x as u8));
    push_f32(&mut buf, &lm.emb_scale);
    buf.extend(lm.w1_q.iter().map(|&x| x as u8));
    push_f32(&mut buf, &lm.w1_scale);
    push_f32(&mut buf, &lm.b1);
    if !lm.tied {
        buf.extend(lm.w2_q.iter().map(|&x| x as u8));
        push_f32(&mut buf, &lm.w2_scale);
    }
    push_f32(&mut buf, &lm.b2);
    buf.extend_from_slice(&Sha256::digest(&buf));
    buf
}

pub(super) fn from_bytes(raw: &[u8]) -> Result<FfnnLm, EngineError> {
    if raw.len() < 64 {
        return Err(EngineError::Format("FFNN LM file truncated".into()));
    }
    let (payload, tail) = raw.split_at(raw.len() - 32);
    if tail != Sha256::digest(payload).as_slice() {
        return Err(EngineError::Integrity("FFNN LM SHA-256 mismatch".into()));
    }
    let mut pos = 0usize;
    if read_slice(payload, &mut pos, 8)? != FFNN_MAGIC {
        return Err(EngineError::Format("bad FFNN LM magic".into()));
    }
    let hdr = [
        read_u32(payload, &mut pos)?,
        read_u32(payload, &mut pos)?,
        read_u32(payload, &mut pos)?,
        read_u32(payload, &mut pos)?,
        read_u32(payload, &mut pos)?,
        read_u32(payload, &mut pos)?,
    ];
    let [ver, vocab, ctx, d_e, d_h, flags] = hdr;
    if ver != FFNN_VERSION || (flags & !FLAG_KNOWN_MASK) != 0 {
        return Err(EngineError::Unsupported("unsupported FFNN header".into()));
    }
    let (v, c, de, dh) = (vocab as usize, ctx as usize, d_e as usize, d_h as usize);
    let tied = (flags & FLAG_TIED) != 0;
    let act = if (flags & FLAG_SILU) != 0 {
        FfnnActivation::Silu
    } else {
        FfnnActivation::Relu
    };
    let emb_q = read_i8(payload, &mut pos, v.saturating_mul(de))?;
    let emb_scale = read_f32(payload, &mut pos, v)?;
    let w1_q = read_i8(payload, &mut pos, dh.saturating_mul(c).saturating_mul(de))?;
    let w1_scale = read_f32(payload, &mut pos, dh)?;
    let b1 = read_f32(payload, &mut pos, dh)?;
    let (w2_q, w2_scale) = if tied {
        (Vec::new(), Vec::new())
    } else {
        (
            read_i8(payload, &mut pos, v.saturating_mul(dh))?,
            read_f32(payload, &mut pos, v)?,
        )
    };
    let b2 = read_f32(payload, &mut pos, v)?;
    if pos != payload.len() {
        return Err(EngineError::Format("trailing bytes in FFNN LM".into()));
    }
    FfnnLm::from_parts(
        v, c, de, dh, tied, act, emb_q, emb_scale, w1_q, w1_scale, b1, w2_q, w2_scale, b2,
    )
}

pub(super) fn save(lm: &FfnnLm, path: &Path) -> Result<(), EngineError> {
    let mut w = BufWriter::new(File::create(path)?);
    w.write_all(&to_bytes(lm))?;
    w.flush()?;
    Ok(())
}

pub(super) fn load(path: &Path) -> Result<FfnnLm, EngineError> {
    let mut raw = Vec::new();
    File::open(path)?.read_to_end(&mut raw)?;
    from_bytes(&raw)
}

fn push_f32(out: &mut Vec<u8>, xs: &[f32]) {
    for &x in xs {
        out.extend_from_slice(&x.to_le_bytes());
    }
}

fn read_slice<'a>(buf: &'a [u8], pos: &mut usize, n: usize) -> Result<&'a [u8], EngineError> {
    let end = pos
        .checked_add(n)
        .ok_or_else(|| EngineError::Format("FFNN offset overflow".into()))?;
    if end > buf.len() {
        return Err(EngineError::Format("unexpected EOF in FFNN LM".into()));
    }
    let s = &buf[*pos..end];
    *pos = end;
    Ok(s)
}

fn read_u32(buf: &[u8], pos: &mut usize) -> Result<u32, EngineError> {
    let b = read_slice(buf, pos, 4)?;
    Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

fn read_i8(buf: &[u8], pos: &mut usize, n: usize) -> Result<Vec<i8>, EngineError> {
    Ok(read_slice(buf, pos, n)?.iter().map(|&b| b as i8).collect())
}

fn read_f32(buf: &[u8], pos: &mut usize, n: usize) -> Result<Vec<f32>, EngineError> {
    let nbytes = n
        .checked_mul(4)
        .ok_or_else(|| EngineError::Format("len overflow".into()))?;
    Ok(read_slice(buf, pos, nbytes)?
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::Isa;

    const GOLDEN_SHA256: &str = "8f46b3295769032bcd0106fd4e5a4a97b31e8bc0aed6b70302ac54821e848b0c";
    const GOLDEN_SCORES: [f32; 8] = [
        -32.460_968,
        -18.040_712,
        -36.501_312,
        -9.991_06,
        -38.605_01,
        -0.004_762_6,
        -38.772_064,
        -5.359_194,
    ];

    fn golden(tied: bool, act: FfnnActivation) -> FfnnLm {
        let (v, c, de, dh) = (8, 4, 6, if tied { 6 } else { 8 });
        let (w2_q, w2_s) = if tied {
            (Vec::new(), Vec::new())
        } else {
            (
                (0..v * dh)
                    .map(|i| (((i * 13 + 5) % 23) as i32 - 11) as i8)
                    .collect(),
                (0..v).map(|k| (k + 2) as f32 / 64.0).collect(),
            )
        };
        FfnnLm::from_parts(
            v,
            c,
            de,
            dh,
            tied,
            act,
            (0..v * de)
                .map(|i| (((i * 17 + 7) % 25) as i32 - 12) as i8)
                .collect(),
            (0..v).map(|k| (k + 1) as f32 / 16.0).collect(),
            (0..dh * c * de)
                .map(|i| (((i * 29 + 3) % 27) as i32 - 13) as i8)
                .collect(),
            (0..dh).map(|j| (j + 1) as f32 / 32.0).collect(),
            (0..dh).map(|j| (j as i32 - 3) as f32 / 8.0).collect(),
            w2_q,
            w2_s,
            (0..v).map(|k| (k as i32 - 4) as f32 / 16.0).collect(),
        )
        .unwrap()
    }

    #[test]
    fn roundtrip_tamper_cross_isa_and_python_parity() {
        let cases = [
            (false, FfnnActivation::Relu),
            (false, FfnnActivation::Silu),
            (true, FfnnActivation::Silu),
        ];
        for (tied, act) in cases {
            let lm = golden(tied, act);
            let bytes = lm.to_bytes();
            let loaded = FfnnLm::from_bytes(&bytes).expect("roundtrip");
            assert_eq!(loaded, lm);
            let mut bad = bytes.clone();
            bad[12] ^= 0x40;
            assert!(matches!(
                FfnnLm::from_bytes(&bad),
                Err(EngineError::Integrity(_))
            ));
            for hist in [&[][..], &[1, 3, 5][..], &[7, 5, 2, 4, 6][..], &[999][..]] {
                let (mut a, mut b) = (vec![0.0; 8], vec![0.0; 8]);
                lm.score_all_isa(Isa::Scalar, hist, &mut a);
                loaded.score_all_isa(Isa::detect(), hist, &mut b);
                for k in 0..8 {
                    assert_eq!(a[k].to_bits(), b[k].to_bits());
                    assert_eq!(loaded.score_next(hist, k as u32).to_bits(), b[k].to_bits());
                }
                assert!((b.iter().map(|x| x.exp()).sum::<f32>() - 1.0).abs() < 1e-5);
            }
            assert_eq!(loaded.score_next(&[1], 99), f32::NEG_INFINITY);
        }
        let g = golden(false, FfnnActivation::Relu);
        let raw = g.to_bytes();
        let sha: String = raw[raw.len() - 32..]
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        assert_eq!(sha, GOLDEN_SHA256);
        let mut out = [0.0f32; 8];
        g.score_all(&[1, 3, 5], &mut out);
        for (got, want) in out.iter().zip(GOLDEN_SCORES) {
            assert!((got - want).abs() < 1e-5, "got={got} want={want}");
        }
        let l2 = FfnnLm::from_parts(
            512,
            6,
            128,
            512,
            false,
            FfnnActivation::Silu,
            vec![0; 512 * 128],
            vec![0.01; 512],
            vec![0; 512 * 6 * 128],
            vec![0.01; 512],
            vec![0.0; 512],
            vec![0; 512 * 512],
            vec![0.01; 512],
            vec![0.0; 512],
        )
        .unwrap();
        assert!(l2.size_bytes() < 1_468_006);
    }
}
