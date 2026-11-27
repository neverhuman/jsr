//! SHA-256 verified binary I/O for `ultra.tf-rescore/v1` (`.utfr`).
//!
//! Layout: 8-byte magic (`ULTRTF01`) + 11 `u32` LE header fields + FP32 embeddings/RMSNorms
//! and per-row symmetric INT8 (`|q| <= 127`) linear weights/scales + 32-byte trailing SHA-256.

use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::path::Path;

use sha2::{Digest, Sha256};

use super::tf_rescore_trie::{
    mul2, push_f32, push_packed_i8, read_f32, read_packed_i8, read_slice,
};
use crate::quant::PackedI8;
use crate::EngineError;

pub const TF_MAGIC: &[u8; 8] = b"ULTRTF01";
pub const TF_VERSION: u32 = 1;
const FLAG_TIED: u32 = 1 << 0;
const HEADER_BYTES: usize = 8 + 11 * 4;

/// Causal scoring direction of a `TfRescorer`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TfDirection {
    L2r,
    R2l,
}

/// Architecture and runtime configuration of a `ultra.tf-rescore/v1` Transformer rescorer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TfConfig {
    pub vocab: usize,
    pub d_model: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub d_ff: usize,
    pub ctx: usize,
    pub direction: TfDirection,
    pub tied: bool,
    pub rope_theta: f32,
    pub rms_eps: f32,
}

impl TfConfig {
    pub fn validate(&self) -> Result<(), EngineError> {
        let ok = (2..=usize::from(u16::MAX)).contains(&self.vocab)
            && self.d_model > 0
            && self.n_layers > 0
            && self.n_heads > 0
            && self.d_model.is_multiple_of(self.n_heads)
            && (self.d_model / self.n_heads).is_multiple_of(2)
            && self.d_ff > 0
            && self.ctx >= 2;
        if !ok {
            return Err(EngineError::Shape("invalid TF rescorer dims".into()));
        }
        if !(self.rope_theta.is_finite()
            && self.rope_theta > 0.0
            && self.rms_eps.is_finite()
            && self.rms_eps > 0.0)
        {
            return Err(EngineError::Format("invalid TF rope/eps".into()));
        }
        Ok(())
    }

    #[inline]
    pub fn head_dim(&self) -> usize {
        self.d_model / self.n_heads
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct TfLayer {
    pub(super) n1_w: Vec<f32>,
    pub(super) qkv: PackedI8,
    pub(super) wo: PackedI8,
    pub(super) n2_w: Vec<f32>,
    pub(super) w12: PackedI8,
    pub(super) w3: PackedI8,
}

/// Pure-Rust per-row INT8 causal Transformer N-best rescorer (`ultra.tf-rescore/v1`).
#[derive(Debug, Clone, PartialEq)]
pub struct TfRescorer {
    pub cfg: TfConfig,
    pub(super) emb: Vec<f32>,
    pub(super) layers: Vec<TfLayer>,
    pub(super) norm_w: Vec<f32>,
    pub(super) head: PackedI8,
    pub(super) rope_cos: Vec<f32>,
    pub(super) rope_sin: Vec<f32>,
}

impl TfRescorer {
    pub fn load(path: &Path) -> Result<Self, EngineError> {
        let mut raw = Vec::new();
        File::open(path)?.read_to_end(&mut raw)?;
        Self::from_bytes(&raw)
    }

    pub fn save(&self, path: &Path) -> Result<(), EngineError> {
        let mut w = BufWriter::new(File::create(path)?);
        w.write_all(&self.to_bytes())?;
        w.flush()?;
        Ok(())
    }

    pub fn from_bytes(raw: &[u8]) -> Result<Self, EngineError> {
        if raw.len() < HEADER_BYTES + 32 {
            return Err(EngineError::Format("UTFR file truncated".into()));
        }
        let (payload, tail) = raw.split_at(raw.len() - 32);
        if tail != Sha256::digest(payload).as_slice() {
            return Err(EngineError::Integrity("UTFR SHA-256 mismatch".into()));
        }
        let mut pos = 0usize;
        if read_slice(payload, &mut pos, 8)? != TF_MAGIC {
            return Err(EngineError::Format("bad UTFR magic".into()));
        }
        let mut h = [0u32; 11];
        for slot in &mut h {
            let b = read_slice(payload, &mut pos, 4)?;
            *slot = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
        }
        let [ver, v, d, nl, nh, f, ctx, dcode, flags, theta_b, eps_b] = h;
        if ver != TF_VERSION || (flags & !FLAG_TIED) != 0 {
            return Err(EngineError::Unsupported("unsupported UTFR header".into()));
        }
        let direction = match dcode {
            0 => TfDirection::L2r,
            1 => TfDirection::R2l,
            _ => return Err(EngineError::Unsupported("unknown UTFR direction".into())),
        };
        let cfg = TfConfig {
            vocab: v as usize,
            d_model: d as usize,
            n_layers: nl as usize,
            n_heads: nh as usize,
            d_ff: f as usize,
            ctx: ctx as usize,
            direction,
            tied: (flags & FLAG_TIED) != 0,
            rope_theta: f32::from_bits(theta_b),
            rms_eps: f32::from_bits(eps_b),
        };
        cfg.validate()?;
        let (v, d, f) = (cfg.vocab, cfg.d_model, cfg.d_ff);
        let emb = read_f32(payload, &mut pos, mul2(v, d)?)?;
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for _ in 0..cfg.n_layers {
            layers.push(TfLayer {
                n1_w: read_f32(payload, &mut pos, d)?,
                qkv: read_packed_i8(payload, &mut pos, 3 * d, d)?,
                wo: read_packed_i8(payload, &mut pos, d, d)?,
                n2_w: read_f32(payload, &mut pos, d)?,
                w12: read_packed_i8(payload, &mut pos, 2 * f, d)?,
                w3: read_packed_i8(payload, &mut pos, d, f)?,
            });
        }
        let norm_w = read_f32(payload, &mut pos, d)?;
        let head = if cfg.tied {
            PackedI8::quantize(&emb, v, d, None)
        } else {
            read_packed_i8(payload, &mut pos, v, d)?
        };
        if pos != payload.len() {
            return Err(EngineError::Format("trailing bytes in UTFR".into()));
        }
        let (rope_cos, rope_sin) = build_rope(cfg.ctx, cfg.head_dim(), cfg.rope_theta);
        Ok(Self {
            cfg,
            emb,
            layers,
            norm_w,
            head,
            rope_cos,
            rope_sin,
        })
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let c = self.cfg;
        let mut buf = Vec::with_capacity(self.memory_bytes() + 128);
        buf.extend_from_slice(TF_MAGIC);
        for v in [
            TF_VERSION,
            c.vocab as u32,
            c.d_model as u32,
            c.n_layers as u32,
            c.n_heads as u32,
            c.d_ff as u32,
            c.ctx as u32,
            u32::from(c.direction == TfDirection::R2l),
            u32::from(c.tied),
            c.rope_theta.to_bits(),
            c.rms_eps.to_bits(),
        ] {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        push_f32(&mut buf, &self.emb);
        for ly in &self.layers {
            push_f32(&mut buf, &ly.n1_w);
            push_packed_i8(&mut buf, &ly.qkv);
            push_packed_i8(&mut buf, &ly.wo);
            push_f32(&mut buf, &ly.n2_w);
            push_packed_i8(&mut buf, &ly.w12);
            push_packed_i8(&mut buf, &ly.w3);
        }
        push_f32(&mut buf, &self.norm_w);
        if !c.tied {
            push_packed_i8(&mut buf, &self.head);
        }
        buf.extend_from_slice(&Sha256::digest(&buf));
        buf
    }

    pub fn memory_bytes(&self) -> usize {
        let pb = |p: &PackedI8| p.parts().0.len() + (p.parts().1.len() + p.parts().2.len()) * 4;
        let base = (self.emb.len() + self.norm_w.len() + self.rope_cos.len() + self.rope_sin.len())
            * 4
            + pb(&self.head);
        base + self
            .layers
            .iter()
            .map(|l| {
                (l.n1_w.len() + l.n2_w.len()) * 4 + pb(&l.qkv) + pb(&l.wo) + pb(&l.w12) + pb(&l.w3)
            })
            .sum::<usize>()
    }
}

fn build_rope(ctx: usize, hd: usize, theta: f32) -> (Vec<f32>, Vec<f32>) {
    let half = hd / 2;
    let inv: Vec<f32> = (0..half)
        // f64 then round: libm f32 powf/sin_cos differ between glibc and macOS,
        // which flipped 3/730 panel words; f64-rounded tables are bit-identical.
        .map(|j| (1.0 / f64::from(theta).powf((2 * j) as f64 / hd as f64)) as f32)
        .collect();
    let (mut cos, mut sin) = (
        Vec::with_capacity(ctx * half),
        Vec::with_capacity(ctx * half),
    );
    for p in 0..ctx {
        for &f in &inv {
            let (s, c) = f64::from((p as f32) * f).sin_cos();
            let (s, c) = (s as f32, c as f32);
            cos.push(c);
            sin.push(s);
        }
    }
    (cos, sin)
}

#[cfg(test)]
mod rope_tests {
    use super::build_rope;

    fn fnv(v: &[f32]) -> u64 {
        v.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, x| {
            (h ^ u64::from(x.to_bits())).wrapping_mul(0x0100_0000_01b3)
        })
    }

    /// Golden hash shared by x86_64 glibc/musl, aarch64 linux and macOS:
    /// the t2 reranker must give the same words on every release target.
    #[test]
    fn rope_tables_are_platform_identical() {
        let (c, s) = build_rope(512, 64, 10_000.0);
        assert_eq!((fnv(&c), fnv(&s)), (GOLD_COS, GOLD_SIN));
    }

    const GOLD_COS: u64 = 0xdbae_a756_5b6e_1d6c;
    const GOLD_SIN: u64 = 0xb905_ab5c_0729_1438;
}
