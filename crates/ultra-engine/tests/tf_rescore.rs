//! Unit tests for `ultra.tf-rescore/v1`: format roundtrip, tamper rejection, FP32-vs-INT8
//! logp parity & correlation, L2R vs R2L direction, trie-batch identity, and cross-ISA parity.

use sha2::{Digest, Sha256};
use ultra_engine::lm::{TfConfig, TfDirection, TfRescorer, TfWorkspace, TF_MAGIC, TF_VERSION};
use ultra_engine::{EngineError, Isa};

struct Rng(u64);
impl Rng {
    fn f32(&mut self, scale: f32) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((((self.0 >> 32) as u32) as f64 / u32::MAX as f64) * 2.0 - 1.0) as f32 * scale
    }
    fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| self.f32(scale)).collect()
    }
}

fn push_f32(buf: &mut Vec<u8>, xs: &[f32]) {
    xs.iter()
        .for_each(|x| buf.extend_from_slice(&x.to_le_bytes()));
}

fn push_quant(buf: &mut Vec<u8>, w: &[f32], n: usize, k: usize) {
    let mut s = vec![0.0f32; n];
    for r in 0..n {
        let row = &w[r * k..(r + 1) * k];
        let mx = row.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
        s[r] = mx / 127.0;
        let inv = if mx > 0.0 { 127.0 / mx } else { 0.0 };
        for &v in row {
            buf.push(((v * inv).round_ties_even().clamp(-127.0, 127.0) as i8) as u8);
        }
    }
    push_f32(buf, &s);
}

#[derive(Clone)]
struct Fp32Tf {
    cfg: TfConfig,
    emb: Vec<f32>,
    n1: Vec<Vec<f32>>,
    qkv: Vec<Vec<f32>>,
    wo: Vec<Vec<f32>>,
    n2: Vec<Vec<f32>>,
    w12: Vec<Vec<f32>>,
    w3: Vec<Vec<f32>>,
    norm: Vec<f32>,
}

impl Fp32Tf {
    fn random(cfg: TfConfig, seed: u64) -> Self {
        let (mut rng, v, d, f, nl) = (Rng(seed), cfg.vocab, cfg.d_model, cfg.d_ff, cfg.n_layers);
        let ones = |r: &mut Rng| (0..d).map(|_| 1.0 + r.f32(0.05)).collect();
        Self {
            cfg,
            emb: rng.vec(v * d, 0.25),
            n1: (0..nl).map(|_| ones(&mut rng)).collect(),
            qkv: (0..nl).map(|_| rng.vec(3 * d * d, 0.2)).collect(),
            wo: (0..nl).map(|_| rng.vec(d * d, 0.2)).collect(),
            n2: (0..nl).map(|_| ones(&mut rng)).collect(),
            w12: (0..nl).map(|_| rng.vec(2 * f * d, 0.2)).collect(),
            w3: (0..nl).map(|_| rng.vec(d * f, 0.2)).collect(),
            norm: ones(&mut rng),
        }
    }

    fn to_utfr(&self) -> Vec<u8> {
        let (c, d, f, mut buf) = (self.cfg, self.cfg.d_model, self.cfg.d_ff, Vec::new());
        buf.extend_from_slice(TF_MAGIC);
        let hdr = [
            TF_VERSION,
            c.vocab as u32,
            d as u32,
            c.n_layers as u32,
            c.n_heads as u32,
            f as u32,
            c.ctx as u32,
            u32::from(c.direction == TfDirection::R2l),
            u32::from(c.tied),
            c.rope_theta.to_bits(),
            c.rms_eps.to_bits(),
        ];
        hdr.iter()
            .for_each(|v| buf.extend_from_slice(&v.to_le_bytes()));
        push_f32(&mut buf, &self.emb);
        for i in 0..c.n_layers {
            push_f32(&mut buf, &self.n1[i]);
            push_quant(&mut buf, &self.qkv[i], 3 * d, d);
            push_quant(&mut buf, &self.wo[i], d, d);
            push_f32(&mut buf, &self.n2[i]);
            push_quant(&mut buf, &self.w12[i], 2 * f, d);
            push_quant(&mut buf, &self.w3[i], d, f);
        }
        push_f32(&mut buf, &self.norm);
        buf.extend_from_slice(&Sha256::digest(&buf));
        buf
    }

    fn score_fp32(&self, tokens: &[u32]) -> f32 {
        let c = self.cfg;
        let (v, d, nh, hd, f) = (c.vocab, c.d_model, c.n_heads, c.head_dim(), c.d_ff);
        let mut seq = vec![0usize];
        for i in 0..tokens.len().min(c.ctx - 1) {
            let t = if c.direction == TfDirection::R2l {
                tokens[tokens.len() - 1 - i]
            } else {
                tokens[i]
            } as usize;
            seq.push(if t < v { t } else { 0 });
        }
        seq.push(0);
        let t_len = seq.len() - 1;
        let rms = |x: &[f32], w: &[f32]| -> Vec<f32> {
            x.chunks_exact(d)
                .flat_map(|xr| {
                    let ss: f32 = xr.iter().map(|&z| z * z).sum();
                    let inv = 1.0 / (ss / d as f32 + c.rms_eps).sqrt();
                    xr.iter().zip(w).map(move |(&z, &wi)| z * inv * wi)
                })
                .collect()
        };
        let mm = |x: &[f32], w: &[f32], n: usize, k: usize| -> Vec<f32> {
            (0..t_len)
                .flat_map(|t| {
                    (0..n).map(move |j| (0..k).map(|kk| x[t * k + kk] * w[j * k + kk]).sum())
                })
                .collect()
        };
        let mut h: Vec<f32> = seq[..t_len]
            .iter()
            .flat_map(|&tok| self.emb[tok * d..(tok + 1) * d].iter().copied())
            .collect();
        let inv_f: Vec<f32> = (0..hd / 2)
            .map(|j| (1.0 / f64::from(c.rope_theta).powf((2 * j) as f64 / hd as f64)) as f32)
            .collect();
        let sc = 1.0 / (hd as f32).sqrt();
        for l in 0..c.n_layers {
            let mut qkv = mm(&rms(&h, &self.n1[l]), &self.qkv[l], 3 * d, d);
            for t in 0..t_len {
                for hh in 0..nh {
                    for off in [0, d] {
                        let sl =
                            &mut qkv[t * 3 * d + off + hh * hd..t * 3 * d + off + (hh + 1) * hd];
                        for j in 0..hd / 2 {
                            let (s, cs) = f64::from((t as f32) * inv_f[j]).sin_cos();
                            let (s, cs) = (s as f32, cs as f32);
                            let (a, b) = (sl[2 * j], sl[2 * j + 1]);
                            (sl[2 * j], sl[2 * j + 1]) = (a * cs - b * s, a * s + b * cs);
                        }
                    }
                }
            }
            let mut att = vec![0.0f32; t_len * d];
            for t in 0..t_len {
                for hh in 0..nh {
                    let scores: Vec<f32> = (0..=t)
                        .map(|k| {
                            (0..hd)
                                .map(|j| {
                                    qkv[t * 3 * d + hh * hd + j] * qkv[k * 3 * d + d + hh * hd + j]
                                })
                                .sum::<f32>()
                                * sc
                        })
                        .collect();
                    let mx = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let sum: f32 = scores.iter().map(|&z| (z - mx).exp()).sum();
                    for (k, &s) in scores.iter().enumerate() {
                        let p = (s - mx).exp() / sum;
                        for j in 0..hd {
                            att[t * d + hh * hd + j] += p * qkv[k * 3 * d + 2 * d + hh * hd + j];
                        }
                    }
                }
            }
            for (a, b) in h.iter_mut().zip(mm(&att, &self.wo[l], d, d)) {
                *a += b;
            }
            let ff12 = mm(&rms(&h, &self.n2[l]), &self.w12[l], 2 * f, d);
            let ff12_s = ff12.as_slice();
            let mid: Vec<f32> = (0..t_len)
                .flat_map(|t| {
                    (0..f).map(move |j| {
                        let g = ff12_s[t * 2 * f + j];
                        (g / (1.0 + (-g).exp())) * ff12_s[t * 2 * f + f + j]
                    })
                })
                .collect();
            for (a, b) in h.iter_mut().zip(mm(&mid, &self.w3[l], d, f)) {
                *a += b;
            }
        }
        let logits = mm(&rms(&h, &self.norm), &self.emb, v, d);
        (0..t_len)
            .map(|t| {
                let row = &logits[t * v..(t + 1) * v];
                let mx = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                row[seq[t + 1]] - (mx + row.iter().map(|&z| (z - mx).exp()).sum::<f32>().ln())
            })
            .sum()
    }
}

#[test]
fn format_roundtrip_tamper_fp32_parity_trie_batch_and_r2l() {
    let cfg_l = TfConfig {
        vocab: 24,
        d_model: 32,
        n_layers: 2,
        n_heads: 4,
        d_ff: 64,
        ctx: 32,
        direction: TfDirection::L2r,
        tied: true,
        rope_theta: 10000.0,
        rms_eps: 1e-6,
    };
    let fp_l = Fp32Tf::random(cfg_l, 123);
    let raw = fp_l.to_utfr();
    let m_l = TfRescorer::from_bytes(&raw).expect("load l2r");
    assert_eq!(TfRescorer::from_bytes(&m_l.to_bytes()).expect("rt"), m_l);
    let mut bad = raw.clone();
    bad[14] ^= 0x5A;
    assert!(matches!(
        TfRescorer::from_bytes(&bad),
        Err(EngineError::Integrity(_))
    ));

    let seqs: [&[u32]; 5] = [
        &[],
        &[4],
        &[4, 7, 11, 15],
        &[4, 7, 12, 19, 3],
        &[8, 5, 2, 14],
    ];
    for act7 in [false, true] {
        let mut ws = TfWorkspace::default();
        ws.set_act7(act7);
        let batch_s = m_l
            .score_batch(Isa::Scalar, &seqs, &mut ws)
            .expect("scalar");
        let batch_d = m_l
            .score_batch(Isa::detect(), &seqs, &mut ws)
            .expect("detect");
        let mut token_logp = Vec::new();
        let with_tokens = m_l
            .score_batch_with_token_logp(Isa::detect(), &seqs, &mut ws, &mut token_logp)
            .expect("token scores");
        assert_eq!(with_tokens, batch_d);
        let mut remaining = token_logp.as_slice();
        for score in with_tokens {
            let (tokens, rest) = remaining.split_at(score.n_tok);
            assert_eq!(tokens.iter().sum::<f32>(), score.tok_logp);
            remaining = rest;
        }
        assert!(remaining.is_empty());
        for (i, s) in seqs.iter().enumerate() {
            let single = m_l.score_sequence(s, &mut ws).expect("single");
            assert!((batch_d[i].total_logp - single.total_logp).abs() < 1e-5);
            assert!((batch_s[i].total_logp - batch_d[i].total_logp).abs() < 1e-4);
            assert!((batch_d[i].total_logp - fp_l.score_fp32(s)).abs() < 0.25);
        }
    }
    let mut fp_r = fp_l.clone();
    fp_r.cfg.direction = TfDirection::R2l;
    let m_r = TfRescorer::from_bytes(&fp_r.to_utfr()).expect("load r2l");
    let mut ws = TfWorkspace::default();
    let fwd = m_l.score_sequence(&[4, 7, 11, 15], &mut ws).expect("fwd");
    let rev = m_r.score_sequence(&[15, 11, 7, 4], &mut ws).expect("rev");
    assert!((fwd.total_logp - rev.total_logp).abs() < 1e-5);
    assert!(
        (fwd.total_logp
            - m_r
                .score_sequence(&[4, 7, 11, 15], &mut ws)
                .expect("a")
                .total_logp)
            .abs()
            > 1e-3
    );
}
