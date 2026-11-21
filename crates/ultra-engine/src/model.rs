//! `ultra.r1-weights/v1` loader: `arch.json` + `weights.safetensors` into packed weights.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::Deserialize;
use ultra_core::receipt::sha256_hex;
use ultra_frontend::{GlobalCmvn, TrainerMelConfig, MEL_BINS};
use ultra_gp::TokenTable;

use crate::kernels::PackedF32;
use crate::quant::PackedI8;
use crate::safetensors::SafeTensors;
use crate::EngineError;

pub const WEIGHTS_SCHEMA: &str = "ultra.r1-weights/v1";

#[derive(Deserialize)]
struct RawArch {
    schema: String,
    model: RawModel,
    source: RawSource,
    files: BTreeMap<String, RawFile>,
    features: RawFeatures,
    decoder: RawDecoder,
}

#[derive(Deserialize)]
struct RawModel {
    family: String,
    config: RawConfig,
    encoder: RawEncoder,
}

#[derive(Deserialize)]
struct RawConfig {
    vocab_rnnt: usize,
    pred_emb: usize,
    joiner: usize,
    joiner_act: String,
}

#[derive(Deserialize)]
struct RawEncoder {
    n_mels: usize,
    d_model: usize,
    n_blocks: usize,
    n_heads: usize,
    ffn_mult: usize,
    conv_kernel: usize,
    stem_channels: usize,
    chunk: usize,
    left_context: usize,
    post_block_norm: bool,
    family: String,
}

#[derive(Deserialize)]
struct RawSource {
    checkpoint_sha256: String,
}

#[derive(Deserialize)]
struct RawFile {
    path: String,
    sha256: String,
}

#[derive(Deserialize)]
struct RawFeatures {
    sample_rate: u32,
    frame_length: usize,
    frame_shift: usize,
    n_fft: usize,
    n_mels: usize,
    log_floor: f32,
    cmvn: RawCmvn,
}

#[derive(Deserialize)]
struct RawCmvn {
    mean: Vec<f32>,
    istd: Vec<f32>,
}

#[derive(Deserialize)]
struct RawDecoder {
    kind: String,
    context: usize,
    vocab_size: usize,
    joiner_activation: String,
    blank_id: usize,
    max_symbols: usize,
    ctx_init: [usize; 2],
}

/// Validated architecture (dense family).
#[derive(Debug, Clone, PartialEq)]
pub struct Arch {
    pub n_mels: usize,
    pub d: usize,
    pub n_blocks: usize,
    pub heads: usize,
    pub dh: usize,
    pub ffn: usize,
    pub kernel: usize,
    pub stem_ch: usize,
    /// Frequency bins after the first / second stride-2 convolution (39 / 19 for 80 mels).
    pub f1: usize,
    pub f2: usize,
    pub chunk: usize,
    pub left: usize,
    pub post_block_norm: bool,
    pub vocab: usize,
    pub pred_emb: usize,
    pub joiner: usize,
    pub blank: usize,
    pub max_symbols: usize,
    pub ctx_init: [usize; 2],
    pub log_floor: f32,
    pub checkpoint_sha256: String,
    pub weights_sha256: String,
}

/// A linear layer `y = W x + b`: the fp32 packing always, the int8 packing when requested.
#[derive(Debug, Clone)]
pub struct Linear {
    pub f: PackedF32,
    pub q: Option<PackedI8>,
}

impl Linear {
    fn new(w: &[f32], n: usize, k: usize, b: Option<&[f32]>) -> Self {
        Self {
            f: PackedF32::pack(w, n, k, b),
            q: None,
        }
    }
    pub fn n(&self) -> usize {
        self.f.n
    }
    pub fn k(&self) -> usize {
        self.f.k
    }
}

#[derive(Debug, Clone)]
pub struct Norm {
    pub g: Vec<f32>,
    pub b: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct Block {
    pub ff1_norm: Norm,
    pub ff1_w1: Linear,
    pub ff1_w2: Linear,
    pub att_norm: Norm,
    pub qkv: Linear,
    pub att_out: Linear,
    pub conv_norm: Norm,
    pub pw1: Linear,
    /// Depthwise taps transposed to `[kernel][d]` so one tap is a contiguous channel vector.
    pub dw_w: Vec<f32>,
    pub dw_b: Vec<f32>,
    pub dnorm: Norm,
    pub pw2: Linear,
    pub ff2_norm: Norm,
    pub ff2_w1: Linear,
    pub ff2_w2: Linear,
    pub post_norm: Option<Norm>,
}

/// Everything the engine needs, loaded and packed once, shared read-only by all workers.
#[derive(Debug)]
pub struct Model {
    pub arch: Arch,
    pub root: PathBuf,
    /// Layer-1 stem conv: `[9][stem_ch]` taps (kh * 3 + kw, channel contiguous) and bias.
    pub c1_w: Vec<f32>,
    pub c1_b: Vec<f32>,
    /// Layer-2 stem conv as a GEMM with K ordered (kh, kw, c_in).
    pub c2: Linear,
    /// Stem projection with K ordered (f, c) to match the channel-last layer-2 output.
    pub stem_out: Linear,
    pub blocks: Vec<Block>,
    pub final_norm: Option<Norm>,
    pub enc_proj: Linear,
    /// Predictor embedding table `[vocab][pred_emb]`.
    pub emb: Vec<f32>,
    pub pred_proj: Linear,
    pub joiner_out: Linear,
    pub rope_inv: Vec<f32>,
    pub mel_cfg: Arc<TrainerMelConfig>,
    pub tokens: Arc<TokenTable>,
}

/// Resolve a declared file inside `root` (no separators, no escaping symlinks) and, when
/// `check` is set, verify its SHA-256 against `arch.json`.
fn verify(root: &Path, f: &RawFile, check: bool) -> Result<PathBuf, EngineError> {
    if f.path.is_empty() || f.path.contains('/') || f.path.contains('\\') || f.path.starts_with('.')
    {
        return Err(EngineError::Format(format!(
            "file path {} escapes the directory",
            f.path
        )));
    }
    let base = root.canonicalize()?;
    let p = base.join(&f.path).canonicalize()?;
    if p.parent() != Some(base.as_path()) {
        return Err(EngineError::Format(format!(
            "file {} resolves outside the weight directory",
            f.path
        )));
    }
    if check {
        let got = sha256_hex(&std::fs::read(&p)?);
        if got != f.sha256 {
            return Err(EngineError::Integrity(format!(
                "{}: sha256 {got} != arch.json {}",
                f.path, f.sha256
            )));
        }
    }
    Ok(p)
}

/// Positive and at most `max` (guards every allocation product derived from arch.json).
fn bounded(name: &str, v: usize, max: usize) -> Result<usize, EngineError> {
    if v == 0 || v > max {
        return Err(EngineError::Shape(format!(
            "{name} = {v} outside 1..={max}"
        )));
    }
    Ok(v)
}

fn file<'a>(raw: &'a RawArch, key: &str) -> Result<&'a RawFile, EngineError> {
    raw.files
        .get(key)
        .ok_or_else(|| EngineError::Format(format!("arch.json has no files.{key}")))
}

impl Model {
    /// Load and pack an `ultra.r1-weights/v1` directory. `verify_hashes` checks every file's
    /// SHA-256 against `arch.json` first.
    pub fn load(root: &Path, verify_hashes: bool) -> Result<Self, EngineError> {
        let text = std::fs::read_to_string(root.join("arch.json"))?;
        let raw: RawArch = serde_json::from_str(&text)
            .map_err(|e| EngineError::Format(format!("arch.json: {e}")))?;
        if raw.schema != WEIGHTS_SCHEMA {
            return Err(EngineError::Format(format!(
                "schema {} != {WEIGHTS_SCHEMA}",
                raw.schema
            )));
        }
        let e = &raw.model.encoder;
        if raw.model.family != "dense" || e.family != "dense" {
            return Err(EngineError::Unsupported(format!(
                "encoder family {} (the engine implements dense)",
                e.family
            )));
        }
        if raw.model.config.joiner_act != "tanh" {
            return Err(EngineError::Unsupported(format!(
                "joiner activation {}",
                raw.model.config.joiner_act
            )));
        }
        bounded("d_model", e.d_model, 4096)?;
        bounded("n_heads", e.n_heads, 64)?;
        bounded("n_blocks", e.n_blocks, 256)?;
        bounded("ffn_mult", e.ffn_mult, 16)?;
        bounded("conv_kernel", e.conv_kernel, 255)?;
        bounded("stem_channels", e.stem_channels, 1024)?;
        bounded("chunk", e.chunk, 64)?;
        bounded("left_context", e.left_context, 4096)?;
        bounded("vocab_rnnt", raw.model.config.vocab_rnnt, 1 << 20)?;
        bounded("pred_emb", raw.model.config.pred_emb, 4096)?;
        bounded("joiner", raw.model.config.joiner, 4096)?;
        bounded("max_symbols", raw.decoder.max_symbols, 64)?;
        if e.n_mels != MEL_BINS
            || !e.d_model.is_multiple_of(e.n_heads)
            || !(e.d_model / e.n_heads).is_multiple_of(2)
        {
            return Err(EngineError::Shape("n_mels/d_model/n_heads".into()));
        }
        let (fe, de) = (&raw.features, &raw.decoder);
        // The frontend (ultra-frontend TrainerLogMel) and greedy rule the engine implements.
        if (
            fe.sample_rate,
            fe.frame_length,
            fe.frame_shift,
            fe.n_fft,
            fe.n_mels,
        ) != (16_000, 400, 160, 512, MEL_BINS)
            || !(fe.log_floor > 0.0 && fe.log_floor.is_finite())
        {
            return Err(EngineError::Unsupported(
                "feature spec differs from TrainerLogMel".into(),
            ));
        }
        if de.kind != "stateless"
            || de.context != 2
            || de.vocab_size != raw.model.config.vocab_rnnt
            || de.joiner_activation != "tanh"
        {
            return Err(EngineError::Unsupported(
                "decoder spec (stateless, context 2, tanh)".into(),
            ));
        }
        if e.chunk == 0 || e.left_context == 0 || e.conv_kernel < 2 {
            return Err(EngineError::Shape("chunk/left_context/conv_kernel".into()));
        }
        let wfile = file(&raw, "weights")?;
        let f1 = (e.n_mels - 3) / 2 + 1;
        let arch = Arch {
            n_mels: e.n_mels,
            d: e.d_model,
            n_blocks: e.n_blocks,
            heads: e.n_heads,
            dh: e.d_model / e.n_heads,
            ffn: e.d_model * e.ffn_mult,
            kernel: e.conv_kernel,
            stem_ch: e.stem_channels,
            f1,
            f2: (f1 - 3) / 2 + 1,
            chunk: e.chunk,
            left: e.left_context,
            post_block_norm: e.post_block_norm,
            vocab: raw.model.config.vocab_rnnt,
            pred_emb: raw.model.config.pred_emb,
            joiner: raw.model.config.joiner,
            blank: raw.decoder.blank_id,
            max_symbols: raw.decoder.max_symbols,
            ctx_init: raw.decoder.ctx_init,
            log_floor: raw.features.log_floor,
            checkpoint_sha256: raw.source.checkpoint_sha256.clone(),
            weights_sha256: wfile.sha256.clone(),
        };
        if arch.blank >= arch.vocab || arch.ctx_init.iter().any(|&c| c >= arch.vocab) {
            return Err(EngineError::Format(
                "blank_id / ctx_init out of vocabulary".into(),
            ));
        }
        let st = SafeTensors::read(&verify(root, wfile, verify_hashes)?)?;
        if st.metadata.get("checkpoint_sha256") != Some(&arch.checkpoint_sha256) {
            return Err(EngineError::Integrity(
                "weights.safetensors metadata checkpoint_sha256 differs from arch.json".into(),
            ));
        }
        let tokens = Arc::new(TokenTable::from_file(
            &verify(root, file(&raw, "tokens")?, verify_hashes)?,
            arch.vocab,
        )?);
        let cmvn = GlobalCmvn::new(&raw.features.cmvn.mean, &raw.features.cmvn.istd)?;
        let mel_cfg = Arc::new(TrainerMelConfig::from_fbank_file(
            &verify(root, file(&raw, "mel_fbank")?, verify_hashes)?,
            arch.log_floor,
            cmvn,
        )?);
        let ckpt_mean = st.take("cmvn.mean", &[arch.n_mels])?;
        if ckpt_mean != raw.features.cmvn.mean {
            return Err(EngineError::Integrity(
                "arch.json cmvn differs from weights".into(),
            ));
        }
        Self::from_tensors(arch, &st, root, mel_cfg, tokens)
    }

    fn from_tensors(
        arch: Arch,
        st: &SafeTensors,
        root: &Path,
        mel_cfg: Arc<TrainerMelConfig>,
        tokens: Arc<TokenTable>,
    ) -> Result<Self, EngineError> {
        let a = &arch;
        let (d, ch, ffn) = (a.d, a.stem_ch, a.ffn);
        let lin = |name: &str, n: usize, k: usize| -> Result<Linear, EngineError> {
            let w = st.get(&format!("{name}.weight"), &[n, k])?;
            let b = st.get(&format!("{name}.bias"), &[n])?;
            Ok(Linear::new(&w.data, n, k, Some(&b.data)))
        };
        let norm = |name: &str| -> Result<Norm, EngineError> {
            Ok(Norm {
                g: st.take(&format!("{name}.weight"), &[d])?,
                b: st.take(&format!("{name}.bias"), &[d])?,
            })
        };
        // Stem layer 1: (ch, 1, 3, 3) -> [ch][9].
        let c1 = st.take("encoder.stem.c1.weight", &[ch, 1, 3, 3])?;
        let mut c1_w = vec![0.0f32; 9 * ch];
        for c in 0..ch {
            for t in 0..9 {
                c1_w[t * ch + c] = c1[c * 9 + t];
            }
        }
        let c1_b = st.take("encoder.stem.c1.bias", &[ch])?;
        // Stem layer 2: (ch_out, ch_in, 3, 3) -> rows ch_out, K index (kh*3+kw)*ch + c_in.
        let w2 = st.get("encoder.stem.c2.weight", &[ch, ch, 3, 3])?;
        let mut w2p = vec![0.0f32; ch * 9 * ch];
        for o in 0..ch {
            for i in 0..ch {
                for t in 0..9 {
                    w2p[o * 9 * ch + t * ch + i] = w2.data[(o * ch + i) * 9 + t];
                }
            }
        }
        let c2 = Linear::new(
            &w2p,
            ch,
            9 * ch,
            Some(&st.take("encoder.stem.c2.bias", &[ch])?),
        );
        // Stem projection: K index c * f2 + f in torch; f * ch + c here.
        let kin = ch * a.f2;
        let wo = st.get("encoder.stem.out.weight", &[d, kin])?;
        let mut wop = vec![0.0f32; d * kin];
        for o in 0..d {
            for c in 0..ch {
                for f in 0..a.f2 {
                    wop[o * kin + f * ch + c] = wo.data[o * kin + c * a.f2 + f];
                }
            }
        }
        let stem_out = Linear::new(&wop, d, kin, Some(&st.take("encoder.stem.out.bias", &[d])?));
        let mut blocks = Vec::with_capacity(a.n_blocks);
        for li in 0..a.n_blocks {
            let p = format!("encoder.blocks.{li}");
            let dw = st.get(&format!("{p}.conv.dw.weight"), &[d, 1, a.kernel])?;
            let mut dw_w = vec![0.0f32; a.kernel * d];
            for c in 0..d {
                for t in 0..a.kernel {
                    dw_w[t * d + c] = dw.data[c * a.kernel + t];
                }
            }
            blocks.push(Block {
                ff1_norm: norm(&format!("{p}.ff1.norm"))?,
                ff1_w1: lin(&format!("{p}.ff1.w1"), ffn, d)?,
                ff1_w2: lin(&format!("{p}.ff1.w2"), d, ffn)?,
                att_norm: norm(&format!("{p}.att.norm"))?,
                qkv: lin(&format!("{p}.att.qkv"), 3 * d, d)?,
                att_out: lin(&format!("{p}.att.out"), d, d)?,
                conv_norm: norm(&format!("{p}.conv.norm"))?,
                pw1: lin(&format!("{p}.conv.pw1"), 2 * d, d)?,
                dw_w,
                dw_b: st.take(&format!("{p}.conv.dw.bias"), &[d])?,
                dnorm: norm(&format!("{p}.conv.dnorm"))?,
                pw2: lin(&format!("{p}.conv.pw2"), d, d)?,
                ff2_norm: norm(&format!("{p}.ff2.norm"))?,
                ff2_w1: lin(&format!("{p}.ff2.w1"), ffn, d)?,
                ff2_w2: lin(&format!("{p}.ff2.w2"), d, ffn)?,
                post_norm: if a.post_block_norm {
                    Some(norm(&format!("{p}.norm"))?)
                } else {
                    None
                },
            });
        }
        let final_norm = if a.post_block_norm {
            None
        } else {
            Some(norm("encoder.final_norm")?)
        };
        let rope_inv = st.take("engine.rope_inv_freq", &[a.dh / 2])?;
        Ok(Self {
            c1_w,
            c1_b,
            c2,
            stem_out,
            blocks,
            final_norm,
            enc_proj: lin("joiner.enc_proj", a.joiner, d)?,
            emb: st.take("predictor.emb.weight", &[a.vocab, a.pred_emb])?,
            pred_proj: lin("predictor.proj", a.joiner, 2 * a.pred_emb)?,
            joiner_out: lin("joiner.out", a.vocab, a.joiner)?,
            rope_inv,
            mel_cfg,
            tokens,
            root: root.to_path_buf(),
            arch,
        })
    }

    /// Pack int8 copies (per-output-channel scales) of every encoder and joiner linear layer.
    pub fn quantize_int8(&mut self) {
        self.quantize_int8_where(|_, _| true);
    }

    /// Pack int8 copies of the layers for which `keep(kind, block)` is true (ablations);
    /// the others stay fp32 in the int8 engine.
    pub fn quantize_int8_where(&mut self, keep: impl Fn(&str, Option<usize>) -> bool) {
        for (kind, blk, l) in self.linears_mut() {
            l.q = keep(kind, blk).then(|| PackedI8::from_packed_f32(&l.f));
        }
    }

    /// Every quantisable linear layer with its kind and block index.
    pub fn linears_mut(&mut self) -> Vec<(&'static str, Option<usize>, &mut Linear)> {
        let mut v: Vec<(&'static str, Option<usize>, &mut Linear)> = vec![
            ("stem_c2", None, &mut self.c2),
            ("stem_out", None, &mut self.stem_out),
        ];
        for (i, b) in self.blocks.iter_mut().enumerate() {
            let s = Some(i);
            v.extend([
                ("ff1_w1", s, &mut b.ff1_w1),
                ("ff1_w2", s, &mut b.ff1_w2),
                ("qkv", s, &mut b.qkv),
                ("att_out", s, &mut b.att_out),
                ("pw1", s, &mut b.pw1),
                ("pw2", s, &mut b.pw2),
                ("ff2_w1", s, &mut b.ff2_w1),
                ("ff2_w2", s, &mut b.ff2_w2),
            ]);
        }
        v.push(("enc_proj", None, &mut self.enc_proj));
        v.push(("joiner_out", None, &mut self.joiner_out));
        v
    }
}
