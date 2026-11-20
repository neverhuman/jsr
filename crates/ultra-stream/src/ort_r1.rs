//! Pure-Rust ONNX Runtime streaming transducer backend (`OrtTransducerBackend`) for R1
//! (`dense` `S` and `multirate` `S_MR`) consuming `ultra.r1-export/v1` bundles.
//!
//! Parity status: `R1Precision::Fp32` achieves `400/400` (`100.00%`) token-and-text parity with
//! `BundleRecognizer` on `dev-clean`. `R1Precision::Int8` is **EXPERIMENTAL** on dynamic
//! per-tensor activation-quantized bundles (`355/400 = 88.75%` due to batch-dependent
//! `DynamicQuantizeLinear` scales) pending static TRAIN-calibrated QDQ export.

use std::collections::{BTreeMap, VecDeque};
use std::ffi::CString;
use std::path::Path;
use std::sync::Arc;

use ultra_core::events::Emitter;
use ultra_frontend::{TrainerLogMel, MEL_BINS};
use ultra_gp::{checked_argmax_f32, FeatureRing, GpError, R1Bundle, R1Precision, RnntStreamState};

use crate::api::{AsrStream, StreamClock, StreamConfig, StreamError};
use crate::ort_c_api::{OrtEnv, OrtSession, OrtTensorIn, OrtTensorOut};
use crate::ort_r1_stream::{OrtR1AsrStream, OrtR1RawStream};

/// Samples per 25 Hz encoder frame (`40 ms @ 16 kHz = 640` samples).
pub const R1_SAMPLES_PER_ENC_FRAME: u64 = 640;

#[derive(Debug, Clone)]
pub(crate) enum StateBuf {
    F32(Vec<f32>),
    I64(Vec<i64>),
}

/// Reusable contiguous scratch buffers eliminating per-chunk heap allocations across streams.
#[derive(Default)]
pub struct BatchWorkspace {
    feats_in: Vec<f32>,
    t0_in: Vec<i64>,
    state_ins: Vec<StateBuf>,
    state_outs: Vec<StateBuf>,
    state_shapes: Vec<Vec<i64>>,
    enc_out: Vec<f32>,
    dec_ctx: Vec<i64>,
    dec_out: Vec<f32>,
    joi_enc: Vec<f32>,
    joi_logits: Vec<f32>,
}

impl BatchWorkspace {
    pub fn new() -> Self {
        Self::default()
    }
}

fn ensure_all_finite(slice: &[f32], label: &'static str) -> Result<(), StreamError> {
    if slice.iter().all(|v| v.is_finite()) {
        Ok(())
    } else {
        Err(GpError::Output(format!("non-finite {label} tensor entry")).into())
    }
}

/// Immutable R1 streaming transducer engine (`Send + Sync`, shared across CCX workers).
pub struct OrtTransducerBackend {
    pub ort: Arc<OrtEnv>,
    pub bundle: R1Bundle,
    pub threads: i32,
    enc: OrtSession,
    dec: OrtSession,
    joi: OrtSession,
    state_in_names: Vec<CString>,
    state_out_names: Vec<CString>,
}

impl OrtTransducerBackend {
    pub fn load(
        bundle_dir: &Path,
        precision: R1Precision,
        threads: i32,
        verify_hashes: bool,
    ) -> Result<Arc<Self>, StreamError> {
        let ort = OrtEnv::load_default()?;
        Self::load_with_ort(ort, bundle_dir, precision, threads, verify_hashes)
    }

    pub fn load_with_ort(
        ort: Arc<OrtEnv>,
        bundle_dir: &Path,
        precision: R1Precision,
        threads: i32,
        verify_hashes: bool,
    ) -> Result<Arc<Self>, StreamError> {
        let bundle = R1Bundle::load(bundle_dir, precision, verify_hashes)?;
        let enc = OrtSession::new(Arc::clone(&ort), &bundle.encoder_onnx, threads)?;
        let dec = OrtSession::new(Arc::clone(&ort), &bundle.decoder_onnx, threads)?;
        let joi = OrtSession::new(Arc::clone(&ort), &bundle.joiner_onnx, threads)?;

        let mut state_in_names = Vec::with_capacity(bundle.streaming.states.len());
        let mut state_out_names = Vec::with_capacity(bundle.streaming.states.len());
        for s in &bundle.streaming.states {
            state_in_names.push(
                CString::new(s.name.as_str())
                    .map_err(|_| StreamError::Protocol(format!("NUL in state {}", s.name)))?,
            );
            state_out_names.push(
                CString::new(format!("{}_out", s.name))
                    .map_err(|_| StreamError::Protocol(format!("NUL in state {}_out", s.name)))?,
            );
        }

        Ok(Arc::new(Self {
            ort,
            bundle,
            threads: threads.max(1),
            enc,
            dec,
            joi,
            state_in_names,
            state_out_names,
        }))
    }

    pub fn create_raw_stream(self: &Arc<Self>) -> OrtR1RawStream {
        let states = self
            .bundle
            .streaming
            .states
            .iter()
            .map(|spec| {
                if spec.is_int64 {
                    StateBuf::I64(vec![0i64; spec.per_stream_elems])
                } else {
                    StateBuf::F32(vec![0.0f32; spec.per_stream_elems])
                }
            })
            .collect();
        OrtR1RawStream {
            engine: Arc::clone(self),
            mel: TrainerLogMel::new(Arc::clone(&self.bundle.mel_cfg)),
            mel_scratch: Vec::with_capacity(8),
            ring: FeatureRing::new(),
            t_next: 0,
            finished: false,
            states,
            rnnt: RnntStreamState::new(self.bundle.decoder.ctx_init),
        }
    }

    /// Decode one ready step across a slice of streams (grouping by chunk length `n` identically
    /// to `bundle_infer.py:decode_streams`). Returns `(max_group_batch, ort_wakeups)`.
    pub fn decode_streams(
        &self,
        streams: &mut [&mut OrtR1RawStream],
        ws: &mut BatchWorkspace,
    ) -> Result<(usize, u64), StreamError> {
        let mut groups: BTreeMap<usize, Vec<(usize, usize)>> = BTreeMap::new();
        for (idx, s) in streams.iter().enumerate() {
            if let Some((t0, t1)) = s.chunk_bounds() {
                groups.entry(t1 - t0).or_default().push((idx, t0));
            }
        }
        let mut max_b = 0usize;
        let mut wakeups = 0u64;
        for (n, items) in groups {
            max_b = max_b.max(items.len());
            self.run_group(streams, &items, n, ws)?;
            wakeups += 1;
        }
        Ok((max_b, wakeups))
    }

    fn ensure_ws_states(&self, b: usize, ws: &mut BatchWorkspace) {
        let specs = &self.bundle.streaming.states;
        if ws.state_ins.len() != specs.len() {
            ws.state_ins = specs
                .iter()
                .map(|s| {
                    if s.is_int64 {
                        StateBuf::I64(Vec::new())
                    } else {
                        StateBuf::F32(Vec::new())
                    }
                })
                .collect();
            ws.state_outs = ws.state_ins.clone();
            ws.state_shapes = vec![Vec::new(); specs.len()];
        }
        for (si, spec) in specs.iter().enumerate() {
            let total = b * spec.per_stream_elems;
            match (&mut ws.state_ins[si], &mut ws.state_outs[si]) {
                (StateBuf::F32(a), StateBuf::F32(o)) => {
                    a.resize(total, 0.0);
                    o.resize(total, 0.0);
                }
                (StateBuf::I64(a), StateBuf::I64(o)) => {
                    a.resize(total, 0);
                    o.resize(total, 0);
                }
                _ => {}
            }
            ws.state_shapes[si].clear();
            ws.state_shapes[si].extend(spec.per_stream_dims.iter().map(|&d| d as i64));
            ws.state_shapes[si][spec.batch_axis] = b as i64;
        }
    }

    fn run_group(
        &self,
        all_streams: &mut [&mut OrtR1RawStream],
        items: &[(usize, usize)],
        n: usize,
        ws: &mut BatchWorkspace,
    ) -> Result<(), StreamError> {
        let b = items.len();
        let feat_rows = 4 * n + 3;
        let feat_stride = feat_rows * MEL_BINS;
        ws.feats_in.resize(b * feat_stride, 0.0);
        ws.t0_in.resize(b, 0);
        self.ensure_ws_states(b, ws);

        for (bi, &(s_idx, t0)) in items.iter().enumerate() {
            let s = &mut *all_streams[s_idx];
            s.ring.extract_chunk_into(
                t0,
                n,
                &mut ws.feats_in[bi * feat_stride..(bi + 1) * feat_stride],
            );
            ws.t0_in[bi] = t0 as i64;
            for (si, spec) in self.bundle.streaming.states.iter().enumerate() {
                let outer: usize = spec.per_stream_dims[..spec.batch_axis].iter().product();
                let inner: usize = spec.per_stream_dims[spec.batch_axis + 1..].iter().product();
                match (&s.states[si], &mut ws.state_ins[si]) {
                    (StateBuf::F32(src), StateBuf::F32(dst)) => {
                        for o in 0..outer {
                            let d0 = (o * b + bi) * inner;
                            dst[d0..d0 + inner].copy_from_slice(&src[o * inner..(o + 1) * inner]);
                        }
                    }
                    (StateBuf::I64(src), StateBuf::I64(dst)) => {
                        for o in 0..outer {
                            let d0 = (o * b + bi) * inner;
                            dst[d0..d0 + inner].copy_from_slice(&src[o * inner..(o + 1) * inner]);
                        }
                    }
                    _ => {}
                }
            }
        }

        let joiner_dim = self.bundle.streaming.joiner_dim;
        ws.enc_out.resize(b * n * joiner_dim, 0.0);

        let feats_shape = [b as i64, feat_rows as i64, MEL_BINS as i64];
        let t0_shape = [b as i64];
        let enc_shape = [b as i64, n as i64, joiner_dim as i64];

        let mut ort_ins = Vec::with_capacity(2 + self.state_in_names.len());
        ort_ins.push((c"feats", OrtTensorIn::f32(&ws.feats_in, &feats_shape)));
        ort_ins.push((c"t0", OrtTensorIn::i64(&ws.t0_in, &t0_shape)));
        for (si, name) in self.state_in_names.iter().enumerate() {
            let sh = &ws.state_shapes[si];
            let view = match &ws.state_ins[si] {
                StateBuf::F32(v) => OrtTensorIn::f32(v, sh),
                StateBuf::I64(v) => OrtTensorIn::i64(v, sh),
            };
            ort_ins.push((name.as_c_str(), view));
        }

        let mut ort_outs = Vec::with_capacity(1 + self.state_out_names.len());
        ort_outs.push((c"enc", OrtTensorOut::f32(&mut ws.enc_out, &enc_shape)));
        for (si, out_buf) in ws.state_outs.iter_mut().enumerate() {
            let sh = &ws.state_shapes[si];
            let view = match out_buf {
                StateBuf::F32(v) => OrtTensorOut::f32(v, sh),
                StateBuf::I64(v) => OrtTensorOut::i64(v, sh),
            };
            ort_outs.push((self.state_out_names[si].as_c_str(), view));
        }

        self.enc.run_into(&ort_ins, &mut ort_outs)?;
        drop(ort_outs);
        drop(ort_ins);
        ensure_all_finite(&ws.enc_out, "encoder")?;

        for (bi, &(s_idx, t0)) in items.iter().enumerate() {
            let s = &mut *all_streams[s_idx];
            for (si, spec) in self.bundle.streaming.states.iter().enumerate() {
                let outer: usize = spec.per_stream_dims[..spec.batch_axis].iter().product();
                let inner: usize = spec.per_stream_dims[spec.batch_axis + 1..].iter().product();
                match (&ws.state_outs[si], &mut s.states[si]) {
                    (StateBuf::F32(src), StateBuf::F32(dst)) => {
                        for o in 0..outer {
                            let s0 = (o * b + bi) * inner;
                            dst[o * inner..(o + 1) * inner].copy_from_slice(&src[s0..s0 + inner]);
                        }
                    }
                    (StateBuf::I64(src), StateBuf::I64(dst)) => {
                        for o in 0..outer {
                            let s0 = (o * b + bi) * inner;
                            dst[o * inner..(o + 1) * inner].copy_from_slice(&src[s0..s0 + inner]);
                        }
                    }
                    _ => {}
                }
            }
            s.t_next = t0 + n;
        }

        self.greedy_step(all_streams, items, n, ws)
    }

    /// Exact active-set transducer greedy inner loop matching `BundleRecognizer._greedy`:
    /// calls `decoder.onnx` on `ctx[active]` (`[a, 2] -> [a, d]`) and `joiner.{fp32,int8}.onnx`
    /// on `(enc[active, t], dec[active])`, removing blanked rows each symbol round.
    fn greedy_step(
        &self,
        all_streams: &mut [&mut OrtR1RawStream],
        items: &[(usize, usize)],
        n: usize,
        ws: &mut BatchWorkspace,
    ) -> Result<(), StreamError> {
        let b = items.len();
        let d = self.bundle.streaming.joiner_dim;
        let v = self.bundle.decoder.vocab_size;
        let blank = self.bundle.decoder.blank_id;
        let max_sym = self.bundle.decoder.max_symbols;

        let mut active: Vec<usize> = Vec::with_capacity(b);
        let mut next_active: Vec<usize> = Vec::with_capacity(b);

        for t in 0..n {
            active.clear();
            active.extend(0..b);
            let mut hits = 0usize;
            while !active.is_empty() && hits < max_sym {
                let a = active.len();
                ws.dec_ctx.resize(a * 2, 0);
                ws.dec_out.resize(a * d, 0.0);
                ws.joi_enc.resize(a * d, 0.0);
                ws.joi_logits.resize(a * v, 0.0);

                for (ai, &bi) in active.iter().enumerate() {
                    let ctx = all_streams[items[bi].0].rnnt.ctx;
                    ws.dec_ctx[ai * 2] = ctx[0];
                    ws.dec_ctx[ai * 2 + 1] = ctx[1];
                    let enc_off = (bi * n + t) * d;
                    ws.joi_enc[ai * d..(ai + 1) * d]
                        .copy_from_slice(&ws.enc_out[enc_off..enc_off + d]);
                }

                let ctx_sh = [a as i64, 2];
                let joi_in_sh = [a as i64, d as i64];
                let joi_out_sh = [a as i64, v as i64];

                self.dec.run_into(
                    &[(c"ctx", OrtTensorIn::i64(&ws.dec_ctx, &ctx_sh))],
                    &mut [(c"dec", OrtTensorOut::f32(&mut ws.dec_out, &joi_in_sh))],
                )?;
                ensure_all_finite(&ws.dec_out, "decoder")?;

                self.joi.run_into(
                    &[
                        (c"enc", OrtTensorIn::f32(&ws.joi_enc, &joi_in_sh)),
                        (c"dec", OrtTensorIn::f32(&ws.dec_out, &joi_in_sh)),
                    ],
                    &mut [(
                        c"logits",
                        OrtTensorOut::f32(&mut ws.joi_logits, &joi_out_sh),
                    )],
                )?;

                next_active.clear();
                for (ai, &bi) in active.iter().enumerate() {
                    let k = checked_argmax_f32(&ws.joi_logits[ai * v..(ai + 1) * v])? as i64;
                    if k != blank {
                        let st = &mut all_streams[items[bi].0].rnnt;
                        st.tokens.push(k);
                        st.ctx = [st.ctx[1], k];
                        next_active.push(bi);
                    }
                }
                std::mem::swap(&mut active, &mut next_active);
                hits += 1;
            }
            for &bi in &active {
                all_streams[items[bi].0].rnnt.forced_advances += 1;
            }
        }
        Ok(())
    }

    pub fn create_stream_with_clock(
        self: &Arc<Self>,
        cfg: &StreamConfig,
        clock: StreamClock,
    ) -> Box<dyn AsrStream> {
        let model_id = format!(
            "ultra-r1-{}-{}",
            self.bundle.family.as_str(),
            self.bundle.precision.as_str()
        );
        let eff_model_id = if cfg.model_config_id == "default-model-v1" {
            model_id.as_str()
        } else {
            cfg.model_config_id.as_str()
        };
        Box::new(OrtR1AsrStream {
            raw: self.create_raw_stream(),
            ws: BatchWorkspace::new(),
            clock,
            emitter: Emitter::new(&cfg.stream_id, eff_model_id),
            cfg: cfg.clone(),
            accepted_samples: 0,
            processed_samples: 0,
            wakeups: 0,
            events_emitted: 0,
            last_hyp: String::new(),
            finished: false,
            cancelled: false,
            events: VecDeque::new(),
        })
    }
}
