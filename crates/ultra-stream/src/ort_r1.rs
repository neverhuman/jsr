//! Pure-Rust ONNX Runtime streaming transducer backend (`OrtTransducerBackend`) for R1
//! (`dense` `S` and `multirate` `S_MR`) consuming `ultra.r1-export/{v1,v1.1}` bundles:
//! zero-copy `b == 1` state ping-pong, pre-evaluated `ctx_init` + per-stream predictor context
//! cache (`ort_r1_greedy::tests`), AVX2 finite checks, and zero-alloc `OrtRunScratch`.

use std::collections::BTreeMap;
use std::ffi::CString;
use std::path::Path;
use std::sync::Arc;

use ultra_frontend::{TrainerLogMel, MEL_BINS};
use ultra_gp::{FeatureRing, R1Bundle, R1Precision, RnntStreamState};

use crate::api::StreamError;
use crate::ort_c_api::{OrtEnv, OrtSession, OrtTensorIn, OrtTensorOut};
use crate::ort_r1_stream::{validate_state_outputs, OrtR1RawStream};
pub(crate) use crate::ort_r1_ws::{ensure_all_finite, thread_cpu_ns, StateBuf};
pub use crate::ort_r1_ws::{BatchWorkspace, R1PhaseCounters};

/// Samples per 25 Hz encoder frame (`40 ms @ 16 kHz = 640` samples).
pub const R1_SAMPLES_PER_ENC_FRAME: u64 = 640;

type StreamSnapshot = ([i64; 2], usize, u64, Option<[i64; 2]>, Vec<f32>);

/// Immutable R1 streaming transducer engine (`Send + Sync`, shared across CCX workers).
pub struct OrtTransducerBackend {
    pub ort: Arc<OrtEnv>,
    pub bundle: R1Bundle,
    pub threads: i32,
    pub(crate) enc: OrtSession,
    pub(crate) dec: OrtSession,
    pub(crate) joi: OrtSession,
    state_in_names: Vec<CString>,
    state_out_names: Vec<CString>,
    b1_state_shapes: Vec<Vec<i64>>,
    dec_init_00: Vec<f32>,
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
        let mut b1_state_shapes = Vec::with_capacity(bundle.streaming.states.len());
        for s in &bundle.streaming.states {
            state_in_names.push(
                CString::new(s.name.as_str())
                    .map_err(|_| StreamError::Protocol(format!("NUL in state {}", s.name)))?,
            );
            state_out_names.push(
                CString::new(format!("{}_out", s.name))
                    .map_err(|_| StreamError::Protocol(format!("NUL in state {}_out", s.name)))?,
            );
            let mut sh = Vec::with_capacity(s.per_stream_dims.len());
            for &dim in &s.per_stream_dims {
                sh.push(
                    i64::try_from(dim)
                        .map_err(|_| StreamError::ResourceLimit("state dim > i64::MAX".into()))?,
                );
            }
            b1_state_shapes.push(sh);
        }

        let d = bundle.streaming.joiner_dim;
        let ctx_init = bundle.decoder.ctx_init;
        let mut dec_init_00 = vec![0.0f32; d];
        dec.run_into(
            &[(c"ctx", OrtTensorIn::i64(&ctx_init, &[1, 2]))],
            &mut [(c"dec", OrtTensorOut::f32(&mut dec_init_00, &[1, d as i64]))],
        )?;
        ensure_all_finite(&dec_init_00, "decoder_init")?;

        Ok(Arc::new(Self {
            ort,
            bundle,
            threads: threads.max(1),
            enc,
            dec,
            joi,
            state_in_names,
            state_out_names,
            b1_state_shapes,
            dec_init_00,
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
        let ctx_init = self.bundle.decoder.ctx_init;
        OrtR1RawStream {
            engine: Arc::clone(self),
            mel: TrainerLogMel::new(Arc::clone(&self.bundle.mel_cfg)),
            mel_scratch: Vec::with_capacity(8),
            ring: FeatureRing::new(),
            t_next: 0,
            finished: false,
            fail_after_chunks: None,
            states,
            alt_states: Vec::new(),
            cached_ctx: Some(ctx_init),
            cached_dec: self.dec_init_00.clone(),
            rnnt: RnntStreamState::new(ctx_init),
        }
    }

    /// Decode one ready step across a slice of streams (grouping by chunk length `n` identically
    /// to `bundle_infer.py:decode_streams`). Returns `(max_group_batch, compatible_groups)`.
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
            let step = self.bundle.int8_max_batch_enc.unwrap_or(items.len()).max(1);
            for sub in items.chunks(step) {
                max_b = max_b.max(sub.len());
                if sub.len() == 1 {
                    self.run_single_zero_copy(streams, sub, n, ws)?;
                } else {
                    self.run_group(streams, sub, n, ws)?;
                }
            }
            ws.0.phases.compatible_groups += 1;
            wakeups += 1;
        }
        Ok((max_b, wakeups))
    }

    /// Zero-copy `b == 1` fast path: binds `s.states` directly as ORT inputs and `s.alt_states`
    /// directly as ORT outputs (`0` gather/scatter bytes), swapping only on complete step success.
    fn run_single_zero_copy(
        &self,
        all_streams: &mut [&mut OrtR1RawStream],
        items: &[(usize, usize)],
        n: usize,
        ws_outer: &mut BatchWorkspace,
    ) -> Result<(), StreamError> {
        let ws = &mut *ws_outer.0;
        let prof = ws.profiling_enabled;
        let tp0 = thread_cpu_ns(prof);
        let (s_idx, t0) = items[0];
        let feat_rows = 4usize
            .checked_mul(n)
            .and_then(|r| r.checked_add(3))
            .ok_or_else(|| StreamError::ResourceLimit("chunk feature rows overflow".into()))?;
        let feat_stride = feat_rows
            .checked_mul(MEL_BINS)
            .ok_or_else(|| StreamError::ResourceLimit("chunk feature stride overflow".into()))?;
        ws.feats_in.resize(feat_stride, 0.0);
        ws.t0_in.resize(1, 0);
        ws.t0_in[0] = t0 as i64;
        let joiner_dim = self.bundle.streaming.joiner_dim;
        ws.enc_out.resize(n * joiner_dim, 0.0);

        {
            let s = &mut *all_streams[s_idx];
            s.ensure_alt_states();
            s.ring.extract_chunk_into(t0, n, &mut ws.feats_in);
            let tp1 = thread_cpu_ns(prof);
            ws.phases.cache_pack_cpu_ns += tp1.saturating_sub(tp0);

            let feats_shape = [1i64, feat_rows as i64, MEL_BINS as i64];
            let t0_shape = [1i64];
            let enc_shape = [1i64, n as i64, joiner_dim as i64];

            let mut ort_ins = Vec::with_capacity(2 + self.state_in_names.len());
            ort_ins.push((c"feats", OrtTensorIn::f32(&ws.feats_in, &feats_shape)));
            ort_ins.push((c"t0", OrtTensorIn::i64(&ws.t0_in, &t0_shape)));
            for (si, name) in self.state_in_names.iter().enumerate() {
                let sh = &self.b1_state_shapes[si];
                let view = match &s.states[si] {
                    StateBuf::F32(v) => OrtTensorIn::f32(v, sh),
                    StateBuf::I64(v) => OrtTensorIn::i64(v, sh),
                };
                ort_ins.push((name.as_c_str(), view));
            }

            let mut ort_outs = Vec::with_capacity(1 + self.state_out_names.len());
            ort_outs.push((c"enc", OrtTensorOut::f32(&mut ws.enc_out, &enc_shape)));
            for (si, out_buf) in s.alt_states.iter_mut().enumerate() {
                let sh = &self.b1_state_shapes[si];
                let view = match out_buf {
                    StateBuf::F32(v) => OrtTensorOut::f32(v, sh),
                    StateBuf::I64(v) => OrtTensorOut::i64(v, sh),
                };
                ort_outs.push((self.state_out_names[si].as_c_str(), view));
            }
            ws.ort_ins_cap = ws.ort_ins_cap.max(ort_ins.capacity());

            let te0 = thread_cpu_ns(prof);
            self.enc
                .run_into_with_scratch(&ort_ins, &mut ort_outs, &mut ws.ort_scratch)?;
            let te1 = thread_cpu_ns(prof);
            ws.phases.encoder_cpu_ns += te1.saturating_sub(te0);
            ws.phases.encoder_ort_calls += 1;
            ws.phases.zero_copy_single_steps += 1;
            drop(ort_outs);
            drop(ort_ins);

            let tv0 = thread_cpu_ns(prof);
            ensure_all_finite(&ws.enc_out, "encoder")?;
            validate_state_outputs(
                &self.bundle.streaming.states,
                self.bundle.streaming.left_context_frames,
                &s.alt_states,
            )?;
            let tv1 = thread_cpu_ns(prof);
            ws.phases.finite_validation_cpu_ns += tv1.saturating_sub(tv0);
        }

        self.commit_greedy_or_rollback(all_streams, items, n, ws_outer, true)
    }

    fn ensure_ws_states(&self, b: usize, ws_outer: &mut BatchWorkspace) -> Result<(), StreamError> {
        let ws = &mut *ws_outer.0;
        if b == 0 || b > 4096 {
            return Err(StreamError::ResourceLimit(format!(
                "invalid R1 batch size {b} (expected 1..=4096)"
            )));
        }
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
            let total = b.checked_mul(spec.per_stream_elems).ok_or_else(|| {
                StreamError::ResourceLimit(format!("state {} batch allocation overflow", spec.name))
            })?;
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
            ws.state_shapes[si].extend_from_slice(&self.b1_state_shapes[si]);
            ws.state_shapes[si][spec.batch_axis] = b as i64;
        }
        Ok(())
    }

    fn run_group(
        &self,
        all_streams: &mut [&mut OrtR1RawStream],
        items: &[(usize, usize)],
        n: usize,
        ws_outer: &mut BatchWorkspace,
    ) -> Result<(), StreamError> {
        self.ensure_ws_states(items.len(), ws_outer)?;
        let ws = &mut *ws_outer.0;
        let prof = ws.profiling_enabled;
        let tp0 = thread_cpu_ns(prof);
        let b = items.len();
        let feat_rows = 4usize
            .checked_mul(n)
            .and_then(|r| r.checked_add(3))
            .ok_or_else(|| StreamError::ResourceLimit("chunk feature rows overflow".into()))?;
        let feat_stride = feat_rows
            .checked_mul(MEL_BINS)
            .ok_or_else(|| StreamError::ResourceLimit("chunk feature stride overflow".into()))?;
        let feat_total = b
            .checked_mul(feat_stride)
            .ok_or_else(|| StreamError::ResourceLimit("batch feature buffer overflow".into()))?;
        ws.feats_in.resize(feat_total, 0.0);
        ws.t0_in.resize(b, 0);

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
        let tp1 = thread_cpu_ns(prof);
        ws.phases.cache_pack_cpu_ns += tp1.saturating_sub(tp0);

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
        ws.ort_ins_cap = ws.ort_ins_cap.max(ort_ins.capacity());

        let te0 = thread_cpu_ns(prof);
        self.enc
            .run_into_with_scratch(&ort_ins, &mut ort_outs, &mut ws.ort_scratch)?;
        let te1 = thread_cpu_ns(prof);
        ws.phases.encoder_cpu_ns += te1.saturating_sub(te0);
        ws.phases.encoder_ort_calls += 1;
        ws.phases.batched_steps += 1;
        drop(ort_outs);
        drop(ort_ins);

        let tv0 = thread_cpu_ns(prof);
        ensure_all_finite(&ws.enc_out, "encoder")?;
        validate_state_outputs(
            &self.bundle.streaming.states,
            self.bundle.streaming.left_context_frames,
            &ws.state_outs,
        )?;
        let tv1 = thread_cpu_ns(prof);
        ws.phases.finite_validation_cpu_ns += tv1.saturating_sub(tv0);

        self.commit_greedy_or_rollback(all_streams, items, n, ws_outer, false)
    }

    fn commit_greedy_or_rollback(
        &self,
        all_streams: &mut [&mut OrtR1RawStream],
        items: &[(usize, usize)],
        n: usize,
        ws: &mut BatchWorkspace,
        zero_copy_single: bool,
    ) -> Result<(), StreamError> {
        let prof = ws.0.profiling_enabled;
        let snapshots: Vec<StreamSnapshot> = items
            .iter()
            .map(|&(s_idx, _)| {
                let s = &all_streams[s_idx];
                (
                    s.rnnt.ctx,
                    s.rnnt.tokens.len(),
                    s.rnnt.forced_advances,
                    s.cached_ctx,
                    s.cached_dec.clone(),
                )
            })
            .collect();

        let injected_err = items.iter().find_map(|&(s_idx, _)| {
            all_streams[s_idx]
                .fail_after_chunks
                .filter(|&rem| rem == 0)
                .map(|_| StreamError::Runtime("injected late joiner/decoder failure".into()))
        });
        let step_res = match injected_err {
            Some(e) => Err(e),
            None => self.greedy_step(all_streams, items, n, ws),
        };
        if let Err(err) = step_res {
            for (&(s_idx, _), (ctx, tok_len, fa, c_ctx, c_dec)) in items.iter().zip(snapshots) {
                let s = &mut all_streams[s_idx];
                s.rnnt.ctx = ctx;
                s.rnnt.tokens.truncate(tok_len);
                s.rnnt.forced_advances = fa;
                s.cached_ctx = c_ctx;
                s.cached_dec = c_dec;
            }
            return Err(err);
        }

        let ts0 = thread_cpu_ns(prof);
        let b = items.len();
        for (bi, &(s_idx, t0)) in items.iter().enumerate() {
            let s = &mut *all_streams[s_idx];
            if let Some(rem) = s.fail_after_chunks.as_mut() {
                *rem = rem.saturating_sub(1);
            }
            if zero_copy_single {
                std::mem::swap(&mut s.states, &mut s.alt_states);
            } else {
                for (si, spec) in self.bundle.streaming.states.iter().enumerate() {
                    let outer: usize = spec.per_stream_dims[..spec.batch_axis].iter().product();
                    let inner: usize = spec.per_stream_dims[spec.batch_axis + 1..].iter().product();
                    match (&ws.0.state_outs[si], &mut s.states[si]) {
                        (StateBuf::F32(src), StateBuf::F32(dst)) => {
                            for o in 0..outer {
                                let s0 = (o * b + bi) * inner;
                                dst[o * inner..(o + 1) * inner]
                                    .copy_from_slice(&src[s0..s0 + inner]);
                            }
                        }
                        (StateBuf::I64(src), StateBuf::I64(dst)) => {
                            for o in 0..outer {
                                let s0 = (o * b + bi) * inner;
                                dst[o * inner..(o + 1) * inner]
                                    .copy_from_slice(&src[s0..s0 + inner]);
                            }
                        }
                        _ => {}
                    }
                }
            }
            s.t_next = t0 + n;
        }
        let ts1 = thread_cpu_ns(prof);
        ws.0.phases.scatter_cpu_ns += ts1.saturating_sub(ts0);
        Ok(())
    }
}
