//! Active-set transducer greedy inner loop with per-stream predictor context caching,
//! `OrtRunScratch` reuse, and `CLOCK_THREAD_CPUTIME_ID` phase attribution.

use ultra_gp::checked_argmax_f32;

use crate::api::StreamError;
use crate::ort_c_api::{OrtTensorIn, OrtTensorOut};
use crate::ort_r1::OrtTransducerBackend;
use crate::ort_r1_stream::OrtR1RawStream;
use crate::ort_r1_ws::{ensure_all_finite, thread_cpu_ns, BatchWorkspace};

impl OrtTransducerBackend {
    /// Active-set transducer greedy inner loop matching `BundleRecognizer._greedy`:
    /// reuses each stream's cached `decoder(ctx)` vector whenever `ctx` has not changed since the
    /// last non-blank emission, calling `decoder.onnx` only on newly changed contexts and recording
    /// actual workload counts in `phases.{decoder_cache_hits, decoder_ort_calls, joiner_ort_calls}`.
    /// `decoder_vectors_match_across_batch_shapes_m1_m4_m16` verifies raw `decoder.onnx` output
    /// equality across miss-batch sizes `m = 1, 4, 16`.
    pub(crate) fn greedy_step(
        &self,
        all_streams: &mut [&mut OrtR1RawStream],
        items: &[(usize, usize)],
        n: usize,
        ws_outer: &mut BatchWorkspace,
    ) -> Result<(), StreamError> {
        let ws = &mut *ws_outer.0;
        let prof = ws.profiling_enabled;
        let t_greedy0 = thread_cpu_ns(prof);
        let b = items.len();
        let d = self.bundle.streaming.joiner_dim;
        let v = self.bundle.decoder.vocab_size;
        let blank = self.bundle.decoder.blank_id;
        let max_sym = self.bundle.decoder.max_symbols;

        let mut active: Vec<usize> = Vec::with_capacity(b);
        let mut next_active: Vec<usize> = Vec::with_capacity(b);
        let mut miss_slots: Vec<usize> = Vec::with_capacity(b);

        for t in 0..n {
            active.clear();
            active.extend(0..b);
            let mut hits = 0usize;
            while !active.is_empty() && hits < max_sym {
                let a = active.len();
                ws.joi_enc.resize(a * d, 0.0);
                ws.joi_dec.resize(a * d, 0.0);
                ws.joi_logits.resize(a * v, 0.0);
                miss_slots.clear();

                for (ai, &bi) in active.iter().enumerate() {
                    let s = &all_streams[items[bi].0];
                    let enc_off = (bi * n + t) * d;
                    ws.joi_enc[ai * d..(ai + 1) * d]
                        .copy_from_slice(&ws.enc_out[enc_off..enc_off + d]);
                    if s.cached_ctx == Some(s.rnnt.ctx) && s.cached_dec.len() == d {
                        ws.joi_dec[ai * d..(ai + 1) * d].copy_from_slice(&s.cached_dec);
                        ws.phases.decoder_cache_hits += 1;
                    } else {
                        miss_slots.push(ai);
                    }
                }

                if !miss_slots.is_empty() {
                    let m = miss_slots.len();
                    ws.dec_ctx.resize(m * 2, 0);
                    ws.dec_out.resize(m * d, 0.0);
                    for (mi, &ai) in miss_slots.iter().enumerate() {
                        let ctx = all_streams[items[active[ai]].0].rnnt.ctx;
                        ws.dec_ctx[mi * 2] = ctx[0];
                        ws.dec_ctx[mi * 2 + 1] = ctx[1];
                    }
                    let ctx_sh = [m as i64, 2];
                    let dec_sh = [m as i64, d as i64];
                    let td0 = thread_cpu_ns(prof);
                    self.dec.run_into_with_scratch(
                        &[(c"ctx", OrtTensorIn::i64(&ws.dec_ctx, &ctx_sh))],
                        &mut [(c"dec", OrtTensorOut::f32(&mut ws.dec_out, &dec_sh))],
                        &mut ws.ort_scratch,
                    )?;
                    let td1 = thread_cpu_ns(prof);
                    ws.phases.decoder_cpu_ns += td1.saturating_sub(td0);
                    ws.phases.decoder_ort_calls += 1;
                    ensure_all_finite(&ws.dec_out, "decoder")?;
                    for (mi, &ai) in miss_slots.iter().enumerate() {
                        let src = &ws.dec_out[mi * d..(mi + 1) * d];
                        ws.joi_dec[ai * d..(ai + 1) * d].copy_from_slice(src);
                        let st = &mut all_streams[items[active[ai]].0];
                        st.cached_ctx = Some(st.rnnt.ctx);
                        st.cached_dec.clear();
                        st.cached_dec.extend_from_slice(src);
                    }
                }

                let tj0 = thread_cpu_ns(prof);
                if let Some(mb) = self.bundle.int8_max_batch_joi {
                    let step = mb.max(1);
                    let mut off = 0usize;
                    while off < a {
                        let sub = (a - off).min(step);
                        let sub_in = [sub as i64, d as i64];
                        let sub_out = [sub as i64, v as i64];
                        self.joi.run_into_with_scratch(
                            &[
                                (
                                    c"enc",
                                    OrtTensorIn::f32(
                                        &ws.joi_enc[off * d..(off + sub) * d],
                                        &sub_in,
                                    ),
                                ),
                                (
                                    c"dec",
                                    OrtTensorIn::f32(
                                        &ws.joi_dec[off * d..(off + sub) * d],
                                        &sub_in,
                                    ),
                                ),
                            ],
                            &mut [(
                                c"logits",
                                OrtTensorOut::f32(
                                    &mut ws.joi_logits[off * v..(off + sub) * v],
                                    &sub_out,
                                ),
                            )],
                            &mut ws.ort_scratch,
                        )?;
                        ws.phases.joiner_ort_calls += 1;
                        off += sub;
                    }
                } else {
                    let joi_in_sh = [a as i64, d as i64];
                    let joi_out_sh = [a as i64, v as i64];
                    self.joi.run_into_with_scratch(
                        &[
                            (c"enc", OrtTensorIn::f32(&ws.joi_enc, &joi_in_sh)),
                            (c"dec", OrtTensorIn::f32(&ws.joi_dec, &joi_in_sh)),
                        ],
                        &mut [(
                            c"logits",
                            OrtTensorOut::f32(&mut ws.joi_logits, &joi_out_sh),
                        )],
                        &mut ws.ort_scratch,
                    )?;
                    ws.phases.joiner_ort_calls += 1;
                }
                let tj1 = thread_cpu_ns(prof);
                ws.phases.joiner_cpu_ns += tj1.saturating_sub(tj0);

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
        let t_greedy1 = thread_cpu_ns(prof);
        ws.phases.greedy_cpu_ns += t_greedy1.saturating_sub(t_greedy0);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use ultra_gp::R1Precision;

    use super::*;

    #[test]
    fn decoder_vectors_match_across_batch_shapes_m1_m4_m16() {
        let root = std::env::var("JSR_DATA")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("/mnt/xbabe3-external/jsr"));
        let Some(bdir) = [
            "bundles/r1-d1-v1.1b",
            "bundles/r1-d1-reviewed-20261120-v2",
            "bundles/r1-d1",
        ]
        .into_iter()
        .map(|rel| root.join(rel))
        .find(|p| p.join("manifest.json").exists()) else {
            return;
        };
        let eng = OrtTransducerBackend::load(&bdir, R1Precision::Fp32, 1, false)
            .expect("load R1 bundle for decoder batch-shape test");
        let d = eng.bundle.streaming.joiner_dim;
        let v = eng.bundle.decoder.vocab_size as i64;
        let ctxs: Vec<[i64; 2]> = (0..16i64)
            .map(|i| [(i * 131) % v, ((i + 3) * 977) % v])
            .collect();

        // Evaluate each context individually at m = 1.
        let mut single_vecs = vec![0.0f32; 16 * d];
        for (i, ctx) in ctxs.iter().enumerate() {
            eng.dec
                .run_into(
                    &[(c"ctx", OrtTensorIn::i64(ctx, &[1, 2]))],
                    &mut [(
                        c"dec",
                        OrtTensorOut::f32(&mut single_vecs[i * d..(i + 1) * d], &[1, d as i64]),
                    )],
                )
                .expect("dec m=1");
        }

        // Compare against m = 4 and m = 16 batched calls bit-for-bit.
        for m in [4usize, 16] {
            for (chunk_idx, chunk) in ctxs.chunks(m).enumerate() {
                let flat: Vec<i64> = chunk.iter().flat_map(|c| [c[0], c[1]]).collect();
                let mut batched = vec![0.0f32; m * d];
                eng.dec
                    .run_into(
                        &[(c"ctx", OrtTensorIn::i64(&flat, &[m as i64, 2]))],
                        &mut [(
                            c"dec",
                            OrtTensorOut::f32(&mut batched, &[m as i64, d as i64]),
                        )],
                    )
                    .expect("dec batched");
                let expected = &single_vecs[chunk_idx * m * d..(chunk_idx + 1) * m * d];
                for (a, b) in expected.iter().zip(&batched) {
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "decoder vector bit mismatch at m={m}: {a} vs {b}"
                    );
                }
            }
        }
    }
}
