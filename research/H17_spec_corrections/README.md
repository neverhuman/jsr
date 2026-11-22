# H17: spec corrections for the next waves

Status: note only. No training run and no engine or stream edit on `lane/grok`. Every speed and WER figure already in `docs/ultra/STATUS.md` stays exploratory.

The six documents in `tips/` were ranked in `docs/ultra/NEXT_WAVES_PROPOSAL.md` (H12–H16). This note records claims those specs make that our measurements or the published recipes do not support, and three live bugs H12–H15 did not schedule. `lane/runtime` owns bugs 1 and 2. `lane/engine` owns bug 3. This lane does not patch those trees.

## Qualification bugs (owners, not this lane)

1. **Fleet offset applied twice.** `open_slot` stores `(logical + offset) % n` as the slot's utterance id. `utt_samples` must index that stored id. Adding `utt_offset` again returns a different length, so `simulate_grid` drops a finished utterance from `audio_completed_samples` (`cur >= full_len` fails). Fleet workers pass `(w * 137) % n` (`native_engine_fleet.rs`). Offset 0, which is what a pre-sliced single process uses, hides it. Sherpa already indexed the stored id. On `lane/runtime` (unmerged), native and ORT `utt_samples` index the stored id. The spin test is in `tests/grid_sim_p0.rs`. The offset pin is not: lengths `[100, 200, 300]`, offset 1, one unpaced stream, completed ids `[1, 2, 0]`, `audio_completed_samples == 600`.
2. **Persistent decode failure spun the grid.** `decode_ready_step` stored `last_error` and still returned `ready_n > 0`. The inner loop stopped only when `ready_n == 0`, so `simulate_grid` never returned and the error check after it never ran. `lane/runtime` checks `take_error` before and after `decode_ready_step`. A test bank must set the error only inside `decode_ready_step` and clear it in `take_error`. An error that is always present fails in `accept` first, and the decode call count stays 0.
3. **Ternary plus `act7` is two models.** `linear_t` forces `act7` off for the call. `rows > 8` (`TLUT2_MAX_ROWS`) uses `linear_i8`, which honours `act7` (qmax 63 versus 127). Crossing 8 rows changes the model. `vpmaddubsw` saturates for u8 codes at `[0, 254]` with 8-bit weights, so T-LUT2 cannot honour `act7` by keeping those codes. Refuse the combination at the start of `Engine::decode` until both kernels share one width. Leave `with_act7` as `Self` (the only caller is `engine_eval`, default false). Do not change `encoder/block.rs` or `linear_t` to paper over it. `lane/engine` owns this; it is not in their tree yet.

## Trainer facts, not edited here

`lane/trainer` owns `python/ultra/train/**`.

- `Trainer.payload` saves qkv caps and does not save `loss_ema`, `best_ema`, or `best_wer`. `resume` leaves them `None`. After a resume the divergence guard's baseline starts over (`trainer.py` `payload` / `resume`).
- `init_cmvn` still estimates from the first 8 batches (`trainer.py`).
- `data.py` `_decode` speed-perturbs, then drops audio longer than `max_seconds` (default 20). A factor of 0.9 lengthens the utterance (`output length = input / f`), so slowed copies of 18–20 s utterances are the ones removed. Fast copies stay. Worth counting before H13 treats speed perturbation as uniform coverage.

## Claims not to plan on

| Claim in the specs | What we already measured or published |
|---|---|
| Zen 2 is bound by one multiply pipe, so an unsigned 7-bit tile is 1.33–1.6× | `act7` on the real kernel was +3–10%, and single-stream did not move. H15 B2 benches the tile before anyone writes it into the engine. |
| About 40% of single-stream cost is weight fetch | Batching amortises ≤ 8% on Zen 2 and about 0 on M3. The engine is compute-bound, about 62 GMAC/s per core, not 27–36. |
| Ternary wins once the weights fit in L3 | R4T fit and still lost to int8 at batch. Compute-bound. |
| HYPER 1.95–2.35% dev-clean and 0.0039 CPU-s/audio-s on LibriSpeech 960 h | That stacks an unbuilt architecture, an unmeasured kernel, and a recipe we have not run. It is not a target. |
| u8 activations at `[0, 254]` are exact on AVX2 | `vpmaddubsw` saturates (`2·254·127` exceeds i16). Exact only at 7-bit weights or 7-bit activations. |

## Kill thresholds for H13 and H14

**H13 is the icefall recipe, not a promise to match R0.** R0 Zipformer (2.85% dev-clean in our harness) is the icefall streaming Zipformer: pruned RNN-T ([arXiv:2206.13236](https://arxiv.org/abs/2206.13236)), BPE-500, on the order of 30 epochs, then average the last checkpoints (`pruned_transducer_stateless7_streaming`, epoch 30, avg 9, in the icefall LibriSpeech results). A streaming Conformer on full LibriSpeech, stopped near 25 epochs, landed around 3.9–4.7% test-clean in icefall's own `pruned_transducer_stateless` table, not 2.85%. Pruned loss + a ~512-piece vocabulary + EMA can close a training-economics gap. It does not, by itself, separate recipe from architecture. Keep the public causal-Zipformer control (NEXT_WAVES A10) for that.

**H14's published gain is small.** Mode-consistency for a unified RNN-T (MCR-RNNT, [arXiv:2604.19079](https://arxiv.org/abs/2604.19079), April 2026) keeps offline WER and trims the streaming gap. The paper's open-leaderboard average is about 5.8–5.9% for a 0.6B model trained on much more than LibriSpeech. It does not support a 55–70% relative cut. Run H14 only after H12's error table says the remaining gap is the streaming penalty, and kill it if the relative gain on dev-other is under 5% at equal exposure.

**FastEmit is a latency tool, and it can cost WER.** On LibriSpeech, FastEmit ([arXiv:2010.11148](https://arxiv.org/abs/2010.11148)) took a late-emitting ContextNet from 4.4%/8.9% to 3.1%/7.5% and p90 partial latency from 210 ms to 30 ms. On Voice Search the same idea has cost about +0.2 absolute WER. Use it only if H12 shows late emission. Report endpoint delay separately from WER.

**The certified joiner cannot pay for itself yet.** Frontier §11's own stop rule is ≥ 3% whole-service CPU. The joiner is about 3% of the CPU while the encoder is the rest. A blank-first exact argmax can land with the bit-identical engine work. Full branch-and-bound waits until that work shrinks the encoder.

**KV state is a density limit.** 6.7 MiB per stream matters when many streams share a CCX. It is not why a single stream is slow. INT8 or FP16 K/V and the left-context sweep (H15 B3) are the density levers. A Gated DeltaNet pyramid waits until a matched Conformer control exists.

## Owner decisions still open

R0 as a teacher, R0 weights inside `ultra-engine`, an AVX-512 host, and data beyond LibriSpeech. This note does not start any of them.
