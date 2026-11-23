# H47 / alwayson-bench: always-on competitor rows, the ULTRA-AO benchmark, gain panels and Q5b data

Lane: `alwayson-bench`. Date: 2026-11-23. Status: closed by the owner's research wrap-up at 09:55Z.

**Every number here is exploratory.** The speech is LibriSpeech DEV only; nothing from test was used. Libri-Light small appears only in the Q5b stream, which has no references. No CPU or energy number is quoted: those come only from xbabe0 through bench-open, and this lane never touched xbabe0.

The receipts live on the data drive, under `/mnt/xbabe3-external/jsr/evals/alwayson-bench/` and `/mnt/xbabe3-external/jsr/alwayson-bench/`. Each receipt holds the traces, 1 Hz RSS logs and run JSON, with model, binary and audio hashes and the host. `results/` holds the summaries, and `MANIFEST` lists the sha256 of every file in this folder.

## 1. What was measured

| Item | Systems | Status |
|---|---|---|
| False words on eval-lane's ten 1 h non-speech blocks | Zipformer R0 (sherpa-onnx INT8), Moonshine Tiny and Small, Vosk Small and Large, Nemotron 560 and 160 ms, Whisper.cpp tiny, plus B0 C0 and B0 C1 | done; Nemotron 160 partial (4.07 h); C1 partial (3 h) |
| False words on 20 extra non-speech hours (new seeds) | Moonshine Tiny and Small (20 h each), Zipformer (9 h) | the rest were stopped by the wrap-up |
| Zipformer on the 27.56 h continuous stream | no endpoint (paper setting) and sherpa endpoint resets | both stopped early (§3) |
| ULTRA-AO tier A (21 cells, 21.7 h) | B0 C0, Zipformer, Vosk Small, Moonshine Tiny (complete); B0 C1 (20 cells); Moonshine Small (11 cells) | Nemotron, Whisper, Vosk Large and Zipformer-endpoint tier A rows were stopped |
| Gain panels (tier G), raw and with a warm causal AGC | B0 C0 | done |
| Q5b fresh data for the gate's default-on test | — | built, hashed and handed to h47-sleep |

## 2. Scoring rules (named in every table)

- **eval-lane rule:** `score_alwayson.py` assigns each word at its emission time minus a 0.5 s allowance.
- **Plain tolerant rule:** a word goes to the last utterance that started at or before its emission, if it falls within a tail after that utterance's end. Otherwise it is a gap (false) word.
- **Lag-compensated tolerant rule (the cross-system comparator):**
  - Each emission is shifted back by the system's own median lag before assignment, with a 3.5 s tail.
  - The shift is one global value per system: the median of per-cell lag p50 over the 12 gapped duty cells of the plain pass. It is recorded in `results/compensated_rule_shifts.json`.
  - eval-lane accepted it (09:14Z) with three conditions: the shift is measured and never tuned, there is one shift per system, and the rule is named in every table.
  - Reported lags always use true emission times.
  - Shifts: B0 C0 0.305 s, B0 C1 0.336, Zipformer 0.527, Vosk Small 0.291, Moonshine Tiny 0.713, Moonshine Small 0.695.
- **Assignment-free stream WER**, for contiguous cells (the panel): the final full transcript is aligned against all references in stream order (`results/panel_stream_wer.json`). On contiguous speech any assignment rule can misattribute late revisions.
- **Non-speech bounds:** exact Poisson (Garwood) one-sided 95%, or a block bootstrap if larger. All counts use the pinned normaliser `ultra-en-wer-norm/v1`.
- **For accuracy between systems, eval-lane's isolated full-dev WER stays primary.** Stream WER is the always-on secondary measure.

## 3. Results

### 3.1 False words on non-speech (normalised words; ours is H22ctl at 0 in 10 h)

| System | Words in eval-lane's 10 h | Extra hours (seeds 20261123 / 24) | Notes |
|---|---:|---|---|
| Zipformer R0 | 0 | 0 in 9 h | |
| Moonshine Tiny | 0 | 0 in 20 h | 30 h total, upper bound 0.10/h. VAD-gated: it skips decoding on non-speech |
| Moonshine Small | 0 | 0 in 20 h | 30 h total, upper bound 0.10/h |
| Nemotron 560 ms | 0 | – | |
| Nemotron 160 ms | 0 in 4.07 h | – | partial: the xbabe1 memory guard stopped it when other users' VMs took the memory; no text was ever shown |
| Vosk Small | 948 | – | "huh" at every forced 20.16 s endpoint in 6 noise blocks |
| Vosk Large | 1,790 | – | 179 per block in every block, silence included, mostly "the" |
| Whisper.cpp tiny | 1,438 | – | all in pink −20 dBFS, e.g. "(engine revving)" ×710 |
| B0 C0 (A1-49000 EMA INT8, continuous) | 6 | – | one single-letter token ("R" or "N") early in 6 noise blocks |
| B0 C1 (T = 1500 / R = 1000) | 0 in 3 h | – | partial |

Two lessons for the gate. Endpoint policy creates false words, not only the acoustic model: Vosk emits one LM-favoured word per forced close. "0 false words" alone no longer separates ULTRA from VAD-gated systems, so the comparison has to include first-word recall and latency.

### 3.2 Zipformer on the 27.56 h stream (`results/zipformer_endless27h_*.run.json`)

- **No endpoint, hypothesis read every chunk (paper setting):**
  - The 6 GB watchdog stopped the run at 6.98 h, with anonymous memory at 6,152 MB, growing super-linearly with hypothesis length.
  - Prefix scores:
    - eval-lane rule: WER 6.906%, 98 gap words;
    - tolerant rule: 6.780%, 10 gap words (2.3/h, upper bound 3.9/h), lag 531 / 730 ms.
  - Ours (H22ctl) on the same prefix: 9.763% / 9.759%, 1 / 0 gap words.
- **Read-policy control (40 min of speech, 2,640 words each):**

  | Read policy | Reads | Anonymous memory |
  |---|---:|---|
  | Every chunk | 7,499 | 137 → 191 MB |
  | Every 60 s | 39 | flat at 141 MB |
  | Only at the end | 0 | flat at 141 MB |

  The O(n) whole-hypothesis read is what grows memory, not the recogniser state.
- **sherpa endpoint resets:**
  - The wrap-up stopped the run at 18.0 h, after 16,939 resets. Anonymous memory went from 136.6 to 151.2 MB, a linear +0.55 MB per audio hour.
  - Prefix: tolerant WER 6.726%, 28 gap words in 11.12 gap hours (2.5/h, upper bound 3.45/h), lag 529 / 729 ms.
  - Ours on the same prefix: T0 9.773% and T1500 / R1000 9.145%, both with **0 gap words** (upper bound 0.27/h).
- **Ours over the full 27.56 h (tolerant rule):** T0 9.89% and T1500 / R1000 9.24%; 0 gap words in 17.06 h (upper bound 0.18/h); lag 310 / 439 ms.

### 3.3 ULTRA-AO tier A, compensated rule, 21 matched cells (`results/tierA_*`)

| System | WER (%) | First-word recall | First word correct by +300 ms | Gap words / 12.97 h | Lag p50 (ms) |
|---|---:|---:|---:|---:|---:|
| B0 C0 | 11.95 | 0.782 | **0.391** | **5** | **305** |
| Zipformer R0 | **7.57** | **0.855** | 0.059 | 20 | 529 |
| Vosk Small | 34.46* | 0.696 | 0.300 | 392 | 292 |
| Moonshine Tiny | 13.26 | 0.800 | 0.028 | 125 | 715 |

\* Vosk's WER is inflated on the contiguous panels by assignment (§2). Its assignment-free panel stream WER is 17.07% (Q) and 22.52% (N20).

**Panel stream WER (Q / N20):**

| System | Q (%) | N20 (%) |
|---|---:|---:|
| B0 C0 | 8.13 | 10.74 |
| B0 C1 | 8.05 | 10.78 |
| Zipformer | 5.96 | 6.43 |
| Moonshine Tiny | 9.92 | 13.02 |
| Vosk Small | 17.07 | 22.52 |

**Reset policy, B0 C0 vs C1 (20 matched cells):**
- Stream resets recover most of the first-word loss after pauses: first-word recall goes from 0.783 to **0.857**, WER from 12.19% to **10.47%**, and gap words from 5 to **0**.
- The cost is speed: first word correct by +300 ms falls from 0.392 to 0.298, and lag p50 rises by 30 ms.
- The gains are largest in noise and on short items. On duty05_r6 N20 (single words at 6 per minute), WER falls from 53.4% to 26.8% and first-word recall rises from 0.50 to 0.72.

Partial rows on 11 matched cells: B0 C0 16.26% / 0.699; B0 C1 11.99% / 0.827; Moonshine Small 10.43% / 0.797 with lag p50 712 ms.

First-word recall by onset class and pre-gap is in `results/tierA_cells.json`. These splits are confounded with item kind, so treat them as descriptive only.

### 3.4 Gain panels and warm AGC, B0 C0 always-on (`results/gain_agc_B0C0.json`)

The AGC is causal (`aobench/agc_stream.py`), with a speech-gated level estimate, attack 0.3 s, release 3 s and gain bounded −10…+40 dB. Its target is calibrated on the 0 dB panel, and its state carries across all 730 utterances.

| Input gain | WER raw (%) | WER AGC (%) | First-word error raw | First-word error AGC |
|---|---:|---:|---:|---:|
| 0 dB | 8.13 | 8.05 | 10.1% | 10.1% |
| −10 dB | 8.06 | 8.08 | 10.5% | 10.1% |
| −20 dB | 8.76 | 8.09 | 11.1% | 10.0% |
| −30 dB | 10.20 | 8.17 | 12.6% | 10.3% |

- **Recovery:** 98-107% of the WER loss and 95-115% of the first-word loss at −20 / −30 dB.
- **Paired speaker bootstrap, AGC − raw at 0 dB:** −0.08 pp [−0.21, +0.04].
- This always-on row met the lead's AGC criteria 1 and 2. The lead's 10:20Z decision (warm AGC flag; gain augmentation dropped from v3.0 training) cites it.

### 3.5 Q5b data (`results/q5b/`)

Q5b is 52.0 h that no earlier run had decoded, built for the gate's default-on false-word test. The sha256 list (file sha256 7fd33038…, 44 files) was posted at 10:00:54Z, before any decode.

- **Non-speech, 30 h:** eval-lane's `gen_alwayson.py nonspeech` (8b963231…), unchanged, with seeds 20261125-27.
- **Junction stream, 22.0 h:** `gen_q5b.py` (39ebf93e…), seed 20261125.
  - Contents: 3,559 Libri-Light small chunks from 271 speakers (11.17 h of speech), cut at Libri-Light's own VAD pauses, and 1,779 speech → noise junctions.
  - It reuses gen_alwayson's structure and noise code: U(2, 20) s gaps, alternating digital silence and the 9 colour × level noises.
  - Each Libri-Light member was fetched by HTTPS byte range and checked against the md5-verified archive index (`FETCH.json`).
  - There are no references; Q5b's rule (gated minus twin) needs none.

## 4. The benchmark: ULTRA-AO v1 (`results/tier_sets/`, `results/code_sha256.json`)

The generator is `aobench/gen_ao_bench.py`. Its output format is identical to eval-lane's always-on stream (`part-N.wav`, `index.jsonl`, `refs.jsonl`, `words.jsonl`, `MANIFEST.json`), so `score_alwayson.py` scores every cell as is. Each speech item carries MFA onset phone and class, a quiet-onset flag and its pre-gap.

| Tier | Contents | Streams | Hours | Generator | Where |
|---|---|---:|---:|---|---|
| A | duty 5 / 20 / 50% × 1 or 6 onsets per minute × Q / N20 (1 h each); 730-utterance panel × Q / N20; torture T1 (unvoiced fricative onset) and T2 (quiet onset) at 3 s and 30 s gaps; T3 (onset after 8 s of noise) | 21 | 21.69 | 3dd55780 | data drive `aobench/tierA/` (INDEX 6badba61…) |
| B | the full grid at 2 h per cell × Q / N20 / N10; panels; T1 / T1v / T2 at 0.4 / 3 / 30 s gaps; T3; T4 at −20 and −30 dB | 52 of 53 (T4 −40 dB not built) | 96.69 | 9dd6b183 (T4 −30: e73e4fad) | xbabe1 / xbabe2 `~/jsr-data/alwayson-bench/aobench/tierB/` |
| G | the 730-utterance panel at 0 / −10 / −20 / −30 dB, raw and AGC, with per-utterance WAVs | 8 | 11.69 | 96233083 + `agc_stream.py` f7bb67ef | data drive `aobench/tierG/` |
| Extra non-speech | 10 conditions × seeds 20261123 / 24 | 20 | 20 | gen_alwayson 8b963231 | data drive `aobench/nonspeech-extra/` |

- **Reproducibility:**
  - Tier B was split across two hosts after a memory stop. duty50_r6_N20 (xbabe2) and duty50_r6_Q (xbabe1) have identical segments and items.
  - h47-sleep's Mac rebuild of the 29 tier B torture cells matched the x86 PCM hashes byte for byte.
  - Every generator version that built a tier is kept by hash in `aobench/versions/`.
- **Known deviation:** the 5%-occupancy, 6-onset-per-minute cell realises 6.7% (tier A) and 6.5% (tier B) occupancy, because DEV has too few short items. Reports use the realised value.

## 5. Stopped or not done (wrap-up, 09:55Z, by recorded PID / PGID)

- **Stopped or never run:**
  - Nemotron 560 / 160, Whisper tiny, Vosk Large and Zipformer-endpoint tier A rows;
  - 10 Moonshine Small tier A cells;
  - all tier B runs;
  - B0 C0 / C1 on the extra non-speech hours;
  - 9 B0 C1 cells, and Zipformer on 11 extra non-speech hours.
- **Never measured:** CPU, wake and energy on xbabe0.
- **Planned and not done:** Vosk Large tier A and the paced (wall-clock) runs.

## 6. Code (on the data drive, hashes in `results/code_sha256.json`)

- **`drivers/`:**
  - the competitor drivers: `zipformer_alwayson.py`, `moonshine_alwayson.cc`, `vosk_alwayson.py`, `nemo_alwayson.py`, `whispercpp_alwayson.py`, `common.py`;
  - the guarded runner `runq.py` (CPU band, memory floor, adoption by PGID) and `handover.sh`;
  - the finalizers.
- **`aobench/`:**
  - the generators: `gen_ao_bench.py`, `gen_q5b.py`, `agc_stream.py`;
  - the scorers: `score_ao_bench.py` (`--assign`, `--shift-s`), `system_shift.py`, `stream_wer.py`, `rollup_tier.py`, `export_cells.py`.
