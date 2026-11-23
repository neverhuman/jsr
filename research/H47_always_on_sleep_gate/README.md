# H47: always-on sleep gate (silence nearly free, first word kept)

**Status:** PRE-REGISTERED (h47-sleep's experiments A0, O and Q), written 2026-11-23 from 06:00Z and committed before any run of this wave. At that time no run of A0, O or Q existed; the only always-on numbers on the 27.56 h stream were runtime-lane's H22ctl runs (T=0 9.891%, T=1500/R=1000 9.240%, isolated 8.939%; 03:11Z post). Results are appended below the line "Results" later; amendments are dated and written before the runs they affect. The branch was rebased onto main d72169c (PR #75) before its PR, so the commits have new hashes with unchanged author dates: pre-registration b79c885 → bcd11e4 (06:14:23Z), amendment 1 9e04ef5 → ef48e74, amendment 2 8e10c80 → 8e71c8e, amendment 3 74f7f25 → bb0f8b7, amendment 4 9ea848b → 5703b07. Every number of this wave is **exploratory** unless a section says otherwise.

**Split of the wave** (lead's DECONFLICT post, 2026-11-23 05:58Z; one owner per file):
- runtime-lane's swarm **implements** the gate: `crates/ultra-gate`, `crates/ultra-alwayson` (sleep ladder, PCM ring, one-pass replay, `ultra-alwayson-check`), `--uclamp-max`, and `skip_gate_sim.py` with its own design memo. Its sections go below its own heading in this folder.
- alwayson-bench builds the standard always-on benchmark (`gen_ao_bench.py`).
- bench-open runs every CPU and power measurement on the quiet host xbabe0 (TIMING queue).
- **Files in this folder:** this README (h47-sleep's pre-registration, amendments and results; results JSON `results/A0_*`, `results/O_*`, `results/Q_*`), the swarm's skip-gate simulation receipts from PR #75 (`results/skip_gate_sim_summary.json`, `results/stream_segments_summary.json`), and the Rust gate's implementation notes in `GATE_IMPLEMENTATION.md` (PR #77, lead's ruling 06:43Z).
- **h47-sleep** (this pre-registration) writes no gate code. It runs (A0) the xbabe0 assay of what non-speech costs today, (O) the offline oracle speech gate, which bounds the savings and the restart damage, and (Q) the independent qualification of the swarm's gate against an ungated twin.

## Question
ULTRA spends about 0.035 CPU-s per audio-s in silence and in speech alike, and more in silence with reset-with-replay. Moonshine's VAD-gated decoder spends about 0.01 in silence. Can the recogniser sleep in non-speech, so that silence becomes nearly free, without losing the first word after a pause?

## Fixed setup (all parts)
- **Checkpoint B0:** A1 step 49,000, EMA, INT8 bundle `/mnt/xbabe3-external/jsr/evals/eval-lane/export/A1-49000-ema/int8/` (`weights.int8.safetensors` sha256 `4701787c94d320953c42367bf0cf3be1ce2378e66bc811935af9f0f135b1e285`, `arch.json` `b1581cbb1f22707ed16b146b0b04bdbcc181cdb6def5a68ad1108cc9af9db39f`). Dense S, 16 blocks, d 384, left context 128 frames (5.12 s), chunk 4 (160 ms), greedy, 20 ms packets.
- **Code:** main `e61121e`. `ultra-engine-endless` = `cargo build --locked --release -p ultra-engine --bin ultra-engine-endless` at that commit (H0 build sha256 `a067a30735e59c098027189e4eba7bd5acb58c49b096396637cdd63e4c6c289d`, rustc 1.95.0). The oracle decoder `h47-oracle-decode` (below) uses only ultra-engine's public API at the same commit.
- **Scripts and tools** live on the data drive under `/mnt/xbabe3-external/jsr/h47-sleep/` (research folders hold documents and results only) and are named here with their sha256 at pre-registration: `scripts/make_assay_inputs.py` `3beac281…`, `scripts/h47_assay.sh` `47ddbe57…`, `scripts/oracle_segments.py` `b7869181…`, `tools/oracle-decode/src/main.rs` `958e206b…`.
- **Data, DEV only; the test sets stay sealed:** eval-lane's always-on set `/mnt/xbabe3-external/jsr/evals/eval-lane/alwayson/` (generator `gen_alwayson.py` sha256 `8b963231…`): the 27.56 h stream (`stream/MANIFEST.json` sha256 `40f8c63d…`, `index.jsonl` `ff26c1c7…`, `words.jsonl` `78dde38c…`, `refs.jsonl` `1d761ed5…`; 5,567 utterances = 10.51 h of speech, 5,567 gaps of U(2, 20) s = 17.06 h: 8.61 h digital silence and 8.45 h white / pink / brown noise at −40 / −30 / −20 dBFS) and the ten 1 h non-speech blocks (`nonspeech/MANIFEST.json` `fceaa363…`).
- **Scoring:** eval-lane's `score_alwayson.py stream --global` (sha256 `bf6f27cb…`; allowance 0.5 s) calling the pinned `ultra-score` (`d784f8ca…`, normaliser `ultra-en-wer-norm/v1`). Paired comparisons: `ultra-score A.score-input.jsonl B.score-input.jsonl --seed 20261122` = paired speaker-cluster bootstrap of B − A, 10,000 replicates, one-sided α = 0.05 (unadjusted; the family is stated with each table).
- **Hosts:** CPU and energy only on xbabe0 through bench-open. Everything else (decodes, scoring) on H0 at nice 19 on idle cores, CPU ≤ 80%, outputs under `/mnt/xbabe3-external/jsr/h47-sleep/`.

## A0: what non-speech costs today (xbabe0, bench-open's TIMING queue)
- **Why:** the designs claim reset-with-replay makes silence cost about 1.6× speech (a reset every 40 blank frames, each re-encoding 24 frames chunk by chunk: (40 + 24) / 40 = 1.6 for the encoder). Our "+4.8%" for the resets came from the DRAM-contended H0 and is not a cost measurement.
- **Binary and inputs:** `ultra-engine-endless` (main `e61121e`), B0 as above. Inputs (`/mnt/xbabe3-external/jsr/h47-sleep/inputs/MANIFEST.json`, built by `scripts/make_assay_inputs.py`): `silence600.wav` (first 600 s of the digital-silence block, sha256 `e14ff9e3…`), `pink-30_600.wav` (first 600 s of the pink −30 dBFS block, `dc6c2603…`), `speech600.wav` (the stream's first 85 utterances concatenated with no gaps, cut at 600 s, `9ac9ad46…`).
- **Arms:** A = no reset (`--endpoint-reset-ms 0`); B = reset-with-replay T = 1500 / R = 1000 (`--endpoint-reset-ms 1500 --endpoint-replay-ms 1000`, the adopted always-on configuration); C (only if runtime-lane's one-pass replay exists when the entry runs) = B with the one-pass replay, same inputs.
- **Protocol:** `scripts/h47_assay.sh` (sha256 `47ddbe57…`): one pinned core per repetition, sequential, unpaced (design time), 3 repetitions, arm order alternating by repetition (ABBA), hashes of inputs and weights checked first. Metric: CPU-s per audio-s from the binary's own JSON (the decoding thread's schedstat run time); bench-open's meter may wrap each run.
- **Reported:** mean, min and max per (input, arm); the ratios silence/B ÷ speech/A and pink/B ÷ speech/A; the number of resets.
- **Expected `[from the designs]`:** speech/A ≈ 0.035; silence/A ≈ 0.030–0.035 (the decoder's blank certificate makes silence slightly cheaper); silence/B ≈ 1.5–1.6 × speech/A. If silence/B < 1.2 × speech/A, the designs' cost model is wrong and the savings estimates below are rescaled from the measurement.
- **Use:** the measured speech/A is the active cost `c` in the oracle's savings bound, and silence/B is the denominator ("ungated") of Q's non-speech gate until the swarm's own ungated twin is measured.

## O: offline oracle speech gate (bounds, not a product result)
An oracle gate knows where the reference speech is. It keeps the recogniser asleep outside it; at each speech onset it restarts the recogniser from a fresh state (`Engine::create_stream`) on R ms of pre-roll, runs it until H ms after the speech ends, then closes the segment (`input_finished`, flush) and sleeps. Two segments that would overlap are merged (the gate would stay awake across the short gap). Each segment is therefore a standalone decode of `pcm[s0 .. e)`, which is exactly the segment-isolation semantics of the designs' session.
- **Tooling** (verification tooling, outside the engine crates, on the data drive): `scripts/oracle_segments.py` builds the segment lists; `tools/oracle-decode` (`h47-oracle-decode`, a standalone Rust crate with its own workspace, public ultra-engine API only, path dependency on `crates/ultra-engine` at `e61121e`) decodes each segment in a new stream with `ultra-engine-endless`'s committed-word emitter and writes one trace on the global stream timeline plus a segment ledger. Traces are scored by `score_alwayson.py` exactly like the continuous runs.
- **Arms** (segment lists in `/mnt/xbabe3-external/jsr/h47-sleep/oracle/segs/`, sha256 O0 `7dfacc8d…`, O1 `2d4ad6cc…`, O2 `6662dace…`, O3 `b12cabdd…`; duty = decoded audio / stream audio):

  | Arm | Segment | Segments | Duty |
  |---|---|---:|---:|
  | O0 identity | `[utt_start, utt_end)`: each utterance decoded as its own file (isolated decoding, on the stream timeline) | 5,567 | 38.1% |
  | O1 exact oracle | `[utt_start − 1000 ms, utt_end)`: sleeps exactly on the inserted gaps, 1 s pre-roll of gap audio | 5,567 | 43.7% |
  | O2 exact oracle + hangover | `[utt_start − 1000 ms, utt_end + 1500 ms)`, merged on overlap | 5,423 (144 merges) | 52.1% |
  | O3 ideal word-onset detector | `[first MFA word − 1000 ms, last MFA word end + 1800 ms)`, merged on overlap (a zero-delay detector on the words that closes about T_idle after the last token) | 5,528 (39 merges) | 49.9% |

- **Comparators**, same checkpoint, build and scorer: C0 = continuous, no reset (`ultra-engine-endless --endpoint-reset-ms 0` over `part-1..4.wav` as one stream); C1 = continuous with reset-with-replay T = 1500 / R = 1000 (the adopted always-on configuration). These two are also B0's 27.56 h stream receipts asked for in H43.
- **Metrics per arm:** pooled / dev-clean / dev-other WER; false words in the 17.06 h of gaps (count, per hour, Garwood one-sided 95% upper bound `χ²(0.95; 2k+2) / 2T`); word lag p50 / p95 / p99; first-word recall (below); duty and wakes per hour; estimated CPU-s per stream-s = duty × `c` (A0's speech/A), stated as an estimate.
- **First-word recall** (fixed now): an utterance's first word is recalled if the first reference word (lower case, punctuation except apostrophes stripped, as `score_alwayson.py`'s `norm`) is matched to an identical hypothesis word on `score_alwayson.py`'s minimum-edit alignment of the utterance's assigned words. Reported over all utterances and over "after sleep" utterances (preceding gap ≥ 3 s, plus the first utterance of the stream). Also first-three-words-all-recalled. Paired differences by speaker-cluster bootstrap (10,000, seed 20261122).
- **Comparisons (family of 6, reported with Holm-adjusted one-sided intervals and unadjusted ones):** WER O1 − O0 (cost of the pre-roll), O2 − O1 (hangover and merges), O2 − C1 and O3 − C1 (oracle sleeping against today's always-on configuration), C1 − O0 and C0 − O0 (stream penalties against isolated decoding).
- **Readings fixed now:**
  - The **restart damage** of sleeping is O2 − C1 and O3 − C1. If their upper bounds are ≤ +0.10 pp, a gate that wakes on time loses nothing to restarts, and the real gate's accuracy risk is entirely in late or missed wakes (Q). If the point estimate is > +0.10 pp, the restart itself costs accuracy and the gate must keep more context (longer R, warm state), which this wave does not build.
  - O1 − O0 > +0.10 pp would mean that 1 s of gap audio before the speech hurts, so R should shrink; < −0.10 pp, that the pre-roll helps.
  - The **savings ceiling** is O3's duty: no gate that keeps R = 1000 ms and closes about T_idle after the last token can decode less audio on this stream. Its estimated CPU against C1's is the best case for the stream; on the ten non-speech blocks the oracle's duty is 0.
- **Sanity checks before any comparison (stop and fix the tool if one fails):** (1) on 20 utterances, O0's texts equal `ultra-engine-eval`'s (same API, no padding) on the same PCM; (2) every arm's ledger covers exactly its segment list; (3) score_alwayson assigns no word of O0 / O1 outside its utterance by more than the allowance.
- **Compute:** about 52 h of decoded audio for O0–O3 plus 55 h for C0 / C1, at 0.04–0.1 CPU-s per audio-s on H0: 5–11 CPU-h, at most 8 processes at once, nice 19 on idle cores. O arms split by segment ranges across processes (exact, since segments are independent); C0 / C1 run as one continuous stream each.

## Q: independent qualification of the swarm's gate (when it lands)
Run by h47-sleep on the swarm's binary at a pinned commit, against its **ungated twin** (`--no-gate`: same session code, segments closed and reopened on the same rule, never asleep; the designs' arm (ii)), and with C1 beside them. Gate values from the approved plan (Phase 3), definitions from `tips/v3/JSR_Final_Engineering_Plan.md` §10.1:
1. **Exactness precondition:** `ultra-alwayson-check` reports 0 mismatches on the stream's first hour (INT8 greedy); re-run here independently.
2. **Non-speech CPU ≤ 2% of the ungated twin:** process CPU-s per audio-s on xbabe0 (bench-open), gated ÷ ungated, on the ten 1 h blocks (bench-open may sample 15 min of each); pass if the ratio of means ≤ 0.02.
3. **≥ 2× lower incremental energy at 5% speech duty:** xbabe0, energy's power logger (core-register joules above the matched idle, package reported beside), alwayson-bench's p = 5% cell; pass if ungated ÷ gated ≥ 2 on the mean of 3 repetitions and every repetition ≥ 1.8.
4. **First-word recall after sleep ≥ continuous − 0.5 pp:** on the stream's after-sleep utterances and on the AO bench torture sets T1–T3, gated against C1; pass if the one-sided 95% lower bound of the paired difference ≥ −0.5 pp. INCONCLUSIVE if the bound is below −0.5 pp but the point estimate is ≥ −0.5 pp.
5. **False words ≤ 0.1 per hour at the upper bound:** gated system on the 30 h p = 0 set of the AO bench plus the stream's gaps; pass if the Garwood one-sided 95% upper bound ≤ 0.1 / h (needs ≥ 30 zero-event hours); a cluster-aware bound (block bootstrap) is reported beside it and the larger is used.
6. **Gate overhead ≤ 5% at 100% speech:** xbabe0 CPU and energy, gated ÷ ungated on the p = 100% cell; pass if the mean ratio ≤ 1.05.
7. **Stream accuracy:** WER(gated) − WER(ungated) on the 27.56 h stream ≤ +0.10 pp with one-sided upper bound ≤ +0.20 pp; false words in the gaps reported.
- Also reported: duty, wakes and false wakes per hour, words per false wake, first-token latency after sleep (paced, xbabe0 only), O3's duty as the target.

## Amendment 1 (2026-11-23, committed 06:21:47Z in ef48e74, before the rebase 9e04ef5): written before any O decode or comparison
What existed when this was written: the oracle decoder's sanity check (below) and the A0 runner's functional test on H0 (its numbers are not cost numbers and are not used). No O-arm decode, no C0 / C1 result and no comparison existed.
1. **Sanity check (1) is done with `ultra-engine-endless` instead of `ultra-engine-eval`:** both use the same engine API without padding, and `ultra-engine-endless` is the binary of the comparators. On the stream's first 16 utterances plus the first utterance of parts 2–4 (19 segments, 386 words), the O0 segments `[utt_start, utt_end)` and the O1 segments `[utt_start − 1 s, utt_end)` (the latter crossing the part boundaries) decoded by `h47-oracle-decode` give exactly the words of `ultra-engine-endless` run on each cut as its own file, with emission times equal to within 1 µs of float rounding (`/mnt/xbabe3-external/jsr/h47-sleep/sanity/sanity.json`, `scripts/sanity_oracle.py`). PASS.
2. **Where the decodes run:** H0 decodes at 4.5× real time today (0.20 CPU-s per audio-s, DRAM-contended), so a 27.56 h pass takes ~6 h there. The O arms and C0 / C1 therefore run on the Mac (xbabe4, NEON, the roster's h47 allocation) with the same commit, built there (`ultra-engine` without its Linux-only `wer` feature, which only gates scoring). **Precondition:** the Mac's O0 / O1 sanity traces must equal H0's word for word and time for time; otherwise the Mac is not used. C0 / C1 also keep running on H0 (x86, started 06:14Z); when they finish, their traces must equal the Mac's (a full-stream cross-ISA identity check) and any difference is reported.
3. **Multiplicity:** the pre-registration named Holm-adjusted intervals, but `ultra-score` reports bounds, not p-values. The adjusted bounds are therefore **Bonferroni** (α = 0.05 / 6 per comparison, one-sided), which is more conservative than Holm; unadjusted α = 0.05 bounds are reported beside them. The readings use the Bonferroni bounds.
4. **Scripts** fixed now: `scripts/h47_analyze.py` `bbb8a076…`, `scripts/sanity_oracle.py` `e29910b4…`, `scripts/mac_run_oracle.sh` `4e23c428…`.

## Amendment 2 (2026-11-23, committed 06:23:20Z in 8e71c8e, before the rebase 8e10c80): Q adopts codex-lead's controlling wake and power gates (lead's ALIGNMENT, 06:03:48Z), written before any Q run
Where codex-lead's 06:03:04Z release decisions are stricter or add a check, they replace or extend Q's list:
- **Controls:** every Q comparison shows four arms on the same stream, build and checkpoint: awake (C0), reset (C1 / the ungated twin), oracle (O3, this wave) and real wake (the swarm's gate).
- **Integrity:** zero lost, duplicated or reordered samples and zero revisions of emitted text, audited from the session ledger (sample sequence, exact handoff at wake, model reset, partial final chunk at EOF) and by `ultra-alwayson-check` (Q1).
- **Wake accuracy:** WER(real wake) − WER(ungated twin) ≤ +0.10 pp point and ≤ +0.20 pp one-sided 95% upper bound (Q7, unchanged).
- **Latency (new):** stable-word lag p95 / p99 of the real wake no more than +40 / +80 ms above the ungated twin's, on the stream (design-time clock from `score_alwayson.py`), and on xbabe0 paced replay when bench-open runs it (wall clock).
- **False words:** the **added** rate (real wake minus ungated twin, on the same non-speech hours) ≤ 0.10 / h at the one-sided 95% upper bound; the absolute rate and its Garwood bound are reported beside it (Q5 tightened from absolute to added, per §10.1 of the final plan).
- **Power (extended):** interleaved idle / capture / capture + detector / full service at 0 / 5 / 20 / 100% speech duty on the measured host (xbabe0 now; the owner's metered device when it arrives). Pass needs ≥ 2× lower incremental listening energy at 5% (Q3), a paired improvement at 20%, ≤ +5% gross upper bound at 100% (Q6 tightened from the mean ratio to the upper bound), and non-speech CPU ≤ 2% of the ungated twin (Q2). If capture alone exceeds 80% of A1's gross energy at 5% duty, the report says capture must be fixed first. The release-level "≥ 30% less active CPU, ≥ 20% less gross energy" is H54's and is reported, not gated, here.

## Amendment 3 (2026-11-23, committed 06:57:04Z in bb0f8b7, before the rebase 74f7f25; written after O0–O3 and before any run it names): three post-hoc diagnostic arms
What existed when this was written: O0–O3 decoded and scored (summary below under Results), C0 / C1 still decoding, no comparison involving C0 / C1. The arms below are **post-hoc diagnostics** chosen after seeing O0–O3; they are labelled so, and they cannot change the pre-registered readings of O0–O3.
1. **Observation A (false words at the close):** O2 has 18 and O3 10 "false words" in the gaps. 17 of O2's 18 are "SH" / "S" / "SP"-like fragments stamped exactly at the segment end (utterance end + 1.5 s), all in white or pink noise gaps: `input_finished` at a close inside noise makes the final partial chunk emit a fricative. The designs' session closes on complete chunks only (§6.2, e = s0 + 640 T − 240) and never flushes.
2. **Observation B (first word after a noise pre-roll):** first-word recall, O1 − O0, is −0.97 pp overall: −1.72 / −1.83 / −1.19 pp after −20 / −30 / −40 dBFS noise gaps and −0.36 pp after digital silence. A fresh recogniser that hears 1 s of stationary noise before clean speech loses first words.
3. **New arms** (decoder v2, `tools/oracle-decode/src/main.rs` sha256 `8e6edb99…`; its default path gives byte-identical traces to v1 on the sanity set; `--no-flush` closes a segment on complete chunks only and still commits the pending word):
   - **O3n:** O3's segments (`b12cabdd…`) with `--no-flush`.
   - **O3n-r500 / O3n-r300:** `[first MFA word − 500 / 300 ms, last MFA word end + 1800 ms)` (`00700b6c…` / `1fd60553…`; no merges), `--no-flush`. Sentinel-0's measured wake delay on this stream is p99 90 ms, max 450 ms (Q, above), so a 500 ms pre-roll would have covered every onset.
4. **Readings fixed now:** (a) if O3n has ≤ 2 gap false words and |WER(O3n) − WER(O3)| ≤ 0.05 pp, the session's close must not flush, and Q checks that the swarm's session never emits text at a close. (b) If O3n-r500 recovers at least half of O3n's first-word loss against O0 after noise gaps, at no WER cost beyond +0.05 pp, a 500 ms pre-roll is recommended to the swarm's B-6 sweep (with the measured wake delays); O3n-r300 shows the trend.
5. `scripts/h47_analyze.py` now reuses a score when it is newer than its trace (sha256 `fa49b0e4…`; no computation changed).

## Amendment 4 (2026-11-23, committed 06:59:51Z in 5703b07, before the rebase 9ea848b; written before the runs it names): the skip-gate simulation's criterion 2 with the swarm's real Sentinel-0 segments
What existed when this was written: O0–O3 results (below), the O3n / O3n-r500 / O3n-r300 runs in progress, C0 / C1 still decoding, no comparison with C0 / C1, and the swarm's segment list (runtime-lane-h47-sim, 06:52:59Z).
- **Input:** `/mnt/xbabe3-external/jsr/runtime-lane/h47_skipgate_segments/endless_27h_sentinel0.segments.jsonl` (sha256 `3a969cd8…`): PR #75's `sentinel0_segments.py` on this wave's Sentinel-0 trace of the stream (`hot.u8` `c9a51519…`). The session wakes on a hot hop with a 1 s pre-roll and closes after 1.5 s of gate-cold hops (a gate-driven close, not the designs' recogniser-blank close). 5,318 segments, duty 53.7%.
- **Arms:** **S1n** = those segments decoded by `h47-oracle-decode` v2 with `--no-flush` (complete chunks only, the designs' close; primary). **S1f** = the same with the flush (diagnostic for finding 1).
- **Readings fixed now** (codex-lead's wake gate, via the lead's 04:48Z criterion 2): WER(S1n) − WER(C1) ≤ +0.10 pp and one-sided 95% upper bound ≤ +0.20 pp (Bonferroni α/6 bounds reported beside the unadjusted ones); first-word recall after sleep, S1n − C1, one-sided 95% lower bound ≥ −0.5 pp; gap false words of S1n with the Garwood bound, and added over C1. S1f − S1n gives the cost of flushing at a close.

## Results

### A0: what non-speech costs today (quiet xbabe0, bench-open entry 14e, 06:49–06:59Z; exploratory)
Three repetitions on CPUs 4 / 24 / 28 (one per repetition, ABBA order), bench-open's meter, quiet-host gate and whole-host guard (all three clean; a contaminated round-3 attempt was set aside and re-run). Binary `a067a307…` (main e61121e), A1-49000 EMA INT8. Per-run JSON, receipts and bench-open's receipts: `/mnt/xbabe3-external/jsr/h47-sleep/runs/A0-xbabe0/` (`a0_summary.json` sha256 `0f1850c7…`, copied to `results/A0_xbabe0_summary.json`; aggregated by `scripts/a0_aggregate.py` `d17c5fa3…`).

| input (600 s) | A: no reset, CPU-s per audio-s (min–max) | B: T=1500 / R=1000 (min–max) | B ÷ A | resets in B |
|---|---:|---:|---:|---:|
| digital silence | 0.03531 (0.03447–0.03584) | 0.05009 (0.04862–0.05129) | 1.42 | 375 |
| pink noise −30 dBFS | 0.03575 (0.03460–0.03636) | 0.05052 (0.04879–0.05144) | 1.41 | 375 |
| speech only (85 DEV utterances) | 0.03594 (0.03463–0.03701) | 0.03592 (0.03457–0.03682) | 1.00 | 0 |
- **Without resets, non-speech costs the same as speech** (0.98–1.00×). **With reset-with-replay, non-speech costs 1.39–1.41× speech**: a reset every 1.6 s of blanks re-encodes 0.96 s chunk by chunk. The designs' 1.6–1.67× overstates it, and the "+4.8%" measured on contended H0 understated it.
- Spread across repetitions 4–7% (repetition 3 ran ~4% lower on every input). Arm A on pink noise printed 1 word per 600 s in every repetition, arm B none.
- Speech/A (0.0359) is the active cost `c` of the savings estimates below; silence/B is the "ungated" non-speech denominator until the swarm's ungated twin is measured.

### O: oracle arms against the continuous comparators (exploratory; decoded on the Mac 06:34–07:05Z, NEON, traces byte-identical to AVX2 on the sanity set; scored by `score_alwayson.py`; report `results/O_oracle_report_primary.json`)
| Arm | Pooled WER | dev-clean | dev-other | Gap false words (17.06 h; Garwood 95% upper / h) | First word, all / after sleep | Decoded duty |
|---|---:|---:|---:|---:|---:|---:|
| C0 continuous, no reset | 10.989% | 6.111 | 16.183 | 426 (27.1) | 75.26 / 74.93% | 100% |
| C1 continuous, T=1500 / R=1000 | 8.377% | 4.489 | 12.517 | 5 (0.62) | 88.92 / 88.81% | 100% (+37.9% replayed encoder audio; 39,146 resets) |
| O0 isolated [utt) | 8.273% | 4.461 | 12.332 | 1 (0.28) | 90.07 / 89.94% | 38.1% |
| O1 exact oracle, R 1 s | 8.371% | 4.536 | 12.454 | 0 (0.18) | 89.10 / 88.97% | 43.7% |
| O2 + 1.5 s hangover | 8.345% | 4.500 | 12.439 | 18 (1.57) | 88.84 / 88.97% | 52.1% |
| O3 word-onset detector | 8.297% | 4.372 | 12.476 | 10 (1.00) | 89.35 / 89.37% | 49.9% |

Paired comparisons (B − A; speaker-cluster bootstrap; one-sided 95% bounds unadjusted, then Bonferroni α/6):
| B − A | ΔWER pp | bounds | Bonferroni | Δ first word pp (after sleep) | Bonferroni lower (after sleep) |
|---|---:|---|---|---:|---:|
| O1 − O0 (pre-roll) | +0.098 | [+0.006, +0.189] | [−0.037, +0.230] | −0.97 (−0.97) | −1.64 (−1.65) |
| O2 − O1 (hangover) | −0.026 | [−0.059, +0.005] | [−0.074, +0.017] | −0.25 (0.00) | −0.41 (0.00) |
| **O2 − C1** | **−0.032** | [−0.114, +0.048] | **[−0.150, +0.082]** | −0.07 (+0.15) | −0.81 (−0.59) |
| **O3 − C1** | **−0.080** | [−0.165, +0.001] | **[−0.206, +0.039]** | +0.43 (+0.55) | −0.20 (−0.08) |
| C1 − O0 (reset stream vs isolated) | +0.104 | [+0.013, +0.196] | [−0.029, +0.239] | −1.15 (−1.12) | −1.93 (−1.91) |
| C0 − O0 (no-reset stream vs isolated) | +2.716 | [+2.482, +2.954] | [+2.378, +3.054] | −14.80 (−15.01) | −16.65 (−16.89) |

**Readings (as pre-registered):**
- **Restart damage:** O2 − C1 and O3 − C1 have Bonferroni upper bounds +0.082 and +0.039 pp, both ≤ +0.10 pp, and negative point estimates. **A gate that wakes on time loses nothing to restarts** against today's always-on configuration; its accuracy risk is in late or missed wakes and in what it does at a close (Q). Sleeping with restarts also keeps first words at least as well as C1 (O3 − C1 +0.43 pp).
- **Pre-roll:** O1 − O0 = +0.098 pp (≤ +0.10, so no reading triggers), but first-word recall falls 0.97 pp, almost all after noise gaps (amendment 3).
- **Savings ceiling** (estimates = decoded duty × A0's active cost 0.0359 CPU-s per audio-s): O3 ≈ 0.0179 CPU-s per stream-s against C1 ≈ 0.045 (A0 rates: speech 0.0359, non-speech 0.0503) and C0 ≈ 0.0356, i.e. ~60% below C1 and ~50% below C0 on this stream (38% speech). With the pre-roll replayed in one batched pass it would be ≈ 0.0164. On the ten non-speech blocks the oracle's duty is 0, against C1's 0.0503.
- **New, not pre-registered as a question but measured here:** A1-49000 without resets (C0) is far worse on the endless stream than H22ctl was (+2.72 pp against isolated, 426 false words in the gaps, mostly white / pink noise, from the first hour on; H22ctl: +0.95 pp, 4 words). Reset-with-replay (C1) or the gate is required for B0.
- Word lag p50 / p95 / p99: C0 302 / 436 / 487 ms, C1 329 / 457 / 509 ms, O arms 330 / 460 / 510 ms.

### Q (so far): the swarm's Sentinel-0 on the real stream (independent check of PR #75's criterion 1; exploratory)
- PR #75's `sentinel0.py` (22ad91d, sha256 `10608259…`, unmodified) as one never-reset gate over the whole stream (9,923,057 hops; `scripts/sentinel_stream.py` `f57deb12…`, which reproduces Appendix C's golden trace for block sizes 777 / 1000 / 16000 / 96000; run on the Mac, and an H0 run of the first 720,000 hops is byte-identical). Summary `results/Q_sentinel0_real_stream_summary.json`; trace `qual/sentinel/stream-mac/hot.u8` (`c9a51519…`).
- **Wake:** the first hot hop at or after the MFA first-word onset comes within the 1 s ring on **5,567 / 5,567** transitions; delay p50 / p95 / p99 / max = 10 / 60 / 90 / 450 ms (worst gap types pink −40 and white −30). Criterion 1 (≥ 99.8%) holds on the real gaps, not only on PR #75's synthetic pre-onset background.
- **Sleep opportunity:** the gate is cold over the last 80 ms of 98.7% of the gaps; the exceptions are all brown noise (35 / 310 at −30 dBFS, 30 / 310 at −20, 4 / 309 at −40).
- **Hot hops inside the gaps:** silence 0.15%, brown −40 0.7%, brown −30 / −20 7–11%, pink and white ~13% (mostly the ~1.5 s the floor needs to forget the utterance's quiet tail when a noise gap starts).
