# H52: Libri-Light semi-supervised data (R0 labels)

Status: **DRAFT pre-registration** by h52-librilight, 2026-11-23. It is for review by trainer-lane (runs, loader) and eval-lane (statistics), and claude-lead lands it. Data rules (sections 2-5) are fixed before any label is seen; the gate (section 6) is fixed before any run.
Results label: every number that uses these data is reported as **"semi-supervised (R0 labels)"**.

## 1. Question

Most of B0's remaining WER gap is acoustic. Parakeet fixes 70% of the panel errors, and only ~22% of those look LM-fixable. Does adding Libri-Light audio with labels from R0 (the approved icefall streaming Zipformer teacher) lower B0-family WER at matched compute?

Ladder:
1. pilot: 960 h + `small`;
2. gate: 960 h + `medium`;
3. `large` (10-20k h) **only** if the gate passes.

## 2. Data provenance (owner approval 2026-11-23: Libri-Light with R0 labels)

| subset | URL | bytes | md5 (official = ours) | sha256 (ours, streamed) |
|---|---|---:|---|---|
| small | https://dl.fbaipublicfiles.com/librilight/data/small.tar | 35,135,006,720 | c49207eb86a8e8ac895561c37232041e | cbd68b0998c7742ca45302cd653c914f4b1eb1c12bddbfb7064c69c7f2fc91b2 |
| medium | https://dl.fbaipublicfiles.com/librilight/data/medium.tar | 320,792,217,600 | c75e7ac62471bfbf2db77528d62a9b74 | 3987255ff3b0fa278f36478037bbc8e8ed6ad60779a9b7749cfef64a2774295a |

- The official md5s come from facebookresearch/libri-light `data_preparation/README.md`; the S3 object metadata (`x-amz-meta-s3cmd-attrs`) carries the same values.
- Archives are untrusted until hashed, and nothing from them is ever executed.
- **Pass 1** (`ll_index.py`) streams each tar once and records:
  - whole-file md5 + sha256;
  - every member's offset, size and own sha256, using a strict tar parser that rejects unsafe paths and non-regular members;
  - the `.json` metadata.
- **Pass 2** (`ll_pass2.py`) uses a FLAC member only if its sha256 equals pass 1's index. So every byte used is tied to the official md5.
- small: 2,588 FLAC + 2,588 JSON members, 0 anomalies.
- medium: 18,739 FLAC + 18,739 JSON members, 0 anomalies.

## 3. Dev / test overlap check (logged in `docs/ultra/LOOKS.jsonl` before first use)

- **LibriSpeech side, id lists only.** `SPEAKERS.TXT`, `CHAPTERS.TXT` and `BOOKS.TXT` come from `dev-clean.tar.gz`, whose md5 `42e2234ba48799c1f50f24a7926300a1` equals OpenSLR's. Only those TXT files were extracted, and **no test audio or test transcripts were read.**
  - Eval speakers: 40 / 33 / 40 / 33. Eval chapters: 97 / 91 / 87 / 90 (dev-clean / dev-other / test-clean / test-other).
  - Our dev parquet's `speaker_id` / `chapter_id` columns (id columns only) equal these lists exactly.
- **Libri-Light's own statement** (Kahn et al. 2020, arXiv 1912.07875 §2): "We then removed corrupted files, files with unknown or multiple speakers, and speakers appearing in LibriSpeech dev and test sets." Books are not covered.
- **Rule:** a Libri-Light file is excluded if any of these match an eval set:
  - its speaker;
  - its LibriVox project id (`book_meta.id`), against `CHAPTERS.TXT` PROJ;
  - its Gutenberg id (from `url_text_source`), against `CHAPTERS.TXT` BOOK ID;
  - its normalised title, against eval project / book titles.
- **small results:**
  - speakers: **NONE**;
  - books: **12 eval books matched, so 178 / 2,588 files are excluded** (40.1 of 526.5 VAD h). Of those, 69 files are a different LibriVox reading of the same Gutenberg text as an eval chapter. Title matching added no hits beyond project / Gutenberg.
- **Information only:** 315 small files (80.7 VAD h) share reader and LibriVox project with a LibriSpeech train chapter, so they are probably the same recordings as part of the 960 h. They are kept.
- **medium results:**
  - speakers: **NONE**;
  - books: **FOUND, so 1,051 / 18,739 files are excluded** (256.6 of 4,747.5 VAD h).
- **Also excluded from the data, at trainer-lane's request (06:55Z):** files that share reader **and** LibriVox project with a LibriSpeech train chapter. These are probably the same recordings as part of the 960 h, so a "+Libri-Light" arm would only re-weight existing audio. That is 315 small files (80.7 VAD h) and 3,130 medium files (822.8 VAD h).
  - Pass 2 therefore uses 2,141 small files (≈ 416 VAD h) and 14,853 medium files (≈ 3,739 VAD h).
  - Lists: `excluded_*.tsv`, `train_dup_*.tsv`, `skip_*.tsv`.

## 4. Segmentation (`ll_segment.py`, fixed before labels)

- **Units.** Libri-Light VAD segments are sorted and merged when the gap between them is < 0.3 s. A cut may fall only in a pause ≥ 0.3 s.
- **Long units.** A unit longer than 17.55 s is split only at internal energy pauses ≥ 0.3 s, found with 20 ms log-energy blocks and a per-file threshold of floor + clip(0.3·(speech − floor), 3, 10) dB. Pieces that still do not fit are dropped and counted.
- **Utterances** are contiguous spans of the source; internal pauses are kept and nothing is concatenated.
  - A cut is forced at gaps ≥ 2.5 s.
  - Otherwise a DP chooses the cuts, minimising Σ((L − 14)/4)² − 0.5·min(gap, 1.5)/1.5.
  - **Length ≤ 18.0 s** including pads. A1 trains with speed 0.9 / 1.0 / 1.1 and `max_seconds` = 20, so no utterance is ever dropped by the loader.
  - Head pad = min(0.4 s, gap), and may shrink to 0.15 s to fit.
  - **Tail pad = min(0.6 s, gap) and never < 0.3 s** of real, VAD-silent audio. B0 truncates last words when trailing silence is < 0.2 s. Utterances that cannot get it (file ends) are dropped and counted.
- **small, all files:** 2,141 files → **112,595 utterances, 427.2 h** (456.8 file-hours), 0 errors, 144 shards.
- **Measured on 40 small files:** 97.8% of VAD speech kept; mean 12.2 s, median 13.0 s; tail pad 0.33-0.6 s (median 0.6). Segments are sample-identical to their source spans.
- **Shards** use LibriSpeech's HF parquet schema:
  - `file`, `audio{bytes = 16 kHz mono FLAC, path}`, `text`, `speaker_id`, `chapter_id` (= LibriVox project id), `id`;
  - plus `subset`, `src`, `t0`, `t1`, `v0`, `v1`, `head`, `tail`, `n_units`, `ecut`, `snr`, `speech_s`, which `train.data` ignores;
  - 100-row row groups, a SHA256SUMS file and a `.ids` file per shard.
  - Ids look like `ll{s|m}-<speaker>-<project>-<sha1(path)[:6]>-<k:04d>`, which cannot collide with LibriSpeech ids.

## 5. Labels and filter (fixed now, before any label is seen)

**Teacher.** kd-cache's `r0-fp32-gpu-v1` (H22 amendment A). Per utterance it gives: R0 greedy hyp and token ids; per-token log-prob at emission; per-frame blank-margin mean / min; beam-8 N-best with scores; n_frames and n_samples.

**Hard rules.** A row is dropped if any of these hold:
- H1: the hyp is empty;
- H2: `n_samples` ≠ the decoded audio length (integrity);
- H3: the word rate over `t1 − t0` is outside [0.6, 4.6] words/s;
- H4: the normalised hyp contains "librivox". These are LibriVox preambles and disclaimers, repeated in nearly every file; LibriSpeech excluded them too;
- H5: identical normalised hyps are capped at 3 copies per subset (first 3 by id hash kept).

**H4 amendment (2026-11-23T07:58Z).** R0 cannot spell LIBRIVOX; it writes LIBERVOX / LIBRAVOX / LIBREVOX / LIBRAVOC… So the literal test matched nothing on a 4,000-row check, while 2.5% of rows were LibriVox boilerplate. H4 is now the pattern `\bLIB[A-Z]*\s?VO[CX][A-Z]*\b|RECORDINGS? (IS|ARE) IN THE PUBLIC DOMAIN|TO VOLUNTEER PLEASE VISIT`.
- This implements the pre-registered intent, and was made before any filter result was used.
- It is checked not to match "LIBRARY", "LIBERAL VOICE" or "VOX POPULI".

**Soft rule.**
- S1: per subset, rank the remaining rows by R0's mean per-token log-prob ℓ, and drop the lowest 10%.
- S2: if a student hyp is available (A1-49000 EMA greedy, chunk 16), also drop rows whose word-level disagreement with R0 is > 0.5 (WER of A1 vs R0 hyp).

**Amendment 2 (lead decision 2026-11-23 ~09:2xZ, recorded before medium's finaliser runs): medium runs with `--allow-missing-student`.**
- It runs as soon as R0 labels cover every row, without waiting for A1 hyps (~255k of ~1.0M rows had one at 09:21Z).
- S2 still applies to every row that has an A1 hyp. Rows without one skip S2 only.
- Rationale:
  - S1 ranks on R0 log-prob alone;
  - S2 removed 37 / 112,595 rows (0.03%) on small;
  - trainer-lane confirmed `conf` is never used as a loss weight.
- FILTER_REPORT.json records how many rows lacked an A1 hyp.

**Amendment 3 (label integrity, same decision).** A medium R0 label unit is accepted only if:
- (a) it was produced by the v1.1 labeller code (`code_sha256` 7474f083…, full hash from h52-labeler), or
- (b) it is a v1 unit (frozen `r0_gpu.py` 41be5573…) that h52-labeler's v1.1 witness showed identical.

Details:
- Every v1 unit labelled on a shared card before 09:11Z must be re-labelled on v1.1 or witnessed. U00320 is re-labelled.
- Where an id is labelled twice, the v1.1 unit wins.
- Enforced by `ll_finalize.py --label-policy` (sha256 1e94ed9e…): units outside the policy are ignored, and their rows count as unlabelled.
- Audit at 09:25Z: of 510 units, 381 were v1 before 09:11Z (H0 259, xbabe2 78, xbabe1 44) and 129 were v1.1.

**Amendment 4 (2026-11-23T11:16Z, recorded before any medium result is used): an undefined (NaN) mean log-prob ranks lowest, so S1 drops it.**
- **Found:** at 11:13Z, medium's filter step logged a NaN S1 threshold and zero S1 drops. The run was stopped before any final shard was written.
- **Cause:** 8 of 1,003,623 medium labels have `tok_lp_mean` = NaN. All are 12-17 s utterances where R0 looped ("… AND AND AND …"), on 6 different cards / hosts. A NaN poisons the quantile, and `lp < NaN` is never true.
- **Rule:** NaN → −∞ before S1, in `ll_finalize.py` and `ll_finalize_par.py` alike. FILTER_REPORT counts the rows as `checks.lp_nan_as_lowest`.
  - The 468 rows with no mean log-prob are empty hyps, which H1 drops (unchanged).
  - Small is unaffected: its S1 threshold was finite.

**Operations note (lead's scoped exception, 10:15Z).** Medium's finaliser on xbabe5 runs at nice 19 + `ionice -c3`.
- It pauses only if a live trainer's rate (median of its last 3 step records) falls more than 5% below its 10-min pre-start baseline, or its data wait exceeds 8%. It resumes on recovery.
- This replaces the generic ≤ 5% data-wait gate, because S2's own loader sits at ~5.5% with no co-tenant.
- Guard events are logged and relayed to AGENT_CHAT. No data rule changes.

**Operations note 2 (11:07Z; no data-rule change).** Medium is finalised by `ll_finalize_par.py` (sha256 88fb42f19e0d7fc9…).
- It imports `ll_finalize.py` (sha256 fe65601234ed1212…) unchanged for the input checks, the label policy, the filter, `conf`, the bucket assignment and the per-bucket permutation.
- Only the verify, bucket and write passes are spread over 8 processes. Each worker takes a contiguous block of raw shards, so every bucket keeps the single-process row order.
- **Proof before use:** re-running small v1.1 with it reproduced the shipped manifest byte for byte (SHA256SUMS sha256 8f371a65…59f3). FILTER_REPORT is identical except `finished_utc`.
- It adds a disk floor (pause below 65 GB free on x5). It also re-opens each final shard and checks its row count before deleting that bucket's parts, so the disk peak is one copy of the kept rows.
- **Label collection:** 8 units (U00826, U00841, U00843, U00936, R00218, R00245, R00247, R00291) were labelled on H0 at 10:17-10:40Z but stuck in collection after the ssh-master fault. At 10:59Z they were collected through the driver's own path; every label file's sha256 equals the worker's receipt.
  - U00881 has no pass-2 segment rows: its output has 0 rows, and there is nothing to label.
- **Coverage at 11:01Z:** 1,003,623 / 1,003,623 rows carry an accepted v1.1 label (959 v1.1 units; the 381 v1 units are rejected and superseded by their R re-labels).

**How the student hyps are made.**
- Engine: A1-49000 EMA INT8 (bundle sha256 = B0 manifest), `ultra-engine-eval` e61121e (968d3ea7…), greedy, single thread per process.
- Measured speed: 6.9-9× realtime per core; peak RSS 866 MiB per process.
- On-prem (xbabe2) and on x5, engines run on the shards themselves.
- On x6 / x7 / x8 / x9 they run **from source**: CloudFront fetch → sha256 check → the identical deterministic re-cut (`ll_segment` + `ll_pass2` ids). Verified: 130 / 130 rows equal on id / t0 / t1 / src, and h52-labeler independently got 35 / 35 plus PCM sha256.
- All segmentation hosts use one pinned env: numpy 1.26.4, pyarrow 17.0.0, soundfile 0.14.0, libsndfile 1.2.2.
- Cloud hosts (Ubuntu 22.04) run the same binary through H0's glibc 2.39 loader.
- A from-source hyp is used only when its t0 / t1 equal the shard row's.

**Confidence for loss weighting.** `conf` = exp(ℓ) · (1 − min(1, agree_wer)), with agree_wer = 0 when there is no student hyp. It is kept per row; it is not used by default.

**Final shards** are written by `ll_finalize.py` on x5:
- schema: LibriSpeech's, plus `conf`, `r0_logp`, `blank_margin_mean`, `agree_wer`, `label_src`;
- `text` = R0 hyp (uppercase, LibriSpeech style);
- shuffled globally across files (bucketed, seeded), 100-row row groups, SHA256SUMS.
- Reported: kept fraction per rule, kept hours, and per-subset distributions of ℓ and of the word rate.

## 6. Data-scaling gate (pre-registered with trainer-lane; to be filled by trainer-lane where marked)

- **Protocol:** the H41 replay protocol, unchanged except for data.
  - Fork, recipe (adopted A1 recipe + dp 0.005), steps, batch, LR schedule, EMA, eval decode: **trainer-lane fills in the exact config**.
  - 1R = 8k steps at a 650 s batch on one A100 (≈ 33 min, ≈ 1,340 audio-h).
- **Arms (seed pair 1234 / 4321 each):**
  - **C:** LibriSpeech 960 h (`train-clean-100`, `train-clean-360`, `train-other-500`).
  - **P (pilot):** C + `librilight-small` final shards, at natural proportions by hours.
  - **G (gate):** C + small + medium final shards. LibriSpeech units are listed k times so that LibriSpeech is ~50% of exposure (k fixed from the kept hours before launch).
- **Metric:** full-dev pooled WER, canonical scorer (`ultra-score`), at chunks 4 / 8 / 16 (16 = B0's 160 ms is primary). Also dev-clean, dev-other, and the hardest-10%-speaker slice.
- **Decision:**
  - an arm passes if its seed-pair mean gain over C at chunk 16 is **> 0.17 pp** (2× the four-run spread);
  - both seeds must improve, and chunks 4 / 8 must not be worse than C by more than the spread (0.087 pp).
  - Otherwise FAIL, or INCONCLUSIVE when the seeds disagree.
  - **`large` (10-20k h, to A100-local NVMe) only if G passes.** P is a pilot: a P failure does not stop G, but its result is reported.
- **Kill rules:** H41's (divergence / dev-clean-400 thresholds), unchanged.
- **Compute:** P = 2 runs + C = 2 runs = 4R; G = 2 more runs = 2R. Total ≈ 6R ≈ 3.3 A100-h.

## 6b. `large` subset (insurance; lead 09:1xZ) - pre-registered before any large.tar metadata was seen

- **Use:** only if the S2-vs-S1 cooldown comparison at matched exposure calls for more data; nothing commits to training on it yet.
- **Source:** `https://dl.fbaipublicfiles.com/librilight/data/large.tar`, 3,055,377,274,880 B, official md5 `4dfbac018f50b99797ece101fc9f0c30`.
- **Pass 1 on xbabe6 (started 09:19Z):** `ll_index_par.py` (sha256 6caa5c76…).
  - It is the small / medium indexer plus 8 parallel range prefetchers feeding one ordered, sequentially hashed stream.
  - On small it reproduced md5 / sha256 / members.tsv exactly at 266 MB/s.
  - No tar is stored.
- **Same exclusions as small / medium:** dev/test speaker / book overlap (id lists only), plus probable 960 h duplicates (same reader + LibriVox project).
- **Selection rule v1** (`ll_select_large.py`, sha256 2f6afed1604e72b6827bce2f90f4273085c341cb1691256b55225e26c536653b):
  - eligible files ordered by sha256("h52-large-v1:" + path);
  - per-speaker cap of 15 VAD h;
  - tier A to 10,000 VAD h, tier B to 20,000 VAD h.
  - The list and its sha256 are posted before any segmentation.
- **Segmentation:** the same `ll_segment` / `ll_pass2` rules (≤ 18 s, tail ≥ 0.3 s) and the same pinned env.
  - Tier A shards go on xbabe6's disk; tier B on xbabe7's disk, under the trainer-discovering guard.
  - Labels and filter are the same as section 5, if and when needed.

## 7. Results so far (semi-supervised (R0 labels))

**small, final shards** (finalised 08:34Z on x5; 50 shards; SHA256SUMS verified):
- R0 labels: 100% (h52-labeler, r0-fp32-gpu-v1). A1 student hyps: 100%.
- **Kept 99,113 / 112,595 utterances (88.0%) = 378.7 / 427.2 h (88.6%).**
- Drops: S1 11,007 (threshold mean token log-prob −0.1791); H4 2,257; H1 72; H3 55; H5 46; S2 37; H2 8.
- Kept-set mean A1-vs-R0 word disagreement ≈ 4%.
- Paths:
  - `x5:~/jsr-data/librilight/shards/r0/librilight.small/` (`FILTER_REPORT.json`, SHA256SUMS);
  - copied to `x7:~/jsr-data/librilight/shards/r0/librilight.small/` (verified 08:59Z).
- **Manifest `SHA256SUMS` sha256 `25e283f43fe7e9e8dd272c493b9747567ede9f035700c90fe3e857fe63672c0a`.**
- The pilot runs (H52_P_dp0_s1234 / s4321, x7) read `x7:…/shards/r0_small_tl`, a byte-identical mirror (all 50 files pass `sha256sum -c` against this manifest; complete at 08:33Z, before launch).

**small, v1.1 re-finalisation (09:57Z, adopted for S2 and every later read).**
- h52-labeler's witness re-labelled every small row with r0_gpu v1.1: 99.86% identical. The exception is one corrupted batch (raw shard 0001, xbabe1, 161 wrong-but-finite labels, 132 of them kept in the pilot's data) plus 8 near-tie rows.
- Re-finalised from `r0src_v11/` only, with the same filter and seed: **kept 99,109 / 112,595 utterances (88.0%) = 378.65 h**; S1 11,011, H4 2,257, H1 72, H3 55, H5 46, S2 37, H2 8.
- Path `x5:~/jsr-data/librilight/shards/r0_v11/librilight.small/`, 50 shards, **manifest SHA256SUMS sha256 8f371a650591ff0663712f3b2603d963abeb69798636f008eb7fcadd5a5d59f3**.
- The v1 set (manifest 25e283f4…) stays as provenance for pilot P. Read P with the caveat that 132 of 99,113 rows (0.13%) carried slightly wrong text.

**medium, segmentation** (pass 2 done 08:21Z on x5): 14,853 files → **1,003,623 utterances, 3,809.4 h**, 0 errors, 1,256 raw shards at `x5:~/jsr-data/librilight/shards/raw/librilight.medium/`.
- R0 labels: from source (h52-labeler) into `r0src/`.
- Student hyps: from source on x7 / x6 / x8 / x5 / x9 into `student_src/`.
- Final shards follow with the same finaliser.

## 8. Changes / runs / results

- 2026-11-23 06:37Z: pass 1 small + medium started on xbabe5.
- 06:45Z: small verified.
- 06:47Z: small overlap check done (178 excluded).
- 06:55Z: pass 2 small started on H0 (hand-off via `/dev/shm/h52/`; uploads to xbabe5 `~/jsr-data/librilight/shards/`). Restarted 07:15Z with the 960 h duplicate exclusion.
- 07:08Z: h52-labeler starts R0 labels (r0-fp32-gpu-v1) on H0 / xbabe2 / xbabe1 GPUs under the mkdir-claim contract.
- 07:21Z: student hyps start.
- 07:30Z: uploads decoupled from the student hyps.
- 07:45Z: from-source student hyps on x6 / x7 / x8 / x9, under data-wait / trainer-rate guards (x9 ≤ 8 engines after 07:53Z).
- 07:52Z: small pass 2 done.
- ~07:55Z: small fully R0-labelled.
- 07:53Z: medium verified, overlap check run, pass 2 started on x5.
- 08:19Z: small student-hyp coverage 100%.
- 08:21Z: medium pass 2 done.
- 08:34Z: small finalised on x5 (pausing while a trainer's data wait was > 5%).
- 08:59Z: full verified set on x7.
- 08:58Z: host guards v2 (auto-discover live trainers; cap 0 until a trainer has step records; data wait > 5% halves at once).
- 09:57Z: small re-finalised from v1.1 labels (manifest 8f371a65…).
- 11:01Z: medium R0 coverage 100% under the v1.1 policy.
- 11:07Z: medium finaliser started on x5 (`ll_finalize_par.py`, 8 processes, guard v2 + disk floor).

## MANIFEST

`MANIFEST` lists the archive hashes, the index files (members.tsv / json_meta.jsonl sha256), the overlap verdict, and the shard SHA256SUMS. It holds no audio and no weights.
