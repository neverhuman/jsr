# bench-open: xbabe0 release holds (2026-11-23)

Quiet single-host measurements of the v3.0.0 release decoders on xbabe0 (AMD Threadripper PRO 3995WX, Zen 2), run by
bench-open's TIMING runner (`docs/paper/opponents/open_turn.py`, plan `open-plan-xbabe0.txt`, entries 14g-14t). Every hold
ran under TIMING.lock with the owner-approved timer pause (the owner's timers and CI runner units paused, the host reported
quiet), one measured process on one core (CPUs 4 / 24 / 28 for rounds 1 / 2 / 3, meter on 0 / 1 / 2), and bench-open's host
guard (alone on its CCX, SMT siblings idle, at most 2 other active cores). Watts are CPU-reported (RAPL core register of the
service core and the package) from energy's continuous 1 Hz log, never wall power. Background release-decoder evals ran only
between holds, stopped by process group at least 5 minutes before the next hold.

Receipts are compact copies of `xbabe0:~/jsr-data/opponents/bench-open/runs-v3/1c-assay/cold/round-N/<arm>/` (traces
gzipped); 14q's are `runs-v3/1c/cold/round-N/<arm>/` in `open_summary.py`'s compact form. `joins/` holds the H0-side
joins; `release-eval/` the full-dev release-decoder rows.

**Incident (disclosed):** at 12:30:44Z bench-open's runner exited (a window script renamed the runner's gap file), the lock
lapsed for 108 s and the timer-pause hook resumed the owner's timers and CI runner units; four CI job VMs ran 12:31:11-12:32:32Z,
when bench-open re-took the lock (the hook's pause ended them). 14m's round-1 beam-16 repetition, whose idle phase followed,
was set aside (`round-1/commit-lm16.void-ci-lapse-1231Z.*`, kept on xbabe0) and repeated.

## 14m: the release decoder (B0 + commit pass), quiet cost on one core

One artifact: release binary `ultra-engine-commit` 9a142e8e... (main bb1071a: #106 LM + LODR beam in the commit pass, #107
40 s cap), B0 INT8 (A1-49000 EMA, weights.int8 4701787c...), M5.ulm (72a621c2...). Its full-dev WER is runtime-lane's official
row on the same sha, 6.087% (6,455 edits). Each repetition decodes the 730-utterance panel (5,259.6 s) unpaced in one process
with one GEMM thread: the chunk-4 greedy stream (provisional text) and, at each endpoint, the chunk-16 final pass with a 320 ms
synthetic tail pad, then a reset of the stream (`reset_segment`). 60 s idle before and after each run for the 1 Hz power join.
Variants in Latin-square order over 3 rounds (CPUs 4 / 24 / 28); all repetitions clean. Medians of 3:

| final pass | CPU-s per audio-s (stream + final) | x real time | peak RSS | final-text delay p50 / p95 / max | core / package W per stream |
|---|---|---:|---:|---|---|
| greedy c16 | 0.0648 (0.0341 + 0.0306) | 15.5 | 251 MiB | 180 / 499 / 956 ms | 0.67 / 1.67 |
| beam 4 + M5 / LODR (0.65 / 0.45) | 0.0658 (0.0338 + 0.0317) | 15.3 | 686 MiB | 186 / 518 / 1,002 ms | 0.68 / 1.67 |
| **beam 16 + M5 / LODR (0.80 / 0.60), release** | **0.0702 (0.0339 + 0.0360)** | **14.3** | **686 MiB** | **210 / 589 / 1,127 ms** | **0.72 / 1.79** |

- CPU-s per audio-s from the binary's own per-utterance stream / final-pass wall (one thread on a quiet core, so wall ~ CPU)
  and `/usr/bin/time` (process, including model and LM load). Beam 16 per round: 0.0702 / 0.0701 / 0.0703.
- Final-text delay: the final pass's wall time after the endpoint is declared (the tail pad is synthetic zeros inside the pass;
  the endpointer's silence wait is not included). It scales with segment length.
- Watts per stream: net joules (run phase minus the floor interpolated between the idle phases) per audio second, i.e. the
  draw of one live stream at 1x.
- Session length: the cost per audio-s is flat over each 87.7-min session (per 10-min window 0.0696-0.0701 for beam 16;
  slope +0.1% per hour), because the stream resets at every endpoint.
- Join: `joins/14m.jsonl` (every repetition with its hashes).

## 14q: the v3.0.0 live point (RC1-EMA INT8 stream only, cold)

The release weights (RC1-EMA INT8, **weights.int8 edac19d8...**, the same staged copy as 14p) as the greedy chunk-4 live
stream only (160 ms chunks, 20 ms packets; no final pass, no LM), timed exactly like the paper's other cold rows (6b / 14b /
the anchors): `open-run.sh` -> `run-cold.sh`, process tree from launch to exit with the model load, one pinned core per round
(CPUs 4 / 24 / 28), the 730-utterance panel scored in the same run with the canonical `ultra-score` d784f8ca. The binary is
`ultra-cold-wav` built at the **v3.0.0 tag** (9c8c05e; `cargo build --locked --release -p ultra-bridge --bin ultra-cold-wav`,
rustc 1.95.0, clean tree; sha256 bc0729578eebce8b..., provenance on xbabe0 in `bench-open/anchor-v300/`), because the
e2ef608 cold binary of the earlier rows predates the release engine. Harness 2b74653 (arm `jsr-rc1-ema-int8`, added in
8366eaa; row `rc1-ema-live-zen2-t1`). The hold opened with one H22ctl anchor repetition on CPU 4, as every hold does.

| row (730 panel, one core, cold process tree with the model load) | CPU-s per audio-s (rounds 1 / 2 / 3) | x real time (1 / CPU; wall) | peak RSS | W per stream, core_msr_active (rounds) | WER pooled / clean / other |
|---|---|---|---|---|---|
| **`rc1-ema-live-zen2-t1`**, v3.0.0 live | **0.03434** (0.03420 / 0.03434 / 0.03453) | **29.1**; 29.1 | **74.2 MB** | **0.3445** (0.3328 / 0.3481 / 0.3445) | **6.845** / 4.116 / 10.749 |

All three rounds clean and alone on the host (no other busy core), at 4,149 / 4,153 / 4,168 MHz, 2026-11-23 20:16:59 -
20:26:09Z. The text is identical in the three rounds and identical (730 / 730) to 14p's provisional (live) stream from the
release binary 9a142e8e, so the live WER is the same number by either route. The evidence row is `14q/rows.jsonl`
(`open_summary.py rows`; receipt = round 2's measurement.json d2658af7..., the median round), the table `14q/table.md`.
Energy: `joins/14q-energy.jsonl`, the same join as every cold row (`energy_report.join_run` / `join_lines` over the timed
process window, started + 0.5 s .. + measurement wall_seconds, widened by 1.5 s): the core draws 9.96 W while decoding, the
package 95.1 W, idle 0.054 W; 3.02 kWh per stream-year. The join emits no `core_msr_dynamic` line, because round 1's idle
window before the run is the anchor running on the same core (the tool's rule).

The anchor (H22ctl, `jsr-h22ctl-79800-ema-int8/round-11` and `round-12`) read 0.0349 and 0.0345 CPU-s per audio-s, against
0.0354 in the earlier sessions: this session ran about 1.5-2.5% faster on our engine. Round 11 is the anchor of 14q's first
hold (20:03:30-20:07:00Z), whose three RC1 repetitions exited at once with "unknown cold comparison arm": `run-cold.sh`
routes only listed arms to `open-arms.sh`, and harness 8366eaa had not listed the new arm. Harness 2b74653 lists it, and 14q
ran again as 14qr; nothing of the failed hold is in the row.

## 14s: the v3.0.0 live stream's always-on capacity on one core

Energy's capacity test (TIMING 4b, run on H22ctl) repeated on the release: N = 1, 4, 8, 16 live streams of `ultra stream`
built at the v3.0.0 tag (9c8c05e; `cargo build --locked --release -p ultra-cli --bin ultra`, sha256 6d8ae087...) on the
release weights (RC1-EMA INT8, weights.int8 edac19d8...), one process per stream, all N pinned to one core (CPU 4, meter on
CPU 0), each stream paced at 1x (`--pace 1.0`, 20 ms packets, the release's default per-packet wakes) on energy's
`paced120.wav` (DEV panel u00000-u00021 joined, 127.98 s, sha256 37035776..., byte-identical to energy's file), with 60 s
idle phases before and after every N (assay `capacity-v300`, harness 2b74653; one hold, 20:29:12-20:43:37Z). Watts come from
energy's continuous 1 Hz log: the core register over the N-stream run (12 s trimmed at both ends, as 4b did) against the
idle floor interpolated between the idle phases around it; N streams at 1x decode N audio-s per second, so W per stream =
net W / N (gross = run W / N). The lag is the stable-word lag of every matched reference word (when a word last stops
changing, minus its forced-aligned end in the audio; `energy_report.stable_lags_from_events`, the paced-lag tool's
function), from each stream's own events against `timestamps_100.json` (first 22 utterances, sha256 e0856c92...).
Join: `joins/14s.jsonl` (`watts-join/join_14s.py` on H0).

| streams on one core | core busy | CPU-s per audio-s per stream | W per stream, net / gross (core) | stable-word lag p50 / p95 / max | words matched | last text after the audio ends |
|---:|---:|---:|---|---|---:|---:|
| 1 | 6.2% | 0.0619 | 0.487 / 0.579 | 190 / 420 / 545 ms | 328 / 342 | 0.008 s |
| 4 | 14.7% | 0.0375 | 0.387 / 0.426 | 200 / 430 / 559 ms | 1,312 / 1,368 | 0.020 s |
| 8 | 30.1% | 0.0388 | 0.366 / 0.382 | 215 / 438 / 579 ms | 2,624 / 2,736 | 0.040 s |
| **16** | **53.2%** | **0.0354** | **0.358 / 0.364** | **242 / 457 / 608 ms** | 5,248 / 5,472 | 0.087 s |

The round was clean and alone on the host (guard verdict clean, no other busy core), 20:29:48-20:43:35Z; every stream
finished, with the release's default per-packet wakes. Sixteen live streams fill 53% of one core with the p95 lag within 37 ms
of one stream's, so one core holds them with room to spare. One stream costs more per stream (0.0619 CPU-s per audio-s,
0.49 W net) than the cold unpaced row (14q: 0.0343, 0.34 W), because a lone paced stream wakes the otherwise idle core 50
times a second and its model load is spread over only 128 s of audio; from 4 streams on, the cost per stream approaches the
cold row's. Peak RSS per stream process: 78.5 MB. The package draws 72 W idle and 89 W at 16 streams (about 1.1 W per stream
above the floor). The old H22ctl capacity rows (energy, entry 4b, flagged for foreign load) read 61.8% busy at 16 streams,
0.368 / 0.375 W, lag p50 / p95 303 / 429 ms.

The one-stream line also gives the live row a paced lag: `joins/14s-lag.jsonl` (row `rc1-ema-live-zen2-t1`: p50 190 ms,
p95 420 ms). Its workload is one continuous 128 s stream of 22 clips, not the per-clip lag100 workload of the H0 lag rows, as
its `lag_method` says; the two are not the same measurement.

## 14t: Nemotron 560 ms on the quiet host (one panel73 repetition)

Nemotron Speech Streaming EN 0.6B at 560 ms (NeMo-Speech.cpp, Q8_0 GGUF), the H0 rotation's exact binary (nemo-speech
41b19d48...) and model (d9a01898...; energy's hash-checked copies on xbabe0, linked where `run-cold.sh` looks) with
`run-cold.sh`'s own command (one thread, right context 6 = 480 ms), one repetition on panel73 (73 utterances, 597.1 s) on CPU 4,
as the lead asked: a full-panel repetition (~40 min) would outrun the 30-minute hold. Clean, 2026-11-23 20:53:11-20:57:19Z
(TIMING 14tr, harness 35eb5ce; receipt `14t/round-1/nemotron-560/`, copied whole with its 73 transcripts).

| | CPU-s per audio-s | x real time | peak RSS | W per stream, core_msr_active |
|---|---:|---:|---:|---:|
| **quiet xbabe0, panel73, 1 rep (14tr)** | **0.4156** | **2.41** | 987 MB | **4.33** |
| H0 rotation row `nemotron-en-560-zen2-t1-xbabe3` (full panel, 3 reps, shared host) | 0.4510 | 2.22 | 1,290 MB | 4.09 (core_msr_contended, separate run) |

The quiet host runs it 7.9% faster than the shared-host row. Energy: `joins/14t-energy.json` (`energy_report.join_run` over the
timed process window; the core draws 10.4 W while decoding; one other core crossed 2 W during the run). A single subset
repetition is a cross-check of the shared-host row, not a row of its own. Its first hold (14t, 20:46:37-20:47:15Z) ran only
the anchor: one-core panel73 arms go through `open-arms.sh`, which had no Nemotron arm until harness 35eb5ce.

The three panel73 anchor repetitions of 14s / 14t / 14tr (H22ctl, `anchors-p73/`) read 0.0355 / 0.0354 / 0.0352 CPU-s per
audio-s.

## 14p: the v3.0.0 candidate on the release decoder (one artifact)

RC1's EMA (ckpt-00005095 ee40e833... -> EMA as model 5f528528...), eval-lane's pinned INT8 export **weights.int8
edac19d8...** (weights.safetensors dd4121aa..., arch.json 880a38c0...), on the release binary 9a142e8e with 14m's exact command,
730 panel, one core, 3 repetitions in one hold (CPUs 4 / 24 / 28), all clean. Its full-dev WER on the same artifact is
eval-lane's release-decoder row, 5.476% (5,808 edits; -0.610 pp [-0.737, -0.489] vs the B0 release row).

| artifact (release decoder) | CPU-s per audio-s (stream + final) | x real time | peak RSS | final-text delay p50 / p95 / max | core / package W per stream | revised |
|---|---|---:|---:|---|---|---:|
| **candidate RC1 EMA (edac19d8...)** | **0.0705 (0.0342 + 0.0360)** | **14.3** | **686 MiB** | **211 / 586 / 1,126 ms** | **0.72 / 1.64** | 42.5% |
| B0 (4701787c...), 14m | 0.0702 (0.0339 + 0.0360) | 14.3 | 686 MiB | 210 / 589 / 1,127 ms | 0.72 / 1.79 | 41.4% |

CPU per round 0.0706 / 0.0704 / 0.0705; the package figure carries about +-0.1 W of floor noise per round. Join:
`joins/14p.jsonl`.

**The paper's energy method for this row (`rc1-ema-final-zen2-t1`, the v3.0.0 final point):** `joins/14p-energy.jsonl`,
computed from these receipts with the same tool as every cold row's energy join (`energy_report.join_run` / `join_lines`):
the service core's register over the one `ultra-engine-commit` process (`out/phases.tsv` run_start..run_end, widened by
1.5 s), divided by the panel's 5,259.56 audio-s. **core_msr_active 0.720 W per stream** (rounds 0.705 / 0.724 / 0.720),
core_msr_dynamic 0.719 (the core's idle level in the assay's 60 s idle phases: 0.018 W), package 92.8 W, no other busy core in any round;
receipt = round 3's measurement.json (3e812eda..., the median-CPU round).

## 14k: streaming-L4 fallback vs B0, paced (latency and memory)

panel73 (73 DEV utterances, 597.1 s, staged under full-dev ids), one core per run, physically paced at 1x in 20 ms packets,
L4 / B0 order alternated by round. L4 = B0 INT8 (4701787c...) + M5.ulm (72a621c2...) / LODR in codex-full-fleet's
`h46-native-l4-paced` (d561d65d...; chunk 4, beam 4, 0.8 / 0.6 / 0); B0 = greedy chunk 4 in eval-lane's paced runner
(beb61399...). 6 / 6 repetitions clean.

| arm | CPU-s per audio-s (r1 / r2 / r3) | median | peak RSS | final text after the last packet (mean / p95) |
|---|---|---:|---:|---|
| B0 | 0.0629 / 0.0627 / 0.0630 | 0.0629 | 67-68 MiB | 5.9 / 10.0 ms |
| L4 | 0.0646 / 0.0642 / 0.0643 | 0.0643 | 685 MiB | 5.9 / 10.0 ms |

eval-lane's protected E2 reading of these traces (first correct word, paced capture clock, pooled 3 x 73): onset p50 equal,
onset p95 +40.1 ms [-160, +320], so the streaming-L4 row failed E2 as registered and is a documented row only.

## 14g: B0 power, pair 2 (capture-only proxy vs B0 listening)

One hold = idle 3 min, X 10 min, Y 10 min, idle 3 min (ABBA over holds), 600 s of pink noise at -30 dBFS. 3 / 3 holds clean.
The capture-only proxy (20 ms packets read at 1x, no model) costs nothing measurable (net core +0.03 W over the idle floor).
**B0 listening minus the capture proxy in the same hold: +0.47 W core / +1.7 W package** (median of 3; per hold core
+0.45 / +0.47 / +0.54 W, package +1.23 / +1.72 / +2.52 W). With 14f (one live B0 stream ~0.5 W core / ~2 W package, the same
on speech and on noise). Join: `joins/14fg_power_holds.jsonl`.

## 14i: H47-Q, #77's Sentinel-0 session (CPU and energy)

h47-sleep's `h47_q_cpu.sh` with `ultra-alwayson-check` b6bc7fbd... (`--replay-ms 500 --idle-ms 1500`) on B0 INT8, unpaced, one
core per repetition, gated (G) vs its ungated twin (U) vs today's C1 (T=1500 / R=1000) on the 5%-duty hour. 3 / 3 clean.
CPU-s per audio-s (3-repetition means, load baseline subtracted; spread <= +-0.3%):

| input | G | U | C1 | G / U |
|---|---:|---:|---:|---:|
| silence (900 s) | 0.00010 | 0.0407 | | 0.25% |
| pink -30 dBFS (900 s) | 0.00010 | 0.0408 | | 0.25% |
| white -20 dBFS (900 s) | 0.00024 | 0.0407 | | 0.60% |
| duty05 (1 h, 5% speech) | 0.00292 | 0.0411 | 0.0491 | 7.1% (G / C1 5.9%) |
| panel (900 s, 100% speech) | 0.03533 | 0.03522 | | 1.003 |

Q2 (mean G / U on non-speech <= 0.02): 0.0036, PASS. Q6 (G / U on speech <= 1.05): 1.003, PASS. Energy per stream at 1x on
the 5%-duty hour (net J per audio-s from the 1 Hz log; core register exact, package approximate): G 0.029 W core / 0.09 W
package, U 0.41 / 1.28, C1 0.50 / 1.59. Summary: `joins/14i_q_cpu_summary.json`.

## Release-decoder full-dev evals (between holds)

`release_eval.sh` (v3) on xbabe0's idle cores between holds: 80 single-thread `ultra-engine-commit` processes at nice 19 on
CPUs 24-63 / 88-127, ~20-utterance shards of the 5,567 full-dev utterances, canonical `ultra-score` (d784f8ca...). The binary
is pinned per run. Settings: `--commit-chunk 16 --tail-pad-ms 320 --threads 1 --commit-lm M5.ulm --commit-beam 16 --lm-scale
0.80 --lodr-scale 0.60 --token-bonus 0.0`. These runs are memory-bandwidth bound (80 processes), so their CPU totals are not
throughput numbers; 14m is.

| model | binary | dev-clean | dev-other | pooled | edits | provisional (c4 greedy) |
|---|---|---:|---:|---:|---:|---:|
| B0 (A1-49000 EMA, 4701787c...) | fb8a95e5 (#106, 30 s cap) | 3.256 | 9.131 | 6.102 | 6,471 | 8.272 |
| S2-BP1 EMA (S2a step 5,500; weights.int8 0f35bfb1...) | 9a142e8e (bb1071a, 40 s cap) | 3.158 | 8.816 | 5.898 | 6,255 | 7.987 |

The B0 row reproduces runtime-lane's official #106 pad-320 row edit for edit. S2-BP1 vs B0 on the same binary (runtime-lane's
official 9a142e8e row, 6,455 edits): -0.189 pp [-0.282, -0.096] (10,000 paired speaker-cluster bootstrap replicates).

## 14n: warm-AGC v2 added engine CPU (criterion 4; for the record, warm AGC ships opt-in)

#108's `ultra-engine-agc` (02e1fa4c..., the binary eval-lane ran C1-C3 on) on B0 INT8 decodes the 730-utterance panel in the
shipping mode, one process per arm with `--manifest` (AGC level carried across utterances, decoder reset per clip):
`--warm-agc --tau-s 5.0` vs `--no-warm-agc`, first arm alternating by round, paired per utterance on the binary's decode-loop
thread CPU. 3 / 3 rounds clean. Added engine CPU: +0.34% [+0.26, +0.41], +0.52% [+0.44, +0.60], +0.55% [+0.47, +0.62]; pooled
**+0.47%** (on 0.0342 vs off 0.0341 CPU-s per audio-s), under the criterion's 1%. The AGC changes the text of 69 / 730 utterances.
Warm AGC ships opt-in because criterion 3 (the 27.56 h stream) failed (lead, 14:05Z). Join: `joins/14n.jsonl`.

## 14o: #105 per-packet vs per-chunk wakes (one paced B0 stream)

`ultra stream --pace 1.0` (296f469c, #105's merged head ab0a867), 20 ms packets, `--wake-per-packet` (default) vs
`--wake-per-chunk` on 300 s of continuous speech-only DEV audio and 300 s of pink noise at -30 dBFS, ABBA inside each hold.

- CPU-s per audio-s (median of 3 rounds): speech 0.0619 vs 0.0618 (-0.6%), pink 0.0618 vs 0.0613 (-0.8%); core watts per
  stream unchanged within noise (~0.45-0.5 W). The loaded-host smoke's -24% / -33% does not hold on a quiet core.
- Not design-clock identical: on continuous speech the 30 s hard-cap closure lands one revision later in chunk mode (segment
  0: revision 102 vs 103), the previews diverge from end sample 481,280 on, and on this input (an exact multiple of the cap)
  packet mode ends with 10 cap closures and an empty Final while chunk mode ends with 9 and a Final carrying the last segment's
  text. Pink: previews and texts identical; one Status event fewer in chunk mode, and the cap / status events land ~0.3 s later
  in wall time. Deterministic: identical in all 3 rounds. Event files are gzipped.
- Ruling (lead, 14:47Z): `--wake-per-chunk` stays opt-in and experimental; per-packet is the default. Join: `joins/14o.jsonl`.
