# H39: energy per live stream (watts), and an independent check of the per-core method

Status: **EXPLORATORY** (shared hosts, CPU-internal energy model, no wall meter).
Owner: `energy` agent, lane `bench/energy`. Tool: `python/ultra/tools/energy_bench.py`.

## 1. Question

1. How many watts does one live stream cost on a Zen 2 core, for our engine and for the other recognisers?
2. runtime-lane already reports "CorWatt" rows (`power_bench.py`: turbostat per-core watts on one pinned core of xbabe3, 3.5 s idle before and after, one run per arm). Is that method sound, and how much does a per-core figure understate what the whole CPU package draws?
3. Does the figure measured on unpaced audio (as fast as possible) equal the figure for streams paced at 1×?

## 2. Protocol (written 2026-11-22 before any of my own measurements of a recogniser)

What I had seen when this section was written: runtime-lane's posted CorWatt rows (4.8–5.8 W per busy core; 312–1,122 mW per stream), and one 10 s idle reading of xbabe0 (package 84.6 W; the 64 per-core counters summed to 20.9 W). Nothing else.

### 2.1 Counters

| Counter | Source | Covers | Leaves out |
|---|---|---|---|
| Per-core energy | MSR `0xC001029A`, one per physical core (both SMT threads read the same value), 32 bits, unit from MSR `0xC0010299` (2^-16 J here) | that core's modelled energy | L3, memory controller, I/O die, every other core |
| Package energy | MSR `0xC001029B` and `/sys/class/powercap/intel-rapl:0/energy_uj` (same register; the file wraps at `max_energy_range_uj`) | all cores plus the shared (uncore) part of the CPU package | DRAM, voltage-regulator and power-supply loss, fans, disks, board |

- Both are **models inside the CPU, not measurements** (Schöne et al., "Energy Efficiency Aspects of the AMD Zen 2 Architecture", IEEE CLUSTER 2021, arXiv:2108.00808: modelled, no DRAM domain, package below an external meter, data-dependent power not represented). No wall meter or BMC power sensor is used here, so every number in this wave is "CPU-reported", never "at the wall".
- The powercap file `intel-rapl:0:0` ("core") is **core 0 only** on AMD, not the sum of cores (measured: 0.526 W next to cpu0's MSR at 0.527 W while 64 cores summed to 20.9 W). It is not used.
- Counters are root-only. A helper started with `sudo -n` only opens `/dev/cpu/N/msr` and the powercap file read-only and prints samples; the workload never runs as root.

### 2.2 One measured run

`idle window (>= 60 s) -> command on pinned cores -> idle window (>= 60 s)`; consecutive repetitions share the idle window between them. Five seconds after each command are discarded before an idle window starts.

- The helper samples once a second on a core far from the workload (CPU 31), and takes an extra sample at every phase boundary, so a phase's energy is an exact counter difference, not a sum of per-second estimates. Wrap-around is handled sample to sample (the package counter wraps every few minutes).
- **Core view:** energy of the pinned physical cores. Their SMT siblings must stay idle (the register covers both threads). **Package view:** the package counter. Both are reported side by side.
- `idle_watts` = energy over both idle windows / their duration. `net_watts` = run watts − idle watts. `net_joules` = net watts × run seconds.
- **`joules_per_audio_s`** = net joules / audio seconds processed. One stream at 1× consumes one audio-second per second, so this number **is** watts per live stream (`watts_per_stream`).
- **`joules_per_cpu_s`** = net joules / CPU seconds of the process tree (user + system) = watts per busy core.
- Three repetitions; the median is reported with min and max.
- The receipt carries command, cores, kernel, governor, boost setting, load average before and after, per-phase busy time of the pinned CPUs' SMT siblings and of the rest of the machine (`/proc/stat`), mean frequency of the pinned CPUs, and SHA-256 of binary, model and audio manifest.
- No machine-wide setting is changed. Governor and boost are recorded as found (xbabe0: `schedutil`, boost on, driver `acpi-cpufreq`).

### 2.3 Validation, in this order; a failed check stops the measurements until the method is fixed

| # | Check | Pass rule (fixed before running) |
|---|---|---|
| V1 | Idle against idle: the "command" is `sleep 60` | median net watts within the spread of the repetitions; the spread is reported as the noise floor of each view |
| V2 | Busy loop on 1, 2, 4, 8 pinned cores | least-squares line, R² >= 0.99 and every point within 10% of the line, in both views; report the slope (watts per busy core) and the intercept (fixed cost) per view |
| V3 | Our engine, 1, 2, 4, 8 single-thread workers on 8 pinned cores, each worker decoding the same audio | core-view joules per audio-second within 10% across the four worker counts; package-view net watts linear in the worker count (R² >= 0.99) |
| V4 | Core register against the package counter (from V2 and V3) | no pass rule: report package watts per busy core / core-register watts per busy core. This is the factor by which a per-core figure understates the package figure |
| X | Cross-check of runtime-lane's rows: our engine (D2e-16457 INT8) and the Zipformer, same binary, model and 100-utterance audio as its rows, on xbabe0 | report agreement in percent for watts per busy core, seconds per audio-second and mW per stream. Agreement within 10% counts as "the method reproduces" |

### 2.4 What makes me distrust a number (each is a flag in the receipt)

- **Sibling busy:** an SMT sibling of a pinned core was busy more than 1% of the run (its energy lands in my core's register).
- **Foreign load changed:** the other cores' summed watts differ between the run and its idle windows by more than 3 W. The core view survives this; the package view of that repetition does not.
- **Idle drift:** idle before and idle after differ by more than 0.2 W per pinned core (core view) or 3 W (package view).
- **Worker not on-CPU:** CPU seconds below 95% of wall seconds for a single-thread unpaced arm.
- **Repetitions disagree:** joules per audio-second min to max wider than 5% of the median. The arm is re-run.
- **Package noise against signal:** if V1's package noise is more than 10% of one busy core's package rise (V2 slope), single-core package figures are reported as indicative only and the 8-core package figure is used.
- Frequency is recorded because watts per busy core depend on it: one busy core on an idle 64-core part boosts above 4 GHz, which is the most expensive joule per operation this CPU has. A fully loaded server runs each core slower and cheaper, and shares the idle floor between more streams. These numbers are for the measured condition and say so.

### 2.5 Audio

- `wav100`: runtime-lane's power workload (first 100 dev-clean utterances, 739.6 s) for the like-for-like cross-check X.
- `panel730`: the frozen DEV panel (staged-WAV digest `010aa985…`, 5,259.56 s), used where a run fits a 15-minute quiet window.
- `panel73`: one utterance per speaker (every tenth staged file, `u00000, u00010, …`), for arms that need more than 15 minutes on the full panel. Our engine and the Zipformer run both `panel730` and `panel73` so the subset is checked rather than assumed.
- Never LibriSpeech test audio.

### 2.6 Live-paced check

Several streams paced at 1× on one core, for our engine and the Zipformer. Watts per stream = net watts of the steady part / number of streams, compared with the unpaced `joules_per_audio_s`. Model load is outside the steady part. Agreement within 15% means the unpaced figure can be quoted as watts per live stream; otherwise the paced figure is the one to quote and the gap is reported.

### 2.7 Shared host

xbabe0 is shared with other benchmark agents and background services. Before each window I create `xbabe0:~/jsr-data/opponents/QUIET.lock` (`energy until=<UTC>`, at most 15 minutes ahead), wait for runs in flight, measure, remove it. Cores: 16–23 for workloads (SMT siblings 80–87 idle), CPU 31 for the sampler.

### 2.8 Amendments after V1 (2026-11-22T20:32Z, before any recogniser result was read)

V1 ran 20:14–20:26Z. Four of its five repetitions were disturbed: one by my own smoke test on CPU 17, three by unpinned bursts from other work (the package jumped by 19–43 W for a minute and every one of my eight idle cores gained 0.4–0.7 W). The flags caught all four, but the rules as first written would have let ±1.5 W of package noise through. Changes, all made with only V1 in hand:

1. **Tighter rules.** Other cores' power may move at most **1 W** across idle–run–idle (was 3 W), also between the two idle windows; package idle drift at most **2 W** (was 3 W); CPU seconds at least **98%** of wall for an unpaced worker (was 95%).
2. **The headline uses only clean repetitions**, per view: a repetition with `sibling_busy`, `idle_drift_core`, `worker_not_on_cpu` or a failed exit is dropped from both views; `foreign_load_changed` and `idle_drift_package` drop it from the package view only. The receipt says how many were clean; if none was, the number is still printed and flagged `no_clean_…_rep`.
3. **The core view measures only the cores that carry a worker.** V1 and the first ladder measured all eight claimed cores while loading fewer; foreign bursts landed on the idle ones. Ladder fits use the per-core breakdown of the loaded cores.
4. **Package against core registers by regression over every phase** (`energy_bench.py uplift`). V1's ten phases, with foreign load between 4 W and 36 W of summed core-register power, already lie on one line: package = 54.6 W + 1.40 × (sum of the 64 core registers), R² 0.998, residual 0.65 W. So foreign load is not only noise: it is extra points on the line the lead asked for. Where the foreign load moved during a repetition, the package figure can be corrected with that line; a corrected figure is always labelled as such.
5. **Idle against idle is also read off every later run**: each repetition has an idle window before and after, so their difference is one more idle-against-idle sample. The noise floor reported in §3 uses all of them, not just V1's.
6. **Nothing of mine runs on xbabe0 during a measurement** except the sampler and the workload.

## 3. Results (2026-11-22, xbabe0 unless stated; every number is CPU-reported, not wall power)

### 3.1 The answer in one table

Core registers, full 730-utterance panel, one pinned core alone on its 4-core L3 group, three quiet repetitions each (the same runs that produced the timing rows):

| System | CPU-s per audio-s | W per busy core | **J per audio-s = W per live stream** (core registers) | Ratio to ours | Package, densely loaded host (W per stream above the busy floor) |
|---|---:|---:|---:|---:|---:|
| ULTRA dense S INT8, H22ctl step 79,800 EMA (ours, the plotted model; runtime-lane's timing entry 2, joined with the power log) | 0.0360 | 9.3 | **0.334** (0.345 / 0.331 / 0.334) | 1.00 | 0.42 (factor 1.245) |
| ULTRA dense S INT8, D2e step 16,457 (ours, previous point) | 0.0352 | 9.7 | **0.343** (0.344 / 0.341 / 0.343) | 1.03 | 0.43 (factor 1.245, measured) |
| Zipformer INT8 (sherpa-onnx) | 0.0692 | 8.5 | **0.591** (0.597 / 0.586 / 0.591) | 1.77 | 0.65–0.74 (factor 1.10–1.26, one plateau) |

- "W per live stream" is energy per audio-second: a stream at 1× consumes one audio-second per second.
- **Always-on (owner's direction, 22:32Z):** a stream running all year at those watts costs **ours 2.9 kWh (core registers) / 3.6 kWh (package marginal), the Zipformer 5.2 / 5.7–6.5 kWh per stream-year** (W × 8,766 h). Streams one core can carry unpaced (1 / CPU-s per audio-s, no latency bound): **ours 27.8, the Zipformer 14.5**; the measured number at a latency bound comes from the paced check (§3.8).
- One stream alone on an otherwise idle machine raises the package by about 34 W (H22ctl: 97.7 W median while running against 63.3 W in the 13 idle minutes after entry 2), i.e. 1.2 J per audio-second; almost all of that is the step every busy core causes (§3.5), paid once per machine.
- The package column multiplies the core figure by the measured package watts per core-register watt (§3.5). On top of it the package draws about 16 W more as soon as any core is busy, once per machine.
- Rows for the other recognisers come from the continuous power log joined with each agent's timed runs (§3.9), in `evidence/xbabe0-energy-joins.jsonl` as the serial timing queue produces them. So far: **ours, A1 step 30,000 EMA INT8: 0.350 W per live stream** (0.335 / 0.350 / 0.351; bench-open's entry 6b; 0.0355 CPU-s per audio-s; 3.1 kWh per stream-year, 3.8 package marginal), **ours, A1 step 20,000 EMA INT8: 0.337 W per live stream** (0.347 / 0.329 / 0.337; 0.0365 CPU-s per audio-s; 2.9 kWh per stream-year in the core registers, 3.7 package marginal), **Moonshine Tiny 1.077 W per live stream** (0.122 CPU-s per audio-s, 3.2× ours, 9.4 kWh per stream-year) and **Parakeet TDT v3 INT8 (sherpa-onnx, whole utterance) 1.711 W** (0.169 CPU-s per audio-s, 5.1× ours, 15.0 kWh per stream-year).

### 3.2 Instruments agree

- turbostat's `CorWatt` (runtime-lane's source) and this tool read the same register: on one core at the same time (H0 CPU 20, busy loop, 14 s) turbostat's mean over busy samples was 9.821 W and this tool's exact energy over the run 9.792 W (ratio 1.003).
- The powercap package file and the package MSR agree to 0.005% (ratio 0.999948: the kernel rounds the energy unit to 15,258 pJ).
- The powercap "core" file is core 0 only (§2.1).

### 3.3 V1, idle against idle: the core registers pass, the package counter does not

Fifteen pairs of consecutive 60 s idle windows (V1 and the V2 ladder, 20:14–20:46Z, benchmark agents locked out; `derived/idle-against-idle.json`):

| View | median difference | 90th percentile | largest |
|---|---:|---:|---:|
| Pinned cores, per core | 0.20 W | 0.40 W | 0.56 W |
| Package | **15.5 W** | 37.4 W | 39.3 W |

- A busy core reads 8–11 W, so the core view's noise is 2–5% for one repetition and under 1% for a three-repetition median (the cold rounds below repeat within 0.9%).
- The package moves by tens of watts from minute to minute: other tenants' services, file transfers and builds run in bursts of 20–40 W that the lock cannot stop. Per second the idle package ranges from 43 W to over 110 W. **Package-above-idle for one busy core (about 25 W) is not repeatable within 10% on this host**, which is the condition the lead set for making it the headline.
- V1's own five repetitions: one clean (core +0.05 W per core, package +1.4 W), one spoiled by my own smoke test, three by foreign bursts. The flags caught all four.

### 3.4 V2, busy loop on 1, 2, 4, 8 cores

| Busy cores | core registers, loaded cores (W) | per core | clock | package, median second while running (W) |
|---:|---:|---:|---:|---:|
| 1 | 11.41 | 11.4 | 4.29 GHz | 86.3 |
| 2 | 22.25 | 11.1 | | 102.6 |
| 4 | 44.68 | 11.2 | | 125.9 |
| 8 | 84.96 | 10.6 | 4.08 GHz | 168.1 |

- **Core view: PASS.** Line 1.44 + 10.50 W per busy core, R² 0.9993, worst point 4.5% off (`derived/v2-spin-fit.json`). Watts per core fall 7% from one to eight busy cores because the clock falls.
- **Package view by the rule as written (run minus idle, clean repetitions only): cannot pass.** Not one of the ten repetitions was clean for the package; the flagged numbers give R² 0.96 and a worst point 26% off.
- **Package view inside the runs** (median second, which bursts cannot raise): 77.8 + 11.45 W per busy core, R² 0.995, so **1.09 package watts per core-register watt** for a workload that touches no memory.

### 3.5 V4, core registers against the package (what a per-core figure leaves out)

Our engine, 1 to 8 single-thread processes started in steps inside one run, each decoding the whole panel (`derived/engine-stair-a.levels.json`, median second of each plateau):

| Processes (per L3 group) | core registers of the 8 cores (W) | per busy core | package (W) |
|---|---:|---:|---:|
| idle | 0.3 | – | 66.1 |
| 1 | 11.4 | 9.9 | 95.2 |
| 2 (same group) | 20.7 | 9.7 | 109.4 |
| 4 (one full group) | 28.4 | 6.9 | 118.3 |
| 4 (the other group) | 28.3 | 6.9 | 117.8 |
| 6 (2 + 4) | 33.7 | 5.5 | 121.0 |
| 8 (4 + 4) | 38.1 | 4.8 | 130.7 |

- **package = 82.1 W + 1.245 × core-register watts**, R² 0.979, residual 1.9 W. So above the busy floor the package draws **1.245 W per watt the core registers report** for our engine: a per-core figure leaves out about 20% of the marginal package watts.
- **The busy floor is 16 W above idle** (82.1 against 66.1). It appears when the first core becomes busy and is paid once per machine. One stream alone on an idle 64-core part therefore "costs" 25–30 W at the package, almost all of it this step.
- In the quiet cold rounds a second arm on another L3 group added 15–18 W of package for 8.3–9.6 W of core register (factor 1.8–1.9), so on a sparsely loaded host the share a per-core figure leaves out is nearer 45%.
- **Zipformer:** its staircase ran while the other agents were stopping their arms, the package fell 110 W under it and no line can be drawn (`derived/zipformer-stair-a.levels.json`, R² 0.14). One quiet plateau survives: eight workers, 50.2 W in the registers, 138.3 W package. With the busy floor of our engine (82 W) or of the busy loop (74–78 W) that is a factor of 1.10–1.26.
- DRAM, voltage regulators, power supply and fans are in neither counter.

### 3.6 V3, our engine on 1, 2, 4, 8 workers: FAILS as written, and the reason is the workload, not the meter

- The rule was "core-view joules per audio-second within 10% across worker counts". One process alone: 0.343 J per audio-second. The staircase run, dominated by eight processes: **0.488** (+42%), at 0.0958 CPU-s per audio-second instead of 0.0352 (+172%).
- Eight separate `ultra-cold-wav` processes each carry their own copy of the weights (58 MB INT8) and evict each other from the two 16 MB L3 caches: each took 440–530 s for the panel against 185 s alone. A core that waits on memory reads less in its register (4.8 W against 9.9 W), but not proportionally less.
- The meter is linear on a workload that scales (V2). So the check as I wrote it tested the engine's multi-process scaling, which is poor, and not the meter. The plain ladder (2, 4, 8 processes with idle windows) was cancelled when xbabe0 became serial; the staircase stands in for it.
- The serving path is one process with shared weights (24 streams per core in the ladder of H19), not N processes; "8 processes side by side" rows of our engine must be labelled as such.

### 3.7 Cross-check of runtime-lane's rows (`power_bench.py`, xbabe3 core 44, one run, 100 dev-clean utterances)

| | runtime-lane, xbabe3 | this wave, xbabe0, full panel, 3 quiet rounds | difference |
|---|---:|---:|---:|
| ours D2e: W per busy core | 5.84 | 9.69 | −40% |
| ours D2e: seconds per audio-second | 0.0607 | 0.0352 | +72% |
| **ours D2e: J per audio-second (active)** | **0.3546** | **0.3426** | **+3.5%** |
| Zipformer: W per busy core | 5.15 | 8.48 | −39% |
| Zipformer: seconds per audio-second | 0.1245 | 0.0692 | +80% |
| **Zipformer: J per audio-second (active)** | **0.6416** | **0.591** | **+8.6%** |

- **The method reproduces for joules per audio-second (within 10%), and only for that.** On xbabe3 the measured core shared its L3 with a busy neighbour and the host was loaded; the core drew less and took longer, and the two nearly cancel.
- **Watts and seconds must come from the same run.** Carrying 5.84 W over to the quiet-host speed gives 0.205 J per audio-second, 40% too low.
- The same holds under bus contention on xbabe0: the Zipformer alone on its L3 group read 8.26 W at 0.0722 s per audio-second on a quiet host and 6.37 W at 0.0942 with 16 other busy cores: **0.597 / 0.600 / 0.597 J per audio-second**.
- What `power_bench.py` does not give: repetitions and a spread, a check that the SMT sibling and the L3 group were idle, a package figure that survives a shared host (its 8-core protocol subtracts 4 s idle windows), and a correct name for its time base (`cpu_s_per_audio_s` is wall seconds). Its docstring says the package counter includes DRAM; on AMD it does not.

### 3.8 Live-paced check

- **Word lag** is the stable-word lag the lead ruled for the paper (end to end, input paced at real time): a reference word counts as emitted at the first event after which it is never withdrawn from the hypothesis; lag = that time minus the word's end in the audio (MFA word ends, `word-lag/timestamps_100_mfa.json`, the first 22 utterances of the panel concatenated). Event payloads are bounded windows (the tail of the hypothesis, about 60 bytes), so `energy_report.py paced-lag` places each window in the final hypothesis. Check against runtime-lane's harness on the same model: one paced stream of ours on an idle H0 core gave 346 / 442 ms (p50 / p95), its trace gave 350 / 494 ms; the Zipformer 593 / 769 ms against 570 / 770 ms.
**Always-on capacity test (TIMING entry 4b, xbabe0, 23:32–23:50Z, the host otherwise idle):** N live streams paced at 1×, one `ultra stream --pace 1.0` process each, all on one core (ours on CPU 16, the Zipformer on CPU 20, side by side; 128 s of audio = the first 22 panel utterances; 12 s cut at both ends for the watts; `derived/capacity-test-entry4b.jsonl`, receipts in `paced/`):

| System | N streams on one core | core busy | stable-word lag p50 / p95 (max) | W per stream, net of the core's idle floor | W per stream, gross | words matched |
|---|---:|---:|---|---:|---:|---:|
| ours (H22ctl) | 1 | 7% | 305 / 430 ms (501) | 0.224 | 0.485 | 96% |
| ours | 4 | 17% | 313 / 439 ms (519) | 0.390 | 0.426 | 96% |
| ours | 8 | 35% | 309 / 437 ms (514) | 0.335 | 0.372 | 96% |
| ours | **16** | 62% | **303 / 429 ms (513)** | **0.368** | **0.375** | 96% |
| Zipformer | 1 | 12% | 539 / 744 ms (929) | 0.303 | 0.588 | 95% |
| Zipformer | 2 | 23% | 552 / 754 ms (942) | 0.561 | 0.663 | 95% |
| Zipformer | 4 | 40% | 546 / 751 ms (939) | 0.520 | 0.595 | 95% |
| Zipformer | **8** | 65% | **548 / 755 ms (958)** | **0.592** | **0.639** | 95% |

- **Latency does not move with load up to the largest N tested:** ours holds p95 429–439 ms from 1 to 16 streams on one core (62% busy); the Zipformer 744–755 ms from 1 to 8 (65% busy). So the measured always-on capacity at a p95 ≤ 450 ms (ours) or ≤ 760 ms (Zipformer) bound is **at least 16 and at least 8 streams per core**; from the paced CPU cost the ceiling is about 26 and 12 (1 / CPU-s per audio-s per stream: 0.0387 and 0.0815, model load included). The lag of ours is lower than the Zipformer's at every N.
- **Watts per stream, paced against unpaced (pass rule 15%):** at 8 or more streams per core the paced figure agrees with the unpaced one: ours 0.335–0.375 W against 0.334 (+0 to +12%), the Zipformer 0.592–0.639 against 0.591 (+0 to +8%). **Pass**: the unpaced joules per audio-second can be quoted as watts per live stream for a core that carries several streams.
- **A core with one stream is a different regime.** The idle floor of the core (0.1–0.4 W) is then carried by one stream (gross 0.485 W for ours), while the governor keeps the core at 2.3 GHz instead of 4.2 GHz, so the work itself costs less (net 0.224 W). Both numbers have a large relative uncertainty (the idle floor drifted by 0.3 W between the windows, flagged).
- The Zipformer's last word arrives up to 0.54 s after the end of the audio at N = 8 (0.08 s at N = 1); ours 0.01–0.05 s.
- A first attempt on H0 (xbabe3, 21:31–22:15Z) is discarded: every repetition was flagged (SMT siblings busy with other agents' work, foreign load changing by tens of watts, idle drift), net core watts came out negative, and the engine ran 4–7× slower than on xbabe0 because of memory-bus load. Receipts kept in `/mnt/xbabe3-external/jsr/runs/energy-h39/runs/`.

### 3.9 Watts for every other row: the power log join

- From 21:28:48Z one sampler logs all 64 core registers and the package once a second with UTC timestamps (`xbabe0:~/jsr-data/opponents/energy/powerlog/power-YYYYMMDD.jsonl`, format in `energy_bench.py sample`).
- `energy_report.py join` turns a timed repetition (service CPUs, start, end, audio seconds) into joules per audio-second from the same run, lists how many other cores were busy and gives the package level during it. Nobody else has to read counters.
- From 21:27Z xbabe0 times one arm at a time (the lead's protocol v3, after the finding of §4.2), so every joined row is a quiet-host row.

## 4. What went wrong or surprised us

1. **The quiet lock did not make the package quiet.** It stopped benchmark agents, not other tenants. Four of V1's five repetitions were disturbed. Consequence: tighter flags, clean-repetition headlines, median-second levels and staircases (§2.8).
2. **Speed on xbabe0 depends on what the rest of the host does, even for a core alone on its L3 group.** With 16 busy benchmark cores the Zipformer cost 30% more CPU (0.0942 against 0.0722) at a clock only 3% lower; bench-cross saw 1.2–1.75× for our engine with 16–24 busy cores. The shared resource is the memory bus. The lead replaced the load rules with "one timed arm on the whole host at a time" at 21:27Z.
3. **Watts per busy core is not a property of a recogniser.** Ours reads 9.7 W alone, 6.9 W with four processes in one L3 group, 4.8 W with eight, 5.8 W on a contended xbabe3 core. Joules per audio-second is the stable quantity.
4. **I first reported 11.4 W for our engine alone.** That was the sum of all eight sampled cores; seven idle neighbours add 0.2 W each. Corrected within the hour (9.8 W). The fix in the tool: the core view covers only cores that carry a worker.
5. **My own smoke test ran inside my first measurement window**, and a helper I started 20 s too early would have collided with two cold rounds had they not just finished. Both are reasons for §2.8 rule 6.
6. **The first staircase line I posted (1.33, 14 W) used the end points**; least squares over the six plateaus gives 1.245 and 16 W.
7. **The INT8 engine needs the FP32 weights file to load** (`strace`: it opens `weights.safetensors`; without it: "No such file or directory"), so the shipped artifact is 304 MB, not 58 MB, until the loader changes.
8. **H0 was unusable for the paced check at high stream counts**: the memory bus there is so loaded that the same engine binary runs 4–7× slower.

## 5. Decision / consequence

- **Headline energy method for the Zen 2 one-core group: `core_msr_active`** (core register while running, joules per audio-second of the same run). It repeats within 1% and does not move with host load. The caption must say it leaves out the shared part of the CPU: about 20% of the marginal package watts for our engine, plus a one-off 16 W whenever any core is busy.
- **`power_bench.py` (runtime-lane, on main since PR #38) is sound for that quantity** and is not replaced. This wave adds what it lacks as a second, independent tool (own sampler, no turbostat): repetitions, idle windows, distrust flags, the package view, paced streams, the power-log join. Both write the same evidence line (`energy_report.py sidecar`, with `--power-bench` for its summaries). Whether to fold the two into one is the lead's call; my suggestion is to keep `power_bench.py`'s arm list and let it call this meter.
- **Rules for the paper's loader:** joules per audio-second from one run only; a joule figure may be attached to a cost row of another session on the same CPU, a "watts per busy core" may not; package figures only from inside one run (staircase), never run-minus-idle on a shared host.
- **Not done here:** wall power (no meter, no BMC sensor), DRAM power, the Mac (needs sudo for `powermetrics`), the cloud VMs (no counters), boost off (owner decision, not made).

## 6. Pointers

- Tool and tests: `python/ultra/tools/energy_bench.py` (meter), `python/ultra/tools/energy_report.py` (fits, staircase levels, join, evidence lines), `python/ultra/tests/test_energy_bench.py`; driver `docs/paper/run-energy.sh`.
- Receipts (small, in git, pinned in `raw/MANIFEST.txt`): `docs/paper/data/raw/energy/` — `v1-idle-vs-idle.json`, `v2-spin-{1,2,4,8}.json`, `engine-stair-a.json`, `zipformer-stair-a.json` (receipts); `series/` (per-second series of the two staircases); `cold/<arm>/round-N/` (cold-contract receipts of the anchor rounds) and `cold/sampler-cpus-16-20.jsonl` (the counter log under them); `derived/` (fits, levels, idle pairs, cold-round energies); `evidence/xbabe0-energy.jsonl` (schema-v2 rows and energy lines for the paper loader).
- On the hosts: `xbabe0:~/jsr-data/opponents/energy/` (binaries, models, audio sets, every run directory, the power log); `/mnt/xbabe3-external/jsr/runs/energy-h39/` (the paced runs on H0).
- Hashes: `ultra-cold-wav` 457a4db7… and `ultra` f4c1c751… (clean e2ef608 builds), runtime-lane's `ultra-cold-wav` aa596637… (byte-identical hypotheses), D2e-16457 INT8 export 5633a93e…, Zipformer bundle manifest 2f98a3f2…, sherpa-onnx c72c8b56… with ONNX Runtime 4b3607ae…, `ultra-measure` 98a8ca0f…, `ultra-score` d784f8ca…, panel digest 010aa985….
- Method background: R. Schöne et al., "Energy Efficiency Aspects of the AMD Zen 2 Architecture", IEEE CLUSTER 2021 (arXiv:2108.00808).
