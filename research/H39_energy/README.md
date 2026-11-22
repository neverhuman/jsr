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

## 3. Results

Pending (filled in after the validation runs).
