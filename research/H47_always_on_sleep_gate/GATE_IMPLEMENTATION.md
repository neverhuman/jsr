# H47 Rust Sleep Gate & Ladder Implementation (`crates/ultra-gate`, `crates/ultra-alwayson`)

**Status:** IMPLEMENTATION CONTRACT (complements the wave pre-registration in `README.md`, owned by `@h47-sleep`).  
**Scope:** Runtime sleep gate + 5-state always-on ladder (`crates/ultra-gate`, `crates/ultra-alwayson`), one-pass batched wake replay, exact lifecycle/recovery continuity, and opt-in Linux per-thread `--uclamp-max` pacing. Zero model weights modified; zero training required; DEV-only evaluation (LibriSpeech test sets remain sealed).

---

## 1. Question & Hypothesis

An always-on streaming recogniser spends most of wall time in non-speech. Today ULTRA executes the full Conformer encoder on every `160 ms` chunk continuously (`0.036` CPU-s/audio-s), and with periodic reset-with-replay (`T_idle = 1.5 s`, `R = 1.0 s`) silence costs `~1.6×` active speech due to repeated pre-roll re-encoding.

**Hypothesis:** Placing a zero-float, pure-integer wake gate (`Sentinel-0`, `160`-sample `10 ms` hops, Martin min-stats `1.25 s` noise floor + `-72 dBFS` absolute floor) in front of a fixed `3.0 s` (`48_000` samples = `96 KB`) circular PCM16 ring and an explicit 5-state sleep ladder (`Listening`, `Waking`, `Active`, `Draining`, `Recovery` — explicitly **no** warm-pause or suppressed-emission state) with one-pass batched `1.0 s` wake replay (`with_max_chunks(16)`):

1. Cuts non-speech CPU cost to `<= 2%` of continuous active CPU (`0` engine calls in `0%` speech).
2. Cuts total energy (J/stream-s) by `>= 2×` at `5%` speech duty (paired physical measurement on `xbabe0` and metered target device; core + package + gross input scopes reported separately).
3. Preserves exact per-segment isolated-decode identity (`100%` zero-mismatch tokens against standalone decode of `pcm[s0..e]`, `e = s0 + 640*T - 240`, cross-ISA x86 AVX2 == aarch64 NEON).
4. Holds first-word recall within `>= continuous - 0.5 pp` and adds `<= 5%` gate overhead at `100%` speech.
5. Maintains false words `<= 0.10 / h` on non-speech (verified on a `>= 30 h` zero-event suite — see §4).

---

## 2. Pre-Registered Arms & Controls

All arms run on the exact same frozen audio stream, checkpoint (`B0 = A1-49000 EMA INT8`), and build:

| Arm | Description | Role |
|---|---|---|
| **A0 (Continuous)** | `ultra-engine-endless` with no reset and no gate (`100%` duty) | Baseline continuous always-on |
| **A1 (Ungated Reset-Replay)** | `AlwaysOnSession` with `gate_enabled = false`, `T_idle = 1500 ms`, `R = 1000 ms`, one-pass batched replay | Matched ungated control |
| **A2 (Oracle Speech Gate)** | Ideal MFA-aligned speech intervals with `1.0 s` pre-roll | Upper-bound ceiling for duty & recall |
| **A3 (Sentinel-0 Ladder)** | `AlwaysOnSession` with `Sentinel-0` defaults (`theta_fast=640`, `theta_slow=256`, `nabs_fast=3943`, `nabs_slow=4455`), `ring=3.0 s`, `R=500 ms` (`8_000` samples), `T_idle=1500 ms` (Q pass 2 verified optimum on `27.56 h` real stream: `8.127%` WER, `-0.250 pp` vs ungated C1, `+1.01 pp` first words, `0/48` hot-close duplicates, `50.9%` duty; `1500 ms` prevents mid-utterance restarts and hot-close word duplication under recogniser-blank close, whereas `(500, 700)` applies only to PR #75's gate-driven close; validation enforces `R <= T_idle - 300 ms`), `flush_on_sleep_close=false` (complete chunks only at mid-stream sleep close), depths `20 / 80 / 160 ms` | Primary H47 candidate |
| **A4 (Sentinel-0 + `uclamp`)** | Arm A3 with opt-in per-thread `--uclamp-max` sweep `{1024, 640, 512, 384, 256}` (O18-approved per-thread `sched_setattr`, no machine-wide change) | Energy-pacing arm |

**Seeds & Datasets (DEV only; test sets sealed):**
- Golden cross-ISA determinism vector: LCG seed `12345`, `96_000` samples (`FNV-1a-64` PCM `0x6f5d0b63084f8ffd`, trace `0xc7b5b4e8da91b73f`).
- Always-on benchmark (`ULTRA-AO v1`, DEV-only): duty `{0%, 5%, 20%, 50%, 100%}` × onset rate `{1, 6}` / min × `{silence, pink 20 dB SNR, white 10 dB SNR}` + torture sets `T1–T4` (fricative/quiet onsets, post-noise, `-20 / -40 dB`), bootstrap seed `20261122` (`10_000` speaker-cluster resamples).
- Full 27.56 h DEV endless stream (`part-1..4.wav`) + `30 h` non-speech false-word suite.

---

## 3. Acceptance Gates (Fixed Before Looking)

| Gate | Metric | Pass Criterion |
|---|---|---|
| **G-H47-1 (Bit-Exact Identity)** | `ultra-alwayson-check --verify-identity` | `0` segment token mismatches across `100%` of wake segments; x86 AVX2 == aarch64 NEON; exact sample conservation (`samples_in == samples_dropped + ring_len`, `ring_underflows == 0`) |
| **G-H47-2 (Non-Speech CPU)** | Process CPU-s/audio-s at `0%` speech duty | `<= 2%` of continuous active CPU (`0` engine decode calls on digital silence and sub-`-72 dBFS` noise) |
| **G-H47-3 (5% Duty Energy)** | Measured J/stream-s at `5%` speech duty (`xbabe0` / metered device) | `>= 2.0×` lower J than continuous control (A0) under paired physical metering (gross/package/core scopes reported separately) |
| **G-H47-4 (First-Word Recall & WER)** | First-word recall & pooled stream WER vs continuous/A1 | First-word recall `>= continuous - 0.5 pp`; stream WER delta `<= +0.10 pp` point (`<= +0.20 pp` 95% upper bound) |
| **G-H47-5 (False Words)** | False words per hour on non-speech | `<= 0.10 / h` (95% one-sided Poisson upper bound; see §4) |
| **G-H47-6 (100% Speech Overhead)** | Active CPU at `100%` speech duty (A3 vs A1) | `<= +5%` gate + ring overhead |
| **G-H47-7 (Wake Latency Tail)** | Cold vs warm stable-word latency (`p50` / `p95`) | Within `+40 ms` mean / `+80 ms` tail at depth 1; one-pass wake replay (`wake_decode_calls == 1`) |

**Kill Rule:** If `G-H47-1` (bit-exact segment identity or sample conservation) fails on any chunk size or ISA, stop immediately. If `Sentinel-0` fails `G-H47-4` after up to three `-64` (`-0.75 dB`) `theta_slow` calibration steps on DEV, record as negative and escalate to sub-band/PCEN gate calibration without touching weights.

---

## 4. Poisson Sample-Size Requirement for False Words (`<= 0.10 / h`)

By the exact one-sided 95% Poisson upper bound for `k = 0` observed false words over `T` hours of non-speech audio:

$$\lambda_{\text{upper}, 95\%}(k=0) = \frac{-\ln(0.05)}{T} \approx \frac{2.9957}{T}$$

- A `10 h` zero-event run bounds the false-word rate only to `<= 0.300 / h`, which is **insufficient** to close the `<= 0.10 / h` gate.
- Closing `<= 0.10 / h` at 95% confidence with zero observed false words requires at least **`T >= 29.96` zero-event hours** (`30 h` non-speech suite spanning digital silence, stationary white/pink/brown noise, and non-speech room tone).

---

## 5. Artifact & Repository Hygiene Contract

- **Zero weights committed** under `research/H47_always_on_sleep_gate/`; directory stays strictly `< 5 MB` (summaries, configs, and JSON receipts only).
- **No warm-pause / suppressed-emission** in H47: the 5 states are strictly `Listening`, `Waking`, `Active`, `Draining`, and `Recovery`.
- **Lifecycle & recovery continuity:**
  - `finish()` is idempotent; any `push_pcm` after `finish()` returns `Err(AlwaysOnError::AlreadyFinished)` in both gated and ungated modes.
  - A mid-call backend failure returns `AlwaysOnError::BackendFault { consumed_samples, capture_end, message }`, records an explicit `ContinuityGap`, resets the ring timeline to `capture_end` while conserving `samples_in == samples_dropped + len`, and resumes post-`recover()` segments at exact absolute capture coordinates (zero timestamp drift).
  - If `finish()` arrives while `Listening` before Sentinel-0 exits its `70 ms` startup warm-up (`gate.hops() < KS`) and buffered audio exceeds `-72 dBFS` (`nabs_fast`), `flush_short_onset_on_eof = true` (default) replays and decodes the short onset as `SegmentCause::EofShortOnset` (digital silence stays at `0` engine calls; opt-out reports `unflushed_short_onset_samples` explicitly).
- **Per-episode isolation:** every segment starts from a fresh `Engine::create_stream()` with explicit per-episode reset counters (`ultra-alwayson` does not rely on `EngineStream::restart` or `endpoint_reset_ms`).
- **O18 power compliance:** `--uclamp-max` uses per-thread `sched_setattr(pid=0, SCHED_FLAG_UTIL_CLAMP_MAX)` only; if the host kernel lacks `CONFIG_UCLAMP_TASK`, the tool logs `Unsupported` and continues unclamped cleanly.
