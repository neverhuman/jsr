W00 Instrument Verification: PASS [H0-VERIFIED, deterministic + live C-ABI smoke, 2026-11-20]

# W00: Measuring Instrument & Streaming Contract Proof

- **Wave:** `W00` (Spec v1.0-B §31, §38.1, §38.3; ULTRA v1 §0.7 Day 2–3)
- **Evidence tier:** `E1` (deterministic contract, equivalence, and failure-probe checks)
- **Source crates:** `crates/ultra-core`, `crates/ultra-eval`, `crates/ultra-frontend`, `crates/ultra-model`, `crates/ultra-stream`, `crates/ultra-cli`
- **Verdict:** **PASS**

## Verified Instrument Guarantees

| Probe / Contract | Test / Command | Result |
|---|---|---|
| Transport packetization invariance (`10 ms`, `20 ms`, `37 ms`, short final packet) | `cargo test -p ultra-stream --test contract_probes transport_packetization` & `cargo test -p ultra-frontend logmel_and_cochlea_are_packet_size_invariant` | **PASS** — identical `TranscriptEvent` streams, log-mel frames, and integer Cochlea A7 frames across packet sizes |
| Deliberate difficult-word omission (§38.3 probe 1) | `probe_omitting_difficult_words_improves_conditional_delay_but_worsens_cq_and_wer` | **PASS** — conditional `p95` delay drops (`600 ms -> 50 ms`) while `CQ(750 ms)` drops (`1.00 -> 0.50`) and WER rises (`0% -> 50%`) |
| Reference-time shift isolation (§38.3 probe 2) | `probe_shifting_scorer_reference_times_never_changes_runtime_events` | **PASS** — shifting scorer word-end times alters reported `p50_s` (`300 ms -> 100 ms`) while runtime events and transcripts stay bit-identical |
| Causal endpoint hangover & late refiner rejection (§38.3 probes 3 & 4) | `probe_endpoint_hangover_exhausts_causal_budget_and_late_refiner_is_rejected_yet_charged` | **PASS** — hangover consumes causal `final_budget_ms` without reset; stale refiner transaction after new speech is rejected atomically while charging `0.125 CPU-s` and `8 MiB` |
| Multibyte UTF-8 Commit/Final replay, `Gap` prefix preservation & uninterruptible cancel (§38.3 probes 5 & 6) | `probe_replay_dedup_multibyte_utf8_gap_and_uninterruptible_cancel_and_1h_session` | **PASS** — identical `(stream_id, event_sequence, sha256)` replay is a no-op; altered payload errors; `Gap` preserves committed bytes; cancelled native call holds memory slot until reaped |
| 1-hour continuous session stability (§38.1) | `probe_replay_dedup_multibyte_utf8_gap_and_uninterruptible_cancel_and_1h_session` | **PASS** — `57,600,000` samples (`3,600 s`) hit `120` 30-second segment caps with bounded `< 32 KiB` retained audio |
| Reference-blind `.npz` audio vs scorer separation (§12.1, §35) | `ultra_frontend::npz::{load_npz_audio_only, load_npz_references}` | **PASS** — inference phase reads only `offsets.npy` + `pcm.npy`; `texts.npy` is opened only in the separate post-inference scoring phase |
| Native C-ABI struct layout equivalence vs `c-api.h` | `c_abi_struct_layout_matches_sherpa_header_when_present` | **PASS** — all 10 `sizeof` / `offsetof` checks across `SherpaOnnxOnline*` and `SherpaOnnxOffline*` match `/usr/bin/cc` against `sherpa-onnx/c-api/c-api.h` |
| Bundle & candidate lock loader guards (§11.2, §34.2) | `cargo test -p ultra-model` & `ultra candidate validate --verify-files` | **PASS** — refuses `REQUIRED` / `UNPINNED` placeholders, `..` path traversal, unsupported `avx512vnni` ISA on H0, and corrupted payload SHA-256 |
| CPU-only process verification | `ultra doctor --candidate <id>` | **PASS** — `/proc/self/maps` verified free of `libcuda.so` and GPU ONNX execution providers (`cpu_only_verified: true`) |

## Reproduce

```bash
cargo test --workspace
ultra doctor --candidate zipformer-int8 --hardware docs/ultra/HARDWARE.json
```
