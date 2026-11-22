# Owner rulings (spec §16)

| ID | Ruling | Date | Source |
|---|---|---|---|
| O1 | Trainer may be Python in `research/ultra/`. This is a standalone repo, so ADR-0006 does not apply. | 2026-11-20 | owner session |
| O2 | Approved: LibriSpeech 960 h + dev/test (CC BY 4.0); pretrained, internal evaluation only: icefall LibriSpeech streaming Zipformer, Nemotron Speech Streaming 0.6B + NeMo-Speech.cpp, Moonshine v2, Whisper small.en. Not approved: other corpora, other public eval sets. | 2026-11-20 | owner |
| O11 | First-week shock claim: stream density (real-time streams per core at unchanged WER). | 2026-11-20 | owner |
| O12 | Spec v1.0-B sets the order of work; DSL, ternary and cross-ISA exactness are optional lanes. | 2026-11-20 | owner-supplied spec |
| O3 | Compute: one RTX 3090, shared. Do not stop the h464 jobs; train when the GPU is free. | 2026-11-20 | owner |
| O4 | H0 = this host (Threadripper PRO 3995WX, AVX2), provisional. Machines A–E are UNTESTED. | 2026-11-20 | default |
| O6 | Gates of §1.4 and margins of §11.6 as written. | — | default |
| O7 | Audio operators live in an unregistered dialect table. | — | default |
| O8 | Both DICTATE and COMMIT are researched. | — | default |
| O10 | Repo is public (`neverhuman/jsr`); specs published as-is. | 2026-11-20 | owner |
| O13 | Benchmark qualifying window: right after D2 finishes (about 2026-11-22 09:00–14:00Z, GPU idle), before D1MR. No D2 interruption. | 2026-11-20 | owner |
| O14 | Word-timing reference: the public Montreal-Forced-Aligner LibriSpeech word alignments (timing only; hashed in DATA.lock; 20-utterance listening spot check). | 2026-11-20 | owner |
| O15 | Out-of-domain evaluation-only sets approved: TED-LIUM 3 test, VoxPopuli EN test, AMI IHM test. Sealed until the bench freeze; one read. LibriSpeech stays the primary board. | 2026-11-20 | owner |
| O14a | MFA source: LibriSpeech alignments from Zenodo 2619474 v1.0 (Lugosch et al. 2019), CC BY 4.0. Timing reference only; attribution in docs/ultra/data/word_timing.md. | 2026-11-20 | data lane, lead |
| O15a | OOD licences, via hf-audio/open-asr-leaderboard @ de79e57a, evaluation only: VoxPopuli EN test is CC0 1.0 (plus the Europarl legal notice); AMI IHM test is CC BY 4.0 (attribution to the AMI Meeting Corpus). TED-LIUM 3 is BLOCKED_RIGHTS: withdrawn upstream for copyright; not fetched and not substituted. Default unless the owner rules otherwise: v1 runs with 2 OOD sets. | 2026-11-20 | data lane, lead (owner decision pending on TED-LIUM) |
| O16 | More external recognisers approved for internal evaluation, benchmarking only: Vosk (en-us 0.22, small en-us 0.15); Whisper.cpp streaming (base, tiny); Moonshine streaming (tiny, small, medium); Nemotron 3.5 ASR Streaming 0.6B; Parakeet TDT 0.6B v3 (parakeet.cpp, onnx-asr, sherpa-onnx); Parakeet Redux with Photon; Parakeet Unified EN 0.6B; Nemotron Labs Unified ASR 0.6B. DEV audio only, every artifact hash-locked in CANDIDATES.lock, results exploratory. Picovoice Cheetah only if the owner supplies an access key; otherwise it is cited as vendor-reported. | 2026-11-22 | owner |
| O17 | The LibriSpeech language-model text (openslr 11) may be used for a decoding language model. Results that use it are reported in separate "+LM" rows. | 2026-11-22 | owner |
| O18 | Energy: CPU energy counters may be read with sudo for benchmarking. Machine-wide CPU settings (boost, governor, frequency limits) are not changed without a separate owner decision. | 2026-11-22 | owner |
