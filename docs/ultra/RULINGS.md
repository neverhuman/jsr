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
