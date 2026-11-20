# Owner rulings (spec §16)

| ID | Ruling | Date | Source |
|---|---|---|---|
| O1 | Trainer may be Python in `research/ultra/`. This is a standalone repo, so ADR-0006 does not apply. | 2026-11-20 | owner session |
| O2 | Approved: LibriSpeech 960 h + dev/test (CC BY 4.0). Not approved yet: other corpora, public eval sets, pretrained external models. | 2026-11-20 | owner |
| O3 | Compute: one RTX 3090, shared. Do not stop the h464 jobs; train when the GPU is free. | 2026-11-20 | owner |
| O4 | H0 = this host (Threadripper PRO 3995WX, AVX2), provisional. Machines A–E are UNTESTED. | 2026-11-20 | default |
| O6 | Gates of §1.4 and margins of §11.6 as written. | — | default |
| O7 | Audio operators live in an unregistered dialect table. | — | default |
| O8 | Both DICTATE and COMMIT are researched. | — | default |
| O10 | Repo is public (`neverhuman/jsr`); specs published as-is. | 2026-11-20 | owner |
