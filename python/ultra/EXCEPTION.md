# Python exception: ULTRA training and data preparation

- **Granted:** 2026-11-20 (owner ruling O1, `docs/ultra/RULINGS.md`).
- **Expires:** 2027-06-30, or at the R5 release, whichever comes first.
- **Scope:** advanced-ML training (PyTorch trainer, losses, quantisation-aware training), dataset download and bank building, and evaluation glue that drives the Python trainer.
- **Not in scope:** product truth, the streaming runtime, scoring authority, proof lanes and release gates. Those are Rust (`crates/`).
- **Why:** no Rust training stack exists for transducer ASR with GPU kernels; the spec names Python for the trainer.
- **Exit:** trained models export to a bundle consumed only by the Rust runtime.
