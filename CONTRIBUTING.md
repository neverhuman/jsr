# Contributing to JSR / ULTRA

This guide is for new contributors, including external coding agents. Read it before your first PR.

## 1. What the project is
ULTRA is a CPU-first streaming speech recogniser. It has two halves:
- **R1:** a streaming Conformer transducer, trained in PyTorch (`python/ultra/train`).
- **The engine:** a pure-Rust INT8 inference engine (`crates/ultra-engine`). Its output is bit-identical across x86 AVX2 and ARM NEON.

The goal is the most live streams per CPU core at a bounded latency, with competitive WER. Today the throughput side is strong. Accuracy is still behind (see the README snapshot).

Read, in order:
1. `README.md`: the current numbers.
2. `docs/ultra/STATUS.md` and `docs/ultra/PLAN_TO_V1.md`: where we are and what's left.
3. `docs/ultra/FREEZE.md`: the qualifying protocol.
4. `research/README.md`: research waves H1…; one folder per wave.
5. `AGENTS.md` and `agent/JANKURAI_STANDARD.md`: repo rules for agents.

## 2. Non-negotiable rules
- **Data hygiene:**
  - The **LibriSpeech test sets are sealed** until `bench-freeze-v1` (`docs/ultra/LOOKS.jsonl`). Choose and tune on DEV only.
  - Never commit audio, checkpoints or build output. Downloaded data is untrusted until hashed (`docs/ultra/DATA.lock`).
- **Numbers:**
  - **Label every number exploratory** unless it comes from the FREEZE qualifying run.
  - Never pair a WER from one checkpoint with a speed from another.
  - Never combine ratios from separate runs.
  - Every number in docs and the paper traces to a receipt, a wave README or a generated data file.
- **Push gate:** `just required` (fmt, clippy `-D warnings`, tests, pytest, security, jankurai audit). The jankurai score must stay **≥ 85 with 0 caps**.
- **Workflow:**
  - **One writer per branch.** Work on your own branch and open a PR against `main`. Don't push to another contributor's branch; send a PR into it, or comment.
  - **Small PRs.** Each one says what it changes, how it was tested, and which research wave it belongs to.
  - **Licences.** Don't vendor code without a compatible licence. Credit reimplementations (e.g. ScaledAdam from icefall, Apache-2.0).

## 3. Setup
```bash
just venv        # Python env in .venv (python/ultra/requirements.txt)
just fast        # fmt, clippy, Rust tests, pytest
just required    # the full push gate
```
Build with `CARGO_INCREMENTAL=0`; the Justfile sets it.

For local Jankurai audit hooks, run these commands from the repository root.

```bash
bash -o pipefail -c 'curl --proto "=https" --tlsv1.2 -fsSL https://raw.githubusercontent.com/neverhuman/jankurai/v1.7.0/jankurai-installer.sh | bash -s -- --tag v1.7.0'
export PATH="$HOME/.local/bin:$PATH"
jankurai init --yes
jankurai hooks install --yes
```

The pre-commit hook runs `jankurai diff-audit --base-ref HEAD`. Run `just score`
for the full repository audit and `jankurai upgrade` to upgrade the tool.

## 4. Data
- **Location:** set `JSR_DATA` to a directory with at least 100 GB free. The maintainers use `/mnt/xbabe3-external/jsr`.
- **Download:** `python/ultra/scripts/fetch_librispeech_hf.py` fetches LibriSpeech (CC BY 4.0) from the `openslr/librispeech_asr` parquet mirror into `$JSR_DATA`. Check the hashes against `docs/ultra/DATA.lock`.
- **Usage:** `python/ultra/ultra/libri.py` reads the manifests. Training uses `train-clean-100`, `train-clean-360` and `train-other-500`; evaluation uses `dev-clean` and `dev-other` only.

## 5. Training and evaluation
- **Trainer:** `python -m train.trainer --run-dir <dir> …`, run from `python/ultra`. Run scripts and recipes are in `python/ultra/train/ops/` (see its README).
- **Logging:** each run writes `train.jsonl` (`kind: train / eval`) and its checkpoints to its run directory.
- **Current best recipe:**
  - model S (d384, 16 Conformer blocks, 160 ms chunks);
  - SentencePiece-512 for new runs;
  - pruned RNN-T (k2) plus CTC, EMA weights, speed perturbation, a qkv cap, and a divergence guard.
  - Decode with beam 4 plus averaging of the last 3 checkpoints.
- **Hardware limits:** keep GPU and CPU at a 5-minute mean of at most 90% on shared hosts. Wrap long runs with `ultra/fleet_guard.py`.

## 6. Open workstreams
Before starting, check the open PRs and issues; ask in an issue or PR comment if unsure. The maintainers' agent coordination file is local and not in git, so use GitHub.

| Area | Task | Where |
|---|---|---|
| Accuracy | Close the WER gap to Zipformer (2.85% dev-clean) without losing density: distillation (H22), recipe and optimiser studies (H21; H23 options `optimizer` / `grad_clip_mode`), LM fusion in beam search | `python/ultra/train`, `research/` |
| Multi-GPU | Data-parallel trainer: exact global-batch sharding, native DDP overlap, step-synchronised throttle; DiLoCo as a fallback for slow links. `speed_pack.py` needs fixes before it can be wired in: keep `Batch.kd` on transfer, SpecAugment draw parity, a TF32 quality check | `python/ultra/train` (`trainer.py` DDP options), `speed_pack.py`, `diloco.py` |
| Engine | Rust beam search with LM fusion; AVX-512 VNNI and NEON dot-product INT8 paths; keep cross-ISA bit-exactness | `crates/ultra-engine`, `crates/ultra-kernels` |
| Benchmark | Qualifying single-harness run (FREEZE §4); more baselines (Moonshine, other Zipformer sizes) in the same harness | `crates/ultra-cli` (`ultra bench-suite`), `crates/ultra-bridge` |
| Paper | Independent review of `docs/paper/` claims against receipts; refresh `data/metrics.tex` from new checkpoints | `docs/paper`, `python/ultra/tools/paper_data` |
| Tooling | Training dashboard improvements | `python/ultra/tools/fleet_dashboard` |

## 7. Research waves
- **New waves:** each new experiment gets the next free `H<n>` in `research/README.md` and a folder `research/H<n>_<slug>/README.md` with question, protocol, pre-registered decision rule, results and pointers.
- **Content:** keep summaries in git and leave raw data out. Record the commit the runs used.
- **Negative results:** they count, so record them (e.g. the ternary R4 study).
