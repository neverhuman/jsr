**Verdict: PASS (pipeline fits). Loss reached 0.005 per token. The dense CTC head transcribes 16/16 utterances exactly. The transducer greedy decode gets 9/16 exact at the spec cap of 8 symbols per frame, and 12/16 at cap 32. The misses come from memorised burst emission, not from a bug.**

# D0: overfit 16 utterances

- **Date:** 2026-11-20.
- **Lane:** trainer (`lane/trainer`).
- **Run:** `/mnt/xbabe3-external/jsr/runs/train/D0_overfit16/` (`train.jsonl`, `config.json`, checkpoints 500 and 1000).

## Setup

- **Data:** the first 16 utterances of train-clean-100 (226 s of audio). One batch of 16, no SpecAugment, no dropout.
- **Model:** preset `D0`, 6.48M parameters.
  - Encoder: d=192, 4 blocks, 4 heads, chunk 4 frames (160 ms), left context 128 frames.
  - Same code path as S: causal stem, stateless predictor, joiner 320, 4,096-piece transducer vocabulary, 1,024-piece CTC.
- **Optimiser:** AdamW, lr 2e-3, warm-up 50, cosine to 1,500 steps.
- **Device:** this ran on the GPU. The trainer picks CUDA when it is available, and the GPU had just freed. That is not a CPU result. CPU throughput is measured separately below.
- **Command:**

```
python -m train.trainer --run-dir .../D0_overfit16 --preset D0 --overfit 16 --threads 16 \
  --set specaug=false warmup=50 total_steps=1500 eval_every=100 lr=0.002
python -m train.evaluate --run-dir .../D0_overfit16 --checkpoint .../ckpt-00001000.pt \
  --split train-clean-100 --limit 16 [--max-symbols 8|32]
```

## Results

| step | loss/token | RNNT/token | CTC/token |
|---:|---:|---:|---:|
| 50 | 5.83 | 2.30 | 11.76 |
| 130 | 0.56 | 0.085 | 1.59 |
| 1500 | 0.0052 | 0.0052 | 0.0001 |

Checkpoint 1000 (the step-1500 checkpoint write was stuck in fsync on the data drive):

| decoder | WER | exact utterances | forced advances |
|---|---:|---:|---:|
| CTC best path (dense 25 Hz) | 0.0% | 16/16 | — |
| transducer greedy, cap 8 | 15.2% | 9/16 | 77 |
| transducer greedy, cap 32 | 17.9% | 12/16 | 9 |

- CTC feasibility: 0 of 24,000 sequence-steps were infeasible.

## Why the transducer misses

- With only 16 utterances, the cheapest solution is to recognise the utterance from its first frames and emit the whole transcript at once.
- Measured per frame with cap 64: in every utterance, nearly all tokens (21 to 52) come out in one or two frames between frame 1 and frame 20.
- The stateless predictor sees only the last two tokens. Inside such a burst a repeated bigram (for example "in the") can send it into another memorised transcript. That accounts for the insertions and for the cap hits.
- This is a property of memorising 16 utterances. It does not show an alignment bug.
- The loss is validated against brute-force enumeration (`tests/test_train_loss.py`). Greedy decoding reproduces targets on a non-degenerate tiny fixture.
- D1 measures this properly: on real data, forced advances are logged with every dev evaluation.

## CPU training throughput (deliverable 4)

- **Model:** S preset, 58.7M parameters.
- **Placement:** pinned to `taskset -c 0-15,64-79` (the trainer block of the core partition), 16 torch threads.
- **Batches:** 200 audio-second batches with SpecAugment, using the same code path as the trainer.
- **Load:** 38.6 before and 52.8 after. The D1 data loader shared this block.

| torch build | audio-hours per wall-hour (16 cores) | peak RSS |
|---|---:|---:|
| system torch 2.7.0 (Lambda, `BLAS_INFO=generic`, Eigen) | far below 0.2: the first step did not finish in 7 min | — |
| PyPI torch 2.7.0+cpu (MKL), `/mnt/xbabe3-external/jsr/venvs/torch270-cpu` | **3.77** | 11.7 GB |

- **Recommendation: CPU training is not a viable fallback.** 3.8 h/h per 16 cores is a fifth of the 20 h/h threshold. Even all 64 cores, scaling perfectly, would give about 15 h/h, which is about 7 h per train-clean-100 epoch, and would take the whole machine.
- The GPU (RTX 3090) runs the same S model at **~373 audio-h per GPU-hour**, with bf16 and 1,000 audio-second batches. That is about 100× the 16-core CPU number.
- The system torch has no optimised BLAS. Any CPU torch work needs the MKL wheel above, through `PYTHONPATH=/mnt/xbabe3-external/jsr/venvs/torch270-cpu`.
