"""H49 witness: with the H49 terms off, the patched tree trains bit-identically to the unpatched one
(pruned RNN-T + CTC + SpecAugment + chunk modes + EMA, the A1C path; TINY model, CPU, 4 steps)."""
import dataclasses, hashlib, json, sys
tree = sys.argv[1]
sys.path.insert(0, tree)
import torch
from train.asr import PRESETS
from train.data import Example, FixedBatches
from train.trainer import Trainer, TrainConfig
torch.manual_seed(0)
g = torch.Generator().manual_seed(1)
exs = [Example(f"u{i}", "x", torch.randn(60 + 8 * i, 80, generator=g), [1 + i % 9, 2, 3 + i % 5], [1, 2]) for i in range(8)]
mcfg = dataclasses.replace(PRESETS["TINY"], rnnt_loss="pruned", prune_range=3)
cfg = TrainConfig(preset="TINY", total_steps=8, warmup=2, log_every=1, eval_every=0, ckpt_every_steps=1000,
                  specaug=True, specaug_start=0, chunk_modes="2,4", ema_decay=0.9, rnnt_loss="pruned", prune_range=3)
torch.manual_seed(11)
tr = Trainer("/dev/shm/h49-seqkd/work/run-" + hashlib.sha1(tree.encode()).hexdigest()[:6], cfg,
             FixedBatches(exs, batch_size=4), model_cfg=mcfg, device="cpu", log=False)
rows = []
it = iter(tr._batches())
for _ in range(4):
    r = tr.train_step(next(it)[1])
    rows.append([r["loss"], r["rnnt"], r["ctc"], r["grad_norm"]])
h = hashlib.sha256()
for k, v in sorted(tr.model.state_dict().items()):
    h.update(k.encode()); h.update(v.detach().cpu().numpy().tobytes())
for k, v in sorted((tr.ema or {}).items()):
    h.update(k.encode()); h.update(v.detach().cpu().numpy().tobytes())
print(json.dumps({"rows": rows, "params_ema_sha256": h.hexdigest(), "rng": hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()}))
