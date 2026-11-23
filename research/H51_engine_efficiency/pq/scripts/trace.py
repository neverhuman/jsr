"""TRAIN activation traces through A1-49000 (engine-faithful INT8 emulation, chunk 4, greedy).

For every engine int8 GEMM input (130 Linears + stem conv 2 as im2col rows) it accumulates:
  * the input second moment Sigma = sum x x^T (float64) for activation-weighted PQ;
  * real group sparsity of the int8 codes the kernel would see, per row and as row intersections
    (aligned 4-frame chunk steps; random 16-row multi-stream batches), for group widths 2 / 4 / 8
    and the pqscan skip unit (a pair of subspaces = 2s values);
    - quantiser: the engine's per-row CENTRED int8 (deployed) and a zero-anchored symmetric int8;
    - rotation: none, block-16 Walsh-Hadamard (H24 FWHT16), and a seeded random orthogonal;
  * row statistics: crest factor max|x|/rms, centre offset |c| / half-range, zero-code fraction.
No synthetic masks: every number comes from real TRAIN audio.

usage: python -I trace.py <threads> <out_dir> [n_utts]
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, "/mnt/xbabe3-external/jsr/h51-pq/scripts")
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import h51lib as L  # noqa: E402

threads, out_dir = int(sys.argv[1]), sys.argv[2]
u_start, u_end = int(sys.argv[3]), int(sys.argv[4])
torch.set_num_threads(threads)
os.makedirs(out_dir, exist_ok=True)

model, p = L.load_model("/dev/shm/h51-pq/A1-49000-ema-as-model.pt")
names = L.apply_int8(model)
all_utts = L.load_npz_utts("/dev/shm/h51-pq/train300.npz")
utts = all_utts[u_start:u_end]
exs, tok, _ = L.featurize(p, utts)

GROUPS = (2, 4, 8)
ROTS = ("none", "fwht16", "randorth")
QUANTS = ("centred", "symmetric")
_rot_cache: dict = {}


def rot(x, kind):
    if kind == "none":
        return x
    if kind == "fwht16":
        return L.fwht16(x)
    k = x.shape[-1]
    if k not in _rot_cache:
        path = f"/dev/shm/h51-pq/rot/randorth_{k}.npy"
        _rot_cache[k] = torch.from_numpy(np.load(path))
    return x @ _rot_cache[k]


class Acc:
    def __init__(self, k):
        self.k = k
        self.sigma = torch.zeros(k, k, dtype=torch.float64)
        self.sum = torch.zeros(k, dtype=torch.float64)
        self.rows = 0
        self.cal_rows = 0
        self.held = []
        self.zero = {}  # (quant, rot, unit) -> [zero groups, groups]
        self.inter4 = {}  # (quant, rot, s) -> [zero pairs over 4-row chunks, pairs]
        self.crest = {r: 0.0 for r in ROTS}
        self.centre = 0.0
        self.zcode = {(q, r): 0.0 for q in QUANTS for r in ROTS}
        self.pz_samples = {q: [] for q in QUANTS}  # s=4 pair-zero rows (no rotation), for M=16

    def add(self, x: torch.Tensor, chunked: bool, rng: np.random.Generator, sigma: bool = True):
        x = x.float()
        t, k = x.shape
        if state["held"] and sigma:
            take = rng.choice(t, size=min(t, state["per_utt"]), replace=False)
            self.held.append(x[torch.as_tensor(take)].clone())
        elif sigma:
            self.sigma += (x.t() @ x).double()  # fp32 per call (<= ~10^3 rows), fp64 across calls
            self.sum += x.sum(0).double()
            self.cal_rows += t
        self.rows += t
        # Deployed case (centred int8, no rotation) on every row; the other quantiser / rotation
        # variants on a subsample of aligned 4-row blocks (<= 64 rows per call).
        if chunked and t >= 4:
            nb = t // 4
            pick = np.sort(rng.choice(nb, size=min(nb, 16), replace=False))
            sub = torch.as_tensor((pick[:, None] * 4 + np.arange(4)[None]).reshape(-1))
        else:
            sub = torch.as_tensor(np.sort(rng.choice(t, size=min(t, 64), replace=False)))
        xsub = x[sub]
        rotated = {r: rot(xsub, r) for r in ROTS}
        for r in ROTS:
            for qn in QUANTS:
                full = r == "none" and qn == "centred"
                xr = x if full else rotated[r]
                ts = xr.shape[0]
                if qn == "centred":
                    rms = xr.pow(2).mean(-1).sqrt().clamp(min=1e-30)
                    self.crest[r] += float((xr.abs().amax(-1) / rms).sum()) * (t / ts)
                    q, sc, c = L.quant_rows_centred(xr)
                    if r == "none":
                        half = sc * L.QMAX
                        self.centre += float((c.abs() / half.clamp(min=1e-30)).sum())
                else:
                    q, sc = L.quant_rows_symmetric(xr)
                z = q == 0
                self.zcode[(qn, r)] += float(z.float().mean(-1).sum()) * (t / ts)
                for s in GROUPS:
                    for unit, w in (("g", s), ("pair", 2 * s)):
                        if k % w:
                            continue
                        gz = z.view(ts, k // w, w).all(-1)
                        a = self.zero.setdefault((qn, r, f"{unit}{s}"), [0, 0])
                        a[0] += int(gz.sum())
                        a[1] += gz.numel()
                        if unit == "pair" and chunked and ts >= 4 and (full or ts % 4 == 0):
                            t4 = ts // 4 * 4
                            iz = gz[:t4].view(t4 // 4, 4, -1).all(1)
                            b = self.inter4.setdefault((qn, r, s), [0, 0])
                            b[0] += int(iz.sum())
                            b[1] += iz.numel()
                        if unit == "pair" and s == 4 and r == "none":
                            take = rng.choice(ts, size=min(ts, 48), replace=False)
                            self.pz_samples[qn].append(gz[torch.as_tensor(take)].numpy())

    def summary(self, rng):
        out = {"k": self.k, "rows": self.rows, "cal_rows": self.cal_rows,
               "held_rows": int(sum(h.shape[0] for h in self.held))}
        n = max(1, self.rows)
        out["crest"] = {r: self.crest[r] / n for r in ROTS}
        out["centre_over_half"] = self.centre / n
        out["zero_code_frac"] = {f"{q}/{r}": self.zcode[(q, r)] / n for q in QUANTS for r in ROTS}
        out["zero_group_frac"] = {f"{q}/{r}/{u}": a[0] / max(1, a[1])
                                  for (q, r, u), a in sorted(self.zero.items())}
        out["zero_pair_frac_4row"] = {f"{q}/{r}/s{s}": a[0] / max(1, a[1])
                                      for (q, r, s), a in sorted(self.inter4.items())}
        for qn in QUANTS:
            if self.pz_samples[qn]:
                allp = np.concatenate(self.pz_samples[qn])
                fr = []
                for _ in range(200):
                    idx = rng.choice(len(allp), size=16, replace=False)
                    fr.append(allp[idx].all(0).mean())
                out.setdefault("zero_pair_frac_16row_random", {})[f"{qn}/none/s4"] = float(np.mean(fr))
        return out


accs: dict[str, Acc] = {}
N_CAL = int(os.environ.get("H51_NCAL", "250"))  # Sigma from utts [0, N_CAL); held-out rows after
state = {"chunked": True, "held": False, "per_utt": 24}
rng = np.random.default_rng(5151)


joiner_buf: list = []


def hook_for(name):
    def pre(mod, inp):
        x = inp[0]
        x2 = x.reshape(-1, x.shape[-1])
        if name == "joiner.out":  # one row per greedy step: buffer, process per utterance
            joiner_buf.append(x2.detach().clone())
            return
        if name not in accs:
            accs[name] = Acc(x2.shape[-1])
        accs[name].add(x2, state["chunked"], rng)
    return pre


def flush_joiner():
    if not joiner_buf:
        return
    x = torch.cat(joiner_buf)
    joiner_buf.clear()
    if "joiner.out" not in accs:
        accs["joiner.out"] = Acc(x.shape[-1])
    accs["joiner.out"].add(x, False, rng)


for n in names:
    L.get_mod(model, n).register_forward_pre_hook(hook_for(n))

# stem conv 2: im2col rows (K = 128 * 9), 19 frequency positions per encoder frame
stem = model.encoder.stem
_orig_c2 = None


def c2_pre(mod, inp):
    x = inp[0]  # (B, C, T, F) padded input of conv 2
    cols = F.unfold(x.float(), mod.kernel_size, stride=mod.stride)  # (B, K, L)
    rows = cols.transpose(1, 2).reshape(-1, cols.shape[1])
    take = torch.as_tensor(np.sort(rng.choice(rows.shape[0], size=min(rows.shape[0], 256), replace=False)))
    if "encoder.stem.c2" not in accs:
        accs["encoder.stem.c2"] = Acc(rows.shape[-1])
    accs["encoder.stem.c2"].add(rows[take], False, rng, sigma=False)


# With int8_qat the stem calls train.qat.int8_conv2d_as_gemm(x, c2) instead of c2.forward, so
# wrap that function (looked up at call time inside CausalStem.forward).
import train.qat as _qat  # noqa: E402

_orig_gemm = _qat.int8_conv2d_as_gemm


def _c2_wrapped(x, conv):
    c2_pre(conv, (x,))
    return _orig_gemm(x, conv)


_qat.int8_conv2d_as_gemm = _c2_wrapped

from train.data import collate  # noqa: E402
from train.transducer import greedy_decode  # noqa: E402

t0 = time.time()
with torch.no_grad():
    for i, e in enumerate(exs):
        state["held"] = (u_start + i) >= N_CAL
        b = collate([e])
        enc, lens = model.encode(b.feats, b.feat_lens, None)
        enc_p = model.joiner.enc_proj(enc.float())
        greedy_decode(model.predictor, model.joiner, enc_p, lens, 8)
        flush_joiner()
        if i % 10 == 0 or i < 3:
            print(f"{L.now()} utt {i}/{len(exs)} {time.time() - t0:.0f}s", flush=True)

for a_ in accs.values():
    a_.pz_samples = {q: (np.concatenate(v) if v else np.zeros((0, 1), bool)) for q, v in a_.pz_samples.items()}
    a_.held = torch.cat(a_.held) if a_.held else None
meta = {"utts": len(exs), "u_start": u_start, "u_end": u_end,
        "seconds": sum(len(u["pcm"]) for u in utts) / 16000.0,
        "chunk": model.encoder.cfg.chunk, "left_context": model.encoder.cfg.left_context,
        "precision": "engine int8 emulation (all int8 GEMMs)", "elapsed_s": time.time() - t0,
        "checkpoint_sha256": "4055c989c3822b326c52f5a0abb055810ddef7dab7ea85acffa388d134a3a70c"}
torch.save({"meta": meta, "accs": {n: a_.__dict__ for n, a_ in accs.items()}},
           os.path.join(out_dir, f"part_{u_start:03d}_{u_end:03d}.pt"))
print(json.dumps(meta), flush=True)
