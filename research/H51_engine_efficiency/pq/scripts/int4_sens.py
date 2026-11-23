"""H51 Addendum A: 4-bit weight-only screen (int4 symmetric per output channel, INT8 data path).

Fits every encoder GEMM (stem.out + 128 block GEMMs) with GPTQ (H = TRAIN Sigma + 1% damping,
static order, per-row scale max|w|/7) or round-to-nearest; reports held-out TRAIN output distortion
(centred int8 activations x int4 weights, i.e. the deployed arithmetic, vs fp32); writes the int4
values through the engine's own int8 packing into one joint variant and decodes the 730-utt DEV
panel with the native engine; scores with ultra-score against the INT8 baseline.

usage: python -I int4_sens.py --arm gptq|rtn [--group 0|128] [--threads 4]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import torch  # noqa: E402

import h51lib as L  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--arm", choices=("gptq", "rtn"), required=True)
ap.add_argument("--group", type=int, default=0, help="0 = per output channel; >0 = column-group scales")
ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--damp", type=float, default=0.01)
ap.add_argument("--keep-int8", default="", help="regex of encoder GEMMs left at INT8 (Amendment A.2)")
ap.add_argument("--tag", default="")
ap.add_argument("--decode-if-fails", default="", help="Amendment A.2 trigger: after fitting, wait for this arm JSON and decode only if its pooled ΔWER > +0.30 pp")
a = ap.parse_args()
torch.set_num_threads(a.threads)

ROOT = "/dev/shm/h51-pq"
TRACE = "/mnt/xbabe3-external/jsr/h51-pq/derived/trace"
OUT = "/mnt/xbabe3-external/jsr/h51-pq/results/int4"
ENGINE = "/mnt/xbabe3-external/jsr/h51-pq/bench/ultra-engine-eval"  # 968d3ea7..., e61121e
SCORE = "/home/ubuntu/jsr/target/paper-evidence/runtime/ultra-score"
BASELINE = "/mnt/xbabe3-external/jsr/h51-pq/results/engine_sens/base_c4.score-input.jsonl"
PANEL = os.path.join(ROOT, "panel730.npz")
NR, KG, QM = 16, 4, 7
tag = a.tag or (f"int4_{a.arm}" + (f"_g{a.group}" if a.group else ""))
os.makedirs(OUT, exist_ok=True)

model, p = L.load_model(os.path.join(ROOT, "A1-49000-ema-as-model.pt"))
names = [n for n in L.target_names(model) if n.startswith("encoder.")]
assert len(names) == 129, len(names)
if a.keep_int8:
    import re as _re
    kept = [n for n in names if _re.search(a.keep_int8, n)]
    names = [n for n in names if not _re.search(a.keep_int8, n)]
    print(f"keeping {len(kept)} GEMMs at INT8, {len(names)} in int4", flush=True)


def perm_for(n):
    if n != "encoder.stem.out":
        return None
    ch, f2 = 128, 19
    return torch.tensor([c * f2 + f for f in range(f2) for c in range(ch)])


def scales_for(W, group):
    if group == 0:
        return (W.abs().amax(1, keepdim=True) / QM).clamp(min=1e-12)  # (N, 1)
    N, K = W.shape
    s = (W.reshape(N, K // group, group).abs().amax(2) / QM).clamp(min=1e-12)  # (N, K/g)
    return s.repeat_interleave(group, dim=1)


def rtn(W, group):
    s = scales_for(W, group)
    return torch.clamp(torch.round(W / s), -QM, QM) * s


def gptq(W, H, group, damp, block=128):
    """GPTQ (Frantar et al. 2022) with lazy block updates; returns dequantised weights."""
    W = W.clone().double()
    N, K = W.shape
    H = H.double().clone()
    dead = torch.diagonal(H) <= 0
    H[dead, dead] = 1.0
    W[:, dead] = 0.0
    H += damp * torch.diagonal(H).mean() * torch.eye(K, dtype=torch.float64)
    Hinv = torch.cholesky_inverse(torch.linalg.cholesky(H))
    U = torch.linalg.cholesky(Hinv, upper=True)
    S = scales_for(W.float(), group).double().expand(N, K)  # fixed from the original weights
    Q = torch.zeros_like(W)
    for i1 in range(0, K, block):
        i2 = min(i1 + block, K)
        W1 = W[:, i1:i2].clone()
        E1 = torch.zeros_like(W1)
        U1 = U[i1:i2, i1:i2]
        for i in range(i2 - i1):
            if group and (i1 + i) % group == 0:
                # reference GPTQ groupsize behaviour: the group's scale comes from the current,
                # error-updated weights when the group starts (group divides the block)
                assert block % group == 0
                S = S.clone() if S.stride(1) == 0 else S
                S[:, i1 + i:i1 + i + group] = (W1[:, i:i + group].abs().amax(1, keepdim=True) / QM).clamp(min=1e-12)
            w = W1[:, i]
            s = S[:, i1 + i]
            q = torch.clamp(torch.round(w / s), -QM, QM) * s
            Q[:, i1 + i] = q
            e = (w - q) / U1[i, i]
            W1[:, i:] -= e[:, None] * U1[i, i:][None, :]
            E1[:, i] = e
        W[:, i2:] -= E1 @ U[i1:i2, i2:]
    return Q.float()


def pack_int8(w):
    n, k = w.shape
    kp = -(-k // KG) * KG
    panels = -(-n // NR)
    w = w.astype(np.float32)
    mx = np.abs(w).max(1).astype(np.float32)
    s = (mx / np.float32(127.0)).astype(np.float32)
    inv = np.where(mx > 0, np.float32(127.0) / np.where(mx > 0, mx, 1), 0).astype(np.float32)
    q = np.clip(np.round(w * inv[:, None]), -127, 127).astype(np.int8)
    qp = np.zeros((panels * NR, kp), np.int8)
    qp[:n, :k] = q
    data = qp.reshape(panels, NR, kp // KG, KG).transpose(0, 2, 1, 3).reshape(-1)
    scale = np.zeros(panels * NR, np.float32)
    scale[:n] = s
    wsum = np.zeros(panels * NR, np.float32)
    wsum[:n] = (q.astype(np.int64).sum(1).astype(np.float32) * s).astype(np.float32)
    return data, scale, wsum, q, s


def engine_name(n):
    import re
    m = re.match(r"encoder\.blocks\.(\d+)\.(\w+)\.(\w+)$", n)
    if m:
        b, mod, w = m.groups()
        kind = {("ff1", "w1"): "ff1_w1", ("ff1", "w2"): "ff1_w2", ("ff2", "w1"): "ff2_w1",
                ("ff2", "w2"): "ff2_w2", ("att", "qkv"): "qkv", ("att", "out"): "att_out",
                ("conv", "pw1"): "pw1", ("conv", "pw2"): "pw2"}[(mod, w)]
        return f"blocks.{b}.{kind}"
    return {"encoder.stem.out": "stem_out"}[n]


class St:
    def __init__(self, path):
        self.raw = bytearray(open(path, "rb").read())
        n = struct.unpack("<Q", self.raw[:8])[0]
        self.hdr = json.loads(self.raw[8:8 + n])
        self.base = 8 + n

    def put(self, name, arr):
        o0, o1 = self.hdr[name]["data_offsets"]
        b = arr.tobytes()
        assert len(b) == o1 - o0, (name, len(b), o1 - o0)
        self.raw[self.base + o0:self.base + o1] = b


t0 = time.time()
st = St(os.path.join(ROOT, "engine_base", "weights.int8.safetensors"))
rows = []
for n in names:
    lin = L.get_mod(model, n)
    W = lin.weight.data.float()
    pm = perm_for(n)
    sig = torch.from_numpy(np.load(os.path.join(TRACE, f"sigma.{L.short(n)}.npy")))
    x = torch.from_numpy(np.load(os.path.join(TRACE, f"held.{L.short(n)}.npy"))).float()
    if pm is not None:
        W, sig, x = W[:, pm], sig[pm][:, pm], x[:, pm]
    Wq = gptq(W, sig, a.group, a.damp) if a.arm == "gptq" else rtn(W, a.group)
    d, sc, ws, q8, s8 = pack_int8(Wq.numpy())
    st.put(engine_name(n) + ".q", d)
    st.put(engine_name(n) + ".scale", sc)
    st.put(engine_name(n) + ".wsum", ws)
    with torch.no_grad():
        b = lin.bias.data.float()
        y = x @ W.t() + b
        var = (y - y.mean(0)).pow(2).sum()
        xq = L.fake_quant_rows_fwd(x)
        r8 = float(((xq @ L.int8_weight(W).t() + b - y) ** 2).sum() / var)
        r4 = float(((xq @ Wq.t() + b - y) ** 2).sum() / var)
        w_rq = torch.from_numpy(q8.astype(np.float32) * s8[:, None])
        r4rq = float(((xq @ w_rq.t() + b - y) ** 2).sum() / var)
        sigd = sig.double()
        E = (W - Wq).double()
        werr = float(torch.einsum("nk,kl,nl->", E, sigd, E) / torch.einsum("nk,kl,nl->", W.double(), sigd, W.double()))
    rows.append({"layer": n, "n": int(W.shape[0]), "k": int(W.shape[1]), "rel_mse_int8": r8,
                 "rel_mse_int4": r4, "rel_mse_int4_int8requant": r4rq, "rel_werr_sigma": werr})
fit_s = time.time() - t0
vd = os.path.join(ROOT, "variants", tag)
shutil.rmtree(vd, ignore_errors=True)
os.makedirs(vd)
for f in ("weights.safetensors", "bpe.model", "tokens.txt", "mel_fbank.f32"):
    os.link(os.path.join(ROOT, "engine_base", f), os.path.join(vd, f))
with open(os.path.join(vd, "weights.int8.safetensors"), "wb") as fh:
    fh.write(st.raw)
sha = hashlib.sha256(st.raw).hexdigest()
arch = json.load(open(os.path.join(ROOT, "engine_base", "arch.json")))
arch["files"]["weights_int8"]["sha256"] = sha
json.dump(arch, open(os.path.join(vd, "arch.json"), "w"), indent=2)
if a.decode_if_fails:
    # the fit is WER-blind; the decode waits for the trigger arm and runs only if it failed the rule
    while not os.path.exists(a.decode_if_fails):
        time.sleep(10)
    time.sleep(2)
    trig = json.load(open(a.decode_if_fails))["summary"]["d_pooled_pp"]
    print(f"trigger arm pooled delta {trig:+.3f} pp", flush=True)
    if trig <= 0.30:
        print("trigger not met (trigger arm passed the WER rule): not decoding", flush=True)
        shutil.rmtree(vd, ignore_errors=True)
        raise SystemExit(0)
run = vd + "_run"
shutil.rmtree(run, ignore_errors=True)
t1 = time.time()
r = subprocess.run([ENGINE, "--weights", vd, "--npz", PANEL, "--limit", "730", "--precision", "int8",
                    "--streams", "32", "--packet-ms", "160", "--out", run], capture_output=True, text=True)
if r.returncode != 0:
    raise SystemExit(r.stderr[-2000:])
dec_s = time.time() - t1
ev = json.load(open(os.path.join(run, "engine_eval.json")))
z = np.load(PANEL)
meta = {str(z["uids"][i]): (str(z["speakers"][i]), str(z["splits"][i]), str(z["texts"][i]),
                            round(float(z["offsets"][i + 1] - z["offsets"][i]) / 16000.0, 3))
        for i in range(len(z["uids"]))}
hp = os.path.join(OUT, f"{tag}.score-input.jsonl")
with open(hp, "w") as fh:
    for u in ev["rows"]:
        spk, split, ref, dur = meta[u["id"]]
        fh.write(json.dumps({"id": u["id"], "speaker": spk, "dataset": split, "ref": ref,
                             "hyp": u["hyp"], "duration_s": dur}) + "\n")
sc = subprocess.run([SCORE, BASELINE, hp, "--seed", "20261122", "--replicates", "10000", "--alpha", "0.05",
                     "--md-out", os.path.join(OUT, f"{tag}.score.md")], capture_output=True, text=True)
score = json.loads(sc.stdout)
c = score["comparison"]
summary = {"tag": tag, "arm": a.arm, "group": a.group, "damp": a.damp, "layers": len(rows), "keep_int8": a.keep_int8,
           "int4_bytes_mb": round(sum(x["n"] * x["k"] for x in rows) / 2e6, 2),
           "variant_int8_sha256": sha, "fit_s": round(fit_s, 1), "decode_s": round(dec_s, 1),
           "wer_pooled": 100 * c["pooled_delta"]["observed_b"], "d_pooled_pp": 100 * c["pooled_delta"]["delta"],
           "d_lo": 100 * c["pooled_delta"]["lower"], "d_hi": 100 * c["pooled_delta"]["upper"],
           "d_clean": 100 * c["per_dataset"]["dev-clean"]["delta"], "d_other": 100 * c["per_dataset"]["dev-other"]["delta"],
           "mean_rel_mse_int8": sum(x["rel_mse_int8"] for x in rows) / len(rows),
           "mean_rel_mse_int4": sum(x["rel_mse_int4"] for x in rows) / len(rows),
           "mean_rel_mse_int4_int8requant": sum(x["rel_mse_int4_int8requant"] for x in rows) / len(rows),
           "engine": {k: ev.get(k) for k in ("wer_pct", "errors", "substitutions", "deletions", "insertions",
                                              "executable_sha256", "int8_prepared_sha256")}}
json.dump({"summary": summary, "layers": rows, "score": score}, open(os.path.join(OUT, f"{tag}.json"), "w"), indent=1)
shutil.rmtree(vd, ignore_errors=True)
shutil.rmtree(run, ignore_errors=True)
print(json.dumps(summary))
