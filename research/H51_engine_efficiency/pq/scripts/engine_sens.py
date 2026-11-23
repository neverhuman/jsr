"""Per-layer PQ sensitivity, WER measured with the native engine (ultra-engine-eval, origin/main
e61121e, unmodified) on the 730-utt DEV panel, chunk 4, greedy, 16 streams.

Per target GEMM: fit activation-weighted gain-shape PQ on TRAIN statistics (h51lib.PQFit), take
its dequantised weights W' through the engine's own int8 packing (per-row scale, same rule as
ultra-engine-prepare), write a variant model dir (hard links + one patched weights.int8 file +
arch.json with the new hash), decode the panel, score it against the INT8 baseline with the
canonical ultra-score (paired bootstrap). Held-out TRAIN distortion is reported for the exact
pqscan arithmetic (7-bit LUT), the LUT-free PQ product, the PQ->int8 requantised weights the
engine run uses, and the INT8 baseline, so the two arithmetic gaps are visible.

usage: python -I engine_sens.py --tag s4r2 --s 4 --R 2 --shard i/n [--layers all|a,b|group:<re>]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import torch  # noqa: E402

import h51lib as L  # noqa: E402

ROOT = os.environ.get("H51_ROOT", "/dev/shm/h51-pq")
OUT_DEFAULT = os.environ.get("H51_OUT", "/mnt/xbabe3-external/jsr/h51-pq/results/engine_sens")
ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--s", type=int, default=4)
ap.add_argument("--R", type=int, default=2)
ap.add_argument("--layers", default="all")
ap.add_argument("--shard", default="0/1")
ap.add_argument("--chunk", type=int, default=0)
ap.add_argument("--streams", type=int, default=32)
ap.add_argument("--unweighted", action="store_true")
ap.add_argument("--trace", default=os.path.join(ROOT, "trace"))
ap.add_argument("--base", default=os.path.join(ROOT, "engine_base"))
ap.add_argument("--work", default=os.path.join(ROOT, "variants"))
ap.add_argument("--out", default=OUT_DEFAULT)
ap.add_argument("--baseline", default=os.path.join(OUT_DEFAULT, "base_c4.score-input.jsonl"))
ap.add_argument("--keep", action="store_true")
ap.add_argument("--kc", type=int, default=16, help="codewords per subspace (16 = H24 pqscan, 4 = staged)")
ap.add_argument("--rows-suffix", default="", help="separate rows file for parallel group runs")
ap.add_argument("--no-distortion", action="store_true")
ap.add_argument("--gptq", action="store_true", help="GPTQ/GPTVQ-style error-feedback fit (h51lib.PQFitGPTQ)")
ap.add_argument("--reverse", action="store_true", help="walk this shard from the end (second host)")
ap.add_argument("--fit-cache", default=os.path.join(ROOT, "fits"))
ap.add_argument("--check-pack", action="store_true", help="verify packing reproduces the base file")
a = ap.parse_args()
torch.set_num_threads(1)

ENGINE = os.environ.get("H51_ENGINE", "/dev/shm/h51-pq/target/release/ultra-engine-eval")
SCORE = os.environ.get("H51_SCORE", "/home/ubuntu/jsr/target/paper-evidence/runtime/ultra-score")
PANEL = os.path.join(ROOT, "panel730.npz")
NR, KG = 16, 4

model, p = L.load_model(os.path.join(ROOT, "A1-49000-ema-as-model.pt"))
names = L.target_names(model)


def engine_name(n: str) -> str:
    m = re.match(r"encoder\.blocks\.(\d+)\.(\w+)\.(\w+)$", n)
    if m:
        b, mod, w = m.groups()
        kind = {("ff1", "w1"): "ff1_w1", ("ff1", "w2"): "ff1_w2", ("ff2", "w1"): "ff2_w1",
                ("ff2", "w2"): "ff2_w2", ("att", "qkv"): "qkv", ("att", "out"): "att_out",
                ("conv", "pw1"): "pw1", ("conv", "pw2"): "pw2"}[(mod, w)]
        return f"blocks.{b}.{kind}"
    return {"encoder.stem.out": "stem_out", "joiner.enc_proj": "enc_proj", "joiner.out": "joiner_out"}[n]


def perm_for(n: str):
    """Engine K order for stem_out is f * ch + c (torch: c * f2 + f); PQ subspaces follow the
    deployed order, so weights / Sigma / activations are permuted to it."""
    if n != "encoder.stem.out":
        return None
    ch, f2 = 128, 19
    perm = np.empty(ch * f2, np.int64)
    for c in range(ch):
        for f in range(f2):
            perm[f * ch + c] = c * f2 + f
    return torch.from_numpy(perm)


def eng_weight(n: str) -> torch.Tensor:
    w = L.get_mod(model, n).weight.data
    pm = perm_for(n)
    return w if pm is None else w[:, pm]


def eng_sigma(n: str) -> torch.Tensor:
    sig = torch.from_numpy(np.load(os.path.join(a.trace, f"sigma.{L.short(n)}.npy")))
    pm = perm_for(n)
    return sig if pm is None else sig[pm][:, pm]


def pack_int8(w: np.ndarray):
    """ultra-engine PackedI8::quantize (WeightRule::Engine), float32 arithmetic."""
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


class St:
    """Patch tensors of a safetensors file in place of their byte ranges (same dtype/shape)."""

    def __init__(self, path):
        self.raw = bytearray(open(path, "rb").read())
        n = struct.unpack("<Q", self.raw[:8])[0]
        self.hdr = json.loads(self.raw[8:8 + n])
        self.base = 8 + n

    def get(self, name, dtype):
        o0, o1 = self.hdr[name]["data_offsets"]
        return np.frombuffer(bytes(self.raw[self.base + o0:self.base + o1]), dtype=dtype)

    def put(self, name, arr):
        o0, o1 = self.hdr[name]["data_offsets"]
        b = arr.tobytes()
        assert len(b) == o1 - o0, (name, len(b), o1 - o0)
        self.raw[self.base + o0:self.base + o1] = b


base_st = St(os.path.join(a.base, "weights.int8.safetensors"))
arch = json.load(open(os.path.join(a.base, "arch.json")))

if a.check_pack:
    bad = 0
    for n in names:
        en = engine_name(n)
        d, sc, ws, _, _ = pack_int8(eng_weight(n).numpy())
        ok = (np.array_equal(d, base_st.get(en + ".q", np.int8)) and
              np.array_equal(sc, base_st.get(en + ".scale", np.float32)) and
              np.allclose(ws, base_st.get(en + ".wsum", np.float32), rtol=0, atol=1e-6))
        bad += not ok
        if not ok:
            print("MISMATCH", n, en)
    print(json.dumps({"check_pack": "ok" if bad == 0 else f"{bad} mismatches", "layers": len(names)}))
    sys.exit(0 if bad == 0 else 1)


def write_variant(key, patches):
    vd = os.path.join(a.work, key)
    shutil.rmtree(vd, ignore_errors=True)
    os.makedirs(vd)
    for f in ("weights.safetensors", "bpe.model", "tokens.txt", "mel_fbank.f32"):
        os.link(os.path.join(a.base, f), os.path.join(vd, f))
    st = St(os.path.join(a.base, "weights.int8.safetensors"))
    for en, (d, sc, ws) in patches.items():
        st.put(en + ".q", d)
        st.put(en + ".scale", sc)
        st.put(en + ".wsum", ws)
    wp = os.path.join(vd, "weights.int8.safetensors")
    with open(wp, "wb") as fh:
        fh.write(st.raw)
    sha = hashlib.sha256(st.raw).hexdigest()
    ar = json.loads(json.dumps(arch))
    ar["files"]["weights_int8"]["sha256"] = sha
    with open(os.path.join(vd, "arch.json"), "w") as fh:
        json.dump(ar, fh, indent=2)
    return vd, sha


utt_meta = {}
z = np.load(PANEL)
for i in range(len(z["uids"])):
    utt_meta[str(z["uids"][i])] = (str(z["speakers"][i]), str(z["splits"][i]), str(z["texts"][i]),
                                   round(float(z["offsets"][i + 1] - z["offsets"][i]) / 16000.0, 3))


def run_engine(vd, outdir):
    shutil.rmtree(outdir, ignore_errors=True)
    pk = 40 * (a.chunk or 4)  # one chunk per stream per round: every stream steps together
    cmd = [ENGINE, "--weights", vd, "--npz", PANEL, "--limit", "730", "--precision", "int8",
           "--streams", str(a.streams), "--packet-ms", str(pk), "--out", outdir]
    if a.chunk:
        cmd += ["--chunk", str(a.chunk)]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"engine failed: {r.stderr[-2000:]}")
    ev = json.load(open(os.path.join(outdir, "engine_eval.json")))
    return ev, time.time() - t0


def score_rows(ev):
    rows = []
    for u in ev["rows"]:
        spk, split, ref, dur = utt_meta[u["id"]]
        rows.append({"id": u["id"], "speaker": spk, "dataset": split, "ref": ref,
                     "hyp": u["hyp"], "duration_s": dur})
    return rows


def ultra_score(a_path, b_path, md):
    r = subprocess.run([SCORE, a_path, b_path, "--seed", "20261122", "--replicates", "10000",
                        "--alpha", "0.05", "--md-out", md], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-2000:])
    return json.loads(r.stdout)


def held_distortion(name, fit, wq_deq):
    path = os.path.join(a.trace, f"held.{L.short(name)}.npy")
    if not os.path.exists(path):
        return {}
    x = torch.from_numpy(np.load(path)).float()
    pm = perm_for(name)
    if pm is not None:
        x = x[:, pm]
    lin0 = L.get_mod(model, name)
    lin = torch.nn.Linear(lin0.in_features, lin0.out_features)
    lin.weight.data = eng_weight(name).clone()
    lin.bias.data = lin0.bias.data.clone()
    with torch.no_grad():
        y = x @ lin.weight.float().t() + lin.bias.float()
        var = (y - y.mean(0)).pow(2).sum()
        y8 = L.Int8Emu(lin)(x)
        pq = L.PQLinear(fit, lin.bias.data)
        ypq = pq(x)
        pq.lut_exact = False
        ynl = pq(x)
        yrq = L.fake_quant_rows_fwd(x) @ wq_deq.t() + lin.bias.float()
    sig = eng_sigma(name).double()
    W = lin.weight.double()
    den = torch.einsum("nk,kl,nl->", W, sig, W)
    E = W - fit.dequant().double()
    w_err = float(torch.einsum("nk,kl,nl->", E, sig, E) / den)
    Ef = W - fit.dequant_float().double()
    w_err_f = float(torch.einsum("nk,kl,nl->", Ef, sig, Ef) / den)
    Eq = W - wq_deq.double()
    w_err_rq = float(torch.einsum("nk,kl,nl->", Eq, sig, Eq) / den)
    return {"held_rows": int(x.shape[0]),
            "rel_werr_sigma_cal_floatcb": w_err_f,
            "rel_werr_sigma_cal_int8requant": w_err_rq,
            "rel_mse_pq_exact": float((ypq - y).pow(2).sum() / var),
            "rel_mse_pq_nolut": float((ynl - y).pow(2).sum() / var),
            "rel_mse_pq_int8requant": float((yrq - y).pow(2).sum() / var),
            "rel_mse_int8": float((y8 - y).pow(2).sum() / var),
            "rel_werr_sigma_cal": w_err,
            "lut_shift_hist": pq.stats["sh_hist"].tolist()}


FIT_KEYS = ("N", "K", "s", "R", "C", "kc", "codes", "cb_int", "cb_scale", "gain", "cb_float")


def get_fit(name):
    """Seeded, deterministic fit; cached per (format, weighting, layer) so group runs reuse it."""
    fdir = os.path.join(a.fit_cache, f"s{a.s}R{a.R}k{a.kc}{'u' if a.unweighted else 'w'}{'g' if a.gptq else ''}")
    os.makedirs(fdir, exist_ok=True)
    fp = os.path.join(fdir, L.short(name) + ".pt")
    if os.path.exists(fp):
        st = torch.load(fp, weights_only=False)
        fit = L.PQFit.__new__(L.PQFit)
        for k_, v_ in st.items():
            setattr(fit, k_, v_)
        fit.codes = fit.codes.long()
        return fit
    w_ = eng_weight(name)
    sig_ = torch.eye(w_.shape[1]) if a.unweighted else eng_sigma(name)
    if a.gptq:
        fit = L.PQFitGPTQ(w_, sig_, s=a.s, R=a.R, kc=a.kc)
    else:
        fit = L.PQFit(w_, sig_, s=a.s, R=a.R, kc=a.kc, weighted=not a.unweighted)
    st = {k_: getattr(fit, k_) for k_ in FIT_KEYS}
    st["codes"] = fit.codes.to(torch.uint8)
    torch.save(st, fp + ".tmp")
    os.replace(fp + ".tmp", fp)
    return fit


os.makedirs(os.path.join(a.out, a.tag, "hyps"), exist_ok=True)
os.makedirs(a.work, exist_ok=True)

if a.layers == "baseline":
    ev, dt = run_engine(a.base, os.path.join(a.work, "_base_run"))
    rows = score_rows(ev)
    with open(a.baseline, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(json.dumps({"baseline": a.baseline, "decode_s": round(dt, 1),
                      "summary": {k: v for k, v in ev.items() if k != "utterances"}}, default=str)[:3000])
    sys.exit(0)

if a.layers == "all":
    todo = [[n] for n in names]
elif a.layers.startswith("group:"):
    rx = re.compile(a.layers[6:])
    todo = [[n for n in names if rx.search(n)]]
else:
    todo = [[n] for n in a.layers.split(",")]
si, sn = map(int, a.shard.split("/"))
todo = todo[si::sn]
if a.reverse:
    todo = todo[::-1]
res_path = os.path.join(a.out, a.tag, f"rows.{si}of{sn}{a.rows_suffix}.jsonl")
import glob as _glob  # noqa: E402
done = set()
for _f in _glob.glob(os.path.join(a.out, a.tag, "rows.*.jsonl")):  # any sharding resumes
    done |= {json.loads(l)["key"] for l in open(_f) if l.strip()}

def refresh_done():
    d = set()
    for _f in _glob.glob(os.path.join(a.out, a.tag, "rows.*.jsonl")):
        d |= {json.loads(l)["key"] for l in open(_f) if l.strip()}
    return d


for group in todo:
    key = group[0] if len(group) == 1 else "group:" + a.layers[6:]
    if key in done or key in refresh_done():  # rows from other hosts may be synced in
        continue
    t0 = time.time()
    patches, info = {}, []
    for name in group:
        fit = get_fit(name)
        wd = fit.dequant().numpy()
        d, sc, ws, q, s = pack_int8(wd)
        patches[engine_name(name)] = (d, sc, ws)
        wq_deq = torch.from_numpy(q.astype(np.float32) * s[:, None])
        n, k = wd.shape
        info.append({"layer": name, "engine_name": engine_name(name), "n": n, "k": k,
                     "int8_bytes": n * k, "pq_bytes": fit.code_bytes(), "cb_scale": fit.cb_scale,
                     **({} if a.no_distortion else held_distortion(name, fit, wq_deq))})
    fit_s = time.time() - t0
    fkey = re.sub(r"[^A-Za-z0-9_.-]", "_", L.short(key))[:100]
    wkey = f"{a.tag}__{fkey}"  # tag in the work path: same-regex runs of other tags must not collide
    vd, sha = write_variant(wkey, patches)
    ev, dec_s = run_engine(vd, os.path.join(a.work, wkey + "_run"))
    rows = score_rows(ev)
    hp = os.path.join(a.out, a.tag, "hyps", f"{fkey}.score-input.jsonl")
    with open(hp, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    sc_json = ultra_score(a.baseline, hp, os.path.join(a.work, wkey + ".md"))
    row = {"key": key, "tag": a.tag, "s": a.s, "R": a.R, "kc": a.kc, "weighted": not a.unweighted, "gptq": a.gptq,
           "chunk": a.chunk or 4, "layers": info, "variant_int8_sha256": sha,
           "fit_s": round(fit_s, 1), "decode_s": round(dec_s, 1), "score": sc_json,
           "engine": {k: ev.get(k) for k in ("cpu_s_per_audio_s", "cpu_s", "audio_s", "wer_pct",
                                              "errors", "substitutions", "deletions", "insertions",
                                              "executable_sha256", "int8_prepared_sha256")},
           "time": L.now()}
    with open(res_path, "a") as fh:
        fh.write(json.dumps(row) + "\n")
    if not a.keep:
        shutil.rmtree(vd, ignore_errors=True)
        shutil.rmtree(os.path.join(a.work, wkey + "_run"), ignore_errors=True)
    print(json.dumps({"key": key, "fit_s": round(fit_s, 1), "decode_s": round(dec_s, 1),
                      "rel_mse_pq_exact": [i.get("rel_mse_pq_exact") for i in info][:2]}), flush=True)
