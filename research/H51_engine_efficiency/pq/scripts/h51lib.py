"""H51-PQ shared library: model loading, engine-faithful INT8 emulation, activation hooks,
activation-weighted gain-shape PQ fitting and an exact emulation of the `pqscan` integer contract
(crates/ultra-kernels/src/pqscan.rs, ULTRA-SU §6.2).

Nothing here changes repository code; the model classes are imported from an origin/main export.
"""
from __future__ import annotations

import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

SRC = os.environ.get("H51_SRC", "/mnt/xbabe3-external/jsr/h51-pq/src/python/ultra")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

QMAX = 127.0


# ----------------------------------------------------------------------------- model / data
def load_model(ckpt: str):
    from train import checkpoint
    from train.asr import ASRModel, ModelConfig

    p = checkpoint.load(ckpt, restore_rng=False)
    cfg = ModelConfig.from_dict(p["model_config"])
    model = ASRModel(cfg)
    model.load_state_dict(p["model"])
    model.eval()
    return model, p


def tokenizers(p):
    from train.asr import ModelConfig
    from train.tokenizer import TOKDIR, Tokenizer, tokenizer_path

    mcfg = ModelConfig.from_dict(p["model_config"])
    # The checkpoint's tokenizer (zipformer bpe.model, sha256 c53433de...) is not on H0; the INT8
    # export carries the same file. Verified against the checkpoint's tokenizer_sha256.
    import hashlib

    path = os.environ.get("H51_BPE", "/dev/shm/h51-pq/bpe.model")
    got = hashlib.sha256(open(path, "rb").read()).hexdigest()
    if got != p.get("tokenizer_sha256"):
        raise ValueError(f"tokenizer sha256 {got} != checkpoint {p.get('tokenizer_sha256')}")
    tok = Tokenizer(path)
    return tok, tok


def load_npz_utts(path: str):
    z = np.load(path)
    offs, pcm = z["offsets"], z["pcm"]
    uids, spk, txt, spl = z["uids"], z["speakers"], z["texts"], z["splits"]
    out = []
    for i in range(len(uids)):
        out.append(dict(uid=str(uids[i]), speaker=str(spk[i]), text=str(txt[i]),
                        split=str(spl[i]), pcm=pcm[offs[i]:offs[i + 1]]))
    return out


def featurize(p, utts):
    from train.data import Featurizer

    tok, tok_c = tokenizers(p)
    feat = Featurizer(tok, tok_c)
    exs = [feat.example(u["uid"], u["text"], u["pcm"]) for u in utts]
    return exs, tok, tok_c


# ----------------------------------------------------------------------------- target layers
def target_names(model) -> list[str]:
    """The engine's int8 GEMMs (weights.int8.safetensors kinds), in execution order."""
    names = ["encoder.stem.out"]
    for i in range(len(model.encoder.blocks)):
        b = f"encoder.blocks.{i}."
        names += [b + "ff1.w1", b + "ff1.w2", b + "att.qkv", b + "att.out",
                  b + "conv.pw1", b + "conv.pw2", b + "ff2.w1", b + "ff2.w2"]
    names += ["joiner.enc_proj", "joiner.out"]
    return names


def short(name: str) -> str:
    return (name.replace("encoder.blocks.", "b").replace("encoder.", "").replace(".", "_"))


def fake_quant_rows_fwd(x: torch.Tensor) -> torch.Tensor:
    from train.qat import fake_quant_rows

    return fake_quant_rows(x)


def int8_weight(w: torch.Tensor) -> torch.Tensor:
    """Engine rule: s = max|w| / 127, q = round_half_even(w * (127 / max)), w' = q * s."""
    wf = w.float()
    mx = wf.abs().amax(dim=1, keepdim=True)
    inv = torch.where(mx > 0, QMAX / mx, torch.zeros_like(mx))
    q = torch.clamp(torch.round(wf * inv), -QMAX, QMAX)
    return q * (mx / QMAX)


class Int8Emu(nn.Module):
    """The engine's int8 linear: per-row centred int8 inputs, per-channel int8 weights."""

    def __init__(self, lin: nn.Linear):
        super().__init__()
        self.weight_q = nn.Parameter(int8_weight(lin.weight.data), requires_grad=False)
        self.bias = nn.Parameter(lin.bias.data.float().clone(), requires_grad=False)
        self.in_features, self.out_features = lin.in_features, lin.out_features

    def forward(self, x):
        return F.linear(fake_quant_rows_fwd(x), self.weight_q, self.bias)


def get_mod(model, name):
    m = model
    for part in name.split("."):
        m = getattr(m, part)
    return m


def set_mod(model, name, new):
    parent = get_mod(model, name.rsplit(".", 1)[0])
    setattr(parent, name.rsplit(".", 1)[1], new)


def apply_int8(model, names=None, stem_c2=True):
    """All engine int8 GEMMs emulated (as deployed); fp32 originals kept in `model._fp32`."""
    names = names or target_names(model)
    model._fp32 = {n: get_mod(model, n) for n in names}
    for n in names:
        set_mod(model, n, Int8Emu(model._fp32[n]))
    if stem_c2:
        model.encoder.stem.int8_qat = True
    return names


# ----------------------------------------------------------------------------- decoding / WER
def decode_panel(model, exs, tok, batch=16, chunk=None, threads=None, max_symbols=8):
    from train.data import collate
    from train.transducer import greedy_decode

    hyps = {}
    order = sorted(range(len(exs)), key=lambda i: exs[i].feats.shape[0])
    with torch.no_grad():
        for i in range(0, len(order), batch):
            idx = order[i:i + batch]
            b = collate([exs[j] for j in idx])
            enc, lens = model.encode(b.feats, b.feat_lens, chunk)
            enc_p = model.joiner.enc_proj(enc.float())
            toks, _ = greedy_decode(model.predictor, model.joiner, enc_p, lens, max_symbols)
            for j, t in zip(idx, toks):
                hyps[j] = tok.decode(t)
    return [hyps[j] for j in range(len(exs))]


def wer_rows(utts, hyps):
    """score-input rows for ultra-score plus a provisional pooled WER (ultra.wer)."""
    from ultra.wer import corpus_wer

    rows = [{"id": u["uid"], "speaker": u["speaker"], "dataset": u["split"], "ref": u["text"],
             "hyp": h, "duration_s": round(len(u["pcm"]) / 16000.0, 3)} for u, h in zip(utts, hyps)]
    res = {}
    for ds in ("dev-clean", "dev-other"):
        sel = [(r["ref"], r["hyp"]) for r in rows if r["dataset"] == ds]
        res[ds] = corpus_wer([a for a, _ in sel], [b for _, b in sel])
    allw = corpus_wer([r["ref"] for r in rows], [r["hyp"] for r in rows])
    res["pooled"] = allw
    return rows, res


# ----------------------------------------------------------------------------- rotations
def fwht16(x: torch.Tensor) -> torch.Tensor:
    """Block-16 orthonormal Walsh-Hadamard (H_16 / 4) on the last axis (H24's FWHT16)."""
    k = x.shape[-1]
    assert k % 16 == 0
    h = torch.tensor([[1.0]])
    for _ in range(4):
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    h = (h / 4.0).to(x.dtype)
    return (x.reshape(*x.shape[:-1], k // 16, 16) @ h).reshape(x.shape)


def random_orthogonal(k: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(k, k, generator=g, dtype=torch.float64)
    q, r = torch.linalg.qr(a)
    q = q * torch.sign(torch.diagonal(r))[None]
    return q.float()


# ----------------------------------------------------------------------------- quantisers
def quant_rows_centred(x: torch.Tensor):
    """Engine per-row centred int8: (codes as float, scale, centre); constant rows -> 0 codes."""
    xf = x.float()
    hi = xf.amax(dim=-1, keepdim=True)
    lo = xf.amin(dim=-1, keepdim=True)
    half = 0.5 * hi - 0.5 * lo
    c = 0.5 * hi + 0.5 * lo
    inv = QMAX / half
    ok = (half > 0) & torch.isfinite(inv)
    q = torch.clamp(torch.round((xf - c) * torch.where(ok, inv, torch.zeros_like(inv))), -QMAX, QMAX)
    q = torch.where(ok, q, torch.zeros_like(q))
    return q, torch.where(ok, half / QMAX, torch.zeros_like(half)), c


def quant_rows_symmetric(x: torch.Tensor):
    """Zero-anchored per-row symmetric int8 (absmax): exact 0 -> code 0."""
    xf = x.float()
    mx = xf.abs().amax(dim=-1, keepdim=True)
    inv = torch.where(mx > 0, QMAX / mx, torch.zeros_like(mx))
    q = torch.clamp(torch.round(xf * inv), -QMAX, QMAX)
    return q, mx / QMAX


# ----------------------------------------------------------------------------- PQ fitting
def _kmeans_batched(v, wts, k=16, iters=25, seed=0, init=None):
    """Weighted k-means for C independent subspaces at once.
    v (C, N, s) float64 data, wts (N,) weights; returns centroids (C, k, s), assign (C, N)."""
    C, N, s = v.shape
    g = torch.Generator().manual_seed(seed)
    if init is None:
        # k-means++ (weighted), batched
        cent = torch.empty(C, k, s, dtype=v.dtype)
        first = torch.multinomial(wts.expand(C, N).float(), 1, generator=g).squeeze(1)
        cent[:, 0] = v[torch.arange(C), first]
        d2 = ((v - cent[:, :1]) ** 2).sum(-1)
        for j in range(1, k):
            pr = (d2 * wts[None]).float().clamp(min=0) + 1e-30
            nxt = torch.multinomial(pr, 1, generator=g).squeeze(1)
            cent[:, j] = v[torch.arange(C), nxt]
            d2 = torch.minimum(d2, ((v - cent[:, j:j + 1]) ** 2).sum(-1))
    else:
        cent = init.clone()
    for _ in range(iters):
        d = torch.cdist(v, cent) ** 2  # (C, N, k)
        a = d.argmin(-1)
        oh = F.one_hot(a, k).to(v.dtype) * wts[None, :, None]  # (C, N, k)
        num = oh.transpose(1, 2) @ v  # (C, k, s)
        den = oh.sum(1)[..., None]
        newc = torch.where(den > 0, num / den.clamp(min=1e-30), cent)
        # re-seed empty clusters at the worst-served points
        empty = (den.squeeze(-1) <= 0)
        if empty.any():
            dmin = d.gather(-1, a[..., None]).squeeze(-1) * wts[None]
            for ci, j in empty.nonzero().tolist():
                far = int(dmin[ci].argmax())
                newc[ci, j] = v[ci, far]
                dmin[ci, far] = 0
        cent = newc
    d = torch.cdist(v, cent) ** 2
    return cent, d.argmin(-1)


class PQFit:
    """Gain-shape additive PQ of W (N, K): W[n] ~ g[n] * sum_r cb[r, c, code[r, n, c]] per
    subspace c of width s; codebooks shared by all rows; activation-weighted distortion
    (x^T e)^2 with the input second moment Sigma; int8 codebooks with one global scale."""

    def __init__(self, W, Sigma, s=4, R=2, iters=25, outer=3, seed=0, weighted=True, kc=16):
        W = W.double()
        N, K = W.shape
        assert K % (2 * s) == 0, (K, s)
        assert kc in (4, 16)
        self.kc = kc
        self.N, self.K, self.s, self.R, self.C = N, K, s, R, K // s
        C = self.C
        Sig = Sigma.double() if weighted else torch.eye(K, dtype=torch.float64)
        # per-subspace Cholesky of the s x s diagonal blocks (ridge for safety)
        blocks = torch.stack([Sig[c * s:(c + 1) * s, c * s:(c + 1) * s] for c in range(C)])
        tr = torch.diagonal(blocks, dim1=1, dim2=2).mean(-1).clamp(min=1e-12)
        blocks = blocks + 1e-6 * tr[:, None, None] * torch.eye(s, dtype=torch.float64)
        L = torch.linalg.cholesky(blocks)  # (C, s, s), Sig_c = L L^T
        self.L = L
        g = W.pow(2).mean(1).sqrt().clamp(min=1e-12)  # row RMS init
        cents = [None] * R
        for it in range(outer):
            U = (W / g[:, None]).reshape(N, C, s).permute(1, 0, 2)  # (C, N, s)
            V = (L.transpose(1, 2)[:, None] @ U[..., None]).squeeze(-1)  # v = L^T u
            wts = g.pow(2)
            res = V.clone()
            codes, cbv = [], []
            for r in range(R):
                c_r, a_r = _kmeans_batched(res, wts, k=kc, iters=iters,
                                           seed=seed + 101 * r + 7 * it, init=cents[r])
                cents[r] = c_r
                codes.append(a_r)
                cbv.append(c_r)
                res = res - c_r.gather(1, a_r[..., None].expand(-1, -1, s))
            # one refinement round of the additive stages (re-fit each stage on its residual)
            if R > 1:
                for r in range(R):
                    other = sum(cbv[q].gather(1, codes[q][..., None].expand(-1, -1, s))
                                for q in range(R) if q != r)
                    c_r, a_r = _kmeans_batched(V - other, wts, k=kc, iters=max(5, iters // 3),
                                               seed=seed, init=cbv[r])
                    cbv[r], codes[r], cents[r] = c_r, a_r, c_r
            # back to weight space: cb = L^{-T} v
            Linv_T = torch.linalg.inv(L.transpose(1, 2))
            cbw = [(Linv_T[:, None] @ cb[..., None]).squeeze(-1) for cb in cbv]  # (C, 16, s)
            recon = sum(cbw[r].gather(1, codes[r][..., None].expand(-1, -1, s)) for r in range(R))
            rn = recon.permute(1, 0, 2).reshape(N, K)  # unit-gain reconstruction per row
            # closed-form gain under the full Sigma metric
            SR = rn @ Sig
            g_new = (SR * W).sum(1) / (SR * rn).sum(1).clamp(min=1e-30)
            g = torch.where(g_new.abs() > 1e-12, g_new, g)
        self.codes = torch.stack(codes)  # (R, C, N)
        cbw_all = torch.stack(cbw)  # (R, C, 16, s)
        # int8 codebooks, one global scale (the kernel sums all stages / subspaces directly)
        self.cb_float = cbw_all.clone()
        mx = cbw_all.abs().max()
        self.cb_scale = float(mx / QMAX)
        self.cb_int = torch.clamp(torch.round(cbw_all / self.cb_scale), -QMAX, QMAX)
        self.gain = g.float()

    def dequant(self) -> torch.Tensor:
        """Deployed weights W' (N, K) = gain * cb_scale * sum_r cb_int[r, c, code]."""
        R, C, N = self.codes.shape
        s = self.s
        rec = sum(self.cb_int[r].double().gather(1, self.codes[r][..., None].expand(-1, -1, s))
                  for r in range(R))  # (C, N, s)
        return (rec.permute(1, 0, 2).reshape(N, C * s) * (self.gain.double()[:, None] * self.cb_scale)).float()

    def dequant_float(self) -> torch.Tensor:
        """Same codes and gains with float (not int8) codebooks: isolates the int8-codebook loss."""
        R, C, N = self.codes.shape
        s = self.s
        rec = sum(self.cb_float[r].gather(1, self.codes[r][..., None].expand(-1, -1, s))
                  for r in range(R))
        return (rec.permute(1, 0, 2).reshape(N, C * s) * self.gain.double()[:, None]).float()

    def code_bytes(self) -> dict:
        bits = 4 if self.kc == 16 else 2
        codes = self.N * self.C * self.R * bits / 8
        cbook = self.R * self.C * self.kc * self.s  # int8 codewords
        side = 4 * self.N * 3  # gain, wsum, bias (f32)
        return {"codes": int(codes), "codebooks": int(cbook), "side": int(side),
                "total": int(codes + cbook + side), "bits_per_weight": bits * self.R / self.s,
                "kc": self.kc}


class PQLinear(nn.Module):
    """Exact emulation of the pqscan integer contract + an INT8-style dequant epilogue:
    a = centred int8(x); v[r,c,j] = a_c . cb[r,c,j]; per-row shift sh = min{sh: max range >> sh <= 127};
    e = (v - min_c) >> sh; S[n] = sum_{r,c} e[r,c,code]; acc = (S << sh) + sum min + round_bias;
    y = acc * (sa * gain * cb_scale) + ca * wsum + b, wsum[n] = sum_k W'[n,k]."""

    def __init__(self, fit: PQFit, bias: torch.Tensor, chunk_rows: int = 2048, lut_exact=True):
        super().__init__()
        self.s, self.R, self.C, self.N = fit.s, fit.R, fit.C, fit.N
        self.kc = fit.kc
        self.register_buffer("cb", fit.cb_int.float())  # (R, C, kc, s)
        self.register_buffer("gw", (fit.gain * fit.cb_scale).float())  # (N,)
        wd = fit.dequant()
        self.register_buffer("wsum", wd.double().sum(1).float())
        self.register_buffer("bias", bias.float().clone())
        self.register_buffer("wdeq", wd)  # for the no-LUT-rounding ablation
        # one-hot scan matrix (R*C*16, N)
        R, C, N = fit.codes.shape
        if self.kc == 16:  # H24 pqscan: one 16-entry LUT per subspace
            T_, cc = C, fit.codes
        else:  # swarm #6 staged format: subspace pairs share one 4x4 = 16-entry LUT
            T_ = C // 2
            cc = fit.codes[:, 0::2] + 4 * fit.codes[:, 1::2]
        oh = torch.zeros(R * T_ * 16, N)
        idx = (torch.arange(R)[:, None, None] * T_ + torch.arange(T_)[None, :, None]) * 16 + cc
        oh.scatter_(0, idx.reshape(R * T_, N), 1.0)
        self.register_buffer("oh", oh)
        self.ntab = T_
        self.chunk_rows = chunk_rows
        self.lut_exact = lut_exact
        self.in_features, self.out_features = fit.K, fit.N
        self.stats = {"rows": 0, "sh_hist": np.zeros(32, dtype=np.int64)}

    def acc_int(self, a: torch.Tensor) -> torch.Tensor:
        """Integer pqscan accumulators (T, N) for int8 activation codes a (T, K): the kernel's
        i32 output before the dequant epilogue (for the bit-exactness check against Rust)."""
        T = a.shape[0]
        v = torch.einsum("tci,rcji->trcj", a.float().view(T, self.C, self.s), self.cb)
        if self.kc == 4:
            v = (v[:, :, 0::2, None, :] + v[:, :, 1::2, :, None]).reshape(T, self.R, self.C // 2, 16)
        mn = v.amin(-1)
        ri = (v.amax(-1) - mn).amax(-1).amax(-1).to(torch.int64)
        sh = torch.zeros_like(ri)
        while bool(((ri >> sh) > 127).any()):
            sh = sh + ((ri >> sh) > 127).to(torch.int64)
        e = ((v.to(torch.int64) - mn.to(torch.int64)[..., None]) >> sh[:, None, None, None]).float()
        S = (e.reshape(T, -1) @ self.oh).to(torch.int64)
        nact = self.R * self.ntab
        off = mn.to(torch.int64).sum((1, 2)) + ((nact * ((1 << sh) - 1)) >> 1)
        return S * (1 << sh)[:, None] + off[:, None]

    def forward(self, x):
        shp = x.shape
        xf = x.reshape(-1, shp[-1]).float()
        out = torch.empty(xf.shape[0], self.N)
        for t0 in range(0, xf.shape[0], self.chunk_rows):
            xt = xf[t0:t0 + self.chunk_rows]
            if getattr(self, "act", "centred") == "sym":  # zero-anchored codes, no centre term
                a, sa = quant_rows_symmetric(xt)
                ca = torch.zeros_like(sa)
            else:
                a, sa, ca = quant_rows_centred(xt)
            if not self.lut_exact:
                out[t0:t0 + len(xt)] = (a @ self.wdeq.t()) * sa + ca * self.wsum + self.bias
                continue
            T = a.shape[0]
            v = torch.einsum("tci,rcji->trcj", a.view(T, self.C, self.s), self.cb)  # exact ints
            if self.kc == 4:  # pair LUT: v[j0 | j1 << 2] = v_even[j0] + v_odd[j1]
                v = (v[:, :, 0::2, None, :] + v[:, :, 1::2, :, None]).reshape(T, self.R, self.C // 2, 16)
            mn = v.amin(-1)  # (T, R, C)
            rng = (v.amax(-1) - mn).amax(-1).amax(-1)  # (T,)
            ri = rng.to(torch.int64)
            emax = (1 << getattr(self, "lut_bits", 7)) - 1  # H24 contract: 7-bit entries (127)
            sh = torch.zeros_like(ri)
            while True:
                m = (ri >> sh) > emax
                if not bool(m.any()):
                    break
                sh = sh + m.to(torch.int64)
            d = v.to(torch.int64) - mn.to(torch.int64)[..., None]
            nact = self.R * self.ntab
            if getattr(self, "lut_round", "trunc_bias") == "nearest":  # variant: no global bias
                half = torch.where(sh > 0, 1 << (sh - 1).clamp(min=0), torch.zeros_like(sh))
                e = torch.clamp((d + half[:, None, None, None]) >> sh[:, None, None, None], max=emax).float()
                off = mn.to(torch.int64).sum((1, 2))
            else:  # H24 contract: truncate, add the mean truncation back once per frame
                e = (d >> sh[:, None, None, None]).float()
                off = mn.to(torch.int64).sum((1, 2)) + ((nact * ((1 << sh) - 1)) >> 1)
            S = e.reshape(T, -1) @ self.oh  # exact: entries <= 255, sums < 2^24
            acc = S.to(torch.int64) * (1 << sh)[:, None] + off[:, None]
            out[t0:t0 + T] = acc.float() * (sa * self.gw[None]) + ca * self.wsum[None] + self.bias[None]
            self.stats["rows"] += T
            self.stats["sh_hist"] += np.bincount(sh.numpy(), minlength=32)[:32]
        return out.reshape(*shp[:-1], self.N)


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class PQFitGPTQ(PQFit):
    """PQ with GPTQ/GPTVQ-style error feedback (same deployed format as PQFit).

    Gains are fixed first (row RMS, then the closed-form refit at the end); subspaces are
    quantised left to right on the gain-normalised, error-updated weights. Block c uses the
    metric M_c = inv((H^-1)_cc) (the Schur complement: the error the later, still free columns
    cannot absorb) for its weighted k-means, then its error is pushed into the remaining columns
    with E_c inv((H^-1)_cc) (H^-1)_{c,rest}. H = Sigma + damp * mean(diag) * I.
    """

    def __init__(self, W, Sigma, s=4, R=2, iters=25, seed=0, kc=16, damp=0.01):
        W = W.double()
        N, K = W.shape
        assert K % (2 * s) == 0
        self.N, self.K, self.s, self.R, self.C, self.kc = N, K, s, R, K // s, kc
        C = self.C
        H = Sigma.double().clone()
        H += damp * torch.diagonal(H).mean() * torch.eye(K, dtype=torch.float64)
        Hinv = torch.linalg.inv(H)
        g = W.pow(2).mean(1).sqrt().clamp(min=1e-12)
        U = (W / g[:, None]).clone()  # (N, K) normalised rows, updated in place
        wts = g.pow(2)
        codes = torch.zeros(R, C, N, dtype=torch.long)
        cbw = torch.zeros(R, C, kc, s, dtype=torch.float64)
        for c in range(C):
            sl = slice(c * s, (c + 1) * s)
            Hcc = Hinv[sl, sl]
            M = torch.linalg.inv(Hcc)  # s x s
            M = 0.5 * (M + M.t())
            Lm = torch.linalg.cholesky(M + 1e-12 * torch.eye(s, dtype=torch.float64))
            V = U[:, sl] @ Lm  # rows v = u L  so ||v1 - v2||^2 = (u1-u2) M (u1-u2)^T
            res = V.clone()[None]  # (1, N, s)
            q = torch.zeros(N, s, dtype=torch.float64)
            for r in range(R):
                cent, a_ = _kmeans_batched(res, wts, k=kc, iters=iters, seed=seed + 31 * c + r)
                codes[r, c] = a_[0]
                sel = cent[0][a_[0]]  # (N, s) in v-space
                res = res - sel[None]
                cbw[r, c] = cent[0] @ torch.linalg.inv(Lm)  # back to u-space
                q = q + cbw[r, c][a_[0]]
            E = U[:, sl] - q  # (N, s)
            U[:, sl] = q
            if c + 1 < C:
                rest = slice((c + 1) * s, K)
                B = Hinv[sl, rest]  # (s, r) of the inverse conditioned on blocks < c fixed
                U[:, rest] -= E @ (M @ B)
                # condition the inverse on block c being fixed (Schur complement update)
                Hinv[rest, rest] -= B.t() @ (M @ B)
        rn = U  # unit-gain reconstruction (quantised blocks)
        Sig = Sigma.double()
        SR = rn @ Sig
        g_new = (SR * W).sum(1) / (SR * rn).sum(1).clamp(min=1e-30)
        self.gain = torch.where(g_new.abs() > 1e-12, g_new, g).float()
        self.codes = codes
        self.cb_float = cbw
        mx = cbw.abs().max()
        self.cb_scale = float(mx / QMAX)
        self.cb_int = torch.clamp(torch.round(cbw / self.cb_scale), -QMAX, QMAX)
