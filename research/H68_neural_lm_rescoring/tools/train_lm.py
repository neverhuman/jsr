"""H68: packed causal Transformer LM on the shipped BPE-500 token stream (L2R or R2L).

Stream: uint16, token = sp_id + 1, 0 = sentence boundary (acts as BOS and EOS). Random windows of
`ctx` tokens are drawn from the book-ordered corpus, so the model also learns cross-sentence context
(usable later for session-mode rescoring). Keeps only the latest checkpoint (overwritten atomically).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.w, self.eps = nn.Parameter(torch.ones(d)), eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype) * self.w


def rope_tables(ctx: int, hd: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    inv = 1.0 / theta ** (torch.arange(0, hd // 2, dtype=torch.float32) * 2 / hd)
    ang = torch.arange(ctx, dtype=torch.float32)[:, None] * inv[None]
    return ang.cos(), ang.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # interleaved pairs (2j, 2j+1), identical to crates/ultra-engine/src/lm/tf_rescore.rs::apply_rope
    a, b = x[..., 0::2], x[..., 1::2]
    c, s = cos[: x.shape[-2]].to(x.dtype), sin[: x.shape[-2]].to(x.dtype)
    return torch.stack((a * c - b * s, a * s + b * c), dim=-1).flatten(-2)


class Block(nn.Module):
    """Engine-compatible block (ULTRTF01): RMSNorm, qkv (no bias), interleaved RoPE, wo, RMSNorm, SwiGLU w12/w3."""

    def __init__(self, d: int, h: int, ff: int, drop: float) -> None:
        super().__init__()
        self.h, self.drop, self.ff = h, drop, ff
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
        self.qkv, self.wo = nn.Linear(d, 3 * d, bias=False), nn.Linear(d, d, bias=False)
        self.w12, self.w3 = nn.Linear(d, 2 * ff, bias=False), nn.Linear(ff, d, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        q, k, v = (t.view(B, T, self.h, D // self.h).transpose(1, 2) for t in self.qkv(self.n1(x)).chunk(3, -1))
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.drop if self.training else 0.0)
        x = x + F.dropout(self.wo(a.transpose(1, 2).reshape(B, T, D)), self.drop, self.training)
        g, u = self.w12(self.n2(x)).split(self.ff, -1)
        return x + F.dropout(self.w3(F.silu(g) * u), self.drop, self.training)


class LM(nn.Module):
    def __init__(self, vocab: int = 501, d: int = 512, layers: int = 8, heads: int = 8, ff: int = 1536, ctx: int = 256, drop: float = 0.0, theta: float = 10000.0) -> None:
        super().__init__()
        self.cfg = dict(vocab=vocab, d=d, layers=layers, heads=heads, ff=ff, ctx=ctx, drop=drop, theta=theta)
        self.emb = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList([Block(d, heads, ff, drop) for _ in range(layers)])
        self.norm = RMSNorm(d)
        cos, sin = rope_tables(ctx, d // heads, theta)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        nn.init.normal_(self.emb.weight, std=0.02)
        for b in self.blocks:
            for lin in (b.qkv, b.w12):
                nn.init.normal_(lin.weight, std=0.02)
            for lin in (b.wo, b.w3):
                nn.init.normal_(lin.weight, std=0.02 / math.sqrt(2 * layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.emb(x)
        for b in self.blocks:
            h = b(h, self.cos, self.sin)
        return F.linear(self.norm(h), self.emb.weight)  # tied head


def load_lm(path: str, device: str = "cpu") -> LM:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = LM(**ck["cfg"])
    m.load_state_dict(ck["model"])
    m.reverse = bool(ck.get("reverse", False))
    return m.to(device).eval()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/mnt/xbabe3-external/jsr/wer-push/H68/data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--reverse", action="store_true")
    ap.add_argument("--d", type=int, default=512)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--ff", type=int, default=1536)
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--drop", type=float, default=0.0)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    tr = np.fromfile(f"{a.data}/train.u16", dtype=np.uint16)
    va = np.fromfile(f"{a.data}/valid.u16", dtype=np.uint16)
    if a.reverse:
        tr, va = tr[::-1].copy(), va[::-1].copy()
    dev = torch.device("cuda")
    trt = torch.from_numpy(tr.view(np.int16))  # values < 501: reinterpret, no extra host copy
    model = LM(501, a.d, a.layers, a.heads, a.ff, a.ctx, a.drop).to(dev)
    nparam = sum(p.numel() for p in model.parameters())
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    nodecay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": nodecay, "weight_decay": 0.0}], lr=a.lr, betas=(0.9, 0.98), fused=True)
    cmodel = model  # no triton in the shared venv
    T = a.ctx + 1

    def batch() -> torch.Tensor:
        idx = torch.from_numpy(rng.integers(0, len(trt) - T, a.batch))
        w = trt[idx[:, None] + torch.arange(T)[None]]
        return w.to(dev, non_blocking=True).long()

    @torch.no_grad()
    def evaluate() -> float:
        model.eval()
        v = torch.from_numpy(va.astype(np.int64))
        n = (len(v) - 1) // a.ctx
        tot, cnt = 0.0, 0
        for s in range(0, min(n, 2048), 64):
            rows = torch.stack([v[i * a.ctx:(i + 1) * a.ctx + 1] for i in range(s, min(s + 64, n))]).to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lo = model(rows[:, :-1]).float()
            tot += F.cross_entropy(lo.reshape(-1, 501), rows[:, 1:].reshape(-1), reduction="sum").item()
            cnt += rows[:, 1:].numel()
        model.train()
        return tot / cnt

    log = open(f"{a.out}/train.jsonl", "a")
    t0, best = time.time(), 9e9
    start = 1
    if a.resume and os.path.exists(f"{a.out}/latest.pt"):
        ck = torch.load(f"{a.out}/latest.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        start = int(ck["step"]) + 1
        print(json.dumps({"resumed_from_step": ck["step"], "note": "optimizer state not saved; 500-step re-warmup"}), flush=True)
    model.train()
    for step in range(start, a.steps + 1):
        rw = min(1.0, (step - start + 1) / 500) if start > 1 else 1.0
        lr = rw * a.lr * min(1.0, step / a.warmup) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1.0, step / a.steps))))
        for g in opt.param_groups:
            g["lr"] = lr
        w = batch()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = F.cross_entropy(cmodel(w[:, :-1]).float().reshape(-1, 501), w[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if not math.isfinite(loss.item()):
            raise SystemExit(f"non-finite loss at {step}")
        if step % 200 == 0:
            rec = {"step": step, "loss": round(loss.item(), 4), "lr": lr, "tok_per_s": round((step - start + 1) * a.batch * a.ctx / (time.time() - t0))}
            if step % a.save_every == 0 or step == a.steps:
                vn = evaluate()
                rec["valid_nll"] = round(vn, 4)
                ck = {"cfg": model.cfg, "model": model.state_dict(), "reverse": a.reverse, "step": step, "valid_nll": vn, "params": nparam, "args": vars(a)}
                torch.save(ck, f"{a.out}/latest.pt.tmp")
                os.replace(f"{a.out}/latest.pt.tmp", f"{a.out}/latest.pt")
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
    print(json.dumps({"done": True, "params": nparam, "elapsed_s": round(time.time() - t0)}), flush=True)


if __name__ == "__main__":
    main()
