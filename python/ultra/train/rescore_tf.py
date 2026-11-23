"""Full-sentence causal L2R & R2L Transformer N-best rescorers (H55 Wave 1B-2).

Torch-free at module import: ``torch`` is imported lazily inside model/training
helpers so light CI lanes can import planning/scoring utilities without PyTorch.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
import warnings

import numpy as np

from train.vram_pack import apply_gpu_mem_fraction

VOCAB = 501  # 0 = BOS/EOS sentence boundary, 1..500 = BPE-500 token id + 1
MAX_RESCORER_VRAM_FRAC = 0.42  # <= 10.5 GB on 24 GB 3090 (H0 co-tenancy guard)
DEFAULT_GPU_DUTY_CAP = 0.75  # keeps 5-min GPU mean strictly <= 75% (below 90% host rule)
PRESETS = {
    "3.5m": {"d_model": 256, "n_layers": 4, "n_heads": 4, "d_ff": 1024, "ctx": 128},
    "10m": {"d_model": 384, "n_layers": 6, "n_heads": 6, "d_ff": 1024, "ctx": 128},
}


def compute_duty_sleep_s(step_s: float, duty_cap: float) -> float:
    """Per-step sleep duration to keep GPU duty <= ``duty_cap`` in ``(0, 1]``."""
    if not math.isfinite(duty_cap) or duty_cap <= 0.0 or duty_cap > 1.0:
        raise ValueError(f"duty_cap must be in (0.0, 1.0], got {duty_cap!r}")
    return 0.0 if (step_s <= 0.0 or duty_cap >= 1.0) else step_s * (1.0 / duty_cap - 1.0)


def should_yield_to_cotenant(my_pid: int | None = None, gpu_index: int = 0, *,
                             util_thr: int = 20, min_free_mb: int = 10548) -> bool:
    """Yield ONLY if another GPU PID is active AND (util > 20% OR free VRAM < 10.3 GiB)."""
    me = os.getpid() if my_pid is None else int(my_pid)
    try:
        r1 = subprocess.run(["nvidia-smi", "-i", str(gpu_index), "--query-compute-apps=pid", "--format=csv,noheader"],
                            capture_output=True, text=True, timeout=5, check=False)
        if r1.returncode != 0 or not any(t.strip().isdigit() and int(t.strip()) != me for t in r1.stdout.splitlines()):
            return False
        r2 = subprocess.run(["nvidia-smi", "-i", str(gpu_index), "--query-gpu=utilization.gpu,memory.free",
                             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5, check=False)
    except (OSError, subprocess.SubprocessError):
        return False
    parts = [p.strip() for p in r2.stdout.splitlines()[0].split(",")] if (r2.returncode == 0 and r2.stdout.strip()) else []
    return len(parts) >= 2 and (int(float(parts[0])) > util_thr or int(float(parts[1])) < min_free_mb)


def build_sentence_positions_and_mask(x_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-sentence RoPE positions, same-sentence causal mask, and valid-loss mask."""
    is_zero, sid = x_ids == 0, np.cumsum(x_ids == 0, axis=1, dtype=np.int32)
    B, T = x_ids.shape
    pos = np.zeros((B, T), dtype=np.int64)
    for t in range(1, T):
        pos[:, t] = np.where(is_zero[:, t], 0, pos[:, t - 1] + 1)
    tril = np.tril(np.ones((T, T), dtype=bool))[None, :, :]
    return pos, (sid[:, :, None] == sid[:, None, :]) & tril, sid > 0


def make_model(arch: str = "3.5m", *, direction: str = "l2r"):
    """Pre-Norm RMSNorm + RoPE + SwiGLU causal Transformer LM with tied embeddings."""
    import torch
    from torch import nn
    import torch.nn.functional as F

    if arch not in PRESETS or direction not in ("l2r", "r2l"):
        raise ValueError(f"invalid arch={arch!r} or direction={direction!r}")
    cfg = PRESETS[arch]

    class RMSNorm(nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(dim))

        def forward(self, x):
            return (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.weight

    class Block(nn.Module):
        def __init__(self, d: int, nh: int, d_ff: int):
            super().__init__()
            self.nh, self.hd = nh, d // nh
            self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
            self.qkv, self.wo = nn.Linear(d, 3 * d, bias=False), nn.Linear(d, d, bias=False)
            self.w12, self.w3 = nn.Linear(d, 2 * d_ff, bias=False), nn.Linear(d_ff, d, bias=False)

        def forward(self, x, cos, sin, mask):
            B, T, C = x.shape
            q, k, v = (t.view(B, T, self.nh, self.hd).transpose(1, 2)
                       for t in self.qkv(self.n1(x)).chunk(3, dim=-1))
            q = torch.stack((q[..., ::2] * cos - q[..., 1::2] * sin, q[..., ::2] * sin + q[..., 1::2] * cos), -1).flatten(-2)
            k = torch.stack((k[..., ::2] * cos - k[..., 1::2] * sin, k[..., ::2] * sin + k[..., 1::2] * cos), -1).flatten(-2)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
            x = x + self.wo(y.transpose(1, 2).contiguous().view(B, T, C))
            g, u = self.w12(self.n2(x)).chunk(2, dim=-1)
            return x + self.w3(F.silu(g) * u)

    class CausalTransformerLM(nn.Module):
        def __init__(self):
            super().__init__()
            d, nh, hd = cfg["d_model"], cfg["n_heads"], cfg["d_model"] // cfg["n_heads"]
            self.arch, self.direction, self.cfg = arch, direction, dict(cfg)
            self.emb = nn.Embedding(VOCAB, d)
            self.blocks = nn.ModuleList([Block(d, nh, cfg["d_ff"]) for _ in range(cfg["n_layers"])])
            self.norm = RMSNorm(d)
            self.register_buffer("inv_freq", 1.0 / (10000.0 ** (torch.arange(0, hd, 2).float() / hd)), persistent=False)
            nn.init.normal_(self.emb.weight, std=0.02)

        def forward(self, x, pos, mask=None):
            h = self.emb(x)
            ang = pos.float().unsqueeze(-1) * self.inv_freq
            cos, sin = ang.cos().unsqueeze(1).to(h.dtype), ang.sin().unsqueeze(1).to(h.dtype)
            mask = torch.ones((x.shape[1], x.shape[1]), dtype=torch.bool, device=x.device).tril() if mask is None else mask
            for blk in self.blocks:
                h = blk(h, cos, sin, mask)
            return F.linear(self.norm(h), self.emb.weight)

    return CausalTransformerLM()


def _prep_packed_batch(chunk_np: np.ndarray, direction: str, device):
    import torch

    seq = torch.from_numpy(chunk_np[:, ::-1].copy() if direction == "r2l" else chunk_np).to(device, non_blocking=True)
    x, y = seq[:, :-1], seq[:, 1:]
    T, is_zero = x.shape[1], x == 0
    sid, ar = is_zero.cumsum(dim=1), torch.arange(T, device=device)
    pos = ar - torch.where(is_zero, ar, torch.zeros_like(ar)).cummax(dim=1).values
    tril = torch.ones((T, T), dtype=torch.bool, device=device).tril()
    return x, y, pos, ((sid[:, :, None] == sid[:, None, :]) & tril).unsqueeze(1), sid > 0


def eval_stream_ppl(model, vstream: np.ndarray, device, *, batch_size: int = 128) -> dict:
    """Sentence-isolated held-out PPL: non-zero token PPL (comparable to FFNN-6) + full EOS PPL."""
    import torch
    import torch.nn.functional as F

    ctx, n_win = model.cfg["ctx"], max(1, (len(vstream) - 1) // model.cfg["ctx"])
    tok_nll = sent_nll = 0.0
    tok_cnt = sent_cnt = 0
    model.eval()
    with torch.no_grad():
        for i in range(0, n_win, batch_size):
            arr = np.stack([vstream[k * ctx : k * ctx + ctx + 1] for k in range(i, min(n_win, i + batch_size))]).astype(np.int64)
            x, y, pos, mask, valid = _prep_packed_batch(arr, model.direction, device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                nll = F.cross_entropy(model(x, pos, mask).float().reshape(-1, VOCAB), y.reshape(-1), reduction="none").view_as(y)
            tok_m = valid & (y > 0)
            tok_nll, tok_cnt = tok_nll + float(nll[tok_m].sum()), tok_cnt + int(tok_m.sum())
            sent_nll, sent_cnt = sent_nll + float(nll[valid].sum()), sent_cnt + int(valid.sum())
    model.train()
    return {"valid_tok_ppl": round(math.exp(tok_nll / max(1, tok_cnt)), 4),
            "valid_sent_ppl": round(math.exp(sent_nll / max(1, sent_cnt)), 4),
            "valid_toks": tok_cnt, "valid_sent_toks": sent_cnt}


def score_sequences(model, token_seqs: list[list[int]], device, *, batch_size: int = 128) -> list[dict]:
    """Score BPE-500 token sequences (1..500) returning tok_logp, eos_logp, total_logp, n_tok."""
    import torch
    import torch.nn.functional as F

    model.eval()
    ctx, out = model.cfg["ctx"], []
    with torch.no_grad():
        for i in range(0, len(token_seqs), batch_size):
            seqs = [[0] + (list(reversed(s)) if model.direction == "r2l" else list(s))[: ctx - 1] + [0]
                    for s in token_seqs[i : i + batch_size]]
            mlen, lens = max(len(s) for s in seqs), [len(s) - 1 for s in seqs]
            x = torch.zeros((len(seqs), mlen - 1), dtype=torch.long, device=device)
            y = torch.zeros((len(seqs), mlen - 1), dtype=torch.long, device=device)
            for b, (s, n) in enumerate(zip(seqs, lens, strict=True)):
                x[b, :n], y[b, :n] = torch.tensor(s[:-1], device=device), torch.tensor(s[1:], device=device)
            pos = torch.arange(mlen - 1, device=device).unsqueeze(0).expand(len(seqs), -1)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                tok_lp = F.log_softmax(model(x, pos).float(), dim=-1).gather(-1, y.unsqueeze(-1)).squeeze(-1).cpu().numpy()
            for b, n in enumerate(lens):
                t_sum, e_lp = float(tok_lp[b, : max(0, n - 1)].sum()), float(tok_lp[b, n - 1]) if n > 0 else 0.0
                out.append({"tok_logp": t_sum, "eos_logp": e_lp, "total_logp": t_sum + e_lp, "n_tok": max(0, n - 1)})
    return out


def load_checkpoint(ckpt_path: str, device):
    """Load a saved ``best.pt`` checkpoint into an eval CausalTransformerLM."""
    import torch

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model = make_model(ck["arch"], direction=ck["direction"]).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck


def train_rescorer(args) -> dict:
    """Train causal L2R/R2L Transformer with strict VRAM cap, duty throttle, and cotenant yield."""
    import torch
    import torch.nn.functional as F

    compute_duty_sleep_s(0.01, args.gpu_duty_cap)
    torch.set_num_threads(min(4, max(1, args.threads)))
    torch.manual_seed(args.seed)
    rng, dev = np.random.default_rng(args.seed), torch.device(args.device)
    gpu_idx = (dev.index or 0) if dev.type == "cuda" else 0
    if dev.type == "cuda" and args.yield_on_cotenant:
        while should_yield_to_cotenant(gpu_index=gpu_idx):
            time.sleep(5.0)
    vram_cap = None
    if dev.type == "cuda":
        vram_cap = apply_gpu_mem_fraction(min(float(args.vram_frac), MAX_RESCORER_VRAM_FRAC), gpu_idx)
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        stream = np.memmap(os.path.join(args.data, "train.u16"), dtype=np.int16, mode="r")
        vstream = np.fromfile(os.path.join(args.data, "valid.u16"), dtype=np.int16)
    model, start_step, best_ppl = make_model(args.arch, direction=args.direction).to(dev), 0, float("inf")
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location="cpu", weights_only=True)
        model.load_state_dict(ck["state_dict"])
        start_step, best_ppl = int(ck.get("step", 0)), float(ck.get("valid_tok_ppl", float("inf")))
    n_params, ctx, B, steps = sum(p.numel() for p in model.parameters()), model.cfg["ctx"], args.batch, args.steps
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.95), fused=(dev.type == "cuda"))
    max_start, log_path = len(stream) - (ctx + 1), os.path.join(args.out, "train.jsonl")
    os.makedirs(args.out, exist_ok=True)
    t0, acc_loss, acc_n, last_val = time.time(), 0.0, 0, {}

    for step in range(start_step + 1, steps + 1):
        if dev.type == "cuda" and args.yield_on_cotenant and step % 100 == 0:
            torch.cuda.synchronize(dev)
            time.sleep(0.5)
            while should_yield_to_cotenant(gpu_index=gpu_idx):
                time.sleep(5.0)
        t_step = time.time()
        lr = args.lr * (step / max(1, args.warmup)) if step <= args.warmup else (
            args.lr * (0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * (step - args.warmup) / max(1, steps - args.warmup))))
        )
        for pg in opt.param_groups:
            pg["lr"] = lr
        starts = rng.integers(0, max_start, size=B)
        arr = stream[starts[:, None] + np.arange(ctx + 1, dtype=np.int64)[None, :]].astype(np.int64)
        x, y, pos, mask, valid = _prep_packed_batch(arr, args.direction, dev)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            loss = F.cross_entropy(model(x, pos, mask).float().reshape(-1, VOCAB), y.reshape(-1), reduction="none").view_as(y)[valid].mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if dev.type == "cuda" and args.gpu_duty_cap < 1.0:
            torch.cuda.synchronize(dev)
            nap = compute_duty_sleep_s(time.time() - t_step, args.gpu_duty_cap)
            if nap > 0.0:
                time.sleep(nap)
        acc_loss, acc_n = acc_loss + float(loss.detach()), acc_n + 1

        if step == start_step + 1 or step % args.log_every == 0 or step == steps:
            rec = {"step": step, "loss": round(acc_loss / max(1, acc_n), 4), "lr": lr,
                   "sec": round(time.time() - t0, 1), "duty_cap": args.gpu_duty_cap,
                   "vram_peak_gb": round(torch.cuda.max_memory_allocated(dev) / 2**30, 3) if dev.type == "cuda" else 0.0}
            acc_loss, acc_n = 0.0, 0
            if step % args.eval_every == 0 or step == steps:
                last_val = eval_stream_ppl(model, vstream, dev)
                rec.update(last_val)
                if last_val["valid_tok_ppl"] < best_ppl:
                    best_ppl = last_val["valid_tok_ppl"]
                    torch.save({"arch": args.arch, "direction": args.direction, "cfg": model.cfg,
                                "step": step, "n_params": n_params, "state_dict": model.state_dict(),
                                **last_val}, os.path.join(args.out, "best.pt"))
            open(log_path, "a", encoding="utf-8").write(json.dumps(rec) + "\n")
            print(json.dumps(rec), flush=True)

    summary = {"arch": args.arch, "direction": args.direction, "n_params": n_params, "steps": steps,
               "batch": B, "ctx": ctx, "best_valid_tok_ppl": best_ppl, "final": last_val,
               "vram_cap": vram_cap, "duty_cap": args.gpu_duty_cap, "elapsed_s": round(time.time() - t0, 1)}
    open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8").write(json.dumps(summary, indent=2) + "\n")
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    tr = ap.add_subparsers(dest="cmd", required=True).add_parser("train")
    for fl, kw in (("--data", {"required": True}), ("--out", {"required": True}), ("--resume", {"default": ""}),
                   ("--arch", {"default": "3.5m", "choices": sorted(PRESETS)}), ("--direction", {"default": "l2r", "choices": ("l2r", "r2l")}),
                   ("--steps", {"type": int, "default": 20000}), ("--batch", {"type": int, "default": 256}), ("--lr", {"type": float, "default": 1.5e-3}),
                   ("--wd", {"type": float, "default": 0.05}), ("--warmup", {"type": int, "default": 500}), ("--log-every", {"type": int, "default": 500}),
                   ("--eval-every", {"type": int, "default": 2000}), ("--device", {"default": "cuda"}), ("--vram-frac", {"type": float, "default": MAX_RESCORER_VRAM_FRAC}),
                   ("--gpu-duty-cap", {"type": float, "default": DEFAULT_GPU_DUTY_CAP}), ("--yield-on-cotenant", {"action": "store_true", "default": True}),
                   ("--no-yield-on-cotenant", {"action": "store_false", "dest": "yield_on_cotenant"}), ("--threads", {"type": int, "default": 4}), ("--seed", {"type": int, "default": 42})):
        tr.add_argument(fl, **kw)
    if ap.parse_args(argv).cmd == "train":
        train_rescorer(ap.parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
