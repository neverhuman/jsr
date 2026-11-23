"""Wave 1B-2 full-sentence commit-pass neural LM rescorer (`ULTRA-LM-2M` GRU & `L-ref` Transformer).

Supports L2R and R2L (`--reverse`) training on OpenSLR-11 text, train-960h-derived compound
alignment, per-row INT8 serialization, and fast CPU N-best rescoring (`lm_scale`/`ilme_scale`/`r2l_scale`).
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import time
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from train.errors import DataError, NonFiniteLossError, ShapeError
from train.ffnn_lm_export import quantize_rows_i8
from train.train_ffnn_lm import iter_text_lines
from ultra.wer import corpus_wer

BOS = 0
DEFAULT_COMPOUND_MAP: dict[str, str] = {
    "deathlike": "death like", "everyone": "every one", "alright": "all right",
    "macgregor": "mc gregor", "anyone": "any one", "someone": "some one",
}


def build_compound_align_map(
    train_lines: Iterable[str], candidates: Iterable[tuple[str, str]] | None = None,
    *, min_split_count: int = 5, ratio_thresh: float = 1.5,
) -> dict[str, str]:
    """Derive a fused->split compound alignment map strictly from `train-960h` counts."""
    uni, bi = Counter(), Counter()
    for line in train_lines:
        ws = line.strip().lower().split()
        uni.update(ws)
        bi.update(f"{ws[i]} {ws[i + 1]}" for i in range(len(ws) - 1))
    pairs = list(candidates) if candidates is not None else list(DEFAULT_COMPOUND_MAP.items())
    out = {f.lower(): s.lower() for f, s in pairs if bi[s.lower()] >= min_split_count and bi[s.lower()] >= ratio_thresh * max(1, uni[f.lower()])}
    if candidates is None:
        for bg, sc in bi.items():
            if sc >= min_split_count and (fc := uni[bg.replace(" ", "")]) > 0 and sc >= ratio_thresh * fc:
                out[bg.replace(" ", "")] = bg
    return out


def apply_compound_align(text: str, cmap: dict[str, str]) -> str:
    """Rewrite fused compound tokens using `cmap` (case-insensitive word match)."""
    return " ".join(cmap.get(w.lower(), w) for w in text.split()) if cmap else text


@dataclass(frozen=True)
class RescoreLmConfig:
    arch: str = "ultra_lm_2m"; vocab: int = 501; d_model: int = 256; d_hidden: int = 384  # noqa: E702
    n_layers: int = 2; n_heads: int = 4; ctx: int = 128; reverse: bool = False; dropout: float = 0.0  # noqa: E702

    def validate(self) -> None:
        if self.arch not in ("ultra_lm_2m", "l_ref_tf") or self.vocab < 4 or self.d_model <= 0 or self.ctx < 2:
            raise ShapeError(f"invalid RescoreLmConfig: {self}")
        if self.arch == "l_ref_tf" and (self.d_model % self.n_heads != 0 or self.n_layers < 1):
            raise ShapeError(f"l_ref_tf requires d_model % n_heads == 0 (got {self.d_model}, {self.n_heads})")


RESCORE_PRESETS: dict[str, RescoreLmConfig] = {
    "ultra_lm_2m": RescoreLmConfig("ultra_lm_2m", vocab=501, d_model=256, d_hidden=384, n_layers=2, n_heads=1, ctx=128),
    "l_ref_tf": RescoreLmConfig("l_ref_tf", vocab=501, d_model=256, d_hidden=1024, n_layers=4, n_heads=4, ctx=128),
}


class UltraLm2m(nn.Module):
    """`Emb(V, 256) + 2xGRU(384) + proj(384->256) + tied Emb^T` (~1.85M params)."""

    def __init__(self, cfg: RescoreLmConfig) -> None:
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab, cfg.d_model)
        self.gru = nn.GRU(cfg.d_model, cfg.d_hidden, num_layers=cfg.n_layers, batch_first=True, dropout=cfg.dropout if cfg.n_layers > 1 else 0.0)
        self.proj = nn.Linear(cfg.d_hidden, cfg.d_model)
        self.out_bias = nn.Parameter(torch.zeros(cfg.vocab))
        nn.init.normal_(self.emb.weight, std=0.05)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.gru(self.emb(x))
        return F.linear(self.proj(h), self.emb.weight, self.out_bias)


class _CausalBlock(nn.Module):
    def __init__(self, d: int, heads: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.heads, self.drop = heads, dropout
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.o_proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.ff1, self.ff2 = nn.Linear(d, d_ff), nn.Linear(d_ff, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        q, k, v = (t.view(B, T, self.heads, D // self.heads).transpose(1, 2) for t in self.qkv(self.ln1(x)).chunk(3, dim=-1))
        a = F.scaled_dot_product_attention(q, k, v, dropout_p=self.drop if self.training else 0.0, is_causal=True)
        x = x + self.o_proj(a.transpose(1, 2).contiguous().view(B, T, D))
        return x + self.ff2(F.gelu(self.ff1(self.ln2(x))))


class LRefTransformer(nn.Module):
    """4-layer Causal Transformer (`d=256, 4 heads, ctx=128`, ~3.3M params) for L2R or R2L."""

    def __init__(self, cfg: RescoreLmConfig) -> None:
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.emb, self.pos = nn.Embedding(cfg.vocab, cfg.d_model), nn.Embedding(cfg.ctx, cfg.d_model)
        self.blocks = nn.ModuleList([_CausalBlock(cfg.d_model, cfg.n_heads, cfg.d_hidden, cfg.dropout) for _ in range(cfg.n_layers)])
        self.ln_f, self.out_bias = nn.LayerNorm(cfg.d_model), nn.Parameter(torch.zeros(cfg.vocab))
        nn.init.normal_(self.emb.weight, std=0.05)
        nn.init.normal_(self.pos.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[1]
        if T > self.cfg.ctx:
            raise ShapeError(f"sequence len {T} exceeds ctx={self.cfg.ctx}")
        h = self.emb(x) + self.pos(torch.arange(T, device=x.device))[None, :, :]
        for blk in self.blocks:
            h = blk(h)
        return F.linear(self.ln_f(h), self.emb.weight, self.out_bias)


def build_rescore_lm(cfg: RescoreLmConfig) -> nn.Module:
    return UltraLm2m(cfg) if cfg.arch == "ultra_lm_2m" else LRefTransformer(cfg)


def save_rescore_ckpt(model: nn.Module, path: str | Path, *, quantize_int8: bool = True, compound_map: dict[str, str] | None = None) -> int:
    """Save FP32 or compact per-row symmetric INT8 checkpoint; returns serialized byte count."""
    tensors = {k: (dict(zip(("q", "s"), map(torch.from_numpy, quantize_rows_i8(v.detach().cpu().float())))) if (quantize_int8 and v.ndim == 2) else v.detach().cpu().float()) for k, v in model.state_dict().items()}
    out_p = Path(path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"schema": "ultra.rescore-lm/v1", "config": asdict(model.cfg), "int8": quantize_int8, "compound_map": compound_map or {}, "tensors": tensors}, out_p)
    return out_p.stat().st_size


def load_rescore_ckpt(path: str | Path, device: str = "cpu") -> tuple[nn.Module, dict[str, str]]:
    """Load an FP32 or INT8 rescore checkpoint into a ready `eval()` model."""
    p = torch.load(path, map_location="cpu", weights_only=False)
    if p.get("schema") != "ultra.rescore-lm/v1":
        raise DataError(f"unexpected rescore checkpoint schema {p.get('schema')!r}")
    model = build_rescore_lm(RescoreLmConfig(**p["config"]))
    model.load_state_dict({k: (v["q"].float() * v["s"][:, None] if isinstance(v, dict) else v.float()) for k, v in p["tensors"].items()})
    return model.to(device).eval(), dict(p.get("compound_map") or {})


def _pad_batch(seqs: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
    mlen = max(len(s) for s in seqs)
    x, y = torch.full((len(seqs), mlen), BOS, dtype=torch.long), torch.full((len(seqs), mlen), -100, dtype=torch.long)
    for i, s in enumerate(seqs):
        t = torch.tensor(s, dtype=torch.long)
        y[i, : len(s)] = t
        if len(s) > 1:
            x[i, 1 : len(s)] = t[:-1]
    return x, y


def _iter_seq_batches(seq_factory: Callable[[], Iterable[Sequence[int]]], ctx: int, vocab: int, batch_size: int, *, reverse: bool = False) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    buf: list[list[int]] = []
    while True:
        saw = False
        for seq in seq_factory():
            if s := [int(t) for t in (reversed(seq) if reverse else seq) if 0 < int(t) < vocab][:ctx]:
                saw = True
                buf.append(s)
                if len(buf) == batch_size:
                    yield _pad_batch(buf)
                    buf = []
        if not saw:
            raise DataError("rescore LM stream produced zero valid sequences")


@torch.no_grad()
def score_sequences(model: nn.Module, seqs: Sequence[Sequence[int]], batch_size: int = 64) -> list[float]:
    """Full-sentence log-probability `sum_i ln P(t_i | t_{<i})` (L2R or R2L per `model.cfg.reverse`)."""
    cfg: RescoreLmConfig = model.cfg  # type: ignore[assignment]
    dev, out = next(model.parameters()).device, [0.0] * len(seqs)
    items = [(i, s) for i, raw in enumerate(seqs) if (s := [int(t) for t in (reversed(raw) if cfg.reverse else raw) if 0 < int(t) < cfg.vocab])]
    for off in range(0, len(items), batch_size):
        chunk = items[off : off + batch_size]
        if max(len(s) for _, s in chunk) <= cfg.ctx:
            xb, yb = _pad_batch([s for _, s in chunk])
            lp = F.log_softmax(model(xb.to(dev)).float(), dim=-1)
            mask, tgt = yb.to(dev) != -100, yb.to(dev).clamp(min=0)
            for (oi, _), sc in zip(chunk, (lp.gather(2, tgt.unsqueeze(-1)).squeeze(-1) * mask).sum(dim=1).tolist(), strict=True):
                out[oi] = float(sc)
        else:
            for oi, s in chunk:
                tot = 0.0
                for p in range(0, len(s), cfg.ctx):
                    xb, yb = _pad_batch([s[max(0, p - cfg.ctx + 1) : p + cfg.ctx][-cfg.ctx :]])
                    lp, keep = F.log_softmax(model(xb.to(dev)).float(), dim=-1)[0], min(cfg.ctx, len(s) - p)
                    tot += float(lp[-keep:].gather(1, yb[0, -keep:].to(dev).unsqueeze(-1)).sum().item())
                out[oi] = tot
    return out


def rescore_nbest_records(
    records: Sequence[dict[str, Any]], *, l2r_model: nn.Module | None = None, r2l_model: nn.Module | None = None,
    lm_scale: float = 0.0, r2l_scale: float = 0.0, ilme_scale: float = 0.0, batch_size: int = 64,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Rescore `beam=16/32` N-best candidate JSONL rows with L2R + R2L + ILME on CPU/CUDA."""
    flat = [c[0] for r in records for c in r["nbest"]]
    l2r_sc = score_sequences(l2r_model, flat, batch_size) if (l2r_model is not None and lm_scale != 0.0) else [0.0] * len(flat)
    r2l_sc = score_sequences(r2l_model, flat, batch_size) if (r2l_model is not None and r2l_scale != 0.0) else [0.0] * len(flat)
    rescored, refs, hyps, pos = [], [], [], 0
    for r in records:
        scored = []
        for c in r["nbest"]:
            ilme = float(c[3]) if len(c) > 3 else 0.0
            scored.append((float(c[1]) + lm_scale * l2r_sc[pos] + r2l_scale * r2l_sc[pos] - ilme_scale * ilme, c[0], str(c[2])))
            pos += 1
        best = max(scored, key=lambda t: t[0])
        refs.append(r.get("ref", ""))
        hyps.append(best[2])
        rescored.append({"uid": r["uid"], "ref": r.get("ref", ""), "hyp": best[2], "score": round(best[0], 5), "tokens": best[1]})
    return rescored, {"utterances": len(records), "lm_scale": lm_scale, "r2l_scale": r2l_scale, "ilme_scale": ilme_scale, **(corpus_wer(refs, hyps) if any(refs) else {})}


def train_rescore_lm(
    cfg: RescoreLmConfig, train_factory: Callable[[], Iterable[Sequence[int]]], val_seqs: Sequence[Sequence[int]],
    out_path: str | Path, *, steps: int = 2000, batch_size: int = 64, lr: float = 1e-3, weight_decay: float = 1e-2,
    seed: int = 42, device: str = "auto", bf16: bool = True, quantize_int8: bool = True, compound_map: dict[str, str] | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    """Train `UltraLm2m` or `LRefTransformer` (L2R/R2L) and save INT8/FP32 checkpoint."""
    cfg.validate()
    torch.manual_seed(seed)
    dev = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device))
    amp_ctx = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if (dev.type == "cuda" and bf16 and torch.cuda.is_bf16_supported()) else contextlib.nullcontext
    model = build_rescore_lm(cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    batches = _iter_seq_batches(train_factory, cfg.ctx, cfg.vocab, batch_size, reverse=cfg.reverse)
    init_loss, final_loss, t0 = 0.0, 0.0, time.perf_counter()
    model.train()
    for step in range(1, steps + 1):
        for pg in opt.param_groups:
            pg["lr"] = lr * (0.05 + 0.475 * (1.0 + math.cos(math.pi * (step - 1) / max(1, steps))))
        xb, yb = (t.to(dev) for t in next(batches))
        opt.zero_grad(set_to_none=True)
        with amp_ctx():
            loss = F.cross_entropy(model(xb).reshape(-1, cfg.vocab), yb.reshape(-1), ignore_index=-100)
        val = float(loss.item())
        if not math.isfinite(val):
            raise NonFiniteLossError(step, f"loss={val}", str(out_path))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        init_loss, final_loss = (val if step == 1 else init_loss), val
    model = model.cpu().eval()
    v_tok = sum(len([t for t in s if 0 < int(t) < cfg.vocab]) for s in val_seqs)
    val_nll = -sum(score_sequences(model, val_seqs, batch_size=batch_size)) / max(1, v_tok)
    n_bytes = save_rescore_ckpt(model, out_path, quantize_int8=quantize_int8, compound_map=compound_map)
    return model, {"arch": cfg.arch, "reverse": cfg.reverse, "params": sum(p.numel() for p in model.parameters()), "init_loss": round(init_loss, 4), "final_loss": round(final_loss, 4), "val_nll": round(val_nll, 4), "val_ppl": round(math.exp(min(val_nll, 50.0)), 3), "size_bytes": n_bytes, "elapsed_s": round(time.perf_counter() - t0, 2), "out_path": str(out_path)}


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Wave 1B-2 full-sentence neural LM trainer & N-best rescorer.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    tr, sc = sub.add_parser("train"), sub.add_parser("score-nbest")
    for fl, kw in (("--arch", {"default": "ultra_lm_2m", "choices": sorted(RESCORE_PRESETS)}), ("--reverse", {"action": "store_true"}), ("--train", {"nargs": "+", "required": True}), ("--val", {"nargs": "*"}), ("--tokenizer", {"required": True}), ("--compound-align-map", {}), ("--out", {"required": True}), ("--steps", {"type": int, "default": 2000}), ("--batch-size", {"type": int, "default": 64}), ("--lr", {"type": float, "default": 1e-3}), ("--seed", {"type": int, "default": 42}), ("--device", {"default": "auto"})):
        tr.add_argument(fl, **kw)
    for fl, kw in (("--nbest", {"required": True}), ("--l2r", {}), ("--r2l", {}), ("--lm-scale", {"type": float, "default": 0.0}), ("--r2l-scale", {"type": float, "default": 0.0}), ("--ilme-scale", {"type": float, "default": 0.0}), ("--out", {"required": True})):
        sc.add_argument(fl, **kw)
    a = ap.parse_args(argv)
    if a.cmd == "train":
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor(model_file=a.tokenizer)
        cmap = DEFAULT_COMPOUND_MAP if a.compound_align_map == "default" else (json.loads(Path(a.compound_align_map).read_text(encoding="utf-8")) if a.compound_align_map else {})
        letters = "".join(sp.id_to_piece(i) for i in range(sp.get_piece_size()) if not sp.is_control(i))
        fold = str.upper if sum(c.isupper() for c in letters) > sum(c.islower() for c in letters) else str.lower
        def _enc(paths: Sequence[str], max_n: int = 0) -> Iterator[list[int]]:
            for i, line in enumerate(iter_text_lines(paths), 1):
                if ids := [p + 1 for p in sp.encode(fold(apply_compound_align(line.strip(), cmap)))]:
                    yield ids
                if max_n > 0 and i >= max_n:
                    break
        b = RESCORE_PRESETS[a.arch]
        cfg = RescoreLmConfig(b.arch, sp.get_piece_size() + 1, b.d_model, b.d_hidden, b.n_layers, b.n_heads, b.ctx, a.reverse)
        val_seqs = list(_enc(a.val, 1024)) if a.val else [s for i, s in enumerate(_enc(a.train, 10000)) if i % 50 == 0]
        print(json.dumps(train_rescore_lm(cfg, lambda: _enc(a.train), val_seqs, a.out, steps=a.steps, batch_size=a.batch_size, lr=a.lr, seed=a.seed, device=a.device, compound_map=cmap)[1]), flush=True)
    else:
        rows = [json.loads(ln) for ln in Path(a.nbest).read_text(encoding="utf-8").splitlines() if ln.strip()]
        rescored, met = rescore_nbest_records(rows, l2r_model=load_rescore_ckpt(a.l2r)[0] if a.l2r else None, r2l_model=load_rescore_ckpt(a.r2l)[0] if a.r2l else None, lm_scale=a.lm_scale, r2l_scale=a.r2l_scale, ilme_scale=a.ilme_scale)
        Path(a.out).write_text("\n".join(json.dumps(r) for r in rescored) + "\n", encoding="utf-8")
        print(json.dumps(met), flush=True)


if __name__ == "__main__":
    main()
