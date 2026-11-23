"""H46 compact `FFNN-6` GPU/CPU streaming LM trainer and `ultra.lm-ffnn/v1` exporter."""
from __future__ import annotations

import argparse
import contextlib
import gzip
import json
import math
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from train.errors import DataError, NonFiniteLossError
from train.ffnn_lm_export import PRESETS, FfnnLmConfig, QuantizedFfnnLm, export_ffnn_v1, preset_config, verify_parity


class FfnnLmNet(nn.Module):
    """Fixed-context feed-forward LM matching `ffnn_lm_export.export_ffnn_v1`."""

    def __init__(self, cfg: FfnnLmConfig) -> None:
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab, cfg.d_e)
        self.fc1 = nn.Linear(cfg.ctx * cfg.d_e, cfg.d_h)
        self.fc2: nn.Linear | None = None if cfg.tied else nn.Linear(cfg.d_h, cfg.vocab)
        self.out_bias: nn.Parameter | None = nn.Parameter(torch.zeros(cfg.vocab)) if cfg.tied else None
        nn.init.normal_(self.emb.weight, mean=0.0, std=0.05)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.out_bias if self.fc2 is None else self.fc2.bias)  # type: ignore[arg-type]

    def forward(self, ctx_ids: torch.Tensor) -> torch.Tensor:
        h = self.fc1(self.emb(ctx_ids).flatten(-2))
        h = F.silu(h) if self.cfg.activation == "silu" else F.relu(h)
        return F.linear(h, self.emb.weight, self.out_bias) if self.cfg.tied else self.fc2(h)  # type: ignore[misc]

    @torch.no_grad()
    def score_all_fp32(self, history: Sequence[int]) -> np.ndarray:
        c, take = self.cfg, min(len(history), self.cfg.ctx)
        tail = [int(t) if 0 <= int(t) < c.vocab else 0 for t in history[-take:]] if take else []
        slots = torch.tensor([[0] * (c.ctx - take) + tail], dtype=torch.long, device=self.emb.weight.device)
        return F.log_softmax(self.forward(slots).float(), dim=-1)[0].cpu().numpy()

    @classmethod
    def from_quantized(cls, qlm: QuantizedFfnnLm) -> FfnnLmNet:
        net = cls(qlm.cfg)
        with torch.no_grad():
            pairs = [(net.emb.weight, qlm.emb_q, qlm.emb_scale), (net.fc1.weight, qlm.w1_q, qlm.w1_scale)]
            if not qlm.cfg.tied:
                assert net.fc2 is not None and qlm.w2_q is not None and qlm.w2_scale is not None
                pairs.append((net.fc2.weight, qlm.w2_q, qlm.w2_scale))
            for dst, q, s in pairs:
                dst.copy_(torch.from_numpy(q.astype(np.float32) * s[:, None]))
            net.fc1.bias.copy_(torch.from_numpy(qlm.b1))
            (net.out_bias if qlm.cfg.tied else net.fc2.bias).copy_(torch.from_numpy(qlm.b2))  # type: ignore[union-attr]
        return net.eval()


def iter_text_lines(paths: Sequence[str | Path]) -> Iterator[str]:
    """Stream text lines from `.txt`, `.txt.gz`, `.parquet`, or directories of those files."""
    files: list[Path] = []
    for p in map(Path, paths):
        if not p.exists():
            raise DataError(f"LM corpus path missing: {p}")
        files.extend(sorted(c for c in p.iterdir() if c.suffix in (".parquet", ".txt", ".gz")) if p.is_dir() else [p])
    for f in files:
        if f.suffix == ".parquet":
            import pyarrow.parquet as pq
            for batch in pq.ParquetFile(f).iter_batches(columns=["text"]):
                yield from (str(t) for t in batch.column("text").to_pylist() if t)
        else:
            with (gzip.open if f.suffix == ".gz" else open)(f, "rt", encoding="utf-8") as fh:
                yield from (line for line in fh if line.strip())


def iter_token_sequences(paths: Sequence[str | Path], tokenizer_path: str | Path, *, id_shift: int = 1, max_lines: int = 0) -> Iterator[list[int]]:
    """Stream SentencePiece BPE token sequences from text or parquet inputs."""
    import sentencepiece as spm
    if not Path(tokenizer_path).is_file():
        raise DataError(f"tokenizer model missing: {tokenizer_path}")
    sp = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    letters = "".join(sp.id_to_piece(i) for i in range(sp.get_piece_size()) if not sp.is_control(i))
    fold = str.upper if sum(c.isupper() for c in letters) > sum(c.islower() for c in letters) else str.lower
    for count, line in enumerate(iter_text_lines(paths), start=1):
        if ids := [p + id_shift for p in sp.encode(fold(line.strip()))]:
            yield ids
        if max_lines > 0 and count >= max_lines:
            break


def seq_to_windows(seq: Sequence[int], ctx: int, vocab: int) -> tuple[np.ndarray, np.ndarray]:
    """Convert one token sequence into `(ctx -> next_tok)` windows with BOS=0 left-padding."""
    arr = np.asarray(seq, dtype=np.int64)
    if arr.size == 0:
        return np.empty((0, ctx), dtype=np.int64), np.empty((0,), dtype=np.int64)
    x = np.lib.stride_tricks.sliding_window_view(np.concatenate((np.zeros(ctx, dtype=np.int64), arr))[:-1], ctx)
    valid = (arr >= 0) & (arr < vocab)
    xv = np.where((x[valid] >= 0) & (x[valid] < vocab), x[valid], 0)
    return np.ascontiguousarray(xv, dtype=np.int64), np.ascontiguousarray(arr[valid], dtype=np.int64)


def iter_ngram_batches(
    seq_factory: Callable[[], Iterable[Sequence[int]]], ctx: int, vocab: int, batch_size: int,
    *, seed: int = 42, shuffle_buf: int = 0, repeat: bool = True,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Stream `(ctx -> next_tok)` mini-batches with optional deterministic shuffle buffer."""
    rng, cap, buf, n_buf = np.random.default_rng(seed), max(batch_size, shuffle_buf), [], 0
    while True:
        saw = False
        for seq in seq_factory():
            xw, yw = seq_to_windows(seq, ctx, vocab)
            if yw.size:
                saw, n_buf = True, n_buf + int(yw.size)
                buf.append((xw, yw))
                if n_buf >= cap:
                    xa, ya = np.concatenate([b[0] for b in buf]), np.concatenate([b[1] for b in buf])
                    if shuffle_buf > 0:
                        p = rng.permutation(ya.size)
                        xa, ya = xa[p], ya[p]
                    cut = (ya.size // batch_size) * batch_size
                    yield from ((torch.from_numpy(xa[o : o + batch_size]), torch.from_numpy(ya[o : o + batch_size])) for o in range(0, cut, batch_size))
                    buf, n_buf = ([(xa[cut:], ya[cut:])], int(ya.size - cut)) if cut < ya.size else ([], 0)
        if not saw:
            raise DataError("FFNN LM corpus produced zero valid token windows")
        if not repeat:
            if n_buf:
                xa, ya = np.concatenate([b[0] for b in buf]), np.concatenate([b[1] for b in buf])
                yield from ((torch.from_numpy(xa[o : o + batch_size]), torch.from_numpy(ya[o : o + batch_size])) for o in range(0, ya.size, batch_size))
            break


@torch.no_grad()
def evaluate_ppl(model: FfnnLmNet, val_seqs: Iterable[Sequence[int]], batch_size: int = 4096) -> tuple[float, float]:
    """Return `(mean_nll, perplexity)` on held-out token sequences."""
    dev, was_training, loss_sum, n_tok = model.emb.weight.device, model.training, 0.0, 0
    model.eval()
    for xb, yb in iter_ngram_batches(lambda: val_seqs, model.cfg.ctx, model.cfg.vocab, batch_size, repeat=False):
        loss_sum += float(F.cross_entropy(model(xb.to(dev)).float(), yb.to(dev), reduction="sum").item())
        n_tok += int(yb.numel())
    model.train(was_training)
    mean_nll = loss_sum / max(1, n_tok)
    return mean_nll, math.exp(min(mean_nll, 50.0))


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 5000; batch_size: int = 4096; lr: float = 3e-3; weight_decay: float = 1e-2  # noqa: E702
    warmup_steps: int = 200; val_every: int = 500; grad_clip: float = 1.0; seed: int = 42  # noqa: E702
    shuffle_buf: int = 65536; device: str = "auto"; bf16: bool = True; parity_tol: float = 0.25  # noqa: E702


def _cosine_lr(step: int, total: int, base_lr: float, warmup: int) -> float:
    return base_lr * (step + 1) / warmup if warmup > 0 and step < warmup else base_lr * (0.05 + 0.475 * (1.0 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, total - warmup)))))


def train_ffnn_lm(
    cfg: FfnnLmConfig, train_factory: Callable[[], Iterable[Sequence[int]]], val_seqs: Sequence[Sequence[int]],
    out_path: str | Path, tcfg: TrainConfig = TrainConfig(), *, log_path: str | Path | None = None,
) -> tuple[FfnnLmNet, dict[str, Any]]:
    """Train `FfnnLmNet`, export INT8 `ultra.lm-ffnn/v1`, and verify parity automatically."""
    cfg.validate()
    torch.manual_seed(tcfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(tcfg.seed)
    dev = torch.device("cuda" if tcfg.device == "auto" and torch.cuda.is_available() else ("cpu" if tcfg.device == "auto" else tcfg.device))
    amp_ctx = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if (dev.type == "cuda" and tcfg.bf16 and torch.cuda.is_bf16_supported()) else contextlib.nullcontext
    model = FfnnLmNet(cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
    batches = iter_ngram_batches(train_factory, cfg.ctx, cfg.vocab, tcfg.batch_size, seed=tcfg.seed, shuffle_buf=tcfg.shuffle_buf)
    init_loss, final_loss, t0 = 0.0, 0.0, time.perf_counter()
    with contextlib.ExitStack() as stack:
        log_fh = stack.enter_context(Path(log_path).open("a", encoding="utf-8")) if log_path else None
        model.train()
        for step in range(1, tcfg.steps + 1):
            lr = _cosine_lr(step - 1, tcfg.steps, tcfg.lr, tcfg.warmup_steps)
            for pg in opt.param_groups:
                pg["lr"] = lr
            xb, yb = (t.to(dev, non_blocking=True) for t in next(batches))
            opt.zero_grad(set_to_none=True)
            with amp_ctx():
                loss = F.cross_entropy(model(xb), yb)
            val = float(loss.item())
            if not math.isfinite(val):
                raise NonFiniteLossError(step, f"loss={val}", str(out_path))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
            opt.step()
            init_loss, final_loss = (val if step == 1 else init_loss), val
            if tcfg.val_every > 0 and (step % tcfg.val_every == 0 or step == tcfg.steps) and log_fh is not None:
                vl, vp = evaluate_ppl(model, val_seqs, batch_size=tcfg.batch_size)
                log_fh.write(json.dumps({"step": step, "lr": round(lr, 6), "train_loss": round(val, 4), "val_loss": round(vl, 4), "val_ppl": round(vp, 3), "elapsed_s": round(time.perf_counter() - t0, 2)}) + "\n")
                log_fh.flush()
    val_loss, val_ppl = evaluate_ppl(model, val_seqs, batch_size=tcfg.batch_size)
    model, out_p = model.cpu().eval(), Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    raw = export_ffnn_v1(model, out_p)
    hists = [[], [1]] + [list(s[: min(len(s), cfg.ctx)]) for s in val_seqs[:8] if len(s) >= 2]
    parity = verify_parity(model.score_all_fp32, raw, hists, max_abs_tol=tcfg.parity_tol)
    return model, {"init_loss": init_loss, "final_loss": final_loss, "val_loss": val_loss, "val_ppl": val_ppl, "size_bytes": len(raw), "sha256": raw[-32:].hex(), "parity": parity, "out_path": str(out_p)}


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Train and export an H46 INT8 FFNN-6 streaming LM.")
    for flag, kw in (
        ("--preset", {"default": "1.4m", "choices": sorted(PRESETS)}), ("--activation", {"choices": ("relu", "silu")}), ("--tied", {"action": "store_true"}), ("--no-bf16", {"action": "store_true"}),
        ("--train", {"nargs": "+", "required": True}), ("--val", {"nargs": "*"}), ("--tokenizer", {"required": True}), ("--out", {"required": True}), ("--log-jsonl", {}),
        ("--vocab", {"type": int}), ("--ctx", {"type": int}), ("--d-e", {"type": int}), ("--d-h", {"type": int}), ("--id-shift", {"type": int, "default": 1}), ("--max-lines", {"type": int, "default": 0}),
        ("--steps", {"type": int, "default": 5000}), ("--batch-size", {"type": int, "default": 4096}), ("--lr", {"type": float, "default": 3e-3}), ("--weight-decay", {"type": float, "default": 1e-2}),
        ("--warmup-steps", {"type": int, "default": 200}), ("--val-every", {"type": int, "default": 500}), ("--seed", {"type": int, "default": 42}), ("--device", {"default": "auto"}), ("--parity-tol", {"type": float, "default": 0.25}),
    ):
        ap.add_argument(flag, **kw)
    a = ap.parse_args(argv)
    base = preset_config(a.preset, vocab=a.vocab)
    cfg = FfnnLmConfig(base.vocab, a.ctx or base.ctx, a.d_e or base.d_e, a.d_h or base.d_h, a.activation or base.activation, a.tied or base.tied)
    cfg.validate()
    val_seqs = list(iter_token_sequences(a.val, a.tokenizer, id_shift=a.id_shift, max_lines=2048)) if a.val else [s for i, s in enumerate(iter_token_sequences(a.train, a.tokenizer, id_shift=a.id_shift, max_lines=min(a.max_lines or 20000, 20000))) if i % 100 == 0]
    train_fn = (lambda: iter_token_sequences(a.train, a.tokenizer, id_shift=a.id_shift, max_lines=a.max_lines)) if a.val else (lambda: (s for i, s in enumerate(iter_token_sequences(a.train, a.tokenizer, id_shift=a.id_shift, max_lines=a.max_lines)) if i % 100 != 0))
    tcfg = TrainConfig(a.steps, a.batch_size, a.lr, a.weight_decay, a.warmup_steps, a.val_every, 1.0, a.seed, 65536, a.device, not a.no_bf16, a.parity_tol)
    _, summary = train_ffnn_lm(cfg, train_fn, val_seqs, a.out, tcfg, log_path=a.log_jsonl)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
