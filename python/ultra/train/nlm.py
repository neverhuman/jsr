"""H46 item 3: small feed-forward neural token LMs for shallow fusion ("+LM" rows, ruling O17).

  python -m train.nlm tokens --text librispeech-lm-norm.txt.gz --tokenizer bpe.model \
      --exclude leak_matches.txt --out DATA [--procs 16]
  python -m train.nlm train --data DATA --arch 2.0m --out RUN [--steps N] [--device cuda]
  python -m train.nlm ppl --run RUN --cache ENC [--kenlm M5.bin]
  python -m train.nlm export --run RUN --out model.ulmff   (needs train.ffnn_lm_export, PR #79)

Model (runtime-lane's `ultra.lm-ffnn/v1`, PR #79): the last `ctx` output tokens (oldest first;
positions before the sentence start are 0 = BOS) index an embedding E [501, d_e]; the slots are
concatenated, passed through one hidden layer (ReLU or SiLU) and an output layer to 501 logits;
ln P(k | ctx) = log_softmax(logits)[k] for output tokens k = 1..500 (index 0 is never a
target). In the beam, a non-blank token k extending hypothesis h scores lm_weight * ln P(k | h) - lodr_weight * ln P_LODR(k | h[-1]); blank
gets 0 and no end-of-sentence term exists, exactly as train.lm_fusion's n-gram fusion.

Training text: the OpenSLR 11 LM text tokenised word by word with the model's BPE (as H42's
`train.lm_decode tokens`), minus every line that contains one of the dev transcripts found
verbatim in it (H42's leak list). Every 1000th line is held out for perplexity. The last
`qat_frac` of the steps train through the exporter's symmetric per-row int8 round trip on E, fc1
and fc2 (activations stay float, as in the Rust runtime), so the exported tables are the trained
ones.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import time

import numpy as np

from train.errors import LmEvalError

VOCAB = 501  # model indices 0..500: 0 is BOS / padding (never a target), k is output token k
SCHEMA = "ultra.lm-ffnn/v1"  # runtime-lane's format (PR #79, crates/ultra-engine/src/lm/ffnn.rs)
ARCHS = {
    # name: (ctx, d_e, d_h, activation), one hidden layer as the Rust FfnnLm implements;
    # the first three are PR #79's presets at our vocabulary
    "0.5m": (6, 96, 384, "relu"),     # 0.45 MiB INT8: the 0.5 MiB rung
    "1.4m": (6, 160, 896, "silu"),    # 1.35 MiB: FFNN-6 class
    "2.0m": (6, 192, 1120, "silu"),   # 1.89 MiB: the 2 MiB rung
    "4m": (8, 192, 2048, "silu"),     # ~4.2 MiB: the 8 MiB rung within the ~5% CPU budget
}


# ---------------------------------------------------------------- tokenisation

def _encode_batch(args):
    """(first line number, lines, tokenizer path) -> (lines, train bytes, valid bytes, stats):
    word-by-word encoding (as train.lm_decode.token_line), uint16 tokens with a 0 after each
    sentence; line n goes to the validation stream when n % 1000 == 0."""
    first, lines, tokenizer = args
    import sentencepiece as spm

    if getattr(_encode_batch, "path", None) != tokenizer:
        _encode_batch.sp = spm.SentencePieceProcessor(model_file=tokenizer)
        _encode_batch.path, _encode_batch.cache = tokenizer, {}
    sp, cache = _encode_batch.sp, _encode_batch.cache
    out = {"train": [], "valid": []}
    st = {"train_tokens": 0, "valid_tokens": 0, "train_lines": 0, "valid_lines": 0}
    for i, line in enumerate(lines):
        toks = []
        for w in line.split():
            got = cache.get(w)
            if got is None:
                got = cache[w] = [t + 1 for t in sp.encode(w)]
            toks.extend(got)
        if not toks:
            continue
        part = "valid" if (first + i) % 1000 == 0 else "train"
        out[part].extend(toks)
        out[part].append(0)
        st[f"{part}_tokens"] += len(toks)
        st[f"{part}_lines"] += 1
    return (len(lines), np.asarray(out["train"], dtype=np.uint16).tobytes(),
            np.asarray(out["valid"], dtype=np.uint16).tobytes(), st)


def tokens(a) -> None:
    """Tokenise the LM text into DATA/{train,valid}.u16 (uint16 token streams, 0 after every
    sentence) + DATA/meta.json. Lines containing an excluded string are removed first by
    `grep -F -v -f` (fixed strings, the way H42 found them); the rest is upper-cased and encoded
    by `--procs` workers in order."""
    import shlex
    import subprocess
    from multiprocessing import Pool

    os.makedirs(a.out, exist_ok=True)
    pat = os.path.join(a.out, "exclude.patterns")
    n_excl = 0
    with open(pat, "w") as fh:
        if a.exclude:
            for raw in open(a.exclude):
                parts = raw.strip().split(None, 1)  # "<count> <transcript>" (uniq -c format)
                if len(parts) == 2 and parts[0].isdigit():
                    fh.write(parts[1] + "\n")
                    n_excl += 1
    cat = "zcat" if a.text.endswith(".gz") else "cat"
    src = f"{cat} {shlex.quote(a.text)}"
    dropped = int(subprocess.run(["bash", "-c", f"{src} | grep -c -F -f {shlex.quote(pat)} || true"],
                                 check=True, capture_output=True, text=True).stdout.strip() or 0) \
        if n_excl else 0
    cmd = f"{src} | grep -v -F -f {shlex.quote(pat)}" if n_excl else src
    proc = subprocess.Popen(["bash", "-c", cmd], stdout=subprocess.PIPE, text=True,
                            encoding="utf-8", errors="strict")

    def batches():
        n, buf = 1, []  # line numbers from 1: every 1000th line is held out
        for line in proc.stdout:
            buf.append(line.strip().upper())
            if len(buf) == a.chunk:
                yield (n, buf, a.tokenizer)
                n += len(buf)
                buf = []
        if buf:
            yield (n, buf, a.tokenizer)

    stats = {"lines": 0, "dropped_lines": dropped, "train_tokens": 0, "valid_tokens": 0,
             "train_lines": 0, "valid_lines": 0}
    with open(os.path.join(a.out, "train.u16"), "wb") as ft, \
            open(os.path.join(a.out, "valid.u16"), "wb") as fv, Pool(a.procs) as pool:
        for nl, tb, vb, st in pool.imap(_encode_batch, batches(), chunksize=1):
            ft.write(tb)
            fv.write(vb)
            stats["lines"] += nl
            for k, v in st.items():
                stats[k] += v
    if proc.wait() != 0:
        raise LmEvalError(f"text pipeline failed: {cmd}")
    with open(a.tokenizer, "rb") as fh:
        tok_sha = hashlib.sha256(fh.read()).hexdigest()
    stats.update({"tokenizer_sha256": tok_sha, "text": a.text, "exclude": a.exclude,
                  "excluded_strings": n_excl, "valid_every": 1000,
                  "line_numbering": "after exclusion, from 1"})
    with open(os.path.join(a.out, "meta.json"), "w") as fh:
        json.dump(stats, fh, indent=1)
    print(json.dumps(stats))


# ---------------------------------------------------------------- model

def fake_quant_sym_rows(w):
    """Per-row symmetric int8 round trip with a straight-through gradient: the quantisation of
    runtime-lane's exporter (`quantize_rows_i8`: scale = max|w| / 127, q = rint(w / scale))."""
    import torch

    mx = w.abs().amax(dim=1, keepdim=True)
    scale = mx / 127.0
    q = torch.clamp(torch.round(w / torch.where(mx > 0, scale, torch.ones_like(scale))), -127, 127)
    wq = torch.where(mx > 0, q * scale, torch.zeros_like(w))
    return w + (wq - w).detach()


def make_model(arch: str):
    import torch
    from torch import nn

    ctx, d_e, d_h, act = ARCHS[arch]

    class FFNN(nn.Module):
        """E [V, d_e] -> concat ctx slots -> fc1 [d_h, ctx*d_e] -> act -> fc2 [V, d_h] ->
        log-softmax over all V = 501 indices (index 0 is never a target), as FfnnLm scores."""

        def __init__(self):
            super().__init__()
            self.ctx, self.activation, self.qat = ctx, act, False
            self.emb = nn.Embedding(VOCAB, d_e)
            self.fc1 = nn.Linear(ctx * d_e, d_h)
            self.fc2 = nn.Linear(d_h, VOCAB)

        def _w(self, w):
            return fake_quant_sym_rows(w) if self.qat else w

        def forward(self, c):  # c (N, ctx) int64, oldest first -> (N, V) log-probabilities
            x = self._w(self.emb.weight)[c].flatten(1)
            h = torch.nn.functional.linear(x, self._w(self.fc1.weight), self.fc1.bias)
            h = torch.relu(h) if self.activation == "relu" else torch.nn.functional.silu(h)
            z = torch.nn.functional.linear(h, self._w(self.fc2.weight), self.fc2.bias)
            return torch.log_softmax(z.float(), dim=-1)

    return FFNN()


def contexts(stream, pos, ctx: int):
    """(N, ctx) contexts before positions `pos` of a token stream with 0 between sentences:
    stream[p-ctx:p], with everything at or before the last 0 in the window set to 0 (BOS)."""
    import torch

    offs = torch.arange(-ctx, 0, device=stream.device)
    idx = (pos[:, None] + offs[None]).clamp(min=0)
    w = stream[idx].long()
    w[(pos[:, None] + offs[None]) < 0] = 0
    z = (w == 0).flip(1).int().cummax(1).values.flip(1).bool()  # a 0 at or after this slot
    return w.masked_fill(z, 0)


def train(a) -> None:
    import torch

    torch.manual_seed(a.seed)
    dev = torch.device(a.device)
    data = np.fromfile(os.path.join(a.data, "train.u16"), dtype=np.uint16)
    valid = np.fromfile(os.path.join(a.data, "valid.u16"), dtype=np.uint16)
    stream = torch.from_numpy(data.astype(np.int16)).to(dev)
    vstream = torch.from_numpy(valid.astype(np.int16)).to(dev)
    vpos = torch.nonzero(vstream > 0).flatten()
    model = make_model(a.arch).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd, betas=(0.9, 0.98))
    steps, qat_from = a.steps, int(a.steps * (1 - a.qat_frac))
    os.makedirs(a.out, exist_ok=True)
    log = open(os.path.join(a.out, "train.jsonl"), "a")  # noqa: SIM115
    n = stream.shape[0]
    gen = torch.Generator(device=dev).manual_seed(a.seed)
    t0 = time.time()
    for step in range(steps):
        lr = a.lr * min(1.0, (step + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * step / steps))
        for g in opt.param_groups:
            g["lr"] = lr
        model.qat = step >= qat_from
        pos = torch.randint(1, n, (a.batch,), device=dev, generator=gen)
        tgt = stream[pos].long()
        keep = tgt > 0
        pos, tgt = pos[keep], tgt[keep]
        lp = model(contexts(stream, pos, model.ctx))
        loss = -lp.gather(1, tgt[:, None]).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % a.log_every == 0 or step == steps - 1:
            rec = {"step": step, "loss": round(float(loss), 4), "lr": lr, "qat": model.qat,
                   "sec": round(time.time() - t0, 1)}
            if step % (10 * a.log_every) == 0 or step == steps - 1:
                rec["valid_ppl"] = round(_ppl(model, vstream, vpos, model.ctx), 3)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(json.dumps(rec), flush=True)
    torch.save({"arch": a.arch, "state": model.state_dict(), "args": vars(a), "steps": steps,
                "qat_from": qat_from}, os.path.join(a.out, "model.pt"))


def _ppl(model, stream, pos, ctx: int, batch: int = 1 << 16) -> float:
    import torch

    tot, cnt = 0.0, 0
    with torch.no_grad():
        for i in range(0, pos.shape[0], batch):
            p = pos[i:i + batch]
            lp = model(contexts(stream, p, ctx))
            tot += float(-lp.gather(1, stream[p].long()[:, None]).sum())
            cnt += p.shape[0]
    return math.exp(tot / max(cnt, 1))


def load_run(run: str, device: str = "cpu"):
    import torch

    p = torch.load(os.path.join(run, "model.pt"), map_location=device, weights_only=False)
    model = make_model(p["arch"])
    model.load_state_dict(p["state"])
    model.eval()
    model.qat = p["qat_from"] < p["steps"]  # a QAT-finished model is used through its int8 trip
    return model.to(device), p


def dev_token_streams(cache: str):
    """The dev references as one token stream per split (0 between utterances), tokenised word
    by word with the cache's tokenizer path rules (train.lm_decode tokens)."""
    from train.lm_tune import load_utts
    from train.tokenizer import TOKDIR, Tokenizer

    out = {}
    for split in ("dev-clean", "dev-other"):
        utts = load_utts(cache, [split])
        tok = Tokenizer(os.path.join(os.path.dirname(TOKDIR), "models",
                                     "zipformer-en-2023-06-26", "bpe.model"))
        toks = [0]
        for u in utts:
            for w in u["ref"].upper().split():
                toks.extend(i + 1 for i in tok.sp.encode(w))
            toks.append(0)
        out[split] = np.asarray(toks, dtype=np.int16)
    return out


def ppl(a) -> None:
    """Token perplexity of the dev references (no end-of-sentence term, BOS context at the
    start of each utterance), for the neural LM and optionally a KenLM token model."""
    import torch

    res = {}
    streams = dev_token_streams(a.cache)
    if a.run:
        model, p = load_run(a.run, a.device)
        for split, s in streams.items():
            st = torch.from_numpy(s).to(a.device)
            pos = torch.nonzero(st > 0).flatten()
            res[f"nlm_{split}"] = round(_ppl(model, st, pos, model.ctx), 3)
    if a.kenlm:
        import kenlm

        lm = kenlm.Model(a.kenlm)
        for split, s in streams.items():
            tot = cnt = 0.0
            sent: list[str] = []
            for t in s.tolist()[1:]:
                if t == 0:
                    if sent:
                        # full_scores gives per-token log10 P with <s>; drop the </s> term
                        sc = [x[0] for x in lm.full_scores(" ".join(sent), bos=True, eos=False)]
                        tot += -sum(sc) * math.log(10.0)
                        cnt += len(sc)
                    sent = []
                else:
                    sent.append(str(t))
            res[f"kenlm_{split}"] = round(math.exp(tot / cnt), 3)
    print(json.dumps(res))
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(res, fh, indent=1)


# ---------------------------------------------------------------- export and fusion

def export(a) -> None:
    """Write the trained model in runtime-lane's `ultra.lm-ffnn/v1` format with their exporter
    (`train.ffnn_lm_export`, PR #79), check the INT8 file against the model on dev contexts, and
    write a JSON manifest (bytes, sha256, parity)."""
    import torch

    try:
        from train.ffnn_lm_export import export_ffnn_v1, load_ffnn_v1
    except ImportError as e:
        raise LmEvalError("train.ffnn_lm_export (runtime-lane's PR #79) is needed to export") from e
    model, p = load_run(a.run)
    model.qat = False  # the exporter quantises the fp32 weights itself
    raw = export_ffnn_v1(model, a.out, ctx=model.ctx, activation=model.activation)
    qlm = load_ffnn_v1(raw)
    rng = np.random.default_rng(0)
    hist = [list(rng.integers(1, VOCAB, size=int(n))) for n in rng.integers(0, 12, size=200)]
    with torch.no_grad():
        ref = model(torch.tensor([[0] * (model.ctx - min(len(h), model.ctx)) + h[-model.ctx:]
                                  if h else [0] * model.ctx for h in hist])).numpy()
    q = np.stack([qlm.score_all(h) for h in hist])
    man = {"schema": SCHEMA, "arch": p["arch"], "file": a.out, "file_bytes": len(raw),
           "sha256": hashlib.sha256(raw).hexdigest(), "source_run": a.run,
           "qat_steps": p["steps"] - p["qat_from"],
           "int8_vs_model_max_abs_logp": float(np.abs(q[:, 1:] - ref[:, 1:]).max()),
           "int8_vs_model_top1_agreement": float((q[:, 1:].argmax(1) == ref[:, 1:].argmax(1)).mean())}
    with open(a.out + ".json", "w") as fh:
        json.dump(man, fh, indent=1)
    print(json.dumps(man))


class NeuralFusion:
    """The `fusion` argument of train.beam.modified_beam_search / train.lm_fusion.FastBeam for a
    neural token LM: rows are computed per distinct context (batched NumPy over the live
    hypotheses of a frame) and cached. Same interface and LODR handling as NgramFusion."""

    def __init__(self, model, lm_weight: float, lodr=None, lodr_weight: float = 0.0,
                 max_rows: int = 1 << 15):
        from train.lm_fusion import narrow

        if lm_weight < 0 or lodr_weight < 0:
            raise LmEvalError("lm_weight and lodr_weight must be >= 0")
        if lodr_weight and lodr is None:
            raise LmEvalError("lodr_weight needs a LODR model")
        self._narrow = narrow
        self.model, self.lm_weight = model, float(lm_weight)
        self.lodr, self.lodr_weight = (lodr, float(lodr_weight)) if lodr_weight else (None, 0.0)
        self.ctx = model.ctx
        self.max_rows = max_rows
        self.rows: dict = {}
        self.contexts = 0

    def _key(self, seq) -> tuple[int, ...]:
        hist = tuple(seq[-self.ctx:])
        return (0,) * (self.ctx - len(hist)) + hist

    def stack(self, seqs):
        import torch

        keys = [self._key(s) for s in seqs]
        missing = sorted({k for k in keys if k not in self.rows})
        if missing:
            with torch.no_grad():
                lp = self.model(torch.tensor(missing, dtype=torch.long)).numpy()
            for k, row in zip(missing, lp, strict=True):
                full = np.zeros(VOCAB, dtype=np.float32)
                full[1:] = row[1:] * self.lm_weight
                if self.lodr is not None:
                    hist = tuple(t for t in k if t) or ()
                    lctx = (-1, *hist) if len(hist) < self.lodr.order - 1 else \
                        hist[len(hist) - (self.lodr.order - 1):]
                    full -= self.lodr.row(self._narrow(lctx, self.lodr.order - 1)).numpy() \
                        * self.lodr_weight
                if len(self.rows) >= self.max_rows:
                    self.rows.pop(next(iter(self.rows)))
                self.rows[k] = full
                self.contexts += 1
        return np.stack([self.rows[k] for k in keys])

    def __call__(self, seqs):
        import torch

        return torch.from_numpy(self.stack(seqs))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tokens")
    t.add_argument("--text", required=True)
    t.add_argument("--tokenizer", required=True)
    t.add_argument("--exclude", default=None, help="`uniq -c` list of strings: drop lines "
                   "containing any (H42's leak_matches.txt)")
    t.add_argument("--out", required=True)
    t.add_argument("--procs", type=int, default=8)
    t.add_argument("--chunk", type=int, default=20000)
    t.set_defaults(fn=tokens)
    r = sub.add_parser("train")
    r.add_argument("--data", required=True)
    r.add_argument("--arch", default="2.0m", choices=sorted(ARCHS))
    r.add_argument("--out", required=True)
    r.add_argument("--steps", type=int, default=40000)
    r.add_argument("--batch", type=int, default=1 << 16)
    r.add_argument("--lr", type=float, default=2e-3)
    r.add_argument("--wd", type=float, default=0.01)
    r.add_argument("--warmup", type=int, default=1000)
    r.add_argument("--qat-frac", type=float, default=0.1)
    r.add_argument("--seed", type=int, default=1234)
    r.add_argument("--log-every", type=int, default=200)
    r.add_argument("--device", default="cuda")
    r.set_defaults(fn=train)
    p = sub.add_parser("ppl")
    p.add_argument("--cache", required=True, help="encoder cache (dev references)")
    p.add_argument("--run", default=None)
    p.add_argument("--kenlm", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default=None)
    p.set_defaults(fn=ppl)
    e = sub.add_parser("export")
    e.add_argument("--run", required=True)
    e.add_argument("--out", required=True)
    e.set_defaults(fn=export)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
