"""H42 decode driver: encoder cache, beam decodes (N-best, n-gram fusion), LM-text tokeniser.

  python -m train.lm_decode encode --checkpoint CKPT --out DIR --shard 3/10   (one per core)
  python -m train.lm_decode merge --out DIR
  python -m train.lm_decode slim --checkpoint CKPT --out SLIM.pt   (decode-only copy)
  python -m train.lm_decode beam --cache DIR --checkpoint CKPT --configs grid.json \
      --shard 3/14 --out DIR
  zcat lm-text.gz | python -m train.lm_decode tokens --tokenizer bpe.model | lmplz -o 5 ...

`encode` runs the encoder once (EMA weights, batch 8 in length order, the trained chunk: the
`train.evaluate` defaults) and stores the projected encoder frames of every utterance, so each
decode of the wave sees identical acoustics. `beam` decodes one shard of the utterances for a list
of configs with `train.beam.modified_beam_search` (optionally with `train.lm_fusion`) and writes
one JSON line per utterance. `tokens` rewrites text as the decimal token indices the n-gram LM is
trained on. Dev sets only: test-* splits are refused.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

from train.beam import modified_beam_search
from train.evaluate import check_split

CACHE_SCHEMA = "ultra.lm-enc-cache/v1"


def _load_model(ckpt: str, weights: str):
    from train import checkpoint
    from train.asr import ASRModel, ModelConfig
    from train.tokenizer import TOKDIR, Tokenizer, tokenizer_path

    p = checkpoint.load(ckpt, restore_rng=False)
    mcfg = ModelConfig.from_dict(p["model_config"])
    model = ASRModel(mcfg)
    if weights == "ema" and p.get("ema") is None:
        raise ValueError("the checkpoint holds no EMA weights")
    model.load_state_dict(p["ema"] if weights == "ema" else p["model"])
    model.eval()
    tok_override = (p.get("train_config") or {}).get("tokenizer") or ""
    if tok_override and not os.path.exists(tok_override):  # trained elsewhere: same model here
        for cand in (os.path.join(TOKDIR, os.path.basename(tok_override)),
                     os.path.join(os.path.dirname(TOKDIR), "models",
                                  os.path.basename(os.path.dirname(tok_override)),
                                  os.path.basename(tok_override))):
            if os.path.exists(cand):
                tok_override = cand
                break
    tok = Tokenizer(tok_override or tokenizer_path(mcfg.vocab_rnnt - 1))
    tok_c = Tokenizer(tok_override or tokenizer_path(mcfg.vocab_ctc - 1))
    with open(ckpt[:-3] + ".sha256") as fh:
        sha = fh.read().strip()
    # A `slim` file answers for the checkpoint it was cut from. The payload is dropped here.
    return int(p["step"]), model, tok, tok_c, p.get("source_sha256", sha)


def slim(a) -> None:
    """Cut a decode-only copy of a checkpoint: one set of weights plus the configs (about an
    eighth of the file), so that many decode workers do not each load the optimizer state. The
    copy records the sha256 of its source and gets its own completion marker."""
    import hashlib

    from train import checkpoint

    p = checkpoint.load(a.checkpoint, restore_rng=False)
    if a.weights == "ema" and p.get("ema") is None:
        raise ValueError("the checkpoint holds no EMA weights")
    with open(a.checkpoint[:-3] + ".sha256") as fh:
        src = fh.read().strip()
    keep = {"step": p["step"], "model_config": p["model_config"],
            "train_config": {"tokenizer": (p.get("train_config") or {}).get("tokenizer") or ""},
            "model": p["model"] if a.weights == "model" else None,
            "ema": p["ema"] if a.weights == "ema" else None,
            "source_sha256": src, "weights": a.weights}
    torch.save(keep, a.out)
    with open(a.out, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()
    with open(a.out[:-3] + ".sha256", "w") as fh:
        fh.write(digest + "\n")
    print(json.dumps({"out": a.out, "sha256": digest, "source_sha256": src, "step": p["step"],
                      "weights": a.weights}))


def encode(a) -> None:
    """One shard of the encoder pass. Batches are the `train.evaluate` ones (utterances sorted by
    feature frames, stable, in groups of --batch); shard i of n takes every n-th batch, so a set of
    single-threaded workers reproduces the one-process pass exactly. `merge` assembles the parts."""
    from train.data import Featurizer, collate
    from train.transducer import greedy_decode
    from ultra.libri import load_split

    torch.set_num_threads(a.threads)
    shard, nshard = (int(x) for x in a.shard.split("/"))
    step, model, tok, tok_c, sha = _load_model(a.checkpoint, a.weights)
    feat = Featurizer(tok, tok_c)
    os.makedirs(a.out, exist_ok=True)
    utts: list = []
    for split in a.splits:
        check_split(split)
        del utts  # free the previous split's audio before loading the next
        utts = load_split(split, None)
        order = sorted(range(len(utts)), key=lambda i: feat.logmel.num_frames(len(utts[i].pcm)))
        rows, frames, off, t0 = [], [], 0, time.time()
        for bi in range(shard, (len(order) + a.batch - 1) // a.batch, nshard):
            idx = order[bi * a.batch:(bi + 1) * a.batch]
            b = collate([feat.example(utts[i].uid, utts[i].text, utts[i].pcm) for i in idx])
            with torch.no_grad():
                enc, lens = model.encode(b.feats, b.feat_lens, None)
                enc_p = model.joiner.enc_proj(enc.float())
                greedy, _ = greedy_decode(model.predictor, model.joiner, enc_p, lens,
                                          a.max_symbols)
            for j, uid in enumerate(b.uids):
                n = int(lens[j])
                frames.append(enc_p[j, :n].numpy().astype(np.float32))
                rows.append({"uid": uid, "speaker": utts[idx[j]].speaker, "ref": b.texts[j],
                             "offset": off, "frames": n, "pos": bi * a.batch + j,
                             "seconds": float(b.feat_lens[j]) / 100.0, "greedy": greedy[j]})
                off += n
        part = os.path.join(a.out, f"{split}.part{shard:02d}of{nshard:02d}")
        np.save(part + ".npy", np.concatenate(frames))
        meta = {"schema": CACHE_SCHEMA, "split": split, "checkpoint_sha256": sha,
                "step": step, "weights": a.weights, "chunk": model.encoder.cfg.chunk,
                "left_context": model.encoder.cfg.left_context, "batch": a.batch,
                "order": "length", "max_symbols": a.max_symbols, "tokenizer_sha256": tok.sha256,
                "utterances": len(order), "encode_seconds": round(time.time() - t0, 1),
                "utts": rows}
        with open(part + ".json.tmp", "w") as fh:
            json.dump(meta, fh)
        os.replace(part + ".json.tmp", part + ".json")
        print(json.dumps({"split": split, "shard": a.shard, "utts": len(rows), "frames": off,
                          "seconds": meta["encode_seconds"]}), flush=True)


def merge(a) -> None:
    """Assemble the shard parts of `encode` into <split>.enc_p.npy + <split>.index.json (length
    order, as one process would have written them) and remove the parts."""
    import glob

    for split in a.splits:
        check_split(split)
        metas = []
        for f in sorted(glob.glob(os.path.join(a.out, f"{split}.part*of*.json"))):
            with open(f) as fh:
                metas.append((f[:-5] + ".npy", json.load(fh)))
        want = int(metas[0][0].rsplit("of", 1)[1].split(".")[0]) if metas else 0
        if not metas or len(metas) != want:
            raise ValueError(f"{split}: {len(metas)} of {want} encode parts present")
        rows = []
        for npy, meta in metas:
            data = np.load(npy)
            rows.extend((u["pos"], u, data[u["offset"]:u["offset"] + u["frames"]])
                        for u in meta["utts"])
        rows.sort(key=lambda r: r[0])
        head = {k: v for k, v in metas[0][1].items() if k not in ("utts", "encode_seconds")}
        if len(rows) != head["utterances"] or [r[0] for r in rows] != list(range(len(rows))):
            raise ValueError(f"{split}: parts do not cover the split exactly once")
        off, utts = 0, []
        for _, u, x in rows:
            utts.append({**{k: v for k, v in u.items() if k != "pos"}, "offset": off})
            off += u["frames"]
        np.save(os.path.join(a.out, f"{split}.enc_p.npy"), np.concatenate([r[2] for r in rows]))
        index = {**head, "frames": off, "utts": utts,
                 "encode_cpu_seconds": round(sum(m["encode_seconds"] for _, m in metas), 1)}
        tmp = os.path.join(a.out, f"{split}.index.json.tmp")
        with open(tmp, "w") as fh:
            json.dump(index, fh)
        os.replace(tmp, os.path.join(a.out, f"{split}.index.json"))
        for npy, _ in metas:
            os.remove(npy)
            os.remove(npy[:-4] + ".json")
        print(json.dumps({k: v for k, v in index.items() if k != "utts"}), flush=True)


def complete_shards(folder: str, split: str) -> list[str] | None:
    """The shard files of one complete decode of `split` in `folder` (all i of some n present),
    or None. If several shard counts are complete, the one with the fewest files is used."""
    import glob
    import re

    sets: dict[int, set[int]] = {}
    for f in glob.glob(os.path.join(folder, f"{split}.*of*.jsonl")):
        m = re.fullmatch(r"(\d+)of(\d+)\.jsonl", os.path.basename(f)[len(split) + 1:])
        if m:
            sets.setdefault(int(m.group(2)), set()).add(int(m.group(1)))
    for n in sorted(sets):
        if sets[n] == set(range(n)):
            return [os.path.join(folder, f"{split}.{i:02d}of{n:02d}.jsonl") for i in range(n)]
    return None


def load_cache(cache: str, split: str):
    """(index dict, memory-mapped (frames, J) float32 array) of an `encode` output."""
    check_split(split)
    with open(os.path.join(cache, f"{split}.index.json")) as fh:
        index = json.load(fh)
    if index.get("schema") != CACHE_SCHEMA:
        raise ValueError(f"encoder cache schema must be {CACHE_SCHEMA}")
    frames = np.load(os.path.join(cache, f"{split}.enc_p.npy"), mmap_mode="r")
    if frames.shape[0] != index["frames"]:
        raise ValueError(f"{split}: cache has {frames.shape[0]} frames, index {index['frames']}")
    return index, frames


def beam(a) -> None:
    from train.lm_fusion import FastBeam, NgramFusion, TokenNgram

    torch.set_num_threads(1)
    shard, nshard = (int(x) for x in a.shard.split("/"))
    if not 0 <= shard < nshard:
        raise ValueError("--shard is i/n with 0 <= i < n")
    with open(a.configs) as fh:
        configs = json.load(fh)
    _, model, tok, _, sha = _load_model(a.checkpoint, a.weights)
    lms: dict[str, TokenNgram] = {}

    def lm(path):
        if path and path not in lms:
            lms[path] = TokenNgram(path, tok.num_outputs, max_rows=a.lm_rows)
        return lms.get(path)

    fast = FastBeam(model.predictor, model.joiner) if a.fast else None
    caches = {s: load_cache(a.cache, s) for s in a.splits}
    for index, _ in caches.values():
        if index["checkpoint_sha256"] != sha or index["weights"] != a.weights:
            raise ValueError("encoder cache was made from other weights than --checkpoint")
    fusions = [NgramFusion(lm(c["lm"]), c["lm_weight"], lm(c.get("lodr")),
                           c.get("lodr_weight", 0.0)) if c.get("lm") else None for c in configs]
    for split, (index, frames) in caches.items():
        # Utterance-outer loop: the configs of a grid visit nearly the same LM contexts on one
        # utterance, so each LM row is computed once and shared by all of them.
        todo = []
        for cfg, fusion in zip(configs, fusions, strict=True):
            out = os.path.join(a.out, cfg["name"], f"{split}.{shard:02d}of{nshard:02d}.jsonl")
            if not os.path.exists(out) and not complete_shards(os.path.dirname(out), split):
                os.makedirs(os.path.dirname(out), exist_ok=True)
                todo.append((cfg, fusion, out, open(out + ".tmp", "w")))  # noqa: SIM115
        for n, u in enumerate(index["utts"]):
            if n % nshard != shard or not todo:
                continue
            x = np.array(frames[u["offset"]:u["offset"] + u["frames"]])
            for cfg, fusion, _, fh in todo:
                nb: list = []
                t0 = time.perf_counter()
                if fast is not None:
                    nb.append(fast.search(x, cfg["beam"], cfg.get("blank_penalty", 0.0), fusion))
                else:
                    modified_beam_search(
                        model.predictor, model.joiner, torch.from_numpy(x)[None],
                        torch.tensor([u["frames"]]), cfg["beam"],
                        blank_penalty=cfg.get("blank_penalty", 0.0), nbest=nb, fusion=fusion)
                best = nb[0][0][0]
                row = {"uid": u["uid"], "tokens": best, "hyp": tok.decode(best),
                       "score": nb[0][0][1], "sec": round(time.perf_counter() - t0, 4)}
                if cfg.get("nbest"):
                    row["nbest"] = [[t, round(sc, 4), tok.decode(t)] for t, sc in nb[0]]
                fh.write(json.dumps(row) + "\n")
                if fusion is not None:
                    fusion.rows.clear()  # weighted rows are per utterance; raw LM rows stay cached
            if n % (50 * nshard) == shard:
                for *_, fh in todo:
                    fh.flush()  # progress is visible while a shard runs
        for _, _, out, fh in todo:
            fh.close()
            os.replace(out + ".tmp", out)
        print(json.dumps({"done": split, "shard": a.shard, "configs": len(todo),
                          "lm_lookups": {os.path.basename(k): v.lookups for k, v in lms.items()}}),
              flush=True)


def token_line(sp, cache: dict, line: str) -> str:
    """One text line as space-separated decimal output indices (SentencePiece id + 1), encoded word
    by word through `cache`. Pieces never cross a space, so this is the segmentation the model
    gives each word in a short utterance. (The R0 "BPE" model is a SentencePiece unigram model:
    on lines of thousands of characters its whole-line Viterbi can flip a near-tie through
    float32 rounding of the running score, which is why `tokens` counts, not forbids, mismatches.)
    """
    out = []
    for w in line.split():
        got = cache.get(w)
        if got is None:
            got = cache[w] = " ".join(str(i + 1) for i in sp.encode(w))
        if got:
            out.append(got)
    return " ".join(out)


def tokens(a) -> None:
    from train.tokenizer import Tokenizer

    tok = Tokenizer(a.tokenizer)
    cache: dict[str, str] = {}
    fold = str.upper if tok.upper else str.lower
    write = sys.stdout.write
    lines = ntok = checked = differ = 0
    for line in sys.stdin:
        line = fold(line)
        got = token_line(tok.sp, cache, line)
        if lines < a.check:
            checked += 1
            differ += got != " ".join(map(str, tok.encode(line)))
        lines += 1
        if got:
            ntok += got.count(" ") + 1
            write(got + "\n")
    stats = {"lines": lines, "tokens": ntok, "word_types": len(cache), "checked_lines": checked,
             "lines_differing_from_whole_line_encoding": differ}
    print(json.dumps(stats), file=sys.stderr, flush=True)
    if differ > max(1, checked // 1000):
        raise ValueError(f"word-by-word encoding differs from whole-line encoding on {differ} of "
                         f"{checked} checked lines (more than 0.1%)")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("encode")
    e.add_argument("--checkpoint", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--splits", nargs="+", default=["dev-clean", "dev-other"])
    e.add_argument("--weights", default="ema", choices=("ema", "model"))
    e.add_argument("--batch", type=int, default=8)
    e.add_argument("--max-symbols", type=int, default=8)
    e.add_argument("--threads", type=int, default=1)
    e.add_argument("--shard", default="0/1", help="i/n: every n-th batch, starting at batch i")
    e.set_defaults(fn=encode)
    c = sub.add_parser("slim")
    c.add_argument("--checkpoint", required=True)
    c.add_argument("--out", required=True, help="path ending in .pt")
    c.add_argument("--weights", default="ema", choices=("ema", "model"))
    c.set_defaults(fn=slim)
    m = sub.add_parser("merge")
    m.add_argument("--out", required=True)
    m.add_argument("--splits", nargs="+", default=["dev-clean", "dev-other"])
    m.set_defaults(fn=merge)
    b = sub.add_parser("beam")
    b.add_argument("--cache", required=True)
    b.add_argument("--checkpoint", required=True)
    b.add_argument("--configs", required=True, help="JSON list of {name, beam, blank_penalty, "
                   "nbest, lm, lm_weight, lodr, lodr_weight}")
    b.add_argument("--shard", default="0/1")
    b.add_argument("--lm-rows", type=int, default=1 << 15,
                   help="LM rows (one per n-gram context, ~2 kB each) kept per worker")
    b.add_argument("--fast", action="store_true",
                   help="search with train.lm_fusion.FastBeam (NumPy; for weight sweeps) instead "
                        "of train.beam.modified_beam_search (the reference)")
    b.add_argument("--out", required=True)
    b.add_argument("--splits", nargs="+", default=["dev-clean", "dev-other"])
    b.add_argument("--weights", default="ema", choices=("ema", "model"))
    b.set_defaults(fn=beam)
    t = sub.add_parser("tokens")
    t.add_argument("--tokenizer", required=True)
    t.add_argument("--check", type=int, default=20000,
                   help="compare the word-by-word encoding with whole-line encoding on the first "
                        "N lines; the count of differing lines is printed, > 0.1%% is an error")
    t.set_defaults(fn=tokens)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
