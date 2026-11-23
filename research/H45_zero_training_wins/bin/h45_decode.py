#!/usr/bin/env python3
"""h45-wins decode driver: train.evaluate's exact decode loop, for many checkpoints and chunks in one process.

  python h45_decode.py --src <tree>/python/ultra --jsr-data <dir> --out <dir> \
      --ckpt TAG=PATH[@model|@ema] [--ckpt ...] --chunks 4,8,16 [--splits dev-clean,dev-other] \
      [--batch 8] [--max-symbols 8] [--beam 0 [--nbest] [--ctc-score]] [--device cuda] [--threads 2]

Per (tag, chunk, split) it writes <out>/<tag>.c<chunk>.ms<max>[.b<beam>].<split>.jsonl with train.evaluate's
--dump rows (uid, ref, hyp, ctc_hyp, seconds[, beam_hyp, nbest]) and a .json summary line, so eval-lane's
score_row.py / dumps_to_scoreinput.py read them unchanged. The loop is train.evaluate.main's: examples sorted
by duration, collate() batches of --batch, model.encode(feats, lens, chunk), joiner.enc_proj, greedy_decode,
modified_beam_search, CTC argmax. Features are computed once per split and reused for every checkpoint.

With --ctc-score every N-best entry also carries the CTC head's log-likelihood of that hypothesis on the
same encoder output: the hypothesis text is re-tokenised with the checkpoint's CTC tokenizer (never the
transducer ids), and an impossible CTC path is recorded as null (never as zero cost).
N-best rows: [text, transducer_logp, ctc_logp_or_null, n_ctc_tokens].

Dev splits only: any test-* split is refused, and the data directory given should hold only dev caches.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--src", required=True, help="python/ultra directory of the evaluator tree")
    ap.add_argument("--jsr-data", required=True, help="JSR_DATA holding cache/librispeech_pcm/dev-*.npz + tokenizers")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", action="append", required=True, help="TAG=PATH[@model|@ema] (default @model)")
    ap.add_argument("--chunks", default="4")
    ap.add_argument("--splits", default="dev-clean,dev-other")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-symbols", type=int, default=8)
    ap.add_argument("--beam", type=int, default=0)
    ap.add_argument("--nbest", action="store_true")
    ap.add_argument("--ctc-score", action="store_true")
    ap.add_argument("--left-context", type=int, default=-1)
    ap.add_argument("--uids", default=None, help="file of utterance ids: decode only these (batches then "
                    "differ from a full-split decode; compare only with decodes of the same subset)")
    ap.add_argument("--subset-tag", default="", help="inserted into output names, e.g. .S or .C")
    ap.add_argument("--feat-cache", default=None, help="directory caching featurised examples per split / subset")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--gpu-fraction", type=float, default=0.40)
    a = ap.parse_args(argv)

    os.environ["JSR_DATA"] = a.jsr_data
    sys.path.insert(0, a.src)
    import torch
    import torch.nn.functional as F

    from train import checkpoint
    from train.asr import ASRModel, ModelConfig
    from train.beam import modified_beam_search
    from train.data import Featurizer, collate
    from train.tokenizer import TOKDIR, Tokenizer, tokenizer_path
    from train.transducer import greedy_decode
    from ultra.libri import load_split
    from ultra.wer import corpus_wer

    torch.set_num_threads(a.threads)
    if a.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(a.gpu_fraction, 0)
    splits = a.splits.split(",")
    for s in splits:
        if not s.startswith("dev-"):
            raise SystemExit(f"split {s!r} refused: dev-* only")
    chunks = [int(c) for c in a.chunks.split(",")]
    if a.ctc_score and not a.nbest:
        raise SystemExit("--ctc-score needs --nbest")
    if a.nbest and not a.beam:
        raise SystemExit("--nbest needs --beam")
    os.makedirs(a.out, exist_ok=True)

    specs = []
    for item in a.ckpt:
        tag, _, rest = item.partition("=")
        path, _, key = rest.partition("@")
        specs.append((tag, path, key or "model"))

    keep = None
    if a.uids:
        with open(a.uids) as fh:
            keep = {x.strip() for x in fh if x.strip()}

    def suffix(c):
        return a.subset_tag + f".c{c}.ms{a.max_symbols}" + (f".b{a.beam}" if a.beam else "")

    feats_by_split: dict[str, list] = {}
    tok_path_seen = None
    tok = tok_c = None
    for tag, path, key in specs:
        todo = [(c, s) for c in chunks for s in splits
                if not os.path.exists(os.path.join(a.out, f"{tag}{suffix(c)}.{s}.json"))]
        if not todo:
            print(json.dumps({"skip": tag, "reason": "all outputs exist"}), flush=True)
            continue
        t0 = time.time()
        p = checkpoint.load(path, restore_rng=False)
        with open(path[:-3] + ".sha256") as fh:
            ck_sha = fh.read().strip()
        mcfg = ModelConfig.from_dict(p["model_config"])
        model = ASRModel(mcfg)
        if key == "ema":
            if p.get("ema") is None:
                raise SystemExit(f"{path}: no EMA weights")
            model.load_state_dict(p["ema"])
        else:
            model.load_state_dict(p["model"])
        model.eval().to(a.device)
        if a.left_context >= 0:
            import dataclasses

            model.encoder.cfg = dataclasses.replace(model.encoder.cfg, left_context=a.left_context)
        ecfg = model.encoder.cfg
        tok_override = (p.get("train_config") or {}).get("tokenizer") or ""
        if tok_override and not os.path.exists(tok_override):
            tok_override = os.path.join(TOKDIR, os.path.basename(tok_override))
        tpath = (tok_override or tokenizer_path(mcfg.vocab_rnnt - 1),
                 tok_override or tokenizer_path(mcfg.vocab_ctc - 1))
        tsha = (sha256_file(tpath[0]), sha256_file(tpath[1]))
        if p.get("tokenizer_sha256") and p["tokenizer_sha256"] != tsha[0]:
            raise SystemExit(f"{tag}: tokenizer sha256 {tsha[0]} != checkpoint {p['tokenizer_sha256']}")
        if tok_path_seen is None:
            tok_path_seen = tsha
            tok, tok_c = Tokenizer(tpath[0]), Tokenizer(tpath[1])
        elif tsha != tok_path_seen:
            raise SystemExit(f"{tag}: tokenizer files {tsha} differ from the first checkpoint's {tok_path_seen}")
        feat = Featurizer(tok, tok_c)
        for c, split in todo:
            if split not in feats_by_split:
                cpath = None
                if a.feat_cache:
                    os.makedirs(a.feat_cache, exist_ok=True)
                    key_src = "|".join([split, a.uids and sha256_file(a.uids) or "all", tsha[0], tsha[1],
                                        sha256_file(os.path.join(a.src, "train", "features.py")),
                                        sha256_file(os.path.join(a.src, "train", "data.py"))])
                    cpath = os.path.join(a.feat_cache, hashlib.sha256(key_src.encode()).hexdigest()[:16]
                                         + f".{split}.pt")
                if cpath and os.path.exists(cpath):
                    from train.data import Example

                    z = torch.load(cpath, map_location="cpu", weights_only=False)
                    feats_by_split[split] = [Example(*row) for row in z["rows"]]
                else:
                    utts = load_split(split)
                    if keep is not None:
                        utts = [u for u in utts if u.uid in keep]
                    feats_by_split[split] = sorted((feat.example(u.uid, u.text, u.pcm) for u in utts),
                                                   key=lambda e: e.seconds)
                    if cpath:
                        torch.save({"rows": [(e.uid, e.text, e.feats, e.rnnt, e.ctc)
                                             for e in feats_by_split[split]]}, cpath + ".tmp")
                        os.replace(cpath + ".tmp", cpath)
            exs = feats_by_split[split]
            chunk = c or None
            refs, hyps, ctc_hyps, uids, secs = [], [], [], [], []
            beam_hyps, nbest_rows = [], []
            forced = exact = 0
            t1 = time.time()
            for i in range(0, len(exs), a.batch):
                b = collate(exs[i:i + a.batch]).to(a.device)
                with torch.no_grad():
                    enc, lens = model.encode(b.feats, b.feat_lens, chunk)
                    enc_p = model.joiner.enc_proj(enc.float())
                    toks, f = greedy_decode(model.predictor, model.joiner, enc_p, lens, a.max_symbols)
                    nb: list = []
                    if a.beam:
                        beam_hyps.extend(tok.decode(t) for t in modified_beam_search(
                            model.predictor, model.joiner, enc_p, lens, a.beam,
                            nbest=nb if a.nbest else None))
                    logits_c = model.ctc_head(enc.float())
                    best = logits_c.argmax(dim=-1)
                    if a.nbest:
                        ctc_lp = torch.log_softmax(logits_c.float(), dim=-1) if a.ctc_score else None
                        for j, cands in enumerate(nb):
                            rows = []
                            texts = [tok.decode(t) for t, _ in cands]
                            if a.ctc_score:
                                tj = int(lens[j])
                                tgt = [tok_c.encode(t) for t in texts]
                                ok = [len(x) > 0 for x in tgt]
                                lp = [None] * len(texts)
                                idx = [n for n in range(len(texts)) if ok[n]]
                                if idx:
                                    lpj = ctc_lp[j, :tj].unsqueeze(1).expand(-1, len(idx), -1).contiguous()
                                    flat = torch.tensor([k for n in idx for k in tgt[n]], dtype=torch.long,
                                                        device=lpj.device)
                                    tl = torch.tensor([len(tgt[n]) for n in idx], dtype=torch.long)
                                    il = torch.full((len(idx),), tj, dtype=torch.long)
                                    nll = F.ctc_loss(lpj, flat, il, tl, blank=0, reduction="none",
                                                     zero_infinity=False)
                                    for n, v in zip(idx, nll.tolist(), strict=True):
                                        lp[n] = None if not (v < float("inf")) else -float(v)
                                for n in range(len(texts)):
                                    if not ok[n]:  # empty hypothesis: CTC prob of the all-blank path
                                        lp[n] = float(ctc_lp[j, :tj, 0].sum())
                                for n, (t, sc) in enumerate(cands):
                                    rows.append([texts[n], float(sc), lp[n], len(tgt[n])])
                            else:
                                for n, (t, sc) in enumerate(cands):
                                    rows.append([texts[n], float(sc), None, None])
                            nbest_rows.append(rows)
                forced += f
                for j in range(best.shape[0]):
                    seq = best[j, :int(lens[j])].tolist()
                    ctc_hyps.append(tok_c.decode(
                        [k for n, k in enumerate(seq) if k != 0 and (n == 0 or seq[n - 1] != k)]))
                secs.extend(float(x) / 100.0 for x in b.feat_lens)
                uids.extend(b.uids)
                for r, t in zip(b.texts, toks, strict=True):
                    h = tok.decode(t)
                    refs.append(r)
                    hyps.append(h)
                    exact += int(r.lower().split() == h.split())
            base = os.path.join(a.out, f"{tag}{suffix(c)}.{split}")
            with open(base + ".jsonl.tmp", "w") as fh:
                for n, (u, r, h, cc) in enumerate(zip(uids, refs, hyps, ctc_hyps, strict=True)):
                    row = {"uid": u, "ref": r, "hyp": h, "ctc_hyp": cc, "seconds": secs[n]}
                    if a.beam:
                        row["beam_hyp"] = beam_hyps[n]
                    if a.nbest:
                        row["nbest"] = nbest_rows[n]
                    fh.write(json.dumps(row) + "\n")
            os.replace(base + ".jsonl.tmp", base + ".jsonl")
            w = corpus_wer(refs, hyps)
            out = {"tag": tag, "subset": a.uids and os.path.basename(a.uids), "checkpoint": os.path.abspath(path),
                   "checkpoint_sha256": ck_sha,
                   "weights": key, "step": p.get("step"), "split": split, "n": len(refs),
                   "max_symbols": a.max_symbols, "batch": a.batch, "order": "length",
                   "forced_advances": forced, "exact_utts": exact, **w,
                   "ctc_wer": corpus_wer(refs, ctc_hyps)["wer"],
                   "chunk": chunk or ecfg.chunk, "left_context": ecfg.left_context,
                   "seconds_wall": round(time.time() - t1, 1), "src": os.path.abspath(a.src)}
            if a.beam:
                out["beam"] = a.beam
                out["beam_wer"] = corpus_wer(refs, beam_hyps)
            with open(base + ".json.tmp", "w") as fh:
                fh.write(json.dumps(out) + "\n")
            os.replace(base + ".json.tmp", base + ".json")
            print(json.dumps({k: out[k] for k in ("tag", "split", "chunk", "wer", "seconds_wall")}), flush=True)
        del model
        if a.device.startswith("cuda"):
            torch.cuda.empty_cache()
        print(json.dumps({"done": tag, "seconds": round(time.time() - t0, 1)}), flush=True)


if __name__ == "__main__":
    main()
