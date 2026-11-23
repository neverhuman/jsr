"""Build the H49b `ultra.seqkd/v1` candidate store from kd-cache's R0-scored unions (h49-seqkd).

Input: kd-cache's `*_sp1.0.seq.jsonl` files (r0_gpu.py `--mode seq --extra <student N-best>`): per
utterance the reference, R0's beam-8 N-best and the student's N-best, each distinct R0 token sequence
force-scored by R0 (full-sum RNN-T, `r0-fp32-gpu-v1`). Output: one header line and one line per utterance
with the reference plus at most `--max-alt` retained alternatives (codex-lead 06:52Z: reference + at
most 3), selected by the pre-registered rule:

  alternatives = non-reference candidates, one per canonical string (`ultra-en-wer-norm/v1` words; a
                 string canonically equal to the reference is not an alternative), the highest R0
                 full-sum score kept per string, ranked by that score; the top `--max-alt` are retained.

Student ids = R0 SentencePiece ids + 1 (B0 uses R0's own bpe.model, sha256 c53433de...; output index 0
is blank), so the reference's ids equal the trainer's RNN-T targets (`Tokenizer.encode`). Provenance
(sources, ranks, both scores) is kept on every candidate.

  python h49_build_seqkd_store.py --seq-dir DIR --out STORE.jsonl [--max-alt 3] [--subset 10000]
      [--tokenizer bpe.model] [--tau 1.0]
`--subset N` keeps the first N utterances in sha256(uid) order (the deterministic B-10k screen list).
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from h49_headroom import normalise_words  # noqa: E402  (canonical normaliser port, rule table pinned)

BPE_SHA = "c53433de083c4a6ad12d034550ef22de68cec62c4f58932a7b6b8b2f1e743fa5"


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def select(row: dict, max_alt: int):
    cands = row["candidates"]
    refs = [c for c in cands if "ref" in c["sources"]]
    if len(refs) != 1:
        raise ValueError(f"{row['uid']}: {len(refs)} reference candidates")
    ref = refs[0]
    ref_key = " ".join(normalise_words(ref["text"]))
    best = {}
    for c in cands:
        if c is ref or not math.isfinite(float(c["full_sum"])):
            continue
        key = " ".join(normalise_words(c["text"]))
        if key == ref_key:
            continue
        if key not in best or c["full_sum"] > best[key]["full_sum"]:
            best[key] = c
    alts = sorted(best.values(), key=lambda c: -c["full_sum"])[:max_alt]

    def entry(c, src):
        return {"ids": [int(t) + 1 for t in c["r0_ids"]], "teacher": float(c["full_sum"]),
                "one_per_frame": c.get("one_per_frame"), "text": c["text"], "src": src,
                **({"r0_beam_rank": c["r0_beam_rank"]} if "r0_beam_rank" in c else {})}

    return [entry(ref, ["ref"] + [s for s in ref["sources"] if s != "ref"])] + [entry(c, list(c["sources"])) for c in alts]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--seq-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-alt", type=int, default=3)
    ap.add_argument("--subset", type=int, default=0)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--tokenizer", default="/dev/shm/eval-lane/jsr_data/tokenizers/bpe.model")
    a = ap.parse_args(argv)
    tok_sha = sha256_file(a.tokenizer)
    if tok_sha != BPE_SHA:
        raise SystemExit(f"tokenizer sha256 {tok_sha} != R0 bpe {BPE_SHA}")
    files = sorted(glob.glob(os.path.join(a.seq_dir, "*_sp1.0.seq.jsonl")))
    rows, headers = [], []
    for p in files:
        for line in open(p):
            d = json.loads(line)
            if d.get("header"):
                headers.append({k: d.get(k) for k in ("convention", "model_sha256", "tokenizer_sha256", "beam", "speed",
                                                       "code_sha256")})
                continue
            rows.append(d)
    if a.subset:
        rows = sorted(rows, key=lambda d: hashlib.sha256(d["uid"].encode()).hexdigest())[:a.subset]
    rows.sort(key=lambda d: d["uid"])
    conv = {h.get("convention") for h in headers}
    if len(conv) != 1:
        raise SystemExit(f"mixed or missing teacher conventions: {conv}")
    header = {"header": True, "format": "ultra.seqkd/v1", "student_tokenizer_sha256": tok_sha,
              "teacher": f"R0 ({conv.pop()}), full-sum RNN-T (k2 regular)", "tau": a.tau, "speed": 1.0,
              "max_alt": a.max_alt, "selection": "top-max_alt non-reference canonical strings by R0 full-sum",
              "subset": a.subset or None, "n_utts": len(rows),
              "inputs": {os.path.basename(p): sha256_file(p) for p in files},
              "script_sha256": sha256_file(os.path.abspath(__file__))}
    n_alt = 0
    with open(a.out + ".tmp", "w") as fh:
        fh.write(json.dumps(header) + "\n")
        for d in rows:
            cands = select(d, a.max_alt)
            n_alt += len(cands) - 1
            fh.write(json.dumps({"uid": d["uid"], "speed": float(d.get("speed", 1.0)), "cands": cands}) + "\n")
    os.replace(a.out + ".tmp", a.out)
    print(json.dumps({"out": a.out, "utts": len(rows), "alternatives_per_utt": round(n_alt / max(1, len(rows)), 3),
                      "sha256": sha256_file(a.out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
