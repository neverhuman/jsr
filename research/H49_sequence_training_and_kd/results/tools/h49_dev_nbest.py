"""H49 DEV counterpart of the sequence-KD pilot (h49-seqkd): student N-best on full DEV.

Student = B0 = A1-49000 EMA (`A1-49000-ema-as-model.pt`, sha256 4055c989...), chunk 4 (160 ms),
the pinned evaluator tree d5f231e for model / featurizer / tokenizer / greedy, and a batched copy
of the modified-beam law (kd-cache's `beam_nbest_generic`, copied at a recorded sha256 and checked
against `train.beam.modified_beam_search` on the first utterances of each split).

Outputs (one JSONL row per utterance, analysis only, DEV is never a training target):
  {uid, split, speaker, ref, seconds, n_frames, greedy: {ids, text},
   beams: {"4": [[ids, score], ...], "8": [...], "16": [...]}}
plus <out>/META.json and <out>/greedy.score-input.jsonl (ultra-score input of the greedy 1-best, to
reproduce eval-lane's B0 rows: dev-clean 4.480, dev-other 12.373, pooled 8.302).

  JSR_DATA=/dev/shm/eval-lane/jsr_data python h49_dev_nbest.py --out /dev/shm/h49-seqkd/dev \
      [--splits dev-clean,dev-other] [--beams 4,8,16] [--chunk 4] [--verify 16]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

CODE = os.environ.get("H49_EVAL_CODE", "/dev/shm/h49-seqkd/code-d5f231e/python/ultra")
BEAMLIB = os.environ.get("H49_BEAMLIB", "/dev/shm/h49-seqkd/kdc")  # pinned copy of r0_gpu.py
CKPT = "/dev/shm/eval-lane/ckpt/A1-49000-ema-as-model.pt"
CKPT_SHA = "4055c989c3822b326c52f5a0abb055810ddef7dab7ea85acffa388d134a3a70c"
BPE_SHA = "c53433de083c4a6ad12d034550ef22de68cec62c4f58932a7b6b8b2f1e743fa5"
DEV_SPLITS = ("dev-clean", "dev-other")


def sha256_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", default=",".join(DEV_SPLITS))
    ap.add_argument("--beams", default="4,8,16")
    ap.add_argument("--chunk", type=int, default=4)
    ap.add_argument("--max-symbols", type=int, default=8)
    ap.add_argument("--greedy-batch", type=int, default=8, help="the evaluator's batch (bit-identical greedy)")
    ap.add_argument("--beam-frames", type=int, default=60000, help="padded fbank frames per beam batch")
    ap.add_argument("--verify", type=int, default=16, help="check the batched beam on the first N utts per split")
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--tokenizer", default=os.path.join(os.environ.get("JSR_DATA", ""), "tokenizers", "bpe.model"))
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N utterances per split")
    a = ap.parse_args(argv)
    splits = [s for s in a.splits.split(",") if s]
    for s in splits:
        if s not in DEV_SPLITS:  # never a test set
            ap.error(f"split {s!r} refused: DEV only")
    beams = [int(b) for b in a.beams.split(",") if b]
    if sha256_file(a.ckpt) != CKPT_SHA:
        raise SystemExit("checkpoint sha256 mismatch")
    if sha256_file(a.tokenizer) != BPE_SHA:
        raise SystemExit("tokenizer sha256 mismatch")
    sys.path.insert(0, CODE)
    sys.path.insert(0, BEAMLIB)
    import torch

    import r0_gpu  # noqa: E402  (only beam_nbest_generic is used)
    from train import checkpoint
    from train.asr import ASRModel, ModelConfig
    from train.beam import modified_beam_search
    from train.data import Featurizer, collate
    from train.tokenizer import Tokenizer
    from train.transducer import greedy_decode
    import numpy as np
    from ultra.libri import CACHE, Utt

    def load_split(split):
        """ultra.libri.load_split's cached .npz, read once (its per-row z[...] lookups are O(n^2))."""
        z = np.load(os.path.join(CACHE, f"{split}.npz"))
        offs, pcm, uids, spk, texts = z["offsets"], z["pcm"], z["uids"], z["speakers"], z["texts"]
        return [Utt(str(uids[i]), int(spk[i]), str(texts[i]), pcm[offs[i]:offs[i + 1]]) for i in range(len(uids))]

    torch.set_num_threads(4)
    dev = torch.device("cuda")
    # fleet cap: <= 40% of the card when sharing (the caching allocator otherwise keeps every batch shape)
    torch.cuda.set_per_process_memory_fraction(float(os.environ.get("H49_GPU_FRACTION", "0.35")), dev)
    p = checkpoint.load(a.ckpt, restore_rng=False)
    model = ASRModel(ModelConfig.from_dict(p["model_config"]))
    model.load_state_dict(p["model"])
    model.eval().to(dev)
    tok = Tokenizer(a.tokenizer)
    feat = Featurizer(tok, tok)
    os.makedirs(a.out, exist_ok=True)
    meta = {"what": "H49 DEV student N-best (analysis only)", "student": "A1-49000 EMA (B0)",
            "ckpt": a.ckpt, "ckpt_sha256": CKPT_SHA, "tokenizer_sha256": BPE_SHA, "chunk": a.chunk,
            "max_symbols": a.max_symbols, "beams": beams, "eval_code": CODE,
            "beam_law": "modified beam (<=1 token/frame), equal sequences merged by log-sum-exp",
            "beamlib_sha256": sha256_file(os.path.join(BEAMLIB, "r0_gpu.py")),
            "script_sha256": sha256_file(os.path.abspath(__file__)),
            "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0), "limit": a.limit, "splits": {}}
    rows_path = os.path.join(a.out, "dev_student_nbest.jsonl")
    si_path = os.path.join(a.out, "greedy.score-input.jsonl")
    with open(rows_path + ".tmp", "w") as fr, open(si_path + ".tmp", "w") as fs:
        for split in splits:
            t0 = time.time()
            utts = load_split(split)
            if a.limit:
                utts = utts[:a.limit]
            spk = {u.uid: u.speaker for u in utts}
            exs = sorted((feat.example(u.uid, u.text, u.pcm) for u in utts), key=lambda e: e.seconds)
            greedy, nfr = {}, {}
            with torch.no_grad():
                for i in range(0, len(exs), a.greedy_batch):
                    b = collate(exs[i:i + a.greedy_batch]).to(dev)
                    enc, lens = model.encode(b.feats, b.feat_lens, a.chunk)
                    enc_p = model.joiner.enc_proj(enc.float())
                    toks, _ = greedy_decode(model.predictor, model.joiner, enc_p, lens, a.max_symbols)
                    for uid, t, n in zip(b.uids, toks, lens.tolist(), strict=True):
                        greedy[uid], nfr[uid] = t, int(n)
            t_g = time.time() - t0
            nb = {bm: {} for bm in beams}
            verified = mismatched = 0
            i = 0
            with torch.no_grad():
                while i < len(exs):  # length-sorted, so B * Tmax stays under the frame budget
                    j = i + 1
                    while j < len(exs) and exs[j].feats.shape[0] * (j + 1 - i) <= a.beam_frames:
                        j += 1
                    b = collate(exs[i:j]).to(dev)
                    enc, lens = model.encode(b.feats, b.feat_lens, a.chunk)
                    enc_p = model.joiner.enc_proj(enc.float())
                    ln = [int(x) for x in lens]
                    for bm in beams:
                        res = r0_gpu.beam_nbest_generic(model.predictor, model.joiner.logits, enc_p, ln, bm)
                        for uid, hyps in zip(b.uids, res, strict=True):
                            nb[bm][uid] = hyps
                        if verified < a.verify and bm == 8:
                            k = min(len(b.uids), a.verify - verified)
                            ref_nb = []
                            modified_beam_search(model.predictor, model.joiner, enc_p[:k], lens[:k], bm,
                                                 nbest=ref_nb)
                            for mine, ref in zip(res[:k], ref_nb, strict=True):
                                same = [s for s, _ in mine] == [list(s) for s, _ in ref] and \
                                    all(abs(x - y) < 1e-3 for (_, x), (_, y) in zip(mine, ref))
                                verified += 1
                                mismatched += int(not same)
                    i = j
            by_uid = {e.uid: e for e in exs}
            for u in utts:
                e = by_uid[u.uid]
                g = greedy[u.uid]
                row = {"uid": u.uid, "split": split, "speaker": spk[u.uid], "ref": u.text,
                       "seconds": round(e.seconds, 3), "n_frames": nfr[u.uid],
                       "greedy": {"ids": g, "text": tok.decode(g)},
                       "beams": {str(bm): [[s, round(sc, 5)] for s, sc in nb[bm][u.uid]] for bm in beams}}
                fr.write(json.dumps(row) + "\n")
                fs.write(json.dumps({"id": u.uid, "speaker": str(spk[u.uid]), "dataset": split, "ref": u.text,
                                     "hyp": tok.decode(g), "duration_s": round(e.seconds, 3)}) + "\n")
            meta["splits"][split] = {"utts": len(utts), "greedy_wall_s": round(t_g, 1),
                                     "wall_s": round(time.time() - t0, 1),
                                     "beam_verified": verified, "beam_mismatched": mismatched}
            print(json.dumps({"split": split, **meta["splits"][split]}), flush=True)
    os.replace(rows_path + ".tmp", rows_path)
    os.replace(si_path + ".tmp", si_path)
    meta["rows_sha256"] = sha256_file(rows_path)
    meta["score_input_sha256"] = sha256_file(si_path)
    with open(os.path.join(a.out, "META.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
