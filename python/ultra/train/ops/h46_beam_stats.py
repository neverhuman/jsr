"""H46 amendment 4: search statistics of the frame-synchronous beam (train.lm_fusion.FastBeam, the
twin of train.beam) on a fixed utterance subset: live hypotheses per frame (mean / max) and the
distinct new LM contexts scored per frame (an engine scoring full rows would look up 500 entries
per new context; the Rust engine's exact upper bound looks up fewer). Contexts are counted per
utterance, as in train.lm_decode (no cross-utterance row cache).

  cd python/ultra && python -m train.ops.h46_beam_stats --cache ENC --checkpoint SLIM.pt \
      --shards DIR --shard-glob 's0[0-3]' --config '{"name": "L8", "beam": 8, "lm": ..., ...}' \
      --out stats_L8.json
"""
import argparse
import glob
import json
import os
import time

import numpy as np


def subset_uids(shards: str, pattern: str, limit: int | None = None) -> set[str]:
    """The uids of the matching shard files: the first `limit` of each (as `--limit` gives the
    engine), or all."""
    uids: set[str] = set()
    for f in glob.glob(os.path.join(shards, pattern, "dev-*.npz")):
        uids.update(str(u) for u in np.load(f)["uids"][:limit])
    return uids


def search(fast, enc_p, beam: int, pen: float, fusion, rec: dict):
    """FastBeam.search with counters (same arithmetic)."""
    hyps: dict[tuple[int, ...], float] = {(): 0.0}
    for t in range(enc_p.shape[0]):
        seqs = list(hyps)
        rec["frames"] += 1
        rec["hyps_sum"] += len(seqs)
        rec["hyps_max"] = max(rec["hyps_max"], len(seqs))
        h = enc_p[t] + fast.older[[s[-2] if len(s) > 1 else 0 for s in seqs]] \
            + fast.newer[[s[-1] if s else 0 for s in seqs]]
        h = np.tanh(h) if fast.act == "tanh" else np.clip(h, 0.0, 6.0)
        z = h @ fast.out_w + fast.out_b
        z -= z.max(axis=1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
        if pen:
            logp[:, 0] -= np.float32(pen)
        if fusion is not None:
            c0 = fusion.contexts
            logp += fusion.stack(seqs)
            new = fusion.contexts - c0
            rec["ctx_sum"] += new
            rec["ctx_max"] = max(rec["ctx_max"], new)
        scores = (np.array([hyps[s] for s in seqs], dtype=np.float32)[:, None] + logp).ravel()
        k = min(beam, scores.size)
        top = np.argpartition(scores, -k)[-k:]
        top = top[np.argsort(-scores[top], kind="stable")]
        v = fast.out_b.shape[0]
        new_h: dict[tuple[int, ...], float] = {}
        for flat in top.tolist():
            hi, tok = divmod(flat, v)
            s = seqs[hi] if tok == 0 else (*seqs[hi], tok)
            sc = float(scores[flat])
            new_h[s] = float(np.logaddexp(np.float32(new_h[s]), np.float32(sc))) if s in new_h \
                else sc
        hyps = new_h
    return max(hyps.items(), key=lambda kv: kv[1])[0]


def main(argv=None) -> None:
    from train.lm_decode import _load_model, load_cache
    from train.lm_fusion import FastBeam, NgramFusion, TokenNgram

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--shards", required=True)
    ap.add_argument("--shard-glob", default="s0[0-3]")
    ap.add_argument("--limit", type=int, default=None, help="first N utterances of each shard")
    ap.add_argument("--config", required=True, help="JSON: name, beam, blank_penalty[, lm, "
                    "lm_weight, lodr, lodr_weight]")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    cfg = json.loads(a.config)
    _, model, tok, _, sha = _load_model(a.checkpoint, "ema")
    fast = FastBeam(model.predictor, model.joiner)
    fusion = None
    if cfg.get("lm"):
        lodr = TokenNgram(cfg["lodr"], tok.num_outputs) if cfg.get("lodr") else None
        fusion = NgramFusion(TokenNgram(cfg["lm"], tok.num_outputs), cfg["lm_weight"], lodr,
                             cfg.get("lodr_weight", 0.0))
    keep = subset_uids(a.shards, a.shard_glob, a.limit)
    rec = {"frames": 0, "hyps_sum": 0, "hyps_max": 0, "ctx_sum": 0, "ctx_max": 0}
    n_utt, audio, t0, tokens = 0, 0.0, time.time(), 0
    for split in ("dev-clean", "dev-other"):
        index, frames = load_cache(a.cache, split)
        for u in index["utts"]:
            if u["uid"] not in keep:
                continue
            x = np.array(frames[u["offset"]:u["offset"] + u["frames"]])
            best = search(fast, x, cfg["beam"], cfg.get("blank_penalty", 0.0), fusion, rec)
            tokens += len(best)
            if fusion is not None:
                fusion.rows.clear()
                fusion.contexts = 0
            n_utt += 1
            audio += u["seconds"]
    out = {"config": cfg, "checkpoint_sha256": sha, "utterances": n_utt,
           "audio_s": round(audio, 1), "frames": rec["frames"], "output_tokens": tokens,
           "live_hyps_per_frame_mean": round(rec["hyps_sum"] / max(rec["frames"], 1), 3),
           "live_hyps_per_frame_max": rec["hyps_max"],
           "new_lm_contexts_per_frame_mean": round(rec["ctx_sum"] / max(rec["frames"], 1), 3),
           "new_lm_contexts_per_frame_max": rec["ctx_max"],
           "new_lm_contexts_per_audio_s": round(rec["ctx_sum"] / max(audio, 1e-9), 2),
           "full_row_lookups_per_audio_s": round(500 * rec["ctx_sum"] / max(audio, 1e-9), 1),
           "python_search_s": round(time.time() - t0, 1), "shard_glob": a.shard_glob,
           "limit_per_shard": a.limit}
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
