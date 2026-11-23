"""Wave 0 WER diagnostics: train-vs-DEV gap, k2 lattice alignment stats, and N-best oracle.

Strictly enforces the `dev_split_v2` Selection Half (`S`) guard: confirmation-half utterances and
`test-*` splits are unconditionally refused.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import multiprocessing as mp
import os
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

from train.asr import ASRModel, ModelConfig
from train.ctc import ctc_loss
from train.data import Featurizer, collate
from train.errors import DataError
from train.lm_fusion import FastBeam, NgramFusion, TokenNgram
from train.tokenizer import TOKDIR, Tokenizer, tokenizer_path
from train.transducer import BLANK, StatelessPredictor, greedy_decode
from ultra.libri import PARQUET, SPLIT_DIRS, Utt, decode_flac, load_split
from ultra.wer import corpus_wer, edit_counts, normalise

FORBIDDEN_SPLITS = ("test-clean", "test-other")


def load_split_guard(split_json: str | Path) -> tuple[set[str], set[str]]:
    """Return `(selection_uids, confirmation_uids)` from `dev_split_v2.json`."""
    utts = (json.loads(Path(split_json).read_text(encoding="utf-8")).get("utterances") or {})
    sel, conf = set(utts.get("selection") or ()), set(utts.get("confirmation") or ())
    if not sel or not conf or (sel & conf):
        raise DataError(f"malformed split guard {split_json}: sel={len(sel)} conf={len(conf)}")
    return sel, conf


def guard_utts(split: str, utts: Sequence[Utt], confirmation_uids: set[str] | None = None) -> None:
    """Refuse `test-*` splits and any `dev_split_v2` confirmation-half utterance."""
    if split in FORBIDDEN_SPLITS or split.startswith("test"):
        raise DataError(f"refusing forbidden test split {split!r}")
    if confirmation_uids:
        leaked = [u.uid for u in utts if u.uid in confirmation_uids]
        if leaked:
            raise DataError(f"refusing {len(leaked)} confirmation-half utterances (e.g. {leaked[0]})")


def load_selection_dev(split_json: str | Path) -> tuple[list[Utt], dict[str, str]]:
    """Load strictly the `dev_split_v2` Selection Half (`S`) across `dev-clean` and `dev-other`."""
    sel, conf = load_split_guard(split_json)
    out, subsets = [], {}
    for split in ("dev-clean", "dev-other"):
        for u in load_split(split, None):
            if u.uid in sel:
                out.append(u)
                subsets[u.uid] = split
    guard_utts("dev-selection", out, conf)
    return out, subsets


def sample_train_utts(split: str, limit: int = 1000) -> list[Utt]:
    """Deterministically sample `limit` un-augmented train utterances across parquet shards."""
    if split not in SPLIT_DIRS or not split.startswith("train-"):
        raise DataError(f"sample_train_utts requires a train-* split, got {split!r}")
    files = sorted(glob.glob(os.path.join(PARQUET, SPLIT_DIRS[split], "*.parquet")))
    if not files:
        raise DataError(f"no parquet shards for {split}")
    n_shards = min(len(files), max(1, min(4, limit // 50)))
    per_shard, out = math.ceil(limit / n_shards), []
    for si in np.linspace(0, len(files) - 1, n_shards, dtype=int).tolist():
        rows = pq.read_table(files[si], columns=["id", "speaker_id", "text", "audio"]).to_pylist()
        for row in rows[:: max(1, len(rows) // per_shard)][:per_shard]:
            out.append(Utt(row["id"], int(row["speaker_id"]), row["text"], decode_flac(row["audio"]["bytes"])))
            if len(out) >= limit:
                return out
    return out


def load_ckpt_model(ckpt_path: str | Path, weights: str = "model", tokenizer_override: str | None = None) -> tuple[ASRModel, Tokenizer, dict[str, Any]]:
    """Load an `ASRModel` checkpoint and matching SentencePiece tokenizer."""
    cp = Path(ckpt_path)
    p = torch.load(cp, map_location="cpu", weights_only=False)
    mcfg = ModelConfig.from_dict(p["model_config"])
    model = ASRModel(mcfg)
    model.load_state_dict(p["ema"] if (weights == "ema" and p.get("ema") is not None) else p["model"])
    model.eval()
    tok_p = tokenizer_override or (p.get("train_config") or {}).get("tokenizer") or ""
    if tok_p and not os.path.exists(tok_p):
        root = os.path.dirname(TOKDIR)
        cands = (tok_p.replace("/home/ubuntu/jsr-data", root), os.path.join(TOKDIR, os.path.basename(tok_p)), os.path.join(root, "models", os.path.basename(os.path.dirname(tok_p)), os.path.basename(tok_p)))
        tok_p = next((c for c in cands if os.path.exists(c)), tok_p)
    tok = Tokenizer(tok_p or tokenizer_path(mcfg.vocab_rnnt - 1))
    sha_f = cp.with_suffix(".sha256")
    sha = sha_f.read_text(encoding="utf-8").strip() if sha_f.is_file() else hashlib.sha256(cp.read_bytes()).hexdigest()
    return model, tok, {"checkpoint": str(cp), "sha256": sha, "step": int(p.get("step", 0)), "params": model.num_params()}


@torch.no_grad()
def pruned_alignment_batch(
    model: ASRModel, enc: torch.Tensor, enc_p: torch.Tensor, pred_p: torch.Tensor,
    tgt: torch.Tensor, enc_lens: torch.Tensor, tgt_lens: torch.Tensor,
) -> dict[str, Any]:
    """Exact k2 MI alignment entropy `H`, pruned-band edge occupancy, and out-of-band mass."""
    import k2

    c, B, S = model.cfg, enc.shape[0], tgt.shape[1]
    s_range = min(c.prune_range, S + 1)
    bd = torch.zeros(B, 4, dtype=torch.int64, device=enc.device)
    bd[:, 2], bd[:, 3] = tgt_lens, enc_lens
    sym = tgt.long()
    simple, (pxg, pyg) = k2.rnnt_loss_smoothed(
        lm=model.simple_lm(pred_p).float(), am=model.simple_am(enc).float(), symbols=sym,
        termination_symbol=BLANK, lm_only_scale=c.lm_only_scale, am_only_scale=c.am_only_scale,
        boundary=bd, rnnt_type="regular", reduction="none", return_grad=True,
    )
    ranges = k2.get_rnnt_prune_ranges(px_grad=pxg, py_grad=pyg, boundary=bd, s_range=s_range)
    am_p, lm_p = k2.do_rnnt_pruning(am=enc_p, lm=pred_p, ranges=ranges)
    logits = model.joiner.logits(am_p, lm_p).float()
    nll_pruned = k2.rnnt_loss_pruned(logits=logits, symbols=sym, ranges=ranges, termination_symbol=BLANK, boundary=bd, rnnt_type="regular", reduction="none")
    px, py = k2.get_rnnt_logprobs_pruned(logits=logits, symbols=sym, ranges=ranges, termination_symbol=BLANK, boundary=bd, rnnt_type="regular")
    tot, (gx, gy) = k2.mutual_information_recursion(px=px, py=py, boundary=bd, return_grad=True)
    e_sc = (gx * torch.where(gx > 0, px, 0.0)).sum(dim=(1, 2)) + (gy * torch.where(gy > 0, py, 0.0)).sum(dim=(1, 2))
    ent = (tot - e_sc).clamp(min=0.0)
    left_occ, right_occ, leak_vals = [], [], []
    for b in range(B):
        sb, tb = int(tgt_lens[b]), int(enc_lens[b])
        if tb <= 0 or sb <= 0:
            continue
        r0, r1 = ranges[b, :tb, 0].long(), ranges[b, :tb, -1].long()
        ts = torch.arange(tb, device=enc.device)
        if (m_l := r0 > 0).any():
            left_occ.extend(gy[b, r0[m_l], ts[m_l]].tolist())
        if (m_r := r1 < sb).any():
            right_occ.extend(gy[b, r1[m_r], ts[m_r]].tolist())
        in_band = pyg[b, :, :tb].gather(0, ranges[b, :tb].clamp(max=sb).transpose(0, 1).long()).sum(dim=0)
        leak_vals.extend((1.0 - in_band).clamp(min=0.0).tolist())
    return {
        "nll_pruned": nll_pruned.tolist(), "nll_simple": simple.tolist(),
        "mi_max_abs_diff": float((nll_pruned + tot).abs().max().item()),
        "entropy": ent.tolist(), "left_edge_occ": left_occ,
        "right_edge_occ": right_occ, "simple_oob_mass": leak_vals,
    }


@torch.no_grad()
def eval_greedy_nll_and_align(
    model: ASRModel, tok: Tokenizer, utts: Sequence[Utt], *, chunk: int = 4, batch_size: int = 8, max_symbols: int = 8,
) -> dict[str, Any]:
    """Un-augmented greedy WER, teacher-forced pruned/CTC NLL, and k2 alignment audit."""
    feat = Featurizer(tok, tok)
    order = sorted(range(len(utts)), key=lambda i: feat.logmel.num_frames(len(utts[i].pcm)))
    refs, hyps, ents, ent_tok, left_occ, right_occ, oob = [], [], [], [], [], [], []
    rnnt_s = simple_s = ctc_s = tok_s = ctc_tok_s = frame_s = mi_diff = 0.0
    for off in range(0, len(order), batch_size):
        b = collate([feat.example(utts[i].uid, utts[i].text, utts[i].pcm) for i in order[off : off + batch_size]])
        enc, lens = model.encode(b.feats, b.feat_lens, chunk)
        enc = enc.float()
        enc_p = model.joiner.enc_proj(enc).float()
        dec, _ = greedy_decode(model.predictor, model.joiner, enc_p, lens, max_symbols)
        refs.extend(b.texts)
        hyps.extend(tok.decode(t) for t in dec)
        cs, ct, _ = ctc_loss(F.log_softmax(model.ctc_head(enc), dim=-1), lens, b.ctc, b.ctc_lens, None)
        ctc_s, ctc_tok_s = ctc_s + float(cs.item()), ctc_tok_s + int(ct)
        pred_p = model.predictor(StatelessPredictor.contexts(b.rnnt, model.cfg.pred_ctx)).float()
        st = pruned_alignment_batch(model, enc, enc_p, pred_p, b.rnnt, lens, b.rnnt_lens)
        mi_diff = max(mi_diff, st["mi_max_abs_diff"])
        rnnt_s, simple_s = rnnt_s + sum(st["nll_pruned"]), simple_s + sum(st["nll_simple"])
        tok_s, frame_s = tok_s + int(b.rnnt_lens.sum()), frame_s + int(lens.sum())
        ents.extend(st["entropy"])
        ent_tok.extend(e / max(1, int(l)) for e, l in zip(st["entropy"], b.rnnt_lens.tolist(), strict=True))
        left_occ.extend(st["left_edge_occ"])
        right_occ.extend(st["right_edge_occ"])
        oob.extend(st["simple_oob_mass"])
    ea, ed, ob = np.asarray(ents, dtype=np.float64), np.asarray(left_occ + right_occ, dtype=np.float64), np.asarray(oob, dtype=np.float64)
    return {
        "utterances": len(utts), "chunk": chunk, **corpus_wer(refs, hyps),
        "rnnt_nll_per_tok": round(rnnt_s / max(1.0, tok_s), 5), "simple_nll_per_tok": round(simple_s / max(1.0, tok_s), 5),
        "ctc_nll_per_tok": round(ctc_s / max(1.0, ctc_tok_s), 5),
        "align_stats": {
            "prune_range": model.cfg.prune_range, "mi_max_abs_diff": round(mi_diff, 7),
            "entropy_mean": round(float(ea.mean()), 4), "entropy_p50": round(float(np.median(ea)), 4), "entropy_p90": round(float(np.percentile(ea, 90)), 4),
            "entropy_per_tok_mean": round(float(np.mean(ent_tok)), 4), "entropy_per_frame_mean": round(float(ea.sum() / max(1.0, frame_s)), 4),
            "active_edge_frames": int(ed.size), "active_edge_occ_mean": round(float(ed.mean()) if ed.size else 0.0, 5), "active_edge_occ_p95": round(float(np.percentile(ed, 95)) if ed.size else 0.0, 5),
            "active_edge_gt_0p05_frac": round(float((ed > 0.05).mean()) if ed.size else 0.0, 5), "active_edge_gt_0p10_frac": round(float((ed > 0.10).mean()) if ed.size else 0.0, 5),
            "simple_oob_mass_mean": round(float(ob.mean()) if ob.size else 0.0, 5), "simple_oob_gt_0p05_frac": round(float((ob > 0.05).mean()) if ob.size else 0.0, 5),
        },
    }


@torch.no_grad()
def encode_enc_p_cache(model: ASRModel, tok: Tokenizer, utts: Sequence[Utt], *, chunk: int = 16, batch_size: int = 8) -> list[tuple[str, str, np.ndarray]]:
    """Encode utterances once at `chunk` and return `[(uid, ref, enc_p_f32), ...]`."""
    feat = Featurizer(tok, tok)
    order = sorted(range(len(utts)), key=lambda i: feat.logmel.num_frames(len(utts[i].pcm)))
    items: list[tuple[str, str, np.ndarray]] = []
    for off in range(0, len(order), batch_size):
        b = collate([feat.example(utts[i].uid, utts[i].text, utts[i].pcm) for i in order[off : off + batch_size]])
        enc, lens = model.encode(b.feats, b.feat_lens, chunk)
        enc_p = model.joiner.enc_proj(enc.float()).numpy().astype(np.float32)
        items.extend((uid, b.texts[j], enc_p[j, : int(lens[j])].copy()) for j, uid in enumerate(b.uids))
    return items


def _run_beam_shard(shard: list[tuple[str, str, np.ndarray]], predictor, joiner, tok: Tokenizer, beam: int, bp: float, lm_p: str | None, lodr_p: str | None, lm_w: float, lodr_w: float) -> list[dict[str, Any]]:
    fast = FastBeam(predictor, joiner)
    fusion = NgramFusion(TokenNgram(lm_p, tok.num_outputs), lm_w, TokenNgram(lodr_p, tok.num_outputs) if lodr_p else None, lodr_w) if lm_p else None
    rows = []
    for uid, ref, x in shard:
        nb = fast.search(x, beam=beam, blank_penalty=bp, fusion=fusion)
        if fusion is not None:
            fusion.rows.clear()
        rows.append({"uid": uid, "ref": ref, "nbest": [[t, round(float(sc), 5), tok.decode(t)] for t, sc in nb]})
    return rows


def _beam_worker(args: tuple[Any, ...]) -> list[dict[str, Any]]:
    torch.set_num_threads(1)
    shard, ps, js, md, tok_p, beam, bp, lm_p, lodr_p, lm_w, lodr_w = args
    model = ASRModel(ModelConfig.from_dict(md))
    model.predictor.load_state_dict(ps)
    model.joiner.load_state_dict(js)
    return _run_beam_shard(shard, model.predictor, model.joiner, Tokenizer(tok_p), beam, bp, lm_p, lodr_p, lm_w, lodr_w)


def score_nbest_oracle(rows: Sequence[dict[str, Any]], ks: Sequence[int] = (1, 2, 4, 8, 16), subsets: dict[str, str] | None = None) -> dict[str, Any]:
    """Compute top-1 and oracle-K WER/edits on N-best rows."""
    def _agg(sub_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
        tot_w, ed = 0, {k: 0 for k in ks}
        for r in sub_rows:
            rw = normalise(r["ref"])
            tot_w += len(rw)
            c_ed = [sum(edit_counts(rw, normalise(c[2]))) for c in r["nbest"]] or [len(rw)]
            for k in ks:
                ed[k] += min(c_ed[:k])
        return {"utterances": len(sub_rows), "ref_words": tot_w, "oracle_wer": {f"k{k}": round(100.0 * ed[k] / max(1, tot_w), 3) for k in ks}, "oracle_edits": {f"k{k}": ed[k] for k in ks}, "recoverable_edits": ed[ks[0]] - ed[ks[-1]]}
    out = {"pooled": _agg(rows)}
    for name in sorted(set((subsets or {}).values())):
        if sub := [r for r in rows if subsets and subsets.get(r["uid"]) == name]:
            out[name] = _agg(sub)
    return out


def eval_nbest_oracle(
    model: ASRModel, tok: Tokenizer, utts: Sequence[Utt], *, chunk: int = 16, beam: int = 16,
    lm_path: str | None = None, lodr_path: str | None = None, lm_weight: float = 0.6, lodr_weight: float = 0.45,
    blank_penalty: float = 0.0, batch_size: int = 8, workers: int = 1,
    enc_cache: list[tuple[str, str, np.ndarray]] | None = None, subsets: dict[str, str] | None = None, out_jsonl: str | Path | None = None,
) -> dict[str, Any]:
    """Run `beam` search with optional `M5/LODR` fusion and compute N-best oracle ceiling."""
    cache = enc_cache if enc_cache is not None else encode_enc_p_cache(model, tok, utts, chunk=chunk, batch_size=batch_size)
    if workers <= 1:
        rows = _run_beam_shard(cache, model.predictor, model.joiner, tok, beam, blank_penalty, lm_path, lodr_path, lm_weight, lodr_weight)
    else:
        ps, js, md = model.predictor.state_dict(), model.joiner.state_dict(), model.cfg.to_dict()
        tasks = [(cache[i::workers], ps, js, md, tok.path, beam, blank_penalty, lm_path, lodr_path, lm_weight, lodr_weight) for i in range(workers) if cache[i::workers]]
        with ProcessPoolExecutor(max_workers=len(tasks), mp_context=mp.get_context("spawn")) as ex:
            rows = [r for part in ex.map(_beam_worker, tasks) for r in part]
    if out_jsonl is not None:
        out_p = Path(out_jsonl)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return {"chunk": chunk, "beam": beam, "lm": os.path.basename(lm_path) if lm_path else None, "lm_weight": lm_weight if lm_path else 0.0, "lodr_weight": lodr_weight if lodr_path else 0.0, **score_nbest_oracle(rows, subsets=subsets)}


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Wave 0 WER diagnostics on RC1-EMA (S-half only).")
    for flag, kw in (("--checkpoint", {"required": True}), ("--split-json", {"required": True}), ("--out-dir", {"required": True}), ("--weights", {"default": "model", "choices": ("model", "ema")}), ("--tokenizer", {}), ("--lm", {}), ("--lodr", {}), ("--lm-weight", {"type": float, "default": 0.6}), ("--lodr-weight", {"type": float, "default": 0.45}), ("--train-limit", {"type": int, "default": 1000}), ("--beam", {"type": int, "default": 16}), ("--batch", {"type": int, "default": 8}), ("--threads", {"type": int, "default": 8}), ("--workers", {"type": int, "default": 8})):
        ap.add_argument(flag, **kw)
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)
    t0, out_dir = time.perf_counter(), Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model, tok, meta = load_ckpt_model(a.checkpoint, weights=a.weights, tokenizer_override=a.tokenizer)
    s_utts, subsets = load_selection_dev(a.split_json)
    res_tc = eval_greedy_nll_and_align(model, tok, sample_train_utts("train-clean-100", a.train_limit), chunk=4, batch_size=a.batch)
    res_to = eval_greedy_nll_and_align(model, tok, sample_train_utts("train-other-500", a.train_limit), chunk=4, batch_size=a.batch)
    res_sc = eval_greedy_nll_and_align(model, tok, [u for u in s_utts if subsets[u.uid] == "dev-clean"], chunk=4, batch_size=a.batch)
    res_so = eval_greedy_nll_and_align(model, tok, [u for u in s_utts if subsets[u.uid] == "dev-other"], chunk=4, batch_size=a.batch)
    enc16 = encode_enc_p_cache(model, tok, s_utts, chunk=16, batch_size=a.batch)
    orc_nolm = eval_nbest_oracle(model, tok, s_utts, chunk=16, beam=a.beam, workers=a.workers, enc_cache=enc16, subsets=subsets, out_jsonl=out_dir / f"s_half_c16_b{a.beam}_nolm.nbest.jsonl")
    orc_lm = eval_nbest_oracle(model, tok, s_utts, chunk=16, beam=a.beam, lm_path=a.lm, lodr_path=a.lodr, lm_weight=a.lm_weight, lodr_weight=a.lodr_weight, workers=a.workers, enc_cache=enc16, subsets=subsets, out_jsonl=out_dir / f"s_half_c16_b{a.beam}_m5lodr.nbest.jsonl") if a.lm else None
    summary = {"schema": "ultra.wave0.wer-diag/v1", "meta": meta, "split_json": a.split_json, "elapsed_s": round(time.perf_counter() - t0, 1), "wer0_gap": {"train_clean": res_tc, "train_other": res_to, "dev_s_clean": res_sc, "dev_s_other": res_so, "clean_dev_over_train_wer": round(res_sc["wer"] / max(1e-6, res_tc["wer"]), 3), "other_dev_over_train_wer": round(res_so["wer"] / max(1e-6, res_to["wer"]), 3), "clean_dev_over_train_nll": round(res_sc["rnnt_nll_per_tok"] / max(1e-6, res_tc["rnnt_nll_per_tok"]), 3), "other_dev_over_train_nll": round(res_so["rnnt_nll_per_tok"] / max(1e-6, res_to["rnnt_nll_per_tok"]), 3)}, "nbest_oracle": {"nolm": orc_nolm, "m5_lodr": orc_lm}}
    (out_dir / "wave0_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
