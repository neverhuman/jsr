"""Exact Stateless-Predictor Internal Language Model Estimation (ILME).

For a stateless RNN-T predictor (`ctx = [prev2, prev1]`) and `tanh` joiner:
    z = W_out @ tanh(e_avg + pred(prev2, prev1)) + b_out
    ln P_ILM(k | prev2, prev1) = z[k] - logsumexp_{j != blank}(z[j])   for k != blank
    ln P_ILM(blank | prev2, prev1) = -inf

Matches `crates/ultra-engine/src/lm/ilme.rs` (`ultra.ilme-stateless/v1`) to < 1e-4 max abs diff.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
from pathlib import Path

import numpy as np

ILME_SCHEMA = "ultra.ilme-stateless/v1"
NR = 16
KG = 4


def sha256_f32_le(vec: np.ndarray) -> str:
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    h = hashlib.sha256()
    h.update(struct.pack(f"<{arr.size}f", *arr.tolist()))
    return h.hexdigest()


def _unpack_packed_i8_joiner(q_flat: np.ndarray, scale: np.ndarray, v: int, jn: int) -> np.ndarray:
    """Unpack `ultra-engine/packed-i8 v1` panels (`NR=16, KG=4`) into exact dequantized `[V, J]`."""
    n_panels = (v + NR - 1) // NR
    k_groups = (jn + KG - 1) // KG
    panel_len = k_groups * NR * KG
    q = np.asarray(q_flat, dtype=np.int8).reshape(n_panels, panel_len)
    w = np.zeros((v, jn), dtype=np.float32)
    for row in range(v):
        p, j = divmod(row, NR)
        s = float(scale[row])
        panel = q[p]
        for kk in range(jn):
            code = int(panel[((kk // KG) * NR + j) * KG + (kk % KG)])
            w[row, kk] = float(code) * s
    return w


def load_stateless_joiner_tensors(export_dir: str | Path) -> dict[str, np.ndarray | int | list[int]]:
    """Load predictor + joiner weights from an `ultra.r1-weights/v1` directory (FP32 or INT8)."""
    from safetensors import safe_open

    root = Path(export_dir)
    arch = json.loads((root / "arch.json").read_text())
    dec = arch["decoder"]
    cfg = arch["model"]["config"]
    vocab = int(cfg["vocab_rnnt"])
    joiner = int(cfg["joiner"])
    blank = int(dec["blank_id"])
    ctx_init = [int(x) for x in dec["ctx_init"]]
    with safe_open(str(root / arch["files"]["weights"]["path"]), framework="numpy") as st:
        emb = st.get_tensor("predictor.emb.weight").astype(np.float32)
        pred_w = st.get_tensor("predictor.proj.weight").astype(np.float32)
        pred_b = st.get_tensor("predictor.proj.bias").astype(np.float32)
        join_b = st.get_tensor("joiner.out.bias").astype(np.float32)
        join_w = st.get_tensor("joiner.out.weight").astype(np.float32)
    if arch.get("int8") is not None and "joiner.out" not in arch["int8"].get("fp32_kinds", []):
        i8_path = root / arch["files"]["weights_int8"]["path"]
        if i8_path.exists():
            with safe_open(str(i8_path), framework="numpy") as st8:
                if "joiner.out.q" in st8.keys():
                    q = st8.get_tensor("joiner.out.q")
                    s = st8.get_tensor("joiner.out.scale").astype(np.float32)
                    join_w = _unpack_packed_i8_joiner(q, s, vocab, joiner)
    return {
        "vocab": vocab,
        "joiner": joiner,
        "blank": blank,
        "ctx_init": ctx_init,
        "emb": emb,
        "pred_w": pred_w,
        "pred_b": pred_b,
        "join_w": join_w,
        "join_b": join_b,
        "weights_sha256": arch["files"]["weights"]["sha256"],
    }


def stateless_ilme_logprobs(
    emb: np.ndarray,
    pred_w: np.ndarray,
    pred_b: np.ndarray,
    join_w: np.ndarray,
    join_b: np.ndarray,
    e_avg: np.ndarray,
    ctx_pairs: np.ndarray,
    blank: int = 0,
) -> np.ndarray:
    """Compute exact non-blank ILME natural-log probabilities `[N, V]` for context pairs `[N, 2]`."""
    pairs = np.asarray(ctx_pairs, dtype=np.int64).reshape(-1, 2)
    e0 = emb[pairs[:, 0]]
    e1 = emb[pairs[:, 1]]
    pred_in = np.concatenate([e0, e1], axis=-1).astype(np.float64)
    pred = pred_in @ pred_w.astype(np.float64).T + pred_b.astype(np.float64)
    hid = np.tanh(pred.astype(np.float32) + np.asarray(e_avg, dtype=np.float32)[None, :])
    logits = hid.astype(np.float64) @ join_w.astype(np.float64).T + join_b.astype(np.float64)
    logits = logits.astype(np.float32)
    logits[:, blank] = -np.inf
    mx = np.max(logits, axis=-1, keepdims=True)
    shifted = np.where(np.isfinite(logits), np.exp(logits - mx), 0.0)
    lse = mx + np.log(np.sum(shifted, axis=-1, keepdims=True))
    out = logits - lse
    out[:, blank] = -np.inf
    return out.astype(np.float32)


def estimate_eavg_from_frames(
    frames: list[np.ndarray],
    emission_counts: list[np.ndarray],
    checkpoints: tuple[int, ...] = (1000, 2500),
) -> tuple[np.ndarray, int, list[dict[str, float]]]:
    """Compute emission-weighted mean projected encoder vector `e_avg` and convergence diagnostics."""
    if not frames:
        raise ValueError("frames must be non-empty")
    jn = int(frames[0].shape[-1])
    acc = np.zeros(jn, dtype=np.float64)
    total_em = 0
    saved: list[tuple[int, np.ndarray]] = []
    for idx, (enc, cnt) in enumerate(zip(frames, emission_counts, strict=True), start=1):
        w = np.asarray(cnt, dtype=np.float64).reshape(-1)
        mask = w > 0
        if np.any(mask):
            acc += (w[mask, None] * np.asarray(enc, dtype=np.float64)[mask]).sum(axis=0)
            total_em += int(w[mask].sum())
        if idx in checkpoints and total_em > 0:
            saved.append((idx, (acc / total_em).astype(np.float32)))
    if total_em == 0:
        raise ValueError("zero non-blank emissions observed")
    e_avg = (acc / total_em).astype(np.float32)
    conv: list[dict[str, float]] = []
    norm_f = float(np.linalg.norm(e_avg.astype(np.float64)))
    for n, vec in saved:
        va, vb = vec.astype(np.float64), e_avg.astype(np.float64)
        cos = float(np.dot(va, vb) / max(1e-12, float(np.linalg.norm(va)) * norm_f))
        l2 = float(np.linalg.norm(va - vb))
        conv.append({"utts": float(n), "cos_vs_final": cos, "l2_vs_final": l2})
    return e_avg, total_em, conv


def save_ilme_artifact(
    path: str | Path,
    e_avg: np.ndarray,
    num_utts: int,
    num_emissions: int,
    model_weights_sha256: str,
) -> dict:
    arr = np.asarray(e_avg, dtype=np.float32).reshape(-1)
    art = {
        "schema": ILME_SCHEMA,
        "joiner": int(arr.size),
        "num_utts": int(num_utts),
        "num_emissions": int(num_emissions),
        "model_weights_sha256": str(model_weights_sha256),
        "e_avg_sha256": sha256_f32_le(arr),
        "e_avg": [float(x) for x in arr.tolist()],
    }
    p = Path(path)
    if p.parent and str(p.parent) != ".":
        os.makedirs(p.parent, exist_ok=True)
    p.write_text(json.dumps(art, indent=2))
    return art


def load_ilme_artifact(path: str | Path) -> tuple[np.ndarray, dict]:
    art = json.loads(Path(path).read_text())
    if art.get("schema") != ILME_SCHEMA:
        raise ValueError(f"unexpected ILME schema: {art.get('schema')}")
    arr = np.asarray(art["e_avg"], dtype=np.float32)
    if arr.size != int(art["joiner"]):
        raise ValueError("ILME artifact joiner mismatch")
    if sha256_f32_le(arr) != art["e_avg_sha256"]:
        raise ValueError("ILME artifact e_avg_sha256 mismatch")
    return arr, art


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Stateless-predictor ILME verifier / scorer.")
    ap.add_argument("--export", required=True, help="ultra.r1-weights/v1 export directory")
    ap.add_argument("--eavg", required=True, help="ultra.ilme-stateless/v1 JSON path")
    ap.add_argument("--ctx", default="0,0;0,5;5,12", help="semicolon-separated prev2,prev1 pairs")
    ap.add_argument("--out", default=None, help="optional JSON dump of reference log-probs")
    args = ap.parse_args(argv)
    t = load_stateless_joiner_tensors(args.export)
    e_avg, art = load_ilme_artifact(args.eavg)
    pairs = [[int(x) for x in pair.split(",")] for pair in args.ctx.split(";") if pair.strip()]
    lp = stateless_ilme_logprobs(
        t["emb"], t["pred_w"], t["pred_b"], t["join_w"], t["join_b"],
        e_avg, np.asarray(pairs, dtype=np.int64), blank=int(t["blank"]),
    )
    payload = {
        "schema": ILME_SCHEMA,
        "e_avg_sha256": art["e_avg_sha256"],
        "pairs": pairs,
        "log_probs": lp.tolist(),
    }
    if args.out:
        Path(args.out).write_text(json.dumps(payload))
    print(json.dumps({"pairs": len(pairs), "vocab": int(t["vocab"]), "e_avg_sha256": art["e_avg_sha256"]}))


if __name__ == "__main__":
    main()
