"""Dev-eval gated DiLoCo anchor promotion CLI (anchors are candidates until promoted)."""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import os
import tempfile
import time
from pathlib import Path


class PromotionError(ValueError):
    """Raised when a candidate anchor fails receipt integrity or dev-eval promotion gates."""


def _atomic_write(dst: Path, data: bytes) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{dst.name}.", suffix=".tmp", dir=str(dst.parent))
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, dst)
    except OSError:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def promote_anchor(
    anchor_path: str | Path, receipt_path: str | Path, out_path: str | Path, *,
    max_wer: float | None = None, min_gain_pp: float | None = None,
    baseline_wer: float | None = None, provenance_path: str | Path | None = None,
) -> dict:
    """Promote a candidate anchor ONLY when a verified canonical dev eval receipt passes."""
    if max_wer is None and min_gain_pp is None:
        raise ValueError("at least one promotion gate (--max-wer or --min-gain-pp) is required")
    ap, rp, op = Path(anchor_path), Path(receipt_path), Path(out_path)
    if not ap.is_file():
        raise PromotionError(f"candidate anchor not found: {ap}")
    if not rp.is_file():
        raise PromotionError(f"eval receipt not found: {rp}")

    anchor_bytes, receipt_bytes = ap.read_bytes(), rp.read_bytes()
    actual_sha = hashlib.sha256(anchor_bytes).hexdigest()
    receipt_sha = hashlib.sha256(receipt_bytes).hexdigest()
    try:
        receipt = json.loads(receipt_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PromotionError(f"invalid receipt JSON: {exc}") from exc
    if not isinstance(receipt, dict):
        raise PromotionError("receipt must be a JSON object")
    if receipt.get("verified") is False:
        raise PromotionError("eval receipt is marked verified=false")
    claimed_sha = str(receipt.get("anchor_sha256", ""))
    if not claimed_sha or not hmac.compare_digest(actual_sha, claimed_sha):
        raise PromotionError(f"anchor_sha256 mismatch: receipt={claimed_sha!r} actual={actual_sha!r}")

    split = str(receipt.get("split") or receipt.get("dataset") or "").strip()
    if not split:
        raise PromotionError("receipt missing canonical dev split/dataset identifier")
    if "ref_words" in receipt and int(receipt["ref_words"]) <= 0:
        raise PromotionError("receipt ref_words must be > 0")
    try:
        wer = float(receipt["wer"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PromotionError(f"receipt missing numeric wer: {exc}") from exc
    if not math.isfinite(wer) or wer < 0.0 or wer > 100.0:
        raise PromotionError(f"invalid receipt wer={wer}")
    if max_wer is not None and wer > float(max_wer):
        raise PromotionError(f"wer={wer:.4f} exceeds max_wer={float(max_wer):.4f}")

    base = baseline_wer if baseline_wer is not None else receipt.get("baseline_wer")
    gain_pp = round(float(base) - wer, 4) if base is not None else None
    if min_gain_pp is not None:
        if gain_pp is None:
            raise PromotionError("min_gain_pp gate requires --baseline-wer or receipt baseline_wer")
        if gain_pp < float(min_gain_pp):
            raise PromotionError(f"gain_pp={gain_pp:.4f} below min_gain_pp={float(min_gain_pp):.4f}")

    prov_path = Path(provenance_path) if provenance_path is not None else Path(f"{op}.provenance.json")
    provenance = {
        "promoted": True, "promoted_at": time.time(), "source_anchor": str(ap), "promoted_path": str(op),
        "anchor_sha256": actual_sha, "receipt_path": str(rp), "receipt_sha256": receipt_sha, "split": split,
        "cohort": receipt.get("cohort"), "round": receipt.get("round"), "wer": wer,
        "baseline_wer": float(base) if base is not None else None, "gain_pp": gain_pp,
        "max_wer": float(max_wer) if max_wer is not None else None,
        "min_gain_pp": float(min_gain_pp) if min_gain_pp is not None else None,
    }
    _atomic_write(op, anchor_bytes)
    _atomic_write(prov_path, json.dumps(provenance, indent=2, sort_keys=True).encode("utf-8"))
    return provenance


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Promote a dev-eval verified DiLoCo candidate anchor.")
    ap.add_argument("--anchor", required=True, help="candidate anchor .safetensors path")
    ap.add_argument("--receipt", required=True, help="canonical dev eval receipt JSON path")
    ap.add_argument("--out", required=True, help="output path for promoted anchor")
    ap.add_argument("--provenance", default=None, help="optional provenance JSON output path")
    ap.add_argument("--max-wer", type=float, default=None, help="maximum allowed dev WER")
    ap.add_argument("--min-gain-pp", type=float, default=None, help="minimum WER gain (pp) over baseline")
    ap.add_argument("--baseline-wer", type=float, default=None, help="incumbent baseline WER")
    a = ap.parse_args(argv)
    try:
        prov = promote_anchor(a.anchor, a.receipt, a.out, max_wer=a.max_wer, min_gain_pp=a.min_gain_pp,
                              baseline_wer=a.baseline_wer, provenance_path=a.provenance)
    except (PromotionError, ValueError, OSError) as exc:
        print(json.dumps({"promoted": False, "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(prov, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
