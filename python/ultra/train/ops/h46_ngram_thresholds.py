"""H46 amendment 2 (a): lmplz --prune thresholds for the order-3 rungs, from exact counts and the
v2 pack's bytes per entry only (no WER, no perplexity).

  cd python/ultra && python -m train.ops.h46_ngram_thresholds ngram_counts.json > thresholds.json
"""
import json
import sys

from train.errors import LmEvalError

d = json.load(open(sys.argv[1]))
if "kept_bigrams_by_threshold" not in d or "kept_trigrams_by_threshold" not in d:
    raise LmEvalError(f"{sys.argv[1]}: not a train.ops.h46_ngram_counts output")
K2 = {int(k): v for k, v in d["kept_bigrams_by_threshold"].items()}
K3 = {int(k): v for k, v in d["kept_trigrams_by_threshold"].items()}
LODR = 287296          # lodr2.v2.ulm bytes (measured)
FIXED = 8000           # header + 500 unigrams, generous
B2, B3 = 3.5, 2.5      # bytes per bigram (with backoff / child range) and per top-order trigram
out = {}
for rung in (0.5, 2.0, 8.0):
    for fam, extra in (("fusion", 0), ("lodr", LODR)):
        budget = rung * 2 ** 20 * 0.95 - extra
        t2 = min((t for t in sorted(K2) if FIXED + B2 * K2[t] <= 0.5 * budget), default=None)
        if t2 is None:
            out[f"{rung}_{fam}"] = None
            continue
        t3 = min((t for t in sorted(K3) if t >= t2 and FIXED + B2 * K2[t2] + B3 * K3[t] <= budget),
                 default=None)
        est = FIXED + B2 * K2[t2] + (B3 * K3[t3] if t3 is not None else 0) + extra
        out[f"{rung}_{fam}"] = {"prune": f"0 {t2} {t3}", "bigrams": K2[t2], "trigrams": K3[t3],
                                "est_bytes_incl_lodr_if_used": int(est), "budget": int(budget + extra)}
print(json.dumps(out, indent=1))
