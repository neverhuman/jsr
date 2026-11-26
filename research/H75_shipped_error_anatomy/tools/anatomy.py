"""H75 error anatomy over ultra-score score-input JSONL files (id, speaker, dataset, ref, hyp).

Uses the pinned normaliser (ultra.wer.normalise) and a Levenshtein alignment per utterance. Reports:
S/D/I by subset, position, utterance length, speaker, word frequency class (train-960 transcript counts),
top confusions, and the concentration of edits. Usage: anatomy.py OUT.json NAME=score-input.jsonl ...
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict

sys.path[:0] = ["/home/ubuntu/jsr/.agent/worktrees/wer-claude/python/ultra"]
from ultra.wer import normalise  # noqa: E402

TRAIN_TXT = "/mnt/xbabe3-external/jsr/tokenizers/librispeech_train_text.txt"
FUNC = set("A AN THE AND IN OF TO IS WAS IT HE SHE I YOU WE THEY HIS HER THAT THIS AS AT BY FOR ON WITH BE BUT OR NOT HAD HAVE HAS ARE WERE SO NO".split())


def align(r: list[str], h: list[str]) -> list[tuple[str, str | None, str | None, int]]:
    """Return ops (kind, ref_word, hyp_word, ref_index) with kind in C/S/D/I."""
    n, m = len(r), len(h)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j - 1] + (r[i - 1] != h[j - 1]), d[i - 1][j] + 1, d[i][j - 1] + 1)
    ops, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (r[i - 1] != h[j - 1]):
            ops.append(("C" if r[i - 1] == h[j - 1] else "S", r[i - 1], h[j - 1], i - 1)); i -= 1; j -= 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            ops.append(("D", r[i - 1], None, i - 1)); i -= 1
        else:
            ops.append(("I", None, h[j - 1], i)); j -= 1
    return ops[::-1]


def freq_class(c: int) -> str:
    return "OOV(0)" if c == 0 else "rare(1-9)" if c < 10 else "mid(10-99)" if c < 100 else "common(100-999)" if c < 1000 else "top(1000+)"


def analyse(rows: list[dict], wc: Counter) -> dict:
    tot = Counter(); by_sub = defaultdict(Counter); by_pos = defaultdict(Counter); by_len = defaultdict(Counter)
    by_spk = defaultdict(Counter); by_freq = defaultdict(Counter); conf = Counter(); dels = Counter(); ins = Counter()
    utt_edits = []
    for row in rows:
        r, h = [w.upper() for w in normalise(row["ref"])], [w.upper() for w in normalise(row["hyp"])]
        ops = align(r, h)
        e = Counter(k for k, *_ in ops)
        edits = e["S"] + e["D"] + e["I"]
        utt_edits.append(edits)
        L = len(r)
        lb = "01-09" if L < 10 else "10-19" if L < 20 else "20-34" if L < 35 else "35+"
        for key, store in ((row["dataset"], by_sub), (lb, by_len), (str(row["speaker"]), by_spk)):
            store[key].update({"S": e["S"], "D": e["D"], "I": e["I"], "N": L})
        tot.update({"S": e["S"], "D": e["D"], "I": e["I"], "N": L})
        for k, rw, hw, ri in ops:
            pos = "w1" if ri == 0 else "w2-5" if ri < 5 else "w6-15" if ri < 15 else ("last3" if ri >= L - 3 else "w16+")
            if k != "I":
                by_pos[pos]["N"] += 1
                fc = freq_class(wc[rw]); by_freq[fc]["N"] += 1
                by_freq["function" if rw in FUNC else "content"]["N"] += 1
            if k in "SDI":
                by_pos[pos][k] += 1
            if k in "SD":
                by_freq[freq_class(wc[rw])][k] += 1
                by_freq["function" if rw in FUNC else "content"][k] += 1
            if k == "S":
                conf[f"{rw}->{hw}"] += 1
            elif k == "D":
                dels[rw] += 1
            elif k == "I":
                ins[hw] += 1

    def rate(c: Counter) -> dict:
        n = max(1, c["N"])
        return {"N": c["N"], "S": c["S"], "D": c["D"], "I": c["I"], "wer_pct": round(100 * (c["S"] + c["D"] + c["I"]) / n, 3)}

    se = sorted(utt_edits, reverse=True)
    spk = sorted(((k, rate(v)) for k, v in by_spk.items()), key=lambda t: -t[1]["wer_pct"])
    return {
        "pooled": rate(tot), "by_subset": {k: rate(v) for k, v in sorted(by_sub.items())},
        "by_position": {k: rate(v) for k, v in by_pos.items()}, "by_length": {k: rate(v) for k, v in sorted(by_len.items())},
        "by_word_class_SD_only": {k: {"N": v["N"], "S": v["S"], "D": v["D"], "err_pct": round(100 * (v["S"] + v["D"]) / max(1, v["N"]), 3)} for k, v in by_freq.items()},
        "worst_speakers": spk[:10], "best_speakers": spk[-5:],
        "concentration": {"utts": len(se), "zero_error_utts": sum(1 for x in se if x == 0),
                          "top10pct_share": round(sum(se[: max(1, len(se) // 10)]) / max(1, sum(se)), 3)},
        "top_subs": conf.most_common(40), "top_dels": dels.most_common(20), "top_ins": ins.most_common(20),
    }


def main() -> None:
    out = sys.argv[1]
    wc = Counter()
    with open(TRAIN_TXT, encoding="utf-8") as fh:
        for ln in fh:
            wc.update(w.upper() for w in normalise(ln))
    res = {}
    for spec in sys.argv[2:]:
        name, path = spec.split("=", 1)
        rows = [json.loads(x) for x in open(path, encoding="utf-8") if x.strip()]
        res[name] = analyse(rows, wc)
        print(name, json.dumps(res[name]["pooled"]), json.dumps(res[name]["by_subset"]), flush=True)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
