"""Head-to-head: our release operating points against one matched comparator (Zipformer INT8).

Same 730 DEV utterances, same quiet Zen 2 host and cold protocol (the evidence rows), plus a paired
speaker-cluster bootstrap of the WER difference from the pinned `ultra-score` receipts named in
data/systems.json -> "versus". Every ratio is between rows of one host; nothing here is extrapolated.
Emits \\Vs* macros, data/table_versus.tex and the README head-to-head markdown.
"""

from __future__ import annotations

from pathlib import Path
import json
import re

from .cold import speed_text
from .common import DATA, PaperDataError, fmt, sha256_file
from .evidence import HEADLINE, derived
from .heat import SHOW_FINAL, cell as heat_cell, scale
from .records import Records
from .words import signed as _sg, tex

_CHUNK = re.compile(r"(\d+)\s*ms")


def _chunk_ms(row: dict) -> int:
    m = _CHUNK.search(row["configuration"])
    if not m:
        raise PaperDataError(f"versus: no chunk length (\"<n> ms\") in the configuration of {row['id']}")
    return int(m.group(1))


def _paired(root: Path, rel: str) -> dict:
    data = json.loads((root / DATA / rel).read_text(encoding="utf-8"))
    d = data["comparison"]["pooled_delta"]
    return {"a": 100.0 * data["system_a"]["pooled_wer"], "b": 100.0 * data["system_b"]["pooled_wer"],
            "delta": 100.0 * d["delta"], "lo": 100.0 * d["lower"], "hi": 100.0 * d["upper"],
            "utts": data["system_b"]["utterances"], "digest": sha256_file(root / DATA / rel), "src": str(DATA / rel)}


def _verdict(p: dict) -> str:
    if p["hi"] < 0:
        return "lower WER"
    if p["lo"] > 0:
        return "higher WER"
    return "inconclusive"


_TAGS = ("One", "Two", "Three", "Four")


def add_versus(root: Path, rows: list[dict], spec: dict, rec: Records) -> dict:
    """Return {"markdown": str, "table": str, "pairs": [...]}; empty strings when not configured."""
    if not spec:
        return {"markdown": "", "table": "", "pairs": []}
    group = {r["variant"]: r for r in rows if (r["machine"], r["threads"]) == HEADLINE and r["plot"]
             and r["host"] == "xbabe0" and not r["interim"] and r["timing_panel"] == "full"}
    comp = group.get(spec["comparator"])
    if comp is None:
        raise PaperDataError(f"versus: comparator {spec['comparator']!r} has no quiet full-panel H1 row")
    z = derived(comp)
    zc = _chunk_ms(comp)
    pairs = []
    for i, item in enumerate(spec["pairs"]):
        row = group.get(item["variant"])
        if row is None:
            raise PaperDataError(f"versus: {item['variant']!r} has no quiet full-panel H1 row")
        o = derived(row)
        p = _paired(root, item["paired"])
        if abs(p["a"] - z["wer"]) > 0.005 or abs(p["b"] - o["wer"]) > 0.005 or p["utts"] != 730:
            raise PaperDataError(f"versus: paired receipt {item['paired']} disagrees with the evidence rows")
        pairs.append({"row": row, "d": o, "p": p, "chunk": _chunk_ms(row), "mode": item["mode"], "tag": _TAGS[i]})

    def put(rid: str, value: str, raw, unit: str, what: str, src: str, dig: str) -> None:
        rec.add(rid, value, raw=raw, unit=unit, label="exploratory", configuration=what,
                workload=comp["workload"], workload_digest=dig, source=src)

    ev = f"docs/paper/data/evidence/{comp['_where']} id={comp['id']}"
    put("VsZipChunk", str(zc), zc, "ms", "Zipformer INT8 configured chunk", ev, comp["workload_digest"])
    for q in pairs:
        o, p, t, r = q["d"], q["p"], q["tag"], q["row"]
        evs = f"{ev}; docs/paper/data/evidence/{r['_where']} id={r['id']}"
        put(f"VsDelta{t}", tex(_sg(p["delta"], 2)), p["delta"], "pp WER",
            f"{q['mode']} minus Zipformer INT8, pooled 730-utt panel WER (paired)", p["src"], p["digest"])
        put(f"VsLo{t}", tex(_sg(p["lo"], 2)), p["lo"], "pp WER", f"{q['mode']}: one-sided 95% lower bound (5% bootstrap quantile)", p["src"], p["digest"])
        put(f"VsHi{t}", tex(_sg(p["hi"], 2)), p["hi"], "pp WER", f"{q['mode']}: one-sided 95% upper bound (95% bootstrap quantile)", p["src"], p["digest"])
        pr = r.get("paired_vs_anchor")  # rows timed in interleaved blocks: the median of the within-block ratios
        how = "median of the within-block ratios" if pr else "ratio of the two medians"
        speed_x = pr["speed"] if pr else o["speed"] / z["speed"]
        rss_pct = 100.0 * (pr["rss"] if pr else o["rss_mb"] / z["rss_mb"])
        put(f"VsSpeedX{t}", fmt(speed_x, 2), speed_x, "x",
            f"{q['mode']} single-core speed / Zipformer INT8 ({how})", evs, comp["workload_digest"])
        put(f"VsRssPct{t}", fmt(rss_pct, 0), rss_pct, "%",
            f"{q['mode']} peak RSS as % of Zipformer INT8 ({how})", evs, comp["workload_digest"])
        if o["watts"] is not None and z["watts"] is not None:
            en_pct = 100.0 * (pr["energy"] if pr else o["watts"] / z["watts"])
            put(f"VsEnergyPct{t}", fmt(en_pct, 0), en_pct, "%",
                f"{q['mode']} active core energy per audio-second as % of Zipformer INT8 ({how})", evs, comp["workload_digest"])
        verdict = {"lower WER": "lower than", "inconclusive": "not significantly different from", "higher WER": "higher than"}
        put(f"VsVerdict{t}", verdict[_verdict(p)], None, "text", f"{q['mode']}: paired verdict vs Zipformer INT8",
            p["src"], p["digest"])
        cpu_pct = 100.0 / speed_x
        put(f"VsCpuPct{t}", fmt(cpu_pct, 0), cpu_pct, "%",
            f"{q['mode']} CPU-s per audio-s as % of Zipformer INT8 ({how})", evs, comp["workload_digest"])
        put(f"VsChunk{t}", str(q["chunk"]), q["chunk"], "ms", f"{q['mode']} live chunk", evs, comp["workload_digest"])

    # Rows beside the two paired modes: the chunk-matched 320 ms live row and live greedy (same blocks, paired ratios),
    # and the prior release against the Zipformer of its own, earlier session (never against this session's blocks).
    by_variant = {r["variant"]: r for r in rows if (r["machine"], r["threads"]) == HEADLINE and r["host"] == "xbabe0"}
    for t, variant, what in (("Wide", "rc2-live-lm320", "live +LM, 320 ms"), ("Greedy", "rc2-live-greedy", "live greedy, 160 ms")):
        r = by_variant.get(variant)
        if r is None or "paired_vs_anchor" not in r:
            continue
        o, pr, evs = derived(r), r["paired_vs_anchor"], f"docs/paper/data/evidence/{r['_where']} id={r['id']}"
        for key, val, text, unit in (("SpeedX", pr["speed"], fmt(pr["speed"], 2), "x"), ("EnergyPct", 100 * pr["energy"], fmt(100 * pr["energy"], 0), "%"),
                                     ("RssPct", 100 * pr["rss"], fmt(100 * pr["rss"], 0), "%"), ("RssMb", o["rss_mb"], fmt(o["rss_mb"], 1), "MB"),
                                     ("Wer", o["wer"], fmt(o["wer"], 2), "% WER"), ("Speed", o["speed"], speed_text(o["speed"]), "x real time"),
                                     ("Watts", o["watts"], fmt(o["watts"], 3), "J per audio-s")):
            put(f"Vs{t}{key}", text, val, unit, f"{what}: {key} (same interleaved blocks as Zipformer INT8; ratios are medians of within-block ratios)",
                evs, comp["workload_digest"])
    for item in spec.get("prior", []):
        r = by_variant.get({"live": "rc1-ema-live", "final": "rc1-ema-final"}[item["mode"]])
        if r is None:
            continue
        o, p, t = derived(r), _paired(root, item["paired"]), item["mode"].capitalize()
        evs = f"docs/paper/data/evidence/{r['_where']} id={r['id']}"
        for key, val, text, unit in (("Wer", o["wer"], fmt(o["wer"], 2), "% WER"), ("Speed", o["speed"], speed_text(o["speed"]), "x real time"),
                                     ("Watts", o["watts"], fmt(o["watts"], 3), "J per audio-s"), ("RssMb", o["rss_mb"], fmt(o["rss_mb"], 0), "MB"),
                                     ("Delta", p["delta"], tex(_sg(p["delta"], 2)), "pp WER"), ("Lo", p["lo"], tex(_sg(p["lo"], 2)), "pp WER"),
                                     ("Hi", p["hi"], tex(_sg(p["hi"], 2)), "pp WER")):
            src = p["src"] if key in ("Delta", "Lo", "Hi") else evs
            put(f"VsPrior{t}{key}", text, val, unit, f"v3.0.0 {item['mode']} {key} (its own earlier session; paired against that session's Zipformer INT8)",
                src, p["digest"] if key in ("Delta", "Lo", "Hi") else comp["workload_digest"])

    def j(v) -> str:
        return "—" if v is None else fmt(v, 3)

    md = ["### Head-to-head with Zipformer INT8 (same 730 DEV utterances, same quiet Zen 2 core, cold, median of 3)", "",
          "| | Live chunk | Pooled WER (%) | vs. Zipformer, paired one-sided 95% bounds (pp) | 1-core speed (× RT) | "
          "J per audio-second (active core) | Peak RSS (MB) |",
          "|---|---:|---:|---|---:|---:|---:|",
          f"| Zipformer INT8 | {zc} ms | {fmt(z['wer'], 2)} | — | {speed_text(z['speed'])} | {j(z['watts'])} | "
          f"{fmt(z['rss_mb'], 0)} |"]
    summary = []
    for q in pairs:
        o, p, r = q["d"], q["p"], q["row"]
        bounds = f"{_sg(p['delta'], 2)} [{_sg(p['lo'], 2)}, {_sg(p['hi'], 2)}], {_verdict(p)}"
        chunk_md = f"{q['chunk']} ms" if q["mode"] == "live" else f"{q['chunk']} ms live + commit at reference utterance ends"
        md.append(f"| **{r['system']}** | {chunk_md} | {fmt(o['wer'], 2)} | {bounds} | "
                  f"{speed_text(o['speed'])} ({fmt(o['speed'] / z['speed'], 2)}×) | {j(o['watts'])}"
                  + (f" ({fmt(100.0 * o['watts'] / z['watts'], 0)}%)" if o["watts"] is not None and z["watts"] else "")
                  + f" | {fmt(o['rss_mb'], 0)} ({fmt(100.0 * o['rss_mb'] / z['rss_mb'], 0)}%) |")
        summary.append({"system": r["system"], "mode": q["mode"], "chunk": q["chunk"], "wer": o["wer"],
                        "delta": p["delta"], "lo": p["lo"], "hi": p["hi"], "verdict": _verdict(p),
                        "speed_x": o["speed"] / z["speed"], "speed": o["speed"],
                        "energy_pct": (100.0 * o["watts"] / z["watts"]) if o["watts"] is not None and z["watts"] else None,
                        "rss_pct": 100.0 * o["rss_mb"] / z["rss_mb"], "rss_mb": o["rss_mb"]})
    md += ["", "Δ is our pooled WER minus Zipformer's on the same utterances (negative = fewer errors); each bound is "
           "one-sided at 95% (together a central 90% interval) from `ultra-score` (10,000 speaker-cluster replicates). Speed, energy and memory ratios compare rows timed on "
           "the same host in the same protocol. Energy is active core (CPU-domain) joules per second of audio, not "
           "wall power."]
    cols = [("Zipformer", None)] + [("\\ultra{} live", q) for q in pairs if SHOW_FINAL or q["mode"] == "live"]  # final stays in prose

    def cell(q, f):
        return "---" if q is None else f(q)
    body = [
        ("Live chunk (ms)", lambda q: str(zc) if q is None else (str(q["chunk"]) if q["mode"] == "live" else f"{q['chunk']}\\textsuperscript{{a}}")),
        ("Pooled WER (\\%)", lambda q: fmt(z["wer"] if q is None else q["d"]["wer"], 2)),
        ("$\\Delta$ WER vs.\\ Zip.\\ (pp)", lambda q: cell(q, lambda q: tex(_sg(q["p"]["delta"], 2)))),
        ("\\quad one-sided 95\\% bounds", lambda q: cell(q, lambda q: f"[{tex(_sg(q['p']['lo'], 2))}, {tex(_sg(q['p']['hi'], 2))}]")),
        ("1-core speed ($\\times$ RT)", lambda q: speed_text(z["speed"] if q is None else q["d"]["speed"])),
        ("J per audio-s (core, R6)", lambda q: j(z["watts"] if q is None else q["d"]["watts"])),
        ("Peak RSS (MB)", lambda q: fmt(z["rss_mb"] if q is None else q["d"]["rss_mb"], 0)),
    ]
    tx = [r"\begin{table}[t]", r"\centering",
          r"\caption{\textbf{Head-to-head with Zipformer INT8}: same 730 DEV utterances (the selection panel; the "
          r"unselected-half gaps are in \cref{sec:versus}), same quiet Zen~2 core, same interleaved blocks, median of three. "
          r"Ratios are medians of within-block ratios. "
          r"$\Delta$ is paired (speaker-cluster bootstrap); negative means fewer errors. "
          r"Cell colour (speed, energy, memory): green is better, red worse; WER is judged by the paired $\Delta$ row, not by colour.}",
          r"\label{tab:versus}", r"\footnotesize", r"\setlength{\tabcolsep}{1.5pt}",
          r"\begin{tabular}{@{}l" + "r" * len(cols) + r"@{}}", r"\toprule",
          " & ".join(["Metric"] + [c for c, _ in cols]) + r" \\", r"\midrule"]
    heat_rows = {  # label -> (value of Zipformer, value of live, log, higher is better)
        "1-core speed ($\\times$ RT)": (lambda d: d["speed"], True, True),
        "J per audio-s (core, R6)": (lambda d: d["watts"], True, False), "Peak RSS (MB)": (lambda d: d["rss_mb"], True, False)}
    zd = z
    for label, f in body:
        cells = [f(q) for _, q in cols]
        if label in heat_rows:
            get, lg, hb = heat_rows[label]
            vals = [get(zd if q is None else q["d"]) for _, q in cols]
            cells = [heat_cell(c, t) for c, t in zip(scale(vals, log=lg, higher_better=hb), cells)]
        tx.append(" & ".join([label] + cells) + r" \\")
    tx += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return {"markdown": "\n".join(md) + "\n", "table": "\n".join(tx) + "\n", "pairs": summary,
            "zip": {"wer": z["wer"], "speed": z["speed"], "watts": z["watts"], "rss_mb": z["rss_mb"], "chunk": zc}}
