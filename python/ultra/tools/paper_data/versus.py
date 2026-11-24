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
    return "statistical tie"


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
        put(f"VsLo{t}", tex(_sg(p["lo"], 2)), p["lo"], "pp WER", f"{q['mode']}: 95% lower bound", p["src"], p["digest"])
        put(f"VsHi{t}", tex(_sg(p["hi"], 2)), p["hi"], "pp WER", f"{q['mode']}: 95% upper bound", p["src"], p["digest"])
        put(f"VsSpeedX{t}", fmt(o["speed"] / z["speed"], 2), o["speed"] / z["speed"], "x",
            f"{q['mode']} single-core speed / Zipformer INT8", evs, comp["workload_digest"])
        put(f"VsRssPct{t}", fmt(100.0 * o["rss_mb"] / z["rss_mb"], 0), 100.0 * o["rss_mb"] / z["rss_mb"], "%",
            f"{q['mode']} peak RSS as % of Zipformer INT8", evs, comp["workload_digest"])
        if o["watts"] is not None and z["watts"] is not None:
            put(f"VsEnergyPct{t}", fmt(100.0 * o["watts"] / z["watts"], 0), 100.0 * o["watts"] / z["watts"], "%",
                f"{q['mode']} active core energy per audio-second as % of Zipformer INT8", evs, comp["workload_digest"])
        verdict = {"lower WER": "lower than", "statistical tie": "a statistical tie with", "higher WER": "higher than"}
        put(f"VsVerdict{t}", verdict[_verdict(p)], None, "text", f"{q['mode']}: paired verdict vs Zipformer INT8",
            p["src"], p["digest"])
        put(f"VsCpuPct{t}", fmt(100.0 * z["speed"] / o["speed"], 0), 100.0 * z["speed"] / o["speed"], "%",
            f"{q['mode']} CPU-s per audio-s as % of Zipformer INT8", evs, comp["workload_digest"])
        put(f"VsChunk{t}", str(q["chunk"]), q["chunk"], "ms", f"{q['mode']} live chunk", evs, comp["workload_digest"])

    def j(v) -> str:
        return "—" if v is None else fmt(v, 3)

    md = ["### Head-to-head with Zipformer INT8 (same 730 DEV utterances, same quiet Zen 2 core, cold, median of 3)", "",
          "| | Live chunk | Pooled WER (%) | vs. Zipformer, paired 95% bounds (pp) | 1-core speed (× RT) | "
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
    md += ["", "Δ is our pooled WER minus Zipformer's on the same utterances (negative = fewer errors); bounds from "
           "`ultra-score` (10,000 speaker-cluster replicates). Speed, energy and memory ratios compare rows timed on "
           "the same host in the same protocol. Energy is active core (CPU-domain) joules per second of audio, not "
           "wall power."]
    cols = [("Zipformer", None)] + [(f"\\ultra{{}} {q['mode']}", q) for q in pairs]

    def cell(q, f):
        return "---" if q is None else f(q)
    body = [
        ("Live chunk (ms)", lambda q: str(zc) if q is None else (str(q["chunk"]) if q["mode"] == "live" else f"{q['chunk']}\\textsuperscript{{a}}")),
        ("Pooled WER (\\%)", lambda q: fmt(z["wer"] if q is None else q["d"]["wer"], 2)),
        ("$\\Delta$ WER vs.\\ Zip.\\ (pp)", lambda q: cell(q, lambda q: tex(_sg(q["p"]["delta"], 2)))),
        ("\\quad paired 95\\% bounds", lambda q: cell(q, lambda q: f"[{tex(_sg(q['p']['lo'], 2))}, {tex(_sg(q['p']['hi'], 2))}]")),
        ("1-core speed ($\\times$ RT)", lambda q: speed_text(z["speed"] if q is None else q["d"]["speed"])),
        ("J per audio-s (core)", lambda q: j(z["watts"] if q is None else q["d"]["watts"])),
        ("Peak RSS (MB)", lambda q: fmt(z["rss_mb"] if q is None else q["d"]["rss_mb"], 0)),
    ]
    tx = [r"\begin{table}[t]", r"\centering",
          r"\caption{\textbf{Head-to-head with Zipformer INT8}: same 730 DEV utterances, same quiet Zen~2 core, cold, "
          r"median of three. $\Delta$ is paired (speaker-cluster bootstrap); negative means fewer errors. "
          r"\textsuperscript{a}Live stream plus a commit pass at reference utterance boundaries.}",
          r"\label{tab:versus}", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
          r"\begin{tabular}{@{}l" + "r" * len(cols) + r"@{}}", r"\toprule",
          " & ".join(["Metric"] + [c for c, _ in cols]) + r" \\", r"\midrule"]
    tx += [" & ".join([label] + [f(q) for _, q in cols]) + r" \\" for label, f in body]
    tx += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return {"markdown": "\n".join(md) + "\n", "table": "\n".join(tx) + "\n", "pairs": summary,
            "zip": {"wer": z["wer"], "speed": z["speed"], "watts": z["watts"], "rss_mb": z["rss_mb"], "chunk": zc}}
