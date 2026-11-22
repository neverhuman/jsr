"""Numbers and sentences the text quotes from the measured comparison (Figure 1's group).

The headline row of ours is named in data/systems.json ("headline": {"ours": <variant>}). Accuracy and
memory do not depend on the host, so they are compared with every system drawn. Speed does: it is
compared only with systems measured on the same host as the headline row, and when there is none yet
the generated sentence says so instead of printing a cross-host ratio. Sentences about measurements
that do not exist yet (word lag, watts, eight cores, other machines) say so as well.
"""

from __future__ import annotations

from .cold import ENERGY_WORDS, speed_text
from .common import HOST_NAMES, MACHINES, PaperDataError, fmt
from .evidence import HEADLINE, derived
from .records import Records

NOTHING = "\\relax"  # an empty sentence: prints nothing


def _names(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _name(r: dict) -> str:
    return r.get("text") or r["short"]


def _pending(rows: list[dict], say) -> None:
    eight = [r for r in rows if r["machine"] == HEADLINE[0] and r["threads"] == 8 and not r.get("anchor")]
    away = sorted({r["machine"] for r in rows if r["machine"] != HEADLINE[0]})
    say("ColdEightSentence", f"{len(eight)} rows have also been measured with eight pinned cores." if eight else
        "The eight-core budget has not been measured yet.", "eight-core rows")
    say("ColdMachinesSentence", "Rows from " + _names([MACHINES[m]["name"] for m in away]) + " have been admitted."
        if away else "Rows from a second x86 machine with AVX-512 VNNI and from an Apple M-series machine are being "
        "measured; none is reported yet.", "rows from other machines")


def add_comparison(rows: list[dict], headline: dict, rec: Records) -> None:
    group = [r for r in rows if (r["machine"], r["threads"]) == HEADLINE and r["plot"]]
    for r in group:
        r["_d"] = derived(r)
    ours = [r for r in group if r["role"] == "ours"]
    others = [r for r in group if r["role"] == "comparator"]
    first = rows[0] if rows else None
    note = dict(label="reported", workload=first["workload"] if first else "pending (no evidence rows yet)",
                workload_digest=first["workload_digest"] if first else "pending",
                source="docs/paper/data/evidence/*.jsonl")

    def count(rid: str, value: int, what: str) -> None:
        rec.add(rid, str(value), raw=value, unit="rows", configuration=f"number of {what}", **note)

    def say(rid: str, text: str, what: str) -> None:
        rec.add(rid, text, unit="sentence", configuration=f"generated: {what}", **note)

    count("ColdComparatorCount", len(others), "other systems drawn in Figure 1")
    _pending(rows, say)
    subset = [r for r in group if r["timing_panel"] != "full"]
    say("ColdSubsetSentence", NOTHING if not subset else f"{len(subset)} slow system"
        f"{'s are' if len(subset) != 1 else ' is'} timed on a 73-utterance subset of the panel "
        f"({_names([_name(r) for r in subset])}); their WER is from the full panel.", "rows timed on the subset")
    if not ours or not others:
        for rid in ("ColdAbstractSentence", "ColdHeadSentence", "ColdSpeedSentence", "ColdAccuracySentence",
                    "ColdMemorySentence", "ColdOrderSentence", "ColdLagSentence", "ColdWattsSentence"):
            say(rid, "The measured comparison is pending." if rid == "ColdHeadSentence" else NOTHING, "pending")
        say("ColdHeadDetail", "no row of ours is drawn yet", "pending")
        return
    want = headline.get("ours")
    head = (next((r for r in ours if r["variant"] == want), None) if want
            else min(ours, key=lambda r: r["_d"]["wer_clean"]))
    if head is None:
        raise PaperDataError(f"systems.json names the headline row {want!r}, which is not drawn in Figure 1")
    h, n = head["_d"], len(others)

    def src(*rs: dict) -> dict:
        return dict(label=head["label"], workload=head["workload"], workload_digest=head["workload_digest"],
                    source="; ".join(f"docs/paper/data/evidence/{r['_where']} id={r['id']}" for r in rs))

    def num(rid: str, value: float, text: str, unit: str, what: str, *rs: dict) -> None:
        rec.add(rid, text, raw=value, unit=unit, configuration=what, **src(*rs))

    host = HOST_NAMES.get(head["host"], head["host"])
    rec.add("ColdHeadName", _name(head), unit="name", configuration="the headline row of ours (data/systems.json)",
            **src(head))
    rec.add("ColdHeadDetail", head.get("detail") or head["short"], unit="name", configuration="the headline row of "
            "ours: which checkpoint and step", **src(head))
    for rid, key, nd, unit in (("Cpu", "cpu", 3, "CPU-s/audio-s"), ("Wer", "wer", 2, "% WER, pooled"),
                               ("WerClean", "wer_clean", 2, "% WER, dev-clean half"),
                               ("WerOther", "wer_other", 2, "% WER, dev-other half"),
                               ("Rss", "rss_mb", 0, "MB peak resident memory"), ("Model", "model_mb", 0, "MB of model files")):
        num(f"ColdHead{rid}", h[key], fmt(h[key], nd), unit, "the headline row of ours", head)
    num("ColdHeadSpeed", h["speed"], speed_text(h["speed"]), "audio-s per CPU-s (x real time, one core)",
        "the headline row of ours", head)
    say("ColdHeadSentence", f"\\ultra{{}} ({head.get('detail') or head['short']}) costs {fmt(h['cpu'], 3)} CPU-seconds "
        f"per audio-second on host {host}, which is {speed_text(h['speed'])}$\\times$ real time on one core, at "
        f"{fmt(h['wer_clean'], 2)}\\% WER on dev-clean ({fmt(h['wer_other'], 2)}\\% on dev-other, {fmt(h['wer'], 2)}\\% "
        f"pooled), and peaks at {fmt(h['rss_mb'], 0)}\\,MB of memory.", "the headline row of ours")

    # ---- speed: only against systems measured on the same host
    peers = [r for r in others if r["host"] == head["host"]]
    stream = [r for r in peers if r["class"] == "streaming"]
    if stream:
        fast = max(stream, key=lambda r: r["_d"]["speed"])
        f = fast["_d"]
        ratio = h["speed"] / f["speed"]
        num("ColdHeadSpeedOverFast", ratio, fmt(ratio, 1), "x", "speed of ours / speed of the fastest other streaming "
            "system measured on the same host", head, fast)
        rounds = ""
        if head["session"] == fast["session"] and len(head["cpu_reps"]) == len(fast["cpu_reps"]):
            by_round = sorted(a / b for a, b in zip(fast["cpu_reps"], head["cpu_reps"]))
            rounds = f" by the medians and {fmt(by_round[0], 1)} to {fmt(by_round[-1], 1)}$\\times$ round by round"
        slower = sum(r["_d"]["speed"] < h["speed"] for r in peers)
        rank = ("the fastest of them" if slower == len(peers) else f"faster than {slower} of them")
        say("ColdSpeedSentence", f"On the same host the fastest other streaming system, {_name(fast)}, runs at "
            f"{speed_text(f['speed'])}$\\times$ real time: \\ultra is {fmt(ratio, 1)}$\\times$ faster{rounds}. "
            f"{len(peers)} other {'system has' if len(peers) == 1 else 'systems have'} a row from that host, and "
            f"\\ultra is {rank}.", "speed against same-host systems")
        say("ColdAbstractSentence", f"Measured cold on one pinned core, model load included, \\ultra decodes at "
            f"{speed_text(h['speed'])}$\\times$ real time at {fmt(h['wer_clean'], 2)}\\% dev-clean WER, against "
            f"{speed_text(f['speed'])}$\\times$ and {fmt(f['wer_clean'], 2)}\\% for {_name(fast)}, the fastest other "
            f"streaming system on the same host.", "abstract")
    else:
        say("ColdSpeedSentence", f"No other system has a row from host {host} yet, so no speed ratio is given: the "
            "other points come from a shared host that ran other jobs, where every system is slower.",
            "no same-host comparator yet")
        say("ColdAbstractSentence", f"Measured cold on one pinned core, model load included, \\ultra decodes at "
            f"{speed_text(h['speed'])}$\\times$ real time at {fmt(h['wer_clean'], 2)}\\% dev-clean WER; same-host rows "
            f"of the other systems are being measured.", "abstract, no same-host comparator yet")

    # ---- accuracy and memory: host-independent, against every system drawn
    best = min(others, key=lambda r: r["_d"]["wer_clean"])
    b = best["_d"]
    better = sum(r["_d"]["wer_clean"] < h["wer_clean"] for r in others)
    gap = h["wer_clean"] - b["wer_clean"]
    num("ColdHeadWerGapBest", gap, fmt(gap, 2), "pp WER", "dev-clean WER of ours minus the lowest dev-clean WER among "
        "the other systems drawn", head, best)
    count("ColdAccuracyCount", better, "other systems drawn in Figure 1 with a lower dev-clean WER than ours")
    who = ("Every one of the other systems" if better == n else "None of the other systems" if better == 0
           else f"{better} of the {n} other systems")
    say("ColdAccuracySentence", f"{who} {'is' if better in (0, n) else 'are'} more accurate on dev-clean than \\ultra. "
        f"The most accurate, {_name(best)}, scores {fmt(b['wer_clean'], 2)}\\% ({fmt(b['wer_other'], 2)}\\% on "
        f"dev-other, {fmt(b['wer'], 2)}\\% pooled), {fmt(gap, 2)} points below \\ultra.", "accuracy against all drawn")
    light = min(others, key=lambda r: r["_d"]["rss_mb"])
    heavy = max(others, key=lambda r: r["_d"]["rss_mb"])
    ratio = light["_d"]["rss_mb"] / h["rss_mb"]
    num("ColdLightRssOverHead", ratio, fmt(ratio, 1), "x", "peak memory of the lightest other system / ours", head, light)
    say("ColdMemorySentence", f"The other systems need between {fmt(light['_d']['rss_mb'], 0)}\\,MB ({_name(light)}) and "
        f"{fmt(heavy['_d']['rss_mb'], 0)}\\,MB ({_name(heavy)}) of memory, {fmt(ratio, 1)} to "
        f"{fmt(heavy['_d']['rss_mb'] / h['rss_mb'], 1)} times as much as \\ultra.", "memory against all drawn")
    clean = sorted(group, key=lambda r: r["_d"]["wer_clean"])
    other = sorted(group, key=lambda r: r["_d"]["wer_other"])
    moved = [_name(r) for i, r in enumerate(clean) if other[i] is not r]
    say("ColdOrderSentence", "The dev-other half ranks the systems in the same order." if not moved else
        f"The dev-other half ranks {len(moved)} of them differently ({_names(moved)}).",
        "whether dev-other orders the drawn systems as dev-clean does")

    # ---- measurements that may not exist yet
    watts = [r for r in group if r["watts"] is not None and r["host"] == head["host"]]
    if head["watts"] is not None and len(watts) > 1:
        rest = sorted((r for r in watts if r is not head), key=lambda r: r["watts"])
        say("ColdWattsSentence", f"One always-on stream of \\ultra{{}} costs {fmt(head['watts'], 2)}\\,W, against "
            f"{fmt(rest[0]['watts'], 2)}\\,W for {_name(rest[0])}"
            + (f" and {fmt(rest[-1]['watts'], 2)}\\,W for {_name(rest[-1])}" if len(rest) > 1 else "")
            + f" ({ENERGY_WORDS[head['energy_method']]}, same host).", "energy rows")
    else:
        say("ColdWattsSentence", "Energy per audio-second (the watts one always-on stream costs) is being measured; no "
            "value has been admitted yet.", "no energy row is admitted yet")
    where = head.get("lag_host")
    lag = [r for r in group if r.get("lag_p50_ms") is not None and r.get("lag_host") == where]
    if head.get("lag_p50_ms") is not None and len(lag) > 1:
        rest = sorted((r for r in lag if r is not head), key=lambda r: r["lag_p50_ms"])
        kind = "the quiet host" if where == MACHINES[HEADLINE[0]]["primary_host"] else "the shared host"
        say("ColdLagSentence", f"With the input paced at real time on one core of {kind} "
            f"{HOST_NAMES.get(where, where)}, the median word lag of \\ultra{{}} is {fmt(head['lag_p50_ms'], 0)}\\,ms "
            f"(95th percentile {fmt(head['lag_p95_ms'], 0)}\\,ms); "
            + (f"{_name(rest[0])} measured there has {fmt(rest[0]['lag_p50_ms'], 0)}\\,ms." if len(rest) == 1 else
               f"the other streaming systems measured there range from {fmt(rest[0]['lag_p50_ms'], 0)}\\,ms "
               f"({_name(rest[0])}) to {fmt(rest[-1]['lag_p50_ms'], 0)}\\,ms ({_name(rest[-1])})."),
            "lag rows measured on one host")
    else:
        say("ColdLagSentence", "Word lag (end to end, with the input paced at real time) is being measured; no value "
            "has been admitted yet.", "no lag row is admitted yet")
