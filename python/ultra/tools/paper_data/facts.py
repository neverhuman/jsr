"""Page-1 facts, in the order the abstract and the README lead with them: cost, watts, capacity, silence,
the endless stream, the same words everywhere, and the honest line on accuracy.

The numbers come from the evidence rows (Figure 1's group) and the always-on receipts (alwayson.py), so the
abstract (\\Abs* macros), the results (\\Ao* macros) and the README bullets print the same values. Two rows of
ours can appear: the headline row drawn in Figure 1 (systems.json headline.ours) and the row the always-on
tests ran on (headline.always_on). Every number is printed with the checkpoint it was measured on; a
fact whose receipt or row is missing says that it is pending.
"""

from __future__ import annotations

from pathlib import Path
import math

from . import alwayson as ao
from .cold import speed_text
from .common import fmt
from .evidence import HEADLINE, derived
from .records import Records
from .words import checkpoint as _ck, md, name as _name, names as _names, signed as _sg, tex

HOURS_PER_YEAR = 8766.0


def add_facts(root: Path, rows: list[dict], headline: dict, rec: Records) -> dict:
    """Record the facts (TeX); return the README bullets (Markdown) in the same order, and Figure 1's message."""
    group = [r for r in rows if (r["machine"], r["threads"]) == HEADLINE and not r.get("anchor")]
    drawn = [r for r in group if r["plot"]]
    head = (next((r for r in drawn if r["variant"] == headline.get("ours")), None)
            or next((r for r in drawn if r["role"] == "ours"), None))
    aon = next((r for r in group if r["variant"] == headline.get("always_on")), None) or head
    others = [r for r in drawn if r["role"] == "comparator"]
    bullets: list[str] = []

    def put(rid: str, text: str, src: dict, what: str, workload: str, raw=None, unit: str = "sentence") -> None:
        rec.add(rid, text, raw=raw, unit=unit, label="exploratory", configuration=what, workload=workload,
                workload_digest=src.get("digest", "pending"), source=src.get("source", "pending"))

    def rowsrc(*rs: dict) -> dict:
        return {"source": "; ".join(f"docs/paper/data/evidence/{r['_where']} id={r['id']}" for r in rs),
                "digest": rs[0]["workload_digest"]} if rs else {}

    def fact(rid: str, label: str, abstract: str | None, bullet: str | None, src: dict, what: str, wl: str) -> None:
        put(f"Abs{rid}", tex(abstract) if abstract else f"{label}: pending.", src, f"abstract: {what}", wl)
        bullets.append(f"- **{label}.** " + (md(bullet) if bullet else "pending (no admitted receipt yet)."))

    panel = head["workload"] if head else "pending"
    # ---- 1. cost: the headline row against every other system on its host
    if head:
        h = derived(head)
        peers = sorted((r for r in others if r["host"] == head["host"]), key=lambda r: -derived(r)["speed"])
        vs = [f"{speed_text(derived(r)['speed'])}\u00d7 for {_name(r)}" + (" (whole clip)" if r["class"] ==
              "whole_utterance" else "") for r in peers]
        share, x = fmt(100.0 * h["cpu"], 1), speed_text(h["speed"])
        tail = f", against {_names(vs)}" if vs else ""
        m3 = ao.m3(root) if aon and aon["variant"] == ao.XISA_VARIANT else None
        fact("Cost", "Cost", f"One live stream of ULTRA costs {share}% of one CPU core: {x}\u00d7 real time on one "
             f"quiet Zen\u00a02 core{tail}.",
             f"{share}% of one CPU core per live stream: {_name(head)} ({_ck(head)}) runs at {x}\u00d7 real time "
             f"on one quiet Zen 2 core{tail}." + (f" {'The previous checkpoint, ' + _ck(aon) + ',' if aon is not head else _ck(aon)}"
                                                  f" runs at {speed_text(m3['speed'])}\u00d7 on one Apple M3 Ultra core."
                                                  if m3 else ""), rowsrc(head, *peers),
             "share of one core and speed of the headline row and of every other system on its host", panel)
        if m3:
            put("AoMthreeSpeed", speed_text(m3["speed"]), m3, f"speed of {_ck(aon)} on one Apple M3 Ultra core "
                "(macOS, thread count 1, no core pinning)", "730-utterance DEV panel", raw=m3["speed"], unit="x")
    else:
        fact("Cost", "Cost", None, None, {}, "pending", panel)
    # ---- 2. watts: the always-on row against every other system with a same-run energy line on its host
    wrows = sorted((r for r in others if r.get("watts") is not None and aon and r["host"] == aon["host"]),
                   key=lambda r: r["watts"])
    if aon and aon.get("watts") is not None and wrows:
        w = aon["watts"]
        ratio = [(fmt(r["watts"] / w, 1), r) for r in wrows]
        kwh = fmt(w * HOURS_PER_YEAR / 1000.0, 1)
        same = abs(derived(aon)["speed"] / derived(head)["speed"] - 1.0) if head else 0.0
        who = ("" if aon is head else f"Our always-on tests ran on the previous checkpoint, {_ck(aon)} (same "
               f"architecture; speed within {max(1, math.ceil(100.0 * same))}% of {_ck(head)}): ")
        ends = [ratio[0], ratio[-1]] if len(ratio) > 1 else ratio[:1]
        fact("Watts", "Watts", f"{who}{'one' if who else 'One'} stream draws {fmt(w, 2)}\u2009W, "
             + " and ".join(f"{q}\u00d7 less than {_name(r)}" for q, r in ends) + ".",
             f"{fmt(w, 2)} W per always-on stream ({_ck(aon)}; core energy registers, same run), about {kwh} kWh per "
             "stream-year: " + _names([f"{q}× less than {_name(r)} ({fmt(r['watts'], 2)} W)" for q, r in ratio]) + ".",
             rowsrc(aon, *wrows), "watts per always-on stream of the always-on row against the other systems", panel)
        put("AoWatts", fmt(w, 2), rowsrc(aon), f"W per always-on stream, {_ck(aon)}", panel, raw=w, unit="W")
        put("MoneyOursWatts", "\\relax" if aon is head else tex(
            f"No watts are drawn for ours: {_ck(head)} has no energy measurement yet; {_ck(aon)}, the previous "
            f"checkpoint, draws {fmt(w, 2)}\u2009W."), rowsrc(aon), "Figure 1 caption: watts of ours", panel)
        put("AoKwh", kwh, rowsrc(aon), f"kWh per stream-year, {_ck(aon)} (W x 8766 h)", panel,
            raw=w * HOURS_PER_YEAR / 1000.0, unit="kWh")
    else:
        fact("Watts", "Watts", None, None, {}, "pending", panel)
    # ---- 3. capacity on one core
    cap = ao.capacity(root)
    if cap and aon and aon["variant"] in cap["systems"]:
        o, z = cap["systems"][aon["variant"]], cap["systems"]["r0-int8"]
        lag = f"{fmt(o['lo'], 0)}\u2013{fmt(o['hi'], 0)}\u2009ms"
        zlag = f"{fmt(z['lo'], 0)}\u2013{fmt(z['hi'], 0)}\u2009ms"
        flat = f"flat from {o['n_min']} to {o['n']} streams" if o["flat"] else f"over {o['n_min']} to {o['n']} streams"
        wl = "the first 22 panel utterances (128 s) per stream, paced at real time, all streams on one core"
        fact("Capacity", "Capacity", f"One core carries {o['n']} such streams at a 95th-percentile word lag of {lag}, "
             f"{flat}; Zipformer carries {z['n']} at {zlag}.",
             f"{o['n']} live streams on one core ({_ck(aon)}), 95th-percentile word lag {lag}, {flat}; Zipformer holds "
             f"{z['n']} streams at {zlag}. Measured on one core; from the CPU cost the ceilings are about "
             f"{fmt(o['ceiling'], 0)} and {fmt(z['ceiling'], 0)} streams.", cap, "capacity test", wl)
        put("AoCapacity", tex(f"With {o['n']} live streams paced at real time on one core of the quiet host, "
                              f"{_ck(aon)} keeps a 95th-percentile word lag of {lag} ({flat}) and draws "
                              f"{fmt(o['w'], 2)}\u2009W per stream; Zipformer holds {z['n']} streams at {zlag} and "
                              f"{fmt(z['w'], 2)}\u2009W per stream. This is measured on one core. From the CPU cost "
                              f"per stream the core's ceilings are about {fmt(o['ceiling'], 0)} and "
                              f"{fmt(z['ceiling'], 0)} streams; no number of streams per server is claimed."),
            cap, "capacity test, results sentence", wl)
        put("AoCapacityStreams", str(o["n"]), cap, f"live streams of {_ck(aon)} on one core", wl, raw=o["n"],
            unit="streams")
    else:
        fact("Capacity", "Capacity", None, None, {}, "pending", "pending")
    # ---- 4. silence
    s = ao.silence(root)
    if s and aon and aon["variant"] == ao.XISA_VARIANT:
        levels = f"{_sg(s['levels'][0], 0)} to {_sg(s['levels'][-1], 0)}\u2009dBFS"
        words = "no word" if s["words"] == 0 else f"{s['words']} words"
        gaps = "none" if s["gap_words"] == 0 else str(s["gap_words"])
        wl = "ten 1-hour non-speech blocks and the pauses of the endless stream's first hour (eval-lane)"
        fact("Silence", "Silence", f"In {fmt(s['hours'], 0)}\u2009h of silence and synthetic noise it emitted {words}.",
             f"{s['words']} words in {fmt(s['hours'], 0)} hours of digital silence and synthetic white, pink and "
             f"brown noise at {levels} ({s['blocks']} one-hour blocks, {_ck(aon)}), and {s['gap_words']} in the "
             f"{fmt(s['gap_hours'], 2)} h of pauses in the first hour of a continuous stream.", s, "non-speech", wl)
        put("AoSilence", tex(f"Decoded as one stream from a fresh state, each of {s['blocks']} one-hour blocks of "
                             f"digital silence and of white, pink and brown noise at {levels} gave an empty "
                             f"transcript: {s['words']} words in {fmt(s['hours'], 0)}\u2009h. In the first hour of the "
                             f"continuous stream, the {fmt(s['gap_hours'], 2)}\u2009h of pauses held {gaps}."),
            s, "non-speech, results sentence", wl)
        put("AoSilenceHours", fmt(s["hours"], 0), s, "hours of non-speech decoded", wl, raw=s["hours"], unit="h")
    else:
        fact("Silence", "Silence", None, None, {}, "pending", "pending")
    # ---- 5. the endless stream: memory over hours, and accuracy in one continuous hour
    e, st = ao.endless(root), ao.stream(root)
    if e and st and aon and aon["variant"] == ao.XISA_VARIANT:
        rep, t0, pen, left = st["replay"], st["t0"], st["penalty"], st["left"]
        mem = f"{fmt(e['lo'], 1)}\u2013{fmt(e['hi'], 1)}\u2009MB"
        rr = f"T\u00a0=\u00a0{fmt(st['t_s'], 1)}\u2009s, R\u00a0=\u00a0{fmt(st['r_s'], 1)}\u2009s"
        wl = f"eval-lane's endless stream ({fmt(e['total'], 1)} h, every DEV utterance once, seeded pauses)"
        src = {"source": f"{e['source']}; {st['source']}", "digest": e["digest"]}
        fact("Endless", "Endless", f"Its memory stayed at {mem} over the first {fmt(e['hours'], 1)}\u2009h of a "
             f"{fmt(e['total'], 1)}\u2009h continuous stream, and with reset-with-replay the stream's first hour scores "
             f"{fmt(rep['observed_b'], 2)}% WER against {fmt(rep['observed_a'], 2)}% for the same utterances decoded "
             "one by one.",
             f"Memory flat at {mem} over the first {fmt(e['hours'], 1)} h of a {fmt(e['total'], 1)} h continuous stream "
             f"({_ck(aon)}; the run continues). With reset-with-replay ({rr}), one continuous stream matches isolated "
             f"decoding on the first stream hour: {fmt(rep['observed_b'], 2)}% against {fmt(rep['observed_a'], 2)}% "
             f"WER, {st['utts']} utterances ({fmt(t0['observed_b'], 2)}% without the reset).", src, "endless", wl)
        put("AoEndless", tex(
            f"Over the first {fmt(e['hours'], 1)}\u2009h of the {fmt(e['total'], 1)}\u2009h stream (the run continues), "
            f"the resident memory of {_ck(aon)} stayed between {mem}. In the stream's first hour ({st['utts']} "
            f"utterances, {fmt(st['minutes'], 0)}\u2009min) one continuous stream scored {fmt(pen['observed_b'], 2)}% "
            f"pooled WER against {fmt(pen['observed_a'], 2)}% with each utterance decoded alone "
            f"({_sg(pen['delta'], 2)}\u2009pp, bootstrap bounds {_sg(pen['lower'], 2)} to {_sg(pen['upper'], 2)}). "
            f"Of that, {_sg(left['delta'], 2)}\u2009pp comes from the pauses' audio in the encoder's left context "
            "(each utterance decoded alone with its preceding pause). Resetting the state after "
            f"{fmt(st['t_s'], 1)}\u2009s without output and replaying the last {fmt(st['r_s'], 1)}\u2009s brings the "
            f"continuous hour to {fmt(rep['observed_b'], 2)}% ({_sg(rep['delta'], 2)}\u2009pp, "
            f"{_sg(rep['lower'], 2)} to {_sg(rep['upper'], 2)}; {fmt(t0['observed_b'], 2)}% without the reset in the "
            f"same build), with {st['replay_gap_words']} false words in the pauses. The {fmt(e['total'], 1)}\u2009h run "
            "with the reset on is still going."), src, "endless, results sentence", wl)
        put("AoStreamHours", fmt(e["total"], 1), src, "length of the endless stream", wl, raw=e["total"], unit="h")
        put("AoEndlessHours", fmt(e["hours"], 1), src, "hours of the endless run done (snapshot)", wl, raw=e["hours"],
            unit="h")
    else:
        fact("Endless", "Endless", None, None, {}, "pending", "pending")
    # ---- 6. the same words on three instruction sets, and INT8 against FP32 on full dev
    sw, i8 = ao.same_words(root), (ao.int8(root, head["variant"]) if head else None)
    if sw and i8 and aon and aon["variant"] == ao.XISA_VARIANT:
        isas = _names(sw["isas"])
        i8s = (f"on full dev ({i8['utts']:,} utterances) {_ck(head)} scores {fmt(i8['observed_b'], 2)}% pooled WER in "
               f"the INT8 engine against {fmt(i8['observed_a'], 2)}% in FP32 PyTorch ({_sg(i8['delta'], 2)}\u2009pp, "
               f"bounds {_sg(i8['lower'], 2)} to {_sg(i8['upper'], 2)})")
        src = {"source": f"{sw['source']}; {i8['source']}", "digest": sw["digest"]}
        bullets.append(md(f"- **Same words everywhere.** {sw['same']} of {sw['total']} panel transcripts of {_ck(aon)} "
                          f"are identical on {isas}; INT8 costs nothing: {i8s}."))
        m3 = ao.m3(root)
        on_m3 = f" (one M3 core runs it at {speed_text(m3['speed'])}\u00d7 real time, without core pinning)" if m3 else ""
        put("AoSameWords", tex(f"The engine gives the same words on every instruction set: {sw['same']} of "
                               f"{sw['total']} panel transcripts of {_ck(aon)} are identical on {isas}{on_m3}. Its INT8 "
                               f"weights cost no accuracy: {i8s}."), src, "cross-ISA identity and INT8 on full dev",
            "730-utterance DEV panel; full dev")
    else:
        bullets.append("- **Same words everywhere.** pending (no admitted receipt yet).")
    # ---- 7. the honest line: accuracy
    if head and others:
        h = derived(head)
        stream_best = min((r for r in others if r["class"] == "streaming"), key=lambda r: derived(r)["wer_clean"],
                          default=None)
        whole_best = min((r for r in others if r["class"] == "whole_utterance"), key=lambda r: derived(r)["wer_clean"],
                         default=None)
        fast = max((r for r in others if r["class"] == "streaming" and r["host"] == head["host"]),
                   key=lambda r: derived(r)["speed"], default=None)
        pick = [r for r in dict.fromkeys(x["id"] for x in (fast, stream_best, whole_best) if x)]
        pick = sorted((r for r in others if r["id"] in pick), key=lambda r: -derived(r)["wer_clean"])
        better = sum(derived(r)["wer_clean"] < h["wer_clean"] for r in others)
        lm = ((rec.rows["LmEngineGain"]["value"], rec.rows["LmHeldOutGain"]["value"], "H22ctl")
              if "LmEngineGain" in rec.rows and "LmHeldOutGain" in rec.rows else None)
        vs = _names([f"{fmt(derived(r)['wer_clean'], 2)}% for {_name(r)}" for r in pick])
        rest = f" ({head['detail'].split(', ', 1)[1]})" if ", " in (head.get("detail") or "") else ""
        fact("Accuracy", "Where we trail", f"Accuracy is where we trail: {_ck(head)} scores {fmt(h['wer_clean'], 2)}% "
             f"dev-clean WER{rest}, against {vs}.",
             f"{_name(head)} ({head.get('detail') or _ck(head)}) scores {fmt(h['wer_clean'], 2)}% WER on the "
             f"panel's dev-clean half, against {vs}; {better} of the {len(others)} other systems drawn are more "
             "accurate." + (f" An optional n-gram LM lowers the pooled WER of {lm[2]} by {lm[0]} points in the engine "
                            f"({lm[1]} held out in PyTorch); no row here uses it." if lm else ""),
             rowsrc(head, *pick), "accuracy of the headline row against chosen comparators", panel)
        top = max(drawn, key=lambda r: derived(r)["speed"]) is head
        rank = f"the fastest of the {len(drawn)} systems" if top else "one of the faster systems"
        ratio = f", {fmt(h['speed'] / derived(fast)['speed'], 1)}\u00d7 the next streaming system on the same core" if (
            top and fast) else ""
        see = (f"{_name(head)} ({head.get('detail') or _ck(head)}) is {rank}{ratio}, and {better} of the {len(others)} "
               f"others are more accurate.")
        put("MoneyHeadline", tex(see), rowsrc(head, *([fast] if fast else [])), "Figure 1: what to see", panel)
        figure = md(see)
    else:
        fact("Accuracy", "Where we trail", None, None, {}, "pending", panel)
        put("MoneyHeadline", "Speed against accuracy on one CPU core, every system measured here.", {}, "pending",
            panel)
        figure = "Speed against accuracy on one CPU core, every system measured here."
    if aon:
        put("AoCheckpoint", _ck(aon), rowsrc(aon), "the checkpoint the always-on tests ran on", panel, unit="name")
    for rid in ("MoneyOursWatts", "AoCheckpoint", "AoMthreeSpeed", "AoWatts", "AoKwh", "AoCapacity",
                "AoCapacityStreams", "AoSilence", "AoEndless", "AoSameWords", "AoSilenceHours", "AoStreamHours",
                "AoEndlessHours"):
        if rid not in rec.rows:  # defined either way, so the sections build; the value says what is missing
            put(rid, "\\relax" if rid == "MoneyOursWatts" else "pending", {}, "pending: no admitted receipt or row yet",
                "pending")
    return {"bullets": bullets, "figure": figure}
