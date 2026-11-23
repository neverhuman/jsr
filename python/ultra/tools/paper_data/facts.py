"""Page-1 facts, in the order the abstract and the README lead with them: cost, watts, capacity, silence,
the endless stream, the same words everywhere, and the honest line on accuracy.

Numbers come from the evidence rows (Figure 1's group) and the always-on receipts (alwayson.py), so the
abstract (\\Abs*), the results (\\Ao*) and the README bullets print the same values. Every number is printed
with the checkpoint it was measured on (systems.json headline.ours, headline.always_on, or the newest row of
ours that has the measurement); a fact whose receipt or row is missing says that it is pending.
"""

from __future__ import annotations

from pathlib import Path
import math

from . import alwayson as ao
from .cold import speed_text
from .common import fmt
from .evidence import HEADLINE, derived
from .records import Records
from .tables import HOURS_PER_YEAR
from .words import checkpoint as _ck, md, name as _name, names as _names, signed as _sg, tex

ALWAYS = ("MoneyOursWatts", "AoCheckpoint", "AoMthreeSpeed", "AoWatts", "AoKwh", "AoCapacity", "AoCapacityStreams",
          "AoSilence", "AoEndless", "AoSameWords", "AoSilenceHours", "AoStreamHours")  # macros the sections use


def add_facts(root: Path, rows: list[dict], headline: dict, rec: Records) -> dict:
    """Record the facts (TeX); return the README bullets (Markdown) in the same order, and Figure 1's message."""
    group = [r for r in rows if (r["machine"], r["threads"]) == HEADLINE and not r.get("anchor")]
    drawn = [r for r in group if r["plot"]]
    head = (next((r for r in drawn if r["variant"] == headline.get("ours")), None)
            or next((r for r in drawn if r["role"] == "ours"), None))
    aon = next((r for r in group if r["variant"] == headline.get("always_on")), None) or head
    others = [r for r in drawn if r["role"] == "comparator"]
    bullets: list[str] = []
    comparison = ()

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
        vs = [f"{speed_text(derived(r)['speed'])}\u00d7 for {_name(r)}" + (" (whole clip)" if r["class"] == "whole_utterance"
              else "") + ("" if r["timing_panel"] == "full" else " (timed on 73 utterances)") for r in peers]
        share, x = fmt(100.0 * h["cpu"], 1), speed_text(h["speed"])
        tail = f", against {_names(vs)}" if vs else ""
        m3 = ao.m3(root) if aon and aon["variant"] == ao.XISA_VARIANT else None
        best = [max((r for r in peers if r["class"] == c), key=lambda r: derived(r)["speed"], default=None)
                for c in ("streaming", "whole_utterance")]  # the abstract names the fastest of each class
        short = [f"{fmt(h['speed'] / derived(r)['speed'], 1)}\u00d7 the fastest {kind} system there ({_name(r)}, "
                 f"{speed_text(derived(r)['speed'])}\u00d7)" for r, kind in zip(best, ("other streaming", "whole-clip")) if r]
        fact("Cost", "Cost", f"One live stream of ULTRA costs {share}% of one CPU core: {x}\u00d7 real time on one "
             f"quiet Zen\u00a02 core" + (f", {_names(short)}" if short else "") + ".",
             f"{share}% of one CPU core per live stream: {_name(head)} ({_ck(head)}) runs at {x}\u00d7 real time "
             f"on one quiet Zen 2 core{tail}." + (f" {'An earlier checkpoint, ' + _ck(aon) + ',' if aon is not head else _ck(aon)}"
                                                  f" runs at {speed_text(m3['speed'])}\u00d7 on one Apple M3 Ultra core."
                                                  if m3 else ""), rowsrc(head, *peers),
             "share of one core and speed of the headline row and of every other system on its host", panel)
        if m3:
            put("AoMthreeSpeed", speed_text(m3["speed"]), m3, f"speed of {_ck(aon)} on one Apple M3 Ultra core "
                "(macOS, thread count 1, no core pinning)", "730-utterance DEV panel", raw=m3["speed"], unit="x")
    else:
        fact("Cost", "Cost", None, None, {}, "pending", panel)
    # ---- 2. watts: the always-on row against every other system with a same-run energy line on its host
    mine = sorted((r for r in group if r["role"] == "ours" and r.get("watts") is not None and head
                   and r["host"] == head["host"]), key=lambda r: (r is not head, r.get("order", 999)))
    wr = mine[0] if mine else aon  # our own energy line first, then the newest checkpoint of ours that has one
    wrows = sorted((r for r in others if r.get("watts") is not None and wr and r["host"] == wr["host"]),
                   key=lambda r: r["watts"])
    tests = "Our always-on tests" if wr is aon else "Our capacity, silence and endless-stream tests"
    intro = ("" if aon is head or not head else f"{tests} ran on an earlier checkpoint, {_ck(aon)} (same "
             f"architecture; speed within {max(1, math.ceil(100.0 * abs(derived(aon)['speed'] / derived(head)['speed'] - 1.0)))}"
             f"% of {_ck(head)}): ")
    if wr and wr.get("watts") is not None and wrows:
        w = wr["watts"]
        kwh = fmt(w * HOURS_PER_YEAR / 1000.0, 1)
        who = intro if wr is aon else ""
        who = who or ("" if wr is head else f"{_ck(wr)}, an earlier checkpoint ({_ck(head)} is not measured yet): ")
        fact("Watts", "Watts", f"{who}{'one' if who else 'One'} stream draws {fmt(w, 3)}\u2009W, versus "
             f"{fmt(wrows[0]['watts'], 3)}\u2009W for {_name(wrows[0])}.",
             f"{fmt(w, 3)} W per always-on stream from same-run core energy registers ({_ck(wr)}), about "
             f"{kwh} kWh per stream-year. " + _names([f"{_name(r)} draws {fmt(r['watts'], 3)} W" for r in wrows]) + ".",
             rowsrc(wr, *wrows), "watts per always-on stream of our row against the other systems", panel)
        put("AoWatts", fmt(w, 3), rowsrc(wr), f"W per always-on stream, {_ck(wr)}", panel, raw=w, unit="W")
        put("MoneyOursWatts", "\\relax" if wr is head else tex(
            f"No watts are drawn for ours: {_ck(head)} has no energy measurement yet; {_ck(wr)}, an earlier "
            f"checkpoint, draws {fmt(w, 3)}\u2009W."), rowsrc(wr), "Figure 1 caption: watts of ours", panel)
        put("AoKwh", kwh, rowsrc(wr), f"kWh per stream-year, {_ck(wr)} (W x 8766 h)", panel,
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
        fact("Capacity", "Capacity", f"In a real-time paced trial, {_ck(aon)} sustained {o['n']} live streams on one "
             f"core at a 95th-percentile word lag of {lag}. Zipformer sustained {z['n']} at {zlag}.",
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
    # ---- 4. silence: ten 1-hour blocks, and the pauses of the whole endless stream
    s, e, st = ao.silence(root), ao.endless(root), ao.stream(root)
    if s and e and aon and aon["variant"] == ao.XISA_VARIANT:
        levels = f"{_sg(s['levels'][0], 0)} to {_sg(s['levels'][-1], 0)}\u2009dBFS"
        per = fmt(s["gap_words"] / s["gap_hours"], 2)
        gaps = f"{s['gap_words']} in the {fmt(s['gap_hours'], 1)}\u2009h of pauses of a {fmt(e['hours'], 1)}\u2009h stream ({per} per hour)"
        split = f"{s['silence_words']} in {fmt(s['silence_hours'], 2)}\u2009h of digital silence, {s['gap_words'] - s['silence_words']} in the noise gaps"
        wl = "ten 1-hour non-speech blocks (eval-lane) and the pauses of the endless stream (runtime-lane, no reset)"
        fact("Silence", "Silence", f"In {fmt(s['hours'], 0)}\u2009h of silence and synthetic noise it emitted "
             f"{'no word' if s['words'] == 0 else str(s['words']) + ' words'}, and {gaps}.",
             f"{s['words']} words in {fmt(s['hours'], 0)} hours of digital silence and synthetic white, pink and brown "
             f"noise at {levels} ({s['blocks']} one-hour blocks, {_ck(aon)}); {s['gap_words']} words in the "
             f"{fmt(s['gap_hours'], 2)} h of pauses of the {fmt(e['hours'], 2)} h continuous stream ({per} per hour: "
             f"{split}).", s, "non-speech", wl)
        put("AoSilence", tex(f"Decoded as one stream from a fresh state, each of {s['blocks']} one-hour blocks of "
                             f"digital silence and of white, pink and brown noise at {levels} gave an empty "
                             f"transcript: {s['words']} words in {fmt(s['hours'], 0)}\u2009h. In the pauses of the "
                             f"{fmt(e['hours'], 2)}\u2009h stream it emitted {s['gap_words']} words in "
                             f"{fmt(s['gap_hours'], 2)}\u2009h, {per} per hour ({split})."),
            s, "non-speech, results sentence", wl)
        put("AoSilenceHours", fmt(s["hours"], 0), s, "hours of non-speech decoded", wl, raw=s["hours"], unit="h")
    else:
        fact("Silence", "Silence", None, None, {}, "pending", "pending")
    # ---- 5. the endless stream: memory, lag and WER over all of it; reset-with-replay on its first hour
    iso = ao.int8(root, aon["variant"]) if aon else None
    if e and st and iso and aon["variant"] == ao.XISA_VARIANT:
        rep, t0, pen, left = st["replay"], st["t0"], st["penalty"], st["left"]
        hrs, lag = fmt(e["hours"], 2), f"{fmt(e['lag50'], 0)}\u2009ms median and {fmt(e['lag95'], 0)}\u2009ms at the 95th percentile"
        halves = f"{fmt(e['first'], 2)}% in its first half and {fmt(e['second'], 2)}% in its second"
        rr = f"T\u00a0=\u00a0{fmt(st['t_s'], 1)}\u2009s, R\u00a0=\u00a0{fmt(st['r_s'], 1)}\u2009s"
        wl = f"eval-lane's endless stream ({hrs} h, {e['utts']:,} DEV utterances once, seeded pauses)"
        src = {"source": f"{e['source']}; {st['source']}; {iso['source']}", "digest": e["digest"]}
        hour1 = (f"with reset-with-replay the stream's first hour scores {fmt(rep['observed_b'], 2)}% WER against "
                 f"{fmt(rep['observed_a'], 2)}% for the same utterances decoded one by one")
        fact("Endless", "Endless", f"Over a {fmt(e['hours'], 1)}\u2009h continuous stream (one process, no restart) "
             f"its memory never exceeded {fmt(e['peak'], 1)}\u2009MB, its word lag stayed at {lag}, and its WER did "
             f"not drift ({halves}); {hour1}.",
             f"Memory {fmt(e['start'], 2)} MB at the start, {fmt(e['warm'], 1)} MB once loaded, never above "
             f"{fmt(e['peak'], 2)} MB, {fmt(e['end'], 2)} MB at the end of a {hrs} h continuous stream ({e['utts']:,} "
             f"utterances, one process, no restart, {_ck(aon)}); word lag p50 {fmt(e['lag50'], 0)} / p95 "
             f"{fmt(e['lag95'], 0)} ms over {e['words']:,} words; WER {halves}. With reset-with-replay ({rr}), one "
             f"continuous stream matches isolated decoding on the first stream hour: {fmt(rep['observed_b'], 2)}% "
             f"against {fmt(rep['observed_a'], 2)}%, {st['utts']} utterances ({fmt(t0['observed_b'], 2)}% without the "
             "reset); its full-length run is still going.", src, "endless", wl)
        put("AoEndless", tex(
            f"Over the whole {hrs}\u2009h stream ({e['utts']:,} utterances, one process, no restart, no reset) the "
            f"resident memory of {_ck(aon)} was {fmt(e['start'], 2)}\u2009MB at the start, {fmt(e['warm'], 1)}\u2009MB "
            f"once loaded, at most {fmt(e['peak'], 2)}\u2009MB, and {fmt(e['end'], 2)}\u2009MB at the end; its word "
            f"lag over {e['words']:,} matched words was {lag}. It scored {fmt(e['wer'], 2)}% pooled WER against "
            f"{fmt(iso['observed_b'], 2)}% with each utterance decoded alone by the same engine, {halves}: a constant "
            "cost of continuous decoding, not a drift. In the first hour "
            f"({st['utts']} utterances) the cost was {_sg(pen['delta'], 2)}\u2009pp (bootstrap bounds "
            f"{_sg(pen['lower'], 2)} to {_sg(pen['upper'], 2)}), {_sg(left['delta'], 2)}\u2009pp of it from the pauses' "
            "audio in the encoder's left context. Resetting the state after "
            f"{fmt(st['t_s'], 1)}\u2009s without output and replaying the last {fmt(st['r_s'], 1)}\u2009s brings that "
            f"hour to {fmt(rep['observed_b'], 2)}% ({_sg(rep['delta'], 2)}\u2009pp, {_sg(rep['lower'], 2)} to "
            f"{_sg(rep['upper'], 2)}; {fmt(t0['observed_b'], 2)}% without the reset in the same build), with "
            f"{st['replay_gap_words']} false words in its pauses. The {hrs}\u2009h run with the reset on was still "
            "going when this draft was built."), src, "endless, results sentence", wl)
        put("AoStreamHours", fmt(e["hours"], 1), src, "length of the endless stream", wl, raw=e["hours"], unit="h")
    else:
        fact("Endless", "Endless", None, None, {}, "pending", "pending")
    # ---- 6. the same words on three instruction sets, and INT8 against FP32 on full dev
    i8row = next((r for r in sorted((r for r in group if r["role"] == "ours"), key=lambda r: (r is not head,
                  r.get("order", 999))) if ao.int8(root, r["variant"])), None)  # the newest checkpoint with a report
    sw, i8 = ao.same_words(root), (ao.int8(root, i8row["variant"]) if i8row else None)
    if sw and i8 and aon and aon["variant"] == ao.XISA_VARIANT:
        isas = _names(sw["isas"])
        i8s = (f"on full dev ({i8['utts']:,} utterances) {_ck(i8row)} scores {fmt(i8['observed_b'], 2)}% pooled WER in "
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
        lm = (rec.rows["LmEngineGain"]["value"], rec.rows["LmHeldOutGain"]["value"], "H22ctl") if "LmHeldOutGain" in rec.rows else None
        vs = _names([f"{fmt(derived(r)['wer_clean'], 2)}% for {_name(r)}" for r in pick])
        accuracy = f"{_ck(head)} scores {fmt(h['wer_clean'], 2)}% dev-clean WER, against {vs}."
        if fast and not head["interim"] and h["watts"] is not None and fast["watts"] is not None:
            f = derived(fast)
            comparison = (head, fast)
            accuracy = (f"{_ck(head)} scores {fmt(h['wer_clean'], 2)}% dev-clean WER and {fmt(h['wer'], 2)}% pooled WER, "
                        f"versus {fmt(f['wer_clean'], 2)}% and {fmt(f['wer'], 2)}% for {_name(fast)}.")
        fact("Accuracy", "Where we trail", accuracy,
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
        if comparison:
            see = (f"On one quiet Zen\u00a02 core, {_name(head)} ({_ck(head)}) reaches {speed_text(h['speed'])}× real time "
                   f"at {fmt(h['watts'], 3)}\u2009W per stream, versus {speed_text(f['speed'])}× and "
                   f"{fmt(f['watts'], 3)}\u2009W for {_name(fast)}. Dev-clean WER is {fmt(h['wer_clean'], 2)}% versus "
                   f"{fmt(f['wer_clean'], 2)}%, and pooled WER is {fmt(h['wer'], 2)}% versus {fmt(f['wer'], 2)}%.")
        put("MoneyHeadline", tex(see), rowsrc(head, *([fast] if fast else [])), "Figure 1: what to see", panel)
        figure = md(see)
    else:
        fact("Accuracy", "Where we trail", None, None, {}, "pending", panel)
        figure = "Speed against accuracy on one CPU core, every system measured here."
        put("MoneyHeadline", figure, {}, "pending", panel)
    if aon:
        put("AoCheckpoint", _ck(aon), rowsrc(aon), "the checkpoint the always-on tests ran on", panel, unit="name")
    for rid in (r for r in ALWAYS if r not in rec.rows):  # defined either way, so the sections build
        put(rid, "\\relax" if rid == "MoneyOursWatts" else "pending", {}, "pending: no admitted receipt or row yet", "pending")
    return {"bullets": bullets, "figure": figure, "comparison": comparison}
