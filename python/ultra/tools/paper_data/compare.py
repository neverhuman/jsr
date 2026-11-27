"""Numbers and sentences the text quotes from the measured comparison (Figure 1's group).

Supports both states cleanly:
- Before standalone live-stream hold (14q) lands: headline.ours (`rc1-ema-final`) is plotted alongside
  the 4 comparators, while live-mode WER is quoted from the verified 14p provisional text with zero
  borrowed cold timing, RSS, or watts.
- Once 14q lands: both operating points (`rc1-ema-live` and `rc1-ema-final`) on the exact same INT8
  weights are plotted and compared automatically.
"""

from __future__ import annotations

from .cold import ENERGY_WORDS, speed_text
from .common import HOST_NAMES, PaperDataError, fmt
from .evidence import HEADLINE, derived
from .records import Records

NOTHING = "\\relax"


def _names(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _name(r: dict) -> str:
    return r.get("text") or r["short"]


def _rel(h_speed: float, f_speed: float, name: str, ratio: float | None = None) -> str:
    """Our commit pass against a streaming comparator's single-core speed: 'within 2%' or an explicit ratio
    (the median of within-block ratios when the rows were timed in interleaved blocks)."""
    ratio = ratio or h_speed / f_speed
    if abs(ratio - 1.0) <= 0.02:
        return f"within 2\\% of streaming {name}'s single-core speed"
    return f"at {fmt(ratio, 2)}$\\times$ streaming {name}'s single-core speed"


# Full-DEV pooled WER macros of the v3.1.0 rows (candidate.py), keyed by evidence variant.
FULL_DEV = {"rc2-live-lm": "CandLiveLmFullWer", "rc2-live-greedy": "CandLiveGreedyFullWer", "rc2-final": "CandFinalFullWer",
            "v41-live-lm": "LookLiveFullWer", "v41-final": "LookFinalFullWer"}  # v4.1.0: look41.py
CURRENT = ("rc2", "v41")  # releases whose commit delay and capacity were measured on an earlier release


def add_comparison(rows: list[dict], headline: dict, rec: Records, extras: dict | None = None) -> None:
    group = [r for r in rows if (r["machine"], r["threads"]) == HEADLINE and r["plot"]]
    for r in group:
        r["_d"] = derived(r)
    ours = [r for r in group if r["role"] == "ours"]
    others = [r for r in group if r["role"] == "comparator"]
    extra_complete = (extras or {}).get("complete", [])
    first = rows[0] if rows else None
    note = dict(label="reported", workload=first["workload"] if first else "pending",
                workload_digest=first["workload_digest"] if first else "pending",
                source="docs/paper/data/evidence/*.jsonl")

    def count(rid: str, value: int, what: str) -> None:
        rec.add(rid, str(value), raw=value, unit="rows", configuration=f"number of {what}", **note)

    def say(rid: str, text: str, what: str) -> None:
        rec.add(rid, text, unit="sentence", configuration=f"generated: {what}", **note)

    count("ColdComparatorCount", len(others), "Zen 2 comparator systems drawn in Figure 1")
    count("ColdCompleteCompCount", len(others) + len(extra_complete),
          "complete-panel public comparators (Zen 2 plotted plus other-host complete-panel)")
    count("ColdCompletePanelCount", len(group) + len(extra_complete),
          "complete-panel systems in the 730-utterance DEV WER ranking")
    subset = [r for r in group if r["timing_panel"] != "full"]
    say("ColdSubsetSentence", NOTHING if not subset else f"{len(subset)} slow system"
        f"{'s are' if len(subset) != 1 else ' is'} timed on a 73-utterance subset of the panel "
        f"({_names([_name(r) for r in subset])}); {'their' if len(subset) != 1 else 'its'} WER is from the full panel.",
        "rows timed on the subset")
    if not ours or not others:
        for rid in ("ColdSubtitle", "ColdHeadSentence", "ColdSpeedSentence", "ColdAccuracySentence",
                    "ColdMemorySentence", "ColdOrderSentence", "ColdWattsSentence", "ColdIntroLiveItem",
                    "ColdIntroContrib", "ColdLimitLive", "ColdConclusionResult", "ColdAbstractLive",
                    "ColdAbstractFinal", "ColdAbstractZip", "ColdAbstractMoreAcc"):
            say(rid, "Pending." if rid == "ColdHeadSentence" else NOTHING, "pending")
        return

    want = headline.get("ours")
    head = next((r for r in ours if r["variant"] == want), None) if want else min(ours, key=lambda r: r["_d"]["wer"])
    if head is None:
        raise PaperDataError(f"systems.json names the headline row {want!r}, which is not drawn in Figure 1")
    want_live = headline.get("ours_live")
    live = next((r for r in ours if r["variant"] == want_live), None) if want_live else None
    has_live = live is not None and live is not head
    h, n = head["_d"], len(others)
    lv = live["_d"] if has_live else None
    live_pair = ((live or {}).get("paired_vs_anchor") or {}) if has_live else {}

    def src(*rs: dict) -> dict:
        return dict(label=head["label"], workload=head["workload"], workload_digest=head["workload_digest"],
                    source="; ".join(f"docs/paper/data/evidence/{r['_where']} id={r['id']}" for r in rs))

    def num(rid: str, value: float, text: str, unit: str, what: str, *rs: dict) -> None:
        rec.add(rid, text, raw=value, unit=unit, configuration=what, **src(*rs))

    all_comp_wers = [r["_d"]["wer"] for r in others] + [e["wer"] for e in extra_complete]
    all_panel_wers = [r["_d"]["wer"] for r in group] + [e["wer"] for e in extra_complete]
    better = sum(w < h["wer"] for w in all_comp_wers)
    beaten_n = sum(w > h["wer"] for w in all_comp_wers)
    wer_rank = 1 + sum(w < h["wer"] for w in all_panel_wers)
    n_comp_all = len(all_comp_wers)
    n_panel_all = len(all_panel_wers)

    if has_live:
        w_part = (f", {fmt(lv['watts'], 3)}\\,J active core energy per audio-second,"
                  if (lv["watts"] is not None and live.get("energy_method") == "core_msr_active") else "")
        rec.add("ColdAbstractLive",
                f"Live decoding runs at {speed_text(lv['speed'])}$\\times$ real time with "
                f"{fmt(lv['rss_mb'], 0)}\\,MB peak RSS{w_part} and {fmt(lv['wer'], 2)}\\% pooled WER.",
                unit="sentence", configuration="abstract live operating point", **src(live))
    else:
        say("ColdAbstractLive", NOTHING, "no live cold row")
    final_lead = "The full live-plus-commit pipeline, committing at reference utterance boundaries,"
    rec.add("ColdAbstractFinal",
            f"{final_lead} reaches {fmt(h['wer'], 2)}\\% pooled WER at {speed_text(h['speed'])}$\\times$ real time.",
            unit="sentence", configuration="abstract reference-boundary final operating point", **src(head))
    zipformer = next((r for r in others if r["variant"].startswith("r0-int8") and r["host"] == head["host"]
                      and not r["interim"] and r["timing_panel"] == "full"), None)
    if zipformer:
        z = zipformer["_d"]
        zip_txt = (f"Streaming Zipformer INT8 reaches {speed_text(z['speed'])}$\\times$ real time and "
                   f"{fmt(z['wer'], 2)}\\% WER under the matched protocol")
        if has_live:
            ratio = fmt(live_pair.get("speed") or lv["speed"] / z["speed"], 1)
            if (lv["watts"] is not None and z["watts"] is not None
                    and live.get("energy_method") == zipformer.get("energy_method") == "core_msr_active"):
                pct = fmt(100.0 * (live_pair.get("energy") or lv["watts"] / z["watts"]), 0)
                zip_txt += f"; Isosyll live delivers {ratio}$\\times$ its throughput at {pct}\\% of its active core energy."
            else:
                zip_txt += f"; Isosyll live delivers {ratio}$\\times$ its throughput."
        else:
            zip_txt += "."
        rec.add("ColdAbstractZip", zip_txt, unit="sentence",
                configuration="abstract matched Zipformer reference",
                **(src(zipformer, live) if has_live else src(zipformer)))
    else:
        say("ColdAbstractZip", NOTHING, "no matched Zipformer row")
    words_num = {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six", 7: "Seven", 8: "Eight", 9: "Nine"}
    if better > 0:
        n_txt = words_num.get(better, str(better))
        verb = "is" if better == 1 else "are"
        say("ColdAbstractMoreAcc",
            f"{n_txt} of the {n_comp_all} public recognisers we measured {verb} more accurate than the final mode.",
            "abstract count of more-accurate comparators")
    else:
        say("ColdAbstractMoreAcc", NOTHING, "no more-accurate comparator")

    host = HOST_NAMES.get(head["host"], head["host"])
    for rid, key, nd, unit in (("Cpu", "cpu", 3, "CPU-s/audio-s"), ("Wer", "wer", 2, "% WER, pooled"),
                               ("WerClean", "wer_clean", 2, "% WER, dev-clean half"),
                               ("WerOther", "wer_other", 2, "% WER, dev-other half"),
                               ("Rss", "rss_mb", 0, "MB peak resident memory"), ("Model", "model_mb", 0, "MB of model files")):
        num(f"ColdHead{rid}", h[key], fmt(h[key], nd), unit, "headline final row", head)
        if has_live:
            num(f"ColdLive{rid}", lv[key], fmt(lv[key], nd), unit, "headline live row", live)
    num("ColdHeadSpeed", h["speed"], speed_text(h["speed"]), "x real time, one core", "headline final row", head)
    final_share = 100.0 * h["cpu"]
    num("ColdFinalCorePct", final_share, fmt(final_share, 1), "% of one core", "CPU share for live + commit", head)
    if h["watts"] is not None:
        num("ColdHeadWatts", h["watts"], fmt(h["watts"], 3), "W per stream (live + commit)", "final core-register W", head)

    if has_live:
        live_share = 100.0 * lv["cpu"]
        num("ColdLiveSpeed", lv["speed"], speed_text(lv["speed"]), "x real time, one core", "headline live row", live)
        num("ColdHeadCorePct", live_share, fmt(live_share, 1), "% of one core", "CPU share for one live stream", live)
        if lv["watts"] is not None:
            num("ColdLiveWatts", lv["watts"], fmt(lv["watts"], 3), "W per live stream", "live core-register W", live)

    peers = [r for r in others if r["host"] == head["host"]]
    stream_peers = [r for r in peers if r["class"] == "streaming"]
    fast = max(stream_peers, key=lambda r: r["_d"]["speed"]) if stream_peers else None
    light = min(others, key=lambda r: r["_d"]["rss_mb"])
    heavy = max(others, key=lambda r: r["_d"]["rss_mb"])
    best = min(others, key=lambda r: r["_d"]["wer"])
    b = best["_d"]
    gap = h["wer"] - b["wer"]
    num("ColdHeadWerGapBest", gap, fmt(gap, 2), "pp WER", "pooled panel WER of ours (final) minus lowest comparator",
        head, best)
    count("ColdAccuracyCount", better, "complete-panel comparators with lower pooled WER than our final mode")
    count("ColdHeadWerRank", wer_rank, "pooled WER rank of our final mode among all complete-panel systems")
    count("ColdHeadWerBeaten", beaten_n, "complete-panel comparators beaten by our final mode on pooled WER")
    ord_rank = f"{wer_rank}{'th' if 11 <= wer_rank % 100 <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(wer_rank % 10, 'th')}"
    say("ColdHeadWerRankOrd", ord_rank, "ordinal pooled WER rank of our final mode")

    if fast and has_live:
        num("ColdHeadSpeedOverFast", lv["speed"] / fast["_d"]["speed"], fmt(lv["speed"] / fast["_d"]["speed"], 1),
            "x", "live speed / fastest same-host streaming comparator", live, fast)
        num("ColdLightRssOverHead", light["_d"]["rss_mb"] / lv["rss_mb"], fmt(light["_d"]["rss_mb"] / lv["rss_mb"], 1),
            "x", "peak RSS of lightest comparator / our live stream", live, light)
        if fast["_d"]["watts"] is not None and lv["watts"] is not None:
            num("ColdZipWattsOverLive", fast["_d"]["watts"] / lv["watts"],
                fmt(fast["_d"]["watts"] / lv["watts"], 1), "x",
                "fastest streaming comparator active core W / our live active core W", live, fast)

    def full(variant: str, old: str, default: str) -> str:
        key = FULL_DEV.get(variant, old)
        return rec.rows[key]["value"] if key in rec.rows else default

    full_final = full(head["variant"], "RelFullFinalPooled", fmt(h["wer"], 2))
    panel_live = rec.rows["RelPanelLivePooled"]["value"] if "RelPanelLivePooled" in rec.rows else "--"
    full_live = full(live["variant"], "RelFullLivePooled", "--") if has_live else "--"
    released_now = head["variant"].startswith(CURRENT)  # the commit delay and live capacity were measured on the prior release
    delay = (f"a post-endpoint commit delay (median {rec.rows['RelCommitDelayPfifty']['value']}\\,ms measured on the prior "
             "release, not re-measured)" if released_now else
             f"a median {rec.rows['RelCommitDelayPfifty']['value']}\\,ms post-endpoint commit delay")
    decode = ("chunked greedy decoding" if has_live and "greedy" in live["configuration"] else
              "chunked beam-12 decoding with a 243\\,MiB compact live language model" if has_live and "beam 12" in live["configuration"] else
              "chunked decoding with a 34.5\\,MiB live language model")
    pair_x = (head.get("paired_vs_anchor") or {}).get("speed")
    speed_rel = _rel(h["speed"], fast["_d"]["speed"], _name(fast), pair_x) if fast else ""
    say("ColdFinalVsZipSpeed", f"{speed_rel} ({speed_text(h['speed'])}$\\times$ utterance-segmented vs.\\ "
        f"{speed_text(fast['_d']['speed'])}$\\times$ streaming)" if fast else NOTHING, "final commit pass speed against Zipformer")

    if has_live:
        w_lv = f", {fmt(lv['watts'], 3)}\\,J per audio-second of active core energy" if lv["watts"] is not None else ""
        rec.add("ColdSubtitle", f"Live: {speed_text(lv['speed'])}$\\times$ real time, {fmt(lv['wer'], 2)}\\% WER "
                f"$\\cdot$ Final (commit pass at utterance ends): {speed_text(h['speed'])}$\\times$, "
                f"{fmt(h['wer'], 2)}\\% WER",
                unit="sentence", configuration="subtitle with separate live and final operating points", **src(live, head))
        say("ColdHeadSentence", f"On one quiet Zen~2 core (host {host}), {_name(live)} decodes cold at "
            f"{speed_text(lv['speed'])}$\\times$ real time ({fmt(100.0 * lv['cpu'], 1)}\\% of one core, "
            f"{fmt(lv['rss_mb'], 0)}\\,MB peak memory) at {fmt(lv['wer'], 2)}\\% pooled panel WER "
            f"({fmt(lv['wer_clean'], 2)}\\% dev-clean, {fmt(lv['wer_other'], 2)}\\% dev-other). "
            f"Adding the utterance-segmented commit pass on the same weights ({_name(head)}) runs cold at "
            f"{speed_text(h['speed'])}$\\times$ real time ({fmt(h['rss_mb'], 0)}\\,MB peak memory) and reaches pooled "
            f"panel WER {fmt(h['wer'], 2)}\\% ({fmt(h['wer_clean'], 2)}\\% dev-clean, {fmt(h['wer_other'], 2)}\\% "
            f"dev-other).", "the two operating points of our release model")
        if fast:
            f = fast["_d"]
            say("ColdSpeedSentence", f"On the quiet full-panel host {host}, {_name(live)} runs at "
                f"{speed_text(lv['speed'])}$\\times$ against {speed_text(f['speed'])}$\\times$ for matched streaming "
                f"{_name(fast)} ({fmt(live_pair.get('speed') or lv['speed'] / f['speed'], 1)}$\\times$), while utterance-segmented "
                f"{_name(head)} runs {_rel(h['speed'], f['speed'], _name(fast), pair_x)} "
                f"({speed_text(h['speed'])}$\\times$ utterance-segmented vs.\\ {speed_text(f['speed'])}$\\times$ "
                f"streaming).", "speed")
        else:
            say("ColdSpeedSentence", NOTHING, "speed")
        say("ColdMemorySentence", f"In live mode, \\ultra{{}} peaks at {fmt(lv['rss_mb'], 0)}\\,MB, "
            f"{fmt(100.0 * (live_pair.get('rss') or lv['rss_mb'] / light['_d']['rss_mb']), 0)}\\% of matched streaming "
            f"{_name(light)} ({fmt(light['_d']['rss_mb'], 0)}\\,MB) and "
            f"{fmt(heavy['_d']['rss_mb'] / lv['rss_mb'], 1)}$\\times$ less than {_name(heavy)} "
            f"({fmt(heavy['_d']['rss_mb'], 0)}\\,MB, timed separately); loading the larger 5-gram LM for the commit pass raises peak RSS to "
            f"{fmt(h['rss_mb'], 0)}\\,MB.", "memory")
        say("ColdIntroLiveItem", f"160\\,ms {decode} with no endpoint detector required, using "
            f"{fmt(100.0 * lv['cpu'], 1)}\\% of one quiet Zen~2 core ({speed_text(lv['speed'])}$\\times$ real time, "
            f"{fmt(lv['rss_mb'], 0)}\\,MB peak RAM{w_lv})---"
            f"{fmt(live_pair.get('speed') or 0, 1)}$\\times$ the throughput of matched streaming Zipformer INT8 at "
            f"{fmt(100.0 * (live_pair.get('energy') or 0), 0)}\\% of its active core energy per audio-second---at {fmt(lv['wer'], 2)}\\% pooled panel "
            f"WER ({full_live}\\% full DEV).", "intro live bullet")
        say("ColdIntroContrib", f"Live streaming on one quiet Zen~2 core ({speed_text(lv['speed'])}$\\times$ "
            f"vs.\\ {speed_text(fast['_d']['speed'])}$\\times$ for {_name(fast)}, "
            f"{fmt(lv['speed'] / fast['_d']['speed'], 1)}$\\times$; {fmt(lv['watts'], 3)}\\,J per audio-second; "
            f"{fmt(lv['rss_mb'], 0)}\\,MB RAM; {fmt(lv['wer'], 2)}\\% pooled panel WER), paired on the same INT8 "
            f"weights with an utterance-segmented commit pass ({speed_text(h['speed'])}$\\times$, "
            f"{fmt(h['watts'], 3)}\\,J per audio-second, {fmt(h['rss_mb'], 0)}\\,MB RAM) that reaches "
            f"{fmt(h['wer'], 2)}\\% pooled panel WER against {fmt(fast['_d']['wer'], 2)}\\% for streaming {_name(fast)} "
            f"(paired comparison in \\cref{{sec:versus}}), {speed_rel}." if (fast and lv["watts"] is not None and h["watts"] is not None)
            else f"Dual operating points on one INT8 artifact (\\cref{{sec:results}}).", "intro contribution 2")
        w_lv_plain = f", {fmt(lv['watts'], 3)}\\,J per audio-second" if lv["watts"] is not None else ""
        say("ColdLimitLive", f"While {_name(live)} streams continuously at {speed_text(lv['speed'])}$\\times$ real time"
            f"{w_lv_plain}, and {fmt(lv['rss_mb'], 0)}\\,MB RAM without an endpoint detector, the "
            f"higher-accuracy {_name(head)} operating point requires utterance endpoints (capped at 40\\,s), adds "
            f"{delay}, and increases peak "
            f"RSS to {fmt(h['rss_mb'], 0)}\\,MB due to the 5-gram LM pack.", "limitations live vs final")
        w_lv_at = f" at {fmt(lv['watts'], 3)}\\,J per audio-second and" if lv["watts"] is not None else " at"
        w_hd_c = f", {fmt(h['watts'], 3)}\\,J per audio-second," if h["watts"] is not None else ","
        vs_zip = (f"---{speed_rel} ({speed_text(h['speed'])}$\\times$ "
                  f"utterance-segmented vs.\\ {speed_text(fast['_d']['speed'])}$\\times$ streaming) at "
                  f"{fmt(h['wer'], 2)}\\% vs.\\ {fmt(fast['_d']['wer'], 2)}\\% pooled WER" if fast else "")
        say("ColdConclusionResult", f"two operating points on one INT8 artifact: "
            f"{speed_text(lv['speed'])}$\\times$ real-time continuous live streaming{w_lv_at} "
            f"{fmt(lv['rss_mb'], 0)}\\,MB RAM ({fmt(lv['wer'], 2)}\\% panel WER), and an utterance-segmented commit "
            f"pass reaching {fmt(h['wer'], 2)}\\% pooled panel WER ({full_final}\\% on full DEV) at "
            f"{speed_text(h['speed'])}$\\times$ real time ({fmt(h['rss_mb'], 0)}\\,MB RAM{w_hd_c} "
            f"{'commit delay not re-measured' if released_now else 'median ' + rec.rows['RelCommitDelayPfifty']['value'] + '\\,ms delay'}){vs_zip}", "conclusion summary")
    else:
        w_hd = f", {fmt(h['watts'], 3)}\\,J per audio-second" if h["watts"] is not None else ""
        w_hd_and = f" and {fmt(h['watts'], 3)}\\,J per audio-second" if h["watts"] is not None else ""
        say("ColdSubtitle", f"Utterance-segmented final text: {fmt(h['wer'], 2)}\\% panel WER ({full_final}\\% full "
            f"DEV) at {speed_text(h['speed'])}$\\times$ real time{w_hd} on one Zen~2 core", "subtitle")
        say("ColdHeadSentence", f"On one quiet Zen~2 core (host {host}), {_name(head)} decodes cold at "
            f"{speed_text(h['speed'])}$\\times$ real time ({fmt(final_share, 1)}\\% of one core, "
            f"{fmt(h['rss_mb'], 0)}\\,MB peak memory) and reaches {fmt(h['wer'], 2)}\\% pooled panel WER "
            f"({fmt(h['wer_clean'], 2)}\\% dev-clean, {fmt(h['wer_other'], 2)}\\% dev-other). On the same weights, "
            f"the 160\\,ms greedy live stream scores {panel_live}\\% pooled panel WER ({full_live}\\% full DEV); its "
            f"standalone cold single-core timing and energy hold on host {host} is pending and is not plotted until "
            f"its own cold receipt arrives.", "headline row while live cold hold is pending")
        if fast:
            f = fast["_d"]
            say("ColdSpeedSentence", f"On host {host}, utterance-segmented {_name(head)} runs within 2\\% of "
                f"streaming {_name(fast)}'s single-core speed ({speed_text(h['speed'])}$\\times$ utterance-segmented "
                f"vs.\\ {speed_text(f['speed'])}$\\times$ streaming), and {fmt(h['speed'] / best['_d']['speed'], 1)}"
                f"$\\times$ faster than offline {_name(best)}.", "speed")
        else:
            say("ColdSpeedSentence", NOTHING, "speed")
        say("ColdMemorySentence", f"Loading the 5-gram LM for the commit pass brings {_name(head)} peak RSS to "
            f"{fmt(h['rss_mb'], 0)}\\,MB, against {fmt(light['_d']['rss_mb'], 0)}\\,MB for {_name(light)} and "
            f"{fmt(heavy['_d']['rss_mb'], 0)}\\,MB for {_name(heavy)}; standalone live-mode RSS is reported once the "
            f"stream-only cold hold completes.", "memory")
        say("ColdIntroLiveItem", f"160\\,ms chunked greedy decoding with no endpoint detector required, scoring "
            f"{panel_live}\\% pooled panel WER ({full_live}\\% full DEV) on the exact same INT8 weights (standalone "
            f"cold single-core timing, memory, and watts pending its dedicated quiet-host hold).", "intro live bullet")
        say("ColdIntroContrib", f"Utterance-segmented commit pass on one compact INT8 artifact that reaches "
            f"{fmt(h['wer'], 2)}\\% pooled panel WER against {fmt(fast['_d']['wer'], 2)}\\% for streaming {_name(fast)} within "
            f"2\\% of its single-core speed ({speed_text(h['speed'])}$\\times$ utterance-segmented vs.\\ "
            f"{speed_text(fast['_d']['speed'])}$\\times$ streaming, \\cref{{sec:results}})." if fast
            else "Single INT8 release artifact (\\cref{sec:results}).", "intro contribution 2")
        say("ColdLimitLive", f"The higher-accuracy {_name(head)} operating point requires utterance endpoints (capped "
            f"at 40\\,s), adds a median {rec.rows['RelCommitDelayPfifty']['value']}\\,ms post-endpoint commit delay, "
            f"and peaks at {fmt(h['rss_mb'], 0)}\\,MB RSS due to the 5-gram LM pack; the standalone stream-only cold "
            f"hold for the endpoint-free live mode is pending.", "limitations live vs final")
        vs_zip = (f"---within 2\\% of streaming {_name(fast)}'s single-core speed ({speed_text(h['speed'])}$\\times$ "
                  f"utterance-segmented vs.\\ {speed_text(fast['_d']['speed'])}$\\times$ streaming) at "
                  f"{fmt(h['wer'], 2)}\\% vs.\\ {fmt(fast['_d']['wer'], 2)}\\% pooled WER---" if fast else "---")
        say("ColdConclusionResult", f"an utterance-segmented commit pass reaching {fmt(h['wer'], 2)}\\% pooled panel "
            f"WER ({full_final}\\% on full DEV) at {speed_text(h['speed'])}$\\times$ real time{w_hd_and}{vs_zip}"
            f"plus an endpoint-free 160\\,ms live stream ({panel_live}\\% panel WER) on the exact same INT8 weights",
            "conclusion summary")

    def _ord(k: int) -> str:
        return f"{k}{'th' if 11 <= k % 100 <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(k % 10, 'th')}"

    beat_txt = (f"{_name(head)} ({fmt(h['wer'], 2)}\\% pooled) ranks {_ord(wer_rank)} of {n_panel_all} "
                f"complete-panel systems (lower point-estimate WER than {beaten_n} of {n_comp_all} public comparators, including "
                f"streaming {_name(fast)} at {fmt(fast['_d']['wer'], 2)}\\%). "
                if (beaten_n and fast and fast["_d"]["wer"] > h["wer"])
                else f"{_name(head)} ({fmt(h['wer'], 2)}\\% pooled) ranks {_ord(wer_rank)} of {n_panel_all} "
                     f"complete-panel systems (lower point-estimate WER than {beaten_n} of {n_comp_all} public comparators). "
                if beaten_n else "")
    who = f"{better} of the {n_comp_all} comparators" if better not in (0, n_comp_all) else (
        "All comparators" if better == n_comp_all else "No comparator")
    say("ColdAccuracySentence", f"{beat_txt}{who} achieve lower pooled WER. Among configurations "
        f"with plotted throughput, {_name(best)} has the lowest pooled WER at "
        f"{fmt(b['wer'], 2)}\\% ({fmt(b['wer_clean'], 2)}\\% dev-clean, {fmt(b['wer_other'], 2)}\\% dev-other), "
        f"{fmt(gap, 2)} points below {_name(head)}.", "accuracy against all drawn")

    clean_ord = sorted(group, key=lambda r: r["_d"]["wer_clean"])
    other_ord = sorted(group, key=lambda r: r["_d"]["wer_other"])
    moved = sum(other_ord[i] is not r for i, r in enumerate(clean_ord))
    say("ColdOrderSentence", "The dev-other split ranks all systems in the same order as dev-clean." if not moved else
        f"The dev-other split changes the rank order of {moved} systems relative to dev-clean (notably the \\ultra{{}} "
        f"final mode has lower dev-other WER than Zipformer INT8).",
        "whether dev-other orders the drawn systems as dev-clean does")

    rest_w = sorted((r for r in others if r["watts"] is not None and r["host"] == head["host"]),
                    key=lambda r: r["watts"])
    if h["watts"] is not None and rest_w:
        if has_live and lv["watts"] is not None:
            ours_w = f"{fmt(lv['watts'], 3)}\\,J per audio-second for {_name(live)} and {fmt(h['watts'], 3)}\\,J for {_name(head)}"
        else:
            ours_w = f"{fmt(h['watts'], 3)}\\,J per audio-second for {_name(head)}"
        say("ColdWattsSentence", f"Unpaced active core (CPU-domain) energy on host {host} is {ours_w}, against "
            + _names([f"{fmt(r['watts'], 3)}\\,J for {_name(r)}" for r in rest_w])
            + f" ({ENERGY_WORDS[head['energy_method']]}; unpaced CPU core domain only, no board-power claim).", "energy rows")
    else:
        say("ColdWattsSentence", "Core-register energy rows are pending.", "no energy row")
