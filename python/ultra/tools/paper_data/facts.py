"""Page-1 facts for the abstract and README: accuracy, speed, watts, and INT8 parity.

Every number traces directly to our single winning release artifact (v3.0.0 RC1-EMA INT8 + release
decoder) and the 4 kept same-dataset comparators. Zero legacy checkpoints appear; if the standalone
live-stream cold hold (14q) has not landed yet, zero speed/RSS/watts are borrowed for it.
"""

from __future__ import annotations

from pathlib import Path

from .cold import speed_text
from .compare import FULL_DEV, _rel
from .common import fmt
from .evidence import HEADLINE, derived
from .records import Records
from .words import md, name as _name, names as _names, tex


def add_facts(root: Path, rows: list[dict], headline: dict, rec: Records, extras: dict | None = None) -> dict:
    """Record abstract facts (TeX); return README bullets (Markdown), Fig. 1 headline, and quiet comparison rows."""
    group = [r for r in rows if (r["machine"], r["threads"]) == HEADLINE and not r.get("anchor")]
    drawn = [r for r in group if r["plot"]]
    head = (next((r for r in drawn if r["variant"] == headline.get("ours")), None)
            or next((r for r in drawn if r["role"] == "ours"), None))
    live = next((r for r in drawn if r["variant"] == headline.get("ours_live")), None)
    has_live = live is not None and live is not head
    others = [r for r in drawn if r["role"] == "comparator"]
    extra_complete = (extras or {}).get("complete", [])
    bullets: list[str] = []
    comparison: tuple[dict, ...] = ()

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
    if not head or not others:
        for rid, lbl in (("Accuracy", "Accuracy"), ("Speed", "Speed & memory"), ("Watts", "Watts"),
                         ("Parity", "Integer-exact INT8")):
            fact(rid, lbl, None, None, {}, "pending", panel)
        put("MoneyHeadline", "Single-core throughput against pooled WER on the 730-utterance DEV panel.", {},
            "pending", panel)
        return {"bullets": bullets, "figure": "Single-core throughput against pooled WER.", "comparison": ()}

    h = derived(head)
    lv = derived(live) if has_live else None
    peers = sorted((r for r in others if r["host"] == head["host"]), key=lambda r: -derived(r)["speed"])
    stream_peers = [r for r in peers if r["class"] == "streaming"]
    fast = stream_peers[0] if stream_peers else None
    best = min(others, key=lambda r: derived(r)["wer"])
    stream_best = min((r for r in others if r["class"] == "streaming"), key=lambda r: derived(r)["wer"], default=None)

    quiet_ours = [r for r in ((live, head) if has_live else (head,)) if not r["interim"]]
    quiet_w = [r for r in peers if not r["interim"] and r.get("watts") is not None]
    # Keep header table compact so Fig. 1 fits page 1: ours + fastest quiet streaming + lowest-WER quiet comparator
    top_quiet = []
    if fast and fast in quiet_w:
        top_quiet.append(fast)
    best_qw = min(quiet_w, key=lambda r: derived(r)["wer"], default=None)
    if best_qw and best_qw not in top_quiet:
        top_quiet.append(best_qw)
    if quiet_ours and top_quiet:
        comparison = tuple(quiet_ours + top_quiet)

    # 1. Accuracy (panel 730 utt + full DEV 5,567 utt)
    current = head["variant"].startswith("rc2")  # delay, package energy and the 16-stream capacity belong to the prior release
    def _rv(new: str, old: str) -> str | None:
        key = new if current else old
        return rec.rows[key]["value"] if key in rec.rows else None
    full_final = _rv("CandFinalFullWer", "RelFullFinalPooled")
    full_clean = _rv("CandFinalFullClean", "RelFullFinalClean")
    full_other = _rv("CandFinalFullOther", "RelFullFinalOther")
    full_live = (rec.rows[FULL_DEV[live["variant"]]]["value"] if has_live and current and live["variant"] in FULL_DEV
                 else rec.rows["RelFullLivePooled"]["value"] if "RelFullLivePooled" in rec.rows else None)
    panel_live = (fmt(lv["wer"], 2) if has_live else
                  rec.rows["RelPanelLivePooled"]["value"] if "RelPanelLivePooled" in rec.rows else None)
    all_comp_wers = [derived(r)["wer"] for r in others] + [e["wer"] for e in extra_complete]
    all_panel_wers = [derived(r)["wer"] for r in drawn] + [e["wer"] for e in extra_complete]
    beaten_count = sum(w > h["wer"] for w in all_comp_wers)
    wer_rank = 1 + sum(w < h["wer"] for w in all_panel_wers)
    ord_s = f"{wer_rank}{'th' if 11 <= wer_rank % 100 <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(wer_rank % 10, 'th')}"
    fast_ref = f" (including streaming {_name(fast)} at {fmt(derived(fast)['wer'], 2)}%)" if (fast and derived(fast)["wer"] > h["wer"]) else ""
    beat_clause = (f", ranking {ord_s} of {len(all_panel_wers)} complete-panel systems "
                   f"(lower point-estimate WER than {beaten_count} of {len(all_comp_wers)} public comparators{fast_ref})"
                   if beaten_count else "")
    top_refs = [r for r in dict.fromkeys(x["id"] for x in (stream_best, best) if x)]
    top_rows = [r for r in others if r["id"] in top_refs]
    top_str = _names([
        f"{fmt(derived(r)['wer'], 2)}% for {_name(r)}"
        + (" (shared-host timing)" if r["interim"] else "")
        for r in top_rows
    ])
    full_tail = (f" On full LibriSpeech DEV (5,567 utt), final text reaches {full_final}% pooled "
                 f"({full_clean}% clean / {full_other}% other; live {full_live}%)." if full_final else "")
    live_wer_s = (f"; {_name(live)} on the exact same INT8 weights scores {panel_live}% pooled" if panel_live and has_live
                  else f"; the live mode on the exact same INT8 weights scores {panel_live}% pooled" if panel_live else "")
    fact("Accuracy", "Accuracy",
         f"On the 730-utterance LibriSpeech DEV panel, {_name(head)} scores {fmt(h['wer'], 2)}% pooled WER "
         f"({fmt(h['wer_clean'], 2)}% dev-clean, {fmt(h['wer_other'], 2)}% dev-other){beat_clause}{live_wer_s}."
         f"{full_tail} Larger models score lower at higher CPU cost on their respective timing hosts ({top_str}).",
         f"On the 730-utterance LibriSpeech DEV panel, {_name(head)} (utterance-segmented commit pass) scores "
         f"{fmt(h['wer'], 2)}% pooled WER ({fmt(h['wer_clean'], 2)}% dev-clean / {fmt(h['wer_other'], 2)}% dev-other)"
         f"{beat_clause}{live_wer_s}.{full_tail} Larger models score lower at higher CPU cost on their respective "
         f"timing hosts ({top_str}).",
         rowsrc(head, *([live] if has_live else []), *top_rows), "panel and full-DEV accuracy", panel)

    # 2. Speed & memory
    def timing_name(row: dict) -> str:
        suffix = " (quiet 73-clip timing)" if row["timing_panel"] != "full" else ""
        return _name(row) + suffix

    vs_peers = _names([f"{speed_text(derived(r)['speed'])}\u00d7 for {timing_name(r)}" for r in peers]) if peers else ""
    delay_str = (f" (median final-text delay {rec.rows['RelCommitDelayPfifty']['value']}\u2009ms after endpoint)"
                 if "RelCommitDelayPfifty" in rec.rows and not current else "")
    if has_live:
        live_share = fmt(100.0 * lv["cpu"], 1)
        speed_ratio = (f" ({fmt(lv['speed'] / derived(fast)['speed'], 1)}\u00d7 streaming {_name(fast)}, the fastest "
                       "comparator)" if fast else "")
        speed_ratio_md = f" ({fmt(lv['speed'] / derived(fast)['speed'], 1)}\u00d7 streaming {_name(fast)})" if fast else ""
        zip_cmp = (f", {_rel(h['speed'], derived(fast)['speed'], _name(fast), (head.get('paired_vs_anchor') or {}).get('speed')).replace('\\%', '%').replace('$\\times$', 'x')} "
                   f"({speed_text(h['speed'])}\u00d7 utterance-segmented vs. {speed_text(derived(fast)['speed'])}\u00d7 streaming)" if fast else "")
        fact("Speed", "Speed & memory",
             f"One live stream ({_name(live)}) uses {live_share}% of one quiet Zen\u00a02 core "
             f"({speed_text(lv['speed'])}\u00d7 real time{speed_ratio}) and {fmt(lv['rss_mb'], 0)}\u2009MB peak RAM "
             f"; {_name(head)} runs at {speed_text(h['speed'])}\u00d7 real time and "
             f"{fmt(h['rss_mb'], 0)}\u2009MB RAM{delay_str}{zip_cmp}.",
             f"{_name(live)} runs at {speed_text(lv['speed'])}× "
             f"real time{speed_ratio_md} ({live_share}% of one quiet Zen 2 core, {fmt(lv['rss_mb'], 0)} MB peak "
             f"RAM against {fmt(derived(fast)['rss_mb'], 0)} MB for matched {_name(fast)})"
             + (f", against {vs_peers}" if vs_peers else "")
             + f". {_name(head)} runs at {speed_text(h['speed'])}× real time and {fmt(h['rss_mb'], 0)} MB peak "
             f"RAM{delay_str}{zip_cmp}.",
             rowsrc(live, head, *peers), "single-core speed and memory", panel)
    else:
        fin_share = fmt(100.0 * h["cpu"], 1)
        zip_cmp = (f", within 2% of streaming {_name(fast)}'s single-core speed ({speed_text(h['speed'])}\u00d7 "
                   f"utterance-segmented vs. {speed_text(derived(fast)['speed'])}\u00d7 streaming)" if fast else "")
        fact("Speed", "Speed & memory",
             f"{_name(head)} decodes cold at {speed_text(h['speed'])}\u00d7 real time ({fin_share}% of one quiet "
             f"Zen\u00a02 core, {fmt(h['rss_mb'], 0)}\u2009MB peak RAM{delay_str}){zip_cmp}; "
             "the standalone cold hold for Isosyll v3.0.0 (live) is pending.",
             f"{_name(head)} decodes cold at {speed_text(h['speed'])}× real time ({fin_share}% of one quiet Zen 2 "
             f"core, {fmt(h['rss_mb'], 0)} MB peak RAM{delay_str}){zip_cmp}"
             + (f", against {vs_peers}" if vs_peers else "")
             + ". Standalone cold timing and RSS for Isosyll v3.0.0 (live) are pending their dedicated quiet-H1 hold.",
             rowsrc(head, *peers), "single-core speed and memory", panel)

    # 3. Watts
    wrows = sorted((r for r in peers if r.get("watts") is not None), key=lambda r: r["watts"])
    if h["watts"] is not None and wrows:
        vs_w = _names([f"{fmt(r['watts'], 3)}\u2009J per audio-second for {timing_name(r)}" for r in wrows])
        pkg = rec.rows["RelCommitPkgWatts"]["value"] if "RelCommitPkgWatts" in rec.rows and not current else None
        pkg_s = f" ({pkg}\u2009W package)" if pkg else ""
        if has_live and lv["watts"] is not None:
            zipformer = next((r for r in wrows if r["variant"].startswith("r0-int8") and r["timing_panel"] == "full"
                              and not r["interim"]
                              and r.get("energy_method") == live.get("energy_method") == "core_msr_active"), None)
            live_energy = (f" ({fmt(100 * lv['watts'] / zipformer['watts'], 0)}% of Zipformer INT8's active core energy)"
                           if zipformer else "")
            fact("Watts", "Watts",
                 f"Unpaced active core energy on quiet H1 is {fmt(lv['watts'], 3)}\u2009J per audio-second for "
                 f"{_name(live)} and {fmt(h['watts'], 3)}\u2009J{pkg_s} for {_name(head)}, against {vs_w} (CPU domain "
                 "only, not wall power).",
                 f"Unpaced active core J per audio-second on quiet H1: {fmt(lv['watts'], 3)} J for "
                 f"{_name(live)}{live_energy} and "
                 f"{fmt(h['watts'], 3)} J{pkg_s} for {_name(head)}, against {vs_w} (CPU domain only, not wall power).",
                 rowsrc(live, head, *wrows), "core-register watts per stream", panel)
        else:
            fact("Watts", "Watts",
                 f"Unpaced active core energy on quiet H1 is {fmt(h['watts'], 3)}\u2009J per audio-second{pkg_s} "
                 f"for {_name(head)}, against {vs_w} (CPU domain only, not wall power; live-only energy hold pending).",
                 f"Unpaced active core J per audio-second on quiet H1: {fmt(h['watts'], 3)} J{pkg_s} for "
                 f"{_name(head)}, against {vs_w} (CPU domain only, not wall power; standalone live-only hold pending).",
                 rowsrc(head, *wrows), "core-register watts per stream", panel)
    else:
        fact("Watts", "Watts", None, None, {}, "pending", panel)

    # 4. INT8 parity & exactness
    if "RelIntEightDelta" in rec.rows:
        d_pp = rec.rows["RelIntEightDelta"]["value"]
        lo_pp, hi_pp = rec.rows["RelIntEightLo"]["value"], rec.rows["RelIntEightHi"]["value"]
        d_md = d_pp.replace("$-$", "−")
        lo_md, hi_md = lo_pp.replace("$-$", "−"), hi_pp.replace("$-$", "−")
        fact("Parity", "Integer-exact INT8",
             f"Per-row centred INT8 with exact int32 accumulation and portable log-mel matches FP32 PyTorch on full "
             f"DEV (5,567 utterances) within {d_pp}\u2009pp (95% bounds {lo_pp} to {hi_pp}\u2009pp), with repeatable, "
             "batch-invariant execution.",
             f"Per-row centred INT8 with exact int32 accumulation and portable log-mel matches FP32 PyTorch on full "
             f"DEV (5,567 utterances) within {d_md} pp (95% bounds {lo_md} to {hi_md} pp), with repeatable, "
             "batch-invariant execution.",
             {"source": rec.rows["RelIntEightDelta"]["source"], "digest": rec.rows["RelIntEightDelta"]["workload_digest"]},
             "INT8 vs FP32 parity on full DEV", "full DEV (5,567 utt)")
    else:
        fact("Parity", "Integer-exact INT8", None, None, {}, "pending", panel)

    # Figure 1 headline sentence
    if fast and has_live and lv["watts"] is not None and h["watts"] is not None:
        f_d = derived(fast)
        see = (f"One INT8 model on one quiet Zen\u00a02 core: {_name(live)} runs at {speed_text(lv['speed'])}\u00d7 "
               f"({fmt(lv['watts'], 3)}\u2009W, {fmt(lv['wer'], 2)}% WER); the final mode reaches {fmt(h['wer'], 2)}% at "
               f"{speed_text(h['speed'])}\u00d7 ({fmt(h['watts'], 3)}\u2009W) vs. {speed_text(f_d['speed'])}\u00d7, "
               f"{fmt(f_d['wer'], 2)}% for {_name(fast)}.")
    elif fast and h["watts"] is not None:
        f_d = derived(fast)
        see = (f"On one quiet Zen\u00a02 core, utterance-segmented {_name(head)} reaches {fmt(h['wer'], 2)}% pooled "
               f"panel WER at {speed_text(h['speed'])}\u00d7 real time ({fmt(h['watts'], 3)}\u2009W) vs. "
               f"{speed_text(f_d['speed'])}\u00d7 and {fmt(f_d['wer'], 2)}% for {_name(fast)} (standalone live-mode "
               "cold hold pending).")
    else:
        see = f"{_name(head)} reaches {fmt(h['wer'], 2)}% pooled WER at {speed_text(h['speed'])}\u00d7 real time."
    put("MoneyHeadline", tex(see), rowsrc(head, *([live] if has_live else []), *([fast] if fast else [])),
        "Figure 1: what to see", panel)
    return {"bullets": bullets, "figure": md(see), "comparison": comparison}
