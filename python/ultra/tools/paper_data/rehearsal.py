"""Warm single-harness rehearsal: Job E ladder, Job C panel frontier, Job B ABAB."""

from __future__ import annotations

from pathlib import Path
import json

from .common import (
    ABAB_WORKLOAD,
    CANDIDATES,
    DATA,
    FRONT_DENSE,
    LADDER_WORKLOAD,
    MATCHED_RUNG,
    PANEL_DIGEST,
    PaperDataError,
    RAW,
    csv_text,
    fmt,
    read_json,
    sci_tex,
)
from .derive import add_equal_load_probe, add_ladder_runs
from .records import Records


def add_rehearsal(root: Path, rec: Records, out: dict[str, str]) -> None:
    """Warm single-harness rehearsal (exploratory): Job E ladder (main), Jobs A/D (dense step 45k, range),
    Job C panel frontier, Job B ABAB.

    Jobs C, D and E ran the same `ultra` executable; Job B ran 4fafb45 and Job A 57fc3c5."""
    ldir = RAW / "ladder_e_ce58952_avg3"
    ladder = read_json(root, str(ldir / "density_ladder.json"))
    points = [json.loads(x) for x in (root / ldir / "density_ladder.points.jsonl").read_text().splitlines() if x]
    abab = read_json(root, str(RAW / "abab_timing.json"))
    panel = read_json(root, str(RAW / "frontier_panel.json"))
    lsrc = f"docs/paper/data/{ldir.relative_to(DATA)}/density_ladder.json"
    psrc = f"docs/paper/data/{ldir.relative_to(DATA)}/density_ladder.points.jsonl"
    asrc = "docs/paper/data/raw/abab_timing.json"
    fsrc = "docs/paper/data/raw/frontier_panel.json"
    man = {}
    for sub in ("abab_manifests", "ladder_d_ce58952", "ladder_e_ce58952_avg3", "panel_manifests"):
        for p in sorted((root / RAW / sub).glob("*.job_manifest.json")):
            j = json.loads(p.read_text())
            man[(sub, j["candidate"])] = j
    npz = {j["npz_sha256"] for j in man.values()}
    if len(npz) != 1:
        raise PaperDataError(f"rehearsal jobs ran on different audio caches: {npz}")
    npz_sha = npz.pop()
    exe = {j["executable_sha256"] for (sub, _), j in man.items() if sub != "abab_manifests"}
    if len(exe) != 1:
        raise PaperDataError(f"ladder and panel frontier must share one executable, got {exe}")
    if panel.get("panel_manifest_sha256") != PANEL_DIGEST:
        raise PaperDataError("frontier_panel.json is not on the frozen DEV panel")
    for d in (ladder, abab, panel):
        if d["completion_status"] != "exploratory":
            raise PaperDataError("rehearsal inputs must be labelled exploratory")
    hw = "xbabe0 AMD TR PRO 3995WX (Zen 2), shared host, nice 10"
    lconf = (f"density ladder 4:48:4, {ladder['workers']} single-thread workers on one CCX, observer on a separate CCX, "
             f"{ladder['step_seconds']:.0f} s rungs, one repeat, grid {ladder['grid_ms']} ms, whole-service fleet CPU; "
             f"ultra ce58952 (post-H15 sparse mel); {hw}")
    pconf = f"bench-suite --panel, dev-clean half, threads 1, grid {panel['grid_ms']} ms, warm in-process rusage; ultra ce58952; {hw}"
    pwl = "DEV selection panel, dev-clean half (400 utt, 3,052.165 audio-s)"

    by_cand = {r["candidate"]: r for r in ladder["results"]}
    rows_cost, rows_ladder = [], []
    claims, early, fails = {}, {}, {}
    for key, cand, disp, stem in CANDIDATES:
        # Job E's dense row is the older avg-3 (45/50/55k): its ladder records carry the "DenseOld" stem;
        # the headline dense row (avg 55/60/65k, Job G) is added by jobg.py. Frontier rows keep "Dense" (45k).
        lstem = "DenseOld" if key == "dense" else stem
        res = by_cand[cand]
        pts = sorted((p for p in points if p["candidate"] == cand), key=lambda p: p["total_streams"])
        ok = [p for p in pts if p["all_sustained"] and p["drained"]]
        claim = res["claimed_total_streams"]
        if max(p["total_streams"] for p in ok) != claim:
            raise PaperDataError(f"{cand}: claimed {claim} disagrees with the points file")
        first_fail = next((p for p in pts if not (p["all_sustained"] and p["drained"])), None)
        before_fail = max(p["total_streams"] for p in ok if first_fail is None or p["total_streams"] < first_fail["total_streams"])
        above = [p for p in pts if p["total_streams"] > claim]
        at_claim = next(p for p in pts if p["total_streams"] == claim)
        at_match = next(p for p in pts if p["total_streams"] == MATCHED_RUNG)
        if not (at_match["all_sustained"] and at_match["drained"]):
            raise PaperDataError(f"{cand} did not sustain the matched rung {MATCHED_RUNG}")
        claims[key] = (claim, at_match["rtf_service_cpu"])
        early[key] = before_fail
        wl, dg = LADDER_WORKLOAD, npz_sha
        common = dict(label="exploratory", workload=wl, workload_digest=dg)
        rec.add(f"Den{lstem}CCX", str(claim), raw=claim, unit="streams/CCX",
                configuration=f"{disp}; highest sustained rung (tool claim); {lconf}",
                source=f"{lsrc} results[candidate={cand}].claimed_total_streams", **common)
        rec.add(f"Den{lstem}BeforeFailCCX", str(before_fail), raw=before_fail, unit="streams/CCX",
                configuration=f"{disp}; last sustained rung before the first failing rung (conservative)",
                source=f"{psrc} candidate={cand}", **common)
        rec.add(f"Den{lstem}Core", fmt(claim / ladder["workers"], 0), raw=claim / ladder["workers"],
                unit="streams/core", configuration=f"{disp}; claimed streams / {ladder['workers']} workers",
                source=f"{lsrc} results[candidate={cand}]", **common)
        rec.add(f"DenPninetyNine{lstem}", fmt(at_claim["max_worker_p99_ms"], 0), raw=at_claim["max_worker_p99_ms"],
                unit="ms", configuration=f"{disp}; max worker p99 lateness at the claimed rung",
                source=f"{psrc} candidate={cand} total_streams={claim}", **common)
        rec.add(f"DenRss{lstem}", fmt(at_claim["peak_rss_bytes"] / 1e9, 2), raw=at_claim["peak_rss_bytes"], unit="GB",
                configuration=f"{disp}; peak RSS of the whole 4-worker fleet at the claimed rung",
                source=f"{psrc} candidate={cand} total_streams={claim}", **common)
        if above:
            rec.add(f"DenFail{lstem}CCX", str(above[0]["total_streams"]), raw=above[0]["total_streams"],
                    unit="streams/CCX", configuration=f"{disp}; first rung above the claim (failed)",
                    source=f"{psrc} candidate={cand} total_streams={above[0]['total_streams']}", **common)
            rec.add(f"DenFailPninetyNine{lstem}", fmt(above[0]["max_worker_p99_ms"], 0), raw=above[0]["max_worker_p99_ms"],
                    unit="ms", configuration=f"{disp}; max worker p99 lateness at the first failed rung",
                    source=f"{psrc} candidate={cand} total_streams={above[0]['total_streams']}", **common)
        inv = 1.0 / at_match["rtf_service_cpu"]
        rec.add(f"AudioPerCpu{lstem}", fmt(inv, 1), raw=inv, unit="audio-s per CPU-s",
                configuration=f"{disp}; 1 / whole-service CPU-s per audio-s at the matched {MATCHED_RUNG}-stream rung",
                source=f"derived: {psrc} candidate={cand} total_streams={MATCHED_RUNG}", **common)
        rec.add(f"RtfxLadder{lstem}", fmt(inv, 1), raw=inv, unit="x real time per CPU core",
                configuration=f"{disp}; same as RTFxLadder{lstem} (paper-writer spelling)",
                source=f"derived: {psrc} candidate={cand} total_streams={MATCHED_RUNG}", **common)
        rec.add(f"RTFxLadder{lstem}", fmt(inv, 1), raw=inv, unit="x real time per CPU core",
                configuration=f"{disp}; audio-seconds processed per CPU-second at the matched {MATCHED_RUNG}-stream rung "
                              "(1 / CostLadder)",
                source=f"derived: {psrc} candidate={cand} total_streams={MATCHED_RUNG}", **common)
        util = 100.0 * claim * at_claim["rtf_service_cpu"] / ladder["workers"]
        rec.add(f"UtilClaim{lstem}", fmt(util, 0), raw=util, unit="% projected CPU demand of the CCX's 4 cores",
                configuration=(f"{disp}; PROJECTED CPU DEMAND, not sampled utilisation: nominal claimed streams x "
                               f"CPU-s per audio-s at the claimed rung / {ladder['workers']} cores"),
                source=f"derived: {psrc} candidate={cand} total_streams={claim}", **common)
        cores_s = ladder["workers"] * ladder["step_seconds"]
        for tag, pt in (("", at_claim), ("Fail", above[0] if above else None)):
            if pt is None:
                continue
            dem = 100.0 * pt["audio_processed_seconds"] * pt["rtf_service_cpu"] / cores_s
            rec.add(f"Demand{tag}{lstem}", fmt(dem, 2), raw=dem, unit="% CPU demand of the CCX's 4 cores",
                    configuration=(f"{disp}; processed-audio CPU demand at the {'first failed' if tag else 'claimed'} rung "
                                   f"({pt['total_streams']} streams): processed audio-s x CPU-s per audio-s / "
                                   f"({ladder['workers']} cores x {ladder['step_seconds']:.0f} s)"),
                    source=f"derived: {psrc} candidate={cand} total_streams={pt['total_streams']}", **common)
        rec.add(f"AudioProcessed{lstem}", fmt(at_claim["audio_processed_seconds"], 2), raw=at_claim["audio_processed_seconds"],
                unit="audio-s", configuration=f"{disp}; audio processed during the {ladder['step_seconds']:.0f} s claimed rung",
                source=f"{psrc} candidate={cand} total_streams={claim}", **common)
        rec.add(f"AudioUnfinished{lstem}", fmt(at_claim["audio_unfinished_seconds"], 2),
                raw=at_claim["audio_unfinished_seconds"], unit="audio-s",
                configuration=f"{disp}; accepted audio left unprocessed at the claimed rung's cutoff",
                source=f"{psrc} candidate={cand} total_streams={claim}", **common)
        fails[key] = (claim, above[0]["total_streams"] if above else None)
        rec.add(f"CostLadder{lstem}", fmt(at_match["rtf_service_cpu"], 4), raw=at_match["rtf_service_cpu"],
                unit="CPU-s/audio-s", configuration=f"{disp}; whole-service CPU at the matched {MATCHED_RUNG}-stream rung; {lconf}",
                source=f"{psrc} candidate={cand} total_streams={MATCHED_RUNG}", **common)
        rec.add(f"CostClaim{lstem}", fmt(at_claim["rtf_service_cpu"], 4), raw=at_claim["rtf_service_cpu"],
                unit="CPU-s/audio-s", configuration=f"{disp}; whole-service CPU at its claimed rung",
                source=f"{psrc} candidate={cand} total_streams={claim}", **common)
        fcand = FRONT_DENSE if key == "dense" else cand
        fdisp = "D2-45k dense INT8" if key == "dense" else disp
        fr = next(r for r in panel["rows"] if r["candidate"] == fcand)
        pcommon = dict(label="exploratory", workload=pwl, workload_digest=PANEL_DIGEST)
        fsel = f"{fsrc} rows[candidate={fcand}]"
        rec.add(f"Frontier{stem}", fmt(fr["batch_rtf_cpu"], 4), raw=fr["batch_rtf_cpu"], unit="CPU-s/audio-s",
                configuration=f"{fdisp}; batch-16; {pconf}", source=f"{fsel}.batch_rtf_cpu", **pcommon)
        rec.add(f"FrontierPerCore{stem}", fmt(fr["batch_streams_per_core"], 1), raw=fr["batch_streams_per_core"],
                unit="streams/core", configuration=f"{fdisp}; 1/batch-16 cost", source=f"{fsel}.batch_streams_per_core", **pcommon)
        rec.add(f"FrontierSingle{stem}", fmt(fr["single_rtf_cpu"], 3), raw=fr["single_rtf_cpu"], unit="CPU-s/audio-s",
                configuration=f"{fdisp}; single stream; {pconf}", source=f"{fsel}.single_rtf_cpu", **pcommon)
        rec.add(f"FrontierChunkPninetyNine{stem}", fmt(fr["chunk_ms"]["p99"], 0), raw=fr["chunk_ms"]["p99"], unit="ms",
                configuration=f"{fdisp}; per-chunk compute p99, single stream", source=f"{fsel}.chunk_ms.p99", **pcommon)
        rec.add(f"WerRehearsal{stem}" + ("FortyFive" if key == "dense" else ""), fmt(fr["wer_pct"], 2), raw=fr["wer_pct"], unit="% WER",
                configuration=f"{fdisp}; same harness and executable as the ladder, Rust scorer, greedy",
                source=f"{fsel}.wer_pct", **pcommon)
        if key == "dense":
            wer = rec.raw("WerRehearsalDenseOld")
        else:
            wer = fr["wer_pct"]
        rows_cost.append([key, disp, fmt(at_match["rtf_service_cpu"], 5), fmt(wer, 3), claim, before_fail,
                          fmt(at_claim["rtf_service_cpu"], 5),
                          # the panel frontier timed the 45k dense checkpoint: never put it on the avg-3 row
                          "" if key == "dense" else fmt(fr["batch_rtf_cpu"], 5),
                          "" if key == "dense" else fmt(fr["single_rtf_cpu"], 5)])
        for p in pts:
            rows_ladder.append([key, p["total_streams"], fmt(p["rtf_service_cpu"], 5), fmt(p["max_worker_p99_ms"], 2),
                                int(bool(p["all_sustained"] and p["drained"])), fmt(p["peak_rss_bytes"] / 1e9, 3)])

    base_claim, base_cost = claims["rzero"]
    base_fb = next(r for r in panel["rows"] if r["candidate"] == CANDIDATES[0][1])["batch_rtf_cpu"]
    for key, stem in (("dense", "DenseOld"), ("mr", "MR")):
        c, cost = claims[key]
        common = dict(label="exploratory", workload=LADDER_WORKLOAD, workload_digest=npz_sha)
        rec.add(f"DenRatio{stem}", fmt(c / base_claim, 1), raw=c / base_claim, unit="x",
                configuration=f"claimed streams/CCX {key} over R0, same ladder", source=f"derived: {lsrc}", **common)
        rec.add(f"DenRatio{stem}BeforeFail", fmt(early[key] / early["rzero"], 1), raw=early[key] / early["rzero"], unit="x",
                configuration=f"last-rung-before-first-failure streams/CCX {key} over R0", source=f"derived: {psrc}", **common)
        rec.add(f"CostRatio{stem}", fmt(base_cost / cost, 2), raw=base_cost / cost, unit="x",
                configuration=f"R0 over {key} CPU-s/audio-s at the {MATCHED_RUNG}-stream rung", source=f"derived: {psrc}", **common)
        fcand = FRONT_DENSE if key == "dense" else dict((k, c) for k, c, _, _ in CANDIDATES)[key]
        fb = next(r for r in panel["rows"] if r["candidate"] == fcand)["batch_rtf_cpu"]
        rec.add(f"FrontierRatio{'Dense' if key == 'dense' else stem}", fmt(base_fb / fb, 2), raw=base_fb / fb, unit="x", label="exploratory",
                configuration=f"R0 over {key} batch-16 CPU-s/audio-s, panel frontier", workload=pwl,
                workload_digest=PANEL_DIGEST, source=f"derived: {fsrc}")
    (rc, rf), (mc, mf) = fails["rzero"], fails["mr"]
    if rf is None or mf is None:
        raise PaperDataError("density bracket needs a failed rung for R0 and MR")
    bcommon = dict(label="exploratory", workload=LADDER_WORKLOAD, workload_digest=npz_sha, source=f"derived: {psrc}")
    lo, hi = mc / (rf - 1), (mf - 1) / rc
    rec.add("DenRatioMRBracketLo", fmt(lo, 1), raw=lo, unit="x",
            configuration=f"rung-spacing bracket, low end: MR claim {mc} / (R0 first failed rung {rf} - 1)", **bcommon)
    rec.add("DenRatioMRBracketHi", fmt(hi, 1), raw=hi, unit="x",
            configuration=f"rung-spacing bracket, high end: (MR first failed rung {mf} - 1) / R0 claim {rc}", **bcommon)
    (dc, df) = fails["dense"]
    lo, hi = dc / (rf - 1), (df - 1) / rc
    rec.add("DenRatioDenseOldBracketLo", fmt(lo, 1), raw=lo, unit="x",
            configuration=f"rung-spacing bracket, low end: dense claim {dc} / (R0 first failed rung {rf} - 1)", **bcommon)
    rec.add("DenRatioDenseOldBracketHi", fmt(hi, 1), raw=hi, unit="x",
            configuration=f"rung-spacing bracket, high end: (dense first failed rung {df} - 1) / R0 claim {rc}", **bcommon)
    largs = man[("ladder_e_ce58952_avg3", CANDIDATES[0][1])]["cli_args"]
    limits = {tuple(j["cli_args"][j["cli_args"].index("--limit") + 1:j["cli_args"].index("--limit") + 2])
              for (sub, _), j in man.items() if sub == "ladder_e_ce58952_avg3"}
    if len(limits) != 1:
        raise PaperDataError(f"ladder jobs used different --limit values: {limits}")
    lim = largs[largs.index("--limit") + 1]
    rec.add("LadderUtts", str(lim), raw=int(lim), unit="utterances", label="reported",
            configuration="dev-clean utterances each ladder stream cycles, read from the Job E job manifests (--limit)",
            workload=LADDER_WORKLOAD, workload_digest=npz_sha,
            source=f"docs/paper/data/{ldir.relative_to(DATA)}/*.job_manifest.json cli_args --limit")
    rec.add("LadderWorkload", f"first {lim} utterances cycled per stream (\\texttt{{--limit {lim}}})", raw=int(lim),
            unit="text", label="reported", configuration="ladder workload, read from the Job E job manifests (cli_args)",
            workload=LADDER_WORKLOAD, workload_digest=npz_sha,
            source=f"docs/paper/data/{ldir.relative_to(DATA)}/*.job_manifest.json cli_args --limit")
    add_ladder_runs(root, rec, npz_sha)
    add_equal_load_probe(root, rec, npz_sha, ldir, ladder)
    for rid, v, unit, conf, src, wl in (
        ("DenWorkers", ladder["workers"], "workers", "single-threaded workers on one 4-core CCX", lsrc, LADDER_WORKLOAD),
        ("DenRungSeconds", int(ladder["step_seconds"]), "s", "duration of each ladder rung", lsrc, LADDER_WORKLOAD),
        ("MatchedRung", MATCHED_RUNG, "streams/CCX", "highest rung all three candidates sustained", psrc, LADDER_WORKLOAD),
        ("AbabBlocks", abab["blocks_count"], "blocks", "counterbalanced ABAB blocks", asrc, ABAB_WORKLOAD),
        ("FrontierStreams", 16, "streams", "batch size of the frontier batch row", fsrc, pwl),
    ):
        rec.add(rid, str(v), raw=v, unit=unit, label="reported", configuration=conf, workload=wl,
                workload_digest=PANEL_DIGEST if wl == pwl else npz_sha, source=src)

    for pair in abab["paired_vs_baseline"]:
        stem = "Dense" if "d2-45000" in pair["candidate"] else "MR"
        g = pair["geometric_mean_ratio"]
        lo, hi = pair["geometric_mean_ratio_ci95_t"]
        conf = (f"{stem} vs R0 single-stream warm in-process CPU, {pair['blocks_paired']} paired blocks, "
                f"geometric-mean ratio (secondary), Student-t 95% CI, Holm; ultra 4fafb45; {hw}")
        src = f"{asrc} paired_vs_baseline[candidate~{stem}]"
        common = dict(label="exploratory", configuration=conf, workload=ABAB_WORKLOAD, workload_digest=npz_sha, source=src)
        for rid, v, nd in ((f"AbabRatio{stem}", g, 3), (f"AbabRatio{stem}Lo", lo, 3), (f"AbabRatio{stem}Hi", hi, 3),
                           (f"AbabSpeed{stem}", 1 / g, 2)):
            rec.add(rid, fmt(v, nd), raw=v, unit="x", **common)
        rec.add(f"AbabPholm{stem}", sci_tex(pair["holm_p_value"]), raw=pair["holm_p_value"], unit="p", **common)
        # pre-registered primary statistic (FREEZE section 5): median paired ratio with a bootstrap 95% CI
        med = pair["median_ratio_cand_over_base"]
        mlo, mhi = pair["ratio_ci95_bootstrap"]
        mconf = (f"{stem} vs R0 single-stream warm in-process CPU, median of {pair['blocks_paired']} paired block ratios, "
                 f"bootstrap 95% CI ({pair['bootstrap_replicates']} replicates, seed {pair['seed']}); PRIMARY statistic; "
                 f"ultra 4fafb45; {hw}")
        mcommon = dict(label="exploratory", configuration=mconf, workload=ABAB_WORKLOAD, workload_digest=npz_sha, source=src)
        for rid, v, nd in ((f"AbabMedian{stem}", med, 3), (f"AbabMedian{stem}Lo", mlo, 3),
                           (f"AbabMedian{stem}Hi", mhi, 3), (f"AbabMedianSpeed{stem}", 1 / med, 2)):
            rec.add(rid, fmt(v, nd), raw=v, unit="x", **mcommon)

    out["rehearsal_cost.csv"] = csv_text(
        ["key", "system", "cpu_matched", "wer_panel_clean", "streams_ccx", "streams_before_fail",
         "cpu_at_claim", "cpu_batch16", "cpu_single"], rows_cost)
    out["rehearsal_ladder.csv"] = csv_text(["key", "total_streams", "cpu", "p99_ms", "sustained", "fleet_rss_gb"], rows_ladder)
    # one file per candidate for pgfplots (no filtering needed in TeX)
    for key, *_ in CANDIDATES:
        rows = [r for r in rows_ladder if r[0] == key]
        out[f"ladder_{key}.csv"] = csv_text(["total_streams", "p99_ms", "sustained"], [[r[1], r[3], r[4]] for r in rows])
