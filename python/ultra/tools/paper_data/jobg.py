"""Job G (ultra 43f31a1): the headline dense checkpoint, avg of steps 55/60/65k, greedy and beam 4.

Job G ran on a more heavily loaded xbabe0: its R0 claimed only 16 streams/CCX against 32 in runs A/D/E,
so no ratio is formed with Job G's R0. Ratios use Job E's same-session R0; MR (144) reproduced again."""

from __future__ import annotations

import json

from .common import CANDIDATES, DATA, LADDER_WORKLOAD, PaperDataError, RAW, csv_text, fmt
from .records import Records

NEW_DENSE = "/home/ubuntu/jsr-engine-data/runs/d2-avg3-55-60-65/weights-int8prep"
NEW_DISP = "D2 avg-3 (55/60/65k) dense INT8"
G_RUNS = (("ladder-g1-d2avg556065-greedy", "Dense", "greedy"), ("ladder-g2-d2avg556065-beam4", "DenseBeam", "beam 4"))


def _ladder(root, sub):
    base = RAW / "job_g" / sub
    lad = json.loads((root / base / "density_ladder.json").read_text())
    pts = [json.loads(x) for x in (root / base / "density_ladder.points.jsonl").read_text().splitlines() if x]
    mans = [json.loads(p.read_text()) for p in sorted((root / base).glob("*.job_manifest.json"))]
    return base, lad, pts, mans


def _summary(lad, pts, cand):
    res = next(r for r in lad["results"] if r["candidate"] == cand)
    claim = res["claimed_total_streams"]
    mine = sorted((p for p in pts if p["candidate"] == cand), key=lambda p: p["total_streams"])
    ok = [p for p in mine if p["all_sustained"] and p["drained"]]
    if max(p["total_streams"] for p in ok) != claim:
        raise PaperDataError(f"Job G {cand}: claimed {claim} disagrees with the points file")
    at = next(p for p in mine if p["total_streams"] == claim)
    above = [p for p in mine if p["total_streams"] > claim]
    return claim, at, (above[0] if above else None), mine


def add_job_g(root, rec: Records, out: dict[str, str]) -> None:
    exes, npzs, beams = set(), set(), {}
    for sub, stem, mode in G_RUNS:
        base, lad, pts, mans = _ladder(root, sub)
        for m in mans:
            exes.add(m["executable_sha256"])
            npzs.add(m["npz_sha256"])
            a = m["cli_args"]
            if m["candidate"] == NEW_DENSE:
                beams[sub] = a[a.index("--beam") + 1] if "--beam" in a else "1"
        if lad["completion_status"] != "exploratory" or lad["confirm_seconds"] or lad["repeats_required"] != 1:
            raise PaperDataError(f"Job G {sub}: unexpected ladder protocol")
    if len(exes) != 1 or len(npzs) != 1:
        raise PaperDataError(f"Job G ladders must share one executable and audio cache: {exes} {npzs}")
    if beams != {G_RUNS[0][0]: "1", G_RUNS[1][0]: "4"}:
        raise PaperDataError(f"Job G beam settings unexpected: {beams}")
    npz = npzs.pop()
    common = dict(label="exploratory", workload=LADDER_WORKLOAD, workload_digest=npz)
    hw = "xbabe0 AMD TR PRO 3995WX (Zen 2), shared host under heavier load than Job E"
    claims = {}
    for sub, stem, mode in G_RUNS:
        base, lad, pts, _ = _ladder(root, sub)
        lsrc = f"docs/paper/data/{base.relative_to(DATA)}"
        claim, at, fail, mine = _summary(lad, pts, NEW_DENSE)
        claims[stem] = claim
        disp = f"{NEW_DISP}, {mode}"
        conf = (f"{disp}; Job G ladder 4:48:4, {lad['workers']} single-thread workers on one CCX, "
                f"{lad['step_seconds']:.0f} s rungs, one repeat; ultra 43f31a1; {hw}")
        rec.add(f"Den{stem}CCX", str(claim), raw=claim, unit="streams/CCX", configuration=f"{conf}; highest sustained rung",
                source=f"{lsrc}/density_ladder.json results[candidate={NEW_DENSE}]", **common)
        rec.add(f"Den{stem}Core", fmt(claim / lad["workers"], 0), raw=claim / lad["workers"], unit="streams/core",
                configuration=f"{disp}; claimed streams / {lad['workers']} workers (Job G)",
                source=f"{lsrc}/density_ladder.json", **common)
        rec.add(f"DenPninetyNine{stem}", fmt(at["max_worker_p99_ms"], 0), raw=at["max_worker_p99_ms"], unit="ms",
                configuration=f"{disp}; worst-worker p99 at the claimed rung (Job G)",
                source=f"{lsrc}/density_ladder.points.jsonl total_streams={claim}", **common)
        if fail:
            rec.add(f"DenFail{stem}CCX", str(fail["total_streams"]), raw=fail["total_streams"], unit="streams/CCX",
                    configuration=f"{disp}; first failed rung (Job G)",
                    source=f"{lsrc}/density_ladder.points.jsonl total_streams={fail['total_streams']}", **common)
            rec.add(f"DenFailPninetyNine{stem}", fmt(fail["max_worker_p99_ms"], 0), raw=fail["max_worker_p99_ms"],
                    unit="ms", configuration=f"{disp}; worst-worker p99 at the first failed rung (Job G)",
                    source=f"{lsrc}/density_ladder.points.jsonl total_streams={fail['total_streams']}", **common)
        out[f"ladder_{stem.lower()}_jobg.csv"] = csv_text(
            ["total_streams", "p99_ms", "sustained"],
            [[p["total_streams"], fmt(p["max_worker_p99_ms"], 2), int(bool(p["all_sustained"] and p["drained"]))] for p in mine])
        if sub == G_RUNS[0][0]:
            r_claim, _, r_fail, _ = _summary(lad, pts, CANDIDATES[0][1])
            m_claim, _, _, _ = _summary(lad, pts, CANDIDATES[2][1])
            gsrc = f"{lsrc}/density_ladder.json"
            rec.add("RzeroLoadedCCX", str(r_claim), raw=r_claim, unit="streams/CCX",
                    configuration=(f"R0 Zipformer INT8 in Job G only (heavily loaded host); NOT used for any ratio, "
                                   f"against {rec.raw('DenRzeroCCX'):.0f} in the clean runs A/D/E"),
                    source=gsrc, **common)
            rec.add("RzeroLoadedFailCCX", str(r_fail["total_streams"]), raw=r_fail["total_streams"], unit="streams/CCX",
                    configuration="R0 first failed rung in Job G (loaded host)", source=gsrc, **common)
            rec.add("RzeroLoadedFailPninetyNine", fmt(r_fail["max_worker_p99_ms"], 0), raw=r_fail["max_worker_p99_ms"],
                    unit="ms", configuration="R0 worst-worker p99 at its first failed rung in Job G (loaded host)",
                    source=f"{lsrc}/density_ladder.points.jsonl", **common)
            if m_claim != rec.raw("DenMRCCX"):
                raise PaperDataError(f"MR density changed in Job G ({m_claim}); revisit DenReproRunsMR")
            runs = int(rec.raw("DenReproRuns")) + 1
            rec.add("DenReproRunsMR", str(runs), raw=runs, unit="runs",
                    configuration=f"ladder runs (A, D, E, G) in which MR claimed {m_claim} streams/CCX", source=gsrc, **common)
    if claims["Dense"] != claims["DenseBeam"]:
        raise PaperDataError("greedy and beam-4 densities differ; the 'beam costs no density' wording no longer holds")
    r0 = rec.rows["DenRzeroCCX"]
    ratio = claims["Dense"] / rec.raw("DenRzeroCCX")
    rec.add("DenRatioDense", fmt(ratio, 1), raw=ratio, unit="x",
            configuration=(f"{NEW_DISP} density in Job G over R0's density in the clean same-session Job E "
                           f"({rec.raw('DenRzeroCCX'):.0f}); Job G's own R0 (loaded host) is not used; the same-session "
                           "Job E ratio with the older dense average is DenRatioDenseOld"),
            source=f"derived: Job G + {r0['source']}", **common)
    rf = rec.raw("DenFailRzeroCCX")
    df = rec.raw("DenFailDenseCCX")
    for rid, v, conf in (("DenRatioDenseBracketLo", claims["Dense"] / (rf - 1),
                          f"dense claim {claims['Dense']} (Job G) / (R0 first failed rung {rf:.0f} - 1, Job E)"),
                         ("DenRatioDenseBracketHi", (df - 1) / rec.raw("DenRzeroCCX"),
                          f"(dense first failed rung {df:.0f} - 1, Job G) / R0 claim {rec.raw('DenRzeroCCX'):.0f} (Job E)")):
        rec.add(rid, fmt(v, 1), raw=v, unit="x", configuration=f"rung-spacing bracket: {conf}; different runs",
                source="derived: Job G + Job E ladders", **common)
