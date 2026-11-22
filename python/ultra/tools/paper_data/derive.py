"""Derived metrics: ratios and differences of curated numbers, ladder reproducibility, equal-load probe."""

from __future__ import annotations

from pathlib import Path
import json

from .common import (
    CANDIDATES,
    DATA,
    FRONT_DENSE,
    LADDER_RUNS,
    LADDER_WORKLOAD,
    PaperDataError,
    RAW,
    fmt,
    read_json,
)
from .records import Records


def add_derived(rec: Records) -> None:
    """Ratios and differences of curated numbers (same workload on both sides)."""
    def ratio(rid, a, b, nd, conf):
        x = rec.raw(a) / rec.raw(b)
        ra, rb = rec.rows[a], rec.rows[b]
        rec.add(rid, fmt(x, nd), raw=round(x, 6), unit="x", label="exploratory",
                configuration=f"{conf}; derived {a}/{b}", workload=ra["workload"],
                workload_digest=ra["workload_digest"], source=f"derived: {ra['source']} ; {rb['source']}")

    def diff(rid, a, b, nd, conf):
        x = rec.raw(a) - rec.raw(b)
        ra = rec.rows[a]
        rec.add(rid, fmt(x, nd), raw=round(x, 6), unit="pp WER", label="exploratory",
                configuration=f"{conf}; derived {a}-{b}", workload=ra["workload"],
                workload_digest=ra["workload_digest"], source=f"derived: {ra['source']}")

    ratio("SpeedupVsRzero", "CostRzero", "CostEngineIntEight", 1,
          "R0 vs engine INT8 whole-service CPU, same executable 6df5c8f")
    x = 1.0 / rec.raw("SingleMthreeMR")
    r0 = rec.rows["SingleMthreeMR"]
    rec.add("RTFxMthreeMR", fmt(x, 0), raw=round(x, 6), unit="x real time per CPU core", label="exploratory",
            configuration="1 / SingleMthreeMR: D2MR-final INT8 single stream on M3 Ultra (dev tool)",
            workload=r0["workload"], workload_digest=r0["workload_digest"], source=f"derived: {r0['source']}")
    # No cross-run ratios: ORT FP32 exists only in Run B (db5c14a), so it is compared only with Run B's engine FP32.
    ratio("SpeedupEngineFPVsORT", "CostORTRunB", "CostEngineFPRunB", 2,
          "Run B (db5c14a, one executable): ORT FP32 vs fused engine FP32, 400/400 identical transcripts")
    diff("BeamGainClean", "WerDtwoGreedyClean", "WerDtwoBeamClean", 2, "D2-60k beam 4 vs greedy, full dev-clean")
    diff("BeamGainOther", "WerDtwoGreedyOther", "WerDtwoBeamOther", 2, "D2-60k beam 4 vs greedy, full dev-other")
    diff("AvgGainClean", "WerDtwoGreedyClean", "WerDtwoAvgGreedyClean", 2,
         "D2 avg-3 (50/55/60k) vs single 60k, greedy, full dev-clean")
    diff("AvgGainOther", "WerDtwoGreedyOther", "WerDtwoAvgGreedyOther", 2,
         "D2 avg-3 vs single 60k, greedy, full dev-other")
    diff("StackGainClean", "WerDtwoGreedyClean", "WerDtwoBestClean", 2,
         "D2 avg-3 + beam 4 vs single 60k greedy, full dev-clean")
    diff("StackGainOther", "WerDtwoGreedyOther", "WerDtwoBestOther", 2,
         "D2 avg-3 + beam 4 vs single 60k greedy, full dev-other")


def add_equal_load_probe(root: Path, rec: Records, npz_sha: str, ldir: Path, ladder: dict) -> None:
    """Job E's fixed equal-load probe (same stream count for every system) and the ladder scope."""
    front = read_json(root, str(ldir / "frontier.json"))
    fsrc = f"docs/paper/data/{ldir.relative_to(DATA)}/frontier.json"
    probes = {}
    for key, cand, disp, stem in CANDIDATES:
        row = next(r for r in front["rows"] if r["candidate"] == cand)
        pr = row["sustained_probe"]
        man = json.loads((root / ldir / f"{cand.replace('/', '_')}.job_manifest.json").read_text())
        args = man["cli_args"]
        secs = int(args[args.index("--sustained-seconds") + 1])
        rcpt_name = f"{cand.replace('/', '_')}.sustained.receipt.json"
        rcpt = json.loads((root / ldir / rcpt_name).read_text())
        probes[key] = (pr["total_streams"], secs)
        common = dict(label="exploratory", workload=LADDER_WORKLOAD, workload_digest=npz_sha)
        conf = (f"{disp}; fixed equal-load probe, {pr['workers']} workers x {pr['streams_per_worker']} streams = "
                f"{pr['total_streams']} streams on one CCX for {secs} s (Job E, ultra ce58952)")
        src = f"{fsrc} rows[candidate={cand}].sustained_probe"
        rec.add(f"ProbePninetyNine{stem}", fmt(pr["lateness_p99_ms"], 0), raw=pr["lateness_p99_ms"], unit="ms",
                configuration=f"{conf}; worst-worker p99 lateness", source=src, **common)
        rec.add(f"ProbeSustained{stem}", "yes" if pr["sustained"] else "no", raw=int(bool(pr["sustained"])), unit="flag",
                configuration=f"{conf}; sustained under the A1 rule", source=src, **common)
        rec.add(f"ProbeDrained{stem}", "yes" if pr["drained"] else "no", raw=int(bool(pr["drained"])), unit="flag",
                configuration=f"{conf}; all accepted audio processed at cutoff", source=src, **common)
        cost = rcpt["process_tree_cpu_seconds"] / rcpt["audio_seconds"]
        rec.add(f"ProbeCost{stem}", fmt(cost, 4), raw=cost, unit="CPU-s/audio-s",
                configuration=f"{conf}; process-tree CPU over processed audio (observer excluded)",
                source=f"docs/paper/data/{ldir.relative_to(DATA)}/{rcpt_name}", **common)
    if len(set(probes.values())) != 1:
        raise PaperDataError(f"equal-load probe differs between systems: {probes}")
    n, secs = next(iter(probes.values()))
    common = dict(label="reported", workload=LADDER_WORKLOAD, workload_digest=npz_sha, source=fsrc)
    rec.add("ProbeStreams", str(n), raw=n, unit="streams/CCX", configuration="equal-load probe stream count, every system", **common)
    rec.add("ProbeSeconds", str(secs), raw=secs, unit="s", configuration="equal-load probe duration", **common)
    if ladder["confirm_seconds"] or ladder["repeats_required"] != 1:
        raise PaperDataError("ladder scope changed (confirmation or repeats); update LadderScope")
    scope = f"{ladder['step_seconds']:.0f}~s rungs, one repeat per rung, no confirmation"
    rec.add("LadderScope", scope, raw=None, unit="text", label="reported",
            configuration="ladder protocol of the exploratory rehearsal, from density_ladder.json "
                          "(step_seconds, repeats_required, confirm_seconds)",
            workload=LADDER_WORKLOAD, workload_digest=npz_sha,
            source=f"docs/paper/data/{ldir.relative_to(DATA)}/density_ladder.json")


def add_ladder_runs(root: Path, rec: Records, npz_sha: str) -> None:
    """Reproducibility across the three single-harness ladder runs (A, D, E)."""
    claims: dict[str, list[tuple[int, str]]] = {"rzero": [], "dense": [], "mr": []}
    for sub, dense, note in LADDER_RUNS:
        lad = read_json(root, str(RAW / sub / "density_ladder.json"))
        by = {r["candidate"]: r["claimed_total_streams"] for r in lad["results"]}
        claims["rzero"].append((by[CANDIDATES[0][1]], note))
        claims["dense"].append((by[dense], note))
        claims["mr"].append((by[CANDIDATES[2][1]], note))
    srcs = "; ".join(f"docs/paper/data/raw/{sub}/density_ladder.json" for sub, _, _ in LADDER_RUNS)
    common = dict(label="exploratory", workload=LADDER_WORKLOAD, workload_digest=npz_sha, source=srcs)
    runs = len(LADDER_RUNS)
    for key, stem in (("rzero", "Rzero"), ("mr", "MR")):
        vals = {v for v, _ in claims[key]}
        if len(vals) != 1:
            raise PaperDataError(f"{key} density differs across ladder runs: {claims[key]}; replace the reproducibility "
                                 "macros with a range")
    rec.add("DenReproRuns", str(runs), raw=runs, unit="runs",
            configuration=(f"single-harness ladder runs in which R0 ({claims['rzero'][0][0]}/CCX) and MR "
                           f"({claims['mr'][0][0]}/CCX) gave identical densities: " + "; ".join(n for _, n in claims["rzero"])),
            **common)
    base = claims["rzero"][0][0]
    # Step-45k dense was timed in two runs (A, D); the avg-3 finalist in one (E). Never pool the two checkpoints.
    d45 = [v for (v, n), (_, dense, _) in zip(claims["dense"], LADDER_RUNS) if dense == FRONT_DENSE]
    d45_notes = "; ".join(f"{n}: {v}" for (v, n), (_, dense, _) in zip(claims["dense"], LADDER_RUNS) if dense == FRONT_DENSE)
    src45 = "; ".join(f"docs/paper/data/raw/{sub}/density_ladder.json" for sub, dense, _ in LADDER_RUNS if dense == FRONT_DENSE)
    c45 = dict(label="exploratory", workload=LADDER_WORKLOAD, workload_digest=npz_sha, source=src45)
    rec.add("DenDenseFortyFiveRangeLo", str(min(d45)), raw=min(d45), unit="streams/CCX",
            configuration=f"dense step-45k checkpoint, lowest over its {len(d45)} ladder runs ({d45_notes})", **c45)
    rec.add("DenDenseFortyFiveRangeHi", str(max(d45)), raw=max(d45), unit="streams/CCX",
            configuration=f"dense step-45k checkpoint, highest over its {len(d45)} ladder runs ({d45_notes})", **c45)
    rec.add("DenRatioDenseFortyFiveRangeLo", fmt(min(d45) / base, 1), raw=min(d45) / base, unit="x",
            configuration="dense step-45k over R0, lowest over runs A/D", **c45)
    rec.add("DenRatioDenseFortyFiveRangeHi", fmt(max(d45) / base, 1), raw=max(d45) / base, unit="x",
            configuration="dense step-45k over R0, highest over runs A/D", **c45)
    rec.add("DenDenseFortyFiveRuns", str(len(d45)), raw=len(d45), unit="runs",
            configuration=f"ladder runs that timed dense step 45k ({d45_notes})", **c45)
    dD = dict((sub, v) for (v, _), (sub, _, _) in zip(claims["dense"], LADDER_RUNS))["ladder_d_ce58952"]
    cD = dict(label="exploratory", workload=LADDER_WORKLOAD, workload_digest=npz_sha,
              source="docs/paper/data/raw/ladder_d_ce58952/density_ladder.json")
    rec.add("DenDenseFortyFiveCCX", str(dD), raw=dD, unit="streams/CCX",
            configuration=("dense step 45k in Job D (ultra ce58952, the executable of the Job C panel-frontier row); "
                           f"the same checkpoint gave {min(d45)} in Job A, see DenDenseFortyFiveRange*"), **cD)
    rec.add("DenRatioDenseFortyFive", fmt(dD / base, 1), raw=dD / base, unit="x",
            configuration="dense step 45k over R0 in Job D", **cD)
    # MR over dense: different training runs (dense S vs multirate S_MR) and different dense checkpoints
    mr = claims["mr"][0][0]
    alld = [v for v, _ in claims["dense"]]
    cmv = dict(label="exploratory", workload=LADDER_WORKLOAD, workload_digest=npz_sha, source=srcs)
    rec.add("DenRatioMRvsDenseLo", fmt(mr / max(alld), 1), raw=mr / max(alld), unit="x",
            configuration=(f"MR {mr}/CCX over the highest dense value {max(alld)}; different training runs (dense S vs "
                           "multirate S_MR), not a matched architecture ablation"), **cmv)
    rec.add("DenRatioMRvsDenseHi", fmt(mr / min(alld), 1), raw=mr / min(alld), unit="x",
            configuration=(f"MR {mr}/CCX over the lowest dense value {min(alld)}; different training runs (dense S vs "
                           "multirate S_MR), not a matched architecture ablation"), **cmv)
