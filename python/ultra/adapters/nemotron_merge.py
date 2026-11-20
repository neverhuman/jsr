"""Pool sharded nemotron_run.py results into one corpus line per run identity.

A group is one run identity: split, chunk, concurrency, threads, shard count K, tag, runtime
commit, harness commit and model path. It is pooled only when its shard labels are exactly
{0/K, ..., K-1/K}, each label appears once, and the per-utterance id sets are disjoint.
Duplicate shard labels in one identity are an error, not "take the newest". Groups that are
not complete are reported with complete=false and the missing labels, and are not pooled.

WER pools S/D/I over shards, which equals scoring the whole set at once. CPU and audio are
summed. RSS is the per-process max. With --score-records DIR, each complete group's
per-utterance rows are written as ultra-eval ScoreRecord JSONL ({id, speaker, dataset, ref,
hyp, duration_s}) for Rust rescoring with `ultra-score`.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict


class MergeError(ValueError):
    pass


def _tag(d: dict) -> str:
    if "tag" in d:
        return d["tag"]
    m = re.search(r"_n\d+(?:_s\d+of\d+)?(?:_(.*))?$", d["run_id"])  # legacy lines without a tag field
    return (m.group(1) or "") if m else ""


def _model(d: dict) -> str:
    argv = d.get("argv", [])
    return argv[argv.index("--model") + 1] if "--model" in argv else ""


def identity(d: dict) -> tuple:
    k = d.get("shard", "0/1").split("/")[1]
    return (d["split"], d["chunk_ms"], d["concurrency"], d["threads"], int(k), _tag(d),
            d.get("runtime_commit", ""), d.get("harness_commit", ""), _model(d))


def _rows(d: dict) -> list[dict]:
    path = d.get("per_utterance")
    if not path or not os.path.exists(path):
        raise MergeError(f"{d['run_id']}: per-utterance file missing: {path}")
    return [json.loads(x) for x in open(path) if x.strip()]


def _score_record(row: dict, split: str) -> dict:
    uid = row.get("id", row.get("uid"))
    speaker = row.get("speaker") or uid.split("-")[0]  # LibriSpeech ids are SPEAKER-CHAPTER-UTT
    return {"id": uid, "speaker": str(speaker), "dataset": row.get("dataset", split),
            "ref": row["ref"], "hyp": row["hyp"], "duration_s": row.get("duration_s", row.get("seconds"))}


def merge(lines: list[dict], check_rows: bool = True) -> tuple[list[dict], dict[tuple, list[dict]]]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for d in lines:
        groups[identity(d)].append(d)
    out, records = [], {}
    for ident, ds in sorted(groups.items(), key=lambda kv: tuple(map(str, kv[0]))):
        split, chunk, conc, thr, k, tag = ident[:6]
        labels = [d.get("shard", "0/1") for d in ds]
        dupes = sorted({x for x in labels if labels.count(x) > 1})
        if dupes:
            raise MergeError(f"duplicate shard labels {dupes} in identity {ident}: "
                             f"{[d['run_id'] for d in ds]}; re-run under a new tag instead")
        expected = {f"{i}/{k}" for i in range(k)}
        missing = sorted(expected - set(labels))
        extra = sorted(set(labels) - expected)
        if extra:
            raise MergeError(f"shard labels {extra} are outside 0..{k - 1} in identity {ident}")
        base = dict(schema="ultra.nemotron-merged/v2", split=split, chunk_ms=chunk, concurrency=conc,
                    threads=thr, tag=tag, runtime_commit=ident[6], harness_commit=ident[7],
                    shards=len(ds), shards_expected=k, missing_shards=missing, complete=not missing,
                    evidence="exploratory", run_ids=sorted(d["run_id"] for d in ds))
        if missing:
            out.append(base)
            continue
        if check_rows:
            seen: set[str] = set()
            recs = []
            for d in ds:
                rows = _rows(d)
                if len(rows) != d["utterances"]:
                    raise MergeError(f"{d['run_id']}: {len(rows)} rows for {d['utterances']} utterances")
                for r in rows:
                    rec = _score_record(r, split)
                    if rec["id"] in seen:
                        raise MergeError(f"utterance {rec['id']} appears in two shards of {ident}")
                    seen.add(rec["id"])
                    recs.append(rec)
            records[ident] = sorted(recs, key=lambda r: r["id"])
        S, D, Ii, N = (sum(d["wer"][x] for d in ds) for x in ("sub", "del", "ins", "ref_words"))
        audio = sum(d["audio_seconds"] for d in ds)
        cpu = sum(d["process_cpu_seconds"] for d in ds)
        chunk_lat = [d["bench_run"].get("metrics", {}).get("chunk_latency_ms", {}) for d in ds]
        base.update(
            partition=ds[0]["partition"], honesty_label=ds[0]["honesty_label"],
            utterances=sum(d["utterances"] for d in ds), audio_seconds=round(audio, 2),
            wer={"wer": round(100.0 * (S + D + Ii) / max(N, 1), 3), "sub": S, "del": D, "ins": Ii, "ref_words": N},
            last_word_missing_utts=sum(d["truncation_probe"]["utts_last_ref_word_missing_from_hyp_tail"] for d in ds),
            process_cpu_seconds=round(cpu, 2), cpu_s_per_audio_s=round(cpu / audio, 5),
            streams_per_core_at_full_load=round(audio / cpu, 3),
            wall_rtf_per_process=round(sum(d["process_wall_seconds"] for d in ds) / audio, 5),
            peak_rss_mib_max=max(d["peak_rss_mib"] for d in ds),
            chunk_compute_ms_mean=round(sum(c.get("mean", 0) for c in chunk_lat) / len(ds), 2),
            chunk_compute_ms_p99_max=round(max(c.get("p99", 0) for c in chunk_lat), 2),
            affinity=sorted(c for d in ds for c in d["affinity"]),
            gpu_pids_seen=sorted({p for d in ds for p in d["gpu_pids_seen"]}),
            gpu_check_status=sorted({d.get("gpu_check_status", "not recorded") for d in ds}),
            loadavg_1m_range=[min((d.get("loadavg_before") or [None])[0] or 0 for d in ds),
                              max((d.get("loadavg_after") or [None])[0] or 0 for d in ds)]
            if all("loadavg_before" in d for d in ds) else "not recorded")
        out.append(base)
    return out, records


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="+")
    ap.add_argument("--out")
    ap.add_argument("--score-records", help="directory for per-group ScoreRecord JSONL")
    a = ap.parse_args(argv)
    lines = [json.loads(x) for f in a.results for x in open(f) if x.strip()]
    merged, records = merge(lines)
    text = "".join(json.dumps(m) + "\n" for m in merged)
    sys.stdout.write(text)
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(text)
    if a.score_records:
        os.makedirs(a.score_records, exist_ok=True)
        for ident, recs in records.items():
            name = f"{ident[0]}_{ident[1]}ms_c{ident[2]}_t{ident[3]}_k{ident[4]}{'_' + ident[5] if ident[5] else ''}.score.jsonl"
            with open(os.path.join(a.score_records, name), "w") as fh:
                fh.writelines(json.dumps(r) + "\n" for r in recs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
