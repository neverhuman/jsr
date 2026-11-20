"""Pool sharded nemotron_run.py results (same split/chunk/concurrency) into one corpus line.

WER pools S/D/I over shards (equal to scoring the whole set at once); CPU and audio sum;
wall is the slowest shard (shards ran in parallel, one core each); RSS is the per-process max.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict


def merge(lines: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for d in lines:
        groups[(d["split"], d["chunk_ms"], d["concurrency"], d["threads"], d.get("shard", "0/1").split("/")[1])].append(d)
    out = []
    for (split, chunk, conc, thr, k), ds in sorted(groups.items()):
        # keep the newest line per shard
        by_shard = {}
        for d in ds:
            by_shard[d.get("shard", "0/1")] = d
        ds = list(by_shard.values())
        complete = len(ds) == int(k)
        S = sum(d["wer"]["sub"] for d in ds)
        D = sum(d["wer"]["del"] for d in ds)
        Ii = sum(d["wer"]["ins"] for d in ds)
        N = sum(d["wer"]["ref_words"] for d in ds)
        audio = sum(d["audio_seconds"] for d in ds)
        cpu = sum(d["process_cpu_seconds"] for d in ds)
        chunk_lat = [d["bench_run"].get("metrics", {}).get("chunk_latency_ms", {}) for d in ds]
        out.append(dict(
            schema="ultra.nemotron-merged/v1", split=split, chunk_ms=chunk, concurrency=conc, threads=thr,
            shards=len(ds), shards_expected=int(k), complete=complete,
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
            run_ids=[d["run_id"] for d in ds]))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results", nargs="+")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    lines = [json.loads(x) for f in a.results for x in open(f) if x.strip()]
    merged = merge(lines)
    text = "".join(json.dumps(m) + "\n" for m in merged)
    sys.stdout.write(text)
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
