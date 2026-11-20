"""Harness side of P04: LibriSpeech -> Nemotron process adapter -> provisional WER + receipts.

The harness owns references and scoring; the adapter (nemotron_adapter.py) gets audio only.
Writes one summary JSONL line to --out and per-utterance hypotheses next to it.

Example
  PYTHONPATH=python/ultra .venv/bin/python python/ultra/adapters/nemotron_run.py \
      --split test-clean --chunk-ms 160 --cpus 53 --concurrency 1
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import platform
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from nemotron_adapter import DATA, NemotronAdapter  # noqa: E402

from ultra.libri import load_split  # noqa: E402
from ultra.wer import corpus_wer, edit_counts, normalise  # noqa: E402

RUNS = os.path.join(DATA, "runs", "nemotron")


def _git(*a: str, cwd: str | None = None) -> str:
    try:
        return subprocess.check_output(["git", *a], text=True, cwd=cwd, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def tail_deletions(refs: list[str], hyps: list[str], k: int = 3) -> dict:
    """Truncation probe: how often the last k reference words are missing from the hypothesis end."""
    missing_last = 0
    for r, h in zip(refs, hyps, strict=True):
        rw, hw = normalise(r), normalise(h)
        if rw and (not hw or rw[-1] not in hw[-k:]):
            missing_last += 1
    return {"utts_last_ref_word_missing_from_hyp_tail": missing_last, "utts": len(refs)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test-clean")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--chunk-ms", type=int, default=160)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--cpus", default="", help="comma list for taskset, e.g. 53")
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--trace", action="store_true", help="per-chunk partial trace (latency work)")
    ap.add_argument("--shard", default="0/1", help="i/K: this process takes utterances i, i+K, ...")
    ap.add_argument("--lease", default="", help="flock this file for the whole timed run (bench lease)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default=os.path.join(RUNS, "results.jsonl"))
    args = ap.parse_args(argv)

    shard_i, shard_k = (int(x) for x in args.shard.split("/"))
    utts = load_split(args.split, args.limit)[shard_i::shard_k]
    cpus = [int(c) for c in args.cpus.split(",") if c.strip()]
    adapter = NemotronAdapter(chunk_ms=args.chunk_ms, threads=args.threads, cpus=cpus)
    n = len(utts)
    shard_tag = f"_s{shard_i}of{shard_k}" if shard_k > 1 else ""
    work = os.path.join(RUNS, "work", f"{args.split}_n{n}{shard_tag}_{args.chunk_ms}ms_c{args.concurrency}")
    with contextlib.ExitStack() as stack:
        if args.lease:
            fh = stack.enter_context(open(args.lease, "a"))
            fcntl.flock(fh, fcntl.LOCK_EX)
        load_before = os.getloadavg()
        t0 = time.time()
        hyps, rec = adapter.transcribe([u.pcm for u in utts], work, args.concurrency, trace=args.trace)
        load_after = os.getloadavg()
    refs = [u.text for u in utts]
    audio_s = sum(u.seconds for u in utts)
    w = corpus_wer(refs, hyps)
    run_id = f"{args.split}_{args.chunk_ms}ms_c{args.concurrency}_t{args.threads}_n{n}{shard_tag}{'_' + args.tag if args.tag else ''}"
    per_utt = os.path.join(RUNS, f"{run_id}.utts.jsonl")
    with open(per_utt, "w") as fh:
        for u, h in zip(utts, hyps, strict=True):
            s, d, i = edit_counts(normalise(u.text), normalise(h))
            fh.write(json.dumps({"uid": u.uid, "seconds": round(u.seconds, 3), "ref": u.text, "hyp": h,
                                 "sub": s, "del": d, "ins": i}) + "\n")
    runs = rec.bench.get("runs") or [{}]
    out = dict(
        schema="ultra.nemotron-run/v1", run_id=run_id, started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0)),
        candidate="nemotron-0.6b-q8", split=args.split, partition="PUBLIC-TEST" if args.split.startswith("test") else "DEV",
        honesty_label="unknown overlap", shard=args.shard, utterances=len(utts), audio_seconds=round(audio_s, 2),
        chunk_ms=args.chunk_ms, right_context=adapter.right_context, concurrency=args.concurrency,
        batching=args.concurrency > 1, threads=args.threads, affinity=cpus, wer=w,
        truncation_probe=tail_deletions(refs, hyps),
        loadavg_before=[round(x, 2) for x in load_before], loadavg_after=[round(x, 2) for x in load_after],
        bench_lease=args.lease or None,
        process_cpu_seconds=rec.cpu_seconds, cpu_user_seconds=rec.cpu_user_seconds,
        cpu_system_seconds=rec.cpu_system_seconds, process_wall_seconds=rec.wall_seconds,
        cpu_s_per_audio_s=round(rec.cpu_seconds / audio_s, 5), wall_rtf=round(rec.wall_seconds / audio_s, 5),
        streams_per_core_at_full_load=round(audio_s / rec.cpu_seconds, 2) if rec.cpu_seconds else None,
        peak_rss_mib=round(rec.peak_rss_kib / 1024, 1), max_threads_observed=rec.max_threads_observed,
        gpu_pids_seen=rec.gpu_pids_seen, gpu_check="nvidia-smi compute-apps polled every 5 s",
        bench_load_ms=rec.bench.get("load_ms"), bench_warmup_ms=rec.bench.get("warmup_ms"),
        bench_run=runs[0], argv=rec.argv, host=platform.node(), harness_commit=_git("rev-parse", "--short", "HEAD"),
        runtime_commit=_git("rev-parse", "HEAD", cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(adapter.binary))))),
        per_utterance=per_utt)
    line = json.dumps(out)
    print(line)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "a") as fh:
        fh.write(line + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
