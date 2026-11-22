"""Suggest ddp_rank_weights from a DDP run's log: each rank's frames per second of its own forward.

  python3 rank_weights.py <train.jsonl> [last_n_blocks=20]
Prints one relative speed per rank, the fastest = 1 (pass it as ddp_rank_weights=... at a
checkpoint restart; size budget_s so the fastest rank's share still fits its GPU memory).
"""
import json
import sys


def suggest(path: str, last: int = 20) -> list[float]:
    rows = []
    with open(path) as fh:
        for line in fh:
            if '"rank_fwd_s"' in line:
                rows.append(json.loads(line))
    rows = rows[-last:]
    if not rows:
        raise SystemExit("no rank_fwd_s records (needs a DDP run on this trainer version)")
    world = len(rows[-1]["rank_frames"])
    rows = [r for r in rows if len(r["rank_frames"]) == world]
    speed = [sum(r["rank_frames"][k] for r in rows) / max(1e-9, sum(r["rank_fwd_s"][k] for r in rows))
             for k in range(world)]
    top = max(speed)
    return [round(v / top, 2) for v in speed]


if __name__ == "__main__":
    w = suggest(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 20)
    print("ddp_rank_weights=" + ",".join(str(v) for v in w))
