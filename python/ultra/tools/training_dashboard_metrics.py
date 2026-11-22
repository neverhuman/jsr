"""Normalize training probe records and calculate dashboard summaries."""

from collections import deque
from math import isfinite
from statistics import median

TRAIN_FIELDS = ("step", "time", "loss", "rnnt", "ctc", "audio_s", "wall_s", "epoch", "lr", "grad_norm")
EVAL_FIELDS = ("step", "time", "wer", "ref_words", "sub", "del", "ins")
GPU_FIELDS = ("index", "util_pct", "memory_used_mib", "memory_total_mib", "power_w", "temperature_c")


def number(value):
    if type(value) in (float, int) and 0 <= value < 1e16 and isfinite(value):
        return value
    return None


def label(value, default=""):
    return value[:100] if isinstance(value, str) and value else default


def fields(row, keys):
    return {key: number(row.get(key)) for key in keys}


def records(value, limit):
    if isinstance(value, list):
        return [row for row in value[-limit:] if isinstance(row, dict)]
    return []


def ddp_view(value):
    if not isinstance(value, dict):
        return None
    group = label(value.get("group"))
    rank, world_size = value.get("rank"), value.get("world_size")
    if not group or type(rank) is not int or type(world_size) is not int or not 0 <= rank < world_size or world_size < 2:
        return None
    return {"group": group, "rank": rank, "world_size": world_size, "started_at": number(value.get("started_at"))}


def run_view(run, index):
    train = []
    for row in records(run.get("train"), 240):
        point = fields(row, TRAIN_FIELDS)
        if point["step"] is None:
            continue
        audio, wall = point["audio_s"], point["wall_s"]
        if train and point["time"] is not None:
            previous = train[-1]
            if previous["time"] is not None and point["step"] > previous["step"]:
                elapsed = point["time"] - previous["time"]
                if elapsed > 0:
                    wall = max(wall or 0, elapsed)
        point["rate_wall_s"] = wall
        point["audio_per_second"] = audio / wall if audio is not None and wall else None
        if number(point["audio_per_second"]) is None:
            point["audio_per_second"] = None
        train.append(point)
    evaluations = []
    for row in records(run.get("eval"), 100):
        point = fields(row, EVAL_FIELDS)
        point["split"] = label(row.get("split"), "unknown")
        point["decoder"] = label(row.get("decoder"), "unknown")
        if point["step"] is not None and point["wer"] is not None:
            evaluations.append(point)
    ddp = ddp_view(run.get("ddp"))
    follower = ddp is not None and ddp["rank"] > 0
    return {
        "id": label(run.get("id"), str(index)),
        "name": label(run.get("name"), "Training run"),
        "active": run.get("active") is True,
        "overfit": run.get("overfit") is True,
        "eval_scope": label(run.get("eval_scope"), "unknown"),
        **fields(run, ("pid", "gpu_index", "total_steps", "dev_limit")),
        "ddp": ddp,
        "train": [] if follower else train,
        "eval": [] if follower else evaluations,
    }


def probe_view(payload):
    if not isinstance(payload, dict) or number(payload.get("updated_at")) is None:
        raise ValueError("probe timestamp missing")
    if not isinstance(payload.get("runs"), list):
        raise ValueError("probe runs missing")
    gpus = records(payload.get("gpus"), 16)
    if not gpus and isinstance(payload.get("gpu"), dict):
        gpus = [payload["gpu"]]
    gpus = [dict(fields(gpu, GPU_FIELDS), name=label(gpu.get("name"), "GPU")) for gpu in gpus]
    runs = [run_view(run, index) for index, run in enumerate(records(payload["runs"], 32))]
    return {
        "updated_at": payload["updated_at"],
        "detail": label(payload.get("detail")),
        "runs": list({run["id"]: run for run in runs}.values()),
        "gpu": gpus[0] if gpus else None,
        "gpus": gpus,
        **fields(payload, ("disk_free_bytes", "cpu_load", "cpu_count", "cpu_pct")),
    }


def latest_block(run, node_clock):
    if not run["active"] or not run["train"]:
        return None
    last = run["train"][-1]
    if last["time"] is None:
        return None
    ddp = run.get("ddp")
    if ddp and ddp["started_at"] is not None and last["time"] < ddp["started_at"]:
        return None
    freshness_seconds = max(180, min(600, 3 * (last["wall_s"] or 0)))
    if -30 <= node_clock - last["time"] <= freshness_seconds:
        return last
    return None


def host_view(host, reading, received_at, detail, now):
    sample = reading or {"runs": [], "gpu": None, "gpus": [], "disk_free_bytes": None}
    active = [run for run in sample["runs"] if run["active"]]
    live_link = received_at is not None and now - received_at <= 15 and not detail
    node_clock = sample.get("updated_at", now) + (now - received_at if received_at is not None else 0)
    latest = [latest_block(run, node_clock) if live_link else None for run in sample["runs"]]
    rates = [block["audio_per_second"] for block in latest if block and block["audio_per_second"] is not None]
    has_group_rate = any(block and run["ddp"] and run["ddp"]["rank"] == 0
                         for run, block in zip(sample["runs"], latest))
    if live_link:
        state = "live" if any(latest) else "stale" if active else "idle"
        message = sample["detail"] or ("Waiting for a fresh training block" if state == "stale" else "")
    else:
        state = "connecting" if received_at is None and not detail else "offline"
        message = detail or ("Opening collector" if state == "connecting" else "Collector heartbeat overdue")
    return {
        **sample,
        "host": host,
        "state": state,
        "updated_at": received_at,
        "detail": message,
        "audio_per_second": sum(rates) if rates else None,
        "has_group_rate": has_group_rate,
        "runs": [dict(run, fresh=block is not None) for run, block in zip(sample["runs"], latest)],
    }


def attach_shared_runs(hosts):
    leaders = {}
    connected = [host for host in hosts.values() if host["state"] in ("live", "stale")]
    for host in connected:
        for run in host["runs"]:
            ddp = run["ddp"]
            if ddp and ddp["rank"] == 0 and run["fresh"]:
                group = ddp["group"], ddp["world_size"]
                leaders[group] = None if group in leaders else {"host": host["host"], "id": run["id"], "name": run["name"]}
    for host in connected:
        for run in host["runs"]:
            ddp = run["ddp"]
            if run["active"] and ddp and ddp["rank"] > 0:
                shared = leaders.get((ddp["group"], ddp["world_size"]))
                run["shared_from"] = shared
                if shared:
                    if host["state"] == "stale":
                        host["state"], host["detail"] = "live", ""
                elif host["state"] == "stale":
                    host["detail"] = "Waiting for progress from rank 0"


def fleet_summary(hosts, history: deque, now):
    rates = [host["audio_per_second"] for host in hosts if host["audio_per_second"] is not None]
    fresh = [run["train"][-1] for host in hosts for run in host["runs"] if run["fresh"]]
    losses = [block["loss"] for block in fresh if block["loss"] is not None]
    active_runs = sum(
        run["active"] and (run["ddp"] is None or run["ddp"]["rank"] == 0)
        for host in hosts if host["state"] in ("live", "stale") for run in host["runs"]
    )
    best = {}
    for host in hosts:
        for run in host["runs"]:
            if run["overfit"] or run["eval_scope"] == "overfit":
                continue
            for row in run["eval"]:
                if row["split"] not in ("dev-clean", "dev-other") or not row["ref_words"]:
                    continue
                limit = run["dev_limit"]
                scope = f"First {int(limit):,} utterances" if limit else run["eval_scope"]
                group = (row["split"], row["decoder"], scope, row["ref_words"])
                if group not in best or row["wer"] < best[group]["wer"]:
                    best[group] = dict(row, host=host["host"], run=run["name"], scope=scope)
    point = {
        "audio_per_second": sum(rates) if rates else None,
        "median_loss": median(losses) if losses else None,
        "active_runs": active_runs,
    }
    if not history or now - history[-1]["time"] >= 5 or any(
        value != history[-1][key] for key, value in point.items()
    ):
        history.append(dict(point, time=now))
    return {
        **point,
        "nodes_live": sum(host["state"] == "live" for host in hosts),
        "nodes_total": len(hosts),
        "wer_by_split": [best[key] for key in sorted(best)],
        "history": list(history),
    }
