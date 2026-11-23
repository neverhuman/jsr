"""Normalize training probe records and calculate dashboard summaries."""

from collections import deque
from math import isfinite
from statistics import median

from tools.dashboard_events import (
    DEFAULT_MEASUREMENT_DOMAIN,
    LEGACY_MEASUREMENT_DOMAIN,
    MonotonicEventStore,
    SCHEMA_VERSION,
    aggregate_run_groups,
    annotate_host_hardware,
    extract_artifact,
    extract_cohort,
    extract_sdi,
    infer_measurement_domain,
    is_valid_schema,
    normalize_gate,
    normalize_stale_state,
)

TRAIN_FIELDS = ("step", "time", "loss", "rnnt", "ctc", "audio_s", "wall_s", "epoch", "lr", "grad_norm")
EVAL_FIELDS = ("step", "time", "wer", "ema_wer", "ref_words", "sub", "del", "ins",
               "train_loss", "train_rnnt", "train_ctc")
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


def _normalize_train(rows, limit, store: MonotonicEventStore, stream_key: str, default_run_id: str):
    train = []
    cur = store.cursor(stream_key)
    for row in records(rows, limit):
        point = fields(row, TRAIN_FIELDS)
        if point["step"] is None:
            continue
        prev_att = cur.attempt_id
        accepted, _, ev = store.ingest(
            {**row, "kind": row.get("kind") or "train"},
            default_run_id=default_run_id, stream_key=stream_key,
        )
        if not accepted:
            continue
        if prev_att is not None and cur.attempt_id != prev_att:
            train.clear()
        if ev["rank"] is not None and ev["rank"] > 0:
            continue
        audio, wall = point["audio_s"], point["wall_s"]
        if train and point["time"] is not None:
            previous = train[-1]
            if previous["time"] is not None and point["step"] > previous["step"]:
                elapsed = point["time"] - previous["time"]
                if elapsed > 0:
                    wall = max(wall or 0, elapsed)
        point["rate_wall_s"] = wall
        fallback_rate = number(row.get("planned_exposure_ahh"))
        if fallback_rate is None:
            fallback_rate = number(row.get("planned_audio_h_per_h"))
        if fallback_rate is None:
            fallback_rate = number(row.get("audio_h_per_h"))
        if fallback_rate is None:
            fallback_rate = number(row.get("audio_per_second"))
        if audio is not None:
            point["audio_per_second"] = audio / wall if wall else None
        else:
            point["audio_per_second"] = fallback_rate
        if number(point["audio_per_second"]) is None:
            point["audio_per_second"] = None
        if ev["attempt_id"] is not None:
            point["attempt_id"] = ev["attempt_id"]
        if ev["seq"] is not None:
            point["seq"] = ev["seq"]
        if ev["rank"] is not None:
            point["rank"] = ev["rank"]
        train.append(point)
    return train


def _nearest_train_loss(train_rows, step):
    best = None
    for row in train_rows:
        if row["step"] <= step and row["loss"] is not None:
            best = row["loss"]
        elif row["step"] > step:
            break
    return best


def overfit_summary(train_global, evaluations, is_overfit_run):
    if is_overfit_run or not evaluations:
        return None
    primary = None
    for split in ("dev-clean", "dev-other"):
        for decoder in ("greedy", "beam-4", "beam", "ctc"):
            matches = [row for row in evaluations if row["split"] == split and row["decoder"] == decoder]
            if len(matches) >= 1:
                primary = (split, decoder, sorted(matches, key=lambda r: r["step"]))
                break
        if primary:
            break
    if primary is None:
        return None
    split, decoder, series = primary
    best = min(series, key=lambda row: (row["wer"], row["step"]))
    latest = series[-1]
    delta_pp = round(latest["wer"] - best["wer"], 3)
    best_train = best["train_loss"] if best["train_loss"] is not None else _nearest_train_loss(train_global, best["step"])
    latest_train = latest["train_loss"] if latest["train_loss"] is not None else _nearest_train_loss(train_global, latest["step"])
    train_ratio = round(latest_train / best_train, 4) if best_train and latest_train is not None else None
    worse_streak = 0
    for row in reversed(series):
        if row["step"] > best["step"] and row["wer"] >= best["wer"] + 0.25:
            worse_streak += 1
        else:
            break
    if len(series) < 2:
        status = "insufficient"
    elif delta_pp >= 0.30 and worse_streak >= 2 and (train_ratio is None or train_ratio <= 0.95):
        status = "overfitting"
    elif delta_pp >= 0.25 and (train_ratio is None or train_ratio <= 0.98):
        status = "watch"
    elif len(series) >= 3 and max(r["wer"] for r in series[-3:]) - best["wer"] <= 0.15 and latest["step"] > best["step"]:
        status = "plateau"
    else:
        status = "improving"
    return {
        "status": status, "split": split, "decoder": decoder, "eval_count": len(series),
        "best_step": best["step"], "best_wer": best["wer"], "latest_step": latest["step"],
        "latest_wer": latest["wer"], "latest_ema_wer": latest.get("ema_wer"), "delta_pp": delta_pp,
        "best_train_loss": best_train, "latest_train_loss": latest_train,
        "train_loss_ratio": train_ratio, "worse_streak": worse_streak,
    }


def run_view(run, index):
    if not is_valid_schema(run):
        raise ValueError("unknown dashboard event schema")
    run_id_explicit = label(run.get("run_id"))
    rid = run_id_explicit or label(run.get("id"), str(index))
    ddp = ddp_view(run.get("ddp"))
    raw_rank = run.get("rank")
    explicit_rank = int(raw_rank) if type(raw_rank) is int and raw_rank >= 0 else None
    rank = explicit_rank if explicit_rank is not None else (ddp["rank"] if ddp is not None else 0)
    raw_world = run.get("world_size")
    world_size = int(raw_world) if type(raw_world) is int and raw_world >= 1 else (ddp["world_size"] if ddp is not None else None)
    store = MonotonicEventStore()
    stream_key = f"run:{index}:{rid}"
    train = _normalize_train(run.get("train"), 240, store, stream_key, rid)
    raw_global = run.get("train_global")
    if isinstance(raw_global, list) and raw_global:
        global_store = MonotonicEventStore()
        train_global = _normalize_train(raw_global, 450, global_store, stream_key, rid)
    else:
        train_global = train
    evaluations = []
    cur = store.cursor(stream_key)
    for row in records(run.get("eval"), 400):
        accepted, _, ev = store.ingest({**row, "kind": "eval"}, default_run_id=rid, stream_key=stream_key)
        if not accepted:
            continue
        point = fields(row, EVAL_FIELDS)
        point["split"] = label(row.get("split"), "unknown")
        point["decoder"] = label(row.get("decoder"), "unknown")
        if ev["sdi"] is not None:
            point["sdi"] = ev["sdi"]
            if point["ref_words"] is None and ev["sdi"].get("ref_words") is not None:
                point["ref_words"] = ev["sdi"]["ref_words"]
            if point["sub"] is None:
                point["sub"] = ev["sdi"]["s"]
            if point["del"] is None:
                point["del"] = ev["sdi"]["d"]
            if point["ins"] is None:
                point["ins"] = ev["sdi"]["i"]
        if ev["measurement_domain"] is not None:
            point["measurement_domain"] = ev["measurement_domain"]
        if point["step"] is not None and point["wer"] is not None:
            if point["train_loss"] is None and train_global:
                point["train_loss"] = _nearest_train_loss(train_global, point["step"])
            evaluations.append(point)
    follower = rank > 0
    is_overfit = run.get("overfit") is True
    rank_local = dict(cur.rank_local_ahh)
    if isinstance(run.get("rank_local_ahh"), dict):
        for rk, rv in run["rank_local_ahh"].items():
            if number(rv) is not None:
                try:
                    rank_local[int(rk)] = round(float(rv), 1)
                except ValueError:
                    pass
    if follower:
        f_rate = number(run.get("rank_local_ahh")) if not isinstance(run.get("rank_local_ahh"), dict) else None
        if f_rate is None and train:
            f_rate = train[-1].get("audio_per_second")
        if f_rate is None:
            f_rate = number(run.get("audio_h_per_h")) or number(run.get("planned_exposure_ahh"))
        if f_rate is not None:
            rank_local[rank] = round(float(f_rate), 1)
    run_sdi = extract_sdi(run) or cur.sdi
    run_dom = infer_measurement_domain(run, run_sdi) or cur.measurement_domain
    if run_dom is None and run_sdi is not None:
        run_dom = DEFAULT_MEASUREMENT_DOMAIN if run.get("schema") == SCHEMA_VERSION else LEGACY_MEASUREMENT_DOMAIN
    artifact = extract_artifact(run)
    if cur.artifact:
        artifact = {**(artifact or {}), **cur.artifact}
    cohort = extract_cohort(run)
    if cur.cohort:
        cohort = {**(cohort or {}), **cur.cohort}
    gate = normalize_gate(run.get("wave_gate") or run.get("gate")) or cur.wave_gate
    if gate is None and run.get("active") is True:
        gate = "RUNNING"
    planned = None
    if not follower:
        planned = number(run.get("planned_exposure_ahh"))
        if planned is None:
            planned = number(run.get("planned_audio_h_per_h"))
        if planned is None and cur.planned_exposure_ahh is not None:
            planned = cur.planned_exposure_ahh
        if planned is None and train:
            planned = train[-1].get("audio_per_second")
    latest_step = train[-1]["step"] if (not follower and train) else (train_global[-1]["step"] if (not follower and train_global) else number(run.get("step")))
    attempt_id = cur.attempt_id if cur.attempt_id is not None else run.get("attempt_id")
    seq = cur.seq_by_rank.get(rank) if rank in cur.seq_by_rank else (int(run["seq"]) if type(run.get("seq")) is int else None)
    rejected = cur.rejected_count + (int(run["rejected_events"]) if type(run.get("rejected_events")) is int else 0)
    shared_group = bool(run_id_explicit or ddp is not None or explicit_rank is not None or (world_size and world_size > 1))
    return {
        "schema": SCHEMA_VERSION, "id": label(run.get("id"), str(index)), "run_id": rid,
        "name": label(run.get("name"), "Training run"), "active": run.get("active") is True,
        "overfit": is_overfit, "eval_scope": label(run.get("eval_scope"), "unknown"),
        **fields(run, ("pid", "gpu_index", "total_steps", "dev_limit")),
        "gpu_uuid": label(run.get("gpu_uuid")) or None, "ddp": ddp,
        "rank": rank, "world_size": world_size, "attempt_id": attempt_id, "seq": seq, "step": latest_step,
        "sdi": run_sdi, "measurement_domain": run_dom, "artifact": artifact, "cohort": cohort,
        "planned_exposure_ahh": planned, "rank_local_ahh": rank_local or None, "wave_gate": gate,
        "stale_state": normalize_stale_state("live" if run.get("active") is True else "offline"),
        "rejected_events": rejected, "_shared_group": shared_group,
        "train": [] if follower else train, "train_global": [] if follower else train_global,
        "eval": [] if follower else evaluations,
        "overfit_check": None if follower else overfit_summary(train_global, evaluations, is_overfit),
    }


def probe_view(payload):
    if not isinstance(payload, dict) or number(payload.get("updated_at")) is None:
        raise ValueError("probe timestamp missing")
    if not is_valid_schema(payload):
        raise ValueError("unknown dashboard event schema")
    if not isinstance(payload.get("runs"), list):
        raise ValueError("probe runs missing")
    gpus = records(payload.get("gpus"), 16)
    if not gpus and isinstance(payload.get("gpu"), dict):
        gpus = [payload["gpu"]]
    gpus = [dict(fields(gpu, GPU_FIELDS), name=label(gpu.get("name"), "GPU"), uuid=label(gpu.get("uuid")) or None) for gpu in gpus]
    runs = []
    for index, raw_run in enumerate(records(payload["runs"], 32)):
        if not is_valid_schema(raw_run):
            continue
        runs.append(run_view(raw_run, index))
    timing = payload.get("timing_lock") if isinstance(payload.get("timing_lock"), dict) else None
    other = payload.get("other_load") if isinstance(payload.get("other_load"), dict) else None
    return {
        "schema": SCHEMA_VERSION, "updated_at": payload["updated_at"], "detail": label(payload.get("detail")),
        "runs": list({run["id"]: run for run in runs}.values()),
        "gpu": gpus[0] if gpus else None, "gpus": gpus,
        "timing_lock": timing, "other_load": other,
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
    fresh_entries = []
    for run, block in zip(sample["runs"], latest):
        if block is not None:
            fresh_entries.append({**run, "audio_per_second": block["audio_per_second"], "_host": host})
        elif run["active"] and run.get("rank", 0) > 0 and run.get("rank_local_ahh"):
            fresh_entries.append({**run, "_host": host})
    grouped = aggregate_run_groups(fresh_entries)
    has_group_rate = any(block and run["ddp"] and run["ddp"]["rank"] == 0
                         for run, block in zip(sample["runs"], latest))
    if live_link:
        state = "live" if any(latest) else "stale" if active else "idle"
        message = sample.get("detail", "") or ("Waiting for a fresh training block" if state == "stale" else "")
    else:
        state = "connecting" if received_at is None and not detail else "offline"
        message = detail or ("Opening collector" if state == "connecting" else "Collector heartbeat overdue")
    stale_state = normalize_stale_state(state)
    hw = annotate_host_hardware(
        sample.get("gpus") or [], cpu_load=sample.get("cpu_load"), cpu_count=sample.get("cpu_count"),
        cpu_pct=sample.get("cpu_pct"), other_load=sample.get("other_load"), timing_lock=sample.get("timing_lock"),
    )
    out_runs = []
    for run, block in zip(sample["runs"], latest):
        r_stale = normalize_stale_state("live" if block is not None else state)
        out_runs.append({**run, "fresh": block is not None, "stale_state": r_stale})
    return {
        **sample, "schema": SCHEMA_VERSION, "host": host, "state": state, "stale_state": stale_state,
        "updated_at": received_at, "detail": message,
        "gpus": hw["gpus"], "gpu": hw["gpus"][0] if hw["gpus"] else None,
        "dual_gpu_imbalance": hw["dual_gpu_imbalance"], "cpu_load_pct": hw["cpu_load_pct"], "host_band": hw["host_band"],
        "audio_per_second": grouped["audio_per_second"],
        "planned_exposure_ahh": grouped["planned_exposure_ahh"] if grouped["audio_per_second"] is not None else None,
        "rank_local_ahh": grouped["rank_local_ahh"] or None,
        "has_group_rate": has_group_rate, "runs": out_runs,
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
                        host["state"], host["stale_state"], host["detail"] = "live", "live", ""
                elif host["state"] == "stale":
                    host["detail"] = "Waiting for progress from rank 0"


def fleet_summary(hosts, history: deque, now, *, scorecard=None, diloco=None):
    fresh_entries, active_entries, fresh_blocks = [], [], []
    for host in hosts:
        connected = host["state"] in ("live", "stale")
        for run in host["runs"]:
            if connected and run["active"]:
                active_entries.append({**run, "_host": host["host"]})
            if run.get("fresh") and run.get("train"):
                block = run["train"][-1]
                fresh_blocks.append(block)
                fresh_entries.append({**run, "audio_per_second": block["audio_per_second"], "_host": host["host"]})
    grouped_rates = aggregate_run_groups(fresh_entries)
    grouped_active = aggregate_run_groups(active_entries)
    losses = [block["loss"] for block in fresh_blocks if block["loss"] is not None]
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
        "audio_per_second": grouped_rates["audio_per_second"],
        "median_loss": median(losses) if losses else None,
        "active_runs": grouped_active["live_runs"],
    }
    if not history or now - history[-1]["time"] >= 5 or any(
        value != history[-1][key] for key, value in point.items()
    ):
        history.append(dict(point, time=now))
    timing_locks = {h["host"]: h["timing_lock"] for h in hosts if h.get("timing_lock")}
    other_hosts = {h["host"]: h["other_load"] for h in hosts if h.get("other_load")}
    imbalanced = sorted(h["host"] for h in hosts if h.get("dual_gpu_imbalance"))
    return {
        **point, "schema": SCHEMA_VERSION,
        "planned_exposure_ahh": grouped_rates["planned_exposure_ahh"] if grouped_rates["audio_per_second"] is not None else 0.0,
        "nodes_live": sum(host["state"] == "live" for host in hosts), "nodes_total": len(hosts),
        "wer_by_split": [best[key] for key in sorted(best)], "history": list(history),
        "v3_scorecard": scorecard, "diloco": list(diloco or []),
        "timing_locks": timing_locks, "other_project_hosts": other_hosts, "imbalanced_hosts": imbalanced,
    }
