"""Emit bounded, reference-free trainer and GPU snapshots as JSON lines."""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import time


POLL_SECONDS = 1.0
HEARTBEAT_SECONDS = 4.0
DISCOVERY_SECONDS = 5.0
HARDWARE_SECONDS = 10.0
RECENT_BYTES = 2 * 1024 * 1024
MAX_GLOBAL_TRAIN = 400
MAX_RECENT_TRAIN = 160
MAX_EVALS = 400
MAX_LINE_BYTES = 128 * 1024
PYTHON_NAME = re.compile(r"python(?:\d+(?:\.\d+)*)?[dm]?\Z", re.IGNORECASE)
DEV_SPLITS = {"dev-clean", "dev-other"}
SCHEMA_VERSION = "ultra.dashboard.event/v1"
DEFAULT_MEASUREMENT_DOMAIN = "ultra-en-wer-norm/v1"
GATE_STATES = frozenset({"PASS", "FAIL", "INCONCLUSIVE", "RUNNING"})

try:
    from tools.dashboard_events import (
        attempt_rank as _attempt_rank,
        extract_artifact as _extract_artifact,
        extract_cohort as _extract_cohort,
        extract_sdi as _extract_sdi,
        infer_measurement_domain as _infer_measurement_domain,
        is_valid_schema as _is_valid_schema,
        normalize_gate as _normalize_gate,
        read_timing_status as _read_timing_status,
    )
    from tools.fleet_dashboard.parse import parse_other_load as _parse_other_load
except ImportError:
    def _is_valid_schema(raw):
        schema = raw.get("schema") if raw.get("schema") is not None else raw.get("schema_version")
        return schema is None or schema == SCHEMA_VERSION

    def _attempt_rank(val):
        if val is None:
            return (0, 0)
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            return (1, float(val))
        text = str(val).strip()
        if text.isdigit():
            return (1, float(text))
        match = re.search(r"(\d+)$", text)
        return (2, f"{text[:match.start()]}{int(match.group(1)):012d}") if match else (2, text)

    def _normalize_gate(val):
        if not isinstance(val, str):
            return None
        up = val.strip().upper()
        return up if up in GATE_STATES else None

    def _extract_sdi(rec):
        for src in (rec.get("sdi"), rec, rec.get("legacy_sdi")):
            if not isinstance(src, dict):
                continue
            s = src.get("s", src.get("S", src.get("sub", src.get("subs", src.get("substitutions")))))
            d = src.get("d", src.get("D", src.get("del", src.get("dels", src.get("deletions")))))
            ins = src.get("i", src.get("I", src.get("ins", src.get("insertions"))))
            if s is None or d is None or ins is None:
                continue
            out = {"s": int(s), "d": int(d), "i": int(ins)}
            ref = src.get("ref_words", rec.get("ref_words", src.get("words", src.get("ref_word_count", src.get("n_words")))))
            if ref is not None:
                out["ref_words"] = int(ref)
            return out
        return None

    def _infer_measurement_domain(raw, sdi):
        dom = raw.get("measurement_domain") or raw.get("normaliser_id")
        if isinstance(dom, str) and dom.strip():
            return dom.strip()
        if sdi is not None:
            is_legacy = (
                isinstance(raw.get("legacy_sdi"), dict)
                or any(k in raw for k in ("sub", "del", "subs", "dels", "substitutions", "deletions"))
                or (raw.get("schema") != SCHEMA_VERSION and raw.get("schema_version") != SCHEMA_VERSION)
            )
            return "legacy" if is_legacy else DEFAULT_MEASUREMENT_DOMAIN
        return None

    def _extract_artifact(rec):
        nested = rec.get("artifact") if isinstance(rec.get("artifact"), dict) else {}
        ckpt = nested.get("ckpt_sha256") or rec.get("ckpt_sha256") or rec.get("checkpoint_sha256") or rec.get("artifact_sha256")
        commit = nested.get("git_commit") or rec.get("git_commit") or rec.get("commit")
        tree = nested.get("git_tree") or rec.get("git_tree") or rec.get("tree_sha256")
        art = {}
        if ckpt:
            art["ckpt_sha256"] = str(ckpt).strip()
        if commit:
            art["git_commit"] = str(commit).strip()
        if tree:
            art["git_tree"] = str(tree).strip()
        return art or None

    def _extract_cohort(rec):
        raw = rec.get("cohort")
        out = {}
        if isinstance(raw, dict):
            for key in ("name", "diloco_round", "roster", "mode"):
                if raw.get(key) is not None:
                    out[key] = raw[key]
        elif isinstance(raw, str) and raw.strip():
            out["name"] = raw.strip()
        if rec.get("diloco_round") is not None:
            out["diloco_round"] = int(rec["diloco_round"])
        roster = rec.get("diloco_roster") or rec.get("roster")
        if isinstance(roster, list) and roster:
            out["roster"] = [str(item) for item in roster]
        mode = rec.get("diloco_mode") or rec.get("mode")
        if isinstance(mode, str) and mode.strip():
            out["mode"] = mode.strip()
        return out or None

    def _read_timing_status(opponents_dir=None):
        base = Path(opponents_dir) if opponents_dir is not None else (Path.home() / "jsr-data" / "opponents")
        if not base.is_dir():
            return None
        lock_dir, quiet_file = base / "TIMING.lock", base / "TIMING.quiet"
        status, owner, quiet = "free", "", None
        if lock_dir.is_dir():
            status = "held"
            owner_file = lock_dir / "owner"
            try:
                owner = owner_file.read_text(encoding="utf-8", errors="replace").splitlines()[0].strip() if owner_file.is_file() else "locked"
            except (OSError, IndexError):
                owner = "locked"
        elif lock_dir.is_file():
            status, owner = "held", "locked"
        if quiet_file.is_file():
            try:
                lines = quiet_file.read_text(encoding="utf-8", errors="replace").splitlines()
                quiet = lines[0].strip() if lines else None
            except OSError:
                quiet = None
        return {"status": status, "owner": owner, "quiet": quiet}

    def _parse_other_load(text, gpus, live_runs, cpu_pct=None):
        gpu_mem_mb, cpu_other_pct, labels = 0.0, 0.0, []
        for line in text.splitlines():
            if line.startswith(("OTHER_GPU ", "OTHER_CPU ")):
                parts = line.split(None, 2)
                try:
                    val = float(parts[1])
                    if line.startswith("OTHER_GPU "):
                        gpu_mem_mb += val
                    else:
                        cpu_other_pct += val
                except (IndexError, ValueError):
                    pass
                if len(parts) > 2 and parts[2].strip():
                    labels.append(parts[2].strip()[:32])
        gpu_busy, cpu_busy = gpu_mem_mb >= 256.0, cpu_other_pct >= 15.0
        if not live_runs:
            for g in gpus:
                used = float(g.get("mem_used") or g.get("memory_used_mib") or 0.0)
                total = float(g.get("mem_total") or g.get("memory_total_mib") or 0.0)
                util = float(g.get("util") or g.get("util_pct") or 0.0)
                if util >= 15.0 or (total > 0 and 100.0 * used / total >= 10.0):
                    gpu_busy, gpu_mem_mb = True, max(gpu_mem_mb, used)
            if cpu_pct is not None and cpu_pct >= 25.0:
                cpu_busy, cpu_other_pct = True, max(cpu_other_pct, float(cpu_pct))
        if not gpu_busy and not cpu_busy:
            return {}
        uniq = sorted(set(labels))
        tag = f"other project ({', '.join(uniq[:2])})" if uniq else "other project"
        return {"gpu": gpu_busy, "cpu": cpu_busy, "gpu_mem_mb": round(gpu_mem_mb, 1),
                "cpu_pct": round(cpu_other_pct, 1), "label": tag}


def number(value, minimum=0):
    if type(value) not in (int, float) or not minimum <= value <= sys.float_info.max:
        return None
    return value


def count(value):
    return value if type(value) is int and 0 <= value <= 2**53 - 1 else None


def command(args, timeout=1.5):
    try:
        result = subprocess.run(args, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL, encoding="utf-8", errors="replace",
                                timeout=timeout, check=False)
        return result.stdout if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def mac_argv(pid):
    # KERN_PROCARGS2 returns the executable and individual argv entries.
    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    mib = (ctypes.c_int * 3)(1, 49, pid)
    size = ctypes.c_size_t()
    if libc.sysctl(mib, 3, None, ctypes.byref(size), None, 0) or not 4 <= size.value <= 1024**2:
        return []
    buffer = ctypes.create_string_buffer(size.value)
    if libc.sysctl(mib, 3, buffer, ctypes.byref(size), None, 0):
        return []
    raw = buffer.raw[:size.value]
    argc = struct.unpack_from("=i", raw)[0]
    end = raw.find(b"\0", 4)
    if argc <= 0 or end < 0:
        return []
    return [os.fsdecode(item) for item in raw[end + 1:].lstrip(b"\0").split(b"\0", argc)[:argc]]


def trainer_args(argv):
    if not argv or not PYTHON_NAME.fullmatch(os.path.basename(argv[0])):
        return []
    tokens = iter(enumerate(argv[1:], 1))
    for idx, token in tokens:
        if token == "-m":
            return argv[idx + 2:] if argv[idx + 1:idx + 2] == ["train.trainer"] else []
        if token == "-mtrain.trainer":
            return argv[idx + 1:]
        if token in ("-X", "-W", "--check-hash-based-pycs"):
            next(tokens, None)
        if token == "--" or token.startswith(("-c", "-m")) or not token.startswith("-"):
            return []
    return []


def option(args, flag):
    for idx, token in enumerate(args):
        if not isinstance(token, str):
            continue
        if token == flag and idx + 1 < len(args):
            return args[idx + 1]
        if token.startswith(flag + "="):
            return token[len(flag) + 1:]
    return None


def overfit_size(args):
    raw = option(args, "--overfit")
    try:
        return 0 if raw is None else count(int(raw))
    except (ValueError, TypeError):
        return None


def linux_process(pid):
    folder = Path("/proc") / str(pid)
    try:
        status = (folder / "stat").read_text().rsplit(")", 1)[1].split()
        if status[0] == "Z":
            return [], ""
        args = (folder / "cmdline").read_bytes().rstrip(b"\0").split(b"\0")
        return [os.fsdecode(arg) for arg in args], status[19]
    except (OSError, IndexError):
        return [], ""


def distributed_process(pid, start_ticks):
    if not sys.platform.startswith("linux"):
        return None
    keys = {b"RANK", b"WORLD_SIZE", b"MASTER_ADDR", b"MASTER_PORT", b"TORCHELASTIC_RUN_ID"}
    try:
        entries = (Path("/proc") / str(pid) / "environ").read_bytes().split(b"\0")
        env = {key: value for entry in entries for key, _, value in [entry.partition(b"=")] if key in keys}
        rank, world_size = int(env[b"RANK"]), int(env[b"WORLD_SIZE"])
        if not 0 <= rank < world_size or world_size <= 1:
            return None
        master, port = env[b"MASTER_ADDR"], env[b"MASTER_PORT"]
        identity = b"\0".join((master, port, env[b"WORLD_SIZE"], env.get(b"TORCHELASTIC_RUN_ID", b"")))
        uptime = float(Path("/proc/uptime").read_text().split()[0])
        started_at = time.time() - uptime + int(start_ticks) / os.sysconf("SC_CLK_TCK")
    except (OSError, KeyError, IndexError, ValueError):
        return None
    return {"group": hashlib.sha256(identity).hexdigest()[:24], "rank": rank,
            "world_size": world_size, "started_at": started_at}


def run_path(pid, text):
    path = Path(text)
    if not path.is_absolute():
        try:
            if sys.platform == "darwin":
                fields = (command(["lsof", "-a", "-p", str(pid), "-d", "cwd", "-Fn"]) or "").splitlines()
                cwd = next(field[1:] for field in fields if field.startswith("n"))
            else:
                cwd = os.readlink(f"/proc/{pid}/cwd")
        except (OSError, StopIteration):
            return None
        path = Path(cwd) / path
    try:
        return path.resolve()
    except (OSError, RuntimeError):
        return None


def trainer_processes():
    listing = command(["ps", "-ww", "-axo", "pid=,ppid=,lstart=,command="])
    if listing is None:
        return None
    parents, candidates = {}, {}
    for line in listing.splitlines():
        parts = line.split(maxsplit=7)
        if len(parts) < 7 or not parts[0].isdigit() or not parts[1].isdigit():
            continue
        pid, ppid = int(parts[0]), int(parts[1])
        parents[pid] = ppid
        if len(parts) != 8 or "train.trainer" not in parts[-1]:
            continue
        if sys.platform == "darwin":
            argv, start = mac_argv(pid), " ".join(parts[2:7])
        else:
            argv, start = linux_process(pid)
        argv = trainer_args(argv)
        run_dir = option(argv, "--run-dir")
        if run_dir and (path := run_path(pid, run_dir)) is not None:
            candidates[pid] = (path, start, overfit_size(argv))
    result = defaultdict(dict)
    for pid in candidates:
        path, start, overfit = candidates[pid]
        parent = parents.get(pid, 0)
        while parent > 1 and parent != pid:
            if parent in candidates and candidates[parent][0] == path:
                break
            parent = parents.get(parent, 0)
        else:
            ddp = distributed_process(pid, start)
            rank = (ddp["group"], ddp["rank"]) if ddp else None
            result[path].setdefault(rank, (pid, start, overfit, ddp))
    return dict(result)


def gpu_snapshot():
    tool = shutil.which("nvidia-smi")
    if tool is None:
        return [], {}
    options = ["--format=csv,noheader,nounits"]
    text = command([tool, "--query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total,power.draw",
                    *options], timeout=0.8)
    gpus, indexes = [], {}
    for row in csv.reader((text or "").splitlines(), skipinitialspace=True):
        if len(row) != 7 or not row[0].isdigit():
            continue

        def reading(value):
            try:
                return number(float(value))
            except ValueError:
                return None

        index, uuid, name, util, used, total, power = row
        indexes[uuid] = (int(index), uuid)
        gpus.append({"index": int(index), "uuid": uuid, "name": name[:120], "util_pct": reading(util),
                     "memory_used_mib": reading(used), "memory_total_mib": reading(total),
                     "power_w": reading(power)})
    if not gpus:
        return [], {}
    mapping = defaultdict(set)
    apps = command([tool, "--query-compute-apps=pid,gpu_uuid", *options], timeout=0.8)
    for row in csv.reader((apps or "").splitlines(), skipinitialspace=True):
        if len(row) == 2 and row[0].isdigit():
            mapping[int(row[0])].add(indexes.get(row[1]))
    return sorted(gpus, key=lambda gpu: gpu["index"]), {
        pid: next(iter(devices)) for pid, devices in mapping.items()
        if len(devices) == 1 and None not in devices
    }


@dataclass
class History:
    """Read full train.jsonl from step 0 with monotonic (run_id, attempt_id, rank, seq) filtering."""

    train: dict = field(default_factory=dict)
    train_global: list = field(default_factory=list)
    train_global_stride: int = 1
    train_global_seen: int = 0
    last_train: dict | None = None
    evals: dict = field(default_factory=dict)
    identity: tuple | None = None
    mtime: int = 0
    offset: int = 0
    pending: bytes = b""
    anchor: bytes = b""
    skipping: bool = False
    config_identity: tuple | None = None
    total_steps: int | None = None
    dev_limit: int | None = None
    overfit: int | None = None
    run_id: str | None = None
    attempt_id: str | int | None = None
    world_size: int | None = None
    seq_by_rank: dict = field(default_factory=dict)
    max_train_step_by_rank: dict = field(default_factory=dict)
    rank_local_ahh: dict = field(default_factory=dict)
    artifact: dict | None = None
    cohort: dict | None = None
    sdi: dict | None = None
    measurement_domain: str | None = None
    wave_gate: str | None = None
    rejected_events: int = 0

    def _reset_series(self):
        self.train.clear()
        self.train_global.clear()
        self.train_global_stride = 1
        self.train_global_seen = 0
        self.last_train = None
        self.evals.clear()
        self.seq_by_rank.clear()
        self.max_train_step_by_rank.clear()
        self.rank_local_ahh.clear()

    def _absorb_meta(self, event):
        if event.get("run_id") is not None:
            self.run_id = str(event["run_id"])
        if event.get("world_size") is not None:
            ws = count(event.get("world_size"))
            if ws is not None:
                self.world_size = ws
        art = _extract_artifact(event)
        if art:
            self.artifact = {**(self.artifact or {}), **art}
        coh = _extract_cohort(event)
        if coh:
            self.cohort = {**(self.cohort or {}), **coh}
        dom = event.get("measurement_domain") or event.get("normaliser_id")
        if isinstance(dom, str) and dom.strip():
            self.measurement_domain = dom.strip()
        gate = _normalize_gate(event.get("wave_gate") or event.get("gate"))
        if gate:
            self.wave_gate = gate

    def _record_global_train(self, row):
        if not self.train_global:
            self.train_global.append(row)
            self.train_global_seen = 1
            return
        if row["step"] == self.train_global[-1]["step"]:
            self.train_global[-1] = row
            return
        if row["step"] < self.train_global[-1]["step"]:
            return
        self.train_global_seen += 1
        if self.train_global_seen % self.train_global_stride == 0:
            self.train_global.append(row)
            if len(self.train_global) > MAX_GLOBAL_TRAIN:
                first, last = self.train_global[0], self.train_global[-1]
                self.train_global = [first, *self.train_global[2:-1:2], last]
                self.train_global_stride *= 2

    def global_train_rows(self):
        rows = list(self.train_global)
        if self.last_train is not None and (not rows or self.last_train["step"] > rows[-1]["step"]):
            rows.append(self.last_train)
        return rows

    def _nearest_train(self, step):
        if self.last_train is not None and self.last_train["step"] <= step:
            return self.last_train
        best = None
        for row in self.train_global:
            if row["step"] <= step:
                best = row
            else:
                break
        return best

    def remember(self, rows, key, row, limit):
        previous = rows.get(key)
        if previous == row:
            return False
        if previous is not None and (previous["time"] or 0) > (row["time"] or 0):
            return False
        rows.pop(key, None)
        rows[key] = row
        if len(rows) > limit:
            rows.pop(next(iter(rows)))
        return True

    def record(self, line):
        try:
            event = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            return False
        if not isinstance(event, dict):
            return False
        if not _is_valid_schema(event):
            self.rejected_events += 1
            return False
        kind, step, stamp = event.get("kind"), count(event.get("step")), number(event.get("time"))
        att = event.get("attempt_id")
        advancing_attempt = False
        if att is not None and self.attempt_id is not None:
            cur_k, new_k = _attempt_rank(self.attempt_id), _attempt_rank(att)
            if new_k < cur_k:
                self.rejected_events += 1
                return False
            advancing_attempt = new_k > cur_k
        raw_rank = event.get("rank")
        rank = int(raw_rank) if raw_rank is not None and type(raw_rank) is int and raw_rank >= 0 else 0
        seq = count(event.get("seq"))
        if not advancing_attempt:
            if seq is not None:
                prev_seq = self.seq_by_rank.get(rank)
                if prev_seq is not None and seq <= prev_seq:
                    self.rejected_events += 1
                    return False
            elif kind == "train" and step is not None:
                prev_step = self.max_train_step_by_rank.get(rank)
                if prev_step is not None and step < prev_step:
                    self.rejected_events += 1
                    return False
        if advancing_attempt:
            self._reset_series()
            self.attempt_id = att
            self.wave_gate = None
        elif att is not None and self.attempt_id is None:
            self.attempt_id = att
        if seq is not None:
            self.seq_by_rank[rank] = seq
        if kind == "start":
            self._reset_series()
            if seq is not None:
                self.seq_by_rank[rank] = seq
            self._absorb_meta(event)
            return True
        if kind in ("diverged", "done", "finished"):
            self._absorb_meta(event)
            if kind == "diverged" and self.wave_gate is None:
                self.wave_gate = "FAIL"
            return True
        if step is None:
            return False
        if kind == "resume":
            self.train = {key: row for key, row in self.train.items() if row["step"] <= step}
            self.train_global = [row for row in self.train_global if row["step"] <= step]
            self.last_train = self.train_global[-1] if self.train_global else None
            self.evals = {key: row for key, row in self.evals.items() if row["step"] <= step}
            self.max_train_step_by_rank = {r: min(s, step) for r, s in self.max_train_step_by_rank.items()}
            self._absorb_meta(event)
            return True
        self._absorb_meta(event)
        if kind == "train":
            self.max_train_step_by_rank[rank] = max(step, self.max_train_step_by_rank.get(rank, step))
            audio, wall = number(event.get("audio_s")), number(event.get("wall_s"))
            planned = number(event.get("planned_exposure_ahh"))
            if planned is None:
                planned = number(event.get("planned_audio_h_per_h"))
            rate = number(audio / wall) if audio is not None and wall else (planned if planned is not None else number(event.get("audio_h_per_h")))
            rank_local = number(event.get("rank_local_ahh"))
            if rank_local is None:
                rank_local = number(event.get("rank_audio_h_per_h"))
            if rank_local is not None:
                self.rank_local_ahh[rank] = round(rank_local, 1)
            elif raw_rank is not None and rank > 0 and rate is not None:
                self.rank_local_ahh[rank] = round(rate, 1)
            if raw_rank is not None and rank > 0:
                return True
            position = event.get("pos")
            epoch = position.get("epoch") if isinstance(position, dict) else event.get("epoch")
            row = {"step": step, "time": stamp, "loss": number(event.get("loss")),
                   "rnnt": number(event.get("rnnt")), "ctc": number(event.get("ctc")),
                   "audio_s": audio, "wall_s": wall, "audio_per_second": rate,
                   "epoch": count(epoch), "lr": number(event.get("lr")),
                   "grad_norm": number(event.get("grad_norm"))}
            if planned is not None:
                row["planned_exposure_ahh"] = planned
            if att is not None:
                row["attempt_id"] = att
            if raw_rank is not None:
                row["rank"] = rank
            if seq is not None:
                row["seq"] = seq
            if self.last_train is None or step >= self.last_train["step"]:
                self.last_train = row
            self._record_global_train(row)
            return self.remember(self.train, step, row, MAX_RECENT_TRAIN)
        split = event.get("split")
        if kind != "eval" or not isinstance(split, str) or split not in DEV_SPLITS:
            return False
        decoder = event.get("decoder")
        if isinstance(decoder, str) and (beam_label := re.fullmatch(r"beam[- ]?(\d{1,3})", decoder)):
            decoder = "beam-" + beam_label[1]
        elif decoder not in ("greedy", "ctc", "beam"):
            decoder = "greedy"
        beam = count(event.get("beam"))
        scored = [(decoder, event)]
        if isinstance(event.get("beam_wer"), dict):
            scored.append((f"beam-{beam}" if beam else "beam", event["beam_wer"]))
        paired_train = self._nearest_train(step)
        changed = False
        for decoder, values in scored:
            wer = number(values.get("wer"))
            if wer is not None:
                sdi = _extract_sdi(values) or _extract_sdi(event)
                dom = _infer_measurement_domain(event, sdi)
                if sdi is not None:
                    self.sdi = sdi
                if dom is not None:
                    self.measurement_domain = dom
                row = {"step": step, "time": stamp, "split": split, "wer": wer,
                       "ema_wer": number(values.get("ema_wer")), "decoder": decoder}
                row.update({key: count(values.get(key)) for key in ("ref_words", "sub", "del", "ins")})
                if sdi is not None:
                    row["sdi"] = sdi
                    if row["ref_words"] is None and sdi.get("ref_words") is not None:
                        row["ref_words"] = sdi["ref_words"]
                    if row["sub"] is None:
                        row["sub"] = sdi["s"]
                    if row["del"] is None:
                        row["del"] = sdi["d"]
                    if row["ins"] is None:
                        row["ins"] = sdi["i"]
                if dom is not None:
                    row["measurement_domain"] = dom
                if att is not None:
                    row["attempt_id"] = att
                if seq is not None:
                    row["seq"] = seq
                if paired_train is not None:
                    row["train_loss"] = paired_train.get("loss")
                    row["train_rnnt"] = paired_train.get("rnnt")
                    row["train_ctc"] = paired_train.get("ctc")
                changed |= self.remember(self.evals, (split, step, decoder), row, MAX_EVALS)
        return changed

    def read_log(self, path):
        changed = False
        try:
            with path.open("rb") as source:
                stat = os.fstat(source.fileno())
                identity = (stat.st_dev, stat.st_ino)
                reset = self.identity != identity or stat.st_size < self.offset
                if not reset and stat.st_size == self.offset and stat.st_mtime_ns == self.mtime:
                    return False
                if not reset and self.anchor:
                    source.seek(self.offset - len(self.anchor))
                    reset = source.read(len(self.anchor)) != self.anchor
                if reset:
                    self._reset_series()
                    self.offset = 0
                    self.pending = b""
                    self.skipping = False
                source.seek(self.offset)
                while True:
                    data = source.read(RECENT_BYTES)
                    if not data:
                        break
                    self.offset = source.tell()
                    lines = (self.pending + data).split(b"\n")
                    self.pending = lines.pop()
                    if self.skipping and lines:
                        lines.pop(0)
                        self.skipping = False
                    if self.skipping or len(self.pending) > MAX_LINE_BYTES:
                        self.skipping = True
                        self.pending = b""
                    for line in lines:
                        if len(line) <= MAX_LINE_BYTES:
                            changed |= self.record(line)
                source.seek(max(0, self.offset - 64))
                self.anchor = source.read(min(self.offset, 64))
                self.identity, self.mtime = identity, stat.st_mtime_ns
        except OSError:
            return False
        return changed

    def read_config(self, path):
        try:
            stat = path.stat()
            identity = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
            if self.config_identity == identity or stat.st_size > MAX_LINE_BYTES:
                return
            value = json.loads(path.read_bytes())
            if isinstance(value, dict):
                train = value.get("train", value)
                self.total_steps = count(train.get("total_steps")) if isinstance(train, dict) else None
                self.dev_limit = count(train.get("dev_limit")) if isinstance(train, dict) else None
                argv = value.get("argv")
                self.overfit = overfit_size(argv) if isinstance(argv, list) else None
            self.config_identity = identity
        except (OSError, ValueError):
            pass


def recent_dirs():
    roots = [Path.home() / "jsr-data", Path("/mnt/xbabe3-external/jsr"),
             Path("/mnt/amphora-backup/jsr-data")]
    if os.environ.get("JSR_DATA"):
        roots.insert(0, Path(os.environ["JSR_DATA"]))
    recent = {}
    for root in roots:
        for log in (root / "runs/train").glob("*/train.jsonl"):
            try:
                recent[log.parent.resolve()] = log.stat().st_mtime
            except (OSError, RuntimeError):
                continue
    return sorted(recent, key=recent.get, reverse=True)[:3]


def disk_free(path_list):
    for path in path_list:
        if path is None:
            continue
        try:
            return shutil.disk_usage(path).free
        except OSError:
            continue
    return None


def other_load_lines(live_runs, gpus, cpu_pct, gpu_pids=None, trainer_pids=None):
    lines = []
    known = set(trainer_pids or ())
    extra_gpu_pids = [pid for pid in (gpu_pids or ()) if pid not in known]
    if extra_gpu_pids:
        for pid in extra_gpu_pids[:8]:
            args_txt = command(["ps", "-o", "args=", "-p", str(pid)], timeout=0.5) or ""
            if not re.search(r"train[.](trainer|mwer|evaluate)|ultra-|jsr", args_txt):
                comm = (command(["ps", "-o", "comm=", "-p", str(pid)], timeout=0.5) or "other").strip()
                lines.append(f"OTHER_GPU 512 {comm}")
    ps_txt = command(["ps", "-eo", "pcpu=,comm=,args="], timeout=0.8) or ""
    cpu_hits = 0
    for line in ps_txt.splitlines():
        parts = line.split(None, 2)
        if len(parts) < 2:
            continue
        try:
            pct = float(parts[0])
        except ValueError:
            continue
        if pct >= 15.0 and not re.search(r"train[.](trainer|mwer|evaluate)|ultra-|jsr", line):
            lines.append(f"OTHER_CPU {pct} {parts[1]}")
            cpu_hits += 1
            if cpu_hits >= 4:
                break
    return _parse_other_load("\n".join(lines), gpus, live_runs, cpu_pct)


def run_snapshot(path, history, process, gpu_by_pid):
    pid, start, overfit, ddp = process
    identity = os.fsencode(path)
    if ddp:
        identity += f"\0{ddp['group']}\0{ddp['rank']}".encode()
    if overfit:
        scope = "overfit"
    elif history.dev_limit:
        scope = "dev-prefix"
    else:
        scope = None
    gpu_index, gpu_uuid = gpu_by_pid.get((pid, start), (None, None))
    follower = ddp is not None and ddp["rank"] > 0
    run_rec = {
        "schema": SCHEMA_VERSION, "id": hashlib.sha256(identity).hexdigest()[:20],
        "name": path.name, "active": pid is not None, "pid": pid,
        "gpu_index": gpu_index, "gpu_uuid": gpu_uuid, "total_steps": history.total_steps,
        "dev_limit": history.dev_limit, "eval_scope": scope,
        "overfit": bool(overfit) if overfit is not None else None, "ddp": ddp,
        "rank": ddp["rank"] if ddp else 0, "world_size": ddp["world_size"] if ddp else history.world_size,
        "attempt_id": history.attempt_id, "seq": history.seq_by_rank.get(ddp["rank"] if ddp else 0),
        "sdi": history.sdi, "measurement_domain": history.measurement_domain,
        "artifact": history.artifact, "cohort": history.cohort, "wave_gate": history.wave_gate,
        "rank_local_ahh": dict(history.rank_local_ahh) if history.rank_local_ahh else None,
        "rejected_events": history.rejected_events,
        "train": [] if follower else sorted(history.train.values(), key=lambda row: row["step"]),
        "train_global": [] if follower else history.global_train_rows(),
        "eval": [] if follower else sorted(
            history.evals.values(), key=lambda row: (row["step"], row["split"], row["decoder"])
        ),
    }
    if history.run_id is not None:
        run_rec["run_id"] = history.run_id
    return run_rec


def hardware_snapshot(paths):
    gpus, gpu_by_pid = gpu_snapshot()
    try:
        with open("/proc/stat") as counters:
            ticks = [int(value) for value in counters.readline().split()[1:9]]
        cpu_ticks = (sum(ticks), ticks[3] + ticks[4])
    except (OSError, ValueError, IndexError):
        cpu_ticks = None
    try:
        cpu_load = os.getloadavg()[0]
    except OSError:
        cpu_load = None
    timing_lock = _read_timing_status()
    return gpus, gpu_by_pid, disk_free(paths), cpu_ticks, cpu_load, timing_lock


def live(args):
    pinned = dict.fromkeys(item.expanduser().resolve() for item in args.run_dir)
    histories = {}
    known_dirs = {}
    inferred_dirs = []
    last_processes = {}
    last_info = None
    emitted_at = float("-inf")
    fallback_scan = float("-inf")
    discovery_at = hardware_at = float("-inf")
    gpus, gpu_by_pid, free = [], {}, None
    previous_ticks = cpu_pct = cpu_load = timing_lock = None
    other_load = {}
    detail = None
    with ThreadPoolExecutor(max_workers=2) as worker:
        discovery_job = hardware_job = None
        hardware_processes = {}
        while True:
            begin = time.monotonic()
            if discovery_job is None and begin - discovery_at >= DISCOVERY_SECONDS:
                discovery_job, discovery_at = worker.submit(trainer_processes), begin
            processes = last_processes
            if discovery_job is not None and (last_info is None or discovery_job.done()):
                result = discovery_job.result()
                discovery_job = None
                if result is None:
                    detail = "Process inventory unavailable"
                else:
                    detail, processes = None, result
                    for path in processes:
                        known_dirs[path] = begin
            recent_trainer_dirs = sorted(known_dirs, key=known_dirs.get, reverse=True)[:6]
            known_dirs = {path: known_dirs[path] for path in recent_trainer_dirs}
            if not processes and not pinned and begin - fallback_scan >= 10:
                inferred_dirs, fallback_scan = recent_dirs(), begin
            dirs = dict.fromkeys([*processes, *pinned, *recent_trainer_dirs, *inferred_dirs])
            histories = {path: histories.get(path) or History() for path in dirs}
            changed = processes != last_processes
            for path, history in histories.items():
                changed |= history.read_log(path / "train.jsonl")
                history.read_config(path / "config.json")
            if hardware_job is None and begin - hardware_at >= HARDWARE_SECONDS:
                hardware_processes = processes
                hardware_job = worker.submit(hardware_snapshot, [args.disk_path, *dirs, Path.cwd()])
                hardware_at = begin
            if hardware_job is not None and (args.once or hardware_job.done()):
                gpus, mapping, free, ticks, cpu_load, timing_lock = hardware_job.result()
                if ticks and previous_ticks and ticks[0] > previous_ticks[0]:
                    duration = ticks[0] - previous_ticks[0]
                    cpu_pct = round(100 * (1 - (ticks[1] - previous_ticks[1]) / duration), 1)
                previous_ticks = ticks
                trainer_pids = {
                    pid for ranks in hardware_processes.values() for pid, _, _, _ in ranks.values()
                }
                gpu_by_pid = {
                    (pid, start): mapping[pid]
                    for ranks in hardware_processes.values()
                    for pid, start, _, _ in ranks.values() if pid in mapping
                }
                other_load = other_load_lines({p.name for p in hardware_processes}, gpus, cpu_pct,
                                              gpu_pids=mapping.keys(), trainer_pids=trainer_pids)
                hardware_job = None
            runs = []
            for path, history in histories.items():
                ranks = processes[path].values() if path in processes else [(None, "", history.overfit, None)]
                for process in ranks:
                    runs.append(run_snapshot(path, history, process, gpu_by_pid))
            info = {"schema": SCHEMA_VERSION, "gpu": gpus[0] if gpus else None, "gpus": gpus, "disk_free_bytes": free,
                    "cpu_pct": cpu_pct, "cpu_load": cpu_load, "cpu_count": os.cpu_count(),
                    "timing_lock": timing_lock, "other_load": other_load or None, "runs": runs}
            if detail or not dirs:
                info["detail"] = detail or "No trainer process or recent run log"
            now = time.monotonic()
            visible_change = last_info is None or any(
                info.get(key) != last_info.get(key) for key in ("runs", "gpus", "detail", "timing_lock", "other_load")
            )
            if args.once or changed or visible_change or now - emitted_at >= HEARTBEAT_SECONDS:
                print(json.dumps({"updated_at": time.time(), **info}, allow_nan=False,
                                 separators=(",", ":")), flush=True)
                emitted_at, last_info = now, info
            if args.once:
                return
            last_processes = processes
            time.sleep(max(0, POLL_SECONDS - (time.monotonic() - begin)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", action="append", default=[], type=Path,
                        help="also monitor this run directory when its trainer is inactive")
    parser.add_argument("--disk-path", type=Path, help="report free bytes on the filesystem containing this path")
    parser.add_argument("--once", action="store_true", help="print one snapshot and exit")
    try:
        live(parser.parse_args())
    except (BrokenPipeError, KeyboardInterrupt):
        pass


if __name__ == "__main__":
    main()
