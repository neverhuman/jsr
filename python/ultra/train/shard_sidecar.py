"""H55 S4 data-loader sidecar packing and Option (iv) node-local disjoint shard partitioning.

1. FrameSidecar: loads trainer-lane `dur_table.py` JSONL sidecars
   (`{"file": "...", "size": ..., "mtime_ns": ..., "groups": [[[uid, n_frames], ...], ...]}`)
   so ParquetBatches packs global batches with zero audio-byte reads and zero `sf.info()` calls.
2. NodeShardPlan + partition_items: deterministic seeded disjoint partitioning of extra (Libri-Light)
   shards across nodes (uniform 1/K or GPU-weighted), cutting per-node disk 3-6x while preserving
   global Libri-Light audio share, utterance length distribution, and lockstep DDP resume cursors.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import io
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Callable, Sequence, TypeVar

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

from train.errors import DataError

T = TypeVar("T")


def _rel_key(path: str) -> tuple[str, str]:
    p = os.path.normpath(path)
    return os.path.basename(os.path.dirname(p)), os.path.basename(p)


def _parse_group_rows(raw_group: list) -> list[tuple[str, int, float | None]]:
    out: list[tuple[str, int, float | None]] = []
    for entry in raw_group:
        if not isinstance(entry, (list, tuple)) or len(entry) < 2:
            raise DataError(f"malformed duration sidecar row entry: {entry!r}")
        conf = None if len(entry) < 3 or entry[2] is None else float(entry[2])
        out.append((str(entry[0]), int(entry[1]), conf))
    return out


class FrameSidecar:
    """In-memory per-file, per-row-group `(uid, n_frames, conf|None)` index from JSONL sidecars."""

    def __init__(self, by_fi: list[list[list[tuple[str, int, float | None]]]], sources: tuple[str, ...]):
        self._by_fi, self.sources = by_fi, sources

    @classmethod
    def from_paths(cls, spec: str, files: Sequence[str], num_row_groups: Sequence[int]) -> FrameSidecar:
        paths = tuple(p.strip() for p in spec.split(",") if p.strip())
        if not paths:
            raise DataError("dur_tables spec is empty")
        exact, by_real, by_rel = {}, {}, {}
        for tp in paths:
            if not os.path.isfile(tp):
                raise DataError(f"duration sidecar missing: {tp}")
            with open(tp, "r", encoding="utf-8") as fh:
                for lineno, line in enumerate(fh, 1):
                    s = line.strip()
                    if not s:
                        continue
                    try:
                        rec = json.loads(s)
                        f_key, groups = str(rec["file"]), rec["groups"]
                    except (json.JSONDecodeError, KeyError, TypeError) as exc:
                        raise DataError(f"invalid duration sidecar {tp}:{lineno}: {exc}") from exc
                    if not isinstance(groups, list):
                        raise DataError(f"invalid groups list in {tp}:{lineno}")
                    # Later line for the same file wins (matches trainer-lane fast_audit.py).
                    exact[f_key] = by_real[os.path.realpath(f_key)] = by_rel[_rel_key(f_key)] = groups

        resolved, missing = [], []
        for f, expected_rgs in zip(files, num_row_groups, strict=True):
            raw = exact.get(f) or by_real.get(os.path.realpath(f)) or by_rel.get(_rel_key(f))
            if raw is None:
                missing.append(f)
                continue
            if len(raw) != expected_rgs:
                raise DataError(f"duration sidecar row-group count mismatch for {f}: "
                                f"sidecar={len(raw)} parquet={expected_rgs}")
            resolved.append([_parse_group_rows(g) for g in raw])
        if missing:
            raise DataError(f"duration sidecar lacks {len(missing)} file(s), e.g. {missing[0]}")
        return cls(resolved, paths)

    def row_group(self, fi: int, rg: int) -> list[tuple[str, int, float | None]]:
        return self._by_fi[fi][rg]

    @staticmethod
    def frames_after(
        uid: str, n_samples: int, epoch: int, *, seed: int, speed_perturb: tuple[float, ...],
        max_seconds: float, num_frames_fn: Callable[[int], int],
    ) -> int | None:
        """Exact `ParquetBatches._frames_after` calculation from raw FLAC sample count."""
        n = n_samples
        if speed_perturb:
            h = hashlib.sha256(f"{seed}:{epoch}:{uid}".encode()).digest()
            f = speed_perturb[h[0] % len(speed_perturb)]
            if f != 1.0:
                r = Fraction(f).limit_denominator(100)
                n = -(-n * r.denominator // r.numerator)
        return None if n / 16000.0 > max_seconds else int(num_frames_fn(n))


def _scan_parquet_file(args: tuple[str, bool]) -> dict:
    path, include_conf = args
    pf = pq.ParquetFile(path)
    has_conf = include_conf and ("conf" in pf.schema_arrow.names)
    cols = ["id", "audio", "conf"] if has_conf else ["id", "audio"]
    groups = []
    for rg in range(pf.metadata.num_row_groups):
        t = pf.read_row_group(rg, columns=cols)
        ids, aud = t.column("id").to_pylist(), t.column("audio").to_pylist()
        if has_conf:
            confs = t.column("conf").to_pylist()
            groups.append([[u, int(sf.info(io.BytesIO(a["bytes"])).frames), None if c is None else float(c)]
                           for u, a, c in zip(ids, aud, confs, strict=True)])
        else:
            groups.append([[u, int(sf.info(io.BytesIO(a["bytes"])).frames)]
                           for u, a in zip(ids, aud, strict=True)])
    st = os.stat(path)
    return {"file": os.path.abspath(path), "size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns), "groups": groups}


def _expand_parquet_inputs(inputs: Sequence[str]) -> list[str]:
    files: list[str] = []
    for inp in inputs:
        files.extend(sorted(glob.glob(os.path.join(inp, "*.parquet"))) if os.path.isdir(inp)
                     else ([inp] if os.path.isfile(inp) else sorted(glob.glob(inp))))
    return sorted({os.path.abspath(f) for f in files})


def build_dur_table(
    inputs: Sequence[str], out_path: str, *, workers: int = 4, include_conf: bool = False,
) -> dict:
    """Build or incrementally update a `dur_table.py`-compatible JSONL frame-count sidecar."""
    files = _expand_parquet_inputs(inputs)
    seen: dict[str, tuple[int, int]] = {}
    if os.path.isfile(out_path):
        for line in open(out_path, "r", encoding="utf-8"):
            if not line.strip():
                continue
            rec = json.loads(line)
            seen[str(rec["file"])] = ((int(rec["size"]), int(rec["mtime_ns"]))
                                      if ("size" in rec and "mtime_ns" in rec) else (-1, -1))
    pending = [(f, include_conf) for f in files
               if seen.get(f) not in ((-1, -1), (int(os.stat(f).st_size), int(os.stat(f).st_mtime_ns)))]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    if pending:
        if workers > 1 and len(pending) >= 8:
            with ProcessPoolExecutor(max_workers=min(workers, len(pending)), mp_context=mp.get_context("spawn")) as ex:
                records = list(ex.map(_scan_parquet_file, pending))
        elif workers > 1 and len(pending) > 1:
            with ThreadPoolExecutor(max_workers=min(workers, len(pending))) as ex:
                records = list(ex.map(_scan_parquet_file, pending))
        else:
            records = [_scan_parquet_file(item) for item in pending]
        with open(out_path, "a", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
    return {"table": os.path.abspath(out_path), "files": len(files), "scanned": len(pending)}


@dataclass(frozen=True)
class NodeShardPlan:
    """Option (iv) node-local disjoint shard partitioning configuration."""
    node_rank: int
    node_world: int
    local_rank: int
    local_world: int
    weights: tuple[float, ...]
    prestaged: bool = False
    seed: int = 5504

    @property
    def share(self) -> float:
        return self.weights[self.node_rank] / sum(self.weights)

    def gpu_share(self, global_world: int) -> float:
        return self.local_world / float(global_world) if (global_world > 1 and self.local_world <= global_world) else self.share


def parse_node_shards(spec: str) -> NodeShardPlan | None:
    """Parse Option (iv) spec: e.g. 'node=0/4,local=0/2,weights=2:2:1:1,prestaged=0,seed=5504' or '0/4'."""
    s = (spec or "").strip()
    if not s:
        return None
    kv: dict[str, str] = {}
    if "=" not in s and "/" in s and "," not in s:
        kv["node"] = s
    else:
        for tok in (t.strip() for t in s.split(",") if t.strip()):
            if "=" not in tok:
                raise DataError(f"invalid node_shards token {tok!r} (expected key=value)")
            k, v = tok.split("=", 1)
            kv[k.strip()] = v.strip()
    if "node" not in kv or "/" not in kv["node"]:
        raise DataError("node_shards requires node=<rank>/<world>")
    nr, nw = (int(x) for x in kv["node"].split("/", 1))
    if nw < 1 or not (0 <= nr < nw):
        raise DataError(f"invalid node_shards node={nr}/{nw}")
    if "local" in kv:
        lr, lw = (int(x) for x in kv["local"].split("/", 1))
    else:
        lr, lw = int(os.environ.get("LOCAL_RANK", "0")), int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    if lw < 1 or not (0 <= lr < lw):
        raise DataError(f"invalid node_shards local={lr}/{lw}")
    w = tuple(float(x) for x in kv["weights"].split(":") if x.strip()) if "weights" in kv else (1.0,) * nw
    if len(w) != nw or min(w) <= 0.0:
        raise DataError(f"node_shards weights must have {nw} positive values, got {w}")
    return NodeShardPlan(nr, nw, lr, lw, w, kv.get("prestaged", "0").lower() in ("1", "true", "yes"),
                         int(kv.get("seed", "5504")))


def _apportion(total: int, weights: Sequence[float]) -> list[int]:
    """Largest-remainder (Hamilton) apportionment summing strictly to `total`."""
    wsum = float(sum(weights))
    quotas = [total * float(w) / wsum for w in weights]
    counts = [int(q) for q in quotas]
    for i in sorted(range(len(weights)), key=lambda j: (-(quotas[j] - counts[j]), j))[:total - sum(counts)]:
        counts[i] += 1
    return counts


def partition_items(
    items: Sequence[T], node_world: int, *, weights: Sequence[float] | None = None,
    seed: int = 5504, salt: int = 0, preserve_order: bool = True,
) -> list[list[T]]:
    """Deterministic seeded disjoint partition of `items` into `node_world` buckets."""
    if node_world < 1:
        raise DataError(f"node_world must be >= 1, got {node_world}")
    w = tuple(float(x) for x in weights) if weights is not None else (1.0,) * node_world
    if len(w) != node_world or min(w) <= 0.0:
        raise DataError("partition weights must match node_world and be positive")
    if node_world == 1:
        return [list(items)]
    perm = np.random.default_rng([int(seed), int(salt), 5504]).permutation(len(items))
    out, pos = [], 0
    for c in _apportion(len(items), w):
        idxs = sorted(int(i) for i in perm[pos:pos + c]) if preserve_order else [int(i) for i in perm[pos:pos + c]]
        out.append([items[i] for i in idxs])
        pos += c
    return out


def plan_node_manifests(
    extra_dirs: Sequence[str], node_world: int, *, weights: Sequence[float] | None = None, seed: int = 5504,
) -> dict:
    """Plan offline per-node disjoint shard manifests for staging Option (iv) on local NVMe."""
    w = tuple(float(x) for x in weights) if weights is not None else (1.0,) * node_world
    per_node: list[dict[str, list[str]]] = [{} for _ in range(node_world)]
    totals: dict[str, int] = {}
    for si, d in enumerate(extra_dirs):
        fs = sorted(glob.glob(os.path.join(d, "*.parquet")))
        if not fs:
            raise DataError(f"no parquet shards in {d}")
        totals[d] = len(fs)
        parts = partition_items(fs, node_world, weights=w, seed=seed, salt=100 + si, preserve_order=True)
        for nr in range(node_world):
            per_node[nr][d] = [os.path.basename(f) for f in parts[nr]]
    return {"node_world": node_world, "weights": list(w), "seed": seed, "totals": totals, "nodes": per_node}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build frame-count sidecars or plan Option (iv) node shard manifests.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="Build/update dur_table.py-compatible JSONL sidecar")
    b.add_argument("--out", required=True); b.add_argument("--workers", type=int, default=4)
    b.add_argument("--include-conf", action="store_true"); b.add_argument("inputs", nargs="+")
    p = sub.add_parser("partition", help="Emit disjoint per-node shard manifest JSON")
    p.add_argument("--node-world", type=int, required=True); p.add_argument("--weights", default="")
    p.add_argument("--seed", type=int, default=5504); p.add_argument("--out", default="")
    p.add_argument("extra_dirs", nargs="+")
    a = ap.parse_args(argv)
    if a.cmd == "build":
        res = build_dur_table(a.inputs, a.out, workers=a.workers, include_conf=a.include_conf)
    else:
        w = tuple(float(x) for x in a.weights.split(":") if x.strip()) if a.weights else None
        res = plan_node_manifests(a.extra_dirs, a.node_world, weights=w, seed=a.seed)
        if a.out:
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            Path(a.out).write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
