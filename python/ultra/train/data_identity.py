"""Preflight and compare sealed, label-only trainer inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path

from train.errors import CheckpointError, DataError

LOCAL_PATH_SETTINGS = {"label_map", "label_map_sha256", "extra_splits", "tokenizer", "dur_tables",
                       "node_shards", "noise_dir", "rir_dir"}


def file_sha256(path) -> str:
    with open(path, "rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


def _update(digest, value) -> None:
    digest.update(json.dumps(value, ensure_ascii=False, separators=(",", ":"),
                             sort_keys=True, allow_nan=False).encode("utf-8"))
    digest.update(b"\n")


def _digest(values) -> str:
    digest = hashlib.sha256()
    for value in values:
        _update(digest, value)
    return digest.hexdigest()


def recipe(settings: dict) -> dict:
    """Paths are accounted for by content and sampler identities."""
    return {k: v for k, v in settings.items() if k not in LOCAL_PATH_SETTINGS}


def trainer_identity() -> str:
    train = Path(__file__).parent
    sources = [*train.glob("*.py"), *(train / "optim").glob("*.py")]
    return _digest((str(path.relative_to(train)), file_sha256(path)) for path in sorted(sources))


def seal(source) -> dict:
    """Bind the selected rows and effective sampler to the actual local input bytes."""
    import pyarrow.parquet as pq

    label_map = source.label_map
    if label_map is None or not label_map.only:
        raise DataError("sealed training requires a pinned label map and label_map_only=1")
    shards, seen, retained_extra = [], set(), set()
    selection, durations = hashlib.sha256(), hashlib.sha256()
    for fi, path in enumerate(source.files):
        extra = fi >= source._first_extra
        parquet = pq.ParquetFile(path)
        columns = ["id", "src", "t0", "t1"] if extra else ["id"]
        check_conf = extra and source.min_conf > 0
        if check_conf:
            columns.append("conf")
        missing = set(columns) - set(parquet.schema_arrow.names)
        if missing:
            raise DataError(f"{path}: sealed rows need {sorted(missing)}")
        name = "/".join(Path(path).parts[-2:])
        shards.append({"name": name, "sha256": file_sha256(path), "extra": extra})
        for rg in range(parquet.metadata.num_row_groups):
            rows = parquet.read_row_group(rg, columns=columns).to_pylist()
            sidecar = source._dur_table.row_group(fi, rg) if source._dur_table is not None else None
            if sidecar is not None:
                if [r[0] for r in sidecar] != [r["id"] for r in rows]:
                    raise DataError(f"{path}:{rg}: duration sidecar UID order differs from parquet")
                _update(durations, sidecar)
            for ri, row in enumerate(rows):
                uid = row["id"]
                if not isinstance(uid, str) or not uid or uid in seen:
                    raise DataError(f"{path}:{rg}: invalid or duplicate UID {uid!r}")
                seen.add(uid)
                if not extra:
                    _update(selection, [fi, rg, ri, uid])
                    continue
                if check_conf:
                    conf = row["conf"]
                    if sidecar is not None and sidecar[ri][2] is not None and sidecar[ri][2] != conf:
                        raise DataError(f"{path}:{rg}: duration sidecar confidence differs for {uid}")
                    if conf is None:
                        continue
                    if not math.isfinite(conf):
                        raise DataError(f"{path}:{rg}: non-finite confidence for {uid}")
                    if conf < source.min_conf:
                        continue
                if not label_map.keep(uid):
                    continue
                src, t0, t1 = row["src"], row["t0"], row["t1"]
                if not src or t0 is None or t1 is None or not (0 <= t0 < t1 < float("inf")):
                    raise DataError(f"{path}:{rg}: invalid recording interval for {uid}")
                # Commit both exported bounds and the exact Parquet bytes that contain the cut.
                _update(selection, [fi, rg, ri, uid, src, t0, t1])
                retained_extra.add(uid)
    if not retained_extra:
        raise DataError("sealed training has no eligible mapped extra-split rows")
    if source._node_plan is None and retained_extra != label_map.texts.keys():
        raise DataError("sealed label map includes rows absent from the eligible extra splits")
    train = Path(__file__).parent
    aug = source.fact_aug
    aug_files = {}
    if aug is not None:
        from train.fact_aug import list_train_only_audio

        for name, directory, probability in (("noise", aug.noise_dir, aug.noise_p),
                                               ("rir", aug.rir_dir, aug.rir_p)):
            if probability:
                files = list_train_only_audio(directory, f"{name}_dir")
                aug_files[name] = [[str(Path(path).relative_to(directory)), file_sha256(path)]
                                   for path in files]
    return {
        "version": 2,
        "audio": {
            "shards": shards,
            "ordered_rows_sha256": selection.hexdigest(),
            "map_uid_sha256": _digest(sorted(label_map.texts)),
            "retained_extra_uids": len(retained_extra),
            "duration_rows_sha256": durations.hexdigest() if source._dur_table is not None else None,
            "tokenizers": [source.feat.tok_rnnt.sha256, source.feat.tok_ctc.sha256],
            "source_sha256": {
                name: file_sha256(train / name)
                for name in ("data.py", "fact_aug.py", "features.py", "shard_sidecar.py", "tokenizer.py")
            },
            "augment_files": aug_files,
            "sampler": {
                "seed": source.seed, "speed": list(source.speed_perturb),
                "budget_s": source.budget_s, "max_batch": source.max_batch,
                "max_seconds": source.max_seconds, "window_groups": source.window_groups,
                "pad_frames": source.pad_frames, "pad_tokens": source.pad_tokens,
                "synthetic_augmentation": source.synth_aug,
                "min_conf": source.min_conf, "weights": source._weighted_weights,
                "nodes": asdict(source._node_plan) if source._node_plan is not None else None,
                "global_batches": source.ddp_global_batches, "stratify_lengths": source.stratify_lengths,
                "rank_weights": source.rank_weights, "world": source.world,
            },
        },
        "labels": {"sha256": label_map.sha256, "only": True},
    }


def _arm(directory: Path) -> list[dict]:
    def read(name):
        return json.loads((directory / name).read_text())

    root = read("config.json")
    if not isinstance(root.get("world"), int) or root["world"] < 1:
        raise CheckpointError(f"{directory}: missing rank layout")
    rows = [root] + [read(f"config.rank{i}.json") for i in range(1, root["world"])]
    for i, row in enumerate(rows):
        identity = row.get("data_identity_v2") or {}
        if (row["world"] != len(rows) or row.get("rank") != i or not row.get("fresh_start")
                or identity.get("version") != 2 or not identity.get("labels", {}).get("only")
                or not row.get("initial_checkpoint_sha256")
                or row["initial_checkpoint_sha256"] != root["initial_checkpoint_sha256"]):
            raise CheckpointError(f"{directory}: rank {i} has no fresh sealed preflight")
    return rows


def compare(a: Path, b: Path) -> dict:
    left, right = _arm(a), _arm(b)
    if len(left) != len(right):
        raise CheckpointError("label arms have different world sizes")
    label_hashes, audio_hashes = set(), []
    fields = ("model", "objective_plan", "trainer_source_sha256", "initial_checkpoint_sha256",
              "local_rank", "device_type", "threads")
    for i, (x, y) in enumerate(zip(left, right, strict=True)):
        if any(x[k] != y[k] for k in fields):
            raise CheckpointError(f"rank {i}: model seed, layout or execution settings differ")
        if recipe(x["train"]) != recipe(y["train"]):
            raise CheckpointError(f"rank {i}: effective training recipes differ")
        ix, iy = x["data_identity_v2"], y["data_identity_v2"]
        if ix["audio"] != iy["audio"]:
            raise CheckpointError(f"rank {i}: ordered audio, preprocessing or sampler inputs differ")
        label_hashes.add((ix["labels"]["sha256"], iy["labels"]["sha256"]))
        audio_hashes.append(_digest([ix["audio"]]))
    if len(label_hashes) != 1 or any(x == y for x, y in label_hashes):
        raise CheckpointError("expected two distinct label maps, each fixed across its ranks")
    return {
        "schema": "ultra.label-arm-identity/v1", "world": len(left),
        "initial_checkpoint_sha256": left[0]["initial_checkpoint_sha256"],
        "trainer_source_sha256": left[0]["trainer_source_sha256"],
        "label_map_sha256": list(next(iter(label_hashes))),
        "audio_identity_sha256_by_rank": audio_hashes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm_a", type=Path)
    parser.add_argument("arm_b", type=Path)
    args = parser.parse_args()
    try:
        print(json.dumps(compare(args.arm_a, args.arm_b), indent=2))
    except (CheckpointError, ValueError, OSError, KeyError) as error:
        parser.exit(1, f"label-arm preflight failed: {error}\n")


if __name__ == "__main__":
    main()
