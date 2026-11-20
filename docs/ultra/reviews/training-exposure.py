"""Reconstruct TRAIN duration-filter eligibility without decoding the audio corpus.

Usage: .venv/bin/python docs/ultra/reviews/training-exposure.py RUN_CONFIG NEW_OUTPUT.json
Reads only the declared TRAIN shards; verifies their DATA.lock manifest hashes.
Eligibility is not proof of examples actually consumed by an interrupted run.
"""

import hashlib
import io
import json
import os
import sys
from pathlib import Path

import pyarrow.parquet as pq
import soundfile as sf


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    config_path, output = map(Path, sys.argv[1:])
    require(not output.exists(), "output already exists")
    config = json.loads(config_path.read_text())["train"]
    split_dirs = {"train-clean-100": "train.clean.100", "train-clean-360": "train.clean.360",
                  "train-other-500": "train.other.500"}
    require(all(split in split_dirs for split in config["splits"]), "TRAIN-only audit")
    data = Path(os.environ.get("JSR_DATA", "/mnt/xbabe3-external/jsr"))
    root = data / "corpora/librispeech/hf_parquet"
    manifest = root / "MANIFEST.sha256"
    require(digest(manifest) == "3e21c0e4dc9597cd0379da34ccec5f16cb89e34c7785868daa1f14bd4910bff5",
            "unrecognized DATA.lock manifest")
    expected = dict(line.split(maxsplit=1)[::-1] for line in manifest.read_text().splitlines())
    seen, reports = set(), []
    for split in config["splits"]:
        files = sorted((root / "all" / split_dirs[split]).glob("*.parquet"))
        require(files, f"missing shards: {split}")
        prefix = "all/" + split_dirs[split] + "/"
        require({str(path.relative_to(root)) for path in files} ==
                {name for name in expected if name.startswith(prefix)},
                f"incomplete or extra TRAIN shard set: {split}")
        entries, files_proof, decoded_checks = [], {}, 0
        for path in files:
            relative = str(path.relative_to(root))
            require(relative in expected and digest(path) == expected[relative],
                    f"changed shard: {relative}")
            files_proof[relative] = expected[relative]
            shard = pq.ParquetFile(path)
            for group in range(shard.num_row_groups):
                batch = shard.read_row_group(group, columns=["id", "audio.bytes"], use_threads=False)
                for uid, audio in zip(batch["id"], batch["audio"]):
                    uid = uid.as_py()
                    require(uid not in seen, f"duplicate TRAIN id: {uid}")
                    seen.add(uid)
                    encoded = audio["bytes"].as_py()
                    info = sf.info(io.BytesIO(encoded))
                    require(info.samplerate == 16000 and info.channels == 1 and info.frames > 0,
                            f"unsupported audio: {uid}")
                    eligible = info.frames / 16000 <= config["max_seconds"]
                    if decoded_checks < 20 or not eligible:
                        pcm, rate = sf.read(io.BytesIO(encoded), dtype="int16")
                        require(rate == 16000 and len(pcm) == info.frames, f"header/decode mismatch: {uid}")
                        decoded_checks += 1
                    entries.append(dict(id=uid, samples=info.frames, eligible=eligible))
            print(f"verified {relative}", file=sys.stderr, flush=True)
        eligible = [row for row in entries if row["eligible"]]
        excluded = [row for row in entries if not row["eligible"]]
        reports.append(dict(split=split, unique_utterances=len(entries),
                            unique_eligible=len(eligible), unique_excluded=len(excluded),
                            total_hours=sum(row["samples"] for row in entries) / 16000 / 3600,
                            eligible_hours=sum(row["samples"] for row in eligible) / 16000 / 3600,
                            excluded_hours=sum(row["samples"] for row in excluded) / 16000 / 3600,
                            decoded_header_checks=decoded_checks, files=files_proof, utterances=entries))
    report = dict(schema="ultra.training-eligibility/v1", config_sha256=digest(config_path),
                  script_sha256=digest(Path(__file__)), manifest_sha256=digest(manifest),
                  max_seconds=config["max_seconds"], rows=reports,
                  scope="Reconstructed duration-filter eligibility; not actual consumed exposure or launch provenance")
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
