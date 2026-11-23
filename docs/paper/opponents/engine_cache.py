"""Stage a workload view as npz caches and panel manifests for `ultra-engine-eval`.

    python engine_cache.py PANEL WORKLOAD_DIR
    python engine_cache.py --rows PANEL RUN_DIR

`ultra-cold-wav` cannot be built on macOS (its crates need Linux affinity, prctl and
/proc), so the Apple row of our engine runs the engine's own tool, which reads the
`libri.py` cache format instead of a WAV directory. The first form writes, from the staged
WAVs and the panel's index, `WORKLOAD_DIR/engine/{dev-clean.npz,dev-other.npz,panel.json}`
and the same under `WORKLOAD_DIR/engine-shards-N/shard-K/` for every staged shard. Audio
is byte-identical to the WAVs; transcripts are left empty so the tool stays reference-blind.

The second form turns a finished run's `engine_eval.json` (keyed by corpus uid) into the
receipt files: `predictions.jsonl` keyed by the opaque staged id, and
`engine_eval.summary.json` (the tool's own summary without the per-utterance rows).
"""

import hashlib
import json
from pathlib import Path
import sys
import wave

SPLITS = ("dev-clean", "dev-other")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stage(index, wav_dir, out):
    import numpy as np

    rows = [index[path.stem] for path in sorted(wav_dir.glob("u*.wav"))]
    out.mkdir(parents=True)
    panel = {"schema": "ultra.dev-panel/v1", "seed": "ultra-dev-panel-v1", "splits": []}
    for split in SPLITS:
        part = [row for row in rows if row["dataset"] == split]
        if not part:
            continue
        pcm, offsets = [], [0]
        for row in part:
            with wave.open(str(wav_dir / f"{row['id']}.wav"), "rb") as source:
                samples = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2")
            pcm.append(samples)
            offsets.append(offsets[-1] + len(samples))
        cache = out / f"{split}.npz"
        np.savez(cache, pcm=np.concatenate(pcm), offsets=np.asarray(offsets, dtype="<i8"),
                 uids=np.asarray([row["uid"] for row in part]),
                 speakers=np.asarray([int(row["speaker"]) for row in part], dtype="<i8"),
                 texts=np.asarray(["" for _ in part], dtype="<U1"))
        panel["splits"].append({
            "split": split, "cache_sha256": sha256(cache), "utterances": len(part),
            "uids": [row["uid"] for row in part],
            "speaker_of": {row["uid"]: int(row["speaker"]) for row in part},
        })
    panel["manifest_sha256"] = hashlib.sha256(json.dumps(panel, indent=1, sort_keys=True).encode()).hexdigest()
    with (out / "panel.json").open("x", encoding="utf-8") as output:
        json.dump(panel, output, indent=1, sort_keys=True)
        output.write("\n")


def rows(index, run):
    by_uid = {row["uid"]: row["id"] for row in index.values()}
    parts = sorted(run.glob("worker-*.out")) or [run / "engine-eval"]
    with (run / "predictions.jsonl").open("x", encoding="utf-8") as output:
        summaries = []
        for part in parts:
            result = json.loads((part / "engine_eval.json").read_bytes())
            for row in result.pop("rows"):
                json.dump({"id": by_uid[row["id"]], "hyp": row["hyp"]}, output, ensure_ascii=False)
                output.write("\n")
            summaries.append(result)
    with (run / "engine_eval.summary.json").open("x", encoding="utf-8") as output:
        json.dump(summaries, output, indent=1, sort_keys=True)
        output.write("\n")
    for part in parts:
        (part / "engine_eval.json").unlink()
        part.rmdir()


def main():
    args = sys.argv[1:]
    convert = args[:1] == ["--rows"]
    if convert:
        args = args[1:]
    if len(args) != 2:
        sys.exit(__doc__)
    panel, target = Path(args[0]), Path(args[1])
    index = {row["id"]: row for row in json.loads((panel / "index.json").read_bytes())}
    if convert:
        rows(index, target)
        return
    if not (target / "engine" / "panel.json").is_file():
        stage(index, target / "wav", target / "engine")
    for shards in sorted(target.glob("shards-*")):
        for shard in sorted(shards.glob("shard-*")):
            out = target / f"engine-{shards.name}" / shard.name
            if not (out / "panel.json").is_file():
                stage(index, shard, out)


if __name__ == "__main__":
    main()
