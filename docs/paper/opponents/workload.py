"""Stage hard-linked views of the frozen DEV panel for the opponent runs.

    python workload.py PANEL OUT NAME [--shards N]

PANEL is the verified staged panel (`audio.json`, `wav/u%05d.wav`). NAME selects the view:

    panel730  all 730 utterances (the cost and WER workload)
    panel73   every tenth staged file, one utterance per speaker (slow arms)
    lag100    the first 100 dev-clean utterances (word-lag traces)
    smoke10   five dev-clean and five dev-other utterances (smoke test only)

OUT/NAME/wav holds hard links, OUT/NAME/audio.json the matching manifest rows and
OUT/NAME/workload.json the identity. With --shards N, OUT/NAME/shards-N/shard-K holds N
disjoint parts balanced by audio length for N single-thread workers.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

PANEL_DIGEST = "010aa985e8c677d4daf43561dd9600d1ad27429494e88b74c177825a8cbc3963"
VIEWS = {
    "panel730": lambda count: range(count),
    "panel73": lambda count: range(0, count, 10),
    "lag100": lambda count: range(100),
    "smoke10": lambda count: [0, 1, 2, 3, 4, 400, 401, 402, 403, 404],
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def link_all(panel, rows, target):
    target.mkdir(parents=True)
    for row in rows:
        name = row["id"] + ".wav"
        os.link(panel / "wav" / name, target / name)


def stage(panel, out, name, shards):
    if sha256(panel / "audio.json") != PANEL_DIGEST:
        raise ValueError("staged panel manifest differs from the frozen digest")
    audio = json.loads((panel / "audio.json").read_bytes())
    rows = [audio[index] for index in VIEWS[name](len(audio))]
    root = out / name
    if not (root / "workload.json").is_file():
        link_all(panel, rows, root / "wav")
        with (root / "audio.json").open("x", encoding="utf-8") as output:
            json.dump(rows, output, indent=2)
            output.write("\n")
        identity = {
            "name": name,
            "panel_digest": PANEL_DIGEST,
            "utterances": len(rows),
            "audio_seconds": sum(row["samples"] for row in rows) / 16000,
            "audio_json_sha256": sha256(root / "audio.json"),
        }
        with (root / "workload.json").open("x", encoding="utf-8") as output:
            json.dump(identity, output, indent=2)
            output.write("\n")
    if shards and not (root / f"shards-{shards}" / "shards.json").is_file():
        parts = [[] for _ in range(shards)]
        loads = [0] * shards
        for row in sorted(rows, key=lambda item: (-item["samples"], item["id"])):
            slot = min(range(shards), key=lambda index: (loads[index], index))
            parts[slot].append(row)
            loads[slot] += row["samples"]
        base = root / f"shards-{shards}"
        for index, part in enumerate(parts):
            link_all(panel, sorted(part, key=lambda item: item["id"]), base / f"shard-{index}")
        layout = {
            "shards": shards,
            "audio_seconds": [load / 16000 for load in loads],
            "ids": [sorted(row["id"] for row in part) for part in parts],
        }
        with (base / "shards.json").open("x", encoding="utf-8") as output:
            json.dump(layout, output, indent=2)
            output.write("\n")
    print((root / "workload.json").read_text(), end="")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("panel", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("name", choices=sorted(VIEWS))
    parser.add_argument("--shards", type=int, default=0)
    args = parser.parse_args()
    stage(args.panel, args.out, args.name, args.shards)


if __name__ == "__main__":
    main()
