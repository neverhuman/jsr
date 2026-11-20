"""Fetch LibriSpeech (CC BY 4.0) from the HF mirror openslr/librispeech_asr as parquet shards.

Same audio and transcripts as OpenSLR SLR12, which serves this host at under 1 MB/s.
Records every file's SHA-256 in MANIFEST.sha256 next to the data.
"""
import hashlib
import os
import sys

from huggingface_hub import snapshot_download

ROOT = os.environ.get("JSR_DATA", "/mnt/xbabe3-external/jsr")
DEST = os.path.join(ROOT, "corpora", "librispeech", "hf_parquet")
REVISION = os.environ.get("LS_REVISION", "main")
SPLITS = sys.argv[1:] or ["validation.clean", "validation.other", "test.clean", "test.other",
                          "train.clean.100", "train.clean.360", "train.other.500"]


def main() -> None:
    os.makedirs(DEST, exist_ok=True)
    path = snapshot_download("openslr/librispeech_asr", repo_type="dataset", revision=REVISION,
                             allow_patterns=[f"all/{s}/*" for s in SPLITS], local_dir=DEST,
                             max_workers=8)
    lines = []
    for dirpath, _, files in sorted(os.walk(os.path.join(path, "all"))):
        for f in sorted(files):
            p = os.path.join(dirpath, f)
            h = hashlib.sha256()
            with open(p, "rb") as fh:
                for block in iter(lambda: fh.read(1 << 22), b""):
                    h.update(block)
            lines.append(f"{h.hexdigest()}  {os.path.relpath(p, DEST)}\n")
    with open(os.path.join(DEST, "MANIFEST.sha256"), "w") as fh:
        fh.writelines(lines)
    print("FETCH_DONE", len(lines), "files")


if __name__ == "__main__":
    main()
