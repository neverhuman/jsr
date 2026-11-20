"""Out-of-domain evaluation-only sets (ruling O15): sealed PUBLIC-TEST data.

Sets come from the Open ASR Leaderboard / ESB test-only parquet on Hugging Face
(`hf-audio/open-asr-leaderboard`, pinned revision below), test splits only:
  voxpopuli  VoxPopuli EN test           CC0 (+ Europarl legal notice)
  ami        AMI IHM test (ESB "eval")   CC BY 4.0
  tedlium    TED-LIUM 3 test             BLOCKED_RIGHTS: withdrawn upstream (see BLOCKED)

Sealing. These sets get exactly one read, after the `bench-freeze-v1` tag exists.
- `load_audio(name)` returns audio + ids only (no reference text). It is allowed at any time,
  for timing/cost runs.
- `load_split(name)` returns reference texts. It raises `SealedReadRefused` unless
  (a) env `ULTRA_SEALED_READ=<look-id>` is set,
  (b) `git tag -l bench-freeze-v1` is non-empty in the repo and the tagged commit is an ancestor
      of HEAD, and
  (c) no earlier read of the same set under a different look-id is logged (one read per set).
  Every permitted read appends one line to `docs/ultra/LOOKS.jsonl`.
This guards against accidental reads; it is not a defence against a determined bypass.

Caches (data drive, never git):
  cache/ood_pcm/<name>.audio.npz         pcm (int16 16 kHz, concatenated), offsets, ids
  cache/ood_pcm/sealed/<name>.texts.npz  ids, texts (SEALED; never opened before the freeze)
  cache/ood_pcm/<name>.manifest.json     source, revision, licence, sha256 of every file, counts

CLI (from python/ultra, under `ionice -c3 nice -n 15 taskset -c 48-63`):
  python -m ultra.ood prepare voxpopuli|ami     decode parquet -> caches (never prints text)
"""
from __future__ import annotations

import datetime as _dt
import getpass
import glob
import hashlib
import io
import json
import os
import subprocess
import sys
from dataclasses import dataclass

import numpy as np

DATA = os.environ.get("JSR_DATA", "/mnt/xbabe3-external/jsr")
PARQUET_ROOT = os.path.join(DATA, "corpora", "ood", "open-asr-leaderboard")
CACHE = os.path.join(DATA, "cache", "ood_pcm")
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
FREEZE_TAG = "bench-freeze-v1"
ENV_LOOK = "ULTRA_SEALED_READ"
SR = 16000

HF_REPO = "hf-audio/open-asr-leaderboard"
HF_REVISION = "de79e57adf9983993bcc72586d7313460825f5eb"


@dataclass(frozen=True)
class OodSet:
    name: str
    config: str
    title: str
    licence: str
    homepage: str


SETS = {
    "voxpopuli": OodSet("voxpopuli", "voxpopuli", "VoxPopuli EN test",
                        "CC0 1.0 (transcribed data), also see https://www.europarl.europa.eu/legal-notice/en/",
                        "https://github.com/facebookresearch/voxpopuli"),
    "ami": OodSet("ami", "ami", "AMI Meeting Corpus, IHM, test (ESB 'eval' split)", "CC BY 4.0",
                  "https://groups.inf.ed.ac.uk/ami/corpus/"),
}

BLOCKED = {
    "tedlium": ("BLOCKED_RIGHTS: TED-LIUM was removed from hf-audio/open-asr-leaderboard on "
                "2026-05-19 (PR #9, commit 3edf1c4c); maintainer: 'The original authors (LIUM) also "
                "had to take down the dataset due to copyright/licensing'. Not substituted; owner "
                "decision needed."),
}


class SealedReadRefused(PermissionError):
    """A sealed PUBLIC-TEST read was attempted without the freeze conditions.

    repair_hint: reference texts are readable only after the bench freeze. Tag the freeze commit
    `bench-freeze-v1` (lead, BENCH_PLAN C3), then run once with `ULTRA_SEALED_READ=<look-id>`.
    For timing or cost runs use `load_audio(name)`, which needs no tag. docs: docs/ultra/data/ood_sets.md
    """

    def __init__(self, reason: str):
        super().__init__(f"sealed read refused: {reason}. {self.__doc__.split('repair_hint:')[1].strip()}")
        self.reason = reason


@dataclass
class OodAudio:
    uid: str
    pcm: np.ndarray  # int16, 16 kHz mono

    @property
    def seconds(self) -> float:
        return len(self.pcm) / SR


@dataclass
class OodUtt(OodAudio):
    text: str = ""


def _check_name(name: str) -> OodSet:
    if name in BLOCKED:
        raise LookupError(BLOCKED[name])
    if name not in SETS:
        raise KeyError(f"unknown OOD set {name!r}; have {sorted(SETS)}")
    return SETS[name]


def audio_path(name: str, cache: str = CACHE) -> str:
    return os.path.join(cache, f"{name}.audio.npz")


def sealed_path(name: str, cache: str = CACHE) -> str:
    return os.path.join(cache, "sealed", f"{name}.texts.npz")


def manifest_path(name: str, cache: str = CACHE) -> str:
    return os.path.join(cache, f"{name}.manifest.json")


def looks_path(repo: str = REPO) -> str:
    return os.path.join(repo, "docs", "ultra", "LOOKS.jsonl")


def sha256_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------------------------
# Audio-only access (allowed before the freeze)


def load_audio(name: str, cache: str = CACHE) -> list[OodAudio]:
    _check_name(name)
    z = np.load(audio_path(name, cache))
    offs, pcm, ids = z["offsets"], z["pcm"], z["ids"]
    return [OodAudio(str(ids[i]), pcm[offs[i]:offs[i + 1]]) for i in range(len(ids))]


# ---------------------------------------------------------------------------------------------
# Sealed access


def _git(repo: str, *args: str) -> str:
    r = subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True, check=False)
    return r.stdout.strip() if r.returncode == 0 else ""


def freeze_state(repo: str = REPO) -> dict:
    """Facts the gate decides on: tag presence, its commit, HEAD, and ancestry."""
    tag = _git(repo, "tag", "-l", FREEZE_TAG)
    head = _git(repo, "rev-parse", "HEAD")
    tag_commit = _git(repo, "rev-parse", f"{FREEZE_TAG}^{{commit}}") if tag else ""
    ancestor = bool(tag_commit) and subprocess.run(
        ["git", "-C", repo, "merge-base", "--is-ancestor", tag_commit, "HEAD"],
        capture_output=True, check=False).returncode == 0
    return {"tag": tag, "tag_commit": tag_commit, "head": head, "tag_is_ancestor": ancestor}


def _prior_looks(name: str, repo: str) -> list[dict]:
    p = looks_path(repo)
    if not os.path.exists(p):
        return []
    out = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                if d.get("kind") == "ood-sealed-read" and d.get("set") == name:
                    out.append(d)
    return out


def check_sealed_read(name: str, repo: str = REPO) -> tuple[str, dict]:
    """(look_id, freeze facts) if a sealed read of `name` is permitted now; else raise."""
    _check_name(name)
    look = os.environ.get(ENV_LOOK, "").strip()
    if not look:
        raise SealedReadRefused(f"env {ENV_LOOK}=<look-id> is not set")
    st = freeze_state(repo)
    if not st["tag"]:
        raise SealedReadRefused(f"git tag {FREEZE_TAG} does not exist in {repo}")
    if not st["tag_is_ancestor"]:
        raise SealedReadRefused(f"{FREEZE_TAG} ({st['tag_commit'][:12]}) is not an ancestor of HEAD")
    other = sorted({d["look_id"] for d in _prior_looks(name, repo) if d.get("look_id") != look})
    if other:
        raise SealedReadRefused(f"{name} was already read under look-id(s) {other}; O15 allows one read")
    return look, st


def load_split(name: str, cache: str = CACHE, repo: str = REPO) -> list[OodUtt]:
    """Audio + reference texts. Sealed: see module docstring. Logs the read to LOOKS.jsonl."""
    look, st = check_sealed_read(name, repo)
    sp = sealed_path(name, cache)
    entry = {"ts": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
             "kind": "ood-sealed-read", "look_id": look, "set": name, "user": getpass.getuser(),
             "head": st["head"], "freeze_tag_commit": st["tag_commit"],
             "sealed_sha256": sha256_file(sp)}
    with open(looks_path(repo), "a", encoding="utf-8") as f:  # log before the texts are opened
        f.write(json.dumps(entry, separators=(",", ":")) + "\n")
    audio = load_audio(name, cache)
    t = np.load(sp)
    if [str(x) for x in t["ids"]] != [a.uid for a in audio]:
        raise ValueError(f"sealed ids do not match audio ids for {name}")
    return [OodUtt(a.uid, a.pcm, str(x)) for a, x in zip(audio, t["texts"])]


# ---------------------------------------------------------------------------------------------
# Preparation: parquet -> caches. Never prints or returns reference text.


def _decode(b: bytes) -> tuple[np.ndarray, bool]:
    import soundfile as sf

    pcm, sr = sf.read(io.BytesIO(b), dtype="int16")
    if pcm.ndim != 1:
        raise ValueError(f"expected mono, got shape {pcm.shape}")
    if sr != SR:
        from scipy.signal import resample_poly

        g = np.gcd(sr, SR)
        y = resample_poly(pcm.astype(np.float32), SR // g, sr // g)
        return np.clip(np.round(y), -32768, 32767).astype(np.int16), True
    return pcm, False


def write_caches(name: str, ids: list[str], pcms: list[np.ndarray], texts: list[str],
                 cache: str = CACHE) -> dict:
    """Write audio cache and the separate sealed text file; return file hashes and counts."""
    if not (len(ids) == len(pcms) == len(texts)) or len(set(ids)) != len(ids):
        raise ValueError("ids/pcms/texts must align and ids must be unique")
    os.makedirs(os.path.join(cache, "sealed"), exist_ok=True)
    offs = np.cumsum([0] + [len(p) for p in pcms]).astype(np.int64)
    ap, sp = audio_path(name, cache), sealed_path(name, cache)
    np.savez(ap + ".tmp.npz", pcm=np.concatenate(pcms) if pcms else np.zeros(0, np.int16),
             offsets=offs, ids=np.array(ids))
    os.replace(ap + ".tmp.npz", ap)
    np.savez(sp + ".tmp.npz", ids=np.array(ids), texts=np.array(texts))
    os.replace(sp + ".tmp.npz", sp)
    return {"utterances": len(ids), "hours": round(float(offs[-1]) / SR / 3600, 4),
            "audio_npz_sha256": sha256_file(ap), "sealed_texts_npz_sha256": sha256_file(sp)}


def prepare(name: str, root: str = PARQUET_ROOT, cache: str = CACHE) -> dict:
    import pyarrow.parquet as pq

    s = _check_name(name)
    files = sorted(glob.glob(os.path.join(root, s.config, "test-*.parquet")))
    if not files:
        raise FileNotFoundError(f"no parquet shards for {name} under {root}/{s.config}")
    ids, pcms, texts, resampled = [], [], [], 0
    for f in files:
        pf = pq.ParquetFile(f)
        for rg in range(pf.num_row_groups):
            tab = pf.read_row_group(rg, columns=["id", "audio", "text"])
            for uid, audio, text in zip(tab.column("id").to_pylist(), tab.column("audio").to_pylist(),
                                        tab.column("text").to_pylist()):
                pcm, rs = _decode(audio["bytes"])
                resampled += rs
                ids.append(str(uid)), pcms.append(pcm), texts.append(text)
    man = {"schema": "ultra.ood-set/v1", "set": name, "title": s.title, "split": "test",
           "status": "PUBLIC-TEST (sealed until bench-freeze-v1; one read)",
           "source": {"hf_repo": HF_REPO, "revision": HF_REVISION, "config": s.config,
                      "homepage": s.homepage},
           "licence": s.licence,
           "parquet_sha256": {os.path.relpath(f, root): sha256_file(f) for f in files}}
    man.update(write_caches(name, ids, pcms, texts, cache))
    man["resampled_utterances"] = resampled
    with open(manifest_path(name, cache), "w") as fh:
        json.dump(man, fh, indent=1)
    man["manifest_sha256"] = sha256_file(manifest_path(name, cache))
    return man


def main(argv: list[str]) -> int:
    if len(argv) == 2 and argv[0] == "prepare":
        print(json.dumps(prepare(argv[1]), indent=1))
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
