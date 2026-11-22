"""Stage the pinned paper DEV audio and rescore retained cold-service hypotheses."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import wave


PANEL_SHA256 = "76fead746d1f8b8ae57a30aa341e81364538eb69759af813d6f13d6d1f57fc36"
NORMALIZER_SHA256 = "988c59acad873bf0a0c50bdcf434dde0b5125cf51c2cef887bfc63f4f89113e8"
SPLIT_COUNTS = {"dev-clean": 400, "dev-other": 330}


def sha256(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def read_json(path):
    return json.loads(path.read_bytes())


def write_json(path, value):
    with path.open("x", encoding="utf-8") as output:
        json.dump(value, output, indent=2)
        output.write("\n")


def stage(panel_path, cache_dir, out):
    import numpy as np

    panel = read_json(panel_path)
    declared = panel.pop("manifest_sha256")
    actual = hashlib.sha256(json.dumps(panel, indent=1, sort_keys=True).encode()).hexdigest()
    if actual != declared or actual != PANEL_SHA256:
        raise ValueError("unexpected DEV panel identity")
    splits = panel["splits"]
    if [split["split"] for split in splits] != list(SPLIT_COUNTS):
        raise ValueError("expected the ordered dev-clean and dev-other splits")
    wav_dir = out / "wav"
    wav_dir.mkdir(parents=True)
    audio_manifest = []
    index = []
    for split in splits:
        name = split["split"]
        expected = split["uids"]
        if len(set(expected)) != len(expected) or len(expected) != SPLIT_COUNTS[name]:
            raise ValueError(f"unexpected UID population for {name}")
        path = cache_dir / f"{name}.npz"
        if sha256(path) != split["cache_sha256"]:
            raise ValueError(f"unexpected DEV cache identity for {name}")
        with np.load(path, allow_pickle=False) as source:
            pcm = source["pcm"]
            offsets = source["offsets"]
            uids = source["uids"]
            speakers = source["speakers"]
            texts = source["texts"]
        positions = {uid: pos for pos, uid in enumerate(uids)}
        if len(positions) != len(uids):
            raise ValueError(f"duplicate cache UID in {name}")
        for uid in expected:
            pos = positions[uid]
            speaker = split["speaker_of"][uid]
            if int(speakers[pos]) != speaker:
                raise ValueError(f"speaker mismatch for {uid}")
            ident = f"u{len(index):05d}"
            samples = pcm[offsets[pos]:offsets[pos + 1]].astype("<i2", copy=False)
            wav = wav_dir / f"{ident}.wav"
            with wave.open(str(wav), "wb") as output:
                output.setparams((1, 2, 16000, len(samples), "NONE", "not compressed"))
                output.writeframes(samples.tobytes())
            audio_manifest.append({
                "id": ident, "samples": len(samples),
                "pcm_sha256": hashlib.sha256(samples.tobytes()).hexdigest(), "wav_sha256": sha256(wav),
            })
            index.append({
                "id": ident, "uid": uid, "speaker": str(speaker), "dataset": name,
                "ref": str(texts[pos]), "duration_s": len(samples) / 16000,
            })
    write_json(out / "audio.json", audio_manifest)
    write_json(out / "index.json", index)


def verify(staged):
    audio = read_json(staged / "audio.json")
    expected = {f"u{pos:05d}.wav": row["wav_sha256"] for pos, row in enumerate(audio)}
    actual = {path.name: sha256(path) for path in (staged / "wav").iterdir()}
    if len(expected) != sum(SPLIT_COUNTS.values()) or actual != expected:
        raise ValueError("staged WAV population or audio hashes differ from the manifest")
    if any(row["id"] + ".wav" not in expected for row in audio):
        raise ValueError("invalid WAV identity in the audio manifest")


def hypotheses(run):
    predictions = run / "predictions.jsonl"
    if predictions.is_file():
        rows = map(json.loads, predictions.read_bytes().splitlines())
        results = {}
        for row in rows:
            ident = row["id"]
            if ident in results or not isinstance(row["hyp"], str):
                raise ValueError(f"duplicate or invalid hypothesis for {ident}")
            results[ident] = row["hyp"]
        return results
    results = {}
    for path in (run / "outputs").iterdir():
        if not path.is_file() or path.suffix != ".txt":
            raise ValueError(f"invalid recognizer output {path}")
        results[path.stem] = path.read_text(encoding="utf-8")
    return results


def score(staged, run, scorer, service_cpu):
    measurement = read_json(run / "measurement.json")
    placement = measurement.get("placement") or {}
    cpu = measurement["cpu_seconds_user"] + measurement["cpu_seconds_sys"]
    if (measurement["exit_code"] != 0 or not math.isfinite(cpu) or cpu <= 0
            or measurement["peak_rss_bytes"] <= 0 or not placement.get("separate_l3_and_smt")
            or placement.get("effective_service_cpus_at_spawn") != [service_cpu]):
        raise ValueError("cold service failed, lacked resource accounting, or was misplaced")
    audio = read_json(staged / "audio.json")
    index = read_json(staged / "index.json")
    hyps = hypotheses(run)
    if list(row["id"] for row in audio) != list(row["id"] for row in index):
        raise ValueError("audio manifest and score index disagree")
    ids = {row["id"] for row in index}
    if len(ids) != sum(SPLIT_COUNTS.values()) or ids != hyps.keys():
        raise ValueError(f"wrong hypothesis population, missing {ids - hyps.keys()}, extra {hyps.keys() - ids}")
    score_input = run / "scores.jsonl"
    with score_input.open("x", encoding="utf-8") as out:
        for reference in index:
            ident = reference["id"]
            uid = reference["uid"]
            row = {key: val for key, val in reference.items() if key not in {"id", "uid"}}
            json.dump({**row, "id": uid, "hyp": hyps[ident]}, out)
            out.write("\n")
    scoring = subprocess.run([scorer, score_input], capture_output=True, check=True)
    report = json.loads(scoring.stdout)
    if report["normaliser_sha256"] != NORMALIZER_SHA256:
        raise ValueError("unexpected Rust normalizer")
    (run / "score.json").write_bytes(scoring.stdout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    setup = sub.add_parser("stage")
    setup.add_argument("panel", type=Path)
    setup.add_argument("cache_dir", type=Path)
    setup.add_argument("out", type=Path)
    verification = sub.add_parser("verify")
    verification.add_argument("staged", type=Path)
    scoring = sub.add_parser("score")
    scoring.add_argument("staged", type=Path)
    scoring.add_argument("run", type=Path)
    scoring.add_argument("scorer", type=Path)
    scoring.add_argument("service_cpu", type=int)
    args = parser.parse_args()
    if args.command == "stage":
        stage(args.panel, args.cache_dir, args.out)
    elif args.command == "verify":
        verify(args.staged)
    else:
        score(args.staged, args.run, args.scorer, args.service_cpu)


if __name__ == "__main__":
    main()
