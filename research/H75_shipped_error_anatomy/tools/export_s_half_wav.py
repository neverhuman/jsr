"""Export the dev_split_v2 SELECTION half (S) to 16 kHz int16 WAVs + refs.jsonl (C half never touched)."""
import json, sys, wave
from pathlib import Path
sys.path[:0] = ["/home/ubuntu/jsr/.agent/worktrees/wer-claude/python/ultra"]
from train.wer_diag import load_selection_dev  # guards against C-half uids

split_json, out = sys.argv[1], Path(sys.argv[2])
(out / "wav").mkdir(parents=True, exist_ok=True)
utts, subsets = load_selection_dev(split_json)
with open(out / "refs.jsonl", "w") as fh:
    for u in utts:
        p = out / "wav" / f"{u.uid}.wav"
        with wave.open(str(p), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(u.pcm.tobytes())
        fh.write(json.dumps({"uid": u.uid, "speaker": u.speaker, "subset": subsets[u.uid], "ref": u.text, "samples": int(len(u.pcm))}) + "\n")
print(len(utts), "utts", sum(1 for _ in (out / "wav").iterdir()), "wavs")
