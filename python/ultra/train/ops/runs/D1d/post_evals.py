"""Append each new D1d eval line to AGENT_CHAT.md (append-only), until the run ends."""
import json, os, sys, time, datetime
run, chat = sys.argv[1], sys.argv[2]
pid = int(open(os.path.join(run, "PID")).read())
seen = int(sys.argv[3]) if len(sys.argv) > 3 else 1
def alive():
    try:
        os.kill(pid, 0); return True
    except OSError:
        return False
while True:
    evals = [json.loads(l) for l in open(os.path.join(run, "train.jsonl")) if '"kind": "eval"' in l]
    for r in evals[seen:]:
        trains = [json.loads(l) for l in open(os.path.join(run, "train.jsonl")) if '"kind": "train"' in l]
        t = trains[-1] if trains else {}
        ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
        msg = (f"\n## {ts} | trainer-lane | {os.path.basename(run)} dev-clean eval (auto)\n"
               f"- step {r['step']}, epoch {r.get('epoch')}: **WER {r['wer']}%** (sub {r['sub']} del {r['del']} ins {r['ins']}, "
               f"{r['ref_words']} words, 400 utts, greedy, chunk-masked), forced advances {r['forced_advances']}; "
               f"train RNNT {t.get('rnnt', 0):.3f} CTC {t.get('ctc', 0):.3f}/token; {t.get('audio_h_per_h', 0):.0f} audio-h/GPU-h.\n")
        with open(chat, "a") as fh:
            fh.write(msg)
    seen = len(evals)
    if not alive():
        break
    time.sleep(120)
