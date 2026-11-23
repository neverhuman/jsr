#!/usr/bin/env bash
# h45-wins Arm A on xbabe1 (S half only, chunk 4, batch 32): 12 single checkpoints, 13 pre-registered merges, greedy soup.
cd /mnt/amphora-backup/h45-wins || exit 1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
export H45_CPUS=44-46,108-110
PY=/home/ubuntu/jsr/.venv/bin/python; SRC=src/main-e61121e/python/ultra; W=weights/src
TS="taskset -c $H45_CPUS nice -n 10"
D="$TS $PY src/h45_decode.py --src $SRC --jsr-data jsr_data --feat-cache featcache --batch 32 --device cuda --threads 2 --out out/a --uids src/uids_selection.txt --subset-tag .S --chunks 4"
# 1. singles (B0 first: its S-only decode is also the subset-consistency check)
$D --ckpt E49000=$W/A1-ckpt-00049000.pt@ema --ckpt R49000=$W/A1-ckpt-00049000.pt@model \
   --ckpt E47500=$W/A1-model-00047500.pt@ema --ckpt R47500=$W/A1-model-00047500.pt@model \
   --ckpt E45000=$W/A1-model-00045000.pt@ema --ckpt R45000=$W/A1-model-00045000.pt@model \
   --ckpt E42500=$W/A1-model-00042500.pt@ema --ckpt R42500=$W/A1-model-00042500.pt@model \
   --ckpt E40000=$W/A1-model-00040000.pt@ema --ckpt R40000=$W/A1-model-00040000.pt@model \
   --ckpt A1LE8000=$W/A1L-model-00008000.pt@ema --ckpt A1LR8000=$W/A1L-model-00008000.pt@model || exit 2
# 2. merges
R=""; for n in A-E2 A-E3 A-E4 A-E5 A-R2 A-R3 A-R4 A-R5 A-P4 A-P8 A-M10 A-L50 A-L25; do R="$R recipes/$n.json"; done
$TS $PY src/h45_merge.py --src $SRC --out-dir weights/derived $R || exit 3
C=""; for n in A-E2 A-E3 A-E4 A-E5 A-R2 A-R3 A-R4 A-R5 A-P4 A-P8 A-M10 A-L50 A-L25; do C="$C --ckpt $n=weights/derived/$n.pt@model"; done
$D $C || exit 4
# 3. scores on S
: > out/a/scores_S.jsonl
for t in E49000 R49000 E47500 R47500 E45000 R45000 E42500 R42500 E40000 R40000 A1LE8000 A1LR8000 A-E2 A-E3 A-E4 A-E5 A-R2 A-R3 A-R4 A-R5 A-P4 A-P8 A-M10 A-L50 A-L25; do
  $PY src/h45_score.py row out/a/$t.S.c4.ms8 >> out/a/scores_S.jsonl
done
# 4. greedy soup over the 12 singles
$PY - <<'PYEOF'
import json
W = "/mnt/amphora-backup/h45-wins/weights/src"
src = {"E49000": (f"{W}/A1-ckpt-00049000.pt", "ema"), "R49000": (f"{W}/A1-ckpt-00049000.pt", "model")}
for s in ("47500", "45000", "42500", "40000"):
    src["E" + s] = (f"{W}/A1-model-000{s}.pt", "ema"); src["R" + s] = (f"{W}/A1-model-000{s}.pt", "model")
src["A1LE8000"] = (f"{W}/A1L-model-00008000.pt", "ema"); src["A1LR8000"] = (f"{W}/A1L-model-00008000.pt", "model")
rows = [json.loads(l) for l in open("/mnt/amphora-backup/h45-wins/out/a/scores_S.jsonl")]
sc = {r["system"].split("/")[-1].split(".S.")[0]: r["pooled"]["wer"] for r in rows}
json.dump([{"tag": t, "ckpt": c, "key": k, "s_wer": sc[t]} for t, (c, k) in src.items()],
          open("/mnt/amphora-backup/h45-wins/out/a/singles.json", "w"), indent=1)
PYEOF
$PY src/h45_greedy_soup.py --singles out/a/singles.json --work out/a/gs --chunk 4
echo A-DONE
