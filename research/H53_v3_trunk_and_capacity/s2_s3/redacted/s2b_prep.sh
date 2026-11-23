#!/usr/bin/env bash
# trainer-lane: S2b prep on the sealed, staged medium (run on H0 after MEDIUM STAGED): exact extra_base_weight from the per-row
# FLAC tables (x7), then the Amendment 6 per-batch audit for window_groups 96 / 768 / 1024 / 1536 over the first 2,000 batches.
# Writes state/s2b_prep.json and prints the chosen window_groups by the rule fixed at 11:19Z (lead ACK 11:17:33Z).
set -u
ST=/mnt/xbabe3-external/jsr/trainer-lane/s2/state
LSR=/home/ubuntu/jsr-data/corpora/librispeech/hf_parquet/all
SM=/home/ubuntu/jsr-data/librilight/shards/r0_v11/librilight.small MD=/home/ubuntu/jsr-data/librilight/shards/r0_v11/librilight.medium
TB=/home/ubuntu/trainer-lane/dur_ls_small.jsonl,/home/ubuntu/trainer-lane/dur_medium.jsonl
PYX="cd ~/jsr/.agent/worktrees/trainer-s2/python/ultra && JSR_DATA=~/jsr-data PYTHONPATH=\$PWD PYTHONDONTWRITEBYTECODE=1 nice -n 15 taskset -c 56-59 ~/jsr-data/venvs/train/bin/python"
# make sure the medium table covers every sealed shard (incremental; re-scans changed files)
ssh -n -o BatchMode=yes xbabe7 "cd ~ && nice -n 15 ionice -c3 taskset -c 56-59 ~/jsr-data/venvs/train/bin/python -I ~/keep_dur_table.py ~/trainer-lane/dur_medium.jsonl 4 $MD" > $ST/s2b_prep_dur.txt
BWJ=$(ssh -n -o BatchMode=yes xbabe7 "$PYX -I ~/keep_bw_compute.py $TB $LSR $SM $MD")
BW=$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["extra_base_weight"])' "$BWJ") || { echo "bw failed: $BWJ"; exit 1; }
A=()
for wg in 96 768 1024 1536; do
  A+=("$(ssh -n -o BatchMode=yes xbabe7 "$PYX ~/keep_fast_audit.py $TB $LSR $BW 2000 $wg $SM $MD")")
done
python3 - "$BWJ" "${A[@]}" > $ST/s2b_prep.json <<'PY'
import json, sys
bw = json.loads(sys.argv[1]); au = [json.loads(a) for a in sys.argv[2:]]
by = {a["window_groups"]: a for a in au}
ok = [w for w in (768, 1024) if by[w]["pass"]]
if ok:
    wg, why = ok[0], "smallest of {768, 1024} that passes"
else:
    best = min((768, 1024, 1536), key=lambda w: by[w]["ll_share_p95"])
    wg, why = (768, "neither passes; 768 within 0.01 of the best p95") if by[768]["ll_share_p95"] - by[best]["ll_share_p95"] <= 0.01 \
        else (best, "neither passes; the best p95 of {768, 1024, 1536}")
print(json.dumps({"bw": bw, "audits": au, "chosen_window_groups": wg, "rule": why}, indent=1))
PY
python3 -c 'import json; j = json.load(open("'$ST'/s2b_prep.json")); print("BW", j["bw"]["extra_base_weight"], "WG", j["chosen_window_groups"], "|", j["rule"]); [print(" wg %5d pooled %.4f median %.4f p5 %.4f p95 %.4f pass %s" % (a["window_groups"], a["ll_audio_share_pooled"], a["ll_share_median"], a["ll_share_p5"], a["ll_share_p95"], a["pass"])) for a in j["audits"]]'
