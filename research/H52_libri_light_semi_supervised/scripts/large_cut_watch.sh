#!/usr/bin/env bash
# H0: at 15:30:00Z fire the tier-A cut on x6 (finalize_large_cut_x6.sh), then post the sealed manifest (or an alert).
set -u
D=/home/ubuntu/.h52-labeler/ll; RES=/mnt/xbabe3-external/jsr/librilight/research/H52_libri_light_semi_supervised/results
SUB=${1:-large.s3}; GB=${2:-300}; AT=${3:-153000}
LOG="~/jsr-data/librilight/logs/finalize_large_cut.log"; OUTD="jsr-data/librilight/shards/r0_v11/librilight.$SUB"
until [ "$(date -u +%H%M%S)" -ge "$AT" ]; do sleep 10; done
timeout 60 ssh -o BatchMode=yes xbabe6 "cd ~/jsr-data/librilight && rm -f logs/finalize_large_cut.pid && (setsid nohup bash bin/finalize_large_cut_x6.sh $SUB $GB 8 > logs/finalize_large_cut.log 2>&1 < /dev/null &); sleep 5"
pg=$(timeout 30 ssh -o BatchMode=yes xbabe6 "cat ~/jsr-data/librilight/logs/finalize_large_cut.pid")
echo "$(date -u +%H:%M:%SZ) cut fired on x6, pid $pg"
while :; do
  sleep 60
  body=$(timeout 60 ssh -o BatchMode=yes xbabe6 "cat $LOG" 2>/dev/null) || continue
  if printf '%s\n' "$body" | grep -q "finalise $SUB done"; then break; fi
  alive=$(timeout 60 ssh -o BatchMode=yes xbabe6 "ps -o pid= -p $pg" 2>/dev/null)
  if [ -z "$alive" ]; then
    { echo "@claude-lead @trainer-lane"; echo "- **The tier-A cut stopped on x6 before sealing.** I'm on it. Log tail:"; printf '%s\n' "$body" | tail -10 | sed 's/^/    /'; } > $D/body_cut.md
    $D/chat_post.sh "H52 ALERT: large tier-A cut stopped on x6 before sealing" $D/body_cut.md; exit 1
  fi
done
mkdir -p $RES
timeout 120 scp -q xbabe6:$OUTD/FILTER_REPORT.json $RES/${SUB}_FILTER_REPORT.json
timeout 120 scp -q xbabe6:$OUTD/SHA256SUMS $RES/${SUB}_SHA256SUMS
snap=$(timeout 60 ssh -o BatchMode=yes xbabe6 "grep 'cut snapshot' $LOG | tail -1")
bytes=$(timeout 60 ssh -o BatchMode=yes xbabe6 "du -sb ~/$OUTD | cut -f1")
python3 - "$SUB" "$snap" "$bytes" > $D/body_cut.md <<'PY'
import hashlib, json, sys
tag, snap, nbytes = sys.argv[1], sys.argv[2], int(sys.argv[3] or 0)
d = "/mnt/xbabe3-external/jsr/librilight/research/H52_libri_light_semi_supervised/results/"
r = json.load(open(d + f"{tag}_FILTER_REPORT.json"))
m = hashlib.sha256(open(d + f"{tag}_SHA256SUMS", "rb").read()).hexdigest()
w = r.get("output_shards_written", r["output_shards"]); tot = r.get("output_shards_total", r["output_shards"])
print("@claude-lead @trainer-lane @h53-trunk @eval-lane")
print(f"- **Sealed manifest:** `SHA256SUMS` sha256 **`{m}`**.")
print(f"  - {w} shards written, of {tot} for the full kept set.")
print(f"  - {nbytes / 1e9:.1f} GB at `x6:~/jsr-data/librilight/shards/r0_v11/librilight.{tag}/`.")
print(f"- **Written:** {r.get('rows_written', r['rows_kept']):,} rows = **{r.get('hours_written', r['hours_kept']):,} h**.")
print("  - Each shard is a uniform random sample of the kept rows, so any prefix `0000…N` is too.")
print(f"- **All labelled at the cut:** kept {r['rows_kept']:,} / {r['rows_in']:,} labelled rows ({r['kept_fraction_rows']:.1%}) = {r['hours_kept']:,} h.")
print(f"  - Rows still unlabelled at the cut, dropped: {r['checks'].get('rows_unlabeled', 0):,}.")
print(f"- **Drops:** {json.dumps(r['drop_counts'])}; S1 threshold {r['S1_lp_threshold']:.4f}.")
print(f"  - Checks: {json.dumps(r['checks'])}.")
print(f"- **Inputs (frozen at the cut):** {snap.split('cut snapshot: ')[-1]}. v1.1-only policy and the same filter / seed / amendments 2-4 as medium, including the NaN guard.")
print("- **@trainer-lane:** stage any prefix per node, then `sha256sum -c` its lines from this manifest. Labelling continues on all cards.")
PY
$D/chat_post.sh "H52 RESULT: large tier-A cut finalised on x6 (sealed manifest, shards, hours)" $D/body_cut.md
