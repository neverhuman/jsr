#!/usr/bin/env bash
# trainer-lane (new agent, 10:5xZ): branch-point watcher, run on H0, detached. Every 30 s it scans x5 (node 0) with
# ~/keep_bp_scan.py: for S2a / S2b, BPk = the first 500-step checkpoint past k x 10,000 audio-h of the run's own logged audio
# (Amendment 5). When a BP checkpoint is hard-linked in ~/jsr-data/protected/<run>/ (by keep_ckpts.sh), it hashes the
# protected file, checks it against the trainer's marker, and posts path + sha256 once (gate (a); S2b-BP1 = BP-m, gate (b)).
# It also posts once if x5's free space falls below 60 GiB (re-arms above 70).
set -u
CHAT=/home/ubuntu/jsr/AGENT_CHAT.md; ST=/mnt/xbabe3-external/jsr/trainer-lane/s2/state; mkdir -p $ST; touch $ST/posted
RUNS="H53_S2_small_w6_s7919 H53_S2b_medium_s7919 H53_S3_large_s7919"
post() {  # $1 = title, stdin = body; appended under the same flock other agents take on the chat file
  local body; body=$(cat)
  { flock 9; printf '\n## %s | trainer-lane | %s\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" "$body" >&9; } 9>>"$CHAT"
}
low=0
while :; do
  J=$(timeout 60 ssh -n -o ConnectTimeout=20 -o BatchMode=yes xbabe5 "nice -n 10 python3 -I ~/keep_bp_scan.py $RUNS" 2>/dev/null)
  if [ -n "$J" ]; then
    echo "$J" > $ST/last_scan.json
    free=$(python3 -I -c 'import json,sys; print(int(json.loads(sys.argv[1])["free_gb"]))' "$J")
    if [ "$free" -lt 60 ] && [ $low = 0 ]; then
      low=1; echo "- x5 \`~/jsr-data\` free = **${free} GiB** (floor 60). The medium finaliser must be paused; S2's checkpoints need room. @h52-librilight @claude-lead" | post "x5 DISK BELOW 60 GiB FLOOR (auto)"
    elif [ "$free" -gt 70 ]; then low=0; fi
    python3 -I - "$J" > $ST/ready.txt <<'PY'
import json, sys
j = json.loads(sys.argv[1])
for n, r in j["runs"].items():
    for b in r["bps"]:
        if b["prot_pt"]:
            print(n, b["k"], b["cross"], b["bp"], b["name"], b["marker"], b["links"], "%.0f" % r["cum_h"])
PY
    while read -r N K CROSS BP NAME MARK LINKS CUMH; do
      grep -qx "$N $NAME" $ST/posted && continue
      P=/home/ubuntu/jsr-data/protected/$N/$NAME.pt
      H=$(timeout 300 ssh -n -o BatchMode=yes xbabe5 "nice -n 10 ionice -c3 sha256sum $P | cut -c1-64" 2>/dev/null)
      [ ${#H} = 64 ] || continue
      SZ=$(timeout 30 ssh -n -o BatchMode=yes xbabe5 "stat -c %s $P")
      case $N in
        H53_S2_small*) SEG=S2a; GATE="gate (a) (EMA in the payload, chunk 4, pinned evaluator, greedy; stop rule > +0.20 pp vs B0 8.302)";;
        H53_S3_large*) SEG=S3; GATE="gate (a) (EMA in the payload, chunk 4, pinned evaluator, greedy; stop rule > +0.20 pp vs B0 8.302)";;
        *) SEG=S2b; if [ "$K" = 1 ]; then GATE="**gate (b): this is BP-m = S2b-BP1.** Stage 1 (gate (b) Amendment 1) decodes its stable EMA vs B0 at one-sided 97.5%, alongside gate (a) on the same EMA. Stage 2 (x9 fast cooldown from its RAW weights) runs only if stage 1 does not pass"; else GATE="gate (a)"; fi;;
      esac
      OK=$( [ "$H" = "$MARK" ] && echo "matches the trainer's marker" || echo "**MISMATCH vs marker $MARK**" )
      post "$SEG-BP$K READY: ckpt step $BP (10,000 x $K audio-h crossed at step $CROSS), protected, sha256 ${H:0:16}... (auto) @eval-lane @claude-lead" <<B
- **Path (use this one):** \`xbabe5:~/jsr-data/protected/$N/$NAME.pt\` (hard link, link count $LINKS; same inode as the run dir's copy until the trainer prunes that one). Marker: \`$NAME.sha256\` beside it.
- **sha256** \`$H\` ($OK), $SZ bytes. Run $N, logged exposure at the scan: ${CUMH} audio-h.
- **For:** $GATE. The EMA weights are in the payload (\`ema\`).
- Kept permanently as a branch point (keep_ckpts.sh on x5 never prunes BPs).
B
      echo "$N $NAME" >> $ST/posted; echo "$(date -u +%T) posted $N $NAME $H" >> $ST/bp_watch.log
    done < $ST/ready.txt
  fi
  sleep 30
done
