#!/usr/bin/env bash
# trainer-lane (lead 14:18:41Z, S3 disk decision (b)): MOVE (never just delete) non-canonical research run dirs and x8's
# superseded v1-small copy from x8 / x9 to x7. Per path: sha256 manifest of every regular file at the source, rsync to
# x7:~/jsr-data/moved/from_<x8|x9>/<same relative path> (<= 250 MB/s, spare cores, idle IO), `sha256sum -c` of every file
# AT x7 plus file-count and byte checks, and only then removal of the source (after checking no process uses it).
# Every path is logged with bytes and destination in x7:~/jsr-data/moved/MOVES.txt. Paused while the trunk's logged data
# wait is > 5% on its last 2 records. Starts only after every RC1 torchrun has exited.   moves.sh [--dry-run]
set -u
DRY=${1:-}
ST=/mnt/xbabe3-external/jsr/trainer-lane/s2/state; LOG=$ST/moves.log
X7=${X7_LAN:?}   # x7's cloud-LAN address (private; never posted)
TRUNK=/home/ubuntu/jsr-data/runs/train/H53_S2b_medium_s7919 RC=/home/ubuntu/jsr-data/runs/train/H53_RC1_s7919
log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a $LOG; }
X8="runs/train/D2c_S_warm_pruned_960h runs/train/D2d_S_960h runs/train/D2d_S_960h.raw-start runs/train/D2f_S_960h
 runs/train/H41_chunk_s4321 runs/train/H48c_b_s1234 runs/train/H53_S2p_bridge_s1234 runs/train/F1_S_MR_r0bpe_960h
 runs/train/F1_S_MR_r0bpe_960h.b2000-probe runs/train/L1_S_spm512_pruned_960h runs/train/L1_S_spm512_pruned_960h_probe1200
 runs/train/L1w_S_spm512_warmH13a_960h librilight/shards/r0_small_tl"
X9="runs/train/A1_S_r0bpe_960h runs/train/D2c_S_warm_pruned_960h runs/train/D2c_xhost_bf16 runs/train/D2c_xhost_fp32
 runs/train/D2d_S_960h runs/train/D2d_S_960h.raw-start runs/train/D2e_S_960h runs/train/D2f_S_960h
 runs/train/H41_chunk2_dp0.01_s4321 runs/train/H41_chunk_left256_s4321 runs/train/H41_chunk_specaug
 runs/train/H48b_c2_dp0.0075_s4321 runs/train/H48c_a_s1234 runs/train/H53_S2c_bridge_s1234 runs/train/H53_S2c_bridge_s4321"
valid() { [[ "$1" =~ ^runs/train/[A-Za-z0-9_.+-]+$ || "$1" == librilight/shards/r0_small_tl ]] && [[ "$1" != *H53_S2_small* && "$1" != *H53_S2b* && "$1" != *H53_RC1* && "$1" != *seeds* ]]; }
# Guard (lead 14:18:41Z rule, made baseline-relative at 15:0xZ because the trunk's own logged data wait sat at ~5.7% with
# no move running): pause while the trunk's last 2 records BOTH exceed max(5%, baseline + 0.5 pp) data wait, or its rate is
# below 0.95 x baseline; the baseline = the median of the trunk's last 40 records (~10 min) before the first move.
trunk_tail() { ssh -n -o BatchMode=yes xbabe5 "grep '\"kind\": \"train\"' $TRUNK/train.jsonl | tail -$1" 2>/dev/null; }
read -r BW_BASE BR_BASE <<< "$(trunk_tail 40 | python3 -c 'import json,sys,statistics as s; r=[json.loads(l) for l in sys.stdin]; print(s.median(x["data_wait_pct"] for x in r), s.median(x["audio_h_per_h"] for x in r))')"
guard() {
  while :; do
    w=$(trunk_tail 2 | python3 -c "import json,sys; r=[json.loads(l) for l in sys.stdin]; th=max(5.0, $BW_BASE + 0.5); print(int(len(r) == 2 and (all(x['data_wait_pct'] > th for x in r) or all(x['audio_h_per_h'] < 0.95 * $BR_BASE for x in r))))" 2>/dev/null)
    [ "${w:-1}" = 0 ] && return
    log "guard: trunk data wait / rate outside the baseline band; pausing 30 s"; sleep 30
  done
}
if [ -z "$DRY" ]; then
  for spec in "xbabe5 0" "xbabe7 1" "xbabe8 2" "xbabe9 3"; do set -- $spec
    until ssh -n -o BatchMode=yes $1 "! kill -0 \$(cat $RC/PID.node$2 2>/dev/null) 2>/dev/null" 2>/dev/null; do sleep 15; done
  done
  log "every RC1 torchrun has exited; moves may start"
  eval "$(ssh-agent -s)" > /dev/null; ssh-add -q ~/.ssh/id_ed25519; trap 'ssh-agent -k > /dev/null 2>&1' EXIT
fi
RSH="ssh -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -c aes128-gcm@openssh.com"
move_one() {  # $1 = source host, $2 = role (x8 | x9), $3 = path relative to ~/jsr-data
  local h=$1 r=$2 p=$3 n k nb kb ok dest=jsr-data/moved/from_$2
  valid "$p" || { log "SKIP (not an allowed path): $r $p"; return; }
  read -r n k <<< "$(ssh -n -o BatchMode=yes $h "cd ~/jsr-data && [ -d '$p' ] && echo \$(find '$p' -type f -printf '%s\n' | awk '{s+=\$1} END {print s+0}') \$(find '$p' -type f | wc -l)")"
  [ -n "${n:-}" ] || { log "SKIP (absent): $r $p"; return; }
  if [ -n "$DRY" ]; then log "DRY $r:$p $n bytes $k files -> x7:~/$dest/$p"; return; fi
  guard
  ssh -n -o BatchMode=yes $h "cd ~/jsr-data && find '$p' -type f -print0 | sort -z | nice -n 15 ionice -c3 taskset -c 28-29 xargs -0 sha256sum > /dev/shm/tl_mv.sha256 && wc -l < /dev/shm/tl_mv.sha256" > /dev/null || { log "FAIL source manifest $r $p"; return; }
  ssh -n -o BatchMode=yes $h "cat /dev/shm/tl_mv.sha256" > $ST/tl_mv_$r.sha256
  guard
  ssh -A -n -o BatchMode=yes $h "cd ~/jsr-data && taskset -c 28-29 nice -n 15 ionice -c3 rsync -a --bwlimit=250000 --relative -e '$RSH' --rsync-path='mkdir -p ~/$dest && taskset -c 56-59 nice -n 15 ionice -c3 rsync' './$p' ubuntu@$X7:$dest/" || { log "FAIL rsync $r $p"; return; }
  ssh -o BatchMode=yes xbabe7 "mkdir -p ~/$dest/$(dirname $p) && cat > ~/$dest/$p.sha256" < $ST/tl_mv_$r.sha256
  read -r ok nb kb <<< "$(ssh -n -o BatchMode=yes xbabe7 "cd ~/$dest && nice -n 10 ionice -c3 taskset -c 56-59 sha256sum -c --quiet '$p.sha256' > /dev/null 2>&1 && echo OK || echo BAD; find '$p' -type f -printf '%s\n' | awk '{s+=\$1} END {print s+0}'; find '$p' -type f | wc -l" | tr '\n' ' ')"
  if [ "$ok" = OK ] && [ "$nb" = "$n" ] && [ "$kb" = "$k" ]; then
    busy=$(ssh -n -o BatchMode=yes $h "for q in /proc/[0-9]*; do readlink \$q/cwd; ls -l \$q/fd 2>/dev/null; done 2>/dev/null | grep -c \"jsr-data/$p\"")
    if [ "${busy:-1}" = 0 ]; then
      ssh -n -o BatchMode=yes $h "rm -rf -- \"\$HOME/jsr-data/$p\" && [ ! -e \"\$HOME/jsr-data/$p\" ]" && st="source removed" || st="SOURCE REMOVAL FAILED"
    else st="source KEPT (in use by $busy handles)"; fi
  else st="VERIFY FAILED (sha $ok, bytes $nb/$n, files $kb/$k); source KEPT"; fi
  line="$(date -u +%Y-%m-%dT%H:%M:%SZ) $r:~/jsr-data/$p $n bytes $k files -> x7:~/$dest/$p | sha256 manifest $p.sha256 | verify ${ok:-?} | $st"
  ssh -o BatchMode=yes xbabe7 "cat >> ~/jsr-data/moved/MOVES.txt" <<< "$line"; log "$line"
}
for p in $X8; do move_one xbabe8 x8 "$p"; done &
for p in $X9; do move_one xbabe9 x9 "$p"; done &
wait
log "MOVES DONE"; for h in xbabe8 xbabe9 xbabe7; do log "$h free $(ssh -n -o BatchMode=yes $h 'df -BG --output=avail ~/jsr-data | tail -1')"; done
