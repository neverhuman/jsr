#!/usr/bin/env bash
# trainer-lane: stage S3's large tier-A subset (Amendment 8) on all four island nodes. Run on H0 after h52 seals the large
# manifest on x6 (~/jsr-data/librilight/shards/r0_v11/librilight.large.s3/). x6's egress depends on the destination (measured
# 15:5xZ: x8 18.7 MB/s per stream / 67 MB/s with 4 streams, x7 15.5, x9 15.6, x5 3.1), so: x7 / x8 / x9 each pull one third of
# the subset from x6 (4 rsync streams each), then the cloud LAN fans each third out to the other three nodes (<= 250 MB/s per
# receiver), then `sha256sum -c` of the S3 manifest on all four. Paused (between steps) by the baseline-relative trunk guard.
#   large_stage.sh <h52's sealed SHA256SUMS sha256>
set -u
SEALED=$1
ST=/mnt/xbabe3-external/jsr/trainer-lane/s2/state; LOG=$ST/large_stage.log
D=jsr-data/librilight/shards/r0_v11/librilight.large.s3
X6=$(ssh -G xbabe6 | awk '/^hostname /{print $2}'); X6P=$(ssh -G xbabe6 | awk '/^port /{print $2}'); X6U=$(ssh -G xbabe6 | awk '/^user /{print $2}')
declare -A LAN=([xbabe5]=${X5_LAN:?} [xbabe7]=${X7_LAN:?} [xbabe8]=${X8_LAN:?} [xbabe9]=${X9_LAN:?})
declare -A SPARE=([xbabe5]=56-59 [xbabe7]=56-59 [xbabe8]=28-29 [xbabe9]=28-29)
declare -A FLOOR=([xbabe5]=60 [xbabe7]=20 [xbabe8]=20 [xbabe9]=20)
TRUNK=/home/ubuntu/jsr-data/runs/train/H53_S2b_medium_s7919
log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a $LOG; }
trunk_tail() { ssh -n -o BatchMode=yes xbabe5 "grep '\"kind\": \"train\"' $TRUNK/train.jsonl | tail -$1" 2>/dev/null; }
read -r BW_BASE BR_BASE <<< "$(trunk_tail 40 | python3 -c 'import json,sys,statistics as s; r=[json.loads(l) for l in sys.stdin]; print(s.median(x["data_wait_pct"] for x in r), s.median(x["audio_h_per_h"] for x in r))')"
guard() {
  while :; do
    w=$(trunk_tail 2 | python3 -c "import json,sys; r=[json.loads(l) for l in sys.stdin]; th=max(5.0, $BW_BASE + 0.5); print(int(len(r) == 2 and (all(x['data_wait_pct'] > th for x in r) or all(x['audio_h_per_h'] < 0.95 * $BR_BASE for x in r))))" 2>/dev/null)
    [ "${w:-1}" = 0 ] && return
    log "guard: trunk outside its baseline band; pausing 30 s"; sleep 30
  done
}
X6SSH="ssh -p $X6P -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o BatchMode=yes -c aes128-gcm@openssh.com"
RSH="ssh -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -c aes128-gcm@openssh.com"
eval "$(ssh-agent -s)" > /dev/null; ssh-add -q ~/.ssh/id_ed25519; trap 'ssh-agent -k > /dev/null 2>&1' EXIT
# 1. the sealed manifest and the shard sizes, from x6
ssh -o BatchMode=yes xbabe6 "cat ~/$D/SHA256SUMS" < /dev/null 2>/dev/null > $ST/large_SHA256SUMS
[ "$(sha256sum $ST/large_SHA256SUMS | cut -c1-64)" = "$SEALED" ] || { log "REFUSED: x6 SHA256SUMS != sealed $SEALED"; exit 1; }
ssh -o BatchMode=yes xbabe6 "cd ~/$D && stat -c '%s %n' [0-9]*.parquet" < /dev/null 2>/dev/null > $ST/large_sizes.txt
# 2. the budget: min over nodes (free - floor), minus 6 GB for S3's own run dir
B=999999; for h in xbabe5 xbabe7 xbabe8 xbabe9; do f=$(ssh -n -o BatchMode=yes $h "df -BG --output=avail ~/jsr-data | tail -1 | tr -dc 0-9"); b=$((f - FLOOR[$h] - 6)); log "$h free ${f} GB -> budget ${b} GB"; [ $b -lt $B ] && B=$b; done
cat > $ST/tl_subset.py <<'PY'
import sys
st, budget = sys.argv[1], int(sys.argv[2]) * 2**30
sums = {l.split()[1]: l for l in open(st + "/large_SHA256SUMS") if l.strip()}
size = {n: int(s) for s, n in (l.split() for l in open(st + "/large_sizes.txt"))}
names = sorted(sums)
assert names == ["%04d.parquet" % i for i in range(len(names))], "manifest is not 0000..N-1"
tot, k = 0, 0
for n in names:
    if tot + size[n] > budget: break
    tot += size[n]; k += 1
open(st + "/s3_SHA256SUMS", "w").write("".join(sums[n] for n in names[:k]))
for i, part in enumerate(("xbabe7", "xbabe8", "xbabe9")):
    open(st + "/s3_part_%s.txt" % part, "w").write("".join(n + "\n" for n in names[:k][i::3]))
print(k, tot)
PY
read -r N TOT <<< "$(python3 $ST/tl_subset.py "$ST" "$B")"
[ "${N:-0}" -gt 0 ] || { log "REFUSED: empty subset (budget $B GB)"; exit 1; }
S3SUM=$(sha256sum $ST/s3_SHA256SUMS | cut -c1-64)
log "S3 subset: first $N of $(wc -l < $ST/large_SHA256SUMS) shards, $TOT bytes; S3 SHA256SUMS sha256 $S3SUM"
# 3. x7 / x8 / x9 each pull their third from x6 (4 streams each)
guard
for h in xbabe7 xbabe8 xbabe9; do
  ssh -o BatchMode=yes $h "mkdir -p ~/$D && cat > /dev/shm/tl_s3part.txt" < $ST/s3_part_$h.txt
  ssh -A -n -o BatchMode=yes $h "cd ~/$D && split -n r/4 /dev/shm/tl_s3part.txt /dev/shm/tl_s3p. && for f in /dev/shm/tl_s3p.??; do taskset -c ${SPARE[$h]} nice -n 15 ionice -c3 rsync -a --files-from=\$f -e '$X6SSH' $X6U@$X6:$D/ ./ & done; wait; rm -f /dev/shm/tl_s3p.??" &
done; wait
for h in xbabe7 xbabe8 xbabe9; do log "$h pulled its third: $(ssh -n -o BatchMode=yes $h "cd ~/$D && ls [0-9]*.parquet 2>/dev/null | wc -l") files"; done
# 4. LAN fan-out: each third to the other three nodes (each pair capped at 83 MB/s, so a receiver takes <= 250 MB/s)
guard
for s in xbabe7 xbabe8 xbabe9; do for r in xbabe5 xbabe7 xbabe8 xbabe9; do [ $s = $r ] && continue
  ssh -A -n -o BatchMode=yes $s "cd ~/$D && taskset -c ${SPARE[$s]} nice -n 15 ionice -c3 rsync -a --bwlimit=83000 --files-from=/dev/shm/tl_s3part.txt -e '$RSH' --rsync-path='mkdir -p ~/$D && taskset -c ${SPARE[$r]} nice -n 15 ionice -c3 rsync' ./ ubuntu@${LAN[$r]}:$D/" &
done; done; wait
# 5. the S3 manifest on every node, then sha256sum -c (parallel, spare cores) + file set == manifest set
for h in xbabe5 xbabe7 xbabe8 xbabe9; do ssh -o BatchMode=yes $h "cat > ~/$D/SHA256SUMS" < $ST/s3_SHA256SUMS; done
for h in xbabe5 xbabe7 xbabe8 xbabe9; do
  ( out=$(ssh -n -o BatchMode=yes $h "cd ~/$D && [ \"\$(sha256sum SHA256SUMS | cut -c1-64)\" = $S3SUM ] || { echo 'MANIFEST DIFFERS'; exit 1; }
      diff <(ls | grep '\.parquet\$' | sort) <(awk '{print \$2}' SHA256SUMS | sort) > /dev/null || { echo 'FILE SET DIFFERS'; exit 1; }
      split -n l/2 SHA256SUMS /dev/shm/tl_s3sum.; fail=0
      for f in /dev/shm/tl_s3sum.??; do taskset -c ${SPARE[$h]} nice -n 10 ionice -c3 sha256sum -c --quiet \$f > \$f.out 2>&1 & done; wait
      for f in /dev/shm/tl_s3sum.*.out; do [ -s \$f ] && { cat \$f; fail=1; }; done; rm -f /dev/shm/tl_s3sum.* /dev/shm/tl_s3part.txt
      [ \$fail = 0 ] && echo \"OK \$(ls *.parquet | wc -l) shards \$(du -sh . | cut -f1)\"" 2>&1 | tail -3); log "$h verify: $out" ) &
done; wait
log "S3 STAGED $S3SUM $N"
