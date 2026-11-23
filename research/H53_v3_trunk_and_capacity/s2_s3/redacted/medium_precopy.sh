#!/usr/bin/env bash
# trainer-lane (new agent, 11:0xZ): S2b staging, Amendment 6 condition 2. Run on H0, detached. I am the one writer of
# medium on x7 / x8 / x9. Phase 1 (pre-copy): every 90 s, rsync the finaliser's COMPLETED final shards (NNNN.parquet only:
# never *.part, _buckets/, SHA256SUMS or FILTER_REPORT.json) from x5 over the cloud LAN, pinned to the spare cores at idle
# priority, bandwidth-limited. Phase 2, once sealed (SHA256SUMS present, its line count = the parquet count, unchanged for
# 2 polls): a final rsync with --delete (so no stray parquet can reach the loader's glob) including the manifest, then on
# all four nodes: manifest sha256 equal, file set == manifest set, `sha256sum -c` (parallel, spare cores). Prints MEDIUM STAGED.
set -u
M=jsr-data/librilight/shards/r0_v11/librilight.medium
ST=/mnt/xbabe3-external/jsr/trainer-lane/s2/state; LOG=$ST/medium_copy.log
declare -A SPARE=([xbabe5]=56-59 [xbabe7]=56-59 [xbabe8]=28-29 [xbabe9]=28-29)
declare -A NP=([xbabe5]=4 [xbabe7]=4 [xbabe8]=2 [xbabe9]=2)
# cloud-LAN addresses of the copy targets (private; never post them)
declare -A LAN=([xbabe7]=${X7_LAN:?} [xbabe8]=${X8_LAN:?} [xbabe9]=${X9_LAN:?})
log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" >> $LOG; }
eval "$(ssh-agent -s)" > /dev/null; ssh-add -q ~/.ssh/id_ed25519; trap 'ssh-agent -k > /dev/null 2>&1' EXIT
for h in xbabe7 xbabe8 xbabe9; do ssh -n -o BatchMode=yes $h "mkdir -p ~/$M"; done
RSH="ssh -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -c aes128-gcm@openssh.com"
sync_to() {  # $1 host, $2 bwlimit KB/s (0 = none), rest = extra rsync args
  local h=$1 bw=$2; shift 2
  ssh -A -n -o BatchMode=yes xbabe5 "cd ~/$M 2>/dev/null || exit 0; taskset -c 56-59 nice -n 15 ionice -c3 rsync -a --exclude='*.part' --exclude='_buckets/' --exclude='.sumpart.*' --bwlimit=$bw $* -e '$RSH' --rsync-path='taskset -c ${SPARE[$h]} nice -n 15 ionice -c3 rsync' ./ ubuntu@${LAN[$h]}:$M/" 2>&1 | tail -3
}
log "pre-copy started"
prev=""
while :; do
  S=$(ssh -n -o BatchMode=yes xbabe5 "cd ~/$M 2>/dev/null && { ls | grep -c '^[0-9]*\.parquet\$'; [ -s SHA256SUMS ] && { wc -l < SHA256SUMS; sha256sum SHA256SUMS | cut -c1-64; } || echo '0 none'; }" 2>/dev/null | tr '\n' ' ')
  read -r NPQ NSUM SUM <<< "$S"
  if [ -n "${NPQ:-}" ] && [ "${NPQ:-0}" -gt 0 ]; then
    for h in xbabe7 xbabe8 xbabe9; do sync_to $h 700000 --exclude=SHA256SUMS --exclude=FILTER_REPORT.json & done; wait
    n7=$(ssh -n -o BatchMode=yes xbabe7 "ls ~/$M | grep -c '^[0-9]*\.parquet\$'")
    log "x5 has $NPQ final shards (manifest lines ${NSUM:-0}); x7 has $n7"
  fi
  cur="$NPQ $NSUM $SUM"
  if [ "${SUM:-none}" != none ] && [ "$NPQ" = "$NSUM" ] && [ "$cur" = "$prev" ]; then break; fi
  prev=$cur
  sleep 90
done
log "SEALED on x5: SHA256SUMS $SUM, $NPQ shards; final sync"
for h in xbabe7 xbabe8 xbabe9; do sync_to $h 0 --delete & done; wait
for h in xbabe5 xbabe7 xbabe8 xbabe9; do
  ( out=$(ssh -n -o BatchMode=yes $h "cd ~/$M && [ \"\$(sha256sum SHA256SUMS | cut -c1-64)\" = $SUM ] || { echo 'MANIFEST DIFFERS'; exit 1; }
      diff <(ls | grep '\.parquet\$' | sort) <(awk '{print \$2}' SHA256SUMS | sort) > /dev/null || { echo 'FILE SET DIFFERS'; exit 1; }
      split -n l/${NP[$h]} SHA256SUMS /dev/shm/tl_medsum.; fail=0
      for f in /dev/shm/tl_medsum.*; do taskset -c ${SPARE[$h]} nice -n 10 ionice -c3 sha256sum -c --quiet \$f > \$f.out 2>&1 & done; wait
      for f in /dev/shm/tl_medsum.*.out; do [ -s \$f ] && { cat \$f; fail=1; }; done; rm -f /dev/shm/tl_medsum.*
      [ \$fail = 0 ] && echo \"OK \$(ls *.parquet | wc -l) shards \$(du -sh . | cut -f1)\"" 2>&1 | tail -3); log "$h verify: $out" ) &
done; wait
log "MEDIUM STAGED $SUM $NPQ (see the four verify lines above)"
