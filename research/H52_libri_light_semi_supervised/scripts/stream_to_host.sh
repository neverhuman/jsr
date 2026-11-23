#!/usr/bin/env bash
# H52: copy a finished shard directory x5 -> H0 -> <dest> (worker W of N, by SHA256SUMS line), sha256 both ends;
# worker 0 then waits for all and runs sha256sum -c on <dest>.  usage: stream_to_host.sh <reldir> <dest> <N> <W>
set -u
SD=$1; DST=$2; N=$3; W=$4
DONE=/dev/shm/h52/stream_$(echo "$SD-$DST" | tr '/' '_').done; touch $DONE
C="-o BatchMode=yes -c aes128-gcm@openssh.com -o Compression=no"
log() { echo "$(date -u +%H:%M:%SZ) [$DST w$W] $*"; }
ssh -n $C $DST "mkdir -p ~/$SD"
list=$(timeout 60 ssh -n $C xbabe5 "cat ~/$SD/SHA256SUMS") || { log "cannot read SHA256SUMS"; exit 2; }
total=$(echo "$list" | grep -c .)
i=0
while read -r hs f; do
  [ -z "$f" ] && continue
  i=$((i+1)); [ $(( (i-1) % N )) -eq $W ] || continue
  grep -qx "$f" $DONE && continue
  for attempt in 1 2 3; do
    timeout 1800 ssh -n $C xbabe5 "cat ~/$SD/$f" | timeout 1800 ssh $C $DST "cat > ~/$SD/$f.part"
    hd=$(timeout 300 ssh -n $C $DST "sha256sum ~/$SD/$f.part" | cut -d' ' -f1)
    if [ "$hs" = "$hd" ]; then
      ssh -n $C $DST "mv ~/$SD/$f.part ~/$SD/$f" && echo "$f" >> $DONE && log "copied $f"; break
    fi
    log "MISMATCH $f (attempt $attempt)"; ssh -n $C $DST "rm -f ~/$SD/$f.part"
  done
done <<< "$list"
if [ "$W" -eq 0 ]; then
  while [ "$(sort -u $DONE | wc -l)" -lt "$total" ]; do sleep 10; done
  for x in SHA256SUMS FILTER_REPORT.json; do timeout 120 ssh -n $C xbabe5 "cat ~/$SD/$x" | ssh $C $DST "cat > ~/$SD/$x"; done
  if ssh -n $C $DST "cd ~/$SD && sha256sum -c --quiet SHA256SUMS"; then
    log "ALL $total shards verified on $DST; manifest sha256 $(ssh -n $C $DST "sha256sum ~/$SD/SHA256SUMS" | cut -d' ' -f1)"
  else
    log "FINAL VERIFY FAILED on $DST"
  fi
fi
