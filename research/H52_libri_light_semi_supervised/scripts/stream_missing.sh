#!/usr/bin/env bash
# H52: stream the shards listed in x5's SHA256SUMS that are not yet verified on x7. Worker W of N.
set -u
SUB=$1; N=$2; W=$3
SD=jsr-data/librilight/shards/r0/librilight.$SUB
DONE=/dev/shm/h52/stream_$SUB.done
C="-o BatchMode=yes -c aes128-gcm@openssh.com -o Compression=no"
log() { echo "$(date -u +%H:%M:%SZ) [m$W] $*"; }
list=$(timeout 60 ssh $C xbabe5 "cat ~/$SD/SHA256SUMS") || { log "cannot read SHA256SUMS"; exit 2; }
i=0
echo "$list" | while read -r hs f; do
  [ -z "$f" ] && continue
  i=$((i+1)); [ $(( (i-1) % N )) -eq $W ] || continue
  grep -qx "$f" $DONE && continue
  for attempt in 1 2 3; do
    timeout 1800 ssh -n $C xbabe5 "cat ~/$SD/$f" | timeout 1800 ssh $C xbabe7 "cat > ~/$SD/$f.part"
    hd=$(timeout 300 ssh -n $C xbabe7 "sha256sum ~/$SD/$f.part" | cut -d' ' -f1)
    if [ "$hs" = "$hd" ]; then
      ssh -n $C xbabe7 "mv ~/$SD/$f.part ~/$SD/$f" && echo "$f" >> $DONE && log "streamed $f ($hs)"; break
    fi
    log "MISMATCH $f (attempt $attempt)"; ssh -n $C xbabe7 "rm -f ~/$SD/$f.part"
  done
done
log "share done"
