#!/usr/bin/env bash
# H52: parallel x5 -> H0 -> x7 streaming of final shards. Worker W of N takes shards with number % N == W.
# usage: stream_par.sh <subset> <N> <W>
set -u
SUB=$1; N=$2; W=$3
SD=jsr-data/librilight/shards/r0/librilight.$SUB
DONE=/dev/shm/h52/stream_$SUB.done; touch $DONE
C="-o BatchMode=yes -c aes128-gcm@openssh.com -o Compression=no"
log() { echo "$(date -u +%H:%M:%SZ) [w$W] $*"; }
while :; do
  files=$(timeout 60 ssh $C xbabe5 "cd ~/$SD 2>/dev/null && ls *.parquet 2>/dev/null")
  for f in $files; do
    num=$((10#${f%%.*}))
    [ $((num % N)) -eq $W ] || continue
    grep -qx "$f" $DONE && continue
    hs=$(timeout 300 ssh $C xbabe5 "sha256sum ~/$SD/$f" | cut -d' ' -f1)
    timeout 1800 ssh $C xbabe5 "cat ~/$SD/$f" | timeout 1800 ssh $C xbabe7 "cat > ~/$SD/$f.part"
    hd=$(timeout 300 ssh $C xbabe7 "sha256sum ~/$SD/$f.part" | cut -d' ' -f1)
    if [ -n "$hs" ] && [ "$hs" = "$hd" ]; then
      ssh $C xbabe7 "mv ~/$SD/$f.part ~/$SD/$f" && echo "$f" >> $DONE && log "streamed $f ($hs)"
    else
      log "MISMATCH $f src=$hs dst=$hd; retrying"; ssh $C xbabe7 "rm -f ~/$SD/$f.part"
    fi
  done
  if [ "$W" -eq 0 ] && timeout 60 ssh $C xbabe5 "test -f ~/$SD/FILTER_REPORT.json && test -f ~/$SD/SHA256SUMS"; then
    n=$(timeout 60 ssh $C xbabe5 "wc -l < ~/$SD/SHA256SUMS")
    if [ "$(sort -u $DONE | wc -l)" -ge "$n" ]; then
      for x in SHA256SUMS FILTER_REPORT.json; do timeout 120 ssh $C xbabe5 "cat ~/$SD/$x" | ssh $C xbabe7 "cat > ~/$SD/$x"; done
      if ssh $C xbabe7 "cd ~/$SD && sha256sum -c --quiet SHA256SUMS"; then
        log "ALL $n shards on x7 verified; manifest SHA256SUMS sha256 $(ssh $C xbabe7 "sha256sum ~/$SD/SHA256SUMS" | cut -d' ' -f1)"
        exit 0
      else
        log "FINAL VERIFY FAILED on x7"; exit 3
      fi
    fi
  fi
  [ "$W" -ne 0 ] && timeout 60 ssh $C xbabe5 "test -f ~/$SD/SHA256SUMS" && [ -z "$(for f in $files; do n2=$((10#${f%%.*})); [ $((n2 % N)) -eq $W ] && ! grep -qx "$f" $DONE && echo x; done)" ] && { log "my share done"; exit 0; }
  sleep 10
done
