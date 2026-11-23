#!/usr/bin/env bash
# H52: stream small's final shards x5 -> H0 -> x7 as each is finalised; sha256 checked on both ends.
set -u
SUB=${1:-small}
SD=jsr-data/librilight/shards/r0/librilight.$SUB
DONE=/dev/shm/h52/stream_$SUB.done; touch $DONE
log() { echo "$(date -u +%H:%M:%SZ) $*"; }
ssh -o BatchMode=yes xbabe7 "mkdir -p ~/$SD"
while :; do
  files=$(timeout 60 ssh -o BatchMode=yes xbabe5 "cd ~/$SD 2>/dev/null && ls *.parquet 2>/dev/null")
  for f in $files; do
    grep -qx "$f" $DONE && continue
    hs=$(timeout 300 ssh -o BatchMode=yes xbabe5 "sha256sum ~/$SD/$f" | cut -d' ' -f1)
    timeout 900 ssh -o BatchMode=yes xbabe5 "cat ~/$SD/$f" | timeout 900 ssh -o BatchMode=yes xbabe7 "cat > ~/$SD/$f.part"
    hd=$(timeout 300 ssh -o BatchMode=yes xbabe7 "sha256sum ~/$SD/$f.part" | cut -d' ' -f1)
    if [ -n "$hs" ] && [ "$hs" = "$hd" ]; then
      ssh -o BatchMode=yes xbabe7 "mv ~/$SD/$f.part ~/$SD/$f" && echo "$f" >> $DONE && log "streamed $f ($hs)"
    else
      log "MISMATCH $f src=$hs dst=$hd; retrying"; ssh -o BatchMode=yes xbabe7 "rm -f ~/$SD/$f.part"
    fi
  done
  if timeout 60 ssh -o BatchMode=yes xbabe5 "test -f ~/$SD/FILTER_REPORT.json && test -f ~/$SD/SHA256SUMS"; then
    n=$(timeout 60 ssh -o BatchMode=yes xbabe5 "wc -l < ~/$SD/SHA256SUMS")
    if [ "$(wc -l < $DONE)" -ge "$n" ]; then
      for x in SHA256SUMS FILTER_REPORT.json; do
        timeout 120 ssh -o BatchMode=yes xbabe5 "cat ~/$SD/$x" | ssh -o BatchMode=yes xbabe7 "cat > ~/$SD/$x"
      done
      if ssh -o BatchMode=yes xbabe7 "cd ~/$SD && sha256sum -c --quiet SHA256SUMS"; then
        log "ALL $n shards on x7 verified; manifest SHA256SUMS sha256 $(ssh -o BatchMode=yes xbabe7 "sha256sum ~/$SD/SHA256SUMS" | cut -d' ' -f1)"
        exit 0
      else
        log "FINAL VERIFY FAILED on x7"; exit 3
      fi
    fi
  fi
  sleep 15
done
