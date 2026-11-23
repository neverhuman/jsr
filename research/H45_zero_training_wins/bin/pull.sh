#!/usr/bin/env bash
# h45-wins: stream one checkpoint from its source host to xbabe1's data disk through H0's pipe (nothing written on H0),
# then verify it against the source's own .sha256 marker and write the same marker beside the copy.
#   pull.sh <src_host|local> <src_path.pt> <dst_name.pt>
set -uo pipefail
SRC_HOST=$1 SRC=$2 NAME=$3
DST=/mnt/amphora-backup/h45-wins/weights/src/$NAME
M=${SRC%.pt}.sha256
if [ "$SRC_HOST" = local ]; then want=$(cut -d' ' -f1 < "$M"); else want=$(ssh -o BatchMode=yes "$SRC_HOST" "cut -d' ' -f1 < $M" </dev/null); fi
[ -n "$want" ] || { echo "$(date -u +%FT%TZ) NO-MARKER $SRC_HOST:$SRC"; exit 2; }
have=$(ssh -o BatchMode=yes xbabe1 "cat ${DST%.pt}.sha256 2>/dev/null" </dev/null | cut -d' ' -f1)
if [ "$have" = "$want" ]; then echo "$(date -u +%FT%TZ) SKIP $NAME (already verified $want)"; exit 0; fi
t0=$(date +%s)
if [ "$SRC_HOST" = local ]; then nice -n 19 cat "$SRC"; else ssh -o BatchMode=yes "$SRC_HOST" "nice -n 19 cat $SRC" </dev/null; fi \
  | ssh -o BatchMode=yes xbabe1 "cat > $DST.tmp"
got=$(ssh -o BatchMode=yes xbabe1 "sha256sum $DST.tmp" </dev/null | cut -d' ' -f1)
if [ "$got" != "$want" ]; then echo "$(date -u +%FT%TZ) HASH-MISMATCH $NAME got=$got want=$want"; ssh -o BatchMode=yes xbabe1 "rm -f $DST.tmp" </dev/null; exit 3; fi
ssh -o BatchMode=yes xbabe1 "mv $DST.tmp $DST && echo $want > ${DST%.pt}.sha256" </dev/null
echo "$(date -u +%FT%TZ) OK $NAME $want $SRC_HOST:$SRC $(( $(date +%s) - t0 ))s"
