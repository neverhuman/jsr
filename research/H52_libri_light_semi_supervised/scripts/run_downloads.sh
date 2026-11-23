#!/usr/bin/env bash
# H52: small then medium, sequentially. Each has a follow_hash.py follower started separately.
set -uo pipefail
B=/mnt/xbabe3-external/jsr/librilight
$B/bin/download.sh https://dl.fbaipublicfiles.com/librilight/data/small.tar  $B/dl/small.tar  35135006720  40M
$B/bin/download.sh https://dl.fbaipublicfiles.com/librilight/data/medium.tar $B/dl/medium.tar 320792217600 40M
