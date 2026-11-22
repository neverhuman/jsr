#!/usr/bin/env bash
# H0: start a sync_and_post log mirror for every run dir on <host> matching <glob> (e.g. 'H23*'),
# re-checked every 5 minutes, so new arms are mirrored to H0 (dashboard, research/curves).
#   mirror_pattern.sh <host> <glob>
set -uo pipefail
H=$1 PAT=$2
T=/mnt/xbabe3-external/jsr/runs/train
while :; do
  for d in $(ssh -n "$H" "cd ~/jsr-data/runs/train && ls -d $PAT 2>/dev/null" ); do
    [ -L "$T/$d" ] && continue
    case "$d" in *_current_*) continue;; esac
    # finished/stopped arms: mirrored once (.final), restarted only if the trainer runs again
    if [ -f "$T/$d/.final" ] && ! ssh -n "$H" "kill -0 \$(cat ~/jsr-data/runs/train/$d/PID) 2>/dev/null"; then
      continue
    fi
    if ! pgrep -f "sync_and_post.py $H /home/ubuntu/jsr-data/runs/train/$d " > /dev/null; then
      mkdir -p "$T/$d"
      setsid nohup nice -n 10 python3 "$T/sync_and_post.py" "$H" "/home/ubuntu/jsr-data/runs/train/$d" "$T/$d" "$d" \
        > "/mnt/xbabe3-external/jsr/logs/trainer/sync_$d.log" 2>&1 < /dev/null &
      echo "$(date -u +%FT%TZ) mirroring $H:$d"
    fi
  done
  sleep 300
done
