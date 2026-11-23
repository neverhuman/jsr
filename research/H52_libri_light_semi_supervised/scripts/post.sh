#!/usr/bin/env bash
# Append one post to the shared AGENT_CHAT.md under its lock (append-only, never rewrite).
# usage: post.sh "<title>" <body-file>
set -euo pipefail
title="$1"; body="$2"
chat=/home/ubuntu/jsr/AGENT_CHAT.md; lock=/home/ubuntu/jsr/AGENT_CHAT.lock
ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
{
  printf '\n## %s | h52-librilight | %s\n' "$ts" "$title"
  cat "$body"
} > /mnt/xbabe3-external/jsr/librilight/logs/post.$$.tmp
flock "$lock" -c "cat /mnt/xbabe3-external/jsr/librilight/logs/post.$$.tmp >> $chat"
cat /mnt/xbabe3-external/jsr/librilight/logs/post.$$.tmp >> /mnt/xbabe3-external/jsr/librilight/logs/posts.md
rm -f /mnt/xbabe3-external/jsr/librilight/logs/post.$$.tmp
echo "posted $ts"
