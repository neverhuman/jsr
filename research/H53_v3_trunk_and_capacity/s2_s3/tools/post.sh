#!/usr/bin/env bash
# trainer-lane chat post: post.sh "<title>" < body.md   (body via a QUOTED heredoc at the call site: <<'B')
# Appends "## <UTC now> | trainer-lane | <title>" + body to AGENT_CHAT.md under the file's flock.
set -u
CHAT=/home/ubuntu/jsr/AGENT_CHAT.md
body=$(cat)
{ flock 9; printf '\n## %s | trainer-lane | %s\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" "$body" >&9; } 9>>"$CHAT"
echo "posted: $1"
