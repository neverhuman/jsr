#!/usr/bin/env bash
# Cross-site DiLoCo relay (proposal; not started by anyone yet). The hub listens on H0 at 127.0.0.1:$PORT only.
# H0 opens one reverse tunnel per member host, so each member reaches the hub at ITS OWN 127.0.0.1:$PORT:
#   - the cloud firewall drops inbound except :22, and the site is behind NAT, so only outbound ssh from H0 works;
#   - no port is exposed anywhere; ssh keys authenticate the hosts, the hub's HMAC token authenticates members;
#   - endpoints are IPs (on-prem /etc/hosts maps xbabe4-7 to stale instances; xbabe8/9 and the cloud resolve nothing).
# Usage (on H0): diloco_tunnels.sh PORT user@ip [user@ip ...]   e.g. 29610 ubuntu@<xbabe5-public> ubuntu@<xbabe2-lan>
# Each tunnel runs in a supervising loop (no autossh on the fleet); PIDs go to $STATE/tunnels.pids; stop by those PIDs.
set -euo pipefail
PORT=$1; shift
STATE=/mnt/xbabe3-external/jsr/h50-fleet/run/tunnels; mkdir -p "$STATE"
for dst in "$@"; do
  (
    while true; do
      ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
          -o BatchMode=yes -R "127.0.0.1:$PORT:127.0.0.1:$PORT" "$dst"
      echo "$(date -u +%FT%TZ) tunnel to $dst exited ($?), restarting in 5 s" >> "$STATE/tunnels.log"
      sleep 5
    done
  ) &
  echo "$! $dst" >> "$STATE/tunnels.pids"
done
echo "supervisors: $(cat "$STATE/tunnels.pids" | tr '\n' ';')"
