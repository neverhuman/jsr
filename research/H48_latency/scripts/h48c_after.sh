#!/usr/bin/env bash
# x9: start H48c-<arm> once the GPU lease is free (after H48b s4321's end decodes), on the pinned H48c tree.
#   h48c_after.sh <a|b> <dp>
set -u
ARM=$1 DP=$2; W=$HOME/jsr/.agent/worktrees/trainer-h48c; D=$HOME/jsr-data
WANT=9c178bb5a98f0f16b88e9de94ded09b7fe33897ce11fcd558d9712dad91691a8
until flock --nonblock "$D/gpu-leases/gpu-0.lock" true; do sleep 10; done
GOT=$(cd "$W" && git ls-files -co --exclude-standard python/ultra | grep -E '\.py$' | LC_ALL=C sort | xargs sha256sum | sha256sum | cut -c1-64)
[ "$GOT" = "$WANT" ] || { echo "REFUSED: tree $GOT"; exit 1; }
[ "$(sha256sum "$D/h48c/mfa_anchors_train960_v1.npz" | cut -c1-64)" = 58a8bd0f65231138cb0f10cd62d1b59a28c56c6247732ae4b59f570f13e60b2b ] || { echo "REFUSED: side file"; exit 1; }
[ "$(sha256sum "$D/seeds/A1/ckpt-00049000.pt" | cut -c1-64)" = 2a3dd9fb6c7856126b3455fb65babc4295e27b10e011d18a8806b8be5181e146 ] || { echo "REFUSED: B0"; exit 1; }
echo "$(date -u +%T) launching H48c-$ARM (dp $DP) on tree $GOT"
INIT_FROM=$D/seeds/A1/ckpt-00049000.pt GPU_TARGET=0.87 PY=$D/venvs/train/bin/python bash "$HOME/h48c_launch.sh" "$D" 0-27 0 "H48c_${ARM}_s1234" \
  chunk_modes=4,8,16 eval_chunks=8,16 delay_penalty="$DP" mfa_anchors="$D/h48c/mfa_anchors_train960_v1.npz" \
  anchor_kappa=0.01 anchor_margin=1 anchor_cap=12 anchor_simple=true || exit 1
sleep 5
setsid nohup taskset -c 0-27 bash "$HOME/after_end_fulldev.sh" "$D/runs/train/H48c_${ARM}_s1234" 0 0-27 > "$D/logs/fulldev_H48c_${ARM}_s1234.log" 2>&1 < /dev/null &
echo "$(date -u +%T) launched; end decodes armed"
