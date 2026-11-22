#!/usr/bin/env bash
# usage: bootstrap_uv.sh OPPONENTS_ROOT ASSET   (ASSET = uv-x86_64-unknown-linux-gnu | uv-aarch64-apple-darwin)
set -euo pipefail
O=$1; asset=$2; ver=0.13.9
dl="$O/bench-cross/downloads/uv-$ver"; mkdir -p "$dl" "$O/bench-cross/bin"
cd "$dl"
if [[ ! -x "$O/bench-cross/bin/uv" ]]; then
  curl -sSL --fail -o "$asset.tar.gz" "https://github.com/astral-sh/uv/releases/download/$ver/$asset.tar.gz"
  curl -sSL --fail -o "$asset.tar.gz.sha256" "https://github.com/astral-sh/uv/releases/download/$ver/$asset.tar.gz.sha256"
  want=$(awk '{print $1}' "$asset.tar.gz.sha256")
  if command -v sha256sum >/dev/null; then got=$(sha256sum "$asset.tar.gz" | awk '{print $1}'); else got=$(shasum -a 256 "$asset.tar.gz" | awk '{print $1}'); fi
  [[ "$want" == "$got" ]] || { echo "uv hash mismatch $want $got" >&2; exit 1; }
  tar -xzf "$asset.tar.gz"
  cp "$asset/uv" "$O/bench-cross/bin/uv"
  echo "$got  $asset.tar.gz (uv $ver, https://github.com/astral-sh/uv/releases/download/$ver/$asset.tar.gz)" > "$O/bench-cross/bin/uv.provenance"
fi
"$O/bench-cross/bin/uv" --version
cat "$O/bench-cross/bin/uv.provenance"
