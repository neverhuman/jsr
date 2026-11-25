#!/usr/bin/env bash
# Write dist/SHA256SUMS over every release asset in dist/ (signing is done
# separately with the release key: ssh-keygen -Y sign -n file).
set -Eeuo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
cd "${DIST_DIR:-$root/dist}"
find . -maxdepth 1 -type f ! -name 'SHA256SUMS*' ! -name '*.sig' ! -name '.*' |
	sed 's|^\./||' | LC_ALL=C sort | xargs sha256sum >SHA256SUMS
cat SHA256SUMS
