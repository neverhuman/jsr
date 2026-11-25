#!/usr/bin/env bash
# Build the platform-independent model archive for `isosyll`.
#
#   ops/release/model.sh INT8_BUNDLE.tar LIVE.ulm FINAL.ulm
#
# INT8_BUNDLE.tar is the signed RC2 bundle (int8/arch.json, weights.int8.safetensors,
# tokens.txt, mel_fbank.f32, bpe.model). LIVE.ulm is the 160 ms live LM
# (c4_p0_0_10_40.v2) and FINAL.ulm the commit-pass LM (M5). Writes
# dist/isosyll-v<version>-model.tar.gz holding isosyll-model/ with a MANIFEST of
# sha256 sums that `isosyll model verify` checks.
set -Eeuo pipefail
[ $# -eq 3 ] || { sed -n '3,10p' "$0" >&2; exit 2; }
root=$(cd "$(dirname "$0")/../.." && pwd)
version=$(sed -n 's/^version = "\(.*\)"/\1/p' "$root/crates/isosyll/Cargo.toml" | head -n 1)
out=${DIST_DIR:-$root/dist}
mkdir -p "$out"
stage=$(mktemp -d) # local fs, so file modes are exact
trap 'rm -rf "$stage"' EXIT
m="$stage/isosyll-model"
mkdir -p "$m/lm" "$stage/x"
tar -xf "$1" -C "$stage/x"
src=$(dirname "$(find "$stage/x" -name arch.json | head -n 1)")
for f in arch.json weights.int8.safetensors tokens.txt mel_fbank.f32 bpe.model; do
	[ -f "$src/$f" ] || { echo "error: bundle lacks $f" >&2; exit 1; }
	cp "$src/$f" "$m/$f"
done
cp "$2" "$m/lm/live.ulm"
cp "$3" "$m/lm/final.ulm"
(cd "$m" && find . -type f ! -name MANIFEST | sed 's|^\./||' | LC_ALL=C sort | xargs sha256sum >../MANIFEST.tmp && mv ../MANIFEST.tmp MANIFEST)
find "$m" -type d -exec chmod 755 {} + && find "$m" -type f -exec chmod 644 {} +
tar --sort=name --owner=0 --group=0 --numeric-owner --mtime='@0' \
	-C "$stage" -cf - isosyll-model | gzip -n -9 >"$out/isosyll-v$version-model.tar.gz"
echo "$out/isosyll-v$version-model.tar.gz"
cat "$m/MANIFEST"
