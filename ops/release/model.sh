#!/usr/bin/env bash
# Build the platform-independent model archive for `isosyll`.
#
#   ops/release/model.sh LIVE_BUNDLE.tar LIVE.ulm FINAL.ulm FINAL_BUNDLE T2.utfr CONTENT_FIT.json CONTENT_LEXICON.json
#
# LIVE_BUNDLE.tar is the live INT8 bundle (arch.json, weights.int8.safetensors, tokens.txt, mel_fbank.f32,
# bpe.model; v4.1.0/v4.2.0: H247 step 450). LIVE.ulm is the 160 ms live LM (compact M5.v2), FINAL.ulm the
# commit-pass LM (M5). v4.2.0 adds the final pass: FINAL_BUNDLE (a .tar or a directory with the same five
# files; v4.2.0: the F3 0.30/0.55/0.15 soup), T2.utfr (the left-to-right transformer rescorer) and the frozen
# content-word reranker (fit + lexicon). Writes dist/isosyll-v<version>-model.tar.gz holding isosyll-model/
# with a MANIFEST of sha256 sums that `isosyll model verify` checks.
set -Eeuo pipefail
[ $# -eq 7 ] || { sed -n '3,11p' "$0" >&2; exit 2; }
root=$(cd "$(dirname "$0")/../.." && pwd)
version=$(sed -n 's/^version = "\(.*\)"/\1/p' "$root/crates/isosyll/Cargo.toml" | head -n 1)
out=${DIST_DIR:-$root/dist}
mkdir -p "$out"
stage=$(mktemp -d) # local fs, so file modes are exact
trap 'rm -rf "$stage"' EXIT
m="$stage/isosyll-model"
mkdir -p "$m/lm" "$m/final" "$stage/x" "$stage/f"
bundle() { # bundle SRC SCRATCH DEST: copy the five bundle files from a .tar or a directory
	local src=$1 x=$2 dir
	if [ -d "$src" ]; then x=$src; else tar -xf "$src" -C "$x"; fi
	dir=$(dirname "$(find "$x" -name arch.json | head -n 1)")
	for f in arch.json weights.int8.safetensors tokens.txt mel_fbank.f32 bpe.model; do
		[ -f "$dir/$f" ] || { echo "error: bundle $src lacks $f" >&2; exit 1; }
		cp "$dir/$f" "$3/$f"
	done
}
bundle "$1" "$stage/x" "$m"
bundle "$4" "$stage/f" "$m/final"
cp "$2" "$m/lm/live.ulm"
cp "$3" "$m/lm/final.ulm"
cp "$5" "$m/lm/t2.utfr"
cp "$6" "$m/final/content-fit.json"
cp "$7" "$m/final/content-lexicon.json"
(cd "$m" && find . -type f ! -name MANIFEST | sed 's|^\./||' | LC_ALL=C sort | xargs sha256sum >../MANIFEST.tmp && mv ../MANIFEST.tmp MANIFEST)
find "$m" -type d -exec chmod 755 {} + && find "$m" -type f -exec chmod 644 {} +
tar --sort=name --owner=0 --group=0 --numeric-owner --mtime='@0' \
	-C "$stage" -cf - isosyll-model | gzip -n -9 >"$out/isosyll-v$version-model.tar.gz"
echo "$out/isosyll-v$version-model.tar.gz"
cat "$m/MANIFEST"
