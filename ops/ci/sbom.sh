#!/usr/bin/env bash
# CycloneDX SBOM per workspace crate, for release provenance (docs/release.md).
# Writes target/sbom/<crate>.cdx.json plus SHA256SUMS; nothing is left in crates/.
set -Eeuo pipefail
# shellcheck source=ops/ci/lib.sh
. "$(dirname "$0")/lib.sh"
cd "$(ci_repo_root)"
ci_lane sbom "just sbom"
ci_require_tool cargo-cyclonedx "cargo install --locked cargo-cyclonedx --version $JSR_CARGO_CYCLONEDX_VERSION"
out=target/sbom
rm -rf "$out"
mkdir -p "$out"
ci_step "cargo cyclonedx"
cargo cyclonedx --format json --spec-version 1.5 --override-filename jsr-sbom -q
for bom in crates/*/jsr-sbom.json; do
  crate=$(basename "$(dirname "$bom")")
  mv "$bom" "$out/$crate.cdx.json"
done
(cd "$out" && sha256sum ./*.cdx.json > SHA256SUMS)
ci_step "wrote $(find "$out" -name '*.cdx.json' | wc -l) SBOMs to $out"
