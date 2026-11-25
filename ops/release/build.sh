#!/usr/bin/env bash
# Build release archives of the `isosyll` binary.
#
#   ops/release/build.sh [TRIPLE...]          # default: the host's release triple
#   ops/release/build.sh x86_64-unknown-linux-musl aarch64-unknown-linux-musl
#   ops/release/build-glibc.sh                # x86-64 glibc 2.35 build in ubuntu:22.04 (the timed asset)
#   ops/release/build.sh aarch64-apple-darwin  # run on a macOS arm64 host
#   ops/release/build.sh --package aarch64-apple-darwin path/to/isosyll
#                                             # archive a binary built elsewhere
#
# Each triple produces dist/isosyll-v<version>-<name>.tar.gz holding
# isosyll-v<version>-<name>/{bin/isosyll,LICENSE,README.md}. Binaries are
# stripped, built with --locked on the pinned toolchain, and carry no host paths.
# The model archive is built separately by ops/release/model.sh.
set -Eeuo pipefail

root=$(cd "$(dirname "$0")/../.." && pwd)
cd "$root"
version=$(sed -n 's/^version = "\(.*\)"/\1/p' crates/isosyll/Cargo.toml | head -n 1)
[ -n "$version" ] || { echo "error: no version in crates/isosyll/Cargo.toml" >&2; exit 1; }
out=${DIST_DIR:-$root/dist}
mkdir -p "$out"

name_of() {
	case "$1" in
	x86_64-unknown-linux-gnu) echo x86_64-linux ;; # glibc >= 2.35, the timed build
	x86_64-unknown-linux-musl) echo x86_64-linux-musl ;; # static fallback
	aarch64-unknown-linux-musl | aarch64-unknown-linux-gnu) echo aarch64-linux ;;
	aarch64-apple-darwin) echo aarch64-macos ;;
	*) echo "error: unsupported triple $1" >&2; return 1 ;;
	esac
}

package() { # triple binary
	local name pkg stage
	name=$(name_of "$1")
	pkg="isosyll-v$version-$name"
	stage=$(mktemp -d) # local fs, so file modes are exact
	mkdir -p "$stage/$pkg/bin"
	install -m 755 "$2" "$stage/$pkg/bin/isosyll"
	cp LICENSE "$stage/$pkg/LICENSE"
	cp ops/release/PACKAGE_README.md "$stage/$pkg/README.md"
	chmod 644 "$stage/$pkg/LICENSE" "$stage/$pkg/README.md"
	# Deterministic archive: fixed order, owner and mtime (GNU tar).
	tar --sort=name --owner=0 --group=0 --numeric-owner --mtime='@0' \
		-C "$stage" -cf - "$pkg" | gzip -n -9 >"$out/$pkg.tar.gz"
	rm -rf "$stage"
	echo "    $out/$pkg.tar.gz"
}

if [ "${1:-}" = --package ]; then
	[ $# -eq 3 ] || { echo "usage: build.sh --package TRIPLE BINARY" >&2; exit 2; }
	package "$2" "$3"
	exit 0
fi

if [ $# -eq 0 ]; then
	case "$(uname -s)/$(uname -m)" in
	Linux/x86_64) set -- x86_64-unknown-linux-musl ;;
	Linux/aarch64) set -- aarch64-unknown-linux-musl ;;
	Darwin/arm64) set -- aarch64-apple-darwin ;;
	*) echo "error: pass a target triple" >&2; exit 1 ;;
	esac
fi

export CARGO_PROFILE_RELEASE_STRIP=symbols
export CARGO_PROFILE_RELEASE_DEBUG=false
export RUSTFLAGS="${RUSTFLAGS:-} --remap-path-prefix=$root=. --remap-path-prefix=${CARGO_HOME:-$HOME/.cargo}=cargo"
export ISOSYLL_BUILD_COMMIT=${ISOSYLL_BUILD_COMMIT:-$(git rev-parse --short=12 HEAD 2>/dev/null || echo unknown)}

# Resolve the target directory Cargo will actually use (.cargo/config.toml may set one).
target_dir=$(cargo metadata --format-version 1 --no-deps --offline 2>/dev/null |
	sed -n 's/.*"target_directory":"\([^"]*\)".*/\1/p')
[ -n "$target_dir" ] || { echo "error: cannot resolve cargo target directory" >&2; exit 1; }
export CARGO_TARGET_DIR=$target_dir

for triple in "$@"; do
	name_of "$triple" >/dev/null
	case "$triple" in
	*-linux-musl)
		# Pure-Rust static binary; rust-lld links the bundled musl, no C cross-compiler needed.
		upper=$(echo "$triple" | tr 'a-z-' 'A-Z_')
		export "CARGO_TARGET_${upper}_LINKER=rust-lld"
		;;
	esac
	echo "==> building isosyll $version for $triple"
	cargo build --release --locked -p isosyll --target "$triple"
	bin="$target_dir/$triple/release/isosyll"
	if [ "$(uname -s)" = Darwin ]; then
		echo "    built $bin (package it on Linux with --package)"
	else
		package "$triple" "$bin"
	fi
done
