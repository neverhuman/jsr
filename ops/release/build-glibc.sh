#!/usr/bin/env bash
# Build the x86-64 Linux `isosyll` against glibc 2.35 inside ubuntu:22.04 and archive it as
# dist/isosyll-v<version>-x86_64-linux.tar.gz. This is the timed release build: it links only
# libc, libm and libgcc_s, like the v3.x decoders. (The static musl build is ~2-4% slower on the
# beam-16 commit pass, an allocator effect, so it ships only as the -musl fallback.)
#
#   ops/release/build-glibc.sh          # needs docker and the pinned toolchain under ~/.rustup
set -Eeuo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
cd "$root"
toolchain=$(sed -n 's/^channel *= *"\(.*\)"/\1/p' rust-toolchain.toml)
tc="$HOME/.rustup/toolchains/$toolchain-x86_64-unknown-linux-gnu"
[ -x "$tc/bin/cargo" ] || { echo "error: toolchain $toolchain not installed at $tc" >&2; exit 1; }
target=${GLIBC_TARGET_DIR:-$root/target/release-glibc}
mkdir -p "$root/target"
src=$(mktemp -d "$root/target/glibc-src.XXXXXX") # not /tmp: build inputs stay in the lane
trap 'rm -rf "$src"' EXIT
mkdir -p "$target"
git archive HEAD | tar -x -C "$src"
commit=$(git rev-parse --short=12 HEAD)
docker run --rm --network host -e OWNER="$(id -u):$(id -g)" \
	-e PATH="/rh/toolchains/$toolchain-x86_64-unknown-linux-gnu/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
	-e RUSTUP_HOME=/rh -e CARGO_HOME=/ch -e CARGO_TARGET_DIR=/tgt \
	-e ISOSYLL_BUILD_COMMIT="$commit" \
	-e CARGO_PROFILE_RELEASE_STRIP=symbols -e CARGO_PROFILE_RELEASE_DEBUG=false \
	-e RUSTFLAGS="--remap-path-prefix=/src=. --remap-path-prefix=/ch=cargo" \
	--tmpfs /ch:exec \
	-v "$HOME/.cargo/registry:/ch-registry:ro" -v "$HOME/.rustup:/rh:ro" \
	-v "$src:/src" -v "$target:/tgt" -w /src ubuntu:22.04 sh -c '
		set -e
		apt-get -qq update >/dev/null && apt-get -qq install -y gcc >/dev/null
		ln -s /ch-registry /ch/registry
		cargo build --release --locked --offline -p isosyll
		max=$(objdump -T /tgt/release/isosyll | grep -o "GLIBC_[0-9.]*" | sort -Vu | tail -n 1)
		[ "$(printf "%s\nGLIBC_2.35\n" "$max" | sort -V | tail -n 1)" = GLIBC_2.35 ] ||
			{ echo "error: binary needs $max" >&2; exit 1; }
		chown -R "$OWNER" /tgt # the container runs as root; hand the target dir back'
bash ops/release/build.sh --package x86_64-unknown-linux-gnu "$target/release/isosyll"
