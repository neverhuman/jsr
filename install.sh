#!/bin/sh
# Isosyll installer: downloads a signed release, verifies it, and installs
# the `isosyll` binary and its model under a prefix (default ~/.local).
#
#   curl -fsSL https://github.com/neverhuman/jsr/releases/latest/download/install.sh | sh
#   sh install.sh --prefix /opt/isosyll --version v4.0.0
#
# Nothing here needs root, Python, or a checkout of the repository.
set -eu

REPO="neverhuman/jsr"
VERSION="${ISOSYLL_VERSION:-v4.0.0}"
PREFIX="${ISOSYLL_PREFIX:-$HOME/.local}"
SIGNER="219957145+neverhumanbot@users.noreply.github.com"
SIGNER_KEY="ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIDD7oTCogsiEzaLS9nv6Q0n387XE+De+iqoN+VRKF91O"
VERIFY_SIG=1

say() { printf '%s\n' "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

usage() {
	cat <<EOF
Install Isosyll ($VERSION) from GitHub releases.

usage: install.sh [--prefix DIR] [--version TAG] [--no-verify-signature]

  --prefix DIR             install to DIR/bin and DIR/share/isosyll (default: ~/.local)
  --version TAG            release tag to install (default: $VERSION)
  --no-verify-signature    skip the SSH signature check (checksums are still verified);
                           needed only where ssh-keygen is not installed
EOF
}

while [ $# -gt 0 ]; do
	case "$1" in
	--prefix) [ $# -ge 2 ] || die "--prefix needs a directory"; PREFIX=$2; shift 2 ;;
	--prefix=*) PREFIX=${1#*=}; shift ;;
	--version) [ $# -ge 2 ] || die "--version needs a tag"; VERSION=$2; shift 2 ;;
	--version=*) VERSION=${1#*=}; shift ;;
	--no-verify-signature) VERIFY_SIG=0; shift ;;
	-h | --help) usage; exit 0 ;;
	*) usage >&2; die "unknown option: $1" ;;
	esac
done
[ -n "$PREFIX" ] || die "--prefix must not be empty"

os=$(uname -s)
arch=$(uname -m)
case "$os/$arch" in
Linux/x86_64 | Linux/amd64) target=x86_64-linux ;;
Linux/aarch64 | Linux/arm64) target=aarch64-linux ;;
Darwin/arm64 | Darwin/aarch64) target=aarch64-macos ;;
*) die "no prebuilt binary for $os/$arch (supported: Linux x86-64, Linux arm64, macOS Apple silicon)" ;;
esac

if command -v curl >/dev/null 2>&1; then
	fetch() { curl -fsSL --retry 3 -o "$2" "$1"; }
elif command -v wget >/dev/null 2>&1; then
	fetch() { wget -q -O "$2" "$1"; }
else
	die "curl or wget is required"
fi
if command -v sha256sum >/dev/null 2>&1; then
	sha256() { sha256sum "$1" | cut -d' ' -f1; }
elif command -v shasum >/dev/null 2>&1; then
	sha256() { shasum -a 256 "$1" | cut -d' ' -f1; }
else
	die "sha256sum or shasum is required"
fi
command -v tar >/dev/null 2>&1 || die "tar is required"

tmp=$(mktemp -d 2>/dev/null || mktemp -d -t isosyll)
trap 'rm -rf "$tmp"' EXIT INT TERM

base="${ISOSYLL_BASE_URL:-https://github.com/$REPO/releases/download/$VERSION}"
bin_tgz="isosyll-$VERSION-$target.tar.gz"
model_tgz="isosyll-$VERSION-model.tar.gz"

say "Downloading Isosyll $VERSION for $target"
for f in SHA256SUMS SHA256SUMS.sig "$bin_tgz" "$model_tgz"; do
	fetch "$base/$f" "$tmp/$f" || die "download failed: $base/$f"
done

if [ "$VERIFY_SIG" = 1 ]; then
	command -v ssh-keygen >/dev/null 2>&1 ||
		die "ssh-keygen (OpenSSH) is needed to check the release signature; install it, or rerun with --no-verify-signature to rely on checksums only"
	printf '%s namespaces="file" %s\n' "$SIGNER" "$SIGNER_KEY" >"$tmp/allowed_signers"
	ssh-keygen -Y verify -f "$tmp/allowed_signers" -I "$SIGNER" -n file \
		-s "$tmp/SHA256SUMS.sig" <"$tmp/SHA256SUMS" >/dev/null 2>&1 ||
		die "SHA256SUMS signature does not verify; refusing to install"
	say "Signature verified (release key SHA256:Z6YS73tC3B1XPvdmAWA9z+IM2vc2Tmch7AZmiyX0nqA)"
else
	say "warning: signature check skipped (--no-verify-signature); relying on checksums only"
fi

for f in "$bin_tgz" "$model_tgz"; do
	want=$(awk -v f="$f" '$2 == f || $2 == "*"f { print $1 }' "$tmp/SHA256SUMS")
	[ -n "$want" ] || die "$f is not listed in SHA256SUMS"
	[ "$(sha256 "$tmp/$f")" = "$want" ] || die "checksum mismatch for $f"
done
say "Checksums verified"

mkdir -p "$tmp/bin" "$tmp/model"
tar -xzf "$tmp/$bin_tgz" -C "$tmp/bin"
tar -xzf "$tmp/$model_tgz" -C "$tmp/model"
src_bin=$(find "$tmp/bin" -type f -name isosyll | head -n 1)
src_model=$(dirname "$(find "$tmp/model" -type f -name MANIFEST | head -n 1)")
if [ -z "$src_bin" ] || [ ! -f "$src_model/MANIFEST" ]; then die "release archive layout not recognised"; fi

share="$PREFIX/share/isosyll"
mkdir -p "$PREFIX/bin" "$share"
# One installer per prefix at a time.
lock="$share/.install.lock"
mkdir "$lock" 2>/dev/null || die "another install into $PREFIX is running (remove $lock if not)"
trap 'rm -rf "$tmp"; rmdir "$lock" 2>/dev/null || true' EXIT INT TERM

# Stage the new binary and model next to the old ones and verify them before touching the install.
rm -rf "$share/model.new" "$share/model.old"
cp -R "$src_model" "$share/model.new"
cp "$src_bin" "$PREFIX/bin/.isosyll.new"
chmod 755 "$PREFIX/bin/.isosyll.new"
if ! "$PREFIX/bin/.isosyll.new" model verify --model "$share/model.new" >/dev/null; then
	rm -rf "$share/model.new" "$PREFIX/bin/.isosyll.new"
	die "downloaded model failed verification; the existing install is unchanged"
fi

# Activate: keep the old model until the new pair is in place.
[ -d "$share/model" ] && mv "$share/model" "$share/model.old"
mv "$share/model.new" "$share/model"
mv "$PREFIX/bin/.isosyll.new" "$PREFIX/bin/isosyll"
rm -rf "$share/model.old"

say ""
say "Installed $("$PREFIX/bin/isosyll" --version)"
say "  binary: $PREFIX/bin/isosyll"
say "  model:  $share/model"
case ":$PATH:" in
*":$PREFIX/bin:"*) ;;
*) say ""; say "Add $PREFIX/bin to your PATH, for example:"; say "  export PATH=\"$PREFIX/bin:\$PATH\"" ;;
esac
say ""
say "Try it:  isosyll transcribe recording.wav"
