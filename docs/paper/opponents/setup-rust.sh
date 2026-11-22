#!/usr/bin/env bash
# Build the cold-benchmark Rust binaries in this lane with an isolated toolchain.
# usage: setup-rust.sh OPPONENTS_ROOT TARGET_DIR [CPU_LIST]
# Run from the lane checkout. Nothing is written outside OPPONENTS_ROOT and TARGET_DIR.
set -euo pipefail
O=$1; target=$2; cpus=${3:-}
tool="$O/bench-cross/toolchain"
export RUSTUP_HOME="$tool/rustup" CARGO_HOME="$tool/cargo"
export CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only
channel=$(awk -F'"' '/^channel/ {print $2}' rust-toolchain.toml)
pin=()
if [[ -n $cpus ]]; then
    pin=(taskset -c "$cpus")
fi
sum() {
    if command -v sha256sum >/dev/null; then sha256sum "$1" | awk '{print $1}'; else shasum -a 256 "$1" | awk '{print $1}'; fi
}
if [[ ! -x "$CARGO_HOME/bin/cargo" ]]; then
    case "$(uname -s)-$(uname -m)" in
        Linux-x86_64) triple=x86_64-unknown-linux-gnu ;;
        Darwin-arm64) triple=aarch64-apple-darwin ;;
        *) echo "unsupported host" >&2; exit 2 ;;
    esac
    dl="$O/bench-cross/downloads/rustup"; mkdir -p "$dl" "$tool"
    curl -sSL --fail -o "$dl/rustup-init" "https://static.rust-lang.org/rustup/dist/$triple/rustup-init"
    curl -sSL --fail -o "$dl/rustup-init.sha256" "https://static.rust-lang.org/rustup/dist/$triple/rustup-init.sha256"
    [[ "$(awk '{print $1}' "$dl/rustup-init.sha256")" == "$(sum "$dl/rustup-init")" ]] || { echo "rustup-init hash mismatch" >&2; exit 1; }
    chmod +x "$dl/rustup-init"
    "$dl/rustup-init" -y --no-modify-path --profile minimal --default-toolchain "$channel" >/dev/null
fi
jobs=${SETUP_JOBS:-2}
"${pin[@]}" "$CARGO_HOME/bin/cargo" build --locked --release -j "$jobs" --target-dir "$target" \
    -p ultra-bridge --bin ultra-cold-wav
"${pin[@]}" "$CARGO_HOME/bin/cargo" build --locked --release -j "$jobs" --target-dir "$target" \
    -p ultra-eval --bin ultra-measure --bin ultra-score
mkdir -p "$O/bench-cross/bin"
for bin in ultra-cold-wav ultra-measure ultra-score; do
    cp "$target/release/$bin" "$O/bench-cross/bin/$bin"
done
{
    printf 'source_commit %s\n' "$(git rev-parse HEAD)"
    printf 'dirty %s\n' "$(git status --porcelain --untracked-files=no | wc -l | tr -d ' ')"
    "$CARGO_HOME/bin/rustc" --version
    for bin in ultra-cold-wav ultra-measure ultra-score; do
        printf '%s  %s\n' "$(sum "$O/bench-cross/bin/$bin")" "$bin"
    done
} > "$O/bench-cross/bin/rust-build.provenance"
cat "$O/bench-cross/bin/rust-build.provenance"
