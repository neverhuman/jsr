# Install Isosyll and transcribe audio

## Install the release

Isosyll v4.2.0 runs locally on Linux x86-64, Linux arm64, and macOS on Apple
silicon. The installer needs curl or wget, OpenSSH's `ssh-keygen`, an SHA-256
utility, and tar. It installs the binary and model under `~/.local` by default.
On Linux x86-64 it selects the glibc build with glibc 2.35 or newer and uses the
static build on other systems. [Release record](ultra/releases/v4.2.0.md)

```bash
(
  set -euo pipefail
  curl --proto '=https' --proto-redir '=https' -fsSL \
    https://github.com/neverhuman/jsr/releases/download/v4.2.0/install.sh |
    sh -s -- --version v4.2.0
) &&
export PATH="$HOME/.local/bin:$PATH" &&
isosyll --version &&
isosyll model verify
```

Pass `--prefix DIR` to install elsewhere and add `DIR/bin` to `PATH`. The
installer checks the release signature and checksums. The model can be used
offline after installation. Allow about 720 MB for the unpacked model.

## Transcribe one recording

Replace `recording.wav` with a PCM16 WAV file. Mono and stereo inputs are
supported. The CLI also accepts raw 16 kHz little-endian PCM in `.pcm` or
`.s16le` files.

```bash
isosyll transcribe recording.wav
isosyll transcribe --mode live recording.wav
isosyll stream recording.wav
```

`transcribe` defaults to final text. The live mode uses the 160 ms streaming
decoder. `stream` replays the file with live partials and final text as it plays.
Add `--json` to `transcribe` for one JSON object per input file. The final
mode accepts `--rms-norm` for optional input level normalisation.

## Verify and unpack by hand

Run this from a trusted checkout. Use the checkout's
[`allowed_signers`](ultra/releases/allowed_signers) file to authenticate the
downloaded manifest. Select `x86_64-linux` for Linux x86-64 with glibc 2.35 or
newer, `x86_64-linux-musl` for the static x86-64 build, `aarch64-linux` for
Linux arm64, or `aarch64-macos` for Apple silicon.

```bash
(
  set -euo pipefail
  v=v4.2.0
  t=x86_64-linux
  base="https://github.com/neverhuman/jsr/releases/download/$v"
  bundle="isosyll-$v-$t.tar.gz"
  model="isosyll-$v-model.tar.gz"
  signers="$PWD/docs/ultra/releases/allowed_signers"
  test -r "$signers"
  mkdir -p "$PWD/target"
  dst=$(mktemp -d "$PWD/target/isosyll-manual.XXXXXXXX")
  cd "$dst"
  for f in SHA256SUMS SHA256SUMS.sig; do
    curl --proto '=https' --proto-redir '=https' -fsSL -o "$f" "$base/$f"
  done
  ssh-keygen -Y verify -f "$signers" -I 219957145+neverhumanbot@users.noreply.github.com \
    -n file -s SHA256SUMS.sig < SHA256SUMS
  for f in "$bundle" "$model"; do
    awk -v f="$f" '$2 == f { print; count++ } END { exit count != 1 }' SHA256SUMS > selected.sha256
    curl --proto '=https' --proto-redir '=https' -fsSL -o "$f" "$base/$f"
    if command -v sha256sum >/dev/null 2>&1; then
      sha256sum -c selected.sha256
    else
      shasum -a 256 -c selected.sha256
    fi
  done
  tar -xzf "$bundle"
  tar -xzf "$model"
  "./isosyll-$v-$t/bin/isosyll" model verify --model isosyll-model
  printf 'Verified binary and model in %s\n' "$dst"
)
```

To run the unpacked binary, give it the model directory with `--model DIR`
or set `ISOSYLL_MODEL` to that directory. For paper replay and release
packaging, see the [release guide](release.md).
