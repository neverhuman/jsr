# Isosyll

Streaming speech recognition on one CPU core. Pure Rust; the model is INT8
safetensors.

## Install

The installer puts this binary at `~/.local/bin/isosyll` and the model at
`~/.local/share/isosyll/model`:

    curl -fsSL https://github.com/neverhuman/jsr/releases/latest/download/install.sh | sh

To install by hand, copy `bin/isosyll` onto your `PATH`. Unpack the release's
`isosyll-<version>-model.tar.gz` and point `--model` or `ISOSYLL_MODEL` at the
`isosyll-model` directory inside it.

## Use

    isosyll transcribe recording.wav             # final text
    isosyll transcribe --mode live recording.wav # 160 ms live text
    isosyll stream recording.wav                 # live partials, then final text, in real time
    isosyll model verify                         # check the model files' sha256
    isosyll --help

Input is WAV (16-bit PCM, any sample rate, mono or stereo) or raw 16 kHz
16-bit little-endian PCM (`.pcm`, `.s16le`).

Documentation, benchmarks and the paper: https://github.com/neverhuman/jsr
