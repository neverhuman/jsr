# Grey-name energy: a labeled prediction, plus one H0 core check

This note does not add a paper number. Eight names on the figure have no admitted core-energy receipt. R7 on xbabe0 is the measurement of those arms. The anchors for that campaign passed on 2026-11-26 (Isosyll live 0.3209 J/audio-s, −2.2% vs R6 0.328; Zipformer 0.5668, −1.1% vs R6 0.573). Until a clean R7 row exists for a name, that name stays grey.

## What the admitted rows already show

On this Zen 2, joules per audio-second are the busy-core watts divided by the published single-core speed. The core register, while one decoder is pinned to it, drew between 8.041 W and 11.041 W on every admitted system (`energy_active_w` in the evidence JSONL, method `core_msr_active`):

| System | Receipt row | Published speed | Busy-core W | Measured J/audio-s |
|---|---|---:|---:|---:|
| Zipformer INT8 | `r0-int8-r6-zen2-t1` | 14.0× | 8.041 | 0.573 |
| Moonshine Tiny | `moonshine-tiny-zen2-t1` | 8.20× | 8.850 | 1.077 |
| Parakeet sherpa | `parakeet-tdt-v3-int8-zen2-t1` | 5.92× | 10.120 | 1.711 |
| Isosyll live | `rc2-live-lm-zen2-t1` | 32.7× | 10.720 | 0.328 |
| Nemotron 560 ms | `nemotron-en-560-zen2-t1` | 2.38× | 10.726 | 4.513 |
| Isosyll final | `rc2-final-zen2-t1` | 16.8× | 11.041 | 0.661 |

`busy-core W / published speed` matches the measured joules within the rounding of the published speed. The largest gap is the final row: 11.041 / 16.8 = 0.657 against a receipt of 0.661. Package energy is not used. Idle-against-idle package power on this CPU moves by more than one core.

## Prediction for the eight grey names

Band = `8.04 / speed` to `11.04 / speed`, the observed min and max above. A § speed is the 73-clip timing subset already published. These rows are predictions.

| System | Published speed | Predicted J/audio-s |
|---|---:|---:|
| Vosk Small | 6.88× | 1.17–1.60 |
| Parakeet onnx-asr | 6.69× | 1.20–1.65 |
| Vosk Large | 4.13× | 1.95–2.67 |
| Moonshine Small | 2.02× | 3.98–5.47 |
| Nemotron 160 ms | 1.28× | 6.28–8.63 |
| Whisper.cpp Tiny | 0.91× | 8.84–12.1 |
| Nemotron 3.5 | 0.84× | 9.57–13.1 |
| Whisper.cpp Base | 0.42× | 19.1–26.3 |

Two older joins were left out of the paper and stay out. Vosk Small read 1.279 J with an idle near 0.94 W (`vosk-small-zen2-t1`, busy-core 8.80 W). Parakeet onnx-asr read 1.661 J with an idle near 0.99 W and other cores busy (`parakeet-v3-onnxasr-int8-zen2-t1`, busy-core 11.12 W). Both joule readings sit in the bands above. The idle floor on those joins is not the quiet-host floor, so they are a sanity check only.

## H0 check, 2026-11-26, this machine (xbabe3)

Same CPU, Threadripper PRO 3995WX. Not a quiet host, so this is a method check and not an H1 receipt.

`energy_bench.py run`, core 20, SMT sibling 84 kept idle, observer on CPU 48, one repetition: idle 60 s, integer spin 30 s, idle 60 s. Receipt: `/tmp/h0-energy-check/spin.json` on this host (not committed; it is a local method log).

- Core view, clean: 10.05 W net per busy core (run 10.95 W, idle 0.90 W). Sibling busy during the spin: 0.13%. Mean frequency 4.10 GHz.
- Package view, not clean: flags `foreign_load_changed`, `idle_drift_package`. Idle package 122 W. Other cores were drawing about 48 W. That is why the package number is discarded.

10.05 W sits inside the 8.04–11.04 W band from the admitted decoders. The pass rule was the H39 busy-loop check: watts per core within that band. The core view passes. The package view does not, as expected on a shared host.

No Vosk decode was run here. The staged opponent audio and models are not on this machine, and the other cores were not idle. No GPU decode was run. The RTX 3090 was idle near 27 W, and no paper-model GPU runtime is installed. A board-power reading would be a different quantity from a core-register joule and would not fill a grey cell. Vendor TDP is not used.

## What may enter the paper later

R7 may recolor a grey name only when that arm is clean under the H39 flags. The campaign's own anchor rule already passed for block 1. This band is not a colorbar tick. It is the range those R7 rows should fall in if a pinned Zen 2 core still draws 8–11 W while decoding.
