| arm | panel | round | CPUs | CPU-s/audio-s | x real time (wall) | peak RSS MB | WER clean / other / pooled | MHz | other busy cores | alone |
|---|---|---|---|---|---|---|---|---|---|---|
| jsr-d2e-16457-int8 | full | 1 | 4 | 0.0528 | 18.92 | 78 | 5.84 / 13.74 / 9.09 | 4041 | 16 | yes |
| jsr-d2e-16457-int8-w8 | full | 1 | 8-15 | 0.2097 | 38.00 | 636 | 5.84 / 13.74 / 9.09 | 3970 | 9 | yes |
| parakeet-v3-onnxasr-int8-t8 | full | 1 | 8-15 | 0.4469 | 17.51 | 1678 | 2.52 / 4.58 / 3.36 | 4003 | 15 | yes |
| r0-zipformer-int8 | full | 2 | 24 | 0.0973 | 10.28 | 151 | 3.03 / 8.43 / 5.25 | 4063 | 16 | yes |
| r0-zipformer-int8-w8 | full | 1 | 8-15 | 0.1578 | 50.46 | 1233 | 3.03 / 8.43 / 5.25 | 3964 | 7 | yes |

| arm | panel | reps | median CPU-s/audio-s | spread | x real time | peak RSS MB | model MB | same text |
|---|---|---|---|---|---|---|---|---|
| jsr-d2e-16457-int8 | full | 1 | 0.0528 | 0.0% | 18.92 | 78 | 58 | yes |
| jsr-d2e-16457-int8-w8 | full | 1 | 0.2097 | 0.0% | 38.00 | 636 | 58 | yes |
| parakeet-v3-onnxasr-int8-t8 | full | 1 | 0.4469 | 0.0% | 17.51 | 1678 | 670 | yes |
| r0-zipformer-int8 | full | 1 | 0.0973 | 0.0% | 10.28 | 151 | 73 | yes |
| r0-zipformer-int8-w8 | full | 1 | 0.1578 | 0.0% | 50.46 | 1233 | 73 | yes |
