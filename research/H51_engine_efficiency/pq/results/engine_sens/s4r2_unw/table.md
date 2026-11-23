# Per-layer PQ sensitivity, s4r2_unw (730-utt DEV panel, chunk 4, engine INT8 elsewhere)

8 layers. ΔWER = variant − INT8 baseline (7.385% pooled), percentage points, ultra-score paired speaker-cluster bootstrap 95% interval (10,000 replicates, seed 20261122). rel MSE = held-out TRAIN output error / output variance.

| rank | layer | N×K | INT8 B | PQ B (codes) | rel MSE INT8 | rel MSE PQ (exact LUT) | PQ no-LUT | PQ→int8 (engine run) | ΔWER pooled [95%] | Δclean | Δother |
|---:|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| 1 | blocks.8.ff1.w2 | 384×1536 | 589,824 | 201,216 (147,456) | n/a | n/a | n/a | n/a | -0.014 [-0.108, +0.080] | +0.000 | -0.034 |
| 2 | blocks.8.att.qkv | 1152×384 | 442,368 | 136,704 (110,592) | n/a | n/a | n/a | n/a | +0.048 [-0.056, +0.157] | +0.141 | -0.084 |
| 3 | blocks.8.att.out | 384×384 | 147,456 | 53,760 (36,864) | n/a | n/a | n/a | n/a | +0.083 [+0.000, +0.171] | +0.059 | +0.118 |
| 4 | blocks.8.ff1.w1 | 1536×384 | 589,824 | 178,176 (147,456) | n/a | n/a | n/a | n/a | +0.118 [+0.000, +0.233] | +0.165 | +0.050 |
| 5 | blocks.8.ff2.w2 | 384×1536 | 589,824 | 201,216 (147,456) | n/a | n/a | n/a | n/a | +0.118 [+0.007, +0.231] | +0.129 | +0.101 |
| 6 | blocks.8.conv.pw2 | 384×384 | 147,456 | 53,760 (36,864) | n/a | n/a | n/a | n/a | +0.125 [+0.028, +0.221] | +0.141 | +0.101 |
| 7 | blocks.8.conv.pw1 | 768×384 | 294,912 | 95,232 (73,728) | n/a | n/a | n/a | n/a | +0.159 [+0.042, +0.279] | +0.212 | +0.084 |
| 8 | blocks.8.ff2.w1 | 1536×384 | 589,824 | 178,176 (147,456) | n/a | n/a | n/a | n/a | +0.194 [+0.065, +0.322] | +0.129 | +0.286 |

## By kind

| kind | layers | INT8 MB | mean ΔWER pp | max ΔWER pp | mean rel MSE PQ |
|---|---:|---:|---:|---:|---:|
| ff1_w2 | 1 | 0.59 | -0.014 | -0.014 | 0.0000 |
| att_qkv | 1 | 0.44 | +0.048 | +0.048 | 0.0000 |
| att_out | 1 | 0.15 | +0.083 | +0.083 | 0.0000 |
| ff1_w1 | 1 | 0.59 | +0.118 | +0.118 | 0.0000 |
| ff2_w2 | 1 | 0.59 | +0.118 | +0.118 | 0.0000 |
| conv_pw2 | 1 | 0.15 | +0.125 | +0.125 | 0.0000 |
| conv_pw1 | 1 | 0.29 | +0.159 | +0.159 | 0.0000 |
| ff2_w1 | 1 | 0.59 | +0.194 | +0.194 | 0.0000 |

## Bytes coverable at a per-layer ΔWER threshold

| threshold pp | layers | share of INT8 bytes |
|---:|---:|---:|
| 0.00 | 1 | 17.4% |
| 0.03 | 1 | 17.4% |
| 0.05 | 2 | 30.4% |
| 0.10 | 3 | 34.8% |
| 0.15 | 6 | 73.9% |
| 0.20 | 8 | 100.0% |

## By block depth

| block | mean ΔWER pp | mean rel MSE PQ (exact LUT) | mean rel MSE PQ (no LUT) |
|---:|---:|---:|---:|
| 8 | +0.104 | 0.0000 | 0.0000 |
