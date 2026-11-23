Optimistic 2x table-build reduction (the pre-registered NO-GO test), speedup vs INT8, chunk 1 / 2 / 4:
| assignment | swarm staged format, LUT x2 | H24 format, table share halved |
|---|---|---|
| all 2 bpw | 2.37 / 1.55 / 0.97 | 2.25 / 1.67 / 1.20 |
| all 1 bpw | 3.44 / 2.58 / 1.72 | 3.32 / 2.56 / 1.87 |
| LN-input 2 bpw, rest INT8 | 1.55 / 1.45 / 1.18 | 1.55 / 1.36 / 1.13 |
| LN-input 1 bpw, rest INT8 | 1.74 / 1.65 / 1.43 | 1.74 / 1.61 / 1.41 |
| LN-input 1 bpw, post-SiLU 2 bpw | 2.86 / 1.78 / 1.14 | 2.69 / 2.05 / 1.51 |
(H24 table shares used: 384->1536 11/13/15%, 1536->384 38/45/49% at M 1/2/4.)
