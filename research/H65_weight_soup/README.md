# H65. Interpolating the M1-v4 and M1-v5 EMA weights

**Status:** Closed. The noise gain replicated on confirmation. The clean
non-inferiority gate failed.

H65 tested linear interpolation of the two terminal EMA checkpoints, with
identical tensor names, shapes and dtypes. It selected the lowest clean WER
among the three noise-eligible interpolation weights on the DEV selection
half (S). Both endpoint weights reproduced the respective cold c4 clean
edit counts exactly. The selected weight was 0.50. It beat signed RC2 on S
by 0.0958 pp, with a one-sided upper of −0.0019 pp. The PyTorch RC2 clean
upper on S was +0.013 pp. On the 5 dB MUSAN-TEST condition the soup improved
by 9.24 pp against PyTorch RC2 and by 9.99 pp against M1-v4.

The confirmation protocol and the 0.50 candidate were frozen after selection
and before any DEV confirmation-half (C) decode. H0 decoded the same C-half
inputs for the soup and PyTorch v3.1.0 RC2 with the c4 EMA harness. The score
used 2,785 utterances and 53,858 reference words, with 10,000 paired
speaker-cluster bootstrap replicates from `ultra-score` (`d784f8ca`, seed
20261122).

| C-half leg | Soup WER | PyTorch RC2 WER | Δ (pp) | One-sided upper (pp) | Gate |
|---|---:|---:|---:|---:|---|
| Clean c4 | 6.7325% | 6.6917% | +0.0408 | +0.1439 | FAIL (≤ +0.087 pp) |
| MUSAN-TEST 5 dB c4 | 16.068% | 27.641% | −11.5730 | −10.5638 | PASS (< 0 pp) |

The selected soup passed EMA export and native load checks. INT8 and FP32
transcripts differed on the smoke panel. The C-half clean miss ended the
release path before exact-package qualification.

The [evidence checksums](../v3.2.0-closeout.SHA256SUMS) identify the selection
and confirmation protocols, the soup checkpoint manifest, the export preflight
and both sets of result and gate receipts.
