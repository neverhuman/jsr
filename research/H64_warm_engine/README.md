# H64. Guarded encoder-state carry

**Status:** Closed. Both accuracy gates failed and the CPU result missed its
bound.

The prototype carried M1-v4 terminal EMA INT8 encoder caches across utterance
boundaries when the d2 speaker guard allowed it. It decoded the DEV selection
half (S) with the same c4 stream policy in natural chapter order (NAT) and
with speakers shuffled (SHUF). The cold ALONE reset baseline also matched
`ultra-cold-wav` on 60 clips. The guarded audio-prepend reference re-encoded
5 seconds of prior audio with a 0.3-second gap. The d2 decision required
buffering the first second of new audio. First-partial delay was left unmeasured.

| CARRY_d2 vs ALONE | Point estimate | Upper used for the gate | Threshold |
|---|---:|---:|---:|
| NAT WER change | −0.077 pp | +0.059 pp | < 0 pp |
| SHUF WER change | +0.054 pp | +0.119 pp | ≤ +0.087 pp |
| CPU per audio second | +1.17% | +2.15% | ≤ +1.5% |

The WER uppers are from two-sided 95% paired speaker-cluster intervals
(`ultra-score` `d784f8ca`, 10,000 replicates, seed 20261122). CPU used paired
runs on pinned H0 cores and a one-sided 95% upper. Host contention and
run-order drift prevented a reliable swap-only CPU estimate. NAT audio-prepend
improved WER by 0.167 pp and was the stronger of the two warmed paths.

The [evidence checksums](../v3.2.0-closeout.SHA256SUMS) identify the preregistration,
result and gate receipts, and archived source. No warm-carry implementation
was promoted.
