# H63. Full-horizon turn-switch training with noise

**Status:** Closed. Both noise recipes missed the clean gate.

H63 tested whether training through the full schedule with turn-switch context and
MUSAN TRAIN augmentation could improve held-out noise WER while preserving clean
speech. M1-v4 was the turn-switch comparator. M1-v5 used a 19,500-step schedule
with a 1,000-step clean tail. M1-v6 extended the schedule to 29,250 steps, also
with a 1,000-step clean tail. Both were screened at seed 7919. The stream-context
training implementation is in [#160](https://github.com/neverhuman/jsr/pull/160).

## Results

Terminal EMA, cold c4 on the DEV selection half (S). Clean comparisons use signed
v3.1.0 RC2. The noise comparisons decode the PyTorch RC2 reference and M1-v4
with the same 5 dB MUSAN-TEST mixtures on H0. Deltas are candidate minus comparator
in WER percentage points (pp). Bounds are one-sided 95% speaker-cluster bootstrap
uppers from `ultra-score` (`d784f8ca`, 10,000 replicates, seed 20261122).

| Candidate | Clean WER | Clean Δ / upper vs signed RC2 (pp) | Noise WER | Noise Δ / upper vs PyTorch RC2 (pp) | Noise Δ / upper vs M1-v4 (pp) |
|---|---:|---:|---:|---:|---:|
| M1-v5 | 7.0254% | +0.0786 / +0.1789 | 13.3459% | −11.4051 / −10.1177 | −12.1580 / −10.8062 |
| M1-v6 | 7.2017% | +0.2548 / +0.3579 | 13.5336% | −11.2173 / −9.9557 | −11.9703 / −10.6407 |

Signed RC2 scored 6.9469% on clean c4. PyTorch RC2 and M1-v4 scored 24.7509%
and 25.5039% on the noise condition. M1-v4's clean delta was −0.0517 pp with
an upper of +0.0441 pp. The H63 clean upper had to be at most zero. Both
noise recipes also exceeded the +0.02 pp clean point tolerance in Addendum B.
H63 stopped before further seeds or confirmation. H66 tested a lighter noise
recipe and a longer clean tail.

The [evidence checksums](../v3.2.0-closeout.SHA256SUMS) identify the frozen
protocol and addenda, terminal gates and results, and retained checkpoint
manifests. Two H63 trajectory logs differ from their entries in the broad
score manifest. The terminal gate and selection receipts match their entries.
