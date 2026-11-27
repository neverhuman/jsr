# H286: final-pass reranker on the current best acoustic (H275 F3 soup)

Watcher request (06:58): S/D/I analysis of H275 F3 final (4.5272%) vs H244 (4.6287%); refit H261 on F3 TRAIN lists; target DEV final < 4.50 at ≤ +11% commit cost.

## (1) S/D/I: F3 vs H244 (DEV S, 51,834 ref words; `tools/sdi.py`)
- Edits 2452 → 2396 (net −56): fixed 185, broke 129. Net S −38, D −38, I +20 (203 → 223; broken insertions are AND / A / SIR / THE).
- Remaining: S 1891 (79%), D 282, I 223. Rare content subs (train-960 count < 100): 770, of which 600 have the ref word nowhere in the 16-best.
- Remaining S/D with the right word somewhere in the 16-best: 710 of 2173.
- Top remaining frequent-content subs: MAN→MEN 14, O→OH 9, HONOR→HONOUR 5 (about 15 US/UK spelling), ONE→EVERYONE 5, EDISON→ADDISON 5, ALLAN→ALLEN 5.
- Full output: `/mnt/xbabe3-external/jsr/wer-push/H286/sdi_f3_vs_h244.json`.

## (2) Refit on F3 TRAIN lists
- TRAIN: H261 MLS-en set (9,196 utts, 612 AM-unseen speakers), decoded with F3 .30/.55/.15 INT8 (weights 5c5bfad9…), c32 B16, final.ulm 0.8 / LODR 0.6 on xbabe0 (10 min).
- DEV S lists: codex H275 exact F3 inputs. Frozen H261 offline replay matches native on 2770/2782 utts (offline 4.54).

| run | TRAIN CV (held-out Δ edits) | DEV S | vs F3 native 4.5272 | vs H244 |
|---|---|---|---|---|
| frozen H261 (offline) | – | 4.54 | +0.009 [+0.000, +0.019] | |
| refit, λ 1e-4 (fitF3.json df450299…) | −130 | 4.53 | +0.004 [−0.004, +0.014] | −0.097 [−0.164, −0.025] |
| + MBR consensus | −95 (best λ) | not applied | | |
| + MBR + rank + ins | −111 (H255 lists) | not applied | | |

- TRAIN routing scan (`tools/skip_scan.py`): skip 3.75 → 88% routed, TRAIN WER 5.111; skip 5 → 95%, 5.110; all → 5.111. The current skip is already at the optimum.
- **Verdict: null.** The current LM feature set is saturated on F3. < 4.50 is not reachable on the final pass at ≤ +11% cost with these LMs. The remaining headroom needs a stronger LM (out of budget) or better acoustic lists. H283@2000 and the F7 soups trail F3 (codex H285), so no refit was run on them.

Tools: `sdi.py`, `extra_feats.py`, `rerank_x.py` (H261 reranker with `EXTRA=` features and `SKIP=`), `skip_scan.py`. Bulky data: `/mnt/xbabe3-external/jsr/wer-push/H286/`.
