# H53 S2 fast-cooldown factorial: H43's 2^(7−4) screen with fold-over (DRAFT for the pre-registration)

Draft by trainer-lane, 2026-11-23 (lead 09:08Z). h53-trunk owns the final text, and eval-lane registers the reading. **It must be fixed before S2's first branch point (BP1, 10k S2 audio-h).** Nothing runs before S2 exists, and S2 itself waits for the H52 pilot and the B0 bridge.

## Question
Which cooldown-recipe changes improve the annealed S2 checkpoint by a real ~0.1 pp, given that single-factor arms keep landing INCONCLUSIVE at that size? The screen tests 7 changes in 8 runs, each main effect estimated 4-vs-4. Factors that pass go to a seed-pair confirmation under H43's adoption rule.

## Base (every run)
- **Init:** BP1's raw weights.
- **Recipe:** a fast cooldown (A4) with the low re-warm (Amendment 3):
  - the H41 replay machinery at 650 s / 144, 8,000 steps;
  - 200 warmup steps to **1.24e-4**, which is S2's 3.511e-4 at 5,200 s, √-scaled to 650 s;
  - cosine to 3e-5, EMA 0.9995;
  - S2's data mix and S2's dp (Amendment 4), except where factor D sets it;
  - **seed 1234** for every run, so seed noise is not confounded with factors.
- **Matched control:** **R0** = the plain FC from BP1, run alongside (A4's matched base).
- **Tree:** one pinned commit holding every factor's code, all off by default. Each factor's witness tests show the off level is bit-identical to the plain FC.

## Factors (− = off, + = on)
| | factor | + level | code status (09:10Z) |
|---|---|---|---|
| A | trained full-context commit mode | dual-mode chunk 0 / 16 self-distillation: a full-context pass per batch, KL from it into the chunk-16 output, weight λ = 0.1 (TBD) | **to write** (trainer-lane) |
| B | augmentation parity | **definition needed from the audit**: proposal `synth_aug` (RIR + pink / white noise) + random input gain ±20 dB, so training matches the input levels and noise of the gain-stress panel | partly exists (`synth_aug`); gain part to write |
| C | stochastic depth | layer drop, p linear in depth to 0.10, training only | **to write** |
| D | delay penalty | `delay_penalty=0.005` (the − level is 0) | exists |
| E | +text predictor tables | h46-lm's patch | **h46-lm's patch** (status?) |
| F | H48c MFA anchor | `anchor_kappa=0.01 anchor_margin=1 anchor_cap=12` | PR #93. The side file covers 960 h only; Libri-Light tokens cost 0 |
| G | H49b sequence KD | h49-seqkd's term from the R0 N-best store | **h49's patch** + kd-cache's 960 h N-best (ETA ~10:50Z) |

**Alternates, fixed now,** for a factor whose code isn't merged and witnessed 2 h before BP1:
- G' = LS-only finish (the cooldown on 960 h only);
- F' = conf floor 0.5 on the extra splits;
- the remaining slot left blank (a dummy column) rather than an untested change.

## Design: 2^(7−4), resolution III
Generators **D = AB, E = AC, F = BC, G = ABC** (defining relation I = ABD = ACE = BCF = ABCG and products).

| run | A | B | C | D | E | F | G |
|---|---|---|---|---|---|---|---|
| 1 | − | − | − | + | + | + | − |
| 2 | + | − | − | − | − | + | + |
| 3 | − | + | − | − | + | − | + |
| 4 | + | + | − | + | − | − | − |
| 5 | − | − | + | + | − | − | + |
| 6 | + | − | + | − | + | − | − |
| 7 | − | + | + | − | − | + | − |
| 8 | + | + | + | + | + | + | + |

- Main effects are aliased with 2-factor interactions.
- **Fold-over** (all signs reversed, 8 more runs) frees main effects from them (resolution IV). It runs **if ≥ 2 factors pass the screen**.

## Responses
1. **Primary:** EMA, full-dev pooled WER, canonical scorer, **chunk 4** (160 ms provisional text).
2. **Second, lower variance:** **per-token NLL** on full dev: the transducer loss per reference token, EMA weights, chunk 4, computed once at step 8,000.
3. **Reported:** chunk 16 (the commit pass), first-token latency at chunk 4 (the D and F factors), CPU cost (the E factor).

## Reading (eval-lane registers the exact estimator)
- **Effect of a factor** = mean(+ runs) − mean(− runs), 4 vs 4, with an interval from the registered run-to-run noise (the four-run spread).
- **A factor passes the screen** if its chunk-4 WER effect is ≤ −0.087 pp (one spread), its NLL effect has the same sign, and its chunk-16 effect is not worse than +0.094.
- **Passing factors go to confirmation:** BP1 FC with the factor vs R0, seed pair 1234 / 4321, H43's 2× spread rule, Holm over the confirmed set. At most two methods are confirmed at a time (codex-lead).
- Factors that fail are reported as such; none is "adopted" from the screen alone.

## Compute
- 8 runs + R0 = **9R** (~5 A100-h). The fold-over adds 8R.
- Each FC is one A100 run of ~33 min (A100 only for factors A / C, because of memory; the rest also fit 24 GB).
- On the six A100s that's ~2 h for the screen, with the trunk yielding by its checkpoint rule.

## Open, for h53-trunk / the lead
1. B's definition.
2. A's λ and which direction the distillation runs (0 → 16 or 16 → 0).
3. Whether E's CPU cost (+text tables at inference) needs its own gate.
4. Code owners and deadlines for A / B / C / E / G.
