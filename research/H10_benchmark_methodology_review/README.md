# H10: Independent methodology review → FREEZE

**Status:** ongoing (fixes landing; re-review open).

## Question
Would our planned v1.0.0 claims survive an outside audit?

## Setup
- **First review:** a fresh-context reviewer with no shared conclusions read the code behind every claim (`docs/ultra/reviews/2026-11-21-independent-freeze-review.md`).
- **Re-review:** after the fixes (`2026-11-21-independent-freeze-rereview.md`).
- **Earlier reviews:** codex reviews of the earlier pieces are in `docs/ultra/reviews/`.

## Results: first review, 7 blocking findings
1. **The density ratio compared different hardware.** The engine used 4 workers per CCX; R0 used 1 worker per CCX. The about-11× figure is about 2.8× per core.
2. **Three different A1 implementations** were in use (engine dev tool, bench-suite, Python harness), with different grids and lateness rules.
3. **Nemotron had no density path.**
4. **No tool could produce a qualifying row.**
5. **The DEV selection panel was unused by the tools.**
6. **The claims weren't pre-registered** in a testable form.
7. **Density was quoted without the WER gap.**

## Re-review
- 3 resolved, 9 partial, 3 unresolved.
- **6 new issues**, including:
  - the density ladder could skip a failure;
  - the gate tool exited 0 on FAIL;
  - stale clean-build flags;
  - mixed utterance sets across rows.

## Consequence
- **The cross-harness density claim was retracted** in STATUS.
- **FREEZE.md was rewritten:**
  - bench-suite is the only evidence harness;
  - equal hardware: 4 workers per CCX, and the full machine is 15 CCX plus an observer CCX;
  - one A1 rule;
  - a fixed ladder that stops at the first failure;
  - Nemotron is out of the density and cost claims;
  - the INT8 gate runs on the 730-utterance panel with a bootstrap bound;
  - prior test reads are disclosed;
  - the Holm family is limited to tested claims.
- **Fix work is assigned** to runtime, engine and trainer.

## Lesson
Have an independent reviewer read the code, not the summaries, before freezing claims. Several "fixed" items were claimed but not implemented.

## Pointers
- `docs/ultra/FREEZE.md`, `docs/ultra/PLAN_TO_V1.md`, `docs/ultra/DECISIONS.md`.
- Commits: 17338f3, e0da750, 37df09d.
