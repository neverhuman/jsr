# JSR: ULTRA, an event-clocked, integer-exact streaming speech recogniser for CPUs

[![jankurai](https://github.com/neverhuman/jsr/actions/workflows/jankurai.yml/badge.svg)](https://github.com/neverhuman/jsr/actions/workflows/jankurai.yml)

This is a work in progress. **Nothing here has been trained yet.** See `docs/ultra/STATUS.md` for the current rung and the next task.

The controlling specification is `docs/ultra/specs/ULTRA_ASR_MASTER_ENGINEERING_SPEC.md`. `docs/ultra/specs/cadence_ultra_v3_0_master_engineering_specification.md` is a companion, used for its AVX2 lookup-table kernels and the integer PCEN front-end arm.

## Quick start

```bash
just fast          # fmt, clippy, tests
just venv          # Python trainer env (python/ultra)
just required      # full push gate, including security and jankurai audit
```

## Layout

| Path | What |
|---|---|
| `crates/ultra-core` | numeric contract (§5), output events and ledger (§3), lifecycle, receipts |
| `crates/ultra-kernels` | scalar reference, AVX2 and AVX-512 ternary and int8 kernels, bit-exact property tests |
| `crates/ultra-frontend` | float log-mel control and integer Cochlea |
| `crates/ultra-gp` | `(asr …)` program parser, canonical printer, expression checker, static cost estimator |
| `crates/ultra-model`, `crates/ultra-stream` | integer reference runtime, scheduler, taps, shedding |
| `crates/ultra-eval` | normaliser, WER, delays, paced replay, process-tree CPU accounting, bootstrap |
| `crates/ultra-bench` | the cost-law benchmark (`wstream` port) and kernel bake-off |
| `crates/ultra-adapters`, `crates/ultra-cli` | external recogniser adapters; command-line tool |
| `python/ultra` | PyTorch trainer and data prep (dated Python exception), waves |
| `docs/ultra` | STATUS, DECISIONS, RULINGS, ledgers, specs |

## Gates

`just required` runs fmt, clippy (`-D warnings`), the tests, pytest, the security lane and a jankurai audit. It must pass before every push.

Data, checkpoints and build output live on the data drive (`/mnt/xbabe3-external/jsr`), never in git.
