# Audit rubric

JSR uses the jankurai audit (`just score`, binary pinned at v1.7.6 locally;
CI uses `neverhuman/jankurai-action`, `fail-under: 85`). The report is
`target/jankurai/repo-score.{json,md}`; read `findings[]` and `caps_applied`.

## Rules for fixing findings

- Fix the cause. A finding is resolved by real code, a real test, a real script
  that runs, or a true document, never by adding text whose only purpose is to
  match a detector. After any score gain in a text-matched dimension, remove the
  added text and re-measure to show the gain came from the change.
- Never weaken a test or a gate to pass the audit.
- When a rule does not fit JSR (it assumes a web app, an API, a database), file a
  dated exception under `docs/exceptions/` with owner, expiry, proof lane and
  manifest entry, instead of creating empty template directories.
- Hard findings and caps in touched files block a commit (pre-commit
  `jankurai diff-audit --base-ref HEAD`). Bypass (`JANKURAI_SKIP_HOOKS=1`) only
  when the finding lies solely in files another lane owns, and say so in the
  commit message.

## What every path needs

- An owner in `agent/owner-map.json` and a runnable proof command in
  `agent/test-map.json`.
- Generated or checked files declared in `agent/generated-zones.toml`
  (`docs/generated-zones.md`).
- Typed refusals catalogued in `agent/exceptions.toml` (`docs/errors.md`).

Related: `docs/testing.md` (lanes), `docs/boundaries.md`, `docs/release.md`.
