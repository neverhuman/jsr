# Agent Instructions

Read `agent/JANKURAI_STANDARD.md` first. For explicit phase or MASTER_PLAN work only, read `agent/MASTER_PLAN.md` before `tips/phases/00-phase-index.md`. Keep generated artifacts under their declared source commands.

Routes: architecture `docs/architecture.md`, boundaries `docs/boundaries.md`, tests and lanes `docs/testing.md`, generated zones `docs/generated-zones.md`, errors `docs/errors.md`, audit rules `docs/audit-rubric.md`, exceptions `docs/exceptions/`. `just required` is the push gate.

## Workspace Boundary

- Work only in the user-named active repo/worktree.
- Never switch to sibling clones, archives, backups, resolved symlink targets, `/tmp` worktrees, or duplicate roots.
- Never create repo copies or side folders outside the active repo; preserve work with git branches.
- Before edits, report `pwd`, `git rev-parse --show-toplevel`, and `git status --short --branch`.
- Use Jeryu APIs/CLI for forge remote and merge-request work; no credential scraping or raw forge API calls.

