#!/usr/bin/env bash
# CI pytest lane: the python/ultra tests that need neither torch/k2 nor LaTeX (paper data and
# evidence loader, benchmark tooling, dashboards, guards, OOD seal, WER). The local merge gate
# (`just required`) still runs the whole suite. CI ran no pytest at all before this lane, which let a
# renamed helper break main unnoticed on 2026-11-22 (two PRs merged in the same hour).
set -Eeuo pipefail
# shellcheck source=ops/ci/lib.sh
. "$(dirname "$0")/lib.sh"
cd "$(ci_repo_root)"
ci_lane pytest-light "bash ops/ci/pytest-light.sh"

venv="${RUNNER_TEMP:-$PWD/target}/ci-pytest-venv"
ci_step "python venv (pinned: pytest, pytest-asyncio, numpy, aiohttp, pyarrow)"
python3 -m venv "$venv"
"$venv/bin/pip" install -q pytest==9.2.0 pytest-asyncio==1.5.0 numpy==2.5.5 aiohttp==3.14.5 pyarrow==17.0.0

ci_step "pytest (files listed in ops/ci/pytest-light.txt)"
mapfile -t files < <(grep -vE '^\s*(#|$)' ops/ci/pytest-light.txt)
cd python/ultra
# Two figure tests compile LaTeX (pdflatex), which the CI image does not carry; they stay local-only.
"$venv/bin/python" -m pytest -q -p no:cacheprovider "${files[@]}" \
  --deselect tests/test_paper_data.py::test_figures_compile_standalone_within_column \
  --deselect tests/test_paper_evidence.py::test_label_width_estimates_cover_the_typeset_labels
