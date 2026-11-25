# JSR / ULTRA task runner. `just required` is the push gate (HOST_RULES).
set shell := ["bash", "-euo", "pipefail", "-c"]
export CARGO_INCREMENTAL := "0"
export CARGO_PROFILE_DEV_DEBUG := "line-tables-only"
jk := env_var_or_default("JANKURAI", env_var("HOME") + "/.jeryu/bin176/jankurai")
py := ".venv/bin/python"

default: fast

# Deterministic local health proof.
fast: fmt-check clippy test pytest

fmt-check:
    cargo fmt --all -- --check

clippy:
    cargo clippy --locked --workspace --all-targets -- -D warnings

test:
    cargo test --locked --workspace -q

# Narrow loop for one crate: `just test-crate ultra-core`.
test-crate crate:
    cargo clippy -p {{crate}} --all-targets -- -D warnings
    cargo test -p {{crate}} -q

# Without a full .venv (torch etc.), run at least the torch-free CI lane instead of skipping silently:
# Opt-in CUDA restart witness (C2); run only on reserved CUDA devices. Not part of `required`.
cuda-witness:
    #!/usr/bin/env bash
    set -euo pipefail
    pybin="$PWD/{{py}}"; [ -x "$pybin" ] || pybin="$(git rev-parse --git-common-dir)/../{{py}}"
    cd python/ultra && "$pybin" -m pytest -q -p no:cacheprovider tests_cuda

# a gate that skipped pytest let main go red twice on 2026-11-22/23.
pytest:
    if [ -x {{py}} ]; then {{py}} -m pytest -q python/ultra/tests; else echo "pytest: no .venv; running the torch-free CI lane (ops/ci/pytest-light.sh) instead"; bash ops/ci/pytest-light.sh; fi

venv:
    uv venv --python 3.12 .venv
    uv pip install --python {{py}} -r python/ultra/requirements.txt

# Jankurai score artifacts (full audit).
score:
    mkdir -p target/jankurai
    {{jk}} audit . --mode advisory --no-badge --json target/jankurai/repo-score.json --md target/jankurai/repo-score.md

# Audit and keep a dated copy under audit/ (committed).
audit: score
    mkdir -p audit
    cp target/jankurai/repo-score.md audit/jankurai-$(git rev-parse --short HEAD).md

# Ledger contract tests plus serialised ultra-core values against contracts/events schemas.
contract-drift:
    bash ops/ci/contract-drift.sh

security:
    bash tools/security-lane.sh

# CycloneDX SBOMs for release provenance -> target/sbom/.
sbom:
    bash ops/ci/sbom.sh

# Build stripped, static `isosyll` release archives into dist/ (ops/release/build.sh).
release-build *TRIPLES:
    bash ops/release/build.sh {{TRIPLES}}

# Build the platform-independent model archive into dist/ (ops/release/model.sh).
release-model BUNDLE LIVE FINAL:
    bash ops/release/model.sh {{BUNDLE}} {{LIVE}} {{FINAL}}

# Report every tool the lanes need, with pinned versions (ops/ci/lib.sh).
doctor:
    bash scripts/ci-doctor.sh

# Full merge gate: fast, pytest, contract-drift, security, score (ops/ci/quality-gates.sh).
check:
    bash ops/ci/quality-gates.sh

required: check

# Build the IEEE paper: regenerate metrics/figures (paper_data), then pdflatex + bibtex + pdflatex x2.
# Output docs/paper/paper.pdf (committed). Fails on undefined references or citations, any overfull
# hbox, non-Type-1 fonts, or a missing "Fig. 1." on page 1. SOURCE_DATE_EPOCH pins PDF dates.
paper:
    #!/usr/bin/env bash
    set -euo pipefail
    root="$PWD"
    if [ -e python/ultra/tools/paper_data/__init__.py ]; then
        pybin="$root/{{py}}"; [ -x "$pybin" ] || pybin="$(git rev-parse --git-common-dir)/../{{py}}"
        (cd python/ultra && "$pybin" -m tools.paper_data)
    fi
    export SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH:-$(git log -1 --format=%ct)}" FORCE_SOURCE_DATE=1
    cd docs/paper
    run() { pdflatex -interaction=nonstopmode -halt-on-error paper > /dev/null || { tail -40 paper.log; exit 1; }; }
    run; bibtex paper > /dev/null || { cat paper.blg; exit 1; }; run; run
    fail=0
    if grep -nE "undefined references|Citation .* undefined|Reference .* undefined" paper.log; then echo "paper: undefined references or citations"; fail=1; fi
    if grep -n "Overfull \\\\hbox" paper.log; then echo "paper: overfull hbox"; fail=1; fi
    if pdffonts paper.pdf | tail -n +3 | grep -v "Type 1" | grep .; then echo "paper: non-Type-1 font"; fail=1; fi
    p1="$(pdftotext -f 1 -l 1 paper.pdf -)"; if ! grep -q "Fig. 1\." <<< "$p1"; then echo "paper: Fig. 1. not on page 1"; fail=1; fi
    [ "$fail" = 0 ] || exit 1
    echo "paper: docs/paper/paper.pdf ($(pdfinfo paper.pdf | awk '/^Pages/ {print $2}') pages)"

# One command to refresh the paper from docs/paper/data/evidence/*.jsonl: regenerate the data and
# Fig. 1, rebuild the PDF (just paper), export Fig. 1 as docs/paper/money.png, copy its block into
# README.md, stamp the PNG with the hash of its inputs, and run the paper tests.
paper-refresh: paper
    #!/usr/bin/env bash
    set -euo pipefail
    root="$PWD"
    pybin="$root/{{py}}"; [ -x "$pybin" ] || pybin="$(git rev-parse --git-common-dir)/../{{py}}"
    out="$root/target/paper-figure"; mkdir -p "$out"
    (cd docs/paper && pdflatex -interaction=nonstopmode -halt-on-error -output-directory "$out" figures/money_standalone.tex > /dev/null) \
        || { tail -30 "$out/money_standalone.log"; exit 1; }
    pdftoppm -r 220 -png -singlefile "$out/money_standalone.pdf" docs/paper/money
    (cd python/ultra && "$pybin" -m tools.paper_data --publish)
    (cd python/ultra && "$pybin" -m pytest -q tests/test_paper_data.py tests/test_paper_evidence.py tests/test_paper_claims.py)
    echo "paper-refresh: docs/paper/paper.pdf, docs/paper/money.png and the README block are current"

# Build the two-page CEO and CTO briefs (docs/briefs/isosyll-v4-*-brief.pdf, committed). Numbers come
# from docs/paper/data/metrics.tex. Fails unless each PDF is exactly 2 pages with Type 1 fonts only
# and no overfull boxes. `just briefs --check` verifies the committed PDFs are current.
briefs *args:
    bash docs/briefs/scripts/build_briefs.sh {{args}}

# Real-time fleet dashboard (loss / WER / throughput per run, GPU + CPU per node) on 127.0.0.1:8765.
# View remotely: ssh -L 8765:127.0.0.1:8765 <this host>, then open http://localhost:8765
dashboard:
    cd python/ultra && ../../.venv/bin/python -m tools.fleet_dashboard
