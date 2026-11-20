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
    cargo clippy --workspace --all-targets -- -D warnings

test:
    cargo test --workspace -q

pytest:
    if [ -x {{py}} ]; then {{py}} -m pytest -q research/ultra/tests; else echo "pytest: no .venv yet (run just venv)"; fi

venv:
    uv venv --python 3.12 .venv
    uv pip install --python {{py}} -r research/ultra/requirements.txt

# Jankurai score artifacts (full audit).
score:
    mkdir -p target/jankurai
    {{jk}} audit . --mode advisory --no-badge --json target/jankurai/repo-score.json --md target/jankurai/repo-score.md

# Audit and keep a dated copy under audit/ (committed).
audit: score
    mkdir -p audit
    cp target/jankurai/repo-score.md audit/jankurai-$(git rev-parse --short HEAD).md

contract-drift:
    cargo test -p ultra-core --test output_contract -q

security:
    tools/security-lane.sh

# Full merge gate.
check: fast contract-drift security score

required: check
