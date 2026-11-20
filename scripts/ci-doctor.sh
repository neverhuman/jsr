#!/usr/bin/env bash
# List every tool the ops/ci lanes depend on, with the found and pinned versions.
# Exit 1 when a required tool is missing; a version drift is reported, not fatal.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
# shellcheck source=ops/ci/lib.sh
. ops/ci/lib.sh

missing=0
check() {
  # check <name> <pinned-or-dash> <version command...>
  local name="$1" pin="$2"
  shift 2
  if ! command -v "$1" >/dev/null 2>&1; then
    printf '%-12s MISSING  (pin %s)\n' "$name" "$pin"
    missing=1
    return
  fi
  local found
  found="$("$@" 2>&1 | head -n 3 | tr '\n' ' ')"
  if [ "$pin" != "-" ] && [[ "$found" != *"$pin"* ]]; then
    printf '%-12s DRIFT    pin %s, found: %s\n' "$name" "$pin" "$found"
  else
    printf '%-12s ok       %s\n' "$name" "$found"
  fi
}

check git          -                              git --version
check just         -                              just --version
check cargo        -                              cargo --version
check rustfmt      -                              rustfmt --version
check clippy       -                              cargo clippy --version
check gitleaks     "$JSR_GITLEAKS_VERSION"        gitleaks version
check cargo-audit  "$JSR_CARGO_AUDIT_VERSION"     cargo audit --version
check cyclonedx    "$JSR_CARGO_CYCLONEDX_VERSION" cargo cyclonedx --version
check actionlint   "$JSR_ACTIONLINT_VERSION"      actionlint -version
check uv           -                              uv --version
check shellcheck   "$JSR_SHELLCHECK_VERSION"      shellcheck --version
jk="${JANKURAI:-$HOME/.jeryu/bin176/jankurai}"
if [ -x "$jk" ]; then
  printf '%-12s ok       %s\n' jankurai "$("$jk" version 2>&1 | head -n 1)"
else
  printf '%-12s MISSING  (%s; set JANKURAI)\n' jankurai "$jk"
  missing=1
fi
if [ -x .venv/bin/python ]; then
  printf '%-12s ok       %s\n' python "$(.venv/bin/python --version 2>&1)"
else
  printf '%-12s optional no .venv (just venv) - pytest lane skips\n' python
fi
exit "$missing"
