#!/usr/bin/env bash
# Deterministically build (or --check) the Isosyll v4.0.0 CEO and CTO PDF briefs.
#
# Run from the repository root (or via `just briefs`). Every measured number is a macro from
# docs/paper/data/metrics.tex, which the briefs \input directly. The build fails on:
#   - anything other than exactly 2 pages per brief
#   - undefined references or citations
#   - any Overfull \hbox or \vbox
#   - any embedded font that is not Type 1
#
# Usage:
#   bash docs/briefs/scripts/build_briefs.sh          # build & copy PDFs to docs/briefs/
#   bash docs/briefs/scripts/build_briefs.sh --check  # rebuild in a scratch dir and verify only
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
SRC_DIR="$REPO_ROOT/docs/briefs/src"
OUT_PUB="$REPO_ROOT/docs/briefs"
CHECK_MODE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --check) CHECK_MODE=1; shift ;;
    -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
    *) echo "error: unknown option: $1" >&2; exit 2 ;;
  esac
done

# Fixed epoch (2026-11-23T00:00:00Z) so rebuilds are byte-stable.
export SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH:-1795392000}"
export FORCE_SOURCE_DATE=1

if [[ "$CHECK_MODE" -eq 1 ]]; then
  WORK_DIR="$REPO_ROOT/target/briefs-check"
  rm -rf "$WORK_DIR"
else
  WORK_DIR="$REPO_ROOT/target/briefs"
fi
mkdir -p "$WORK_DIR"

compile_one() {
  local tex_stem="$1"
  local pdf_name="$2"
  local build_dir="$WORK_DIR/$tex_stem"
  mkdir -p "$build_dir"

  local pass
  for pass in 1 2; do
    if ! (
      cd "$SRC_DIR"
      pdflatex -interaction=nonstopmode -halt-on-error -file-line-error \
        -output-directory="$build_dir" \
        '\pdftrailerid{}\pdfsuppressptexinfo=-1\input{'"$tex_stem"'.tex}' >/dev/null
    ); then
      echo "error: pdflatex failed for $tex_stem.tex (pass $pass)" >&2
      tail -40 "$build_dir/$tex_stem.log" >&2 || true
      exit 1
    fi
  done

  local log_file="$build_dir/$tex_stem.log"
  local pdf_file="$build_dir/$tex_stem.pdf"

  if grep -E "LaTeX Warning: (Reference|Citation) .* undefined|There were undefined references" "$log_file" >&2; then
    echo "error: unresolved references or citations in $tex_stem.tex" >&2
    exit 1
  fi
  if grep -E '^(Overfull|Underfull \\vbox)' "$log_file" | grep -E 'Overfull' >&2; then
    echo "error: overfull box in $tex_stem.tex" >&2
    exit 1
  fi
  if pdffonts "$pdf_file" | tail -n +3 | grep -v "Type 1" | grep . >&2; then
    echo "error: non-Type-1 font in $tex_stem.pdf" >&2
    exit 1
  fi

  local pages
  pages="$(pdfinfo "$pdf_file" | awk '/^Pages:/ {print $2}')"
  if [[ "$pages" != "2" ]]; then
    echo "error: expected exactly 2 pages for $pdf_name, got $pages" >&2
    exit 1
  fi

  if [[ "$CHECK_MODE" -eq 1 ]]; then
    if [[ ! -f "$OUT_PUB/$pdf_name" ]]; then
      echo "error: committed PDF missing: docs/briefs/$pdf_name" >&2
      exit 1
    fi
    if ! cmp -s "$pdf_file" "$OUT_PUB/$pdf_name"; then
      echo "error: docs/briefs/$pdf_name is stale; run: just briefs" >&2
      exit 1
    fi
    echo "OK: docs/briefs/$pdf_name is current (2 pages, Type 1 fonts, no overfull boxes)"
  else
    cp "$pdf_file" "$OUT_PUB/$pdf_name"
    echo "Built docs/briefs/$pdf_name ($pages pages, $(wc -c < "$OUT_PUB/$pdf_name") bytes)"
  fi
}

compile_one "ceo-brief" "isosyll-v4-ceo-brief.pdf"
compile_one "cto-brief" "isosyll-v4-cto-brief.pdf"
