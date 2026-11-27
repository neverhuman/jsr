#!/usr/bin/env bash
# Build the arXiv submission package for the Isosyll paper.
#   ops/paper/build_arxiv.sh              full build, container verify, writes dist/arxiv and docs/paper/arxiv
#   ops/paper/build_arxiv.sh --check      no docker; fail if docs/paper/arxiv/{isosyll-arxiv.zip,ARXIV_UPLOAD.md} are stale
#   ops/paper/build_arxiv.sh --stage-only | --no-docker
# Inputs are discovered dynamically (pdflatex -recorder .fls), TikZ figures are precompiled to PDF via
# the TikZ external library, comments are stripped, then the staged tree is rebuilt in a clean
# texlive/texlive:TL2023-historic container and compared with the repo build (pages + pdftotext).
# Outputs: target/arxiv/src (staged), dist/arxiv/isosyll-arxiv-v4.2.0.{tar.gz,zip,pdf}, ARXIV_METADATA.md, SHA256SUMS.
set -euo pipefail

VERSION="${ARXIV_VERSION:-v4.2.0}"
IMAGE="texlive/texlive:TL2023-historic"
DOCKER=1; STAGE_ONLY=0; CHECK=0
for arg in "$@"; do
  case "$arg" in
    --no-docker) DOCKER=0 ;;
    --check) CHECK=1; DOCKER=0 ;;
    --stage-only) STAGE_ONLY=1; DOCKER=0 ;;
    *) echo "usage: $0 [--no-docker] [--stage-only] [--check]" >&2; exit 2 ;;
  esac
done

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
HELPER="python3 -I $ROOT/python/ultra/tools/paper_data/arxiv_stage.py"
A="$ROOT/target/arxiv"
WORK="$A/work"; EXT="$A/work-ext"; SRC="$A/src"; VER="$A/verify"
DIST="$ROOT/dist/arxiv"; COMMITTED="$ROOT/docs/paper/arxiv"; NAME="isosyll-arxiv-$VERSION"
# fixed epoch (1980-01-01): regenerating the zip is byte-stable regardless of the commit date
export SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH:-315532800}" FORCE_SOURCE_DATE=1
fail() { echo "arxiv: FAIL: $*" >&2; exit 1; }
latex() { # latex <dir> [extra flags]; one pdflatex pass on paper
  local d="$1"; shift
  (cd "$d" && pdflatex -interaction=nonstopmode -halt-on-error -file-line-error "$@" paper >/dev/null) \
    || { tail -40 "$d/paper.log" >&2; fail "pdflatex failed in $d"; }
}
check_log() { # check_log <log>
  if grep -nE "undefined references|Citation .* undefined|Reference .* undefined|LaTeX Warning: File .* not found|! LaTeX Error|No file paper.bbl" "$1"; then
    fail "undefined references/citations or missing files in $1"; fi
}

# --- 1. fresh scratch copy of the paper tree (heavy evidence dirs are never read by the build)
rm -rf "$WORK" "$EXT" "$SRC" "$VER"; mkdir -p "$WORK" "$EXT" "$SRC" "$VER" "$DIST"
copy_tree() {
  tar -C docs/paper --exclude=./archive --exclude=./opponents --exclude=./data/raw --exclude=./data/evidence \
      --exclude=./paper.pdf --exclude=./money.png -cf - . | tar -C "$1" -xf -
}
copy_tree "$WORK"

# --- 2. hero figure: ship figures/hero.pdf; compile it standalone if the paper wants it and it is missing
if grep -rqE "hero\.pdf|figures/hero" "$WORK/paper.tex" "$WORK/sections" "$WORK/figures" 2>/dev/null && [ ! -f "$WORK/figures/hero.pdf" ]; then
  echo "arxiv: figures/hero.pdf missing; compiling standalone"
  for s in hero_standalone.tex hero.standalone.tex; do
    if [ -f "$WORK/figures/$s" ]; then
      (cd "$WORK" && pdflatex -interaction=nonstopmode -halt-on-error -output-directory figures "figures/$s" >/dev/null) \
        || fail "hero standalone failed"
      mv "$WORK/figures/${s%.tex}.pdf" "$WORK/figures/hero.pdf"; break
    fi
  done
  [ -f "$WORK/figures/hero.pdf" ] || fail "hero.pdf is referenced but neither shipped nor buildable"
fi

# --- 3. repo-equivalent build: pdflatex, bibtex, pdflatex x2 (same as `just paper`)
latex "$WORK"; (cd "$WORK" && bibtex paper >/dev/null) || fail "bibtex failed"; latex "$WORK"; latex "$WORK" -recorder
check_log "$WORK/paper.log"
cp "$WORK/paper.pdf" "$A/repo-build.pdf"

# --- 4. precompile TikZ figures: each TeX-drawn float body is wrapped in \beginpgfgraphicnamed and built
#        in its own job (pgf manual mode), so the staged paper only includes finished PDFs
copy_tree "$EXT"; cp "$WORK/paper.bbl" "$EXT/"
grep -q '^\\begin{document}' "$EXT/paper.tex" || fail "paper.tex has no \\begin{document} line"
sed -i 's|^\\begin{document}|\\usetikzlibrary{external}\\tikzexternalize\\tikzset{external/up to date check=simple}\\tikzexternaldisable\n\\begin{document}|' "$EXT/paper.tex"
$HELPER wrapfigs --dir "$EXT" --fls "$WORK/paper.fls" --names "$A/figures.txt"
latex "$EXT"   # main pass writes the aux; figures not built yet are typeset inline
while read -r fig; do
  [ -n "$fig" ] || continue
  (cd "$EXT" && pdflatex -interaction=nonstopmode -halt-on-error -file-line-error -jobname "$fig" \
      "\\def\\tikzexternalrealjob{paper}\\input{paper}" >/dev/null) || { tail -40 "$EXT/$fig.log" >&2; fail "figure $fig failed"; }
  [ -s "$EXT/$fig.pdf" ] || fail "figure $fig produced no PDF"
done < "$A/figures.txt"
latex "$EXT"; latex "$EXT" -recorder
check_log "$EXT/paper.log"
[ -s "$EXT/paper.fls" ] || fail "recorder output missing"

# --- 5. stage only what was read, comments stripped, 00README.json
$HELPER stage --fls "$EXT/paper.fls" --work "$EXT" --out "$SRC"
if grep -rnE '(^|[^[:alnum:]])/(home|mnt|tmp|Users|root)/|[A-Za-z]:\\\\' "$SRC" --include='*.tex' --include='*.bbl' --include='*.json'; then
  fail "absolute path in staged sources"; fi
$HELPER scan "$SRC" || fail "privacy scan of staged sources"
[ "$STAGE_ONLY" = 1 ] && { echo "arxiv: staged at $SRC (stage-only)"; exit 0; }

# --- packaging helpers (deterministic: sorted names, mtime 0 / 1980, owner 0, no gzip name or time)
FILES="$VER/files.txt"
(cd "$SRC" && find . -type f | sed 's|^\./||' | LC_ALL=C sort) > "$FILES"
make_tar() {
  tar -C "$SRC" --no-recursion --files-from="$FILES" --mtime=@0 --owner=0 --group=0 --numeric-owner \
      --mode='u=rw,go=r' --format=gnu -cf - | gzip -n -9 > "$1"
}
make_zip() {
python3 -I - "$SRC" "$FILES" "$1" <<'PY'
import sys, zipfile
src, lst, out = sys.argv[1:]
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for name in open(lst).read().split("\n"):
        if not name: continue
        zi = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))  # zip cannot store mtime 0
        zi.compress_type = zipfile.ZIP_DEFLATED; zi.external_attr = 0o644 << 16; zi.create_system = 3
        z.writestr(zi, open(f"{src}/{name}", "rb").read())
PY
}
write_meta() { # write_meta <pdf for page count> <out>
META="$($HELPER metadata --paper "$EXT" --pdf "$1" --aux "$EXT/paper.aux")"
export META
python3 -I - > "$2" <<'PY'
import json, os
m = json.loads(os.environ["META"])
assert m["abstract_len"] <= 1920, f"abstract {m['abstract_len']} chars exceeds arXiv's 1920"
assert not m["leftover_markup"], f"TeX markup left in abstract: {m['leftover_markup']}"
print(f"""# arXiv upload: {m["title"].split(":")[0]}

Generated by `ops/paper/build_arxiv.sh`; do not edit by hand. Source to upload: `isosyll-arxiv.zip` (this folder).

## Title

{m['title']}

## Authors (byline order)

{'; '.join(m['authors'])}

## Abstract (plain text, {m['abstract_len']} of 1920 characters)

{m['abstract']}

## Comments

{m['pages']} pages, {m['figures']} figures

## Classification

- Primary: eess.AS (Audio and Speech Processing)
- Cross-lists: cs.SD, cs.CL, cs.LG
- License (suggested): CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/)

## What the zip is

Sources only: `paper.tex` (top level, `\\pdfoutput=1`), the `.tex` files it reads with comments stripped, `paper.bbl`
(arXiv does not run BibTeX), precompiled figure PDFs, and `00README.json` (pdflatex, toplevel `paper.tex`).
It was rebuilt in a clean `texlive/texlive:TL2023-historic` container with zero errors, no undefined references,
and the same page count and text as the repository build.

## Upload steps

1. Sign in at https://arxiv.org/user and choose Submit, then Start New Submission.
2. Select the license (CC BY 4.0), the primary category eess.AS, and cross-lists cs.SD, cs.CL, cs.LG.
3. Upload `isosyll-arxiv.zip` as the source file. Do not upload a PDF.
4. Wait for processing. Open the preview PDF: it should have the page count above, all figures, and no red "??" marks.
5. Fill the metadata form from this file: title, authors, abstract (paste the plain text above), comments.
6. Preview, read the processing log (an unused-file warning is harmless), then Submit.
7. After announcement, record the arXiv identifier in the repository README.
""", end="")
PY
}

# --- check mode: committed artifacts must equal a fresh build from the current sources
if [ "$CHECK" = 1 ]; then
  make_zip "$VER/fresh.zip"; write_meta "$A/repo-build.pdf" "$VER/fresh.md"
  cmp -s "$VER/fresh.zip" "$COMMITTED/isosyll-arxiv.zip" || fail "docs/paper/arxiv/isosyll-arxiv.zip is stale; run ops/paper/build_arxiv.sh and commit"
  cmp -s "$VER/fresh.md" "$COMMITTED/ARXIV_UPLOAD.md" || fail "docs/paper/arxiv/ARXIV_UPLOAD.md is stale; run ops/paper/build_arxiv.sh and commit"
  echo "arxiv: committed package is current"; exit 0
fi

# --- 6. clean rebuild of the staged tree the way arXiv does it: pdflatex only, no BibTeX, no shell escape
mkdir -p "$VER/out"
VERIFY_SH='set -e; cp -r /src /tmp/build; cd /tmp/build; export SOURCE_DATE_EPOCH='"$SOURCE_DATE_EPOCH"' FORCE_SOURCE_DATE=1
for i in 1 2 3; do pdflatex -interaction=nonstopmode -halt-on-error -file-line-error -no-shell-escape paper >/dev/null || { tail -40 paper.log; exit 1; }; done
cp paper.pdf paper.log /out/'
if [ "$DOCKER" = 1 ]; then
  docker run --rm --network none --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$SRC:/src:ro" -v "$VER/out:/out" "$IMAGE" bash -c "$VERIFY_SH" \
    || fail "container build failed"
  REF="$VER/out/paper.pdf"; LOG="$VER/out/paper.log"
else
  cp -r "$SRC" "$VER/local"; for _ in 1 2 3; do latex "$VER/local" -no-shell-escape; done
  REF="$VER/local/paper.pdf"; LOG="$VER/local/paper.log"
fi
check_log "$LOG"
if grep -nE "^! |Emergency stop|Fatal error|No file " "$LOG"; then fail "errors in verify build"; fi
pages() { pdfinfo "$1" | awk '/^Pages/ {print $2}'; }
P_REPO="$(pages "$A/repo-build.pdf")"; P_ARX="$(pages "$REF")"
[ "$P_REPO" = "$P_ARX" ] || fail "page count differs: repo $P_REPO vs arxiv $P_ARX"
pdftotext "$A/repo-build.pdf" "$VER/repo.txt"; pdftotext "$REF" "$VER/arxiv.txt"
if ! diff -q "$VER/repo.txt" "$VER/arxiv.txt" >/dev/null; then
  # A precompiled figure is one PDF object, so pdftotext may emit its labels in another order.
  # Accept that only if every page still holds exactly the same multiset of words.
  for ((p = 1; p <= P_ARX; p++)); do
    diff <(pdftotext -f $p -l $p "$A/repo-build.pdf" - | tr -s '[:space:]' '\n' | LC_ALL=C sort) \
         <(pdftotext -f $p -l $p "$REF" - | tr -s '[:space:]' '\n' | LC_ALL=C sort) > "$VER/text.diff" \
      || { head -40 "$VER/text.diff" >&2; fail "page $p text differs (words, not just order)"; }
  done
  echo "arxiv: pdftotext order differs only inside precompiled figures; per-page word multisets are identical"
fi
echo "arxiv: verified: $P_ARX pages, no errors, no undefined refs, text matches repo build"


# --- 7. package: dist/ (tar.gz, zip, reference PDF, sums) and the committed copy in docs/paper/arxiv/
NAME="isosyll-arxiv-$VERSION"
rm -f "$DIST/$NAME".* "$DIST/ARXIV_METADATA.md" "$DIST/ARXIV_UPLOAD.md" "$DIST/SHA256SUMS"
make_tar "$DIST/$NAME.tar.gz"; make_zip "$DIST/$NAME.zip"; cp "$REF" "$DIST/$NAME.pdf"
write_meta "$REF" "$DIST/ARXIV_UPLOAD.md"
mkdir -p "$COMMITTED"; cp "$DIST/$NAME.zip" "$COMMITTED/isosyll-arxiv.zip"; cp "$DIST/ARXIV_UPLOAD.md" "$COMMITTED/ARXIV_UPLOAD.md"

# --- 8. privacy scan on everything shipped (zip contents, committed files, PDF text and metadata)
rm -rf "$VER/unzip"; mkdir -p "$VER/unzip" && python3 -I -m zipfile -e "$COMMITTED/isosyll-arxiv.zip" "$VER/unzip"
$HELPER scan "$VER/unzip" "$DIST/$NAME.pdf" "$COMMITTED/ARXIV_UPLOAD.md" || fail "privacy scan hit; remove docs/paper/arxiv outputs before committing"
(cd "$DIST" && sha256sum "$NAME.tar.gz" "$NAME.zip" "$NAME.pdf" ARXIV_UPLOAD.md > SHA256SUMS)
ls -l "$DIST" "$COMMITTED"
echo "arxiv: package ready in $DIST; committed copy in $COMMITTED"
