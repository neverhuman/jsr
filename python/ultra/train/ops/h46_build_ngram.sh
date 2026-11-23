#!/usr/bin/env bash
# H46 item 3 (a): one count-pruned token n-gram from the leak-free token stream: lmplz (modified
# Kneser-Ney, the H42 settings) -> ARPA -> the KenLM trie the Python beam queries (H42's
# build_binary flags) and the ultra.lm-ngram/v2 pack the engine loads (with or without LODR).
#
#   KENLM=<dir with lmplz, build_binary> PACK=<ultra-lm-pack> CODE=<python/ultra> \
#   h46_build_ngram.sh <name> <order> "<prune thresholds>" <train.u16> <out dir> [lodr.arpa]
set -euo pipefail
NAME=${1:?name}; ORDER=${2:?order}; PRUNE=${3:?prune}; STREAM=${4:?train.u16}; OUT=${5:?out}
LODR=${6:-}
: "${KENLM:?}" "${PACK:?}" "${CODE:?}"
PY=${PY:-/home/ubuntu/jsr/.venv/bin/python}
mkdir -p "$OUT/tmp.$NAME"
# shellcheck disable=SC2086  # PRUNE is a list of thresholds
(cd "$CODE" && "$PY" -m train.ops.h46_u16_to_text "$STREAM") \
  | "$KENLM/lmplz" -o "$ORDER" --prune $PRUNE --discount_fallback -S "${MEM:-4G}" \
      -T "$OUT/tmp.$NAME" > "$OUT/$NAME.arpa" 2> "$OUT/$NAME.lmplz.log"
rm -rf "$OUT/tmp.$NAME"
"$KENLM/build_binary" -a 22 -q 8 -b 8 trie "$OUT/$NAME.arpa" "$OUT/$NAME.trie.bin" \
  > "$OUT/$NAME.bb.log" 2>&1
if [ -n "$LODR" ]; then
  "$PACK" --arpa "$OUT/$NAME.arpa" --lodr "$LODR" --vocab 501 --out "$OUT/$NAME.lodr.v2.ulm" \
    > "$OUT/$NAME.pack.log" 2>&1
fi
"$PACK" --arpa "$OUT/$NAME.arpa" --vocab 501 --out "$OUT/$NAME.v2.ulm" >> "$OUT/$NAME.pack.log" 2>&1
gzip -f "$OUT/$NAME.arpa"
echo "$NAME built: $(date -u +%H:%M:%SZ) $(head -12 "$OUT/$NAME.lmplz.log" | grep -c .) log lines"
