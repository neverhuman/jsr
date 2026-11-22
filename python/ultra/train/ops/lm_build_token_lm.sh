#!/usr/bin/env bash
# H42: token-level n-gram LMs for first-pass shallow fusion (owner ruling O17).
#
#   lm_build_token_lm.sh <work-dir> text            tokenise the LM text and the training transcripts
#   lm_build_token_lm.sh <work-dir> lodr            token bigram from the 960 h transcripts (LODR)
#   lm_build_token_lm.sh <work-dir> lm <order> <prune...>   e.g. lm 5 0 0 1 2 3
#   lm_build_token_lm.sh <work-dir> binary <arpa> [build_binary options]
#
# <work-dir> must sit on a fast POSIX disk with room for lmplz's sort files (tens of GB for order 5
# on ~1.6 billion tokens) and holds:
#   dl/librispeech-lm-norm.txt.gz.d/librispeech-lm-norm.txt.gz   OpenSLR 11, sha256-locked
#   tools/bpe.model                       the acoustic model's SentencePiece model
#   tools/librispeech_train_text.txt      960 h TRAINING transcripts only (train.tokenizer writes it)
#   tools/kenlm-bin/{lmplz,build_binary}  from lm_build_kenlm.sh
#   code/                                 a copy of python/ultra (train/, ultra/)
# Tokens are written as decimal output indices (see train.lm_fusion). With 500 tokens the unigram
# counts are too regular for KenLM's discount estimate, hence --discount_fallback. Pin CPUs from
# outside (taskset); LM_MEM caps lmplz's memory.
set -euo pipefail
W=${1:?work dir}
CMD=${2:?text|lodr|lm|binary}
shift 2
PY=${PY:-python3}
LM_MEM=${LM_MEM:-9G}
mkdir -p "$W/out" "$W/tmp" "$W/logs"
tokens() { (cd "$W/code" && "$PY" -m train.lm_decode tokens --tokenizer "$W/tools/bpe.model" "$@"); }
case "$CMD" in
  text)
    zcat "$W/dl/librispeech-lm-norm.txt.gz.d/librispeech-lm-norm.txt.gz" \
      | tokens --check 200000 > "$W/out/lmtext.tok.txt" 2> "$W/logs/tokens.lmtext.log"
    tokens < "$W/tools/librispeech_train_text.txt" > "$W/out/train960.tok.txt" \
      2> "$W/logs/tokens.train960.log"
    (cd "$W/out" && sha256sum lmtext.tok.txt train960.tok.txt)
    ;;
  lodr)
    "$W/tools/kenlm-bin/lmplz" -o 2 -S 2G -T "$W/tmp/" --discount_fallback \
      < "$W/out/train960.tok.txt" > "$W/out/lodr2.train960.arpa" 2> "$W/logs/lmplz.lodr2.log"
    ;;
  lm)
    order=${1:?order}
    shift
    tag="tok$order.p$(printf '%s' "$@")"
    mkdir -p "$W/tmp/$tag"
    "$W/tools/kenlm-bin/lmplz" -o "$order" -S "$LM_MEM" -T "$W/tmp/$tag/" --discount_fallback \
      --prune "$@" < "$W/out/lmtext.tok.txt" > "$W/out/$tag.arpa" 2> "$W/logs/lmplz.$tag.log"
    ;;
  binary)
    arpa=${1:?arpa file}
    shift
    "$W/tools/kenlm-bin/build_binary" -T "$W/tmp/bb" -S 4G "$@" "$arpa" "${arpa%.arpa}.bin"
    ;;
  *)
    echo "unknown command $CMD" >&2
    exit 2
    ;;
esac
