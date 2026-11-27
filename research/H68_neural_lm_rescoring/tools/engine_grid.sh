#!/bin/bash
# engine_grid.sh TAG "w s" ... : native engine full S per weight pair (serial, 24 niced shards each), then speaker 2-fold CV over the grid
H=/mnt/xbabe3-external/jsr/wer-push/H68; cd $H; G=$1; shift
for ws in "$@"; do set -- $ws; N=${G}_w$1_s$2; [ -f engine_fullS/$N.done ] && continue
 ssh -o BatchMode=yes xbabe0 "cd ~/wer-push-claude/lock && for k in \$(seq -w 0 23); do nice -n 10 ./ultra-engine-commit --model model --commit-chunk 16 --tail-pad-ms 320 --commit-beam 16 --commit-lm final.ulm --lm-scale 0.8 --lodr-scale 0.6 --threads 1 --precision int8 --rescore-skip-margin 2.762935630679131 --rescore-l2r $G.utfr --rescore-weight $1 --rescore-m5-sub $2 --manifest s_shards/s\$k.npz --out fullS/${N}_\$k.jsonl >/dev/null 2>&1 & done; wait"
 rsync -a "xbabe0:wer-push-claude/lock/fullS/${N}_*" engine_fullS/ && touch engine_fullS/$N.done; echo "$N done $(TZ=America/Denver date +%H:%M)"
done
