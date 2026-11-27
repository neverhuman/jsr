#!/bin/bash
# prep_feats.sh DIR : DIR/nbest.jsonl -> feats.npz (bf16 content feats), m5.npy (kenlm M5), l2r_int8.npy (engine INT8 kd12L_t2)
set -e; D=$1; PY=/home/ubuntu/jsr/.venv/bin/python; T=/home/ubuntu/jsr/.agent/worktrees/wer-claude/research
nice -n 10 $PY $T/H261_content_word_reranker/tools/feats.py $D/nbest.jsonl $D/feats.npz
PYTHONPATH=/mnt/xbabe3-external/jsr/wer-push/shared/pylib nice -n 10 $PY $T/H68_neural_lm_rescoring/tools/m5_feats.py $D/nbest.jsonl /mnt/xbabe3-external/jsr/lm/models/token/tok5.p00149.trie.bin $D/m5.npy 16 >/dev/null
python3 -c "
import json
with open('$D/cands.jsonl','w') as fh:
    for l in open('$D/nbest.jsonl'):
        r=json.loads(l); fh.write(json.dumps({'uid':r['uid'],'cands':[{'tokens':c[0]} for c in r['nbest'][:16]]})+'\n')"
nice -n 10 /mnt/xbabe3-external/jsr/cargo-target/main/release/examples/tf_rescore_nbest --nbest $D/cands.jsonl --l2r /mnt/xbabe3-external/jsr/wer-push/H68/lock_kd12l24/kd12L_t2_16k.utfr --out $D/l2r_int8.jsonl >/dev/null
python3 -c "
import json,numpy as np
np.save('$D/l2r_int8.npy',np.concatenate([json.loads(l)['l2r_tot'] for l in open('$D/l2r_int8.jsonl')]))"
echo "prep done $D"
