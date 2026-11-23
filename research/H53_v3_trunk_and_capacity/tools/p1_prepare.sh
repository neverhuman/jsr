#!/usr/bin/env bash
# H53 P1 preparation on x5 (draft by h53-trunk for trainer-lane; trainer-lane runs it, h53 does not).
# 1. a SEPARATE worktree for P1 = A1L's tree (d5f231e + trainer-a1l's local diff) + patch 0001a. Never patch trainer-a1l
#    itself: A1C's resume and every launcher pinned to 99bce400... check that tree's content hash and would refuse.
# 2. the grown seeds S-32 / S-24 from rank 0's canonical A1-49000 (2a3dd9fb...), raw "model" weights (P1 arms) and
#    optionally EMA weights (grown-at-init exports for the engine / xbabe0 check, README B3).
# 3. a CPU check on the real checkpoint: grown seed == B0 encoder output, bit for bit (random features, chunk 4).
# 4. prints the P1 tree hash, the seeds' sha256 (for MANIFEST) and the launch lines.
#
#   H53=/mnt/xbabe3-external/jsr/h53-trunk  (copy tools/ + patches/ to x5 first, e.g. ~/h53-trunk/)
#   p1_prepare.sh <h53_dir_on_this_host> [--ema]
set -euo pipefail
H=${1:?h53 dir with tools/ and patches/}; EMA=${2:-}
D=$HOME/jsr-data; REPO=$HOME/jsr; A1L=$REPO/.agent/worktrees/trainer-a1l; P1=$REPO/.agent/worktrees/trainer-h53p1
SRC=$D/runs/train/A1_S_r0bpe_960h/ckpt-00049000.pt; WANT=2a3dd9fb6c7856126b3455fb65babc4295e27b10e011d18a8806b8be5181e146
PY=$D/venvs/train/bin/python
treehash() { (cd "$1" && git ls-files -co --exclude-standard python/ultra | grep -E '\.py$' | LC_ALL=C sort \
              | xargs sha256sum | sha256sum | cut -c1-64); }

[ "$(sha256sum "$SRC" | cut -c1-64)" = "$WANT" ] || { echo "REFUSED: $SRC is not rank 0's A1-49000 ($WANT)"; exit 1; }
[ "$(treehash "$A1L")" = 99bce400e81fbd6748c11b4603950c85b7bfc730abb9d5c9909672201ce04ff9 ] \
  || { echo "REFUSED: trainer-a1l is not A1L's pinned tree"; exit 1; }

if [ ! -d "$P1" ]; then
  git -C "$REPO" worktree add --detach "$P1" d5f231e
  git -C "$A1L" diff | git -C "$P1" apply            # the same local diff as A1L's tree (kd.py, its test, 2 ops scripts)
  [ "$(treehash "$P1")" = 99bce400e81fbd6748c11b4603950c85b7bfc730abb9d5c9909672201ce04ff9 ] \
    || { echo "REFUSED: replicated tree differs from 99bce400"; exit 1; }
  git -C "$P1" apply "$H/patches/0001a-encoder-blocks-a1l-tree-d5f231e.patch"
fi
P1HASH=$(treehash "$P1")
echo "P1 tree: $P1  content sha256 (LC_ALL=C) $P1HASH  (A1L's 99bce400... + 0001a)"
(cd "$P1/python/ultra" && PYTHONPATH=$PWD timeout 300 "$PY" -m pytest -q -p no:cacheprovider tests/test_h53_encoder_blocks_a1l.py)

mkdir -p "$D/seeds/H53"
for n in 32 24; do
  for w in model ${EMA:+ema}; do
    out=$D/seeds/H53/A1-49000_grow${n}_${w}.pt
    [ -f "$out" ] || (cd "$P1/python/ultra" && PYTHONPATH=$PWD:$H/tools nice -n 10 "$PY" "$H/tools/grow_depth.py" \
                        --src "$SRC" --out "$out" --blocks "$n" --weights "$w")
    echo "seed $out sha256 $(cut -c1-64 "${out%.pt}.sha256")"
  done
done

# real-checkpoint check: the grown seed computes exactly B0's encoder output (CPU, 1 thread, ~1 min)
(cd "$P1/python/ultra" && PYTHONPATH=$PWD OMP_NUM_THREADS=1 nice -n 10 "$PY" - "$SRC" "$D/seeds/H53/A1-49000_grow32_model.pt" <<'EOF'
import sys, torch
from train import checkpoint
from train.asr import ASRModel, ModelConfig
torch.set_num_threads(1)
outs = []
for path in sys.argv[1:3]:
    p = checkpoint.load(path, restore_rng=False)
    m = ASRModel(ModelConfig.from_dict(p["model_config"])).eval()
    m.load_state_dict(p["model"])
    feats = torch.randn(1, 4 * 50 + 1, 80, generator=torch.Generator().manual_seed(53))
    with torch.no_grad():
        outs.append(m.encode(feats, torch.tensor([feats.shape[1]]), chunk=4)[0])
    print(path, "blocks", m.encoder.cfg.n_blocks)
same = torch.equal(outs[0], outs[1])
print("grown == B0 encoder output (chunk 4):", same, "max |diff|", float((outs[0] - outs[1]).abs().max()))
sys.exit(0 if same else 1)
EOF
)
cat <<EOF

Launch lines (A1L's launcher, P1's tree; one whole A100 per arm; control = A1C):
  cd $P1 && INIT_FROM=$D/seeds/H53/A1-49000_grow32_model.pt RUN_NAME=H53_P1_G32_s1234 \\
    python/ultra/train/ops/h41_replay.sh $D <cpus> <gpu> P1_G32 chunk_modes=4,8,16 eval_chunks=8,16 delay_penalty=0 encoder_blocks=32
  cd $P1 && INIT_FROM=$D/seeds/H53/A1-49000_grow24_model.pt RUN_NAME=H53_P1_G24_s1234 \\
    python/ultra/train/ops/h41_replay.sh $D <cpus> <gpu> P1_G24 chunk_modes=4,8,16 eval_chunks=8,16 delay_penalty=0 encoder_blocks=24
Pin for both: tree $P1HASH. Kill rule: step-2,000 dev-clean400 EMA chunk 4 > 6.163, or raw > 20%.
EOF
