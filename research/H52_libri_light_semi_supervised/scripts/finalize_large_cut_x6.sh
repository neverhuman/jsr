#!/usr/bin/env bash
# H52 large tier-A CUT on x6 (lead 14:1xZ): freeze a symlink snapshot of every closed raw shard (+ its per-shard v1.1
# labels when present) and every published from-source unit, then ll_finalize_par.py with medium's checks (v1.1-only
# policy, same filter / seed / amendments 2-4 incl. the NaN guard, --allow-unlabeled, disk guard). Writes the first M
# buckets that fit <max GB> (each a uniform random sample of the kept rows); labelling continues meanwhile.
# usage (on x6): finalize_large_cut_x6.sh <subset, e.g. large.s3> <max final GB> <workers>
set -euo pipefail
L=$HOME/jsr-data/librilight; SUB=$1; GB=${2:-300}; W=${3:-8}
echo $$ > $L/logs/finalize_large_cut.pid
SRC=$L/shards/raw/librilight.large; ST=$L/shards/raw/librilight.${SUB}_inputs; OUT=$L/shards/r0_v11
test ! -e $ST && test ! -e $OUT/librilight.$SUB
mkdir -p $ST/r0src
n=0; m=0
for r in $SRC/[0-9]*.ready; do
  k=$(basename $r .ready)
  for e in parquet ready ids; do ln -s $SRC/$k.$e $ST/$k.$e; done
  if [ -e $SRC/$k.labeled ]; then ln -s $SRC/$k.labeled $ST/$k.labeled; ln -s $SRC/$k.r0labels.jsonl $ST/$k.r0labels.jsonl; m=$((m+1)); fi
  n=$((n+1))
done
u=0
for f in $SRC/r0src/*.labeled; do
  b=$(basename $f .labeled); ln -s $f $ST/r0src/$b.labeled; ln -s $SRC/r0src/$b.r0labels.jsonl $ST/r0src/$b.r0labels.jsonl; u=$((u+1))
done
cp $SRC/LABEL_POLICY.json $ST/LABEL_POLICY.json
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) cut snapshot: $n raw shards ($m with per-shard labels), $u from-source units"
cd $L
nice -n 19 ionice -c3 $L/venv/bin/python -I bin/ll_finalize_par.py --raw $ST --out $OUT --subset $SUB --rows-per-shard 2000 \
  --allow-missing-student --allow-unlabeled --label-policy $ST/LABEL_POLICY.json --workers $W --min-free-gb 150 \
  --guard-mode v1 --guard-runs "" --max-final-gb $GB
cd $OUT/librilight.$SUB
split -n l/$W -d SHA256SUMS .sumpart.
ls .sumpart.* | xargs -P $W -I{} nice -n 19 ionice -c3 sha256sum -c --quiet {}
rm -f .sumpart.*
echo "SHA256SUMS verified: $(wc -l < SHA256SUMS) shards"
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) finalise $SUB done"
