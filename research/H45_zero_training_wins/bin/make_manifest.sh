#!/usr/bin/env bash
# h45-wins: MANIFEST of every checkpoint and derived weight file used (paths + sha256); weights are never committed.
OUT=/mnt/xbabe3-external/jsr/h45-wins/MANIFEST
{
  echo "# H45 MANIFEST (generated $(date -u +%FT%TZ)); weights live on xbabe1:/mnt/amphora-backup/h45-wins/weights/ and the source hosts"
  echo "## source copies (pull.sh: streamed from the source host, verified against the source's own .sha256 marker)"
  grep -h " OK " /mnt/xbabe3-external/jsr/h45-wins/logs/pull_all.log /mnt/xbabe3-external/jsr/h45-wins/logs/pull_more.log | awk '{print $4, "xbabe1:weights/src/"$3, "<-", $5}'
  echo "## read in place on xbabe1 (/mnt/amphora-backup/jsr-data/runs/train)"
  ssh -o BatchMode=yes -n xbabe1 'cd /mnt/amphora-backup/jsr-data/runs/train && for d in H41_control H41_chunk_low H41_streamctx H41_crctc_full H41_chunklow_streamctx_s4321; do echo "$(cat $d/model-00008000.sha256) xbabe1:$d/model-00008000.pt"; done'
  echo "## derived weights (h45_merge.py; recipe + every input sha256 inside each file's h45_recipe)"
  ssh -o BatchMode=yes -n xbabe1 'cd /mnt/amphora-backup/h45-wins/weights/derived && for f in *.sha256; do echo "$(cat $f) xbabe1:weights/derived/${f%.sha256}.pt"; done'
  echo "## data and tools"
  echo "bfb27f5857411d7cea5fd8a307bab47793955425fbcb7ce5db5b428a919644b6 dev-clean.npz"
  echo "2944dbff8f3f3158e1192143a4aa6e6b4205681efbb12f1d1fd1c433b21d03c8 dev-other.npz"
  echo "c53433de083c4a6ad12d034550ef22de68cec62c4f58932a7b6b8b2f1e743fa5 bpe.model (R0 tokenizer; transducer and CTC heads)"
  echo "002ac7aee51519b4657d56b887b859660cfda51cf925c76aae61574150f29cac dev_split_v2.json (eval-lane)"
  echo "d784f8ca65a0df52c2dc2f58aa0d22912428f3595effa0625367d6e8aad063e1 ultra-score (pinned)"
  echo "e61121e (origin/main) python/ultra decode tree; d5f231e eval-lane pinned tree (instrument check)"
} > $OUT
wc -l $OUT
