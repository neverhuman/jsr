#!/usr/bin/env bash
# h45-wins: the read-only source copies for H45 (A1 basin, A1L, H41 arms), in priority order.
P=/mnt/xbabe3-external/jsr/h45-wins/bin/pull.sh
A1=/home/ubuntu/jsr-data/runs/train/A1_S_r0bpe_960h
$P xbabe5 $A1/ckpt-00049000.pt A1-ckpt-00049000.pt
for s in 00047500 00045000 00042500 00040000; do $P xbabe5 $A1/model-$s.pt A1-model-$s.pt; done
$P local /mnt/xbabe3-external/jsr/runs/train/H41_chunk/model-00008000.pt H41_chunk-model-00008000.pt
$P xbabe6 /home/ubuntu/jsr-data/runs/train/H41_chunklow_streamctx/model-00008000.pt H41_chunklow_streamctx-model-00008000.pt
$P xbabe7 /home/ubuntu/jsr-data/runs/train/H41_chunk_predctx4/model-00008000.pt H41_chunk_predctx4-model-00008000.pt
$P xbabe7 /home/ubuntu/jsr-data/runs/train/H41_chunk_left256/model-00008000.pt H41_chunk_left256-model-00008000.pt
$P xbabe9 /home/ubuntu/jsr-data/runs/train/H41_chunk_specaug/model-00008000.pt H41_chunk_specaug-model-00008000.pt
$P xbabe8 /home/ubuntu/jsr-data/runs/train/A1L_dp0.005/ckpt-00005907.pt A1L-ckpt-00005907.pt
$P xbabe2 /home/ubuntu/jsr-data/runs/train/H41_aa/model-00008000.pt H41_aa-model-00008000.pt
$P xbabe8 /home/ubuntu/jsr-data/runs/train/H41_chunk_s4321/model-00008000.pt H41_chunk_s4321-model-00008000.pt
echo "$(date -u +%FT%TZ) ALL-DONE"
