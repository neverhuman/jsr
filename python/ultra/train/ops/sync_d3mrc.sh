#!/usr/bin/env bash
until ssh -n xbabe1 "test -f /mnt/amphora-backup/jsr-data/runs/train/D3MRc_S_MR_I8S_960h/PID"; do sleep 300; done
mkdir -p /mnt/xbabe3-external/jsr/runs/train/D3MRc_S_MR_I8S_960h
python3 /mnt/xbabe3-external/jsr/runs/train/sync_and_post.py xbabe1 /mnt/amphora-backup/jsr-data/runs/train/D3MRc_S_MR_I8S_960h /mnt/xbabe3-external/jsr/runs/train/D3MRc_S_MR_I8S_960h D3MRc &
python3 /mnt/xbabe3-external/jsr/runs/train/dev_curve_remote.py xbabe1 /mnt/amphora-backup/jsr-data/runs/train/D3MRc_S_MR_I8S_960h /mnt/xbabe3-external/jsr/runs/train/D3MRc_S_MR_I8S_960h/dev_curve D3MRc 10000
