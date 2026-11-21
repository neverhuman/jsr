#!/usr/bin/env bash
# When R4T ends on xbabe1: export ternary bundle + fp32 CPU reference transcripts there, then copy
# ckpt-12000, model-12000, the export and the references to H0 (lead-designated copy dir).
set -uo pipefail
R=/mnt/amphora-backup/jsr-data/runs/train/R4T_S_MR_T_tc100
H=/mnt/xbabe3-external/jsr/runs/train/R4T_S_MR_T_tc100_copy
until ! ssh -n xbabe1 "kill -0 \$(cat $R/PID)" 2>/dev/null; do sleep 120; done
sleep 30
ssh -n xbabe1 "cd ~/jsr/python/ultra && export JSR_DATA=/mnt/amphora-backup/jsr-data; CK=$R/\$(cat $R/LATEST).pt; \
  taskset -c 0-15 nice -n 10 ~/jsr/.venv/bin/python -m train.export_ternary --checkpoint \$CK --out $R/export_r4t > $R/export_r4t.log 2>&1; \
  echo FIN >> $R/export_r4t.log"
mkdir -p $H
ionice -c3 nice -n 10 rsync -rt --include='ckpt-00012000.*' --include='model-00012000.*' --include='train.jsonl' --include='config.json' --include='export_r4t/***' --include='export_r4t.log' --exclude='*' xbabe1:$R/ $H/
cd $H && for s in *.sha256; do [ "$(sha256sum ${s%.sha256}.pt | cut -d' ' -f1)" = "$(cat $s)" ] && echo "OK ${s%.sha256}" || echo "BAD ${s%.sha256}"; done > $H/sha_check.txt
# Reference transcripts on H0 CPU with the MKL torch build (the posted gate command), guarded.
cat > $H/ref_eval.sh <<EOS
#!/usr/bin/env bash
cd /home/ubuntu/jsr/.agent/worktrees/trainer/python/ultra
CUDA_VISIBLE_DEVICES= PYTHONPATH=/mnt/xbabe3-external/jsr/venvs/torch270-cpu taskset -c 12-15,76-79 nice -n 10 ../../.venv/bin/python -m train.evaluate --run-dir $H --checkpoint $H/ckpt-00012000.pt --split dev-clean --limit 400 --max-symbols 8 --threads 6 --dump $H/export_r4t/ref_dev400_torch_fakequant.jsonl > $H/export_r4t/ref_dev400_summary.json 2>/dev/null
EOS
chmod +x $H/ref_eval.sh
bash /mnt/xbabe3-external/jsr/logs/trainer/start_guarded_cpu.sh /mnt/xbabe3-external/jsr/logs/trainer/r4t_ref $H/ref_eval.sh
until [ -s $H/export_r4t/ref_dev400_summary.json ]; do sleep 30; done
echo DONE > $H/FINISH_DONE
