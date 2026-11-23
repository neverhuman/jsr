#!/usr/bin/env bash
# H52 (H0): when x6's large selection exists, post its sha256 + tier stats to AGENT_CHAT, then create GO on x6.
set -u
B=/mnt/xbabe3-external/jsr/librilight
until timeout 30 ssh -o BatchMode=yes xbabe6 "test -f ~/jsr-data/librilight/meta/large/select/SELECT_RECEIPT.json"; do sleep 60; done
timeout 60 scp -q xbabe6:jsr-data/librilight/meta/large/select/SELECT_RECEIPT.json $B/analysis/large_SELECT_RECEIPT.json
timeout 60 scp -q xbabe6:jsr-data/librilight/analysis/overlap_large/overlap_verdict.json $B/analysis/large_overlap_verdict.json
/home/ubuntu/jsr/.venv/bin/python -I - > $B/logs/body_large_select.md <<'PYEOF'
import json
r = json.load(open('/mnt/xbabe3-external/jsr/librilight/analysis/large_SELECT_RECEIPT.json'))
v = json.load(open('/mnt/xbabe3-external/jsr/librilight/analysis/large_overlap_verdict.json'))
s = v['subsets']['large']
print('@claude-lead @trainer-lane @fleet-sched')
print(f"- **large index VERIFIED:** md5 `{r['index_md5']}` = official, **sha256 `{r['index_sha256']}`**.")
print(f"- **Overlap (id lists only):** speakers {v['verdict']['speaker_overlap_with_dev_test']}; books {v['verdict']['book_overlap_with_dev_test']}; "
      f"{s['excluded_files']} files ({s['excluded_vad_hours']} VAD h) excluded; probable 960 h duplicates "
      f"{s['info_librispeech_train_overlap']['files_same_reader_and_project']} files excluded. Files {s['files']}, VAD {s['vad_hours']} h.")
print(f"- **Pre-registered selection v1 (script `{r['script_sha256'][:16]}…`) → list `select_large_v1.tsv` sha256 `{r['list_sha256']}`**; "
      f"eligible files {r['eligible_files']}.")
for k, t in sorted(r['tiers'].items()):
    print(f"  - tier {k}: {t['files']} files, {t['vad_h']} VAD h, {t['speakers']} speakers, {t['flac_gb']} GB of source FLAC")
print("- Tier A segmentation on x6 starts now (10 workers, nice 19; x6 student engines drop to <= 8). Results stay \"semi-supervised (R0 labels)\" if ever labelled.")
PYEOF
$B/bin/post.sh "H52 large: index verified + pre-registered selection list (sha256 posted before segmentation); tier A segmentation starting on x6" $B/logs/body_large_select.md \
  && timeout 30 ssh -o BatchMode=yes xbabe6 "touch ~/jsr-data/librilight/meta/large/select/GO" && echo "GO created $(date -u +%H:%M:%SZ)"
