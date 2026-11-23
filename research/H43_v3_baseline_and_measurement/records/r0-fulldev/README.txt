eval-lane: R0 Zipformer INT8 canonical full-DEV control (exploratory; dev sets only).
Run: x1 (role label) /dev/shm, 2026-11-23 06:58:12Z - 07:27:18Z, 12 shards x 1 single-thread engine, nice 19, fleet_guard (origin/main 03ca5ff,
--gpu -1 --cpu-target 0.80 --min-duty 0; duty stayed 1.0, host CPU 22-27%). Not a timing measurement.
Runner: ultra-cold-wav e2ef608 release (sha256 457a4db7423b61b6...), backend zipformer-int8, isolated cold WAV per utterance, 320 ms chunks, greedy.
Bundle: /mnt/xbabe3-external/jsr/bundles/r0-zipformer-int8 (manifest 2f98a3f25b2ae1b2..., encoder 563fde436d16cf76..., decoder 98da299f471e38bb...,
joiner d944208d660d67c8..., tokens 49e3c2646595fd90...; upstream csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-26 @ 4062f0f2...).
Libraries: libonnxruntime.so 4b3607aebd1784b2..., libsherpa-onnx-c-api.so c72c8b56d8621a27... (sherpa-onnx 2.0.2 wheel).
Audio: staged full dev (export/fulldev-wav.audio.json sha256 27b1f8c7b5facce2...), 5,567 utterances.
Scorer: ultra-score d784f8ca65a0df52..., normaliser ultra-en-wer-norm/v1 (988c59ac...), 10,000 speaker-cluster replicates, seed 20261122, one-sided 95%.
Files: r0-out.tar (predictions.all.jsonl sha256 3bbd4151cf057983..., per-shard predictions and logs, progress.log, fleet_guard.jsonl);
R0-int8.fulldev.score-input.jsonl; R0-vs-B0-int8-e2ef608.score.json; R0-vs-B0-pytorch-ema-c4.score.json; R0-int8.panel730.score-input.jsonl; summary.json.
Check: the 730-utterance panel subset reproduces the paper row r0-int8-zen2-t1 exactly (3.034 / 8.427 / pooled 5.253).
Built by bin/r0_fulldev_x1.sh and bin/r0_fulldev_score.py.
