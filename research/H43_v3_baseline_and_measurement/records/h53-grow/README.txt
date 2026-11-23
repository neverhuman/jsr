eval-lane: H53 grown-at-init export checks (exploratory, dev panel only).
1) B0-grow32 = h53 grow_depth (sha256 9229901d...) --blocks 32 --weights model on B0's EMA-as-model (4055c989..., from rank-0 ckpt 2a3dd9fb...),
   seed sha256 2b0cf830c6d96b08...; pinned export (d5f231e train.export bundle, r1_to_safetensors 25be8bd7..., ultra-engine-prepare 818644c7...):
   weights.int8 951d9f528e8a0b1f..., arch f6ac66e6af41ae53... (encoder.n_blocks 32). Release engine e2ef608 on the 730 panel: 730 / 730 transcripts
   identical to B0 INT8 (export/B0-grow32-ema/panel-int8/predictions.jsonl vs export/A1-49000-ema/panel-int8/predictions.jsonl).
2) Functional check that the engine EXECUTES the new blocks: bin/h53/perturb_grown.py copied each new block's residual output projections from its
   source block (128 tensors, 16 blocks) -> B0-grow32pert (seed e729d2dda41353ca..., weights.int8 e9837ae6e3b366b5...). On 6 panel clips
   (u00010..u00060) its transcripts differ from B0's in 2 / 6 (pert.jsonl), while B0-grow32 equals B0 in 6 / 6. So blocks 16-31 run.
3) Same-core CPU time on loaded H0 (core 46, 6 clips, 6 alternating reps, /usr/bin/time user s): B0 6.62 14.85 17.14 11.47 8.67 12.51;
   grow32 12.68 17.77 20.56 22.72 11.30 24.39 (per-pair ratio 1.20-1.98, best-of ratio 1.71). Noisy host: NOT a timing receipt; quiet x0 timing is bench-open's.
   Peak RSS 133.6 MB vs 71.0 MB.
