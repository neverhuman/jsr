# H51-PQ scripts (as run, against an origin/main e61121e export)

| script | what it does |
|---|---|
| `prep_data.py` | hash-verified 730-utt DEV panel and the seeded 300-utt TRAIN subset as compact npz files |
| `h51lib.py` | model loading, engine-faithful INT8 emulation, `PQFit` (activation-weighted gain-shape additive PQ), `PQFitGPTQ` (error-feedback variant), `PQLinear` (exact pqscan integer contract + LUT-contract variants) |
| `trace.py`, `merge_trace.py`, `sparsity_report.py` | TRAIN activation traces: input second moments, held-out rows, group-sparsity counters (quantisers × rotations × row intersections) |
| `engine_sens.py` | per-layer / joint PQ variants decoded by the native engine (`ultra-engine-eval`), scored with `ultra-score` against the INT8 baseline |
| `launch_x9.sh`, `launch_groups_x9.sh`, `launch_h0.sh`, `sync_rows.sh` | guarded multi-host launchers (one process group per host under `ultra.fleet_guard`) and the two-way row sync |
| `fit_only.py`, `distortion_all.py` | parallel fit-cache fill; held-out output distortion for every GEMM |
| `gptq_test.py`, `lut_diag.py`, `act_diag.py`, `contract_diag.py` | k-means vs GPTQ; LUT rounding, activation encodings and LUT-contract variants |
| `aggregate.py`, `joint_report.py`, `make_assignment.py` | ranked tables, joint-run summary, regex for a candidate assignment |
| `project.py`, `project2.py` | Amdahl + bandwidth projection from the quiet profile and kernel times |
| `h51_lutcheck.rs` | scratch example run against `ultra_kernels::pqscan::CodebookMatrix` to prove the emulator bit-exact |
| `stage2_h0.sh`, `stage3_h0.sh`, `stage4_h0.sh`, `after_trace.sh`, `wrapup.sh` | the queued fit / distortion / joint-run stages and the clean stop |
| `gptq_vs_kmeans_all.py`, `readme_snippets.py`, `lut_ab.py` | GPTQ vs k-means on all LN-input GEMMs; README tables; PyTorch exact-LUT WER A/B (stopped: too slow on loaded H0) |
| `../bench/h51_prof.sh` | the quiet-profile assay bench-open ran as 14h |
| `int4_sens.py` | Addendum A: int4 per-row GPTQ / RTN fits for the 129 encoder GEMMs, held-out distortion, one joint variant through the engine's int8 packing, native 730-utt decode, `ultra-score` |
| `int4_report.py`, `copy_int4_record.sh` | Addendum A: arm table + stop-rule check (dry runs only; the closed verdict is in the README) and the copy of results into this folder |
| `../bench/h51_int4g_bench.rs` | Addendum A: K2 vs K2g (128-column group scales) kernel cost, bit-exact against a same-order scalar reference |
| `../bench/h51_int4_bench.rs`, `../bench/h51_int4.sh` | Addendum A: scratch int4-weight × int8-activation kernels (K1 naive unpack, K2 offset-binary) checked bit-exact against `gemm_i8`, timed cache-hot and as a 129-matrix sweep; the quiet-assay wrapper staged for bench-open |
