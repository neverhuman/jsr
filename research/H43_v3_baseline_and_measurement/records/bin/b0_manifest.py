#!/usr/bin/env python3
"""eval-lane H43: the frozen B0 receipt manifest (B0 = A1 final, step 49,000, EMA). Read-only over the receipts: it hashes every
listed file and records identities; pending receipts are null with an owner, never zero.
  b0_manifest.py OUT.json"""
import datetime, hashlib, json, os, subprocess, sys

E = "/mnt/xbabe3-external/jsr/evals/eval-lane"; X = f"{E}/export/A1-49000-ema"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def entry(p, role):
    return {"path": p, "role": role, "bytes": os.path.getsize(p), "sha256": sha(p)} if os.path.exists(p) else {"path": p, "role": role, "missing": True}


def score(p):
    d = json.load(open(p))["system_a"]; o = {}
    for ds in ("dev-clean", "dev-other"):
        c = d["datasets"][ds]["counts"]; o[ds] = {"S": c["substitutions"], "D": c["deletions"], "I": c["insertions"], "N": c["ref_words"], "wer_pct": round(100 * d["datasets"][ds]["wer"], 3)}
    o["pooled_wer_pct"] = round(100 * d["pooled_wer"], 3)
    return o


files = []
for f in ("arch.json", "bpe.model", "mel_fbank.f32", "tokens.txt", "weights.int8.safetensors", "weights.safetensors"):
    files.append(entry(f"{X}/int8/{f}", "INT8 engine bundle (all six files are loaded; the FP32 file is still required, H39)"))
files += [entry(f"{X}/SHA256SUMS", "export hashes incl. bundle/ and fp32/ intermediates"), entry(f"{X}/tools.sha256", "converter + preparer hashes"),
          entry(f"{X}/source/A1-49000-ema-as-model.pt", "derived EMA weights-only checkpoint (input of the export)"),
          entry(f"{X}/source/A1-49000-ema-as-model.sha256", "its hash marker"), entry(f"{E}/export/A1-49000-ema.derive.json", "derivation record (source ckpt sha, EMA decay, step)")]
rows = {}
for w in ("ema", "raw"):
    for c in (4, 8, 16):
        p = f"{E}/curve/A1-49000.{w}.c{c}.ms8"
        rows[f"{w}.c{c}"] = score(p + ".score.json")
        for s in ("dev-clean", "dev-other"):
            files.append(entry(f"{p}.{s}.jsonl", f"PyTorch greedy dump {w} chunk {c} {s} (uid, ref, hyp, ctc_hyp)"))
        files += [entry(p + ".score.json", f"canonical score {w} chunk {c}"), entry(p + ".score-input.jsonl", f"canonical score input {w} chunk {c}")]
pb = f"{E}/curve/A1-49000.ema.c4.ms8.b4"
rows["ema.c4.beam4"] = score(pb + ".beam.score.json")
for s in ("dev-clean", "dev-other"):
    files.append(entry(f"{pb}.{s}.jsonl", f"PyTorch beam-4 dump {s} (hyp and beam_hyp)"))
files += [entry(pb + ".beam.score.json", "canonical score beam 4"), entry(pb + ".beam.score-input.jsonl", "canonical score input beam 4")]
fd = f"{X}/fulldev-int8"
files += [entry(f"{fd}/predictions.all.jsonl", "INT8 engine e2ef608 full-dev predictions (12 shards concatenated)"),
          entry(f"{fd}/vs-pytorch.pytorch-v-engine.json", "canonical paired INT8 vs PyTorch, full dev"),
          entry(f"{fd}/vs-pytorch.engine.score-input.jsonl", "INT8 score input"), entry(f"{fd}/vs-pytorch.pytorch.score-input.jsonl", "PyTorch score input (same ids)")]
pn = f"{X}/panel-int8"
files += [entry(f"{pn}/predictions.jsonl", "INT8 engine, 730 panel"), entry(f"{pn}/panel730.pytorch-v-engine.json", "canonical paired panel INT8 vs PyTorch")]
for c in (4, 8):
    files.append(entry(f"{E}/wordlag/A1-49000/A1-49000-c{c}.trace.jsonl", f"paced word-lag trace, 100 MFA clips, chunk {c}"))
    files.append(entry(f"{E}/wordlag/traces/A1-49000-c{c}-compact.trace.jsonl", f"its compact conversion, chunk {c}"))
files += [entry(f"{E}/h43/latency-100clips/A1-49000-c4.json", "H43 latency metrics, 100 clips, 160 ms (both clocks, CQ, first-correct-word, revisions)"),
          entry(f"{E}/h43/dev_split_v2.json", "fixed dev speaker split (selection / confirmation)")]
for c in (2,):
    files.append(entry(f"{E}/h43/latency-fulldev/B0-c{c}.json", "H43 latency metrics, FULL dev, chunk 2 (80 ms): from x5 traces completed on x6 (salvaged, 5,567 / 5,567 complete utterances)"))
    files.append(entry(f"{E}/h43/latency-fulldev/B0-c{c}.utts.jsonl", "H43 per-utterance latency rows, FULL dev, chunk 2"))
    files.append(entry(f"{E}/h43/latency-fulldev/engine-wer/B0-A1-49000-c2.score.json", "B0 engine WER at 80 ms from the trace finals (8.954)"))
for c in (4, 8):
    lat = f"{E}/h43/latency-fulldev/B0-c{c}.json"
    files.append(entry(lat, f"H43 latency metrics, FULL dev (5,567), chunk {c}: both clocks, first-correct-word, CQ, revisions; speaker bootstrap") if os.path.exists(lat) else {"path": lat, "role": f"H43 latency metrics, FULL dev, chunk {c}", "pending": "eval-lane (paced traces running)"})
    files.append(entry(f"{E}/h43/latency-fulldev/B0-c{c}.utts.jsonl", f"H43 per-utterance latency rows, FULL dev, chunk {c}"))
    files.append(entry(f"{E}/h43/traces/B0-A1-49000-c{c}/SHA256SUMS", f"sha256 of the 12 paced full-dev trace + prediction shards, chunk {c} (design clock; H0 loaded, not quiet)"))
files.append(entry(f"{E}/h43/gain-panel/summary.json", "B0 gain-stress panel: 730 panel x 0 / -10 / -20 / -30 dB, WER + first-word error"))
files.append(entry(f"{E}/r0-fulldev/summary.json", "R0 Zipformer INT8 canonical full-dev control (comparison receipt)"))
tools = [entry("/mnt/xbabe3-external/jsr/evals/eval-lane/bin/engine-tools/ultra-cold-wav.e2ef608-release", "release engine e2ef608 (panel + full-dev INT8)"),
         entry("/mnt/xbabe3-external/jsr/evals/eval-lane/bin/engine-tools/ultra-cold-wav.paced-chunk-h41arm10", "paced word-lag engine with --chunk (traces)"),
         entry("/mnt/xbabe3-external/jsr/evals/eval-lane/bin/engine-tools/ultra-engine-prepare.energy-debug-2225", "INT8 preparer (reproduces D2e prepared 5633a93e byte for byte)"),
         entry("/home/ubuntu/jsr/target/paper-evidence/runtime/ultra-score", "canonical scorer (normaliser ultra-en-wer-norm/v1, rule-table sha 988c59ac...)"),
         entry("/mnt/xbabe3-external/jsr/evals/eval-lane/bin/derive_ema_ckpt.py", "EMA derivation"), entry("/mnt/xbabe3-external/jsr/evals/eval-lane/bin/export_model.sh", "export pipeline"),
         entry("/mnt/xbabe3-external/jsr/evals/eval-lane/bin/h43_latency.py", "H43 latency scorer"), entry("/mnt/xbabe3-external/jsr/evals/eval-lane/bin/h43_stats.py", "H43 statistics tool"),
         entry("/mnt/xbabe3-external/jsr/corpora/librispeech/word_timing/dev-clean.jsonl", "locked MFA word times, dev-clean"),
         entry("/mnt/xbabe3-external/jsr/corpora/librispeech/word_timing/dev-other.jsonl", "locked MFA word times, dev-other"),
         entry("/home/ubuntu/jsr/target/paper-evidence/word-lag/timestamps_100_mfa.json", "100-clip MFA word ends (word-lag set)"),
         entry(f"{E}/export/fulldev-wav.index.json", "full-dev staged WAV index (5,567 ids)"), entry(f"{E}/export/fulldev-wav.audio.json", "full-dev staged audio digests")]
src_fp = subprocess.run("cd /dev/shm/eval-lane/src-d5f231e && find . -type f | LC_ALL=C sort | xargs sha256sum | sha256sum", shell=True, capture_output=True, text=True).stdout.split()[0] if os.path.isdir("/dev/shm/eval-lane/src-d5f231e") else None
man = {"schema": "ultra.h43.b0-manifest/v1", "label": "exploratory receipts, read-only record", "made_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
       "b0": {"name": "A1 final", "run": "A1_S_r0bpe_960h", "step": 49000, "weights": "EMA (decay 0.9995)",
              "checkpoint": {"host_path": "xbabe5:~/jsr-data/runs/train/A1_S_r0bpe_960h/ckpt-00049000.pt", "sha256": "2a3dd9fb6c7856126b3455fb65babc4295e27b10e011d18a8806b8be5181e146"},
              "ema_weights_sha256": "4055c989c3822b326c52f5a0abb055810ddef7dab7ea85acffa388d134a3a70c",
              "int8": {"weights.int8.safetensors": "4701787c94d320953c42367bf0cf3be1ce2378e66bc811935af9f0f135b1e285", "arch.json": "b1581cbb1f22707ed16b146b0b04bdbcc181cdb6def5a68ad1108cc9af9db39f"},
              "decoder": "greedy, max 8 symbols per frame (PyTorch) / native greedy (engine)", "chunk_ms_default": 160, "tokenizer": "R0 bpe.model (c53433de...)"},
       "evaluator": {"pinned_source": "git d5f231e python/ultra snapshot", "snapshot_fingerprint_sha256_of_sorted_sha256_list": src_fp},
       "canonical_scorer": {"normaliser_id": "ultra-en-wer-norm/v1", "rule_table_sha256": "988c59acad873bf0a0c50bdcf434dde0b5125cf51c2cef887bfc63f4f89113e8",
                            "bootstrap": "paired speaker clusters, strata = dev subset, 10,000 replicates, seed 20261122, one-sided 95%"},
       "results": {"pytorch_full_dev": rows, "int8_engine_full_dev": None},
       "files": files, "tools": tools,
       "timing_quiet_xbabe0": {"source": "bench-open AGENT_CHAT 2026-11-23T06:21:26Z (TIMING 14br, receipts in bench-open's next PR)", "binary_sha256_prefix": "457a4db7",
                               "panel": "730 utterances, 1 core (CPUs 4 / 24 / 28 at 4.2 GHz), 3 cold reps, chunk 4, greedy", "cpu_s_per_audio_s": [0.03510, 0.03500, 0.03533],
                               "median_cpu_s_per_audio_s": 0.0351, "x_real_time": 28.45, "peak_rss_mb": 74.4, "panel_wer_pct": {"dev-clean": 4.26, "dev-other": 11.86, "pooled": 7.39},
                               "watts_per_live_stream_core_register": [0.334, 0.348, 0.349], "watts_scope": "core-register energy per live stream (energy's tool and method), not whole-device watts"},
       "endless_stream_27h": {"source": "h47-sleep AGENT_CHAT 2026-11-23T07:12:20Z (score_alwayson, A1-49000 INT8, same build)",
                              "wer_pct": {"C0 continuous, no reset": 10.989, "C1 reset-with-replay T=1500 / R=1000": 8.377, "O0 isolated": 8.273},
                              "gap_words": {"C0": 426, "C1": 5, "O0": 1}, "first_word_after_sleep_pct": {"C0": 74.9, "C1": 88.8, "O0": 89.9}},
       "pending": [{"receipt": "TIMING 14br receipt files (values above)", "owner": "bench-open (next PR)", "value": None},
                   {"receipt": "whole-device watts on the owner's metered target device", "owner": "energy / bench-open (H54)", "value": None},
                   {"receipt": "receipt files for the 27.56 h endless-stream values above (h47-sleep branch h47/sleep-gate 7a06cef)", "owner": "h47-sleep (PR)", "value": None},
                   {"receipt": "A1-49000 on the 10 x 1 h non-speech blocks (false words per hour)", "owner": "alwayson-bench (H47)", "value": None},
                   {"receipt": "canonical per-utterance ledger for B0", "owner": "codex-wer-evidence", "value": None}]}
d = json.load(open(f"{fd}/vs-pytorch.pytorch-v-engine.json"))["system_b"]; o = {}
for ds in ("dev-clean", "dev-other"):
    c = d["datasets"][ds]["counts"]; o[ds] = {"S": c["substitutions"], "D": c["deletions"], "I": c["insertions"], "N": c["ref_words"], "wer_pct": round(100 * d["datasets"][ds]["wer"], 3)}
o["pooled_wer_pct"] = round(100 * d["pooled_wer"], 3); man["results"]["int8_engine_full_dev"] = o
lat = {}
for c in (2, 4, 8):
    p = f"{E}/h43/latency-fulldev/B0-c{c}.json"
    if os.path.exists(p):
        L = json.load(open(p)); a = L.get("aggregate", L)
        lat[f"c{c}"] = {k: a[k] for k in ("ftl_end", "ftl_onset", "fcw_end", "fcw_onset", "fcw_coverage", "cq_stable", "first_cq_end", "first_cq_onset", "stable_lag", "revision") if k in a}
        lat[f"c{c}"]["bootstrap_95"] = L.get("bootstrap_95")
man["results"]["latency_full_dev_design_clock"] = lat or None
man["files_sha256_of_list"] = hashlib.sha256(json.dumps([(f["path"], f.get("sha256")) for f in files], sort_keys=True).encode()).hexdigest()
json.dump(man, open(sys.argv[1], "w"), indent=1)
print(json.dumps({"files": len(files), "missing": [f["path"] for f in files if f.get("missing")], "pending": len(man["pending"]), "list_sha": man["files_sha256_of_list"][:16]}))
