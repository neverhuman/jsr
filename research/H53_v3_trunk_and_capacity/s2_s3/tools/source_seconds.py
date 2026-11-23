#!/usr/bin/env python3
"""Exact audio seconds per source from FLAC headers (as the loader's _frames_after reads them, before speed perturbation),
plus row-group counts, for H53 Amendment 6's extra_base_weight. Each arg is name=dir_or_glob (parquet shards).
  source_seconds.py ls=/root/hf_parquet/all/train.*/ small=/dir ...   -> one JSON object"""
import glob, io, json, os, sys, time
import pyarrow.parquet as pq
import soundfile as sf
out = {}
for spec in sys.argv[1:]:
    name, _, pat = spec.partition("=")
    fs = sorted(glob.glob(os.path.join(pat, "*.parquet")) if os.path.isdir(pat) else glob.glob(pat))
    t0 = time.time(); groups = rows = 0; secs = 0.0; over20 = 0.0; n_over20 = 0
    for f in fs:
        pf = pq.ParquetFile(f)
        for g in range(pf.metadata.num_row_groups):
            col = pf.read_row_group(g, columns=["audio"]).column("audio").to_pylist()
            for a in col:
                i = sf.info(io.BytesIO(a["bytes"]))
                d = i.frames / i.samplerate
                secs += d
                if d > 20.0:
                    over20 += d; n_over20 += 1
            rows += len(col); groups += 1
    out[name] = {"files": len(fs), "groups": groups, "rows": rows, "hours": round(secs / 3600, 4),
                 "mean_s_per_group": round(secs / max(1, groups), 4), "rows_per_group": round(rows / max(1, groups), 4),
                 "hours_over_20s": round(over20 / 3600, 4), "rows_over_20s": n_over20, "scan_s": round(time.time() - t0, 1)}
print(json.dumps(out))
