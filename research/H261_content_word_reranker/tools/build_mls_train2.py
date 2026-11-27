"""H261: TRAIN-domain audio held out from the acoustic model: MLS-en TRAIN split, (speaker, book) pairs in
allowed_train_pairs.tsv (LibriSpeech and MLS-eval overlaps removed; Libri-Light overlap NOT removed, so the AM
may have heard LL tier-A audio of some books), speakers absent from the AM's MLS packs
(agy mls_pk06b_2300spk + mls_pl3_pack).

The data drive has unrecovered read errors, so this reads as little as possible: the three metadata columns,
then only the 100-row row groups that hold picked rows (about 6 MB each). Unreadable files are skipped and logged.
Writes mls_train2_tail320.npz (pcm + 320 ms zero tail, offsets, uids) and mls_train_refs.jsonl."""
import csv, glob, io, json, random
import numpy as np, pyarrow.parquet as pq, soundfile as sf

H = "/mnt/xbabe3-external/jsr/wer-push/H261"
A = "/mnt/xbabe3-external/jsr/corpora_v4/mls_en/audit"
seen = set(open(f"{H}/am_seen_mls_speakers.txt").read().split())
allowed = {(r["speaker_id"], r["book_id"]) for r in csv.DictReader(open(f"{A}/allowed_train_pairs.tsv"), delimiter="\t")}
files = sorted(glob.glob("/mnt/xbabe3-external/jsr/corpora_v4/mls_en/parquets/train-*.parquet"))
rng = random.Random(262)
rng.shuffle(files)
prev = set(json.loads(l)["uid"] for l in open(f"{H}/mls_train_refs.jsonl"))
per_spk = {}
for l in open(f"{H}/mls_train_refs.jsonl"):
    s_ = json.loads(l)["speaker"]; per_spk[s_] = per_spk.get(s_, 0) + 1
cap, target, per_file = 36, 9000, 24
pcm, off, refs, bad = [], [0], [], []
for f in files:
    try:
        pf = pq.ParquetFile(f)
        meta = pf.read(columns=["speaker_id", "book_id", "audio_duration"]).to_pylist()
    except OSError as e:
        bad.append(f); print("unreadable", f, e, flush=True); continue
    ok = [i for i, r in enumerate(meta) if r["speaker_id"] not in seen and (r["speaker_id"], r["book_id"]) in allowed
          and 3.0 <= r["audio_duration"] <= 20.0 and per_spk.get(r["speaker_id"], 0) < cap]
    if not ok:
        continue
    rgs = sorted({i // 100 for i in ok})
    rng.shuffle(rgs)
    got = 0
    for g in rgs[:3]:
        rows_in = [i for i in ok if i // 100 == g]
        rng.shuffle(rows_in)
        try:
            tab = pf.read_row_group(g).to_pylist()
        except OSError as e:
            bad.append(f"{f}#rg{g}"); print("unreadable", f, g, e, flush=True); continue
        for i in rows_in[: per_file // 3]:
            r = tab[i - g * 100]
            if r["audio"]["path"].rsplit(".", 1)[0] in prev:
                continue
            spk = r["speaker_id"]
            if per_spk.get(spk, 0) >= cap:
                continue
            a, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="int16")
            assert sr == 16000 and a.ndim == 1
            n = len(a)
            a = np.concatenate([a, np.zeros(5120, np.int16)])
            pcm.append(a); off.append(off[-1] + len(a)); per_spk[spk] = per_spk.get(spk, 0) + 1; got += 1
            refs.append({"uid": r["audio"]["path"].rsplit(".", 1)[0], "speaker": spk, "book": r["book_id"],
                         "ref": r["transcript"].upper(), "samples": n})
    if len(refs) >= target:
        break
np.savez(f"{H}/mls_train2_tail320.npz", pcm=np.concatenate(pcm).astype("<i2"), offsets=np.array(off, dtype=np.int64),
         uids=np.array([r["uid"] for r in refs]))
with open(f"{H}/mls_train2_refs.jsonl", "w") as fh:
    for r in refs:
        fh.write(json.dumps(r) + "\n")
open(f"{H}/unreadable_parquets2.txt", "w").write("\n".join(bad) + "\n")
print("unreadable", len(bad), "utts", len(refs), "speakers", len(per_spk),
      "hours", round(sum(r["samples"] for r in refs) / 16000 / 3600, 1), "files touched", files.index(f) + 1)
