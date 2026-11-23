"""H52 large-subset selection, rule v1 (pre-registered 2026-11-23 ~09:20Z, before any large.tar metadata was seen).

Rule:
  1. Eligible = FLAC members of large.tar that have JSON metadata, minus the overlap tool's dev/test exclusions
     (excluded_large.tsv: speaker, LibriVox project, Gutenberg id or normalised title of a LibriSpeech
     dev/test chapter) and minus probable 960 h duplicates (train_dup_large.tsv: same reader + LibriVox project
     as a LibriSpeech train chapter). Members with no VAD speech are skipped.
  2. Order = ascending sha256("h52-large-v1:" + member path).
  3. Per-speaker cap: a file is skipped while its speaker already has >= 15 VAD hours selected.
  4. Tier A = files in that order until the cumulative VAD speech reaches 10,000 h (the crossing file is in);
     tier B continues to 20,000 h. Nothing else is selected.
Output: select_large_v1.tsv (name, offset_data, size, sha256, speaker, project, vad_s, tier) and
SELECT_RECEIPT.json (counts, hours, list sha256, this script's sha256). Refuses an unverified index.

usage: python -I ll_select_large.py <index_dir> <overlap_dir> <out_dir>
"""
import hashlib
import json
import os
import sys

TIERS = (("A", 10_000 * 3600.0), ("B", 20_000 * 3600.0))
SPEAKER_CAP_S = 15 * 3600.0
SALT = "h52-large-v1:"


def main():
    idx, ov, out = sys.argv[1], sys.argv[2], sys.argv[3]
    rec = json.load(open(os.path.join(idx, "receipt.json")))
    if not rec.get("md5_matches_official"):
        sys.exit("large index receipt does not show an official md5 match; refusing")
    skip = set()
    for fn in ("excluded_large.tsv", "train_dup_large.tsv"):
        for line in open(os.path.join(ov, fn)).read().splitlines()[1:]:
            skip.add(line.split("\t")[0][:-5])
    meta = {}
    for line in open(os.path.join(idx, "json_meta.jsonl")):
        r = json.loads(line)
        try:
            j = json.loads(r["text"])
            vad = sum(float(e) - float(s) for s, e in (j.get("voice_activity") or []))
            meta[r["name"][:-5]] = (str(j["speaker"]).strip(), str((j.get("book_meta") or {}).get("id", "")).strip(), vad)
        except Exception:
            continue
    cands = []
    for line in open(os.path.join(idx, "members.tsv")).read().splitlines()[1:]:
        name, kind, size, oh, od, sha = line.split("\t")
        if kind != "reg" or not name.endswith(".flac"):
            continue
        st = name[:-5]
        if st not in meta or st in skip or meta[st][2] <= 0:
            continue
        cands.append((hashlib.sha256((SALT + name).encode()).hexdigest(), name, int(od), int(size), sha))
    cands.sort()
    per_spk, total, rows = {}, 0.0, []
    tier_i = 0
    for _, name, od, size, sha in cands:
        if tier_i >= len(TIERS):
            break
        spk, proj, vad = meta[name[:-5]]
        if per_spk.get(spk, 0.0) >= SPEAKER_CAP_S:
            continue
        per_spk[spk] = per_spk.get(spk, 0.0) + vad
        total += vad
        rows.append((name, od, size, sha, spk, proj, round(vad, 2), TIERS[tier_i][0]))
        if total >= TIERS[tier_i][1]:
            tier_i += 1
    os.makedirs(out, exist_ok=True)
    lp = os.path.join(out, "select_large_v1.tsv")
    with open(lp, "w") as f:
        f.write("name\toffset_data\tsize\tsha256\tspeaker\tproject\tvad_s\ttier\n")
        for r in rows:
            f.write("\t".join(map(str, r)) + "\n")
    lsha = hashlib.sha256(open(lp, "rb").read()).hexdigest()
    tiers = {}
    for r in rows:
        t = tiers.setdefault(r[7], {"files": 0, "vad_h": 0.0, "bytes": 0, "speakers": set()})
        t["files"] += 1
        t["vad_h"] += r[6] / 3600
        t["bytes"] += r[2]
        t["speakers"].add(r[4])
    receipt = {"rule": "h52-large-v1", "index_sha256": rec["sha256"], "index_md5": rec["md5"],
               "eligible_files": len(cands), "skipped_overlap_or_dup": len(skip),
               "selected_files": len(rows), "list_sha256": lsha,
               "script_sha256": hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest(),
               "tiers": {k: {"files": v["files"], "vad_h": round(v["vad_h"], 1), "flac_gb": round(v["bytes"] / 1e9, 1),
                             "speakers": len(v["speakers"])} for k, v in tiers.items()}}
    with open(os.path.join(out, "SELECT_RECEIPT.json"), "w") as f:
        json.dump(receipt, f, indent=1)
    print(json.dumps(receipt, indent=1))


if __name__ == "__main__":
    main()
