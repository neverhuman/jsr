"""H52 overlap check: Libri-Light speakers / books vs LibriSpeech dev-clean, dev-other, test-clean,
test-other, from id lists only.

LibriSpeech side: SPEAKERS.TXT (reader id + subset), CHAPTERS.TXT (chapter, reader, subset,
LibriVox project id, Gutenberg book id, project title), BOOKS.TXT (Gutenberg id -> title), all from
the md5-verified dev-clean.tar.gz. No test audio and no test transcripts are read. Optionally the
dev parquet `speaker_id` / `chapter_id` columns (dev only) cross-check the dev lists.
Libri-Light side: each file's JSON (speaker, book_meta.id = LibriVox project id, url_text_source ->
Gutenberg id, title) plus the speaker directory in the member path.

A Libri-Light file is EXCLUDED if any of: speaker in an eval set; LibriVox project id equal to an
eval chapter's project; Gutenberg id equal to an eval chapter's book; normalised title equal to an
eval project / book title. Informational: overlap with LibriSpeech *train* (same reader, and same
reader + project, i.e. probably the same recordings as the 960 h).

usage: python -I ll_overlap.py <ls_meta_dir> <out_dir> <subset>:<json_meta.jsonl> [...] [--dev-parquet <root>]
"""
import hashlib
import json
import os
import re
import sys

EVAL = ("dev-clean", "dev-other", "test-clean", "test-other")
GUT = re.compile(r"gutenberg\.org/(?:etext|ebooks|files|cache/epub)/(\d+)", re.I)


def sha256(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def norm_title(t):
    t = (t or "").lower().replace("&", " and ")
    t = re.sub(r"\(.*?\)|\[.*?\]", " ", t)
    t = re.sub(r"[^a-z0-9 ]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"^(the|a|an) ", "", t)
    return t


def rows(path, nmin):
    for line in open(path, encoding="utf-8", errors="replace"):
        if line.startswith(";") or not line.strip():
            continue
        f = [x.strip() for x in line.rstrip("\n").split("|")]
        if len(f) >= nmin:
            yield f


def load_ls(d):
    spk_subset, ch = {}, []
    for f in rows(os.path.join(d, "SPEAKERS.TXT"), 4):
        spk_subset[int(f[0])] = f[2]
    for f in rows(os.path.join(d, "CHAPTERS.TXT"), 8):
        ch.append(dict(chapter=int(f[0]), reader=int(f[1]), subset=f[3], proj=f[4], book=f[5], ptitle=f[-1]))
    books = {}
    for f in rows(os.path.join(d, "BOOKS.TXT"), 2):
        books[f[0]] = f[1]
    ls = {"speakers": {s: set() for s in EVAL}, "proj": {s: set() for s in EVAL},
          "gut": {s: set() for s in EVAL}, "title": {s: {} for s in EVAL}, "chapters": {s: set() for s in EVAL}}
    for s, sub in spk_subset.items():
        if sub in EVAL:
            ls["speakers"][sub].add(s)
    train_spk = {s for s, sub in spk_subset.items() if sub.startswith("train")}
    train_pairs = set()
    for c in ch:
        if c["subset"] in EVAL:
            sub = c["subset"]
            ls["chapters"][sub].add(c["chapter"])
            ls["proj"][sub].add(c["proj"])
            ls["gut"][sub].add(c["book"])
            for t in (c["ptitle"], books.get(c["book"], "")):
                nt = norm_title(t)
                if nt:
                    ls["title"][sub][nt] = t
        elif c["subset"].startswith("train"):
            train_pairs.add((c["reader"], c["proj"]))
    # consistency: every eval chapter's reader is an eval speaker of the same subset
    incons = [c for c in ch if c["subset"] in EVAL and c["reader"] not in ls["speakers"][c["subset"]]]
    return ls, train_spk, train_pairs, incons, len(spk_subset), len(ch), len(books)


def dev_parquet_ids(root):
    import glob

    import pyarrow.parquet as pq
    out = {}
    for sub, dname in (("dev-clean", "validation.clean"), ("dev-other", "validation.other")):
        spk, chap = set(), set()
        for f in sorted(glob.glob(os.path.join(root, dname, "*.parquet"))):
            t = pq.read_table(f, columns=["speaker_id", "chapter_id"])  # id columns only
            spk |= set(t.column("speaker_id").to_pylist())
            chap |= set(t.column("chapter_id").to_pylist())
        out[sub] = (spk, chap)
    return out


def main():
    args = sys.argv[1:]
    dev_root = None
    if "--dev-parquet" in args:
        i = args.index("--dev-parquet")
        dev_root = args[i + 1]
        args = args[:i] + args[i + 2:]
    ls_dir, out_dir, inputs = args[0], args[1], args[2:]
    os.makedirs(out_dir, exist_ok=True)
    ls, train_spk, train_pairs, incons, n_spk, n_ch, n_books = load_ls(ls_dir)
    summary = {"tool": "ll_overlap.py", "librispeech_lists": {
        n: sha256(os.path.join(ls_dir, n)) for n in ("SPEAKERS.TXT", "CHAPTERS.TXT", "BOOKS.TXT")},
        "librispeech_counts": {"speakers": n_spk, "chapters": n_ch, "books": n_books,
                               **{s: {"speakers": len(ls["speakers"][s]), "chapters": len(ls["chapters"][s]),
                                      "projects": len(ls["proj"][s]), "gutenberg_books": len(ls["gut"][s]),
                                      "titles": len(ls["title"][s])} for s in EVAL}},
        "librispeech_eval_chapters_with_reader_not_in_subset": len(incons),
        "test_audio_or_transcripts_read": False,
        "librilight_statement": "Kahn et al. 2020 (arXiv 1912.07875) sec. 2: 'We then removed corrupted "
                                "files, files with unknown or multiple speakers, and speakers appearing in "
                                "LibriSpeech dev and test sets.' and 'All dev or test set audio has been "
                                "removed from training sets.' (books are not mentioned)",
        "subsets": {}}
    if dev_root:
        dp = dev_parquet_ids(dev_root)
        summary["dev_parquet_crosscheck"] = {
            s: {"speakers_equal_SPEAKERS.TXT": dp[s][0] == ls["speakers"][s],
                "chapters_equal_CHAPTERS.TXT": dp[s][1] == ls["chapters"][s],
                "n_speakers": len(dp[s][0]), "n_chapters": len(dp[s][1])} for s in dp}
    all_eval_spk = set().union(*ls["speakers"].values())
    for spec in inputs:
        subset, path = spec.split(":", 1)
        excl = open(os.path.join(out_dir, f"excluded_{subset}.tsv"), "w")
        excl.write("name\tspeaker\tproject\treasons\tvad_s\n")
        dup = open(os.path.join(out_dir, f"train_dup_{subset}.tsv"), "w")
        dup.write("name\tspeaker\tproject\treasons\tvad_s\n")
        n = 0
        vad_total = 0.0
        hit = {s: {"speaker": set(), "project": set(), "gutenberg": set(), "title": set()} for s in EVAL}
        excl_files, excl_vad, reasons_ct = 0, 0.0, {}
        spk_all, proj_all, gut_all = set(), set(), set()
        dir_mismatch, missing_gut, bad_json = 0, 0, 0
        tr_spk_files, tr_pair_files, tr_spk_vad, tr_pair_vad = 0, 0, 0.0, 0.0
        tr_spk_set, tr_pair_set = set(), set()
        for line in open(path):
            r = json.loads(line)
            try:
                j = json.loads(r["text"])
                spk = int(str(j["speaker"]).strip())
                bm = j.get("book_meta") or {}
                proj = str(bm.get("id", "")).strip()
            except Exception:
                bad_json += 1
                continue
            n += 1
            vad = sum(float(e) - float(s) for s, e in j.get("voice_activity") or [])
            vad_total += vad
            if r["name"].split("/")[1] != str(spk):
                dir_mismatch += 1
            m = GUT.search(bm.get("url_text_source") or "")
            gut = m.group(1) if m else None
            if gut is None:
                missing_gut += 1
            nt = norm_title(bm.get("title", ""))
            spk_all.add(spk)
            proj_all.add(proj)
            if gut:
                gut_all.add(gut)
            reasons = []
            for s in EVAL:
                if spk in ls["speakers"][s]:
                    hit[s]["speaker"].add(spk)
                    reasons.append(f"speaker:{s}")
                if proj and proj in ls["proj"][s]:
                    hit[s]["project"].add(proj)
                    reasons.append(f"project:{s}")
                if gut and gut in ls["gut"][s]:
                    hit[s]["gutenberg"].add(gut)
                    reasons.append(f"gutenberg:{s}")
                if nt and nt in ls["title"][s]:
                    hit[s]["title"].add(nt)
                    reasons.append(f"title:{s}")
            if reasons:
                excl_files += 1
                excl_vad += vad
                for x in reasons:
                    reasons_ct[x.split(":")[0]] = reasons_ct.get(x.split(":")[0], 0) + 1
                excl.write(f"{r['name']}\t{spk}\t{proj}\t{','.join(sorted(set(reasons)))}\t{vad:.2f}\n")
            if spk in train_spk:
                tr_spk_files += 1
                tr_spk_vad += vad
                tr_spk_set.add(spk)
            # (reader, LibriVox project) pairs used by LibriSpeech train: likely the same recordings
            if (spk, proj) in train_pairs:
                tr_pair_files += 1
                tr_pair_vad += vad
                tr_pair_set.add((spk, proj))
                dup.write(f"{r['name']}\t{spk}\t{proj}\ttrain_reader_project\t{vad:.2f}\n")
        excl.close()
        dup.close()
        summary["subsets"][subset] = {
            "json_meta": path, "json_meta_sha256": sha256(path), "files": n, "bad_json": bad_json,
            "vad_hours": round(vad_total / 3600, 2), "speakers": len(spk_all), "projects": len(proj_all),
            "files_without_gutenberg_id": missing_gut, "speaker_dir_vs_json_mismatch": dir_mismatch,
            "overlap": {s: {k: sorted(v) for k, v in hit[s].items()} for s in EVAL},
            "speaker_overlap_any_eval": sorted(spk_all & all_eval_spk),
            "excluded_files": excl_files, "excluded_vad_hours": round(excl_vad / 3600, 3),
            "excluded_reason_counts": reasons_ct,
            "info_librispeech_train_overlap": {
                "speakers": len(tr_spk_set), "files_by_train_speaker": tr_spk_files,
                "vad_hours_by_train_speaker": round(tr_spk_vad / 3600, 2),
                "reader_project_pairs": len(tr_pair_set), "files_same_reader_and_project": tr_pair_files,
                "vad_hours_same_reader_and_project": round(tr_pair_vad / 3600, 2)},
        }
    verdict_speakers = all(not v["speaker_overlap_any_eval"] for v in summary["subsets"].values())
    summary["verdict"] = {
        "speaker_overlap_with_dev_test": "NONE" if verdict_speakers else "FOUND",
        "book_overlap_with_dev_test": "NONE" if all(
            not any(v["overlap"][s][k] for s in EVAL for k in ("project", "gutenberg", "title"))
            for v in summary["subsets"].values()) else "FOUND (those files are excluded)",
        "excluded_files": {k: v["excluded_files"] for k, v in summary["subsets"].items()},
    }
    with open(os.path.join(out_dir, "overlap_verdict.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary["verdict"], indent=1))
    for k, v in summary["subsets"].items():
        print(k, {x: v[x] for x in ("files", "vad_hours", "speakers", "projects", "excluded_files",
                                    "excluded_vad_hours", "excluded_reason_counts")})
        print("  train overlap (info):", v["info_librispeech_train_overlap"])


if __name__ == "__main__":
    main()
