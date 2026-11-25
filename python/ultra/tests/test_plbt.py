"""H61 book-text labels (plbt/v1): normaliser, projection rules, index + runner, panel self-check. Torch-free."""
import gzip
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tools.plbt.lh_index import build, parse_cut, rec_key  # noqa: E402
from tools.plbt.norm import ABBR, LEX, NUM, OTHER, book_words, label_words  # noqa: E402
from tools.plbt.project import Cut, project, similarity  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
PANEL = os.path.join(REPO, "docs/paper/data/raw/energy/cold/zipformer-int8/round-3/scores.jsonl")


def kinds(text):
    return [(w.core, w.kind) for w in book_words(text)]


def test_normaliser_tokens_and_kinds():
    assert kinds("“Well,” said Mr. Brown—to-day, in 1867!") == [
        ("WELL", LEX), ("SAID", LEX), ("MR", ABBR), ("BROWN", LEX), ("TO", LEX), ("DAY", LEX), ("IN", LEX),
        ("1867", NUM)]
    assert kinds("Louis XIV. paid $5 and 10s. for the 3rd; 20% of the U.S. & Co.") == [
        ("LOUIS", LEX), ("XIV", NUM), ("PAID", LEX), ("$5", NUM), ("AND", LEX), ("10S", NUM), ("FOR", LEX),
        ("THE", LEX), ("3RD", NUM), ("20%", NUM), ("OF", LEX), ("THE", LEX), ("U", LEX), ("S", LEX),
        ("AND", LEX), ("CO", ABBR)]
    assert kinds("Mc Cloud; a mix; T V; café _very_ naïve [x9y]") == [
        ("MC", LEX), ("CLOUD", LEX), ("A", LEX), ("MIX", LEX), ("T", LEX), ("V", LEX), ("CAFE", LEX),
        ("VERY", LEX), ("NAIVE", LEX), ("X9Y", OTHER)]
    w = book_words("'Tis the boys' ‘quoted’ mary,' don't")
    assert [(x.core, x.lead, x.trail) for x in w] == [("TIS", True, False), ("THE", False, False),
                                                      ("BOYS", False, True), ("QUOTED", True, True),
                                                      ("MARY", False, True), ("DON'T", False, False)]
    assert [x.cap for x in book_words("5 1867 1,000,000 XII")] == [4, 8, 8, 3]
    assert label_words("  the  Day was ") == ["THE", "DAY", "WAS"]


def one(book, hyp, t=(0.0, 10.0), span=(-30.0, 40.0)):
    return project(label_words(hyp), [Cut(span[0], span[1], book_words(book))], *t)


def test_book_word_wins_inside_the_span():
    book = "It rained all night. And then the old horse went slowly home, they said. Nobody saw it."
    assert one(book, "AND THEN THE OLD HORSE WENT SLOWLY HOME").label == "AND THEN THE OLD HORSE WENT SLOWLY HOME".split()
    r = one(book, "AND THEN THE OLD HOUSE WENT SLOWLY HOME")  # substitution
    assert (r.decision, r.label, r.sub) == ("keep", "AND THEN THE OLD HORSE WENT SLOWLY HOME".split(), 1)
    r = one(book, "AND THEN THE HORSE WENT SLOWLY HOME")  # R0 dropped OLD
    assert (r.label, r.dele) == ("AND THEN THE OLD HORSE WENT SLOWLY HOME".split(), 1)
    r = one(book, "AND THEN THE OLD OLD HORSE WENT SLOWLY HOME")  # R0 inserted a word
    assert (r.label, r.ins) == ("AND THEN THE OLD HORSE WENT SLOWLY HOME".split(), 1)


def test_numbers_and_abbreviations():
    r = one("So in 1867 the war ended there.", "IN EIGHTEEN SIXTY SEVEN THE WAR ENDED")
    assert (r.decision, r.label, r.filled) == ("keep", "IN EIGHTEEN SIXTY SEVEN THE WAR ENDED".split(), 3)
    assert one("He saw Louis XIV. at the court.", "SAW LOUIS THE FOURTEENTH AT THE COURT").label[2:4] == ["THE", "FOURTEENTH"]
    for garbled in ("IN EIGHTEEN SIXTY HEAVEN THE WAR ENDED", "IN A TEEN SICKLY HEAVEN THE WAR ENDED", "IN THE WAR ENDED"):
        assert one("So in 1867 the war ended there.", garbled).decision == "nonlexical", garbled
    # an abbreviation with one spoken form is a word: restored even when nothing was heard for it
    r = one("You'll get fixed up comfortable after while Mrs. Shimerda make good house.",
            "YOU'LL GIVE FIXED UP COMFORTABLE AFTER A WHILE SHIMERDA MAKE GOOD HOUSE")
    assert r.label == "YOU'LL GET FIXED UP COMFORTABLE AFTER WHILE MISSUS SHIMERDA MAKE GOOD HOUSE".split()
    assert one("Then Mr. Brown and Dr. Grey came in.", "MISTER BROWN AND DOCTOR GRAY CAME").label == \
        "MISTER BROWN AND DOCTOR GREY CAME".split()
    # several spoken forms: the heard one decides; none heard -> the row is dropped
    assert one("They lived in Baker St. near the park.", "LIVED IN BAKER STREET NEAR THE PARK").label[3] == "STREET"
    assert one("It was St. John who came.", "IT WAS SAINT JOHN WHO CAME").label[2] == "SAINT"
    assert one("They lived in Baker St. near the park.", "LIVED IN BAKER NEAR THE PARK").decision == "nonlexical"


def test_compounds_and_apostrophes():
    r = one("Did anyone come to-day, for ever and ever?", "DID ANY ONE COME TODAY FOREVER AND EVER")
    assert (r.label, r.edits) == ("DID ANYONE COME TO DAY FOR EVER AND EVER".split(), 0)
    assert one("And 'tis the season, 'he said.'", "AND 'TIS THE SEASON HE SAID").label == "AND 'TIS THE SEASON HE SAID".split()
    assert one("Now it's the boys' turn again.", "NOW ITS THE BOYS TURN AGAIN").label == "NOW IT'S THE BOYS TURN AGAIN".split()


def test_edges_are_not_guessed():
    book = "It was a dark night. And the day was long and warm. Then came the rain."
    assert similarity("AND", "AN") >= 0.5 > similarity("NIGHT", "AND")
    r = one(book, "AN THE DAY WAS LONG AND WARM")  # R0 misheard the first word
    assert (r.decision, r.label[0], r.edge_sub) == ("keep", "AND", 1)
    assert one(book, "HAND THE DAY WAS LONG AND WORN").label == "AND THE DAY WAS LONG AND WARM".split()
    assert one(book, "THE DAY WAS LONG AND WARM").label == "THE DAY WAS LONG AND WARM".split()
    assert one(book, "QUITE THE DAY WAS LONG AND WARM").decision == "edge"  # the book's AND is not like QUITE
    assert one(book, "THE DAY WAS LONG AND WARM INDEED").decision == "edge"
    assert one("The day was long.", "WELL THE DAY WAS LONG").decision == "edge"  # nothing before it in the book


def test_edge_repair_needs_timing():
    cuts = [Cut(0.2, 5.0, book_words("And the day was long and warm.")), Cut(5.6, 9.0, book_words("Then came the rain."))]
    hyp = label_words("AND THE DAY WAS LONG AND")  # R0 lost the last word
    r = project(hyp, cuts, 0.0, 5.3)  # the first cut ends inside the row
    assert (r.decision, r.label[-1], r.tail_repair) == ("keep", "WARM", 1)
    r = project(hyp, cuts, 0.0, 4.5)  # it ends after the row: the word may belong to the next row
    assert (r.label[-1], r.tail_repair) == ("AND", 0)
    r = project(label_words("THE DAY WAS LONG AND WARM"), cuts, 0.0, 5.3)
    assert (r.label[0], r.head_repair) == ("AND", 1)
    r = project(label_words("THE DAY WAS LONG AND WARM"), cuts, 1.0, 5.3)  # the cut starts before the row
    assert (r.label[0], r.head_repair) == ("THE", 0)


def test_whole_cut_rows_take_the_whole_book_text():
    cut = [Cut(3.0, 9.0, book_words("\u201cQuite so,\u201d said Mr. Brown in 1867, and left."))]
    r = project(label_words("WHITE SO SAID MISTER BROWN IN EIGHTEEN SIXTY SEVEN AND"), cut, 2.6, 9.5, whole=True)
    assert (r.decision, r.label) == ("keep", "QUITE SO SAID MISTER BROWN IN EIGHTEEN SIXTY SEVEN AND LEFT".split())
    assert (r.sub, r.dele, r.edge_sub, r.tail_repair) == (1, 1, 0, 0)
    assert project(label_words("QUITE SO SAID MISTER BROWN AND LEFT"), cut, 2.6, 9.5, whole=True).decision == "nonlexical"
    plain = [Cut(3.0, 9.0, book_words("Quite so, said the old man, and left."))]
    assert project(label_words("SOMETHING ELSE WAS SAID HERE BY HIM"), plain, 2.6, 9.5, whole=True).decision == "mismatch"


def test_weak_evidence_is_dropped():
    book = "It rained all night. And then the old horse went slowly home, they said. Nobody saw it."
    assert project(label_words("AND THEN"), [], 0.0, 1.0).decision == "no_cut"
    assert one(book, "").decision == "empty_hyp"
    assert one(book, "END THIN THEE OLD HORSE WHEN SLOWLY HOME").decision == "mismatch"
    long = "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen."
    skipped = "ALPHA TWO THREE FOUR FIVE NINE TEN ELEVEN TWELVE THIRTEEN FOURTEEN FIFTEEN"  # the reader left out six seven eight
    assert one(long.replace("one", "alpha"), skipped).decision == "del_run"
    assert one(long, "TWO THREE FOUR SO IT WAS FIVE SIX SEVEN EIGHT NINE TEN ELEVEN TWELVE").decision == "ins_run"
    assert one("He said see x9y here now then.", "SEE EXNINE HERE NOW").decision == "nonlexical"


def test_index_and_runner_end_to_end(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from tools.plbt.run_project import run

    def cut(rec, start, dur, book, asr):
        return json.dumps({"id": f"{rec}_{start}", "start": start, "duration": dur, "channel": 0, "type": "MonoCut",
                           "supervisions": [{"id": "s", "recording_id": rec, "start": 0, "duration": dur,
                                             "custom": {"texts": [book, asr], "pre_texts": ["", ""]}}],
                           "recording": {"id": rec, "sources": [{"type": "file", "source": f"download/librilight/{rec}.flac"}]}})

    rec = "small/100/book_one/01_chapter"
    man = tmp_path / "libriheavy_cuts_small.jsonl.gz"
    with gzip.open(man, "wt", encoding="utf-8") as fh:
        fh.write(cut(rec, 10.0, 6.0, "It rained all night in 1867.",
                     "IT RAINED ALL NIGHT IN EIGHTEEN SIXTY SEVEN") + "\n")
        fh.write(cut(rec, 16.5, 7.0, "And then the old horse went slowly home, Mr. Brown said.",
                     "AND THEN THE OLD HORSE WENT SLOWLY HOME MISTER BROWN SAID") + "\n")
    assert rec_key(f"download/librilight/{rec}.flac") == rec_key(f"{rec}.flac") == rec
    with pytest.raises(ValueError):
        parse_cut(json.dumps({"id": "x", "start": 0, "duration": 1, "supervisions": [], "recording": {"id": rec}}))
    stats = build(str(tmp_path / "lh.sqlite"), [str(man)])
    assert stats["cuts"] == 2 and stats["recordings"] == 1 and stats["manifests"][0]["hours"] == round(13 / 3600, 2)
    with pytest.raises(FileExistsError):
        build(str(tmp_path / "lh.sqlite"), [str(man)])
    shard = tmp_path / "librilight.small"
    shard.mkdir()
    rows = {"id": ["u1", "u2", "u3", "u4"], "src": [rec + ".flac"] * 3 + ["small/9/none/x.flac"],
            "t0": [9.8, 16.3, 16.3, 0.0], "t1": [16.3, 23.9, 23.9, 5.0],
            "text": ["IT RAINED ALL NIGHT IN EIGHTEEN SIXTY SEVEN", "AND THEN THE OLD HOUSE WENT SLOWLY HOME MISTER BROWN",
                     "THIS IS A LIBRIVOX RECORDING", "HELLO THERE"], "audio": [b"x"] * 4}
    pq.write_table(pa.table(rows), shard / "0000.parquet")
    rep = run(str(tmp_path / "lh.sqlite"), str(tmp_path / "out"), "small", [str(shard)])
    assert rep["rows"] == 4 and rep["kept"] == 2 and rep["drops"] == {"mismatch": 1, "no_cut": 1}
    t1 = (tmp_path / "out" / "plbt_v1.small.tsv").read_text().splitlines()
    t0 = (tmp_path / "out" / "r0.rows_of_plbt_v1.small.tsv").read_text().splitlines()
    assert t1 == ["u1\tIT RAINED ALL NIGHT IN EIGHTEEN SIXTY SEVEN",
                  "u2\tAND THEN THE OLD HORSE WENT SLOWLY HOME MISTER BROWN SAID"]
    assert t0[1] == "u2\tAND THEN THE OLD HOUSE WENT SLOWLY HOME MISTER BROWN" and rep["label_differs_from_input"] == 1
    from train.label_map import LabelMap  # the trainer's loader accepts the files as written

    m = LabelMap.load(str(tmp_path / "out" / "plbt_v1.small.tsv"), rep["files"]["plbt_v1.small.tsv"], only=True)
    assert m.keep("u1") and not m.keep("u3") and m.apply({"id": "u2", "text": "x"})["text"].endswith("BROWN SAID")
    audit = [json.loads(x) for x in (tmp_path / "out" / "audit.small.jsonl").read_text().splitlines()]
    assert [a["decision"] for a in audit] == ["keep", "keep", "mismatch", "no_cut"] and audit[1]["tail_repair"] == 1


def test_panel_self_check():
    assert os.path.exists(PANEL), "the committed dev panel file is required"
    from tools.plbt.sim_panel import run

    for layout, bound in (("one_cut", 0.10), ("own_cut", 0.05), ("whole", 0.05)):
        o = run(PANEL, layout=layout)
        assert o["rows"] == 730 and o["kept"] >= 660 and o["worse"] == 0 and o["better"] >= 250
        assert o["kept_r0_wer"] > 3.5 and o["kept_label_wer"] <= bound, o
