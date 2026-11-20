import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ultra import word_timing as wt

TG = '''File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0.0
xmax = 2.0
tiers? <exists>
size = 2
item []:
	item [1]:
		class = "IntervalTier"
		name = "words"
		xmin = 0.0
		xmax = 2.0
		intervals: size = 6
			intervals [1]:
				xmin = 0.0
				xmax = 0.30
				text = ""
			intervals [2]:
				xmin = 0.30
				xmax = 0.60
				text = "wicker"
			intervals [3]:
				xmin = 0.60
				xmax = 0.70
				text = "s"
			intervals [4]:
				xmin = 0.70
				xmax = 1.10
				text = "<unk>"
			intervals [5]:
				xmin = 1.10
				xmax = 1.50
				text = "boat"
			intervals [6]:
				xmin = 1.50
				xmax = 2.0
				text = ""
	item [2]:
		class = "IntervalTier"
		name = "phones"
		xmin = 0.0
		xmax = 2.0
		intervals: size = 1
			intervals [1]:
				xmin = 0.0
				xmax = 2.0
				text = "sil"
'''


def test_parse_drops_silence_and_ignores_phone_tier():
    toks = wt.parse_textgrid_words(TG)
    assert [t[0] for t in toks] == ["wicker", "s", "<unk>", "boat"]
    assert toks[0][1:] == (0.30, 0.60)


def test_attach_merges_apostrophe_pieces_and_fills_unk():
    words = wt.attach_reference(wt.parse_textgrid_words(TG), "WICKER'S GWYNPLAINE BOAT".split())
    assert words == [wt.Word("WICKER'S", 0.3, 0.7), wt.Word("GWYNPLAINE", 0.7, 1.1),
                     wt.Word("BOAT", 1.1, 1.5)]


def test_attach_rejects_mismatch():
    toks = wt.parse_textgrid_words(TG)
    assert wt.attach_reference(toks, "WICKER'S X SHIP".split()) is None
    assert wt.attach_reference(toks, "WICKER'S X".split()) is None  # leftover token
    assert wt.attach_reference(toks, "WICKER'S X BOAT EXTRA".split()) is None


def test_unk_followed_by_s_merges():
    toks = [("<unk>", 0.1, 0.4), ("s", 0.4, 0.5), ("go", 0.5, 0.8)]
    assert wt.attach_reference(toks, "GOBEY'S GO".split()) == [wt.Word("GOBEY'S", 0.1, 0.5),
                                                               wt.Word("GO", 0.5, 0.8)]


def test_jsonl_roundtrip_and_loader(tmp_path):
    words = [wt.Word("HELLO", 0.1, 0.4), wt.Word("WORLD", 0.45, 0.9)]
    (tmp_path / "test-clean.jsonl").write_text(wt.to_jsonl_line("1-2-3", words) + "\n")
    got = wt.load("test-clean", root=str(tmp_path))
    assert got == {"1-2-3": words}
    with pytest.raises(KeyError):
        wt.load("train-clean-100", root=str(tmp_path))


def test_loader_rejects_duplicates_and_bad_order(tmp_path):
    line = wt.to_jsonl_line("1-2-3", [wt.Word("A", 0.1, 0.2)])
    (tmp_path / "dev-clean.jsonl").write_text(line + "\n" + line + "\n")
    with pytest.raises(ValueError):
        wt.load("dev-clean", root=str(tmp_path))
    with pytest.raises(ValueError):
        wt.validate([wt.Word("A", 0.5, 0.6), wt.Word("B", 0.2, 0.3)])
    with pytest.raises(ValueError):
        wt.validate([wt.Word("a", 0.1, 0.2)])
    with pytest.raises(ValueError):
        wt.validate([wt.Word("A", 0.1, 2.5)], audio_s=2.0)


def test_energy_offset_finds_tone_in_noise():
    rng = np.random.default_rng(0)
    sr = 16000
    x = rng.normal(0, 30, 3 * sr)
    t = np.arange(sr) / sr
    x[sr:2 * sr] += 8000 * np.sin(2 * np.pi * 220 * t)
    on, off = wt.energy_offset(x.astype(np.int16))
    assert abs(on - 1.0) < 0.03 and abs(off - 2.0) < 0.03


def test_member_allowlist_rejects_traversal():
    assert wt.member_relpath("test-clean/61/70970/61-70970-0007.TextGrid").endswith(".TextGrid")
    assert wt.member_relpath("test-other/unaligned.txt") == "test-other/unaligned.txt"
    for bad in ["../etc/passwd", "test-clean/../../x.TextGrid", "/abs/61-1-1.TextGrid",
                "train-clean-100/1/2/1-2-3.TextGrid", "test-clean/61/70970/evil.py"]:
        with pytest.raises(ValueError):
            wt.member_relpath(bad)
