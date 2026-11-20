import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ultra.wer import corpus_wer, edit_counts, normalise


def test_known_edits():
    assert edit_counts("a b c d".split(), "a x c d e".split()) == (1, 0, 1)
    assert edit_counts("a b c".split(), "".split()) == (0, 3, 0)
    assert edit_counts([], "x y".split()) == (0, 0, 2)


def test_corpus_pooling_and_normalisation():
    r = corpus_wer(["HELLO WORLD", "IT'S A TEST"], ["hello, world!", "its a test"])
    assert r["ref_words"] == 5 and r["sub"] == 1 and r["wer"] == 20.0
    assert normalise("Don't  STOP.") == ["don't", "stop"]
