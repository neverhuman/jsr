import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ultra.slices import FAST_WPS, LONG_FORM_S, slice_of, words_per_second  # noqa: E402


def test_rules():
    assert slice_of("A B", 21.0) == ["long_form"]
    n = int(FAST_WPS * 2) + 1
    assert slice_of(" ".join(["W"] * n), 2.0) == ["fast_speech"]
    assert slice_of("A B C", LONG_FORM_S) == []
    assert abs(words_per_second("A B C D", 2.0) - 2.0) < 1e-12
