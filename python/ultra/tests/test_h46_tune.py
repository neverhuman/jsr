"""H46 tuning helpers: the dev split v2 reader, its coverage check and the family ordering."""
import json

import numpy as np
import pytest

pytest.importorskip("torch")  # train.h46_tune imports train.lm_tune -> train.lm_decode -> torch

from train.h46_tune import families_of, load_split, masks  # noqa: E402


def _split(tmp_path, sel, conf):
    p = tmp_path / "split.json"
    p.write_text(json.dumps({"utterances": {"selection": sel, "confirmation": conf}}))
    return str(p)


def test_load_split_maps_each_uid_to_its_half(tmp_path):
    half, sha = load_split(_split(tmp_path, ["a", "b"], ["c"]))
    assert half == {"a": "selection", "b": "selection", "c": "confirmation"}
    assert len(sha) == 64


def test_load_split_refuses_a_uid_in_both_halves(tmp_path):
    with pytest.raises(ValueError, match="both halves"):
        load_split(_split(tmp_path, ["a"], ["a"]))


def test_masks_need_exact_coverage(tmp_path):
    half, _ = load_split(_split(tmp_path, ["a"], ["b"]))
    sel, conf = masks([{"uid": "a"}, {"uid": "b"}], half)
    assert sel.tolist() == [True, False] and conf.tolist() == [False, True]
    with pytest.raises(ValueError, match="exactly"):
        masks([{"uid": "a"}], half)  # the split holds an utterance the cache does not
    with pytest.raises(ValueError, match="exactly"):
        masks([{"uid": "a"}, {"uid": "z"}], half)


def test_families_are_ordered_by_preference():
    grid = [{"name": "x2", "family": "f", "lm_weight": 0.5, "lodr_weight": 0.2, "blank_penalty": 0},
            {"name": "x1", "family": "f", "lm_weight": 0.3, "lodr_weight": 0.3, "blank_penalty": 0},
            {"name": "x3", "family": "f", "lm_weight": 0.3, "lodr_weight": 0.1, "blank_penalty": 1},
            {"name": "c", "family": "ctrl", "blank_penalty": 0.25}]
    fams = families_of(grid)
    assert [c["name"] for c in fams["f"]] == ["x3", "x1", "x2"]
    assert [c["name"] for c in fams["ctrl"]] == ["c"]
    assert np.argmin([3, 1, 1]) == 1  # ties resolve to the earliest, i.e. the preferred config
