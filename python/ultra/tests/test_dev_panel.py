import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tools.dev_panel import select


def test_select_is_stratified_deterministic_and_capped():
    uids = [f"{s}-x-{i:04d}" for s in (1, 2, 3) for i in range(20)]
    speakers = [int(u.split("-")[0]) for u in uids]
    a = select(uids, speakers, 5)
    assert len(a) == 15
    assert {u.split("-")[0] for u in a} == {"1", "2", "3"}
    assert all(sum(u.startswith(f"{s}-") for u in a) == 5 for s in (1, 2, 3))
    assert a == select(list(reversed(uids)), list(reversed(speakers)), 5)  # order-independent
    assert a == sorted(a)


def test_select_keeps_all_when_speaker_has_fewer():
    assert select(["9-a", "9-b"], [9, 9], 10) == ["9-a", "9-b"]
