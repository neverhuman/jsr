"""H12 error breakdown: alignment fates, position and duration bins, and dump comparison."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from train.diagnose import align, breakdown, compare, main  # noqa: E402


def test_align_fates_and_tail_insertions():
    assert align("a b c d".split(), "a x c".split()) == (["C", "S", "C", "D"], 0)
    assert align("a b".split(), "a b c c".split()) == (["C", "C"], 2)
    assert align("a b".split(), []) == (["D", "D"], 0)


def test_breakdown_positions_and_bins():
    rows = [{"uid": "1", "ref": "A B C", "hyp": "a b", "seconds": 3.0},
            {"uid": "2", "ref": "A B C", "hyp": "x b c", "seconds": 12.0}]
    out = breakdown(rows)
    assert out["overall"]["wer"] == round(100 * 2 / 6, 3)
    assert out["by_position"]["last"] == {"words": 2, "S": 0, "D": 1, "err_pct": 50.0, "del_pct": 50.0}
    assert out["by_position"]["first"]["S"] == 1 and out["by_position"]["interior"]["S"] == 0
    assert set(out["by_seconds"]) == {"0-5s", "10-15s"}


def test_compare_and_cli(tmp_path, capsys):
    a = [{"uid": "1", "ref": "A B", "hyp": "a"}, {"uid": "2", "ref": "A", "hyp": "a"}]
    b = [{"uid": "1", "ref": "A B", "hyp": "a b"}, {"uid": "2", "ref": "A", "hyp": "x"}]
    assert compare(a, b) == {"utts": 2, "better": 1, "worse": 1, "same": 0}
    pa, pb = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    pa.write_text("\n".join(json.dumps(r) for r in a))
    pb.write_text("\n".join(json.dumps(r) for r in b))
    assert main([str(pb), "--compare", str(pa), str(pb)]) == 0
    res = json.loads(capsys.readouterr().out)
    assert res["compare"]["better"] == 1 and res["overall"]["ref_words"] == 3
