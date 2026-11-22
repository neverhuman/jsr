"""H42 n-gram LM decode: fusion contexts and the beam hook, KenLM rows (when the optional kenlm
module is installed), encoder-cache parts, speaker folds, held-out choice and paired bootstrap."""
import json
import math
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from train import lm_decode, lm_fusion, lm_tune  # noqa: E402
from train.beam import modified_beam_search  # noqa: E402
from train.transducer import Joiner, StatelessPredictor  # noqa: E402

V = 12


def _parts(seed=0):
    torch.manual_seed(seed)
    pred, join = StatelessPredictor(V, emb=8, joiner=16), Joiner(24, 16, V)
    enc_p = join.enc_proj(torch.randn(2, 20, 24))
    return pred.eval(), join.eval(), enc_p.detach(), torch.tensor([20, 13])


def test_context_and_narrow():
    bos = lm_fusion.BOS
    assert lm_fusion.context((), 2) == (bos,)
    assert lm_fusion.context((5,), 2) == (bos, 5)
    assert lm_fusion.context((4, 5, 6), 2) == (5, 6)
    assert lm_fusion.context((4, 5, 6), 0) == ()
    assert lm_fusion.narrow((bos, 5), 1) == (5,)
    assert lm_fusion.narrow((bos,), 1) == (bos,)
    assert lm_fusion.narrow((4, 5, 6), 1) == (6,)
    assert lm_fusion.narrow((bos, 5), 3) == (bos, 5)
    assert lm_fusion.narrow((4, 5), 0) == ()
    with pytest.raises(ValueError):
        lm_fusion.narrow((5,), 3)


class _FakeLm:
    """Stands in for TokenNgram: a row that depends on the context only."""

    def __init__(self, order):
        self.order, self.calls = order, []

    def row(self, ctx):
        self.calls.append(ctx)
        out = torch.zeros(V)
        out[1:] = -1.0 - 0.01 * torch.arange(1, V) - 0.1 * (sum(ctx) % 7)
        return out


def test_fusion_combines_lm_and_lodr_rows_and_caches_by_context():
    lm, lodr = _FakeLm(3), _FakeLm(2)
    f = lm_fusion.NgramFusion(lm, 0.5, lodr, 0.2)
    seqs = [(), (3,), (3, 4), (9, 3, 4)]
    got = f(seqs)
    assert got.shape == (4, V) and bool((got[:, 0] == 0).all())
    for row, s in zip(got, seqs, strict=True):
        want = 0.5 * _FakeLm(3).row(lm_fusion.context(s, 2)) \
            - 0.2 * _FakeLm(2).row(lm_fusion.context(s, 1))
        assert torch.allclose(row, want)
    assert torch.equal(got[2], got[3])  # same last two tokens, same row
    assert np.array_equal(f.stack(seqs), got.numpy()) and f.row((3, 4)).dtype == np.float32
    n = len(lm.calls)
    f(seqs)
    assert len(lm.calls) == n  # second pass is served from the cache
    plain = lm_fusion.NgramFusion(lm, 0.5)
    assert plain.lodr is None and plain.n == 2
    with pytest.raises(ValueError):
        lm_fusion.NgramFusion(lm, -0.1)
    with pytest.raises(ValueError):
        lm_fusion.NgramFusion(lm, 0.1, None, 0.3)


def test_beam_hook_zero_fusion_changes_nothing_and_a_bonus_equals_the_blank_penalty():
    pred, join, enc_p, lens = _parts(3)
    nb0, nb1 = [], []
    base = modified_beam_search(pred, join, enc_p, lens, beam=4, nbest=nb0)
    zero = modified_beam_search(pred, join, enc_p, lens, beam=4, nbest=nb1,
                                fusion=lambda seqs: torch.zeros(len(seqs), V))
    assert zero == base and nb0 == nb1

    def bonus(seqs):  # +1.5 on every non-blank token is a blank penalty of 1.5
        out = torch.full((len(seqs), V), 1.5)
        out[:, 0] = 0.0
        return out

    assert modified_beam_search(pred, join, enc_p, lens, beam=4, fusion=bonus) == \
        modified_beam_search(pred, join, enc_p, lens, beam=4, blank_penalty=1.5)


def test_beam_hook_lm_can_forbid_tokens():
    pred, join, enc_p, lens = _parts(5)

    def only_three(seqs):
        out = torch.full((len(seqs), V), -1e9)
        out[:, 0] = 0.0
        out[:, 3] = 5.0
        return out

    out = modified_beam_search(pred, join, enc_p, lens, beam=4, fusion=only_three)
    assert any(out) and all(k == 3 for s in out for k in s)


ARPA = """\\data\\
ngram 1=6
ngram 2=3

\\1-grams:
-1.0\t<unk>
0\t<s>\t-0.30103
-0.69897\t</s>
-0.39794\t1\t-0.30103
-0.52288\t2\t-0.1
-0.69897\t3

\\2-grams:
-0.1\t<s> 1
-0.2\t1 2
-0.3\t2 1

\\end\\
"""


def test_token_ngram_rows_match_the_arpa_file(tmp_path):
    pytest.importorskip("kenlm")
    path = tmp_path / "toy.arpa"
    path.write_text(ARPA)
    lm = lm_fusion.TokenNgram(str(path), 5)  # tokens 0 (blank) .. 4; "4" is unknown to the LM
    assert lm.order == 2
    ln10 = math.log(10.0)
    start = lm.row((lm_fusion.BOS,))
    assert start[0] == 0.0
    want = [-0.1, -0.30103 - 0.52288, -0.30103 - 0.69897, -0.30103 - 1.0]
    assert torch.allclose(start[1:], torch.tensor(want) * ln10, atol=1e-4)
    after1 = lm.row((1,))
    want = [-0.30103 - 0.39794, -0.2, -0.30103 - 0.69897, -0.30103 - 1.0]
    assert torch.allclose(after1[1:], torch.tensor(want) * ln10, atol=1e-4)
    after3 = lm.row((3,))  # "3" has no backoff weight: plain unigrams
    assert torch.allclose(after3[1:], torch.tensor([-0.39794, -0.52288, -0.69897, -1.0]) * ln10,
                          atol=1e-4)
    fusion = lm_fusion.NgramFusion(lm, 2.0)
    assert torch.allclose(fusion([(7, 1)])[0], 2.0 * after1)
    assert lm.lookups > 0


def test_token_line_encodes_word_by_word_with_a_cache():
    class Sp:
        calls = 0

        def encode(self, w):
            Sp.calls += 1
            return [ord(c) - 65 for c in w]

    cache: dict = {}
    assert lm_decode.token_line(Sp(), cache, "AB  C AB\n") == "1 2 3 1 2"
    assert Sp.calls == 2 and lm_decode.token_line(Sp(), cache, "") == ""


def test_encode_parts_merge_into_one_cache_in_length_order(tmp_path):
    rng = np.random.default_rng(0)
    frames = [3, 5, 2, 4]  # utterance at position i has frames[i] frames
    data = [rng.standard_normal((n, 6)).astype(np.float32) for n in frames]
    head = {"schema": lm_decode.CACHE_SCHEMA, "split": "dev-clean", "checkpoint_sha256": "ab",
            "step": 7, "weights": "ema", "chunk": 4, "left_context": 128, "batch": 1,
            "order": "length", "max_symbols": 8, "tokenizer_sha256": "cd", "utterances": 4}
    for shard in (0, 1):
        pos = [p for p in range(4) if p % 2 == shard]
        rows, off = [], 0
        for p in pos:
            rows.append({"uid": f"u{p}", "speaker": 10 + p, "ref": "A", "offset": off,
                         "frames": frames[p], "pos": p, "seconds": 0.1, "greedy": [p + 1]})
            off += frames[p]
        part = tmp_path / f"dev-clean.part{shard:02d}of02"
        np.save(str(part) + ".npy", np.concatenate([data[p] for p in pos]))
        (tmp_path / f"dev-clean.part{shard:02d}of02.json").write_text(
            json.dumps({**head, "encode_seconds": 1.0, "utts": rows}))

    class A:
        out, splits = str(tmp_path), ["dev-clean"]

    lm_decode.merge(A)
    index, enc = lm_decode.load_cache(str(tmp_path), "dev-clean")
    assert [u["uid"] for u in index["utts"]] == ["u0", "u1", "u2", "u3"]
    assert index["frames"] == sum(frames) and not list(tmp_path.glob("*.part*"))
    for u, x in zip(index["utts"], data, strict=True):
        assert np.array_equal(enc[u["offset"]:u["offset"] + u["frames"]], x)
    with pytest.raises(ValueError):
        lm_decode.load_cache(str(tmp_path), "test-clean")
    with pytest.raises(ValueError):
        lm_decode.merge(A)  # parts are gone


def test_speaker_folds_and_held_out_choice():
    assert lm_tune.speaker_folds([30, 10, 20, 10]) == {10: 0, 20: 1, 30: 0}
    fold = np.array([0, 0, 1, 1])
    err = np.array([[2, 2, 2, 2],    # config 0: never best
                    [0, 0, 3, 3],    # best on fold A
                    [3, 3, 0, 0],    # best on fold B
                    [0, 0, 3, 3]])   # ties with config 1 on fold A: the earlier one wins
    pick, chosen = lm_tune.held_out(err, fold)
    assert chosen == [1, 2]
    assert pick.tolist() == [2, 2, 1, 1]  # each fold is scored with the other fold's choice
    assert lm_tune.choose(err, np.ones(4, dtype=bool)) == 1  # 6 errors, first of three ties
    only_b = np.array([False, False, True, True])
    assert lm_tune.held_out(err, fold, only_b)[1] == [0, 2]  # nothing to tune on in fold A


def test_count_errors_and_wer_row():
    got = lm_tune.count_errors([("A B C", "a x c"), ("A B", "a b d"), ("A B C", "a x c")])
    assert got == [(1, 0, 0), (0, 0, 1), (1, 0, 0)]
    row = lm_tune.wer_row(np.array(got), np.array([3, 2, 3]))
    assert row["wer"] == 37.5 and row["ref_words"] == 8 and row["utts"] == 3
    assert lm_tune.wer_row(np.array(got), np.array([3, 2, 3]),
                           np.array([True, False, False]))["wer"] == 33.333


def test_paired_bootstrap_interval():
    rng = np.random.default_rng(1)
    n = 400
    nref = rng.integers(5, 30, n)
    base = rng.integers(0, 5, n)
    strata = np.array(["a"] * 250 + ["b"] * 150)
    same = lm_tune.bootstrap(base, base, nref, strata, resamples=500)
    assert same["diff_pp"] == 0.0 and same["lo"] == 0.0 and not same["excludes_zero"]
    better = lm_tune.bootstrap(np.maximum(base - 1, 0), base, nref, strata, resamples=500)
    assert better["hi"] < 0 < -better["diff_pp"] and better["excludes_zero"]
    assert better == lm_tune.bootstrap(np.maximum(base - 1, 0), base, nref, strata, resamples=500)
    noise = base + rng.integers(-1, 2, n)
    weak = lm_tune.bootstrap(noise, base, nref, strata, resamples=500)
    assert weak["lo"] < weak["diff_pp"] < weak["hi"]
    spk = np.arange(n) // 20
    by_speaker = lm_tune.bootstrap(np.maximum(base - 1, 0), base, nref, strata, spk, resamples=500)
    assert by_speaker["unit"] == "speaker" and by_speaker["hi"] < 0
    # the cached count matrices are exactly the explicit draws
    arm = np.maximum(base - 1, 0)
    rng = np.random.default_rng(lm_tune.SEED)
    d_err, n_ref = np.zeros(500), np.zeros(500)
    for s in ("a", "b"):
        m = strata == s
        idx = rng.integers(0, int(m.sum()), size=(500, int(m.sum())))
        d_err += (arm[m] - base[m])[idx].sum(axis=1)
        n_ref += nref[m][idx].sum(axis=1)
    lo, hi = np.percentile(100.0 * d_err / n_ref, [2.5, 97.5])
    assert (better["lo"], better["hi"]) == (round(float(lo), 3), round(float(hi), 3))


def test_compare_reports_each_set_and_pooled():
    utts = [{"nref": 10, "split": "dev-clean" if i < 6 else "dev-other", "speaker": i % 4,
             "fold": i % 2} for i in range(12)]
    base = np.array([[1, 0, 0]] * 12)
    arm = np.array([[0, 0, 0]] * 6 + [[1, 0, 0]] * 6)
    res = lm_tune.compare(arm, base, utts)
    assert res["dev-clean"]["arm"]["wer"] == 0.0 and res["dev-clean"]["baseline"]["wer"] == 10.0
    assert res["dev-other"]["bootstrap"]["diff_pp"] == 0.0
    assert res["pooled"]["arm"]["wer"] == 5.0 and res["pooled"]["rel_change_pct"] == -50.0
    cv, pick = lm_tune.cross_validate(np.stack([base, arm]), base, utts, ["none", "lm"])
    assert pick.tolist() == [1] * 12
    assert cv["chosen"] == {"tuned_on_A_scored_on_B": "lm", "tuned_on_B_scored_on_A": "lm"}
    assert cv["held_out"]["pooled"]["arm"]["wer"] == 5.0
    assert cv["tuned_on_itself"]["label"] == "tuned on itself"


def test_fast_beam_matches_the_reference_beam():
    for seed, beam, pen in ((0, 4, 0.0), (1, 8, 0.0), (2, 4, 0.7), (3, 1, 0.0)):
        pred, join, enc_p, lens = _parts(seed)
        fast = lm_fusion.FastBeam(pred, join)
        for fusion in (None, lm_fusion.NgramFusion(_FakeLm(3), 0.4, _FakeLm(2), 0.1)):
            nb = []
            modified_beam_search(pred, join, enc_p, lens, beam=beam, blank_penalty=pen, nbest=nb,
                                 fusion=fusion)
            for b, want in enumerate(nb):
                got = fast.search(enc_p[b, :int(lens[b])].numpy(), beam, pen, fusion)
                assert [t for t, _ in got] == [t for t, _ in want]
                assert np.allclose([s for _, s in got], [s for _, s in want], atol=1e-3)
    with pytest.raises(ValueError):
        fast.search(enc_p[0].numpy(), beam=0)
    pred.context_len = 4  # a longer predictor context is refused, not silently truncated
    with pytest.raises(ValueError):
        lm_fusion.FastBeam(pred, join)


def test_edge_rule_extends_only_axes_whose_choice_sits_on_an_edge():
    grid = lm_tune.make_grid("T", "/lm", "/lodr", kind="reduced")
    fam_f, fam_l = "T_b4_fusion", "T_b4_fusion_lodr"

    def lab(name):
        c = next(x for x in grid if x["name"] == name)
        return {k: c[k] for k in ("name", "lm_weight", "lodr_weight", "blank_penalty") if k in c}

    res = {"families": {
        fam_f: {"chosen": {"a": lab("T_b4_l0.30_r0.00_p0.25"), "b": lab("T_b4_l0.30_r0.00_p0.00")}},
        fam_l: {"chosen": {"a": lab("T_b4_l0.60_r0.50_p0.00"), "b": lab("T_b4_l0.20_r0.50_p0.00")}}}}
    new = lm_tune.extensions(grid, res)
    names = sorted(c["name"] for c in new)
    # fusion: penalty 0.25 is the top of {0, 0.25} -> 0.5 for every LM weight; 0.3 is interior
    assert [n for n in names if "r0.00" in n] == [f"T_b4_l{w:.2f}_r0.00_p0.50"
                                                 for w in (0.1, 0.2, 0.3, 0.4, 0.5)]
    # LODR: 0.6 is the top -> 0.7; 0.2 is the bottom -> 0.1; the ratio axis has one value
    lodr = [c for c in new if c["family"] == fam_l]
    assert sorted({c["lm_weight"] for c in lodr}) == [0.1, 0.7]
    assert all(c["lodr_weight"] == round(0.5 * c["lm_weight"], 6) for c in lodr)
    assert all(c["lm"] == "/lm" for c in new) and all(c["lodr"] == "/lodr" for c in lodr)
    assert not any("lodr" in c for c in new if c["family"] == fam_f)
    assert lm_tune.extensions(grid + new, res) == []  # the same choices add nothing twice


def test_complete_shards_picks_a_full_set_whatever_the_shard_count(tmp_path):
    d = tmp_path / "cfg"
    d.mkdir()
    for i in range(3):
        (d / f"dev-clean.{i:02d}of03.jsonl").write_text(json.dumps({"uid": f"u{i}"}) + "\n")
    for i in (0, 2):
        (d / f"dev-clean.{i:02d}of04.jsonl").write_text("")
    (d / "dev-clean.01of04.jsonl.tmp").write_text("")
    got = lm_decode.complete_shards(str(d), "dev-clean")
    assert [os.path.basename(f) for f in got] == [f"dev-clean.{i:02d}of03.jsonl" for i in range(3)]
    assert lm_decode.complete_shards(str(d), "dev-other") is None
    rows = lm_tune.load_run(str(tmp_path), "cfg", ["dev-clean"], 3)
    assert sorted(rows) == ["u0", "u1", "u2"]
    with pytest.raises(ValueError):
        lm_tune.load_run(str(tmp_path), "cfg", ["dev-other"])
