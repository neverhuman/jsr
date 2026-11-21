"""An exported bundle, read only through train.bundle_infer, reproduces the torch streaming path."""
import os
import shutil
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
from train import checkpoint  # noqa: E402
from train.asr import ASRModel, ModelConfig  # noqa: E402
from train.bundle_infer import BundleError, BundleRecognizer  # noqa: E402
from train.export import export_bundle  # noqa: E402
from train.infer import StreamingRecognizer, run_streaming  # noqa: E402
from train.model import EncoderConfig  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


class _SP:
    def id_to_piece(self, i):
        return ["▁a", "b", "▁c", "d", "e", "▁f", "g", "h", "i", "j"][i]


class _Tok:
    vocab = 10
    sha256 = "test"
    path = __file__
    sp = _SP()

    def decode(self, ids):
        return "".join(self.sp.id_to_piece(i - 1) for i in ids).replace("▁", " ").strip()


class _Utt:
    def __init__(self, uid, pcm):
        self.uid, self.text, self.pcm = uid, "x", pcm

    @property
    def seconds(self):
        return len(self.pcm) / 16000


@pytest.fixture
def workdir(request):
    d = os.path.join(REPO, "target", "test-train", request.node.name)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.mark.parametrize("multirate", [False, True])
def test_bundle_reproduces_streaming_path(workdir, multirate):
    torch.set_num_threads(1)
    torch.manual_seed(0)
    extra = dict(family="multirate", lower_blocks=1, middle_blocks=2, final_blocks=1, chunk=4) \
        if multirate else dict(chunk=3)
    enc = EncoderConfig(d_model=32, n_blocks=2, n_heads=2, stem_channels=8, conv_kernel=5,
                        left_context=6, dropout=0.0, post_block_norm=False, **extra)
    m = ASRModel(ModelConfig(encoder=enc, vocab_rnnt=11, vocab_ctc=7, pred_emb=8, joiner=16))
    with torch.no_grad():
        m.cmvn.mean.fill_(-5.0)
        m.cmvn.istd.fill_(0.5)
        m.joiner.out.bias[0] += 2.0  # mostly blank, like a trained model
    m.eval()
    ck = checkpoint.save(workdir, 1, {"model": m.state_dict(), "model_config": m.cfg.to_dict()})
    bdir = os.path.join(workdir, "bundle")
    man = export_bundle(ck, bdir, int8=True, tokenizer=_Tok(), int8_mode="dynamic")
    assert set(man["files"]) >= {"encoder.fp32", "encoder.int8", "decoder", "joiner.fp32",
                                 "joiner.int8", "tokens", "mel_fbank"}
    g = np.random.default_rng(2)
    utts = [_Utt(f"u{i}", (g.standard_normal(5000 + 3100 * i) * 3000).astype(np.int16))
            for i in range(4)]
    want = run_streaming(StreamingRecognizer(m, _Tok(), threads=1), utts, 3, 30)
    got = run_streaming(BundleRecognizer(bdir, "fp32"), utts, 3, 30)
    assert got == want
    run_streaming(BundleRecognizer(bdir, "int8"), utts, 3, 30)  # int8 loads and runs
    with open(os.path.join(bdir, "tokens.txt"), "a") as fh:
        fh.write("tamper 99\n")
    with pytest.raises(BundleError):
        BundleRecognizer(bdir, "fp32")


def _tiny_dense(multirate=False):
    torch.manual_seed(0)
    extra = dict(family="multirate", lower_blocks=1, middle_blocks=2, final_blocks=1, chunk=4) \
        if multirate else dict(chunk=3)
    enc = EncoderConfig(d_model=32, n_blocks=2, n_heads=2, stem_channels=8, conv_kernel=5,
                        left_context=6, dropout=0.0, post_block_norm=False, **extra)
    m = ASRModel(ModelConfig(encoder=enc, vocab_rnnt=11, vocab_ctc=7, pred_emb=8, joiner=16))
    with torch.no_grad():
        m.cmvn.mean.fill_(-5.0)
        m.cmvn.istd.fill_(0.5)
        m.joiner.out.bias[0] += 2.0
    return m.eval()


@pytest.mark.parametrize("multirate", [False, True])
def test_static_int8_is_batch_invariant(workdir, multirate):
    import onnxruntime as ort

    torch.set_num_threads(1)
    m = _tiny_dense(multirate)
    chunk = m.cfg.encoder.chunk
    ck = checkpoint.save(workdir, 1, {"model": m.state_dict(), "model_config": m.cfg.to_dict()})
    g = np.random.default_rng(5)
    calib = [_Utt(f"c{i}", (g.standard_normal(6000 + 1500 * i) * 2500).astype(np.int16))
             for i in range(12)]
    bdir = os.path.join(workdir, "bundle")
    man = export_bundle(ck, bdir, int8=True, tokenizer=_Tok(), int8_mode="static_qdq",
                        calib_utts=calib, calib_manifest={"selection_sha256": "t", "n": 12, "strata": {}},
                        joiner_int8=True)
    assert man["int8"]["mode"] == "static_qdq" and man["int8"]["batch_invariant"]
    import json
    with open(os.path.join(bdir, "calibration_manifest.json")) as fh:
        cm = json.load(fh)["consumed_rows"]
    assert cm["encoder"]["kept"] == len(cm["encoder"]["rows"]) <= 384
    assert cm["encoder"]["offered"] >= cm["encoder"]["kept"] > 0
    assert all(r["uid"].startswith("c") for r in cm["encoder"]["rows"] + cm["joiner"]["rows"])
    assert all(len(r["ctx"]) == 2 for r in cm["joiner"]["rows"])
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    enc = ort.InferenceSession(os.path.join(bdir, "encoder.int8.onnx"), so,
                               providers=["CPUExecutionProvider"])
    joi = ort.InferenceSession(os.path.join(bdir, "joiner.int8.onnx"), so,
                               providers=["CPUExecutionProvider"])
    names = [i.name for i in enc.get_inputs()]
    assert not any(n.op_type == "DynamicQuantizeLinear"
                   for n in __import__("onnx").load(os.path.join(bdir, "encoder.int8.onnx")).graph.node)

    def enc_batch(bsz, seed):
        r = np.random.default_rng(seed)
        feed = {}
        for i in enc.get_inputs():
            shape = [bsz if (isinstance(d, str) or d is None) and k in (0, 1) and
                     (k == 0) == (i.name in ("feats", "t0", "kv_len", "kvm_len")) else d
                     for k, d in enumerate(i.shape)]
            if i.name == "feats":
                shape = [bsz, 4 * chunk + 3, 80]
            if i.type == "tensor(int64)":
                feed[i.name] = r.integers(0, 6, size=shape).astype(np.int64)
            else:
                feed[i.name] = r.normal(0, 1, size=shape).astype(np.float32)
        return feed

    base = enc_batch(1, 0)
    out1 = enc.run(None, base)
    for bsz in (2, 8, 16):
        f = enc_batch(bsz, bsz)
        for nm in names:
            ax = 1 if nm in ("k", "v", "conv", "km", "vm", "convm") else 0
            idx = [slice(None)] * f[nm].ndim
            idx[ax] = slice(0, 1)
            f[nm][tuple(idx)] = base[nm]
        outb = enc.run(None, f)
        for o1, ob, info in zip(out1, outb, enc.get_outputs(), strict=True):
            ax = 1 if info.name.split("_out")[0] in ("k", "v", "conv", "km", "vm", "convm") else 0
            assert np.array_equal(o1, np.take(ob, [0], axis=ax)), (bsz, info.name)
    e1 = np.random.default_rng(1).normal(0, 1, (1, 16)).astype(np.float32)
    d1 = np.random.default_rng(2).normal(0, 1, (1, 16)).astype(np.float32)
    l1 = joi.run(None, {"enc": e1, "dec": d1})[0]
    for bsz in (2, 8, 16):
        e = np.random.default_rng(bsz).normal(0, 3, (bsz, 16)).astype(np.float32)
        d = np.random.default_rng(bsz + 1).normal(0, 3, (bsz, 16)).astype(np.float32)
        e[0], d[0] = e1[0], d1[0]
        assert np.array_equal(joi.run(None, {"enc": e, "dec": d})[0][:1], l1), bsz


def test_reservoir_is_uniform_and_capped():
    from train.calib import _Reservoir

    counts = np.zeros(100)
    for seed in range(300):
        r = _Reservoir(10, np.random.default_rng(seed))
        for i in range(100):
            r.offer(lambda i=i: i)
        assert len(r.items) == 10 and r.seen == 100
        counts[r.items] += 1
    assert counts.min() > 10 and counts.max() < 60  # expectation 30 per item
