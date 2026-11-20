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
    man = export_bundle(ck, bdir, int8=True, tokenizer=_Tok())
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
