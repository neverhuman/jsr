"""The chunked streaming recogniser reproduces the trained (chunk-masked, whole-utterance) model."""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from train.asr import ASRModel, ModelConfig  # noqa: E402
from train.data import Featurizer, collate  # noqa: E402
from train.infer import StreamingRecognizer, run_streaming  # noqa: E402
from train.model import EncoderConfig  # noqa: E402


class _Tok:
    sha256 = "test"

    def encode(self, text):
        return [1 + (ord(c) % 9) for c in text]

    def decode(self, ids):
        return " ".join(str(i) for i in ids)


class _Utt:
    def __init__(self, uid, pcm):
        self.uid, self.text, self.pcm = uid, "x", pcm

    @property
    def seconds(self):
        return len(self.pcm) / 16000


def _model(post_block_norm, left=6, multirate=False):
    torch.manual_seed(0)
    extra = dict(family="multirate", lower_blocks=1, middle_blocks=2, final_blocks=1,
                 chunk=4) if multirate else dict(chunk=3)
    enc = EncoderConfig(d_model=32, n_blocks=2, n_heads=2, stem_channels=8, conv_kernel=5,
                        left_context=left, dropout=0.0, post_block_norm=post_block_norm, **extra)
    m = ASRModel(ModelConfig(encoder=enc, vocab_rnnt=11, vocab_ctc=7, pred_emb=8, joiner=16))
    with torch.no_grad():
        m.cmvn.mean.fill_(-5.0)
        m.cmvn.istd.fill_(0.5)
    return m.eval()


def _utts(n=5, seed=0):
    g = np.random.default_rng(seed)
    return [_Utt(f"u{i}", (g.standard_normal(4000 + 2770 * i) * 3000).astype(np.int16))
            for i in range(n)]


@pytest.mark.parametrize("arch", ["post", "pre", "multirate"])
@pytest.mark.parametrize("streams,packet_ms", [(1, 20), (3, 37), (5, 160)])
def test_streaming_matches_batch(arch, streams, packet_ms):
    torch.set_num_threads(1)
    m = _model(arch == "post", multirate=arch == "multirate")
    tok = _Tok()
    rec = StreamingRecognizer(m, tok, threads=1)
    rec.keep_states = True
    utts = _utts()
    # Streaming, keeping the stream objects to inspect encoder states.
    made = []
    orig = rec.create_stream

    def create():
        s = orig()
        made.append(s)
        return s

    rec.create_stream = create
    hyps = run_streaming(rec, utts, streams, packet_ms)
    feat = Featurizer(tok, tok)
    for u, s, h in zip(utts, made, hyps, strict=True):
        ex = feat.example(u.uid, u.text, u.pcm)
        b = collate([ex])
        with torch.no_grad():
            enc, lens = m.encode(b.feats, b.feat_lens)
            toks, _ = m.greedy(b.feats, b.feat_lens)
        st = torch.cat(s.states, 0)
        assert st.shape[0] == int(lens[0])
        assert torch.allclose(st, enc[0, :int(lens[0])], atol=1e-4), (st - enc[0]).abs().max()
        assert h == tok.decode(toks[0])


@pytest.mark.parametrize("multirate", [False, True])
def test_left_context_window_rolls(multirate):
    # Utterance much longer than the cache: still identical to the masked whole-utterance model.
    torch.set_num_threads(1)
    m = _model(False, left=4, multirate=multirate)
    rec = StreamingRecognizer(m, _Tok(), threads=1)
    rec.keep_states = True
    u = _Utt("long", (np.random.default_rng(1).standard_normal(48000) * 3000).astype(np.int16))
    s = rec.create_stream()
    s.accept_waveform(16000, u.pcm.astype(np.float32) / 32768.0)
    s.input_finished()
    while rec.is_ready(s):
        rec.decode_stream(s)
    ex = Featurizer(_Tok(), _Tok()).example("long", "x", u.pcm)
    b = collate([ex])
    with torch.no_grad():
        enc, lens = m.encode(b.feats, b.feat_lens)
    assert torch.allclose(torch.cat(s.states, 0), enc[0], atol=1e-4)


@pytest.mark.parametrize("multirate", [False, True])
def test_onnx_backend_matches_torch(multirate):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    torch.set_num_threads(1)
    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    d = os.path.join(repo, "target", "test-train", "onnx")
    os.makedirs(d, exist_ok=True)
    m = _model(False, multirate=multirate)
    utts = _utts(4, seed=3)
    want = run_streaming(StreamingRecognizer(m, _Tok(), threads=1), utts, 2, 30)
    got = run_streaming(StreamingRecognizer(m, _Tok(), threads=1, backend="onnx",
                                            onnx_path=os.path.join(d, f"tiny{int(multirate)}.onnx")),
                        utts, 2, 30)
    assert got == want


def test_onnx_int8_backend_runs_and_stays_close():
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime.quantization")
    torch.set_num_threads(1)
    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    d = os.path.join(repo, "target", "test-train", "onnx")
    os.makedirs(d, exist_ok=True)
    m = _model(False)
    rec = StreamingRecognizer(m, _Tok(), threads=1, backend="onnx-int8",
                              onnx_path=os.path.join(d, "tiny_q.onnx"))
    rec.keep_states = True
    ref = StreamingRecognizer(m, _Tok(), threads=1)
    ref.keep_states = True
    u = _utts(1, seed=4)[0]
    outs = []
    for r in (rec, ref):
        s = r.create_stream()
        s.accept_waveform(16000, u.pcm.astype(np.float32) / 32768.0)
        s.input_finished()
        while r.is_ready(s):
            r.decode_stream(s)
        outs.append(torch.cat(s.states, 0))
    rel = (outs[0] - outs[1]).norm() / outs[1].norm()
    assert outs[0].shape == outs[1].shape and float(rel) < 0.25, float(rel)  # random d=32 weights; real-model check is WER
