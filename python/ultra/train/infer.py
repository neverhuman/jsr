"""Streaming inference for the R1 student, with the recogniser interface the SP-A harness drives.

  rec = StreamingRecognizer.from_checkpoint(path, threads=1)
  s = rec.create_stream(); s.accept_waveform(16000, float32_pcm); s.input_finished()
  while rec.is_ready(s): rec.decode_streams([s])     # or decode_stream(s)
  text = rec.get_result(s)

Execution is chunk by chunk with explicit caches, never over the whole utterance:
  * log-mel frames are computed as soon as their 25 ms window is complete;
  * the causal stem runs on the feature slice a chunk needs (4 t0 - 6 .. 4 t1 - 4);
  * each block keeps its last `left_context` attention keys/values (RoPE already applied at their
    absolute positions) and the last kernel-1 inputs of its causal depthwise convolution;
  * greedy transducer decoding follows each chunk (cap 8 symbols per frame, spec v1.0-B §17.7).
The arithmetic is the trained chunk mask unrolled, so outputs match `ASRModel.greedy` on the
same audio up to float rounding (tests/test_train_infer.py). `decode_streams` batches all ready
streams into one weight pass per chunk; streams whose final chunk is partial run as their own group.

fp32 eager PyTorch on CPU. Use the MKL build (`PYTHONPATH=/mnt/xbabe3-external/jsr/venvs/torch270-cpu`):
the system torch has a generic BLAS and is about 30x slower. `ChunkEncoder.forward` takes and
returns tensors only, so it can be exported (ONNX/TorchScript) for the int8 step.
"""
from __future__ import annotations

import argparse
import json
import resource
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from train.asr import ASRModel, ModelConfig
from train.errors import ShapeError
from train.features import HOP, WIN, LogMel
from train.model import Encoder, apply_rope, stem_out_len
from train.transducer import BLANK


class ChunkEncoder(nn.Module):
    """One chunk of the trained encoder for a batch of streams, with explicit caches.

    feats   (B, 4 n + 3, 80)  normalised log-mel for features 4 t0 - 6 .. 4 t1 - 4 (zeros < 0)
    t0      (B,)              absolute index of the chunk's first encoder frame
    k, v    (layers, B, H, Lc, dh)  attention cache, oldest first
    kv_len  (B,)              valid cache entries (right-aligned: the last kv_len slots)
    conv    (layers, B, kernel - 1, d)  depthwise-convolution input history
    returns states (B, n, d) and the updated caches.
    """

    def __init__(self, enc: Encoder):
        super().__init__()
        self.enc = enc
        c = enc.cfg
        self.h, self.dh, self.left, self.k = c.n_heads, c.d_model // c.n_heads, c.left_context, \
            c.conv_kernel

    def _stem(self, feats: torch.Tensor, t0: torch.Tensor, n: int) -> torch.Tensor:
        st = self.enc.stem
        y = F.silu(st.c1(feats.unsqueeze(1)))  # layer-1 outputs s = 2 t0 - 2 + j
        s_idx = 2 * t0[:, None] - 2 + torch.arange(y.shape[2], device=feats.device)[None]
        y = y.masked_fill((s_idx < 0)[:, None, :, None], 0.0)  # layer-2 left padding is zeros
        y = F.silu(st.c2(y))
        b, ch, t, f = y.shape
        if t != n:
            raise ShapeError(f"stem produced {t} frames, expected {n}")
        return st.out(y.permute(0, 2, 1, 3).reshape(b, t, ch * f))

    def forward(self, feats, t0, k, v, kv_len, conv):
        b = feats.shape[0]
        n = (feats.shape[1] - 3) // 4
        x = self._stem(feats, t0, n)
        lc = k.shape[3]
        dev = x.device
        inv = 1.0 / (10000 ** (torch.arange(0, self.dh, 2, device=dev, dtype=torch.float32)
                               / self.dh))
        pos = (t0[:, None] + torch.arange(n, device=dev)[None]).float()  # (B, n)
        ang = pos[..., None] * inv  # (B, n, half)
        cos, sin = ang.cos()[:, None], ang.sin()[:, None]  # (B, 1, n, half)
        slot = torch.arange(lc, device=dev)[None]
        key_ok = torch.cat([slot >= (lc - kv_len[:, None]),
                            torch.ones(b, n, dtype=torch.bool, device=dev)], dim=1)
        mask = key_ok[:, None, None, :]  # (B, 1, 1, Lc + n): every query sees all valid keys
        new_k, new_v, new_conv = [], [], []
        for li, blk in enumerate(self.enc.blocks):
            x = x + 0.5 * blk.ff1(x)
            att = blk.att
            q, kk, vv = att.qkv(att.norm(x)).view(b, n, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
            q, kk = apply_rope(q, cos, sin), apply_rope(kk, cos, sin)
            kall = torch.cat([k[li], kk], dim=2)
            vall = torch.cat([v[li], vv], dim=2)
            y = F.scaled_dot_product_attention(q, kall, vall, attn_mask=mask)
            x = x + att.out(y.transpose(1, 2).reshape(b, n, -1))
            new_k.append(kall[:, :, -lc:])
            new_v.append(vall[:, :, -lc:])
            cm = blk.conv
            g = F.glu(cm.pw1(cm.norm(x)), dim=-1)
            gall = torch.cat([conv[li], g], dim=1)
            new_conv.append(gall[:, -(self.k - 1):])
            y = cm.dw(gall.transpose(1, 2)).transpose(1, 2)
            x = x + cm.pw2(F.silu(cm.dnorm(y)))
            x = x + 0.5 * blk.ff2(x)
            x = blk.norm(x)
        x = self.enc.final_norm(x)
        kv_len = torch.clamp(kv_len + n, max=lc)
        return x, torch.stack(new_k), torch.stack(new_v), kv_len, torch.stack(new_conv)


_IO_IN = ["feats", "t0", "k", "v", "kv_len", "conv"]
_IO_OUT = ["x", "k_out", "v_out", "kv_len_out", "conv_out"]


def export_onnx(enc: ChunkEncoder, path: str) -> None:
    """Export the chunk encoder (fp32, dynamic stream batch) for onnxruntime."""
    c = enc.enc.cfg
    L, H, dh, lc, n = c.n_blocks, c.n_heads, c.d_model // c.n_heads, c.left_context, c.chunk
    b = 2
    args = (torch.zeros(b, 4 * n + 3, c.n_mels), torch.tensor([0, 4 * n]),
            torch.zeros(L, b, H, lc, dh), torch.zeros(L, b, H, lc, dh), torch.tensor([0, lc]),
            torch.zeros(L, b, c.conv_kernel - 1, c.d_model))
    batch_axis = {"feats": 0, "t0": 0, "k": 1, "v": 1, "kv_len": 0, "conv": 1, "x": 0,
                  "k_out": 1, "v_out": 1, "kv_len_out": 0, "conv_out": 1}
    torch.onnx.export(enc, args, path, input_names=_IO_IN, output_names=_IO_OUT,
                      dynamic_axes={k: ({ax: "B", 1: "F"} if k == "feats" else
                                        {ax: "B", 1: "N"} if k == "x" else {ax: "B"})
                                    for k, ax in batch_axis.items()},
                      opset_version=17, dynamo=False)


def quantize_int8(fp32_path: str, int8_path: str) -> None:
    """Dynamic int8 quantisation (per-channel int8 weights, int8 activations computed per call)
    of the MatMul/Gemm weights; everything else stays fp32."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(fp32_path, int8_path, weight_type=QuantType.QInt8, per_channel=True,
                     op_types_to_quantize=["MatMul", "Gemm"])


class OrtChunkEncoder:
    """onnxruntime execution of an exported ChunkEncoder (same call signature)."""

    def __init__(self, path: str, threads: int):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = max(1, threads)
        so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])

    def __call__(self, *args):
        outs = self.sess.run(None, {k: a.numpy() for k, a in zip(_IO_IN, args, strict=True)})
        return tuple(torch.from_numpy(o) for o in outs)


class Stream:
    """Per-utterance state: unframed samples, features, encoder caches, decoder context."""

    def __init__(self, rec: StreamingRecognizer):
        c = rec.cfg.encoder
        L, H, dh = c.n_blocks, c.n_heads, c.d_model // c.n_heads
        self.samples = np.zeros(0, dtype=np.float32)  # not yet consumed by a full frame
        self.sample_base = 0  # absolute index of samples[0]
        self.n_feat = 0
        self.feats = torch.zeros(0, c.n_mels)  # normalised features from feat_base on
        self.feat_base = 0
        self.t_next = 0
        self.finished = False
        self.k = torch.zeros(L, H, c.left_context, dh)
        self.v = torch.zeros(L, H, c.left_context, dh)
        self.kv_len = 0
        self.conv = torch.zeros(L, c.conv_kernel - 1, c.d_model)
        self.ctx = [BLANK, BLANK]
        self.tokens: list[int] = []
        self.forced = 0
        self.states: list[torch.Tensor] = []
        self._rec = rec

    def accept_waveform(self, sample_rate: int, wav) -> None:
        if sample_rate != 16000:
            raise ShapeError(f"sample rate {sample_rate} != 16000")
        if self.finished:
            raise ShapeError("accept_waveform after input_finished")
        self.samples = np.concatenate([self.samples, np.asarray(wav, dtype=np.float32)])
        self._rec._featurize(self)

    def input_finished(self) -> None:
        self.finished = True


class StreamingRecognizer:
    def __init__(self, model: ASRModel, tokenizer, threads: int = 1, max_symbols: int = 8,
                 backend: str = "torch", onnx_path: str | None = None):
        if threads:
            torch.set_num_threads(threads)
        self.model = model.eval()
        self.cfg = model.cfg
        self.chunk = self.cfg.encoder.chunk
        self.tok = tokenizer
        self.max_symbols = max_symbols
        self.logmel = LogMel()
        self.encoder = ChunkEncoder(model.encoder).eval()
        if backend in ("onnx", "onnx-int8"):
            if onnx_path is None:
                raise ShapeError(f"backend {backend} needs onnx_path")
            export_onnx(self.encoder, onnx_path)
            if backend == "onnx-int8":
                q = onnx_path[:-5] + ".int8.onnx" if onnx_path.endswith(".onnx") else onnx_path + ".int8"
                quantize_int8(onnx_path, q)
                onnx_path = q
            self.encoder = OrtChunkEncoder(onnx_path, threads)
        elif backend != "torch":
            raise ShapeError(f"unknown backend {backend}")
        self.forced_advances = 0
        self.keep_states = False  # tests: keep every stream's encoder states

    @staticmethod
    def from_checkpoint(path: str, threads: int = 1, max_symbols: int = 8, backend: str = "torch",
                        onnx_path: str | None = None):
        from train import checkpoint
        from train.tokenizer import Tokenizer, tokenizer_path

        p = checkpoint.load(path, restore_rng=False)
        model = ASRModel(ModelConfig.from_dict(p["model_config"]))
        model.load_state_dict(p["model"])
        return StreamingRecognizer(model, Tokenizer(tokenizer_path(4096)), threads, max_symbols,
                                   backend, onnx_path)

    # ---- sherpa-onnx-like surface used by ultra.sp_a -----------------------------------------
    def create_stream(self) -> Stream:
        return Stream(self)

    def _chunk_bounds(self, s: Stream) -> tuple[int, int] | None:
        t0 = s.t_next
        if s.finished:
            total = int(stem_out_len(s.n_feat))
            return (t0, min(t0 + self.chunk, total)) if t0 < total else None
        t1 = t0 + self.chunk
        return (t0, t1) if 4 * t1 - 4 <= s.n_feat - 1 else None

    def is_ready(self, s: Stream) -> bool:
        return self._chunk_bounds(s) is not None

    def decode_stream(self, s: Stream) -> None:
        self.decode_streams([s])

    @torch.no_grad()
    def decode_streams(self, streams: list[Stream]) -> None:
        groups: dict[int, list[tuple[Stream, int]]] = {}
        for s in streams:
            b = self._chunk_bounds(s)
            if b is not None:
                groups.setdefault(b[1] - b[0], []).append((s, b[0]))
        for n, items in groups.items():
            self._run_group([s for s, _ in items], [t0 for _, t0 in items], n)

    def get_result(self, s: Stream) -> str:
        return self.tok.decode(s.tokens)

    # ---- internals ---------------------------------------------------------------------------
    def _featurize(self, s: Stream) -> None:
        avail = s.sample_base + len(s.samples)
        n_new = 0 if avail < WIN else 1 + (avail - WIN) // HOP - s.n_feat
        if n_new <= 0:
            return
        start = s.n_feat * HOP - s.sample_base
        seg = torch.from_numpy(s.samples[start:start + WIN + (n_new - 1) * HOP].copy())
        f = self.model.cmvn(self.logmel(seg))
        s.feats = torch.cat([s.feats, f], 0)
        s.n_feat += n_new
        drop = s.n_feat * HOP - s.sample_base  # samples before the next frame's start
        s.samples = s.samples[drop:]
        s.sample_base += drop

    def _slice(self, s: Stream, t0: int, n: int) -> torch.Tensor:
        lo, hi = 4 * t0 - 6, 4 * (t0 + n) - 4  # inclusive
        out = torch.zeros(hi - lo + 1, self.cfg.encoder.n_mels)
        a = max(lo, s.feat_base)
        out[a - lo:] = s.feats[a - s.feat_base:hi + 1 - s.feat_base]
        keep_from = max(0, 4 * (t0 + n) - 6)  # next chunk starts its slice here
        if keep_from > s.feat_base:
            s.feats = s.feats[keep_from - s.feat_base:]
            s.feat_base = keep_from
        return out

    def _run_group(self, streams: list[Stream], t0s: list[int], n: int) -> None:
        feats = torch.stack([self._slice(s, t0, n) for s, t0 in zip(streams, t0s, strict=True)])
        t0 = torch.tensor(t0s)
        k = torch.stack([s.k for s in streams], dim=1)
        v = torch.stack([s.v for s in streams], dim=1)
        kv = torch.tensor([s.kv_len for s in streams])
        conv = torch.stack([s.conv for s in streams], dim=1)
        x, k, v, kv, conv = self.encoder(feats, t0, k, v, kv, conv)
        for i, s in enumerate(streams):
            s.k, s.v, s.kv_len, s.conv = k[:, i], v[:, i], int(kv[i]), conv[:, i]
            s.t_next = t0s[i] + n
            if self.keep_states:
                s.states.append(x[i])
        self._greedy(streams, self.model.joiner.enc_proj(x))

    def _greedy(self, streams: list[Stream], enc_p: torch.Tensor) -> None:
        pred, joiner = self.model.predictor, self.model.joiner
        ctx = torch.tensor([s.ctx for s in streams])
        for t in range(enc_p.shape[1]):
            active = torch.ones(len(streams), dtype=torch.bool)
            for _ in range(self.max_symbols):
                idx = active.nonzero().squeeze(1)
                if idx.numel() == 0:
                    break
                kk = joiner.logits(enc_p[idx, t], pred(ctx[idx])).argmax(dim=-1)
                emit = kk != BLANK
                ei = idx[emit]
                for i, tok in zip(ei.tolist(), kk[emit].tolist(), strict=True):
                    streams[i].tokens.append(tok)
                ctx[ei] = torch.stack([ctx[ei, 1], kk[emit]], dim=1)
                active = torch.zeros_like(active)
                active[ei] = True
            else:
                for i in active.nonzero().squeeze(1).tolist():
                    streams[i].forced += 1
                    self.forced_advances += 1
        for i, s in enumerate(streams):
            s.ctx = ctx[i].tolist()


def run_streaming(rec: StreamingRecognizer, utts, streams: int, packet_ms: int = 20) -> list[str]:
    """Feed utterances in real-time-sized packets through `streams` concurrent slots."""
    packet = 16000 * packet_ms // 1000
    queue = list(range(len(utts)))
    active: dict[int, list] = {}
    hyps: dict[int, str] = {}
    while queue or active:
        for slot in range(streams):
            if slot not in active and queue:
                i = queue.pop(0)
                active[slot] = [i, rec.create_stream(), 0]
        for st in active.values():
            i, s, cur = st
            pcm = utts[i].pcm
            if cur < len(pcm):
                s.accept_waveform(16000, pcm[cur:cur + packet].astype(np.float32) / 32768.0)
                st[2] = cur + packet
                if st[2] >= len(pcm):
                    s.input_finished()
        while True:
            ready = [st[1] for st in active.values() if rec.is_ready(st[1])]
            if not ready:
                break
            rec.decode_streams(ready)
        for slot, (i, s, cur) in list(active.items()):
            if cur >= len(utts[i].pcm) and not rec.is_ready(s):
                hyps[i] = rec.get_result(s)
                del active[slot]
    return [hyps[i] for i in range(len(utts))]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="dev-clean")
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--streams", type=int, default=16)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--packet-ms", type=int, default=20)
    ap.add_argument("--compare", action="store_true", help="also run the trainer's batch eval")
    ap.add_argument("--backend", choices=["torch", "onnx", "onnx-int8"], default="torch")
    ap.add_argument("--onnx-path", default=None, help="where to write the exported chunk encoder")
    a = ap.parse_args()
    from ultra.libri import load_split
    from ultra.wer import corpus_wer

    rec = StreamingRecognizer.from_checkpoint(a.checkpoint, threads=a.threads, backend=a.backend,
                                              onnx_path=a.onnx_path)
    utts = load_split(a.split, a.limit)
    r0 = resource.getrusage(resource.RUSAGE_SELF)
    w0 = time.perf_counter()
    hyps = run_streaming(rec, utts, a.streams, a.packet_ms)
    r1 = resource.getrusage(resource.RUSAGE_SELF)
    cpu = (r1.ru_utime + r1.ru_stime) - (r0.ru_utime + r0.ru_stime)
    audio = sum(u.seconds for u in utts)
    out = {"kind": "streaming", "backend": a.backend, "split": a.split, "n": len(utts), "streams": a.streams,
           "threads": a.threads, "packet_ms": a.packet_ms, "audio_s": round(audio, 1),
           "wall_s": round(time.perf_counter() - w0, 2), "cpu_s": round(cpu, 2),
           "rtf_cpu": round(cpu / audio, 4), "forced_advances": rec.forced_advances,
           **corpus_wer([u.text for u in utts], hyps)}
    if a.compare:
        from train.data import Featurizer, collate
        from train.tokenizer import Tokenizer, tokenizer_path

        feat = Featurizer(rec.tok, Tokenizer(tokenizer_path(1024)))
        exs = [feat.example(u.uid, u.text, u.pcm) for u in utts]
        bh = []
        for i in range(0, len(exs), 8):
            b = collate(exs[i:i + 8])
            toks, _ = rec.model.greedy(b.feats, b.feat_lens)
            bh.extend(rec.tok.decode(t) for t in toks)
        out["batch_eval_wer"] = corpus_wer([u.text for u in utts], bh)["wer"]
        out["identical_hyps"] = sum(x == y for x, y in zip(hyps, bh, strict=True))
    print(json.dumps(out))


if __name__ == "__main__":
    main()
