"""Export a trained R1 checkpoint to the runtime contract `ultra.r1-export/v1`.

  python -m train.export --checkpoint <ckpt.pt> --out <bundle_dir> [--no-int8]

Bundle contents (every file's SHA-256 is recorded in manifest.json):
  encoder.fp32.onnx / encoder.int8.onnx   one chunk for a batch of streams, with explicit caches;
                                          outputs encoder states already projected to joiner width
  decoder.onnx                            stateless predictor: ctx [N, 2] int64 -> dec [N, J]
  joiner.fp32.onnx / joiner.int8.onnx     enc [N, J], dec [N, J] -> logits [N, V]
  tokens.txt                              "<piece> <output index>"; index 0 is <blk>
  bpe.model                               the SentencePiece model the pieces come from
  mel_fbank.f32                           (n_fft/2 + 1) x n_mels float32, little-endian, row-major
  manifest.json                           feature spec, CMVN, streaming metadata, greedy rule

int8 files use onnxruntime dynamic quantisation (per-channel int8 MatMul/Gemm weights). The decoder
is tiny and stays fp32. The reference consumer of a bundle is `train.bundle_infer`, which reads
only these files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess

import torch
from torch import nn

from train import checkpoint
from train.asr import ASRModel, ModelConfig
from train.features import F_MAX, F_MIN, HOP, LOG_FLOOR, N_FFT, N_MELS, SAMPLE_RATE, WIN, mel_filterbank
from train.infer import ChunkEncoder, ChunkEncoderMR, cache_shapes, quantize_int8
from train.tokenizer import Tokenizer, tokenizer_path
from train.transducer import BLANK, CLIP

SCHEMA = "ultra.r1-export/v1.1"
# v1.1: adds manifest["int8"]. int8_mode "static_qdq" (default) = per-channel int8 weights and
# per-tensor activation scales calibrated once on TRAIN audio, so a stream's numbers do not depend
# on which streams share its batch. "dynamic" (v1 behaviour) computes activation scales per call
# over the whole batch and is NOT batch-invariant.


class ExportError(Exception):
    """The checkpoint and the tokenizer or configuration do not form a consistent bundle."""


class EncoderExport(nn.Module):
    """ChunkEncoder(MR) followed by the joiner's encoder projection."""

    def __init__(self, model: ASRModel):
        super().__init__()
        mr = model.cfg.encoder.family == "multirate"
        self.chunk = (ChunkEncoderMR if mr else ChunkEncoder)(model.encoder)
        self.proj = model.joiner.enc_proj

    def forward(self, *args):
        out = self.chunk(*args)
        return (self.proj(out[0]),) + tuple(out[1:])


class DecoderExport(nn.Module):
    def __init__(self, model: ASRModel):
        super().__init__()
        self.pred = model.predictor

    def forward(self, ctx):
        return self.pred(ctx)


class JoinerExport(nn.Module):
    def __init__(self, model: ASRModel):
        super().__init__()
        self.j = model.joiner

    def forward(self, enc, dec):
        return self.j.logits(enc, dec)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                       cwd=os.path.dirname(__file__)).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _state_specs(c) -> list[dict]:
    """Names, shapes (with B for the stream batch) and meaning of every encoder state tensor."""
    sh = cache_shapes(c)
    n25 = sh["k"][0]
    specs = [
        {"name": "k", "shape": [n25, "B", *sh["k"][1:]], "dtype": "float32",
         "meaning": "attention keys (RoPE applied at absolute positions) of the 25 Hz blocks, "
                    "oldest first; valid entries are the last kv_len slots"},
        {"name": "v", "shape": [n25, "B", *sh["k"][1:]], "dtype": "float32",
         "meaning": "attention values, same layout as k"},
        {"name": "kv_len", "shape": ["B"], "dtype": "int64",
         "meaning": "valid cache slots (0 at stream start, saturates at left_context)"},
        {"name": "conv", "shape": [n25, "B", *sh["conv"][1:]], "dtype": "float32",
         "meaning": "last kernel-1 inputs of each block's causal depthwise convolution"},
    ]
    if c.family == "multirate":
        specs += [
            {"name": "km", "shape": [sh["km"][0], "B", *sh["km"][1:]], "dtype": "float32",
             "meaning": "12.5 Hz middle-block attention keys (left_context/2 slots)"},
            {"name": "vm", "shape": [sh["km"][0], "B", *sh["km"][1:]], "dtype": "float32",
             "meaning": "12.5 Hz middle-block attention values"},
            {"name": "kvm_len", "shape": ["B"], "dtype": "int64",
             "meaning": "valid middle cache slots"},
            {"name": "convm", "shape": [sh["convm"][0], "B", *sh["convm"][1:]],
             "dtype": "float32", "meaning": "middle-block depthwise-convolution history"},
        ]
    return specs


class _RowsReader:
    """onnxruntime CalibrationDataReader over captured input rows (one row per call)."""

    def __init__(self, rows: list[dict]):
        self.it = iter(rows)

    def get_next(self):
        return next(self.it, None)


def quantize_static_int8(fp32_path: str, int8_path: str, rows: list[dict],
                         method: str = "minmax", percentile: float = 99.999,
                         exclude: list[str] | None = None) -> None:
    """Static QDQ int8: per-channel int8 weights of MatMul/Gemm with constant B; per-tensor
    activation scales calibrated over `rows` (MinMax or Percentile). Activation-activation
    MatMuls stay fp32, as do nodes named in `exclude`."""
    from onnxruntime.quantization import (CalibrationMethod, QuantFormat, QuantType,
                                          quantize_static)

    calib = {"minmax": CalibrationMethod.MinMax, "percentile": CalibrationMethod.Percentile}
    extra = {"MatMulConstBOnly": True}
    if method == "percentile":
        extra["CalibPercentile"] = percentile
    quantize_static(fp32_path, int8_path, _RowsReader(rows), quant_format=QuantFormat.QDQ,
                    op_types_to_quantize=["MatMul", "Gemm"], per_channel=True,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                    calibrate_method=calib[method], nodes_to_exclude=exclude or [],
                    extra_options=extra)


def export_bundle(ckpt_path: str, out_dir: str, int8: bool = True, tokenizer=None,
                  int8_mode: str = "static_qdq", calib_utts=None, calib_manifest=None,
                  joiner_int8: bool = False, calib_method: str = "minmax",
                  calib_percentile: float = 99.999) -> dict:
    """Write the bundle for checkpoint `ckpt_path` into `out_dir`; returns the manifest."""
    p = checkpoint.load(ckpt_path, restore_rng=False)
    cfg = ModelConfig.from_dict(p["model_config"])
    model = ASRModel(cfg)
    model.load_state_dict(p["model"])
    model.eval()
    c = cfg.encoder
    os.makedirs(out_dir, exist_ok=True)
    tok = tokenizer or Tokenizer(tokenizer_path(cfg.vocab_rnnt - 1))
    if p.get("tokenizer_sha256") not in (None, tok.sha256):
        raise ExportError("checkpoint tokenizer hash differs from the tokenizer on disk")

    # Encoder.
    enc = EncoderExport(model).eval()
    sh = cache_shapes(c)
    b, n = 2, c.chunk

    def z(key):
        L, *rest = sh[key]
        return torch.zeros(L, b, *rest)

    args = [torch.zeros(b, 4 * n + 3, c.n_mels), torch.tensor([0, 4 * n]), z("k"), z("k"),
            torch.tensor([0, c.left_context]), z("conv")]
    names_in = ["feats", "t0", "k", "v", "kv_len", "conv"]
    names_out = ["enc", "k_out", "v_out", "kv_len_out", "conv_out"]
    if c.family == "multirate":
        args += [z("km"), z("km"), torch.tensor([0, c.left_context // 2]), z("convm")]
        names_in += ["km", "vm", "kvm_len", "convm"]
        names_out += ["km_out", "vm_out", "kvm_len_out", "convm_out"]
    batch_axis = {nm: (1 if nm.split("_out")[0] in ("k", "v", "conv", "km", "vm", "convm") else 0)
                  for nm in names_in + names_out}
    dyn = {nm: ({ax: "B", 1: "F"} if nm == "feats" else {ax: "B", 1: "N"} if nm == "enc"
                else {ax: "B"}) for nm, ax in batch_axis.items()}
    files = {}
    path = os.path.join(out_dir, "encoder.fp32.onnx")
    with torch.no_grad():
        torch.onnx.export(enc, tuple(args), path, input_names=names_in, output_names=names_out,
                          dynamic_axes=dyn, opset_version=17, dynamo=False)
    files["encoder.fp32"] = "encoder.fp32.onnx"
    # Decoder and joiner.
    J = cfg.joiner
    with torch.no_grad():
        torch.onnx.export(DecoderExport(model).eval(), (torch.zeros(3, 2, dtype=torch.long),),
                          os.path.join(out_dir, "decoder.onnx"), input_names=["ctx"],
                          output_names=["dec"], dynamic_axes={"ctx": {0: "N"}, "dec": {0: "N"}},
                          opset_version=17, dynamo=False)
        torch.onnx.export(JoinerExport(model).eval(), (torch.zeros(3, J), torch.zeros(3, J)),
                          os.path.join(out_dir, "joiner.fp32.onnx"), input_names=["enc", "dec"],
                          output_names=["logits"],
                          dynamic_axes={"enc": {0: "N"}, "dec": {0: "N"}, "logits": {0: "N"}},
                          opset_version=17, dynamo=False)
    files["decoder"] = "decoder.onnx"
    files["joiner.fp32"] = "joiner.fp32.onnx"
    int8_info = None
    if int8 and int8_mode in ("dynamic", "dynamic_per_row"):
        for part in ("encoder", "joiner"):
            quantize_int8(os.path.join(out_dir, f"{part}.fp32.onnx"),
                          os.path.join(out_dir, f"{part}.int8.onnx"))
            files[f"{part}.int8"] = f"{part}.int8.onnx"
        if int8_mode == "dynamic":
            int8_info = {"mode": "dynamic", "batch_invariant": False}
        else:
            # Same graphs; the contract restricts int8 calls to one stream (encoder) / one query
            # row (joiner), so each activation scale sees only that row: invariant by construction.
            int8_info = {"mode": "dynamic_per_row", "batch_invariant": True, "max_batch": 1,
                         "rule": "run encoder.int8 with B = 1 (one stream per call) and "
                                 "joiner.int8 with N = 1; fp32 graphs may batch freely"}
    elif int8 and int8_mode == "static_qdq":
        from train.calib import calibration_set, capture

        if calib_utts is None:
            calib_utts, calib_manifest = calibration_set(tokenizer=tok)
        enc_rows, joi_rows, prov = capture(model, tok, calib_utts)
        quantize_static_int8(os.path.join(out_dir, "encoder.fp32.onnx"),
                             os.path.join(out_dir, "encoder.int8.onnx"), enc_rows,
                             calib_method, calib_percentile)
        files["encoder.int8"] = "encoder.int8.onnx"
        if joiner_int8:
            quantize_static_int8(os.path.join(out_dir, "joiner.fp32.onnx"),
                                 os.path.join(out_dir, "joiner.int8.onnx"), joi_rows,
                                 calib_method, calib_percentile)
            files["joiner.int8"] = "joiner.int8.onnx"
        calib_manifest = dict(calib_manifest, consumed_rows=prov)
        with open(os.path.join(out_dir, "calibration_manifest.json"), "w") as fh:
            json.dump(calib_manifest, fh, indent=1)
        files["calibration_manifest"] = "calibration_manifest.json"
        int8_info = {"mode": "static_qdq", "batch_invariant": True,
                     "joiner": "int8" if joiner_int8 else "fp32 (joiner.fp32.onnx; no int8 file)",
                     "calibration_method": calib_method if calib_method == "minmax"
                     else f"percentile {calib_percentile}",
                     "weights": "int8 symmetric per output channel (MatMul/Gemm with constant B)",
                     "activations": "uint8 per-tensor, MinMax-calibrated constant scale/zero-point",
                     "not_quantized": "attention score/value MatMuls, convolutions, norms, softmax",
                     "calibration": {"selection_sha256": calib_manifest["selection_sha256"],
                                     "n_utterances": calib_manifest["n"],
                                     "strata": calib_manifest["strata"],
                                     "encoder_rows": len(enc_rows),
                                     "encoder_rows_offered": prov["encoder"]["offered"],
                                     "joiner_rows": len(joi_rows) if joiner_int8 else 0,
                                     "file": "calibration_manifest.json"}}
    elif int8:
        raise ExportError(f"unknown int8_mode {int8_mode}")
    # Tokens, BPE model, mel filterbank.
    with open(os.path.join(out_dir, "tokens.txt"), "w") as fh:
        fh.write(f"<blk> {BLANK}\n")
        for i in range(tok.vocab):
            fh.write(f"{tok.sp.id_to_piece(i)} {i + 1}\n")
    files["tokens"] = "tokens.txt"
    shutil.copyfile(tok.path, os.path.join(out_dir, "bpe.model"))
    files["bpe"] = "bpe.model"
    fb = mel_filterbank().numpy().astype("<f4")
    fb.tofile(os.path.join(out_dir, "mel_fbank.f32"))
    files["mel_fbank"] = "mel_fbank.f32"

    manifest = {
        "schema": SCHEMA,
        "int8": int8_info,
        "model": {"family": c.family, "params": model.num_params(),
                  "checkpoint": os.path.abspath(ckpt_path), "checkpoint_step": p["step"],
                  "checkpoint_sha256": _sha256(ckpt_path), "config": cfg.to_dict(),
                  "exporter_commit": _commit(), "torch": torch.__version__},
        "files": {k: {"path": v, "sha256": _sha256(os.path.join(out_dir, v))}
                  for k, v in files.items()},
        "features": {
            "sample_rate": SAMPLE_RATE, "input": "mono float32 in [-1, 1] (int16 / 32768)",
            "frame_length": WIN, "frame_shift": HOP, "n_fft": N_FFT, "n_mels": N_MELS,
            "framing": "frame i covers samples [160 i, 160 i + 400); no padding; a frame exists "
                       "only once all 400 samples have arrived",
            "per_frame": ["subtract the frame mean (DC removal)",
                          "multiply by a symmetric Hann window of 400 (numpy.hanning(400), "
                          "torch.hann_window(400, periodic=False))",
                          "zero-pad to n_fft and take |rfft|^2 (257 bins)",
                          "mel = power @ mel_fbank (257 x 80)",
                          "logmel = ln(mel + log_floor)",
                          "normalised = (logmel - cmvn.mean) * cmvn.istd"],
            "log_floor": LOG_FLOOR, "mel": {"scale": "htk (1127 ln(1 + f/700))",
                                            "f_min": F_MIN, "f_max": F_MAX,
                                            "file": "mel_fbank.f32"},
            "cmvn": {"mean": model.cmvn.mean.tolist(), "istd": model.cmvn.istd.tolist()},
        },
        "streaming": {
            "encoder_rate_hz": 25, "subsampling": 4, "chunk_frames": c.chunk,
            "chunk_ms": c.chunk * 40, "left_context_frames": c.left_context,
            "encoder_input_rule": (
                "for a chunk of encoder frames [t0, t0+n) feed normalised feature frames "
                "4 t0 - 6 .. 4 (t0 + n) - 4 inclusive (4 n + 3 rows), zero rows for indices < 0. "
                "A full chunk (n = chunk_frames) is ready when feature frame 4 (t0 + n) - 4 "
                "exists. After input_finished the total frame count is "
                "T = ceil(ceil(F / 2) / 2) for F feature frames; the last chunk may be "
                "partial (n = T - t0 >= 1). t0 advances by n."),
            "states": _state_specs(c),
            "state_init": "all zeros; kv_len (and kvm_len) = 0",
            "state_update": "replace every state input by the matching *_out output",
            "batching": "streams may be batched on axis B when they share n; each stream "
                        "keeps its own t0 and caches",
            "encoder_output": {"name": "enc", "shape": ["B", "n", cfg.joiner],
                               "meaning": "encoder states projected to joiner width"},
        },
        "decoder": {
            "kind": "stateless", "context": 2, "blank_id": BLANK, "vocab_size": cfg.vocab_rnnt,
            "ctx_init": [BLANK, BLANK], "ctx_order": "(older, newer)",
            "joiner_activation": cfg.joiner_act + (f" (clip {CLIP})"
                                                   if cfg.joiner_act == "clipped_relu" else ""),
            "greedy": ("for each encoder frame: up to max_symbols times: logits = joiner(enc_t, "
                       "decoder(ctx)); k = argmax; if k == blank: next frame; else emit k, "
                       "ctx = (ctx[1], k). If max_symbols emissions happen at one frame, "
                       "advance to the next frame and count a forced advance."),
            "max_symbols": 8,
            "detokenize": "pieces of the emitted ids (tokens.txt), concatenated, U+2581 -> space, "
                          "strip; lower-case text without punctuation",
        },
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-int8", action="store_true")
    ap.add_argument("--int8-mode", choices=["static_qdq", "dynamic", "dynamic_per_row"], default="static_qdq")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--joiner-int8", action="store_true", help="also quantise the joiner (static)")
    ap.add_argument("--calib-method", choices=["minmax", "percentile"], default="minmax")
    ap.add_argument("--calib-percentile", type=float, default=99.999)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    m = export_bundle(a.checkpoint, a.out, int8=not a.no_int8, int8_mode=a.int8_mode,
                      joiner_int8=a.joiner_int8, calib_method=a.calib_method,
                      calib_percentile=a.calib_percentile)
    print(json.dumps({"out": a.out, "files": m["files"]}, indent=1))


if __name__ == "__main__":
    main()
