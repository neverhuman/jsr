"""Static-INT8 quantisation sweep for an exported bundle (TRAIN calibration, DEV scoring).

  python -m train.quant_sweep --bundle <dir> --checkpoint <ckpt.pt> --out <sweep_dir> \
      --variants minmax pct99.99 minmax-keep2 --limit 400

Calibration rows are captured once (train.calib, TRAIN partition only) and reused for every
variant. Each variant writes encoder/joiner int8 files into <sweep_dir>/<variant>/, then is
checked for batch invariance and scored on DEV against the bundle's fp32 path. Results go to
<sweep_dir>/sweep.jsonl together with per-utterance hypotheses.

Variant grammar: <method>[-keep<K>][-joinerfp32]
  method      minmax | pct<P>   (percentile calibration at P, e.g. pct99.99)
  keep<K>     leave the first K and last K encoder blocks, the stem and projection in fp32
  joinerfp32  keep the joiner fp32 (encoder-only int8)
"""
from __future__ import annotations

import argparse
import json
import os
import re

import torch

from train import checkpoint
from train.asr import ASRModel, ModelConfig
from train.bundle_infer import BundleRecognizer, batch_invariance
from train.calib import calibration_set, capture
from train.export import quantize_static_int8
from train.tokenizer import Tokenizer, tokenizer_path


def _encoder_matmuls(path: str) -> list[str]:
    import onnx

    m = onnx.load(path, load_external_data=False)
    inits = {i.name for i in m.graph.initializer}
    return [n.name for n in m.graph.node
            if n.op_type in ("MatMul", "Gemm") and any(i in inits for i in n.input)]


def parse(variant: str) -> dict:
    mt = re.fullmatch(r"(minmax|pct[0-9.]+)(?:-keep(\d+))?(-joinerfp32)?", variant)
    if not mt:
        raise ValueError(f"bad variant {variant}")
    meth = mt.group(1)
    return {"method": "minmax" if meth == "minmax" else "percentile",
            "percentile": float(meth[3:]) if meth.startswith("pct") else None,
            "keep": int(mt.group(2) or 0), "joiner_fp32": bool(mt.group(3))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", nargs="+", required=True)
    ap.add_argument("--split", default="dev-clean")
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--enc-rows", type=int, default=384)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    from train.infer import run_streaming
    from ultra.libri import load_split
    from ultra.wer import corpus_wer

    os.makedirs(a.out, exist_ok=True)
    p = checkpoint.load(a.checkpoint, restore_rng=False)
    model = ASRModel(ModelConfig.from_dict(p["model_config"]))
    model.load_state_dict(p["model"])
    model.eval()
    tok = Tokenizer(tokenizer_path(model.cfg.vocab_rnnt - 1))
    utts, cman = calibration_set(tokenizer=tok)
    enc_rows, joi_rows = capture(model, tok, utts, max_enc_rows=a.enc_rows)
    with open(os.path.join(a.out, "calibration_manifest.json"), "w") as fh:
        json.dump(cman, fh, indent=1)
    enc_fp32 = os.path.join(a.bundle, "encoder.fp32.onnx")
    joi_fp32 = os.path.join(a.bundle, "joiner.fp32.onnx")
    mms = _encoder_matmuls(enc_fp32)
    nb = model.cfg.encoder.n_blocks
    per = (len(mms) - 2) // nb  # stem out + 8 per block + proj
    dev = load_split(a.split, a.limit)
    refs = [u.text for u in dev]
    fp32 = run_streaming(BundleRecognizer(a.bundle, "fp32", a.threads), dev, 16)
    base = corpus_wer(refs, fp32)["wer"]
    for v in a.variants:
        spec = parse(v)
        vd = os.path.join(a.out, v)
        os.makedirs(vd, exist_ok=True)
        exclude = []
        if spec["keep"]:
            k = spec["keep"]
            exclude = [mms[0], mms[-1]] + mms[1:1 + per * k] + mms[1 + per * (nb - k):1 + per * nb]
        enc_q = os.path.join(vd, "encoder.int8.onnx")
        quantize_static_int8(enc_fp32, enc_q, enc_rows, spec["method"],
                             spec["percentile"] or 99.999, exclude)
        override = {"encoder.int8": enc_q}
        jp = "fp32"
        if not spec["joiner_fp32"]:
            joi_q = os.path.join(vd, "joiner.int8.onnx")
            quantize_static_int8(joi_fp32, joi_q, [{"enc": e[None], "dec": d[None]}
                                                    for e, d in joi_rows],
                                 spec["method"], spec["percentile"] or 99.999)
            override["joiner.int8"] = joi_q
            jp = "int8"
        rec = BundleRecognizer(a.bundle, "int8", a.threads, joiner_precision=jp, override=override)
        hyps = run_streaming(rec, dev, 16)
        w = corpus_wer(refs, hyps)
        inv = batch_invariance(a.bundle, "int8", override=override, joiner_precision=jp)
        rec_out = {"variant": v, **spec, "excluded_nodes": len(exclude), "fp32_wer": base,
                   "int8_wer": w["wer"], "delta": round(w["wer"] - base, 3),
                   "identical_to_fp32": sum(x == y for x, y in zip(hyps, fp32, strict=True)),
                   "n": len(dev), "batch_invariance": inv["max_abs_diff"],
                   "calibration_sha256": cman["sha256"]}
        with open(os.path.join(vd, "dev_hyps.jsonl"), "w") as fh:
            for u, h, f in zip(dev, hyps, fp32, strict=True):
                fh.write(json.dumps({"uid": u.uid, "ref": u.text, "fp32": f, "int8": h}) + "\n")
        with open(os.path.join(a.out, "sweep.jsonl"), "a") as fh:
            fh.write(json.dumps(rec_out) + "\n")
        print(json.dumps(rec_out), flush=True)


if __name__ == "__main__":
    main()
