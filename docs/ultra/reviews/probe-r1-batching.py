"""Synthetic export diagnostic; no audio, labels, WER or timing claims.

From the active root: .venv/bin/python docs/ultra/reviews/probe-r1-batching.py BUNDLE
Compare the identical first feature row alone and beside seven independent rows.
This diagnoses batch sensitivity of the export, not native/reference parity.
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).absolute().parents[3] / "python/ultra"))
from train.bundle_infer import BundleRecognizer  # noqa: E402


def main():
    bundle = Path(sys.argv[1])
    manifest_bytes = (bundle / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest["model"]["family"] != "dense" or manifest["streaming"]["chunk_frames"] != 4:
        raise ValueError("this diagnostic fixture requires the dense four-frame export")
    features = np.random.default_rng(20261120).normal(0, 1, (8, 19, 80)).astype(np.float32)
    rows = []
    for precision in ("fp32", "int8"):
        rec = BundleRecognizer(str(bundle), precision=precision, threads=1)
        if rec.enc.get_providers() != ["CPUExecutionProvider"]:
            raise ValueError("CPU encoder required")
        feed = {"feats": features, "t0": np.zeros(8, dtype=np.int64)}
        for state in rec.states:
            shape = [8 if d == "B" else d for d in state["shape"]]
            dtype = np.float32 if state["dtype"] == "float32" else np.int64
            feed[state["name"]] = np.zeros(shape, dtype=dtype)
        single = {"feats": features[:1], "t0": feed["t0"][:1]}
        for state in rec.states:
            axis = state["shape"].index("B")
            single[state["name"]] = np.take(feed[state["name"]], [0], axis=axis)
        solo = rec.enc.run(["enc"], single)[0][0]
        batched = rec.enc.run(["enc"], feed)[0][0]
        if not np.isfinite(solo).all() or not np.isfinite(batched).all():
            raise ValueError("nonfinite encoder output")
        rows.append(dict(precision=precision, same_input_row=0, batch_sizes=[1, 8],
                         max_abs_encoder_difference=float(np.max(np.abs(solo - batched))),
                         mean_abs_encoder_difference=float(np.mean(np.abs(solo - batched)))))
    print(json.dumps(dict(schema="ultra.export-batch-diagnostic/v1",
                          manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
                          probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                          numpy=np.__version__, ort=ort.__version__, seed=20261120,
                          threads=1, state="zero; all absolute positions zero", results=rows), indent=2))


if __name__ == "__main__":
    main()
