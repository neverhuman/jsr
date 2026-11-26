"""H68: export a train_lm.py checkpoint to the engine's ULTRTF01 `.utfr` (INT8) and check parity.

Parity: NumPy INT8 reference (`QuantizedTfRescorer.score_sequence`, same math as the Rust rescorer)
against the FP32 torch model on N random n-best hypotheses; reports max/mean |delta logp| and the
n-best rank agreement of the LM score alone. Usage: export_utfr.py CKPT OUT.utfr [NBEST.jsonl] [N]
"""
from __future__ import annotations

import hashlib
import json
import random
import sys

import numpy as np
import torch

sys.path[:0] = [__import__("os").path.dirname(__file__), "/home/ubuntu/jsr/.agent/worktrees/wer-claude/python/ultra"]
from tools.export_tf_rescore import TfRescoreConfig, export_tf_tensors, load_tf_v1  # noqa: E402
from train_lm import load_lm  # noqa: E402
from score_nbest import lm_scores  # noqa: E402


def main() -> None:
    ckpt, out = sys.argv[1], sys.argv[2]
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    c = ck["cfg"]
    sd = {k.replace(".n1.w", ".n1.weight").replace(".n2.w", ".n2.weight").replace("norm.w", "norm.weight") if k.endswith(".w") else k: v
          for k, v in ck["model"].items()}
    cfg = TfRescoreConfig(vocab=c["vocab"], d_model=c["d"], n_layers=c["layers"], n_heads=c["heads"], d_ff=c["ff"],
                          ctx=c["ctx"], direction="r2l" if ck.get("reverse") else "l2r", tied=True,
                          rope_theta=float(c["theta"]), rms_eps=1e-5)
    raw = export_tf_tensors(cfg, sd, out)
    info = {"out": out, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw), "step": ck["step"], "direction": cfg.direction}
    if len(sys.argv) > 3:
        n = int(sys.argv[4]) if len(sys.argv) > 4 else 300
        rows = [json.loads(x) for x in open(sys.argv[3])]
        random.Random(7).shuffle(rows)
        q = load_tf_v1(out)
        m = load_lm(ckpt, "cpu").float()
        groups = [[list(map(int, h[0])) for h in r["nbest"][:8]] for r in rows[: max(1, n // 8)]]
        seqs = [s for g in groups for s in g]
        ref = lm_scores(m, seqs, reverse=m.reverse)
        i8 = np.array([q.score_sequence(s)["total_logp"] for s in seqs])
        d = np.abs(ref - i8)
        agree, pos = 0, 0
        for g in groups:
            k = len(g); agree += int(np.argmax(ref[pos:pos + k]) == np.argmax(i8[pos:pos + k])); pos += k
        info.update({"hyps": len(seqs), "max_abs_dlogp": float(d.max()), "mean_abs_dlogp": float(d.mean()),
                     "mean_abs_dlogp_per_tok": float((d / np.array([len(s) + 1 for s in seqs])).mean()),
                     "argmax_agree": f"{agree}/{len(groups)}"})
    print(json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
