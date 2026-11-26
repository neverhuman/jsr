"""H75/H68: RC2 EMA commit-style n-best on DEV S-half: chunk 16, 320 ms zero tail pad, beam B,
M5 (tok5.p00149) + LODR (lodr2.train960) at the shipped final weights 0.80 / 0.60. Also writes the no-LM n-best.
Usage: rc2_nbest.py CKPT SPLIT_JSON LM LODR OUT_DIR [beam] [workers]"""
import dataclasses, json, sys, time
from pathlib import Path
import numpy as np, torch
from train.wer_diag import load_ckpt_model, load_selection_dev, encode_enc_p_cache, eval_nbest_oracle
from train.data import Featurizer, collate


@torch.no_grad()
def encode_gpu(model, tok, utts, chunk=16, batch_size=16, dev="cuda"):
    """Same as wer_diag.encode_enc_p_cache, encoder on GPU (FP32)."""
    feat = Featurizer(tok, tok)
    m = model.to(dev)
    order = sorted(range(len(utts)), key=lambda i: feat.logmel.num_frames(len(utts[i].pcm)))
    items = []
    for off in range(0, len(order), batch_size):
        b = collate([feat.example(utts[i].uid, utts[i].text, utts[i].pcm) for i in order[off:off + batch_size]])
        enc, lens = m.encode(b.feats.to(dev), b.feat_lens.to(dev), chunk)
        enc_p = m.joiner.enc_proj(enc.float()).cpu().numpy().astype(np.float32)
        items.extend((uid, b.texts[j], enc_p[j, :int(lens[j])].copy()) for j, uid in enumerate(b.uids))
    model.to("cpu")
    return items

ckpt, split_json, lm, lodr, out = sys.argv[1:6]
beam = int(sys.argv[6]) if len(sys.argv) > 6 else 16
workers = int(sys.argv[7]) if len(sys.argv) > 7 else 64
torch.set_num_threads(32)
out = Path(out); out.mkdir(parents=True, exist_ok=True)
t0 = time.time()
model, tok, meta = load_ckpt_model(ckpt, weights="ema")
utts, subsets = load_selection_dev(split_json)
pad = np.zeros(5120, dtype=np.int16)  # 320 ms
utts = [dataclasses.replace(u, pcm=np.concatenate([u.pcm, pad])) for u in utts]
enc = encode_gpu(model, tok, utts) if torch.cuda.is_available() else encode_enc_p_cache(model, tok, utts, chunk=16, batch_size=16)
print("encoded", len(enc), round(time.time() - t0), flush=True)
res = {"meta": meta, "beam": beam, "tail_pad_ms": 320}
res["m5_lodr"] = eval_nbest_oracle(model, tok, utts, chunk=16, beam=beam, lm_path=lm, lodr_path=lodr, lm_weight=0.80, lodr_weight=0.60,
                                   workers=workers, enc_cache=enc, subsets=subsets, out_jsonl=out / f"s_half_c16_b{beam}_m5lodr.nbest.jsonl")
print("m5 done", round(time.time() - t0), flush=True)
res["nolm"] = eval_nbest_oracle(model, tok, utts, chunk=16, beam=beam, workers=workers, enc_cache=enc, subsets=subsets,
                                out_jsonl=out / f"s_half_c16_b{beam}_nolm.nbest.jsonl")
res["elapsed_s"] = round(time.time() - t0)
(out / "oracle_summary.json").write_text(json.dumps(res, indent=1))
print(json.dumps({k: (v["pooled"] if isinstance(v, dict) and "pooled" in v else v) for k, v in res.items() if k != "meta"}, indent=1))
