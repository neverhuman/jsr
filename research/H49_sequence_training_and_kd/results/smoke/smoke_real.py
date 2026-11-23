"""H49 smoke test on the REAL model (B0 = A1-49000 EMA weights, S preset, R0 bpe) and real TRAIN
utterances, CPU: two trainer steps with MWER on, two with seq KD on (synthetic union from B0's own beam
+ the reference), timing of the forward with and without each term. Analysis only, no weights written."""
import io, json, os, sys, time
tree = sys.argv[1]
sys.path.insert(0, tree)
os.environ.setdefault("JSR_DATA", "/dev/shm/eval-lane/jsr_data")
import dataclasses
import pyarrow.parquet as pq
import soundfile as sf
import torch
torch.set_num_threads(8)
from train import checkpoint
from train.asr import ModelConfig
from train.data import Example, FixedBatches, Featurizer, collate
from train.tokenizer import Tokenizer
from train.trainer import Trainer, TrainConfig
from train.seqkd import SeqKDStore
from train.mwer import batched_beam_nbest

p = checkpoint.load("/dev/shm/h49-seqkd/ckpt/A1-49000-ema-as-model.pt", restore_rng=False)
mcfg = ModelConfig.from_dict(p["model_config"])
tok = Tokenizer("/dev/shm/eval-lane/jsr_data/tokenizers/bpe.model")
feat = Featurizer(tok, tok)
f = sorted(os.listdir("/mnt/xbabe3-external/jsr/corpora/librispeech/hf_parquet/all/train.other.500"))[3]
rows = pq.ParquetFile(os.path.join("/mnt/xbabe3-external/jsr/corpora/librispeech/hf_parquet/all/train.other.500", f)).read_row_group(5, columns=["id", "text", "audio"]).to_pylist()
rows = sorted(rows, key=lambda r: len(r["audio"]["bytes"]))[:6]
exs = [feat.example(r["id"], r["text"], sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="int16")[0]) for r in rows]
print(json.dumps({"utts": len(exs), "seconds": [round(e.seconds, 2) for e in exs]}))
res = {}
for name, kw in (("base", {}), ("mwer", {"mwer_weight": 1.0, "mwer_beam": 4, "mwer_utts": 6}),):
    cfg = TrainConfig(preset="S", total_steps=10, warmup=1, log_every=1, eval_every=0, ckpt_every_steps=1000,
                      specaug=True, specaug_start=0, chunk_modes="4,8,16", rnnt_pieces=500, ctc_pieces=500,
                      rnnt_loss="pruned", prune_range=5, lr=1e-6, **kw)
    torch.manual_seed(1)
    tr = Trainer(f"/dev/shm/h49-seqkd/work/smoke-{name}", cfg, FixedBatches(exs, batch_size=6), model_cfg=mcfg,
                 tokenizer=tok, device="cpu", log=False)
    tr.model.load_state_dict(p["model"])
    it = iter(tr._batches())
    times, outs = [], []
    for _ in range(2):
        b = next(it)[1]
        t0 = time.time(); r = tr.train_step(b); times.append(round(time.time() - t0, 2)); outs.append(r)
    res[name] = {"step_s": times, "last": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in outs[-1].items()
                                          if k in ("loss", "rnnt", "ctc", "mwer", "mwer_utts", "mwer_active", "mwer_cands",
                                                   "mwer_ref_in_beam", "mwer_top_err", "mwer_exp_err", "mwer_words",
                                                   "gnorm_human", "gnorm_mwer")}}
    print(name, json.dumps(res[name]), flush=True)
# seq KD with a store built from B0's own beam-4 (stand-in teacher scores = B0 beam scores; plumbing test only)
model = tr.model.eval()
with torch.no_grad():
    bb = collate(exs)
    enc, lens = model.encode(model.cmvn(bb.feats) * 0 + bb.feats, bb.feat_lens, 4) if False else model.encode(bb.feats, bb.feat_lens, 4)
    nb = batched_beam_nbest(model.predictor, model.joiner, model.joiner.enc_proj(enc.float()), lens, 4)
store_p = "/dev/shm/h49-seqkd/work/smoke_store.jsonl"
with open(store_p, "w") as fh:
    fh.write(json.dumps({"header": True, "format": "ultra.seqkd/v1", "student_tokenizer_sha256": tok.sha256,
                         "teacher": "PLUMBING TEST: B0 beam scores", "tau": 1.0, "max_alt": 3}) + "\n")
    for e, hyps in zip(exs, nb):
        cands = [{"ids": e.rnnt, "teacher": 0.0, "src": ["ref"]}] + [{"ids": s, "teacher": sc, "src": ["student"]} for s, sc in hyps if s != e.rnnt][:3]
        fh.write(json.dumps({"uid": e.uid, "speed": 1.0, "cands": cands}) + "\n")
store = SeqKDStore(store_p, tokenizer_sha256=tok.sha256)
for e in exs:
    e.seq = store.get_checked(e.uid, 1.0, e.rnnt)
cfg = TrainConfig(preset="S", total_steps=10, warmup=1, log_every=1, eval_every=0, ckpt_every_steps=1000, specaug=True,
                  specaug_start=0, chunk_modes="4,8,16", rnnt_pieces=500, ctc_pieces=500, rnnt_loss="pruned",
                  prune_range=5, lr=1e-6, seq_weight=0.3, seq_store=store_p)
torch.manual_seed(1)
tr = Trainer("/dev/shm/h49-seqkd/work/smoke-seq", cfg, FixedBatches(exs, batch_size=6), model_cfg=mcfg, tokenizer=tok,
             device="cpu", log=False)
tr.model.load_state_dict(p["model"])
t0 = time.time(); r = tr.train_step(collate(exs)); dt = round(time.time() - t0, 2)
print("seq", json.dumps({"step_s": dt, "seqkd": round(r["seqkd"], 4), "seqkd_tokens": r["seqkd_tokens"], "seq_utts": r["seq_utts"],
                         "seq_cands": r["seq_cands"], "seq_q_ref": round(r["seq_q_ref"], 3), "counts": store.counts,
                         "gnorm_human": round(r.get("gnorm_human", -1), 3), "gnorm_seqkd": round(r.get("gnorm_seqkd", -1), 3)}))
