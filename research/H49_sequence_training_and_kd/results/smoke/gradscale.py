"""H49a gradient scale at B0 (owner plan 5.3: record gradient scale): ||grad L_human|| vs ||grad L_mwer||
(lambda = 1) on real TRAIN utterances, CPU, two different 8-utterance batches. Analysis only."""
import io, json, os, sys
sys.path.insert(0, sys.argv[1])
os.environ.setdefault("JSR_DATA", "/dev/shm/eval-lane/jsr_data")
import pyarrow.parquet as pq, soundfile as sf, torch
torch.set_num_threads(8)
from train import checkpoint
from train.asr import ModelConfig
from train.data import FixedBatches, Featurizer
from train.tokenizer import Tokenizer
from train.trainer import Trainer, TrainConfig
p = checkpoint.load(sys.argv[2], restore_rng=False)
import dataclasses
mcfg = ModelConfig.from_dict(p["model_config"])
if os.environ.get("H49_RNNT_LOSS") == "exact":
    mcfg = dataclasses.replace(mcfg, rnnt_loss="exact")
tok = Tokenizer("/dev/shm/eval-lane/jsr_data/tokenizers/bpe.model"); feat = Featurizer(tok, tok)
root = "/mnt/xbabe3-external/jsr/corpora/librispeech/hf_parquet/all"
out = []
for split, shard, rg in (("train.other.500", 7, 2), ("train.clean.360", 11, 4)):
    f = sorted(os.listdir(os.path.join(root, split)))[shard]
    rows = pq.ParquetFile(os.path.join(root, split, f)).read_row_group(rg, columns=["id", "text", "audio"]).to_pylist()
    rows = sorted(rows, key=lambda r: len(r["audio"]["bytes"]))[20:28]
    exs = [feat.example(r["id"], r["text"], sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="int16")[0]) for r in rows]
    cfg = TrainConfig(preset="S", total_steps=10, warmup=1, log_every=1, eval_every=0, ckpt_every_steps=1000, specaug=True,
                      specaug_start=0, chunk_modes="4,8,16", rnnt_pieces=500, ctc_pieces=500, rnnt_loss=os.environ.get("H49_RNNT_LOSS", "pruned"),
                      prune_range=5, lr=1e-9, mwer_weight=1.0, mwer_beam=4, mwer_utts=8)
    torch.manual_seed(1)
    tr = Trainer("/dev/shm/h49-seqkd/work/gs", cfg, FixedBatches(exs, batch_size=8), model_cfg=mcfg, tokenizer=tok,
                 device="cpu", log=False)
    tr.model.load_state_dict({k: v for k, v in p["model"].items() if mcfg.rnnt_loss == "pruned" or not k.startswith(("simple_am", "simple_lm"))})
    r = tr.train_step(next(iter(tr._batches()))[1])
    row = {"rnnt_loss": cfg.rnnt_loss, "batch": f"{split}/{f}/rg{rg}", "seconds": round(sum(e.seconds for e in exs), 1),
           **{k: round(v, 4) for k, v in r.items() if k.startswith(("gnorm_", "mwer")) or k in ("loss", "rnnt")}}
    row["ratio_mwer_to_human_at_lambda1"] = round(r["gnorm_mwer"] / r["gnorm_human"], 4)
    out.append(row); print(json.dumps(row), flush=True)
