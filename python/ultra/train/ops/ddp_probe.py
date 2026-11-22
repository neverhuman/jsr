"""Exercise one shared ASR model under torch.distributed on recorded training audio.

Every run starts from a verified checkpoint with a fresh optimizer. It consumes
the same global batches at any world size, decoding only the rank's own audio.
The probe reports end-to-end throughput and verifies identical final model state
on all ranks. It never writes a production checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import socket
import time
from datetime import timedelta
from pathlib import Path

import pyarrow.parquet as pq
import soundfile as sf
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from train import checkpoint
from train.asr import ASRModel, ModelConfig
from train.data import Featurizer, collate
from train.errors import CheckpointError, DataError
from train.tokenizer import TRAIN_SPLITS, Tokenizer, tokenizer_path
from ultra.libri import PARQUET, SPLIT_DIRS, decode_flac


class ForwardLoss(ASRModel):
    def forward(self, batch):
        return self.losses(batch.feats, batch.feat_lens, batch.rnnt, batch.rnnt_lens,
                           batch.ctc, batch.ctc_lens)


def _batches(split, size, max_seconds, rank, world, feat):
    """Select one deterministic utterance order and decode a disjoint slice per rank."""
    root = Path(PARQUET) / SPLIT_DIRS[split]
    rows = []
    for path in sorted(root.glob("*.parquet")):
        for group in pq.ParquetFile(path).iter_batches(
            batch_size=128, columns=["id", "text", "audio"], use_threads=False,
        ):
            for row in group.to_pylist():
                audio = row["audio"]["bytes"]
                if sf.info(io.BytesIO(audio)).frames > max_seconds * 16000:
                    continue
                rows.append((row["id"], row["text"], audio))
                if len(rows) == size:
                    examples = [feat.example(uid, text, decode_flac(pcm))
                                for uid, text, pcm in rows[rank::world]]
                    yield collate(examples)
                    rows.clear()


def _sum(tensor, world):
    if world > 1:
        dist.all_reduce(tensor)
    return tensor


def _loss(out, cfg, world):
    local = out["rnnt"].new_tensor([out["rnnt_tokens"], out["ctc_tokens"]])
    counts = _sum(local.clone(), world).clamp_min_(1)
    # Undo each local denominator, even when a rank has zero accepted tokens.
    weights = world * local.clamp_min(1) / counts
    rnnt = out["rnnt"]
    if "rnnt_simple" in out:
        rnnt = rnnt + cfg.simple_loss_scale * out["rnnt_simple"]
    return weights[0] * rnnt + cfg.ctc_weight * weights[1] * out["ctc"]


def _state_digest(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().view(-1).view(torch.uint8).numpy())
    return digest.hexdigest()


def _assert_disjoint(rank_results, batch_size, total_steps):
    selected = []
    for batch_by_rank in zip(*(r["batch_ids"] for r in rank_results), strict=True):
        # The rank assignment was a stride, so interleave to recover global order.
        batch = [uids[i] for i in range(max(map(len, batch_by_rank)))
                 for uids in batch_by_rank if i < len(uids)]
        if len(batch) != batch_size:
            raise DataError("rank inputs did not assemble to one complete global batch")
        selected.extend(batch)
    if len(selected) != total_steps * batch_size or len(set(selected)) != len(selected):
        raise DataError("rank input coverage changed or repeated utterances")
    return hashlib.sha256("\n".join(selected).encode()).hexdigest()


def _run(args, rank, world, device):
    payload = checkpoint.load(str(args.seed), restore_rng=False)
    cfg = ModelConfig.from_dict(payload["model_config"])
    rnnt = Tokenizer(tokenizer_path(cfg.vocab_rnnt - 1))
    ctc = Tokenizer(tokenizer_path(cfg.vocab_ctc - 1))
    if payload.get("tokenizers") != {"rnnt": rnnt.sha256, "ctc": ctc.sha256}:
        raise CheckpointError("seed does not identify both local tokenizer digests")
    model = ForwardLoss(cfg).to(device)
    model.load_state_dict(payload["model"])
    del payload
    initial = _state_digest(model)
    ddp = (DistributedDataParallel(model, device_ids=[device.index] if device.type == "cuda" else None,
                                   gradient_as_bucket_view=True, broadcast_buffers=False)
           if world > 1 else model)
    opt = torch.optim.AdamW([
        {"params": [p for p in model.parameters() if p.ndim >= 2], "weight_decay": 0.01},
        {"params": [p for p in model.parameters() if p.ndim < 2], "weight_decay": 0.0},
    ], lr=args.lr, betas=(0.9, 0.98))
    model.eval()  # Keep the mathematical objective deterministic across world sizes.
    source = _batches(args.split, args.batch_size, args.max_seconds, rank, world, Featurizer(rnnt, ctc))
    ids = []
    audio = 0.0
    for index in range(args.steps + args.warmup):
        if index == args.warmup:
            if world > 1:
                dist.barrier()
            started = time.perf_counter()
        t0 = time.perf_counter()
        host_batch = next(source, None)
        if host_batch is None:
            raise DataError("pilot requested more batches than available training audio")
        ids.append(host_batch.uids)
        batch = host_batch.to(device)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=args.bf16):
            out = ddp(batch)
        loss = _loss(out, cfg, world)
        bad = _sum((~torch.isfinite(loss)).to(torch.int32), world)
        if int(bad):
            raise ValueError("at least one rank produced a non-finite loss")
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
        opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        if args.duty < 1.0:
            span = loss.new_tensor(time.perf_counter() - t0)
            if world > 1:
                dist.all_reduce(span, op=dist.ReduceOp.MAX)
            time.sleep(float(span) * (1.0 / args.duty - 1.0))
        if index >= args.warmup:
            audio += host_batch.audio_seconds
        print(json.dumps({"kind": "step", "rank": rank, "step": index + 1,
                          "local_audio_s": host_batch.audio_seconds,
                          "global_grad_norm": float(gnorm)}), flush=True)

    wall = time.perf_counter() - started
    state = _state_digest(model)
    record = {"rank": rank, "host": socket.gethostname(), "initial": initial, "state": state,
              "audio_s": audio, "wall_s": wall, "batch_ids": ids}
    if world > 1:
        ddp_stats = ddp._get_ddp_logging_data()
        record["collective"] = {k: ddp_stats[k] for k in (
            "bucket_sizes", "avg_backward_compute_time", "avg_backward_comm_time",
            "avg_backward_compute_comm_overlap_time",
        ) if k in ddp_stats}
    if world == 1:
        results = [record]
    else:
        results = [None] * world
        dist.all_gather_object(results, record)
    if rank == 0:
        if len({r["initial"] for r in results}) != 1 or len({r["state"] for r in results}) != 1:
            raise ValueError("model state differs across ranks")
        if state == initial:
            raise ValueError("pilot did not update the shared model")
        selected = _assert_disjoint(results, args.batch_size, args.steps + args.warmup)
        total_audio = sum(r["audio_s"] for r in results)
        wall = max(r["wall_s"] for r in results)
        print(json.dumps({"kind": "ddp_probe", "world_size": world, "initial_model_sha256": initial,
                          "synchronized_model_sha256": state, "utterance_order_sha256": selected,
                          "audio_s": total_audio, "wall_s": wall, "audio_h_per_wall_h": total_audio / wall,
                          "rank_hosts": [r["host"] for r in results], "steps": args.steps,
                          "collective": [r.get("collective", {}) for r in results],
                          "seed_sha256": args.seed.with_suffix(".sha256").read_text().strip()}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=Path, required=True)
    parser.add_argument("--split", choices=TRAIN_SPLITS, default="train-clean-100")
    parser.add_argument("--batch-size", type=int, default=32, help="utterances across all ranks")
    parser.add_argument("--max-seconds", type=float, default=20.0)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--duty", type=float, help="fraction of time training, default 0.85 on CUDA")
    parser.add_argument("--bf16", action="store_true")
    a = parser.parse_args()
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if a.steps < 1 or a.warmup < 0 or a.batch_size < world or a.max_seconds <= 0:
        parser.error("require at least one measured step and at least one utterance per rank")
    if a.duty is None:
        a.duty = 0.85 if a.device == "cuda" else 1.0
    if not 0 < a.duty <= 1:
        parser.error("duty must be in (0, 1]")
    torch.set_num_threads(a.threads)
    torch.set_num_interop_threads(1)
    device = torch.device("cpu")
    if a.device == "cuda":
        device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
        torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group("nccl" if a.device == "cuda" else "gloo", timeout=timedelta(minutes=2))
    try:
        _run(a, rank, world, device)
    finally:
        if world > 1:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
