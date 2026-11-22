"""Data parallelism (train/dist.py): two gloo ranks on CPU, each with half of every global batch,
end bit-close to one process on the concatenated batch (global-count loss normalisation,
averaged gradients, clip after sync, identical optimiser steps)."""
import dataclasses
import os
import socket
import sys

import pytest
import torch
import torch.multiprocessing as mp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from train.asr import PRESETS  # noqa: E402
from train.data import FixedBatches  # noqa: E402
from train.trainer import TrainConfig, Trainer  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _examples():
    sys.path.insert(0, os.path.dirname(__file__))
    from test_train_checkpoint import _examples as ex

    return ex(12, seed=3)


def _cfg():
    from test_train_checkpoint import _cfg as c

    base = c()
    return dataclasses.replace(base, encoder=dataclasses.replace(base.encoder, dropout=0.0))


def _train(run_dir, batch_size, steps, rank=0, world=1):
    torch.manual_seed(0)
    cfg = TrainConfig(preset="TINY", warmup=2, total_steps=100, ckpt_every_steps=1000,
                      ckpt_every_minutes=1e9, eval_every=0, log_every=1000, specaug=False,
                      clip=1.0)
    src = FixedBatches(_examples(), batch_size=batch_size, shuffle=False)
    t = Trainer(run_dir, cfg, src, model_cfg=_cfg(), device="cpu", log=False, rank=rank,
                world=world, local_rank=rank)
    if world > 1:
        from train.dist import broadcast_state

        broadcast_state(t.model)
    t.run(max_steps=steps)
    return t


def _worker(rank, world, port, run_dir, out):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank),
                      WORLD_SIZE=str(world), LOCAL_RANK=str(rank))
    torch.set_num_threads(1)
    import torch.distributed as dist

    dist.init_process_group("gloo")
    t = _train(os.path.join(run_dir, f"r{rank}"), 2, 5, rank, world)
    if rank == 0:
        torch.save(t.model.state_dict(), out)
    dist.destroy_process_group()


@pytest.fixture
def workdir(request):
    import shutil

    d = os.path.join(REPO, "target", "test-train", request.node.name)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_two_ranks_equal_one_process_on_the_concatenated_batch(workdir):
    torch.set_num_threads(1)
    torch.manual_seed(0)
    from train.asr import ASRModel

    init = ASRModel(_cfg()).state_dict()
    single = _train(os.path.join(workdir, "single"), 4, 5).model.state_dict()
    moved = [k for k, v in single.items() if v.is_floating_point() and not torch.equal(v, init[k])]
    assert len(moved) > 10  # the comparison is not vacuous: training changed the weights
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    out = os.path.join(workdir, "ddp.pt")
    mp.spawn(_worker, args=(2, port, workdir, out), nprocs=2, join=True)
    ddp = torch.load(out)
    for k, v in single.items():
        assert torch.allclose(ddp[k].float(), v.float(), atol=2e-5, rtol=1e-4), k
