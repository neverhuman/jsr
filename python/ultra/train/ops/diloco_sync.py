"""Dev-gated cross-node weight sharing coordinator (`DiLoCo` + `Greedy Model Soup`).

Connects heterogeneous GPU runs sharing the exact same `(model_config, tokenizer_sha256)`
cohort (e.g. `D2e` 3x A100 + `D2` 3090 + `H13a` 3090, or `H22-kd` A100 + `H22ctl` A100 + `H22t`
4090) without cross-LAN NCCL stalls:
  1. Validates strict cohort compatibility (identical `model_config` and `tokenizer_sha256`).
  2. Packs compact BF16 pseudo-gradient deltas (`~116 MB` for 58M `S`) weighted by each
     worker's audio-hours trained since the anchor.
  3. Produces both `diloco_outer` (`DiLoCoOuterOptimizer`) and `greedy_soup` candidates.
  4. Gates promotion strictly on an evaluation callback (e.g. full-DEV beam-4 WER) so a
     promoted `--swap-weights` checkpoint is written ONLY when it beats the incumbent by
     at least `min_gain_pp`.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Callable, Mapping

import torch

from train.diloco import DiLoCoOuterOptimizer, greedy_model_soup, pack_bf16_delta, unpack_bf16_delta


@dataclass
class WorkerInput:
    name: str
    checkpoint_path: str
    audio_hours: float = 1.0
    use_ema: bool = True


@dataclass
class SyncOutcome:
    promoted: bool
    winner_name: str
    incumbent_wer: float
    winner_wer: float
    gain_pp: float
    candidate_wers: dict[str, float]
    soup_members: list[str]
    out_path: str | None


def extract_weights(payload: Mapping, use_ema: bool = True) -> dict[str, torch.Tensor]:
    if use_ema and payload.get("ema") is not None:
        return {k: v.detach().cpu() for k, v in payload["ema"].items()}
    return {k: v.detach().cpu() for k, v in payload["model"].items()}


def _cohort_key(payload: Mapping) -> tuple[str, str | None]:
    mc = payload.get("model_config") or {}
    # QAT flags don't alter inference graph weights when comparing base architecture,
    # while vocab/depth/width/family/tokenizer MUST match identically.
    norm_mc = {k: v for k, v in mc.items() if k not in ("qat", "qat_stem")}
    return json.dumps(norm_mc, sort_keys=True), payload.get("tokenizer_sha256")


def verify_cohort(anchor_payload: Mapping, worker_payloads: list[tuple[str, Mapping]]) -> None:
    ref_mc, ref_tok = _cohort_key(anchor_payload)
    for name, wp in worker_payloads:
        mc, tok = _cohort_key(wp)
        if mc != ref_mc:
            raise ValueError(f"cohort mismatch for {name}: model_config differs from anchor")
        if ref_tok is not None and tok is not None and tok != ref_tok:
            raise ValueError(f"cohort mismatch for {name}: tokenizer_sha256 {tok} != {ref_tok}")


def atomic_save_checkpoint(template_payload: Mapping, weights: dict[str, torch.Tensor],
                           out_path: str, meta: dict) -> None:
    out_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(out_dir, exist_ok=True)
    payload = {
        "model": {k: v.detach().cpu() for k, v in weights.items()},
        "ema": {k: v.detach().cpu() for k, v in weights.items()},
        "model_config": template_payload.get("model_config"),
        "tokenizer_sha256": template_payload.get("tokenizer_sha256"),
        "step": template_payload.get("step", 0),
        "diloco_sync": meta,
    }
    fd, tmp = tempfile.mkstemp(prefix=".swap_", suffix=".pt", dir=out_dir)
    os.close(fd)
    try:
        torch.save(payload, tmp)
        os.replace(tmp, out_path)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def run_sync_round(
    anchor_payload: Mapping,
    workers: list[tuple[str, Mapping, float]],
    eval_fn: Callable[[dict[str, torch.Tensor]], float],
    *,
    outer_state: dict | None = None,
    outer_lr: float = 0.7,
    outer_momentum: float = 0.9,
    bf16_deltas: bool = True,
    use_ema: bool = True,
    min_gain_pp: float = 0.03,
    out_path: str | None = None,
) -> tuple[SyncOutcome, dict]:
    """Execute one dev-gated cross-node DiLoCo + Greedy Model Soup round."""
    if not workers:
        raise ValueError("workers must be non-empty")
    verify_cohort(anchor_payload, [(n, p) for n, p, _ in workers])

    anchor_w = extract_weights(anchor_payload, use_ema=use_ema)
    opt = DiLoCoOuterOptimizer(anchor_w, lr=outer_lr, momentum=outer_momentum, nesterov=True)
    if outer_state is not None:
        opt.load_state_dict(outer_state)
        opt.anchor = {k: (v.float().cpu().clone() if v.is_floating_point() else v.cpu().clone())
                      for k, v in anchor_w.items()}

    worker_ws: list[tuple[str, dict[str, torch.Tensor], float]] = []
    deltas: list[dict[str, torch.Tensor]] = []
    weights_audio: list[float] = []
    for name, payload, audio_h in workers:
        if audio_h <= 0.0:
            raise ValueError(f"audio_hours for {name} must be > 0 (got {audio_h})")
        w = extract_weights(payload, use_ema=use_ema)
        worker_ws.append((name, w, float(audio_h)))
        d = pack_bf16_delta(opt.anchor, w) if bf16_deltas else {
            k: (opt.anchor[k] - w[k].float().cpu() if opt.anchor[k].is_floating_point() else w[k].cpu())
            for k in opt.anchor
        }
        deltas.append(unpack_bf16_delta(d) if bf16_deltas else d)
        weights_audio.append(float(audio_h))

    diloco_w = opt.step_from_deltas(deltas, weights=weights_audio)

    # Evaluate incumbent, individual workers, and DiLoCo outer candidate.
    incumbent_wer = float(eval_fn(anchor_w))
    cand_wers: dict[str, float] = {"incumbent": incumbent_wer}
    pool: list[tuple[str, dict[str, torch.Tensor], float]] = [("incumbent", anchor_w, incumbent_wer)]

    diloco_wer = float(eval_fn(diloco_w))
    cand_wers["diloco_outer"] = diloco_wer
    pool.append(("diloco_outer", diloco_w, diloco_wer))

    for name, w, _ in worker_ws:
        wer = float(eval_fn(w))
        cand_wers[name] = wer
        pool.append((name, w, wer))

    soup_w, soup_wer, soup_members = greedy_model_soup(pool, eval_fn)
    cand_wers["greedy_soup"] = soup_wer

    best_name, best_w, best_wer = "incumbent", anchor_w, incumbent_wer
    for name, w, wer in pool + [("greedy_soup", soup_w, soup_wer)]:
        if wer < best_wer:
            best_name, best_w, best_wer = name, w, wer

    gain_pp = round(incumbent_wer - best_wer, 4)
    promoted = best_name != "incumbent" and gain_pp >= min_gain_pp
    written_path: str | None = None
    if promoted and out_path:
        meta = {
            "time": time.time(),
            "winner": best_name,
            "incumbent_wer": incumbent_wer,
            "winner_wer": best_wer,
            "gain_pp": gain_pp,
            "soup_members": soup_members,
            "candidates": cand_wers,
            "workers": [{"name": n, "audio_hours": ah} for n, _, ah in worker_ws],
        }
        atomic_save_checkpoint(anchor_payload, best_w, out_path, meta)
        written_path = out_path

    outcome = SyncOutcome(
        promoted=promoted, winner_name=best_name, incumbent_wer=incumbent_wer,
        winner_wer=best_wer, gain_pp=gain_pp, candidate_wers=cand_wers,
        soup_members=soup_members, out_path=written_path,
    )
    return outcome, opt.state_dict()


def _parse_worker_arg(spec: str) -> WorkerInput:
    # Format: name=path[:audio_hours]
    name, rest = spec.split("=", 1)
    if ":" in rest:
        path, ah_s = rest.rsplit(":", 1)
        ah = float(ah_s)
    else:
        path, ah = rest, 1.0
    return WorkerInput(name=name, checkpoint_path=path, audio_hours=ah)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--anchor", required=True, help="incumbent anchor checkpoint .pt")
    ap.add_argument("--worker", action="append", required=True,
                    help="worker spec name=/path/to/ckpt.pt[:audio_hours]")
    ap.add_argument("--eval-cmd", required=True,
                    help="shell command template with {ckpt} printing JSON with 'wer' key")
    ap.add_argument("--out", required=True, help="output path for promoted --swap-weights .pt")
    ap.add_argument("--state", default=None, help="optional DiLoCo outer state .pt")
    ap.add_argument("--receipt", default=None, help="optional JSON receipt output path")
    ap.add_argument("--outer-lr", type=float, default=0.7)
    ap.add_argument("--outer-momentum", type=float, default=0.9)
    ap.add_argument("--min-gain-pp", type=float, default=0.03)
    ap.add_argument("--raw", action="store_true", help="use raw model weights instead of EMA")
    a = ap.parse_args(argv)

    import shlex
    import subprocess

    anchor_p = torch.load(a.anchor, map_location="cpu", weights_only=False)
    worker_inputs = [_parse_worker_arg(s) for s in a.worker]
    workers = [
        (wi.name, torch.load(wi.checkpoint_path, map_location="cpu", weights_only=False), wi.audio_hours)
        for wi in worker_inputs
    ]
    outer_sd = torch.load(a.state, map_location="cpu", weights_only=False) if (a.state and os.path.exists(a.state)) else None

    def eval_fn(weights: dict[str, torch.Tensor]) -> float:
        with tempfile.TemporaryDirectory(prefix="diloco_eval_") as td:
            ckpt = os.path.join(td, "cand.pt")
            atomic_save_checkpoint(anchor_p, weights, ckpt, {"eval_tmp": True})
            argv_cmd = [tok.replace("{ckpt}", ckpt) for tok in shlex.split(a.eval_cmd)]
            out = subprocess.check_output(argv_cmd, text=True)
            data = json.loads(out.strip().splitlines()[-1])
            return float(data["wer"])

    outcome, new_state = run_sync_round(
        anchor_p, workers, eval_fn, outer_state=outer_sd,
        outer_lr=a.outer_lr, outer_momentum=a.outer_momentum,
        use_ema=not a.raw, min_gain_pp=a.min_gain_pp, out_path=a.out,
    )
    if a.state:
        torch.save(new_state, a.state)
    summary = dataclasses.asdict(outcome)
    if a.receipt:
        with open(a.receipt, "w") as fh:
            json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
