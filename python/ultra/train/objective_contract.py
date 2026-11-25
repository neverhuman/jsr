"""Resolve the supported label-only objective and persist rank-local restart state."""
from __future__ import annotations

from train import checkpoint
from train.errors import CheckpointError


def validate(cfg, model_cfg) -> tuple | None:
    two_view = (cfg.cr_ctc_full or cfg.cr_ctc_half) and cfg.cr_ctc_weight > 0
    if two_view and (cfg.mcr_weight or cfg.etm_kd_weight or cfg.loss_trunc_q or cfg.text_loss_weight):
        raise CheckpointError("two-view CR-CTC bypasses the requested MCR, ETM, truncation or text objective")
    if not cfg.seal_data_identity:
        return
    if cfg.mcr_weight or cfg.etm_kd_weight or cfg.kd_weight or cfg.anchor_kappa:
        raise CheckpointError("sealed label-only runs require supervision from the sealed labels")
    if cfg.bpe_dropout_p or cfg.stream_ctx_s:
        raise CheckpointError("sealed restart has no witness for SentencePiece sampling or streaming context")
    if cfg.loss_trunc_q and cfg.loss_trunc_mode == "drop":
        raise CheckpointError("hard loss truncation does not mask the remaining label terms")
    if cfg.cr_ctc_weight > 0 or cfg.text_loss_weight > 0 or cfg.balancer_weight > 0 or cfg.zloss > 0:
        raise CheckpointError("sealed label-only runs support RNN-T, CTC, InterCTC and smooth loss capping")
    if cfg.delay_penalty and model_cfg.rnnt_loss != "pruned":
        raise CheckpointError("delay penalty requires the pruned RNN-T objective")
    if cfg.inter_ctc_weight and (not model_cfg.inter_ctc or model_cfg.encoder.family != "unet3"):
        raise CheckpointError("InterCTC requested without a model InterCTC head")
    if cfg.diloco_hub or cfg.data_world or cfg.ddp_bf16_comm:
        raise CheckpointError("sealed restart requires a fixed data-parallel layout with FP32 gradient reduction")
    if not all(split.startswith("train-") for split in cfg.splits):
        raise CheckpointError("sealed label-only training requires TRAIN splits")
    terms = [("rnnt", "rnnt_tokens"), ("ctc", "ctc_tokens")]
    if model_cfg.rnnt_loss == "pruned":
        terms.append(("rnnt_simple", "rnnt_tokens"))
    if cfg.inter_ctc_weight:
        terms.append(("inter_ctc", "inter_ctc_tokens"))
    return tuple(terms)


def active(tr) -> bool:
    return tr.data_identity is not None or tr.loss_trunc is not None or tr._fact_text is not None


def _layout(tr) -> list:
    return [tr.rank, tr.world, tr.local_rank, tr.data_rank, tr.data_world, tr.device.type]


def _trunc_settings(trunc) -> list:
    return [trunc.q, trunc.mode, trunc.warmup_steps, trunc.min_history, trunc.history.maxlen]


def snapshot(tr) -> dict:
    state = {
        "layout": _layout(tr), "position": tr.pos.to_dict(),
        "rng": checkpoint.rng_state(tr.device),
    }
    if tr.loss_trunc is not None:
        trunc = tr.loss_trunc
        state["truncation"] = {"settings": _trunc_settings(trunc), "history": list(trunc.history)}
    if tr._fact_text is not None:
        state["text"] = tr._fact_text.restart_state()
    return state


def collect(tr) -> dict:
    local = snapshot(tr)
    if tr.world == 1:
        return {"version": 1, "ranks": [local]}
    import torch.distributed as dist

    ranks = [None] * tr.world
    dist.all_gather_object(ranks, local)
    return {"version": 1, "ranks": ranks}


def restore(tr, state: dict) -> dict:
    if state.get("version") != 1 or len(state.get("ranks", ())) != tr.world:
        raise CheckpointError("objective restart state is missing or has a different world size")
    local = state["ranks"][tr.rank]
    if local["layout"] != _layout(tr):
        raise CheckpointError("objective restart rank layout changed")
    trunc, saved = tr.loss_trunc, local.get("truncation")
    if (trunc is None) != (saved is None):
        raise CheckpointError("objective restart loss truncation changed")
    if trunc is not None:
        if _trunc_settings(trunc) != saved["settings"]:
            raise CheckpointError("objective restart loss truncation settings changed")
        trunc.history.clear()
        trunc.history.extend(saved["history"])
    text = local.get("text")
    if (tr._fact_text is None) != (text is None):
        raise CheckpointError("objective restart text helper changed")
    if tr._fact_text is not None:
        tr._fact_text.restore_restart_state(text, tr.device)
    checkpoint.set_rng_state(local["rng"], tr.device)
    return local["position"]
