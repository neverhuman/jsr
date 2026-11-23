"""R0 teacher on GPU, fp32, batched (kd-cache, v3 H49 / H52).

Teacher convention "r0-fp32-gpu-v1" (to be pre-registered as an H22 amendment):
  * R0 = sherpa-onnx streaming zipformer-en-2023-06-26, fp32 ONNX (encoder a423883c..., decoder
    7bf787f9..., joiner 210591f7...), tokenizer bpe.model c53433de... (500 pieces, blank 0).
  * Frontend and perturbation are train/teacher_cache.py's own functions (R0.fbank: kaldi-native-fbank,
    80 mel, dither 0, snip_edges false, high_freq -400, 0.3 s tail pad; perturb: resample_poly with
    Fraction(f).limit_denominator(100), round, clip, int16), imported from the unmodified code.
  * Encoder on onnxruntime-gpu (CUDA EP), batched streams in lock step: windows of T=45 fbank frames
    advanced by 32 (chunk 16 / left 128), full chunks only, 8 encoder frames (25 Hz) per chunk.
    Batch elements are independent in the encoder; outputs past an utterance's last chunk are dropped.
  * Decoder (stateless, context 2) and joiner re-implemented in PyTorch fp32 from the ONNX
    initialisers; `check` compares them and the encoder with onnxruntime on CPU.
  * frame: greedy, one symbol per frame, top-k joiner log-probs per frame (teacher_cache.py's law).
  * seq: modified beam search (at most one symbol per frame, equal token sequences merged by
    log-sum-exp after each frame's top-`beam` selection, as icefall / train.beam), and forced scores
    of any token sequence: full-sum RNN-T log-likelihood (k2 rnnt_type="regular") and the
    one-symbol-per-frame law (k2 rnnt_type="modified").
TRAIN splits only. Results only (caches, scores); no weights are written.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import io
import json
import math
import os
import sys
import time

import numpy as np

CODE = os.environ.get("KD_CODE_PARTS", "/mnt/xbabe3-external/jsr/kd-cache/code/"
                      "e9ca2ce5abcc8c6fa269f1a1ced0b9cc48efb109/python/ultra")
MODEL = os.environ.get("KD_MODEL_FP32", "/mnt/xbabe3-external/jsr/models/zipformer-en-2023-06-26")
TAG = "epoch-99-avg-1-chunk-16-left-128"
HASHES = {"encoder": "a423883ce5754507fd941755ab0b5bc426a84ac670cbe21cf060e9e2c66dc660",
          "decoder": "7bf787f90b194b307e5a4ad6a34fadb4e748304c35f78a8d66358a05b13ee6ef",
          "joiner": "210591f72b3c56b8364f85f345dca240bc2b4c00632848f4aa923630d5639d3b",
          "bpe": "c53433de083c4a6ad12d034550ef22de68cec62c4f58932a7b6b8b2f1e743fa5"}
CONVENTION = "r0-fp32-gpu-v1"
SPLIT_DIRS = {"train-clean-100": "train.clean.100", "train-clean-360": "train.clean.360",
              "train-other-500": "train.other.500"}
WIN, HOP, FPC = 45, 32, 8  # fbank window per chunk, fbank hop per chunk, encoder frames per chunk


def sha256_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def shard_file(split: str, shard: int) -> str:
    if split not in SPLIT_DIRS:
        raise ValueError(f"split {split!r} refused: the teacher only sees TRAIN splits")
    data = os.environ.get("JSR_DATA", "/mnt/xbabe3-external/jsr")
    files = sorted(glob.glob(os.path.join(data, "corpora/librispeech/hf_parquet/all", SPLIT_DIRS[split], "*.parquet")))
    return files[shard]


# ---------------------------------------------------------------- frontend (CPU worker processes)
_W = {}


def _winit(code: str):
    sys.path.insert(0, code)
    from train.teacher_cache import R0, perturb  # the verified frontend and perturbation
    _W["fbank"], _W["perturb"] = R0.fbank, perturb


def _wfeat(args):
    uid, text, flac, speed = args
    import soundfile as sf

    pcm, sr = sf.read(io.BytesIO(flac), dtype="int16")
    if sr != 16000 or pcm.ndim != 1:
        raise ValueError(f"{uid}: expected 16 kHz mono")
    pcm = _W["perturb"](pcm, speed)
    feats = _W["fbank"](pcm)
    return uid, text, feats, len(pcm), hashlib.sha256(pcm.tobytes()).hexdigest()


# ---------------------------------------------------------------- model
class R0GPU:
    def __init__(self, model_dir: str = MODEL, device: str = "cuda", mem_gb: float = 8.0, check_hashes=True):
        import onnx
        import onnxruntime as ort
        import torch
        from onnx import numpy_helper

        self.torch, self.device = torch, torch.device(device)
        torch.backends.cuda.matmul.allow_tf32 = False  # the convention is fp32, not TF32
        torch.backends.cudnn.allow_tf32 = False
        paths = {n: os.path.join(model_dir, f"{n}-{TAG}.onnx") for n in ("encoder", "decoder", "joiner")}
        paths["bpe"] = os.path.join(model_dir, "bpe.model")
        if check_hashes:
            for n, p in paths.items():
                h = sha256_file(p)
                if h != HASHES[n]:
                    raise RuntimeError(f"{p}: sha256 {h} != pinned {HASHES[n]}")
        self.paths = paths
        so = ort.SessionOptions()
        so.intra_op_num_threads = 2
        so.inter_op_num_threads = 1
        prov = [("CUDAExecutionProvider", {"device_id": 0, "gpu_mem_limit": int(mem_gb * 2**30),
                                           "arena_extend_strategy": "kSameAsRequested",
                                           "cudnn_conv_algo_search": "HEURISTIC",
                                           "cudnn_conv_use_max_workspace": "1",
                                           "use_tf32": "0"}), "CPUExecutionProvider"]
        self.enc = ort.InferenceSession(paths["encoder"], so, providers=prov)
        if self.enc.get_providers()[0] != "CUDAExecutionProvider":
            raise RuntimeError(f"encoder not on CUDA: {self.enc.get_providers()}")
        meta = self.enc.get_modelmeta().custom_metadata_map
        assert int(meta["T"]) == WIN and int(meta["decode_chunk_len"]) == HOP, meta
        self.state_in = [i for i in self.enc.get_inputs() if i.name != "x"]
        outs = [o.name for o in self.enc.get_outputs()]
        self.enc_out_name = outs[0]
        self.state_out = outs[1:]
        assert [o for o in self.state_out] == ["new_" + i.name for i in self.state_in], "state order"

        def init(path):
            return {t.name: numpy_helper.to_array(t) for t in onnx.load(path).graph.initializer}

        d, j = init(paths["decoder"]), init(paths["joiner"])
        T = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(self.device)  # noqa: E731
        self.emb, self.conv_w = T(d["decoder.embedding.weight"]), T(d["decoder.conv.weight"])
        self.dproj_w, self.dproj_b = T(d["decoder_proj.weight"]), T(d["decoder_proj.bias"])
        self.jw, self.jb = T(j["output_linear.weight"]), T(j["output_linear.bias"])
        self.groups = self.emb.shape[1] // self.conv_w.shape[1]  # 512 / 4 = 128
        import sentencepiece as spm

        self.sp = spm.SentencePieceProcessor(model_file=paths["bpe"])
        self.ort = ort

    # stateless predictor, context 2: embedding(clamp(y, 0)) * (y >= 0) -> grouped conv -> relu -> proj
    def decoder(self, y):
        F = self.torch.nn.functional
        e = self.emb[y.clamp(min=0)] * (y >= 0).unsqueeze(-1).to(self.emb.dtype)  # (N, 2, 512)
        x = F.conv1d(e.transpose(1, 2), self.conv_w, groups=self.groups)       # (N, 512, 1)
        return F.linear(F.relu(x.squeeze(-1)), self.dproj_w, self.dproj_b)

    def joiner(self, enc, dec):
        return self.torch.nn.functional.linear(self.torch.tanh(enc + dec), self.jw, self.jb)

    # ------------------------------------------------------------ batched streaming encoder
    def encode(self, feats: list[np.ndarray]):
        """feats: list of (n_fb, 80) float32. Returns (enc (B, Tmax, 512) cuda tensor, n_frames list)."""
        torch = self.torch
        B = len(feats)
        nch = [max(0, (len(f) - WIN) // HOP + 1) if len(f) >= WIN else 0 for f in feats]
        cmax = max(nch) if nch else 0
        nfr = [FPC * c for c in nch]
        enc = torch.zeros((B, FPC * cmax, 512), dtype=torch.float32, device=self.device)
        if cmax == 0:
            return enc, nfr
        L = WIN + HOP * (cmax - 1)
        x = np.zeros((B, L, 80), np.float32)
        for b, f in enumerate(feats):
            n = min(len(f), L)
            x[b, :n] = f[:n]
        xg = torch.from_numpy(x).to(self.device)

        def shp(i):
            return [B if not isinstance(dm, int) else dm for dm in i.shape]

        dt = {"tensor(float)": torch.float32, "tensor(int64)": torch.int64}
        st = [torch.zeros(shp(i), dtype=dt[i.type], device=self.device) for i in self.state_in]
        st2 = [torch.empty_like(s) for s in st]
        out = torch.empty((B, FPC, 512), dtype=torch.float32, device=self.device)
        npdt = {torch.float32: np.float32, torch.int64: np.int64}
        for c in range(cmax):
            xc = xg[:, HOP * c: HOP * c + WIN].contiguous()
            io_ = self.enc.io_binding()
            io_.bind_input("x", "cuda", 0, np.float32, list(xc.shape), xc.data_ptr())
            for i, s in zip(self.state_in, st):
                io_.bind_input(i.name, "cuda", 0, npdt[s.dtype], list(s.shape), s.data_ptr())
            io_.bind_output(self.enc_out_name, "cuda", 0, np.float32, list(out.shape), out.data_ptr())
            for name, s in zip(self.state_out, st2):
                io_.bind_output(name, "cuda", 0, npdt[s.dtype], list(s.shape), s.data_ptr())
            # v1.1 (the only change from v1 41be5573): onnxruntime runs on its own CUDA stream, while
            # the input buffers are written by torch kernels on torch's stream. Wait for them before
            # the run, and wait for every stream again before torch reads the bound output.
            torch.cuda.synchronize()
            self.enc.run_with_iobinding(io_)
            torch.cuda.synchronize()
            enc[:, FPC * c: FPC * (c + 1)] = out
            st, st2 = st2, st
        return enc, nfr

    # ------------------------------------------------------------ greedy + top-k (frame cache)
    def greedy_topk(self, enc, nfr, k=8):
        torch = self.torch
        B, Tm, _ = enc.shape
        ctx = torch.zeros((B, 2), dtype=torch.long, device=self.device)
        dec = self.decoder(ctx)
        ids = torch.zeros((B, Tm, k), dtype=torch.int16, device=self.device)
        lps = torch.zeros((B, Tm, k), dtype=torch.float16, device=self.device)
        emit = torch.zeros((B, Tm), dtype=torch.int16, device=self.device)
        lens = torch.tensor(nfr, device=self.device)
        for t in range(Tm):
            live = lens > t
            lp = torch.log_softmax(self.joiner(enc[:, t], dec), dim=-1)
            v, ix = torch.topk(lp, k, dim=-1)
            ids[:, t], lps[:, t] = ix.to(torch.int16), v.to(torch.float16)
            y = ix[:, 0]
            emit[:, t] = y.to(torch.int16)
            upd = live & (y != 0)
            if bool(upd.any()):
                ctx = torch.where(upd[:, None], torch.stack([ctx[:, 1], y], 1), ctx)
                dec = torch.where(upd[:, None], self.decoder(ctx), dec)
        ids, lps, emit = ids.cpu().numpy(), lps.cpu().numpy(), emit.cpu().numpy()
        res = []
        for b in range(B):
            n = nfr[b]
            e = emit[b, :n]
            res.append((ids[b, :n], lps[b, :n], e, self.sp.decode([int(x) for x in e[e > 0]])))
        return res

    # ------------------------------------------------------------ modified beam search, N-best
    def beam_nbest(self, enc, nfr, beam=8):
        return beam_nbest_generic(self.decoder, self.joiner, enc, nfr, beam)

    # ------------------------------------------------------------ forced scores
    def force_scores(self, enc_b, n_frames: int, seqs: list[list[int]]):
        """Full-sum (regular) and one-symbol-per-frame (modified) log P(y|x) for token sequences of
        one utterance. enc_b: (Tmax, 512) cuda tensor. Returns lists of floats (-inf if infeasible)."""
        import k2

        torch = self.torch
        T = n_frames
        if not seqs:
            return [], []
        U = max(1, max(len(s) for s in seqs))
        S = len(seqs)
        ys = torch.zeros((S, U), dtype=torch.long, device=self.device)
        for i, s in enumerate(seqs):
            if s:
                ys[i, :len(s)] = torch.tensor(s, device=self.device)
        ctx_full = torch.cat([torch.zeros((S, 2), dtype=torch.long, device=self.device), ys], 1)  # blank, blank, y1..
        ctx = torch.stack([ctx_full[:, u:u + 2] for u in range(U + 1)], 1)                      # (S, U+1, 2)
        dec = self.decoder(ctx.view(-1, 2)).view(S, U + 1, -1)
        logits = self.joiner(enc_b[None, :T, None, :], dec[:, None, :, :])                       # (S, T, U+1, V)
        bnd = torch.zeros((S, 4), dtype=torch.int64, device=self.device)
        bnd[:, 2] = torch.tensor([len(s) for s in seqs], device=self.device)
        bnd[:, 3] = T
        full = -k2.rnnt_loss(logits=logits, symbols=ys, termination_symbol=0, boundary=bnd,
                             rnnt_type="regular", reduction="none")
        feas = torch.tensor([len(s) <= T for s in seqs], device=self.device)
        mod = -k2.rnnt_loss(logits=logits, symbols=ys, termination_symbol=0, boundary=bnd,
                            rnnt_type="modified", reduction="none")
        mod = torch.where(feas, mod, torch.full_like(mod, -math.inf))
        return full.tolist(), mod.tolist()


def beam_nbest_generic(decoder, joiner, enc, nfr, beam=8):
    """Batched modified beam search (icefall / train.beam law): per frame, the `beam` best
    (hyp, token) extensions of the live hypotheses, then equal token sequences merged by
    log-sum-exp into the best-ranked copy (so the beam may hold fewer than `beam` after a merge);
    at most one non-blank token per frame; context = last two tokens, blank-padded.
    decoder(ctx (N, 2) long) -> (N, J); joiner(enc (..., J), dec (..., J)) -> logits (..., V).
    enc (B, Tmax, J) on the device, nfr frame counts. Returns per utterance [(tokens, score), ...]."""
    import torch

    dev = enc.device
    B, Tm, _ = enc.shape
    K = beam
    P = 1000003
    score = torch.full((B, K), -math.inf, device=dev)
    score[:, 0] = 0.0
    ctx = torch.zeros((B, K, 2), dtype=torch.long, device=dev)
    hsh = torch.zeros((B, K), dtype=torch.int64, device=dev)
    toks = torch.zeros((B, K, max(1, Tm)), dtype=torch.int32, device=dev)
    tlen = torch.zeros((B, K), dtype=torch.long, device=dev)
    lens = torch.tensor(nfr, device=dev)
    ar = torch.arange(B, device=dev)[:, None]
    for t in range(Tm):
        live = lens > t
        dec = decoder(ctx.view(B * K, 2)).view(B, K, -1)
        lp = torch.log_softmax(joiner(enc[:, t, None, :].expand(-1, K, -1), dec).float(), dim=-1)  # (B, K, V)
        V = lp.shape[-1]
        cand = (score[:, :, None] + lp).view(B, K * V)
        cs, ci = torch.topk(cand, K, dim=-1)                                                       # sorted
        src, v = ci // V, ci % V
        nh = torch.where(v == 0, hsh[ar, src], hsh[ar, src] * P + v + 1)
        eq = (nh[:, :, None] == nh[:, None, :]) & torch.isfinite(cs)[:, None, :] & torch.isfinite(cs)[:, :, None]
        first = ~torch.tril(eq, diagonal=-1).any(-1)
        merged = torch.logsumexp(torch.where(eq, cs[:, None, :], torch.full_like(cs[:, None, :], -math.inf)), -1)
        ns = torch.where(first & torch.isfinite(cs), merged, torch.full_like(cs, -math.inf))
        nt = toks[ar, src].clone()
        nl = tlen[ar, src].clone()
        emitm = (v != 0) & torch.isfinite(ns)
        bi, ki = emitm.nonzero(as_tuple=True)
        nt[bi, ki, nl[bi, ki]] = v[bi, ki].to(torch.int32)
        nl = nl + emitm.long()
        pctx = ctx[ar, src]
        nctx = torch.where(emitm[..., None], torch.stack([pctx[..., 1], v], -1), pctx)
        m = live[:, None]
        score = torch.where(m, ns, score)
        hsh = torch.where(m, nh, hsh)
        toks = torch.where(m[..., None], nt, toks)
        tlen = torch.where(m, nl, tlen)
        ctx = torch.where(m[..., None], nctx, ctx)
    score, toks, tlen = score.cpu().numpy(), toks.cpu().numpy(), tlen.cpu().numpy()
    out = []
    for b in range(B):
        hyps = []
        for kk in np.argsort(-score[b], kind="stable"):
            if np.isfinite(score[b, kk]):
                hyps.append(([int(x) for x in toks[b, kk, :tlen[b, kk]]], float(score[b, kk])))
        out.append(hyps)
    return out


# ---------------------------------------------------------------- units and I/O
def read_rows(split, shard, rg_lo, rg_hi):
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(shard_file(split, shard))
    rg_hi = pf.metadata.num_row_groups if rg_hi is None else rg_hi
    rows = []
    for rg in range(rg_lo, rg_hi):
        rows += pf.read_row_group(rg, columns=["id", "text", "audio"]).to_pylist()
    return rows, pf.metadata.num_row_groups


def unit_name(u, speed):
    base = f"{u['split']}_{u['shard']:04d}"
    if u.get("rg") is not None:
        base += f"_rg{u['rg'][0]:02d}-{u['rg'][1]:02d}"
    return f"{base}_sp{float(speed):.1f}"


def batches_by_length(items, max_frames):
    """Group utterances (sorted by length) so that B * Tmax stays under max_frames."""
    order = sorted(range(len(items)), key=lambda i: len(items[i][2]))
    cur, cur_max = [], 0
    for i in order:
        n = len(items[i][2])
        if cur and max(cur_max, n) * (len(cur) + 1) > max_frames:
            yield cur
            cur, cur_max = [], 0
        cur.append(i)
        cur_max = max(cur_max, n)
    if cur:
        yield cur


def run_meta(r0, args):
    import onnxruntime as ort
    import scipy
    import torch

    import kaldi_native_fbank as knf

    return {"convention": CONVENTION, "teacher": "R0 zipformer-en-2023-06-26 chunk-16-left-128 fp32",
            "model_sha256": {n: HASHES[n] for n in ("encoder", "decoder", "joiner")},
            "tokenizer_sha256": HASHES["bpe"], "vocab": 500, "blank": 0,
            "frontend": {"fbank": "kaldi-native-fbank", "knf": knf.__version__, "num_bins": 80, "dither": 0,
                         "snip_edges": False, "high_freq": -400, "samp_freq": 16000, "tail_pad_s": 0.3},
            "perturbation": {"recipe": "scipy.signal.resample_poly(x, F.denominator, F.numerator), "
                                       "F = Fraction(f).limit_denominator(100); round; clip; int16",
                             "scipy": scipy.__version__, "numpy": np.__version__},
            "time_support": {"frame_shift_s": 0.04, "fbank_shift_s": 0.01, "window_fbank_frames": WIN,
                             "chunk_fbank_frames": HOP, "frames_per_chunk": FPC, "left_context_frames": 128,
                             "full_chunks_only": True,
                             "rule": "frame t in chunk c = t // 8 sees fbank frames [32c, 32c + 45)"},
            "runtime": {"onnxruntime": ort.__version__, "provider": "CUDAExecutionProvider",
                        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
                        "decoder_joiner": "pytorch fp32 from ONNX initialisers",
                        "batching": f"length-sorted, max {args.max_frames} padded fbank frames per batch",
                        "tf32": False},
            "code_sha256": sha256_file(os.path.abspath(__file__)),
            "teacher_cache_code_sha256": sha256_file(os.path.join(CODE, "train/teacher_cache.py"))}


def process_unit(r0, pool, u, speeds, args, meta):
    rows, nrg = read_rows(u["split"], u["shard"], *(u.get("rg") or (0, None)))
    outs = {}
    for speed in speeds:
        name = unit_name(u, speed)
        t0 = time.time()
        feats = pool.map(_wfeat, [(r["id"], r["text"], r["audio"]["bytes"], float(speed)) for r in rows], chunksize=4)
        frame = {}
        seq = {}
        for idx in batches_by_length(feats, args.max_frames):
            enc, nfr = r0.encode([feats[i][2] for i in idx])
            if "frame" in args.mode:
                for i, res in zip(idx, r0.greedy_topk(enc, nfr, args.topk)):
                    frame[i] = res
            if "seq" in args.mode and float(speed) in args.seq_speeds:
                nb = r0.beam_nbest(enc, nfr, args.beam)
                for j, i in enumerate(idx):
                    uid, text = feats[i][0], feats[i][1]
                    cands = {}
                    for rank, (s, sc) in enumerate(nb[j]):
                        key = tuple(s)
                        cands.setdefault(key, {"r0_ids": s, "text": r0.sp.decode(s), "sources": [],
                                               "r0_beam_score": sc, "r0_beam_rank": rank})["sources"].append("r0_beam")
                    ref_ids = r0.sp.encode(text.upper()) if text else []
                    cands.setdefault(tuple(ref_ids), {"r0_ids": ref_ids, "text": text, "sources": []})["sources"].append("ref")
                    for extra in args.extra.get(uid, []):
                        ids = r0.sp.encode(extra["text"].upper())
                        c = cands.setdefault(tuple(ids), {"r0_ids": ids, "text": extra["text"], "sources": []})
                        c["sources"].append(extra.get("source", "extra"))
                    lst = list(cands.values())
                    full, mod = r0.force_scores(enc[j], nfr[j], [c["r0_ids"] for c in lst])
                    for c, f, m in zip(lst, full, mod):
                        c["full_sum"], c["one_per_frame"] = f, m
                        if not math.isfinite(f):
                            raise RuntimeError(f"{uid}: non-finite full-sum score")
                    seq[i] = {"uid": uid, "speed": float(speed), "n_frames": nfr[j], "candidates": lst}
        if "frame" in args.mode:
            os.makedirs(os.path.join(args.out, "frame"), exist_ok=True)
            order = list(range(len(feats)))
            uids = [feats[i][0] for i in order]
            fmeta = dict(meta, speed=float(speed), topk=args.topk, split=u["split"],
                         shard=os.path.basename(shard_file(u["split"], u["shard"])),
                         row_groups=list(u["rg"]) if u.get("rg") else [0, nrg])
            tmp = os.path.join(args.out, "frame", name + ".tmp.npz")
            np.savez(tmp, uid=np.array(uids), text=np.array([feats[i][1] for i in order]),
                     hyp=np.array([frame[i][3] for i in order]),
                     n_frames=np.array([len(frame[i][2]) for i in order], np.int32),
                     topk_ids=np.concatenate([frame[i][0] for i in order]) if order else np.zeros((0, args.topk), np.int16),
                     topk_lp=np.concatenate([frame[i][1] for i in order]) if order else np.zeros((0, args.topk), np.float16),
                     emitted=np.concatenate([frame[i][2] for i in order]) if order else np.zeros(0, np.int16),
                     meta=np.array(json.dumps({"teacher": fmeta["teacher"] + " (" + CONVENTION + ")",
                                               "speed": float(speed), "topk": args.topk, "split": u["split"],
                                               "shard": fmeta["shard"], "convention": CONVENTION})))
            final = os.path.join(args.out, "frame", name + ".npz")
            os.replace(tmp, final)
            kd = {"file": name + ".npz", "sha256": sha256_file(final), "meta": fmeta,
                  "utterances": [{"uid": feats[i][0], "n_samples": feats[i][3], "pcm_sha256": feats[i][4],
                                  "n_fbank": int(len(feats[i][2])), "n_frames": int(len(frame[i][2]))} for i in order]}
            with open(os.path.join(args.out, "frame", name + ".kdmeta.json"), "w") as fh:
                json.dump(kd, fh)
            with open(os.path.join(args.out, "frame", "SHA256SUMS"), "a") as fh:
                fh.write(f"{kd['sha256']}  {name}.npz\n")
        if seq:
            os.makedirs(os.path.join(args.out, "seq"), exist_ok=True)
            p = os.path.join(args.out, "seq", name + ".seq.jsonl")
            with open(p + ".tmp", "w") as fh:
                fh.write(json.dumps({"header": True, **meta, "beam": args.beam, "speed": float(speed)}) + "\n")
                for i in range(len(feats)):
                    if i in seq:
                        fh.write(json.dumps(seq[i]) + "\n")
            os.replace(p + ".tmp", p)
            with open(os.path.join(args.out, "seq", "SHA256SUMS"), "a") as fh:
                fh.write(f"{sha256_file(p)}  {name}.seq.jsonl\n")
        audio = sum(f[3] for f in feats) / 16000.0
        outs[name] = {"utts": len(feats), "audio_s": round(audio, 1), "wall_s": round(time.time() - t0, 1)}
        print(json.dumps({"unit": name, **outs[name], "x_rt": round(audio / max(1e-9, time.time() - t0), 1)}), flush=True)
    return outs


def pilot_units():
    """~200 row groups spread over all 126 TRAIN shards (deterministic): row group (7s + 3) mod n of
    shard s, plus a second one halfway round for the first 74 shards."""
    import pyarrow.parquet as pq

    units, s_glob = [], 0
    for split, n in (("train-clean-100", 14), ("train-clean-360", 48), ("train-other-500", 64)):
        for shard in range(n):
            nrg = pq.ParquetFile(shard_file(split, shard)).metadata.num_row_groups
            r = (7 * s_glob + 3) % nrg
            units.append({"split": split, "shard": shard, "rg": [r, r + 1]})
            if s_glob < 74:
                r2 = (r + nrg // 2) % nrg
                units.append({"split": split, "shard": shard, "rg": [r2, r2 + 1]})
            s_glob += 1
    return units


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("cmd", choices=["run", "units-pilot", "check"])
    ap.add_argument("--units", help="JSON list of {split, shard, rg?} (or 'pilot' / 'all')")
    ap.add_argument("--mode", default="frame,seq")
    ap.add_argument("--speeds", default="0.9,1.0,1.1")
    ap.add_argument("--seq-speeds", default="1.0")
    ap.add_argument("--beam", type=int, default=8)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--out", default="/mnt/xbabe3-external/jsr/kd-cache/r0gpu/pilot")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-frames", type=int, default=64000, help="padded fbank frames per batch")
    ap.add_argument("--mem-gb", type=float, default=8.0)
    ap.add_argument("--extra", default="", help="JSONL of {uid, text, source} extra candidates (e.g. student N-best)")
    ap.add_argument("--limit-units", type=int, default=0)
    a = ap.parse_args(argv)
    a.seq_speeds = {float(x) for x in a.seq_speeds.split(",") if x}
    speeds = [x for x in a.speeds.split(",") if x]
    for s in speeds:
        if float(s) not in (0.9, 1.0, 1.1):
            ap.error("speeds must be among 0.9, 1.0, 1.1")
    if a.cmd == "units-pilot":
        print(json.dumps(pilot_units()))
        return 0
    if a.units == "pilot":
        units = pilot_units()
    elif a.units == "all":
        units = [{"split": s, "shard": i} for s, n in (("train-clean-100", 14), ("train-clean-360", 48),
                                                        ("train-other-500", 64)) for i in range(n)]
    else:
        units = json.load(open(a.units))
    if a.limit_units:
        units = units[:a.limit_units]
    extra_path, a.extra = a.extra, {}
    if extra_path:
        for line in open(extra_path):
            d = json.loads(line)
            a.extra.setdefault(d["uid"], []).append(d)
    import multiprocessing as mp

    r0 = R0GPU(mem_gb=a.mem_gb)
    meta = run_meta(r0, a)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "RUN_META.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    done = set()
    prog = os.path.join(a.out, "progress.jsonl")
    if os.path.exists(prog):
        done = {json.loads(l)["unit"] for l in open(prog)}
    ctx = mp.get_context("spawn")
    with ctx.Pool(a.workers, initializer=_winit, initargs=(CODE,)) as pool:
        for u in units:
            names = [unit_name(u, s) for s in speeds]
            if all(n in done for n in names):
                continue
            outs = process_unit(r0, pool, u, [s for s in speeds if unit_name(u, s) not in done], a, meta)
            with open(prog, "a") as fh:
                for n, o in outs.items():
                    fh.write(json.dumps({"unit": n, **o, "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
