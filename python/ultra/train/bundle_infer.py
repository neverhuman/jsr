"""Reference consumer of an `ultra.r1-export/v1` bundle: numpy + onnxruntime only.

It reads nothing but the bundle directory (manifest, ONNX files, tokens, mel filterbank) and is
the Python twin the Rust ORT backend is checked against. Same surface as the SP-A harness uses:
create_stream / accept_waveform / input_finished / is_ready / decode_streams / get_result.

  python -m train.bundle_infer --bundle <dir> [--precision int8] --split dev-clean --limit 400 \
      [--compare-checkpoint <ckpt.pt>]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

import numpy as np

SCHEMAS = ("ultra.r1-export/v1", "ultra.r1-export/v1.1")


class BundleError(Exception):
    """The bundle is incomplete, tampered with, or of another schema."""


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


class BundleStream:
    def __init__(self, rec: BundleRecognizer):
        self.samples = np.zeros(0, dtype=np.float32)
        self.sample_base = 0
        self.n_feat = 0
        self.feats = np.zeros((0, rec.n_mels), dtype=np.float32)
        self.feat_base = 0
        self.t_next = 0
        self.finished = False
        self.state = {s["name"]: np.zeros([1 if d == "B" else d for d in s["shape"]],
                                          dtype=np.float32 if s["dtype"] == "float32" else np.int64)
                      for s in rec.states}
        self.ctx = list(rec.ctx_init)
        self.tokens: list[int] = []
        self.forced = 0
        self._rec = rec

    def accept_waveform(self, sample_rate: int, wav) -> None:
        if sample_rate != self._rec.sr:
            raise BundleError(f"sample rate {sample_rate} != {self._rec.sr}")
        self.samples = np.concatenate([self.samples, np.asarray(wav, dtype=np.float32)])
        self._rec._featurize(self)

    def input_finished(self) -> None:
        self.finished = True


class BundleRecognizer:
    def __init__(self, bundle: str, precision: str = "fp32", threads: int = 1, verify: bool = True,
                 joiner_precision: str | None = None, override: dict | None = None):
        """`override` maps a manifest file key (e.g. "encoder.int8") to another ONNX path; used by
        quantisation sweeps. Overridden files are not hash-checked against the manifest."""
        import onnxruntime as ort

        with open(os.path.join(bundle, "manifest.json")) as fh:
            m = json.load(fh)
        if m.get("schema") not in SCHEMAS:
            raise BundleError(f"schema {m.get('schema')} not in {SCHEMAS}")
        if verify:
            for key, f in m["files"].items():
                if _sha256(os.path.join(bundle, f["path"])) != f["sha256"]:
                    raise BundleError(f"sha256 mismatch for {key} ({f['path']})")
        self.m = m
        fe, st, de = m["features"], m["streaming"], m["decoder"]
        self.sr, self.win, self.hop, self.n_fft = (fe["sample_rate"], fe["frame_length"],
                                                   fe["frame_shift"], fe["n_fft"])
        self.n_mels, self.floor = fe["n_mels"], fe["log_floor"]
        self.fbank = np.fromfile(os.path.join(bundle, fe["mel"]["file"]), dtype="<f4").reshape(
            self.n_fft // 2 + 1, self.n_mels)
        self.window = np.hanning(self.win).astype(np.float32)
        self.mean = np.asarray(fe["cmvn"]["mean"], dtype=np.float32)
        self.istd = np.asarray(fe["cmvn"]["istd"], dtype=np.float32)
        self.chunk = st["chunk_frames"]
        self.states = st["states"]
        self.blank, self.max_symbols = de["blank_id"], de["max_symbols"]
        self.ctx_init = de["ctx_init"]
        so = ort.SessionOptions()
        so.intra_op_num_threads = max(1, threads)
        so.inter_op_num_threads = 1
        f = m["files"]
        enc_key = "encoder.int8" if precision == "int8" else "encoder.fp32"
        jp = joiner_precision or precision
        joi_key = "joiner.int8" if jp == "int8" else "joiner.fp32"
        override = override or {}

        def mk(k):
            path = override.get(k) or os.path.join(bundle, f[k]["path"])
            return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])

        self.enc, self.dec, self.joi = mk(enc_key), mk("decoder"), mk(joi_key)
        # A bundle may declare that its int8 graphs must run one stream (row) per call, e.g.
        # dynamic activation quantisation, whose scale is computed over the whole input tensor.
        i8 = m.get("int8") or {}
        self.max_batch_enc = i8.get("max_batch") if precision == "int8" else None
        self.max_batch_joi = i8.get("max_batch") if jp == "int8" else None
        self.pieces = {}
        with open(os.path.join(bundle, f["tokens"]["path"])) as fh:
            for line in fh:
                piece, idx = line.rstrip("\n").rsplit(" ", 1)
                self.pieces[int(idx)] = piece
        self.forced_advances = 0

    # ---- harness surface ---------------------------------------------------------------------
    def create_stream(self) -> BundleStream:
        return BundleStream(self)

    def _bounds(self, s: BundleStream):
        t0 = s.t_next
        if s.finished:
            total = ((s.n_feat + 1) // 2 + 1) // 2
            return (t0, min(t0 + self.chunk, total)) if t0 < total else None
        t1 = t0 + self.chunk
        return (t0, t1) if 4 * t1 - 4 <= s.n_feat - 1 else None

    def is_ready(self, s: BundleStream) -> bool:
        return self._bounds(s) is not None

    def decode_stream(self, s: BundleStream) -> None:
        self.decode_streams([s])

    def decode_streams(self, streams: list[BundleStream]) -> None:
        groups: dict[int, list] = {}
        for s in streams:
            b = self._bounds(s)
            if b is not None:
                groups.setdefault(b[1] - b[0], []).append((s, b[0]))
        for n, items in groups.items():
            step = self.max_batch_enc or len(items)
            for i in range(0, len(items), step):
                self._run(items[i:i + step], n)

    def get_result(self, s: BundleStream) -> str:
        return "".join(self.pieces[i] for i in s.tokens).replace("▁", " ").strip()

    # ---- internals ---------------------------------------------------------------------------
    def _featurize(self, s: BundleStream) -> None:
        avail = s.sample_base + len(s.samples)
        n_new = 0 if avail < self.win else 1 + (avail - self.win) // self.hop - s.n_feat
        if n_new <= 0:
            return
        start = s.n_feat * self.hop - s.sample_base
        idx = start + np.arange(n_new)[:, None] * self.hop + np.arange(self.win)[None]
        fr = s.samples[idx]
        fr = fr - fr.mean(axis=1, keepdims=True)
        spec = np.abs(np.fft.rfft(fr * self.window, n=self.n_fft)).astype(np.float32) ** 2
        logmel = np.log(spec @ self.fbank + np.float32(self.floor))
        s.feats = np.concatenate([s.feats, ((logmel - self.mean) * self.istd).astype(np.float32)])
        s.n_feat += n_new
        drop = s.n_feat * self.hop - s.sample_base
        s.samples = s.samples[drop:]
        s.sample_base += drop

    def _slice(self, s: BundleStream, t0: int, n: int) -> np.ndarray:
        lo, hi = 4 * t0 - 6, 4 * (t0 + n) - 4
        out = np.zeros((hi - lo + 1, self.n_mels), dtype=np.float32)
        a = max(lo, s.feat_base)
        out[a - lo:] = s.feats[a - s.feat_base:hi + 1 - s.feat_base]
        keep = max(0, 4 * (t0 + n) - 6)
        if keep > s.feat_base:
            s.feats = s.feats[keep - s.feat_base:]
            s.feat_base = keep
        return out

    def _run(self, items, n: int) -> None:
        streams = [s for s, _ in items]
        feed = {"feats": np.stack([self._slice(s, t0, n) for s, t0 in items]),
                "t0": np.asarray([t0 for _, t0 in items], dtype=np.int64)}
        for spec in self.states:
            nm = spec["name"]
            ax = spec["shape"].index("B")
            feed[nm] = np.concatenate([s.state[nm] for s in streams], axis=ax)
        outs = dict(zip([o.name for o in self.enc.get_outputs()], self.enc.run(None, feed),
                        strict=True))
        for spec in self.states:
            nm = spec["name"]
            ax = spec["shape"].index("B")
            parts = np.split(outs[nm + "_out"], len(streams), axis=ax)
            for s, p in zip(streams, parts, strict=True):
                s.state[nm] = p
        for s, (_, t0) in zip(streams, items, strict=True):
            s.t_next = t0 + n
        self._greedy(streams, outs["enc"])

    def _greedy(self, streams, enc: np.ndarray) -> None:
        ctx = np.asarray([s.ctx for s in streams], dtype=np.int64)
        for t in range(enc.shape[1]):
            active = np.arange(len(streams))
            hits = 0
            while active.size and hits < self.max_symbols:
                dec = self.dec.run(None, {"ctx": ctx[active]})[0]
                if self.max_batch_joi:
                    logits = np.concatenate([
                        self.joi.run(None, {"enc": enc[active[j:j + self.max_batch_joi], t],
                                            "dec": dec[j:j + self.max_batch_joi]})[0]
                        for j in range(0, active.size, self.max_batch_joi)])
                else:
                    logits = self.joi.run(None, {"enc": enc[active, t], "dec": dec})[0]
                k = logits.argmax(axis=-1)
                emit = k != self.blank
                for i, kk in zip(active[emit], k[emit], strict=True):
                    streams[i].tokens.append(int(kk))
                ctx[active[emit]] = np.stack([ctx[active[emit], 1], k[emit]], axis=1)
                active = active[emit]
                hits += 1
            for i in active:  # still emitting after max_symbols: forced advance
                streams[i].forced += 1
                self.forced_advances += 1
        for s, c in zip(streams, ctx, strict=True):
            s.ctx = c.tolist()


def batch_invariance(bundle: str, precision: str, sizes=(2, 8, 16), seed: int = 20261120,
                     override: dict | None = None, joiner_precision: str | None = None) -> dict:
    """Max |difference| between a row decoded alone and the same row inside larger batches, for
    every encoder output (including caches) and the joiner. Batch-invariant bundles give 0.0."""
    rec = BundleRecognizer(bundle, precision, threads=1, joiner_precision=joiner_precision,
                           override=override)
    st = rec.m["streaming"]
    n = st["chunk_frames"]
    rng = np.random.default_rng(seed)

    def feed(b):
        f = {"feats": rng.normal(0, 1, (b, 4 * n + 3, rec.n_mels)).astype(np.float32),
             "t0": rng.integers(0, 50, b).astype(np.int64) * n}
        for s in rec.states:
            shape = [b if d == "B" else d for d in s["shape"]]
            if s["dtype"] == "int64":
                f[s["name"]] = rng.integers(0, st["left_context_frames"] + 1, shape).astype(np.int64)
            else:
                f[s["name"]] = rng.normal(0, 1, shape).astype(np.float32)
        return f

    def axis(name):
        for s in rec.states:
            if s["name"] == name:
                return s["shape"].index("B")
        return 0

    base = feed(1)
    names_out = [o.name for o in rec.enc.get_outputs()]
    ref = rec.enc.run(None, base)
    worst = {"encoder": 0.0, "joiner": 0.0}
    for b in sizes:
        f = feed(b)
        for k, v in base.items():
            idx = [slice(None)] * f[k].ndim
            idx[axis(k)] = slice(0, 1)
            f[k][tuple(idx)] = v
        for o_ref, o_b, nm in zip(ref, rec.enc.run(None, f), names_out, strict=True):
            row = np.take(o_b, [0], axis=axis(nm.replace("_out", "")))
            worst["encoder"] = max(worst["encoder"], float(np.max(np.abs(row.astype(np.float64) - o_ref))))
    J = rec.m["streaming"]["encoder_output"]["shape"][-1]
    e1 = rng.normal(0, 1, (1, J)).astype(np.float32)
    d1 = rec.dec.run(None, {"ctx": np.array([[0, 5]], dtype=np.int64)})[0]
    l1 = rec.joi.run(None, {"enc": e1, "dec": d1})[0]
    for b in sizes:
        e = rng.normal(0, 2, (b, J)).astype(np.float32)
        d = rec.dec.run(None, {"ctx": rng.integers(0, 4096, (b, 2)).astype(np.int64)})[0]
        e[0], d[0] = e1[0], d1[0]
        lb = rec.joi.run(None, {"enc": e, "dec": d})[0][:1]
        worst["joiner"] = max(worst["joiner"], float(np.max(np.abs(lb - l1))))
    return {"kind": "batch_invariance", "bundle": bundle, "precision": precision,
            "sizes": list(sizes), "max_abs_diff": worst}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--precision", choices=["fp32", "int8"], default="fp32",
                    help="encoder precision (and joiner unless --joiner-precision)")
    ap.add_argument("--joiner-precision", choices=["fp32", "int8"], default=None)
    ap.add_argument("--split", default="dev-clean")
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--streams", type=int, default=16)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--batch-invariance", action="store_true",
                    help="only check that rows are identical alone and inside batches of 2/8/16")
    ap.add_argument("--dump", default=None,
                    help="write per-utterance JSON lines (uid, ref, hyp[, trainer_hyp]) here")
    ap.add_argument("--compare-checkpoint", default=None,
                    help="also run the trainer's batch eval of this checkpoint on the same audio")
    a = ap.parse_args()
    if a.batch_invariance:
        print(json.dumps(batch_invariance(a.bundle, a.precision)))
        return
    from train.infer import run_streaming
    from ultra.libri import load_split
    from ultra.wer import corpus_wer

    rec = BundleRecognizer(a.bundle, a.precision, a.threads, joiner_precision=a.joiner_precision)
    utts = load_split(a.split, a.limit)
    c0 = time.process_time()
    hyps = run_streaming(rec, utts, a.streams)
    cpu = time.process_time() - c0
    audio = sum(u.seconds for u in utts)
    refs = [u.text for u in utts]
    out = {"kind": "bundle_infer", "bundle": a.bundle, "precision": a.precision,
           "joiner_precision": a.joiner_precision or a.precision, "split": a.split,
           "n": len(utts), "streams": a.streams, "threads": a.threads,
           "rtf_cpu": round(cpu / audio, 4), "forced_advances": rec.forced_advances,
           **corpus_wer(refs, hyps)}
    if a.compare_checkpoint:
        import torch

        from train.asr import ASRModel, ModelConfig
        from train import checkpoint
        from train.data import Featurizer, collate
        from train.tokenizer import Tokenizer, tokenizer_path

        p = checkpoint.load(a.compare_checkpoint, restore_rng=False)
        model = ASRModel(ModelConfig.from_dict(p["model_config"]))
        model.load_state_dict(p["model"])
        model.eval()
        tok = Tokenizer(tokenizer_path(4096))
        feat = Featurizer(tok, Tokenizer(tokenizer_path(1024)))
        exs = [feat.example(u.uid, u.text, u.pcm) for u in utts]
        bh = []
        with torch.no_grad():
            for i in range(0, len(exs), 8):
                b = collate(exs[i:i + 8])
                toks, _ = model.greedy(b.feats, b.feat_lens)
                bh.extend(tok.decode(t) for t in toks)
        out["trainer_eval_wer"] = corpus_wer(refs, bh)["wer"]
        out["identical_hyps"] = sum(x == y for x, y in zip(hyps, bh, strict=True))
    else:
        bh = None
    if a.dump:
        with open(a.dump, "w") as fh:
            for i, u in enumerate(utts):
                rec = {"uid": u.uid, "ref": u.text, "hyp": hyps[i]}
                if bh is not None:
                    rec["trainer_hyp"] = bh[i]
                fh.write(json.dumps(rec) + "\n")
        out["dump"] = a.dump
    print(json.dumps(out))


if __name__ == "__main__":
    main()
