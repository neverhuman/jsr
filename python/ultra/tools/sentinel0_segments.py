"""Streaming H47 Sentinel-0 wake-segment producer for long 16 kHz mono PCM16 WAV streams.

Streams WAV parts in bounded RAM (<= 16 MB) through bit-exact Sentinel0 + Appendix E session
(R=1.0 s pre-roll, T_idle=1.5 s hangover), or consumes a precomputed hot.u8 hop stream,
emitting h47-oracle-decode-compatible non-overlapping [s0, e) JSONL + SHA-256 summary.
"""
from __future__ import annotations

import argparse, hashlib, json, struct
from pathlib import Path
import numpy as np
from tools.sentinel0 import HOP, KF, KS, NABS_F, NABS_S, THETA_F, THETA_S, U, V, Sentinel0

SAMPLE_RATE, DEFAULT_PREROLL_MS, DEFAULT_IDLE_MS, DEFAULT_CHUNK_FRAMES = 16000, 1000, 1500, 400_000


def _read_exact(fh, n: int, hasher) -> bytes:
    buf = fh.read(n)
    if len(buf) != n:
        raise ValueError(f"Unexpected EOF: expected {n} bytes, got {len(buf)}")
    hasher.update(buf)
    return buf


def iter_wav_pcm16_chunks(path: Path, chunk_frames: int = DEFAULT_CHUNK_FRAMES):
    """Yield int16 PCM chunks from a 16 kHz mono PCM16 WAV while hashing WAV & PCM bytes."""
    wav_h, pcm_h = hashlib.sha256(), hashlib.sha256()
    info: dict[str, object] = {"path": str(path), "filename": path.name}
    with path.open("rb") as fh:
        riff = _read_exact(fh, 12, wav_h)
        if riff[:4] != b"RIFF" or riff[8:12] != b"WAVE":
            raise ValueError(f"{path}: not a RIFF/WAVE file")
        fmt_ok = data_ok = False
        while hdr := fh.read(8):
            if len(hdr) != 8:
                raise ValueError(f"{path}: truncated RIFF chunk header")
            wav_h.update(hdr)
            cid, sz = struct.unpack("<4sI", hdr)
            if cid == b"fmt ":
                body = _read_exact(fh, sz + (sz & 1), wav_h)
                afmt, ch, sr, _, _, bps = struct.unpack("<HHIIHH", body[:16]) if len(body) >= 16 else (0,) * 6
                if (afmt, ch, sr, bps) != (1, 1, SAMPLE_RATE, 16):
                    raise ValueError(f"{path}: expected 16 kHz mono PCM16, got {(afmt, ch, sr, bps)}")
                fmt_ok = True
            elif cid == b"data":
                if not fmt_ok or (sz & 1):
                    raise ValueError(f"{path}: invalid data chunk (fmt_ok={fmt_ok}, sz={sz})")
                info["samples"], info["duration_s"] = sz // 2, round((sz // 2) / SAMPLE_RATE, 6)
                rem, step = sz, max(HOP, chunk_frames) * 2
                while rem > 0:
                    raw = _read_exact(fh, min(step, rem), wav_h)
                    pcm_h.update(raw)
                    rem -= len(raw)
                    yield np.frombuffer(raw, dtype="<i2")
                data_ok = True
            else:
                rem = sz + (sz & 1)
                while rem > 0:
                    take = min(1 << 20, rem)
                    _read_exact(fh, take, wav_h)
                    rem -= take
        if not data_ok:
            raise ValueError(f"{path}: missing data chunk")
    info["wav_sha256"], info["pcm_sha256"] = wav_h.hexdigest(), pcm_h.hexdigest()
    return info


class WakeSegmenter:
    """Appendix E sleep-gate session converting Sentinel-0 hot hops into non-overlapping [s0, e)."""

    def __init__(self, preroll_samples: int = SAMPLE_RATE, idle_hops: int = 150) -> None:
        if preroll_samples < 0 or idle_hops < 1:
            raise ValueError(f"Invalid session parameters: preroll={preroll_samples}, idle={idle_hops}")
        self.preroll_samples, self.idle_hops = preroll_samples, idle_hops
        self.active, self.idle, self.cur_s0, self.cur_hot_hops = False, 0, 0, 0
        self.cur_wake_path: int | None = None
        self.cur_wake_hops: list[int] = []
        self.segments: list[dict] = []
        self.raw_wakes = self.total_hot_hops = 0

    def push_hop(self, hop_idx: int, hot: bool, path: int | None = None) -> None:
        now = (hop_idx + 1) * HOP
        if hot:
            self.total_hot_hops += 1
        if self.active:
            if hot:
                self.idle, self.cur_hot_hops = 0, self.cur_hot_hops + 1
            else:
                self.idle += 1
                if self.idle >= self.idle_hops:
                    self._close(now)
        elif hot:
            self.raw_wakes += 1
            s0 = max(0, now - self.preroll_samples)
            if self.segments and s0 < self.segments[-1]["e"]:
                prev = self.segments.pop()
                self.cur_s0, self.cur_wake_path = prev["s0"], prev["wake_path"]
                self.cur_wake_hops, self.cur_hot_hops = list(prev["wake_hops"]) + [hop_idx], prev["hot_hops"] + 1
            else:
                self.cur_s0, self.cur_wake_hops, self.cur_wake_path, self.cur_hot_hops = s0, [hop_idx], path, 1
            self.active, self.idle = True, 0

    def _close(self, end_sample: int) -> None:
        self.segments.append({
            "id": len(self.segments), "s0": self.cur_s0, "e": end_sample,
            "s0_s": round(self.cur_s0 / SAMPLE_RATE, 6), "e_s": round(end_sample / SAMPLE_RATE, 6),
            "wake_hop": self.cur_wake_hops[0], "wake_hops": self.cur_wake_hops,
            "wake_path": self.cur_wake_path, "hot_hops": self.cur_hot_hops,
        })
        self.active, self.idle = False, 0

    def finish(self, total_samples: int) -> list[dict]:
        if self.active:
            self._close(total_samples)
        for idx, seg in enumerate(self.segments):
            seg["id"] = idx
        return self.segments


def _emit_summary(
    seg_sm: WakeSegmenter, total_samples: int, total_hops: int, out_jsonl: Path,
    inputs_meta: list[dict], *, label: str, gate_cfg: dict, extra: dict | None = None,
) -> dict:
    segs = seg_sm.finish(total_samples)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_h = hashlib.sha256()
    with out_jsonl.open("wb") as fh:
        for seg in segs:
            line = (json.dumps(seg, separators=(",", ":")) + "\n").encode("utf-8")
            fh.write(line); out_h.update(line)
    durs = np.array([(s["e"] - s["s0"]) / SAMPLE_RATE for s in segs], dtype=np.float64)
    awake_samples, dur_s = int(sum(s["e"] - s["s0"] for s in segs)), total_samples / SAMPLE_RATE
    duty = awake_samples / total_samples if total_samples else 0.0
    out = {
        "schema": "ultra.h47.sentinel0_segments.v1", "label": label, "gate_constants": gate_cfg,
        "inputs": inputs_meta, "output": {"segments_jsonl": str(out_jsonl), "segments_sha256": out_h.hexdigest()},
        "stream": {"sample_rate": SAMPLE_RATE, "total_samples": total_samples,
                   "total_hops": total_hops, "duration_s": round(dur_s, 6)},
        "metrics": {
            "segment_count": len(segs), "raw_wake_count": seg_sm.raw_wakes,
            "merged_wake_count": seg_sm.raw_wakes - len(segs), "hot_hops": seg_sm.total_hot_hops,
            "hot_hop_fraction": round(seg_sm.total_hot_hops / max(1, total_hops), 6),
            "awake_samples": awake_samples, "awake_s": round(awake_samples / SAMPLE_RATE, 6),
            "awake_duty_fraction": round(duty, 6), "awake_duty_pct": round(100.0 * duty, 4),
            "wakes_per_hour": round(len(segs) / (dur_s / 3600.0), 3) if dur_s > 0 else 0.0,
            "segment_duration_s": {k: round(float(v), 4) if len(durs) else None for k, v in (
                ("min", durs.min() if len(durs) else 0), ("mean", durs.mean() if len(durs) else 0),
                ("p50", np.percentile(durs, 50) if len(durs) else 0),
                ("p95", np.percentile(durs, 95) if len(durs) else 0),
                ("max", durs.max() if len(durs) else 0))},
        },
    }
    return {**out, **(extra or {})}


def stream_wavs_to_segments(
    wav_paths: list[Path], out_jsonl: Path, *, preroll_ms: int = DEFAULT_PREROLL_MS,
    idle_ms: int = DEFAULT_IDLE_MS, chunk_frames: int = DEFAULT_CHUNK_FRAMES,
    theta_f: int = THETA_F, theta_s: int = THETA_S, nabs_f: int = NABS_F, nabs_s: int = NABS_S,
    label: str | None = None,
) -> dict:
    """Stream concatenated WAV files through Sentinel0 + WakeSegmenter and write segments JSONL."""
    if not wav_paths:
        raise ValueError("At least one input WAV path is required")
    gate = Sentinel0(theta_f=theta_f, theta_s=theta_s, nabs_f=nabs_f, nabs_s=nabs_s)
    seg_sm = WakeSegmenter(preroll_samples=(preroll_ms * SAMPLE_RATE) // 1000, idle_hops=idle_ms // 10)
    prev_x, prev_b, rem, hop_idx, total_samples, inputs_meta = 0, 0, np.zeros(0, dtype=np.int64), 0, 0, []
    step, push = gate.step_energies, seg_sm.push_hop
    for wav_path in wav_paths:
        gen = iter_wav_pcm16_chunks(wav_path, chunk_frames=chunk_frames)
        while True:
            try:
                pcm_chunk = next(gen)
            except StopIteration as stop:
                inputs_meta.append(stop.value); break
            total_samples += len(pcm_chunk)
            x = np.concatenate((rem, pcm_chunk.astype(np.int64))) if len(rem) else pcm_chunk.astype(np.int64)
            nh = len(x) // HOP
            if nh == 0:
                rem = x; continue
            blk, rem = x[:nh * HOP], x[nh * HOP:]
            ea_arr = (np.diff(blk, prepend=prev_x) ** 2).reshape(nh, HOP).sum(axis=1).tolist()
            boxes = blk.reshape(nh * (HOP // 4), 4).sum(axis=1)
            eb_arr = (np.diff(boxes, prepend=prev_b) ** 2).reshape(nh, HOP // 4).sum(axis=1).tolist()
            prev_x, prev_b = int(blk[-1]), int(boxes[-1])
            for i in range(nh):
                hot, path, _, _ = step(ea_arr[i], eb_arr[i])
                push(hop_idx, hot, path); hop_idx += 1
    cfg = {"hop": HOP, "v": V, "u": U, "kf": KF, "ks": KS, "theta_f": theta_f, "theta_s": theta_s,
           "nabs_f": nabs_f, "nabs_s": nabs_s, "preroll_ms": preroll_ms, "idle_ms": idle_ms}
    return _emit_summary(seg_sm, total_samples, hop_idx, out_jsonl, inputs_meta, label=label or out_jsonl.stem, gate_cfg=cfg)


def hot_u8_to_segments(
    hot_u8_path: Path, total_samples: int, out_jsonl: Path, *, inputs_meta: list[dict] | None = None,
    preroll_ms: int = DEFAULT_PREROLL_MS, idle_ms: int = DEFAULT_IDLE_MS, label: str | None = None,
) -> dict:
    """Derive exact [s0, e) Appendix E segments from a precomputed Sentinel-0 hot.u8 stream."""
    seg_sm = WakeSegmenter(preroll_samples=(preroll_ms * SAMPLE_RATE) // 1000, idle_hops=idle_ms // 10)
    push, hot_h, hop_idx = seg_sm.push_hop, hashlib.sha256(), 0
    with hot_u8_path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            hot_h.update(chunk)
            for b in chunk:
                push(hop_idx, b != 0, None); hop_idx += 1
    if hop_idx != total_samples // HOP:
        raise ValueError(f"hot.u8 hop count {hop_idx} != total_samples//HOP ({total_samples // HOP})")
    cfg = {"hop": HOP, "v": V, "u": U, "kf": KF, "ks": KS, "theta_f": THETA_F, "theta_s": THETA_S,
           "nabs_f": NABS_F, "nabs_s": NABS_S, "preroll_ms": preroll_ms, "idle_ms": idle_ms}
    return _emit_summary(seg_sm, total_samples, hop_idx, out_jsonl, inputs_meta or [], label=label or out_jsonl.stem,
                         gate_cfg=cfg, extra={"hot_u8": {"path": str(hot_u8_path), "sha256": hot_h.hexdigest()}})


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wavs", nargs="+", default=None); ap.add_argument("--stream-dir", type=Path, default=None)
    ap.add_argument("--hot-u8", type=Path, default=None); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--summary-out", type=Path, default=None); ap.add_argument("--label", type=str, default=None)
    ap.add_argument("--preroll-ms", type=int, default=DEFAULT_PREROLL_MS); ap.add_argument("--idle-ms", type=int, default=DEFAULT_IDLE_MS)
    ap.add_argument("--chunk-frames", type=int, default=DEFAULT_CHUNK_FRAMES)
    args = ap.parse_args(argv)
    if args.hot_u8 is not None:
        m = json.loads((args.stream_dir / "MANIFEST.json").read_text(encoding="utf-8"))
        inputs = [{"path": str(args.stream_dir / p["file"]), "filename": p["file"], "samples": p["end"] - p["start"],
                   "duration_s": round(p["seconds"], 6), "wav_sha256": p["wav_sha256"], "pcm_sha256": p["pcm_sha256"]} for p in m["parts"]]
        summary = hot_u8_to_segments(args.hot_u8, int(m.get("total_samples") or m["parts"][-1]["end"]), args.out,
                                     inputs_meta=inputs, preroll_ms=args.preroll_ms, idle_ms=args.idle_ms, label=args.label)
    else:
        wavs = ([args.stream_dir / p["file"] for p in json.loads((args.stream_dir / "MANIFEST.json").read_text(encoding="utf-8"))["parts"]]
                if args.stream_dir is not None else [Path(p) for item in (args.wavs or []) for p in item.split(",") if p])
        summary = stream_wavs_to_segments(wavs, args.out, preroll_ms=args.preroll_ms, idle_ms=args.idle_ms,
                                          chunk_frames=args.chunk_frames, label=args.label)
    text = json.dumps(summary, indent=2) + "\n"
    if args.summary_out is not None:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True); args.summary_out.write_text(text, encoding="utf-8")
    print(text, end=""); return 0


if __name__ == "__main__":
    raise SystemExit(main())
