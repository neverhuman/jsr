"""Matched p50/p95 stable-word lag benchmark across streaming and whole-utterance ASR systems.

Uses a deterministic 100-utterance LibriSpeech dev-clean subset from the frozen DEV panel,
ground-truth word-end timestamps from Parakeet TDT v3 token alignment, and unified streaming
trace JSONL logs from:
  - ultra-cold-wav (JSR R1 Dense/MR INT8, R0 Zipformer INT8)
  - moonshine-stream (Moonshine Tiny/Small/Medium Streaming)
  - nemo-speech --trace (Nemotron Speech Streaming EN 0.6B 160ms/560ms, Nemotron 3.5 80ms)
  - whole-utterance models (Parakeet Redux + Photon, Parakeet TDT v3 INT8)
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import re
import shutil
import wave

import numpy as np


WORD_RE = re.compile(r"[a-z0-9']+")


def normalize_words(text: str) -> list[str]:
    return WORD_RE.findall((text or "").lower())


def align_ref_to_hyp(ref: list[str], hyp: list[str]) -> list[tuple[int, int]]:
    """Return list of (ref_idx, hyp_idx) exact matches along an optimal Levenshtein alignment."""
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        ri = ref[i - 1]
        row = dp[i]
        prev = dp[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ri == hyp[j - 1] else 1
            a = prev[j] + 1
            b = row[j - 1] + 1
            c = prev[j - 1] + cost
            row[j] = a if a <= b and a <= c else (b if b <= c else c)
    matches = []
    i, j = n, m
    while i > 0 and j > 0:
        if ref[i - 1] == hyp[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
            matches.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif dp[i][j] == dp[i - 1][j - 1] + 1:
            i -= 1
            j -= 1
        elif dp[i][j] == dp[i - 1][j] + 1:
            i -= 1
        else:
            j -= 1
    matches.reverse()
    return matches


def stage_100_from_mfa(panel_dir: Path, out_dir: Path, mfa_jsonl: Path, limit: int = 100) -> Path:
    """Build deterministic 100-clip dev-clean word-end timings from locked MFA alignments."""
    index = json.loads((panel_dir / "index.json").read_text(encoding="utf-8"))
    clean = [row for row in index if row["dataset"] == "dev-clean"][:limit]
    mfa_by_uid: dict[str, list[dict]] = {}
    for line in mfa_jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        mfa_by_uid[rec["id"]] = rec["words"]
    wav_out = out_dir / "wav100"
    wav_out.mkdir(parents=True, exist_ok=True)
    records = []
    for row in clean:
        uid = row["id"]
        src_wav = panel_dir / "wav" / f"{uid}.wav"
        dst_wav = wav_out / f"{uid}.wav"
        if not dst_wav.exists():
            shutil.copy2(src_wav, dst_wav)
        mfa_words = mfa_by_uid.get(row["uid"], [])
        words_aligned = []
        for item in mfa_words:
            for wnorm in normalize_words(item["w"]):
                words_aligned.append({"word": wnorm, "end_s": float(item["end_s"])})
        ref_words = normalize_words(row["ref"])
        hyp_words = [w["word"] for w in words_aligned]
        matches = align_ref_to_hyp(ref_words, hyp_words)
        ref_ends = [None] * len(ref_words)
        for ri, hi in matches:
            ref_ends[ri] = words_aligned[hi]["end_s"]
        records.append({
            "id": uid,
            "uid": row["uid"],
            "duration_s": round(float(row["duration_s"]), 4),
            "ref_words": ref_words,
            "ref_word_end_s": ref_ends,
        })
    out_path = out_dir / "timestamps_100_mfa.json"
    out_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    return out_path


def stage_100_and_align(panel_dir: Path, out_dir: Path, parakeet_dir: Path, limit: int = 100) -> Path:
    import sherpa_onnx

    index = json.loads((panel_dir / "index.json").read_text(encoding="utf-8"))
    clean = [row for row in index if row["dataset"] == "dev-clean"][:limit]
    wav_out = out_dir / "wav100"
    if wav_out.exists():
        shutil.rmtree(wav_out)
    wav_out.mkdir(parents=True)

    rec = sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=str(parakeet_dir / "encoder.int8.onnx"),
        decoder=str(parakeet_dir / "decoder.int8.onnx"),
        joiner=str(parakeet_dir / "joiner.int8.onnx"),
        tokens=str(parakeet_dir / "tokens.txt"),
        num_threads=4,
        provider="cpu",
        model_type="nemo_transducer",
        decoding_method="greedy_search",
    )

    records = []
    for row in clean:
        uid = row["id"]
        src_wav = panel_dir / "wav" / f"{uid}.wav"
        dst_wav = wav_out / f"{uid}.wav"
        shutil.copy2(src_wav, dst_wav)

        with wave.open(str(src_wav), "rb") as w:
            pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
            dur_s = w.getnframes() / float(w.getframerate())
        stream = rec.create_stream()
        stream.accept_waveform(16000, pcm)
        rec.decode_stream(stream)
        tokens = list(stream.result.tokens)
        stamps = list(stream.result.timestamps)
        durs = list(getattr(stream.result, "durations", []))

        # Group SentencePiece tokens into word spans with end_s
        words_aligned = []
        cur_word = ""
        cur_end = 0.0
        for idx, tok in enumerate(tokens):
            end_t = float(stamps[idx]) + (float(durs[idx]) if idx < len(durs) else 0.08)
            starts_word = tok.startswith(" ") or tok.startswith("\u2581")
            piece = tok.lstrip(" \u2581")
            if starts_word and cur_word:
                for wnorm in normalize_words(cur_word):
                    words_aligned.append({"word": wnorm, "end_s": round(cur_end, 4)})
                cur_word = piece
                cur_end = end_t
            else:
                cur_word += piece
                cur_end = max(cur_end, end_t)
        if cur_word:
            for wnorm in normalize_words(cur_word):
                words_aligned.append({"word": wnorm, "end_s": round(cur_end, 4)})

        ref_words = normalize_words(row["ref"])
        hyp_words = [w["word"] for w in words_aligned]
        matches = align_ref_to_hyp(ref_words, hyp_words)
        ref_ends = [None] * len(ref_words)
        for ri, hi in matches:
            ref_ends[ri] = words_aligned[hi]["end_s"]
        records.append({
            "id": uid,
            "duration_s": round(dur_s, 4),
            "ref_words": ref_words,
            "ref_word_end_s": ref_ends,
        })

    out_path = out_dir / "timestamps_100.json"
    out_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    return out_path


def load_trace(trace_path: Path) -> dict[str, list[dict]]:
    by_id: dict[str, list[dict]] = defaultdict(list)
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        ev = json.loads(line)
        if "id" in ev:
            uid = ev["id"]
            audio_s = float(ev["audio_s"])
            cum_ms = float(ev["cum_wall_ms"])
            text = ev.get("text", "")
            is_final = bool(ev.get("final", False))
        else:
            # nemo-speech trace format: {"audio_s": ..., "elapsed_ms": ..., "input": "u00000.wav", "text": "..."}
            uid = Path(ev["input"]).stem
            audio_s = float(ev["audio_s"])
            cum_ms = float(ev["elapsed_ms"])
            text = ev.get("text", "")
            is_final = False
        by_id[uid].append({
            "audio_s": audio_s,
            "cum_wall_ms": cum_ms,
            "words": normalize_words(text),
            "final": is_final,
        })
    for uid in by_id:
        by_id[uid].sort(key=lambda e: (e["audio_s"], e["cum_wall_ms"]))
        # nemo-speech emits a synthetic empty-text marker right before its final flush;
        # filter out empty-text events once words have already been emitted so they don't
        # artificially reset stability to EOF.
        cleaned = []
        seen_nonempty = False
        for ev in by_id[uid]:
            if ev["words"]:
                seen_nonempty = True
                cleaned.append(ev)
            elif not seen_nonempty:
                cleaned.append(ev)
        by_id[uid] = cleaned
    return by_id


def paced_emit_times_ms(events: list[dict]) -> list[float]:
    """Simulate real-time paced audio delivery on 1 CPU core.

    Packet k finishes arriving at audio_s[k]*1000 ms and takes delta_wall_ms = cum[k] - cum[k-1]
    to compute. Execution cannot start before max(prev_finish_ms, audio_s[k]*1000).
    """
    finish_ms = []
    prev_done = 0.0
    prev_cum = 0.0
    for ev in events:
        arrival_ms = ev["audio_s"] * 1000.0
        dt = max(0.0, ev["cum_wall_ms"] - prev_cum)
        prev_cum = ev["cum_wall_ms"]
        done = max(prev_done, arrival_ms) + dt
        prev_done = done
        finish_ms.append(done)
    return finish_ms


def summarize_lags(vals: list[float]) -> dict:
    if not vals:
        return {"count": 0, "p50_ms": None, "p90_ms": None, "p95_ms": None, "mean_ms": None}
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "count": int(len(arr)),
        "p50_ms": round(float(np.percentile(arr, 50)), 1),
        "p90_ms": round(float(np.percentile(arr, 90)), 1),
        "p95_ms": round(float(np.percentile(arr, 95)), 1),
        "mean_ms": round(float(np.mean(arr)), 1),
    }


def score_streaming_trace(timestamps_path: Path, trace_path: Path, name: str) -> dict:
    refs = json.loads(timestamps_path.read_text(encoding="utf-8"))
    traces = load_trace(trace_path)
    alg_lags = []
    e2e_lags = []
    final_flush_ms = []
    total_audio_s = 0.0
    total_wall_ms = 0.0
    matched_utts = 0

    for rec in refs:
        uid = rec["id"]
        events = traces.get(uid)
        if not events:
            continue
        matched_utts += 1
        total_audio_s += rec["duration_s"]
        total_wall_ms += events[-1]["cum_wall_ms"]
        done_ms = paced_emit_times_ms(events)
        final_flush_ms.append(max(0.0, done_ms[-1] - rec["duration_s"] * 1000.0))

        final_words = events[-1]["words"]
        matches = align_ref_to_hyp(rec["ref_words"], final_words)
        for ri, hi in matches:
            end_s = rec["ref_word_end_s"][ri]
            if end_s is None:
                continue
            target_word = final_words[hi]
            # Earliest event k such that for ALL k' >= k, word at index hi equals target_word
            stable_k = len(events) - 1
            for k in range(len(events) - 1, -1, -1):
                w = events[k]["words"]
                if hi < len(w) and w[hi] == target_word:
                    stable_k = k
                else:
                    break
            alg_lag = max(0.0, (events[stable_k]["audio_s"] - end_s) * 1000.0)
            e2e_lag = max(0.0, done_ms[stable_k] - end_s * 1000.0)
            alg_lags.append(alg_lag)
            e2e_lags.append(e2e_lag)

    cpu_s_per_s = (total_wall_ms / 1000.0) / total_audio_s if total_audio_s > 0 else math.nan
    return {
        "system": name,
        "mode": "streaming",
        "utterances": matched_utts,
        "audio_seconds": round(total_audio_s, 2),
        "wall_s_per_audio_s_1core": round(cpu_s_per_s, 4),
        "rtfx_1core": round(1.0 / cpu_s_per_s, 2) if cpu_s_per_s > 0 else None,
        "alg_stable_lag": summarize_lags(alg_lags),
        "e2e_stable_lag": summarize_lags(e2e_lags),
        "finalization_lag": summarize_lags(final_flush_ms),
    }


def score_whole_utterance(timestamps_path: Path, cpu_s_per_s: float, name: str) -> dict:
    """Whole-utterance recognizer cannot emit words until utterance completion + inference."""
    refs = json.loads(timestamps_path.read_text(encoding="utf-8"))
    alg_lags = []
    e2e_lags = []
    fin_lags = []
    for rec in refs:
        dur_ms = rec["duration_s"] * 1000.0
        infer_ms = dur_ms * cpu_s_per_s
        fin_lags.append(infer_ms)
        for end_s in rec["ref_word_end_s"]:
            if end_s is None:
                continue
            wait_ms = max(0.0, dur_ms - end_s * 1000.0)
            alg_lags.append(wait_ms)
            e2e_lags.append(wait_ms + infer_ms)
    return {
        "system": name,
        "mode": "whole_utterance",
        "utterances": len(refs),
        "wall_s_per_audio_s_1core": round(cpu_s_per_s, 4),
        "rtfx_1core": round(1.0 / cpu_s_per_s, 2) if cpu_s_per_s > 0 else None,
        "alg_stable_lag": summarize_lags(alg_lags),
        "e2e_stable_lag": summarize_lags(e2e_lags),
        "finalization_lag": summarize_lags(fin_lags),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_stage = sub.add_parser("stage", help="Stage 100 dev-clean WAVs and generate word-end timestamps")
    p_stage.add_argument("--panel-dir", type=Path, required=True)
    p_stage.add_argument("--out-dir", type=Path, required=True)
    p_stage.add_argument("--mfa-jsonl", type=Path, default=None,
                         help="Locked ultra.word-timing/v1 MFA alignment JSONL (preferred, C1/O14)")
    p_stage.add_argument("--parakeet-dir", type=Path, default=None)
    p_stage.add_argument("--limit", type=int, default=100)

    p_score = sub.add_parser("score", help="Score one streaming trace JSONL")
    p_score.add_argument("--timestamps", type=Path, required=True)
    p_score.add_argument("--trace", type=Path, required=True)
    p_score.add_argument("--name", required=True)
    p_score.add_argument("--out", type=Path, default=None)

    p_off = sub.add_parser("score-offline", help="Compute live-equivalent lag for a whole-utterance model")
    p_off.add_argument("--timestamps", type=Path, required=True)
    p_off.add_argument("--cpu-s-per-s", type=float, required=True)
    p_off.add_argument("--name", required=True)
    p_off.add_argument("--out", type=Path, default=None)

    args = parser.parse_args()
    if args.cmd == "stage":
        if args.mfa_jsonl is not None:
            out = stage_100_from_mfa(args.panel_dir, args.out_dir, args.mfa_jsonl, args.limit)
        else:
            out = stage_100_and_align(args.panel_dir, args.out_dir, args.parakeet_dir, args.limit)
        print(f"Staged {args.limit} clips and aligned word ends -> {out}")
    elif args.cmd == "score":
        res = score_streaming_trace(args.timestamps, args.trace, args.name)
        text = json.dumps(res, indent=2)
        if args.out:
            args.out.write_text(text + "\n", encoding="utf-8")
        print(text)
    elif args.cmd == "score-offline":
        res = score_whole_utterance(args.timestamps, args.cpu_s_per_s, args.name)
        text = json.dumps(res, indent=2)
        if args.out:
            args.out.write_text(text + "\n", encoding="utf-8")
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
