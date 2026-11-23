"""H47 Sentinel-0 full-DEV (5,567 utt), torture T1-T4, and noise skip-gate simulator."""
from __future__ import annotations

import argparse
import io
import json
import math
from pathlib import Path

import numpy as np

from tools.sentinel0 import (
    BIG, HOP, KF, KS, LOG2_LUT, NABS_F, NABS_S, THETA_F, THETA_S, U, V,
    MinStat, Sentinel0, golden_pcm, ilog2_q8, trace_fnv1a64,
)

SAMPLE_RATE = 16000
PREROLL_MS = 1000
IDLE_MS = 1500


def run_sentinel0_fast(pcm: np.ndarray, theta_f: int = THETA_F, theta_s: int = THETA_S,
                       nabs_f: int = NABS_F, nabs_s: int = NABS_S) -> tuple[np.ndarray, np.ndarray]:
    """Bit-exact integer batch runner using int64 hop energies + Sentinel0 decision loop."""
    n_hops = len(pcm) // HOP
    if n_hops == 0:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=np.uint8)
    x = np.asarray(pcm[:n_hops * HOP], dtype=np.int64)
    dx = np.diff(x, prepend=0)
    ea_arr = (dx * dx).reshape(n_hops, HOP).sum(axis=1).tolist()
    db = np.diff(x.reshape(n_hops * (HOP // 4), 4).sum(axis=1), prepend=0)
    eb_arr = (db * db).reshape(n_hops, HOP // 4).sum(axis=1).tolist()
    gate = Sentinel0(theta_f=theta_f, theta_s=theta_s, nabs_f=nabs_f, nabs_s=nabs_s)
    hot, path = np.zeros(n_hops, dtype=bool), np.zeros(n_hops, dtype=np.uint8)
    for idx in range(n_hops):
        hot[idx], path[idx], _, _ = gate.step_energies(ea_arr[idx], eb_arr[idx])
    return hot, path


def measure_non_speech_cpu(hot: np.ndarray, r_hops: int = PREROLL_MS // 10,
                           t_idle_hops: int = IDLE_MS // 10, warmup_hops: int = 150) -> dict:
    """Simulate Appendix E sleep-gate session on non-speech to measure false-wake CPU."""
    n = len(hot)
    if n <= warmup_hops:
        raise ValueError("Stream shorter than warmup_hops")
    active, idle, recent = False, 0, [False] * KS
    asr_hops = false_wakes = hot_at_close = 0
    for i in range(warmup_hops, n):
        h = bool(hot[i])
        recent[i % KS] = h
        if active:
            asr_hops += 1
            idle += 1
            if idle >= t_idle_hops:
                if any(recent):
                    hot_at_close, asr_hops, idle = hot_at_close + 1, asr_hops + r_hops, 0
                else:
                    active = False
        elif h:
            false_wakes, asr_hops, active, idle = false_wakes + 1, asr_hops + r_hops + 1, True, 0
    eval_hops = n - warmup_hops
    dur_s = eval_hops * 0.01
    cpu_frac = asr_hops / eval_hops
    hot_count = int(hot[warmup_hops:].sum())
    return {
        "duration_s": round(dur_s, 2), "hot_hops": hot_count,
        "hot_hop_fraction": round(hot_count / eval_hops, 6),
        "false_wakes": false_wakes, "false_wakes_per_hour": round(false_wakes * 3600.0 / dur_s, 2),
        "hot_at_close_replays": hot_at_close, "false_wake_cpu_fraction": round(cpu_frac, 6),
        "passes_cpu_le_2pct": cpu_frac <= 0.02,
    }


def synth_noise_pcm(kind: str, n: int, dbfs: float, rng: np.random.Generator) -> np.ndarray:
    """Synthesize deterministic 16 kHz PCM16 silence, white, or pink noise."""
    if kind == "silence":
        return np.zeros(n, dtype=np.int16)
    w = rng.standard_normal(n).astype(np.float64)
    if kind == "pink":
        spec = np.fft.rfft(w)
        spec[0] = 0.0
        spec[1:] /= np.sqrt(np.arange(1, len(spec), dtype=np.float64))
        w = np.fft.irfft(spec, n=n)
    elif kind != "white":
        raise ValueError(f"Unsupported noise kind: {kind}")
    w /= math.sqrt(float(np.mean(w * w))) + 1e-12
    return np.rint(w * (10.0 ** (dbfs / 20.0)) * 32768.0).clip(-32768, 32767).astype(np.int16)


def _synth_speech_f64(n: int, rng: np.random.Generator, dbfs: float, fric_ms: int = 0) -> np.ndarray:
    f_n = min(n, (fric_ms * SAMPLE_RATE) // 1000)
    v_n = max(0, n - f_n)
    parts: list[np.ndarray] = []
    if f_n > 0:
        spec = np.fft.rfft(rng.standard_normal(f_n))
        freqs = np.fft.rfftfreq(f_n, 1.0 / SAMPLE_RATE)
        spec[(freqs < 4000.0) | (freqs > 7500.0)] = 0.0
        fric = np.fft.irfft(spec, n=f_n)
        fric /= math.sqrt(float(np.mean(fric * fric))) + 1e-12
        parts.append(fric * (10.0 ** ((dbfs - 14.0) / 20.0)))
    if v_n > 0:
        t = np.arange(v_n, dtype=np.float64) / SAMPLE_RATE
        f0 = float(rng.uniform(125.0, 175.0))
        sig = sum(np.sin(2.0 * math.pi * f0 * k * t) / k for k in range(1, 20))
        freqs = np.fft.rfftfreq(v_n, 1.0 / SAMPLE_RATE)
        gain = sum(np.exp(-0.5 * ((freqs - fc) / bw) ** 2)
                   for fc, bw in ((550.0, 140.0), (1550.0, 220.0), (2600.0, 280.0)))
        voiced = np.fft.irfft(np.fft.rfft(sig) * (gain + 0.05), n=v_n)
        voiced *= (0.6 + 0.4 * np.sin(2.0 * math.pi * 4.0 * t)) * np.minimum(1.0, t / 0.04)
        voiced /= math.sqrt(float(np.mean(voiced * voiced))) + 1e-12
        parts.append(voiced * (10.0 ** (dbfs / 20.0)))
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float64)


def _latency_summary(latencies: list[int], total: int) -> dict:
    woken_1s = sum(1 for v in latencies if v <= PREROLL_MS)
    arr = np.array(latencies, dtype=np.float64) if latencies else np.array([], dtype=np.float64)
    pct = (lambda q: round(float(np.percentile(arr, q)), 1)) if len(arr) else (lambda _: None)
    return {
        "n_utterances": total, "woken_any": len(latencies), "woken_before_1s_preroll": woken_1s,
        "wake_rate_before_1s_pct": round(100.0 * woken_1s / max(1, total), 4),
        "latency_ms": {"p50": pct(50), "p90": pct(90), "p99": pct(99),
                       "max": round(float(arr.max()), 1) if len(arr) else None},
    }


def eval_torture_t1_t4(seed: int = 47, trials: int = 8) -> dict:
    """Evaluate Sentinel0 on synthetic torture sets T1-T4 with settled pre-onset floors."""
    rng = np.random.default_rng(seed)
    pre_n, sp_n = 4 * SAMPLE_RATE, int(1.8 * SAMPLE_RATE)
    onset_hop = pre_n // HOP
    specs = (
        ("T1_fricative_onset", "white", -35.0, -20.0, 200, False, False),
        ("T2_quiet_speech_-35dBFS", "pink", -55.0, -35.0, 0, False, False),
        ("T3_post_burst_noise", "pink", -40.0, -22.0, 0, True, False),
        ("T4_low_snr_floor_transition", "pink", -40.0, -15.0, 0, False, True),
    )
    out: dict[str, dict] = {}
    for name, bg_kind, bg_db, sp_db, fric_ms, add_burst, step_floor in specs:
        lats: list[int] = []
        for _ in range(trials):
            bg = synth_noise_pcm(bg_kind, pre_n + sp_n, bg_db, rng).astype(np.float64) / 32768.0
            if step_floor:
                step_at = pre_n - 2 * SAMPLE_RATE
                bg[step_at:] = synth_noise_pcm(bg_kind, sp_n + 2 * SAMPLE_RATE, -20.0, rng) / 32768.0
            if add_burst:
                b_n, b_s = (150 * SAMPLE_RATE) // 1000, pre_n - (400 * SAMPLE_RATE) // 1000
                bg[b_s:b_s + b_n] += synth_noise_pcm("white", b_n, -10.0, rng) / 32768.0
            bg[pre_n:] += _synth_speech_f64(sp_n, rng, sp_db, fric_ms=fric_ms)
            hot, _ = run_sentinel0_fast(np.rint(bg * 32768.0).clip(-32768, 32767).astype(np.int16))
            hits = np.flatnonzero(hot[onset_hop:])
            if len(hits):
                lats.append(int(hits[0]) * 10)
        out[name] = _latency_summary(lats, trials)
    return out


def eval_pure_noise(duration_s: float = 600.0, seed: int = 47) -> dict:
    """Measure non-speech false-wake CPU fraction across silence, pink, and white noise."""
    rng = np.random.default_rng(seed)
    n = int(round((duration_s + 1.5) * SAMPLE_RATE))
    cases = (("silence", "silence", -120.0), ("pink_-20dBFS", "pink", -20.0),
             ("pink_-30dBFS", "pink", -30.0), ("white_-20dBFS", "white", -20.0),
             ("white_-40dBFS", "white", -40.0))
    return {label: measure_non_speech_cpu(run_sentinel0_fast(synth_noise_pcm(kind, n, db, rng))[0])
            for label, kind, db in cases}


def _energy_onset_hop(pcm: np.ndarray) -> int:
    n_hops = len(pcm) // HOP
    if n_hops == 0:
        return 0
    rms = np.sqrt(np.mean(pcm[:n_hops * HOP].astype(np.float64).reshape(n_hops, HOP) ** 2, axis=1))
    thr = max(float(np.median(rms[:min(15, n_hops)])) * 3.16227766, 32768.0 * (10.0 ** (-48.0 / 20.0)))
    hits = np.flatnonzero(rms >= thr)
    return int(hits[0]) if len(hits) else int(np.argmax(rms))


def _load_mfa(path: Path) -> dict[str, float]:
    if not path.is_file():
        return {}
    rows = (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line)
    return {str(r["id"]): float(r["words"][0]["start_s"]) for r in rows if r.get("words")}


def _iter_dev_split(split: str, cache_dir: Path, parquet_dir: Path, limit: int | None):
    npz_path = cache_dir / f"{split}.npz"
    if npz_path.is_file():
        with np.load(npz_path, allow_pickle=False) as d:
            pcm, off, uids = d["pcm"], d["offsets"], d["uids"]
            for i in range(len(uids) if limit is None else min(len(uids), limit)):
                yield str(uids[i]), pcm[int(off[i]):int(off[i + 1])]
        return
    import pyarrow.parquet as pq
    import soundfile as sf
    sub = "validation.clean" if split == "dev-clean" else "validation.other"
    tbl = pq.read_table(parquet_dir / sub / "0000.parquet", columns=["id", "audio"])
    for i in range(tbl.num_rows if limit is None else min(tbl.num_rows, limit)):
        audio, _ = sf.read(io.BytesIO(tbl["audio"][i].as_py()["bytes"]), dtype="int16")
        yield str(tbl["id"][i].as_py()), np.asarray(audio, dtype=np.int16)


def eval_librispeech_dev(cache_dir: Path, parquet_dir: Path, mfa_dir: Path,
                         limit_per_split: int | None = None) -> dict:
    """Run Sentinel0 across all 5,567 LibriSpeech full-DEV utterances."""
    prefix_hops, prefix_n = 200, 200 * HOP
    splits_out: dict[str, dict] = {}
    all_e, all_m, all_n, all_nm = [], [], 0, 0
    for split in ("dev-clean", "dev-other"):
        mfa, e_lats, m_lats, n_utt, n_mfa = _load_mfa(mfa_dir / f"{split}.jsonl"), [], [], 0, 0
        for uid, utt in _iter_dev_split(split, cache_dir, parquet_dir, limit_per_split):
            n_utt += 1
            e_hop, mfa_s = _energy_onset_hop(utt), mfa.get(uid)
            lead = utt[:max(HOP, min(int((mfa_s or 0.2) * SAMPLE_RATE), 3200, len(utt)))]
            prefix = np.tile(lead, (prefix_n + len(lead) - 1) // len(lead))[:prefix_n]
            hot, _ = run_sentinel0_fast(np.concatenate((prefix, utt)))
            eh = np.flatnonzero(hot[prefix_hops + e_hop:])
            if len(eh):
                e_lats.append(int(eh[0]) * 10)
            if mfa_s is not None:
                n_mfa += 1
                m_hop = min(max(0, int(round(mfa_s * 100.0))), max(0, len(utt) // HOP - 1))
                mh = np.flatnonzero(hot[prefix_hops + m_hop:])
                if len(mh):
                    m_lats.append(int(mh[0]) * 10)
        splits_out[split] = {"energy_onset": _latency_summary(e_lats, n_utt),
                             "mfa_first_word_onset": _latency_summary(m_lats, n_mfa)}
        all_e.extend(e_lats); all_m.extend(m_lats); all_n += n_utt; all_nm += n_mfa
    pooled_e, pooled_m = _latency_summary(all_e, all_n), _latency_summary(all_m, all_nm)
    return {
        "protocol": "2.0s tiled leading room-tone prefix per utterance, strict post-onset wake check",
        "preroll_ring_ms": PREROLL_MS, "splits": splits_out,
        "pooled_energy_onset": pooled_e, "pooled_mfa_first_word_onset": pooled_m,
        "passes_wake_rate_ge_99_8pct": pooled_e["wake_rate_before_1s_pct"] >= 99.8,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", type=Path, default=Path("/mnt/xbabe3-external/jsr/cache/librispeech_pcm"))
    ap.add_argument("--parquet-dir", type=Path, default=Path("/mnt/xbabe3-external/jsr/corpora/librispeech/hf_parquet/all"))
    ap.add_argument("--mfa-dir", type=Path, default=Path("/mnt/xbabe3-external/jsr/corpora/librispeech/word_timing"))
    ap.add_argument("--noise-duration-s", type=float, default=600.0)
    ap.add_argument("--seed", type=int, default=47)
    ap.add_argument("--limit-per-split", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    pcm_h, tr_h, hot_hops = trace_fnv1a64(golden_pcm())
    dev = eval_librispeech_dev(args.cache_dir, args.parquet_dir, args.mfa_dir, args.limit_per_split)
    noise = eval_pure_noise(duration_s=args.noise_duration_s, seed=args.seed)
    golden_ok = pcm_h == 0x6F5D0B63084F8FFD and tr_h == 0xC7B5B4E8DA91B73F
    summary = {
        "schema": "ultra.h47.skip_gate_sim.v1",
        "gate_constants": {"hop": HOP, "v": V, "u": U, "kf": KF, "ks": KS, "theta_f": THETA_F,
                           "theta_s": THETA_S, "nabs_f": NABS_F, "nabs_s": NABS_S,
                           "preroll_ms": PREROLL_MS, "idle_ms": IDLE_MS},
        "golden_vector": {"pcm_fnv1a64": f"0x{pcm_h:016x}", "trace_fnv1a64": f"0x{tr_h:016x}",
                          "hot_hop_count": len(hot_hops), "matches_spec": golden_ok},
        "librispeech_full_dev": dev,
        "synthetic_torture_t1_t4": eval_torture_t1_t4(seed=args.seed),
        "pure_noise_false_wakes": noise,
        "gates": {"dev_wake_ge_99_8pct": dev["passes_wake_rate_ge_99_8pct"],
                  "all_noise_cpu_le_2pct": all(v["passes_cpu_le_2pct"] for v in noise.values()),
                  "golden_parity": golden_ok},
    }
    text = json.dumps(summary, indent=2) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
