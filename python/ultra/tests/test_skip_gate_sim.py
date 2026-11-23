"""Fast deterministic unit tests for H47 Sentinel-0 integer skip-gate simulator."""
from __future__ import annotations

import numpy as np
import pytest

from tools.skip_gate_sim import (
    BIG,
    HOP,
    LOG2_LUT,
    Sentinel0,
    eval_pure_noise,
    eval_torture_t1_t4,
    golden_pcm,
    ilog2_q8,
    measure_non_speech_cpu,
    run_sentinel0_fast,
    synth_noise_pcm,
    trace_fnv1a64,
)


def test_ilog2_q8_and_lut_exact() -> None:
    assert len(LOG2_LUT) == 256
    assert ilog2_q8(1) == 0
    assert ilog2_q8(2) == 256
    assert ilog2_q8(3) == 406
    assert ilog2_q8(1 << 40) == 10240
    assert LOG2_LUT[128] == 150
    with pytest.raises(ValueError):
        ilog2_q8(0)


def test_golden_vector_bit_exact_spec_parity() -> None:
    pcm = golden_pcm(96000)
    assert pcm[:8] == [10, 8, -234, -187, -47, 57, 253, -232]
    pcm_h, tr_h, hot_hops = trace_fnv1a64(pcm)
    assert pcm_h == 0x6F5D0B63084F8FFD
    assert tr_h == 0xC7B5B4E8DA91B73F
    assert hot_hops == list(range(200, 258)) + list(range(401, 456))
    assert len(hot_hops) == 113


def test_fast_batch_matches_scalar_sentinel0_bit_for_bit() -> None:
    pcm = np.array(golden_pcm(96000), dtype=np.int16)
    hot_fast, path_fast = run_sentinel0_fast(pcm)
    gate = Sentinel0()
    hot_ref, path_ref = [], []
    for i in range(len(pcm) // HOP):
        h, p, _, floors = gate.push_hop(pcm[i * HOP:(i + 1) * HOP])
        if i < 7:
            assert h is False
            assert p == 0
            assert floors == [BIG, BIG, BIG, BIG]
        hot_ref.append(h)
        path_ref.append(p)
    assert hot_fast.tolist() == hot_ref
    assert path_fast.tolist() == path_ref


def test_digital_silence_never_wakes() -> None:
    silence = np.zeros(HOP * 2000, dtype=np.int16)
    hot, path = run_sentinel0_fast(silence)
    assert not np.any(hot)
    assert not np.any(path)
    stats = measure_non_speech_cpu(hot)
    assert stats["false_wakes"] == 0
    assert stats["false_wake_cpu_fraction"] == 0.0
    assert stats["passes_cpu_le_2pct"] is True


def test_deterministic_noise_and_low_false_wake_cpu() -> None:
    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(123)
    p1 = synth_noise_pcm("pink", 16000 * 5, -30.0, rng1)
    p2 = synth_noise_pcm("pink", 16000 * 5, -30.0, rng2)
    assert np.array_equal(p1, p2)
    with pytest.raises(ValueError):
        synth_noise_pcm("purple", 1600, -30.0, rng1)
    noise_res = eval_pure_noise(duration_s=20.0, seed=47)
    assert noise_res["silence"]["false_wake_cpu_fraction"] == 0.0
    for label, row in noise_res.items():
        assert row["false_wake_cpu_fraction"] <= 0.05, label


def test_onset_detection_and_torture_sets_t1_t4() -> None:
    res = eval_torture_t1_t4(seed=47, trials=3)
    assert set(res.keys()) == {
        "T1_fricative_onset",
        "T2_quiet_speech_-35dBFS",
        "T3_post_burst_noise",
        "T4_low_snr_floor_transition",
    }
    for name, row in res.items():
        assert row["n_utterances"] == 3
        assert row["woken_before_1s_preroll"] == 3, f"{name}: {row}"
        assert row["latency_ms"]["max"] is not None
        assert row["latency_ms"]["max"] <= 1000.0


def test_push_hop_rejects_wrong_length() -> None:
    gate = Sentinel0()
    with pytest.raises(ValueError):
        gate.push_hop([0] * (HOP - 1))


def test_streaming_wav_segments_bit_exact_and_non_overlapping(tmp_path) -> None:
    import hashlib
    import json
    import wave

    from tools.sentinel0_segments import WakeSegmenter, stream_wavs_to_segments

    pcm = np.array(golden_pcm(96000), dtype=np.int16)
    # Split across two WAV parts at a non-hop-aligned offset (33333 % 160 == 53)
    split = 33333
    p1, p2 = tmp_path / "part-1.wav", tmp_path / "part-2.wav"
    for path, arr in ((p1, pcm[:split]), (p2, pcm[split:])):
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(arr.tobytes())

    out_jsonl = tmp_path / "golden.segments.jsonl"
    # Tiny chunk_frames=377 exercises cross-chunk and cross-part carry logic thoroughly
    summary = stream_wavs_to_segments([p1, p2], out_jsonl, chunk_frames=377, label="golden")

    # Reference scalar Sentinel0 + WakeSegmenter on concatenated PCM
    gate = Sentinel0()
    ref_sm = WakeSegmenter()
    for i in range(len(pcm) // HOP):
        hot, path, _, _ = gate.push_hop(pcm[i * HOP:(i + 1) * HOP])
        ref_sm.push_hop(i, hot, path)
    ref_segs = ref_sm.finish(len(pcm))

    emitted = [json.loads(line) for line in out_jsonl.read_text(encoding="utf-8").splitlines()]
    assert emitted == ref_segs
    # Gap between burst 1 (hop 257) and burst 2 (hop 401) is 144 hops < 150 idle hops -> 1 segment
    assert len(emitted) == 1
    assert emitted[0]["wake_hop"] == 200
    assert summary["metrics"]["hot_hops"] == 113
    assert summary["metrics"]["segment_count"] == 1
    assert summary["stream"]["total_samples"] == len(pcm)
    assert summary["output"]["segments_sha256"] == hashlib.sha256(out_jsonl.read_bytes()).hexdigest()
    assert summary["inputs"][0]["wav_sha256"] == hashlib.sha256(p1.read_bytes()).hexdigest()
    assert summary["inputs"][1]["wav_sha256"] == hashlib.sha256(p2.read_bytes()).hexdigest()
    # With idle_ms=1000 and preroll_ms=200, bursts split into 2 clean non-overlapping segments
    out2 = tmp_path / "golden_split.segments.jsonl"
    s2 = stream_wavs_to_segments([p1, p2], out2, preroll_ms=200, idle_ms=1000, chunk_frames=377)
    em2 = [json.loads(line) for line in out2.read_text(encoding="utf-8").splitlines()]
    assert len(em2) == 2 and s2["metrics"]["segment_count"] == 2
    assert [s["wake_hop"] for s in em2] == [200, 401]
    prev_e = 0
    for idx, seg in enumerate(em2):
        assert seg["id"] == idx
        assert prev_e <= seg["s0"] < seg["e"] <= len(pcm)
        prev_e = seg["e"]


def test_wake_segmenter_merges_overlapping_preroll_and_handles_eof() -> None:
    from tools.sentinel0_segments import WakeSegmenter

    sm = WakeSegmenter(preroll_samples=16000, idle_hops=150)
    # Wake at hop 200 -> closes at hop 350 (e = 351*160 = 56160)
    # Second wake at hop 380 -> s0 = 381*160 - 16000 = 44960 < 56160 -> merges!
    for i in range(420):
        sm.push_hop(i, hot=(i in (200, 380)), path=1 if i == 200 else 2)
    segs = sm.finish(420 * HOP + 55)
    assert len(segs) == 1
    assert segs[0]["s0"] == 201 * HOP - 16000
    assert segs[0]["e"] == 420 * HOP + 55
    assert segs[0]["wake_hops"] == [200, 380]
    assert sm.raw_wakes == 2


def test_hot_u8_matches_wav_streamer_segments_exactly(tmp_path) -> None:
    import json
    import wave

    from tools.sentinel0_segments import hot_u8_to_segments, stream_wavs_to_segments

    pcm = np.array(golden_pcm(96000), dtype=np.int16)
    wav_p = tmp_path / "golden.wav"
    with wave.open(str(wav_p), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(pcm.tobytes())
    hot, _ = run_sentinel0_fast(pcm)
    hot_p = tmp_path / "hot.u8"
    hot_p.write_bytes(hot.astype(np.uint8).tobytes())
    out_w, out_h = tmp_path / "from_wav.jsonl", tmp_path / "from_hot.jsonl"
    stream_wavs_to_segments([wav_p], out_w, preroll_ms=200, idle_ms=1000)
    sh = hot_u8_to_segments(hot_p, len(pcm), out_h, preroll_ms=200, idle_ms=1000)
    sw_segs = [json.loads(l) for l in out_w.read_text(encoding="utf-8").splitlines()]
    sh_segs = [json.loads(l) for l in out_h.read_text(encoding="utf-8").splitlines()]
    assert [{k: v for k, v in s.items() if k != "wake_path"} for s in sw_segs] == [
        {k: v for k, v in s.items() if k != "wake_path"} for s in sh_segs
    ]
    assert sh["metrics"]["segment_count"] == 2
    assert sh["metrics"]["hot_hops"] == 113
