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
