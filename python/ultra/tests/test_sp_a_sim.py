"""Deterministic tests of the SP-A arrival/decode loop with a fake clock and recognizer."""
import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ultra.sp_a import SR, Clock, Item, simulate  # noqa: E402

CHUNK = int(0.32 * SR)


class FakeTime:
    def __init__(self):
        self.t, self.cpu_s = 0.0, 0.0

    def clock(self):
        def sleep(d):
            self.t += max(0.0, d)
        return Clock(now=lambda: self.t, sleep=sleep, cpu=lambda: self.cpu_s)


class FakeStream:
    def __init__(self):
        self.pending, self.done, self.finished = 0, 0, False


class FakeRec:
    """Each decoded chunk costs `fixed` per call plus `per_stream` per stream, in fake seconds."""

    def __init__(self, ft, fixed, per_stream):
        self.ft, self.fixed, self.per_stream = ft, fixed, per_stream

    def create_stream(self):
        return FakeStream()

    def is_ready(self, s):
        return s.pending >= CHUNK or (s.finished and s.pending > 0)

    def _spend(self, n):
        cost = self.fixed + self.per_stream * n
        self.ft.t += cost
        self.ft.cpu_s += cost

    def decode_streams(self, ss):
        self._spend(len(ss))
        for s in ss:
            take = min(CHUNK, s.pending)
            s.pending -= take
            s.done += take

    def decode_stream(self, s):
        self.decode_streams([s])

    def get_result(self, s):
        return "ok"


def _items(lengths_s):
    return [Item(f"u{i}", "spk", "ok", np.zeros(int(L * SR), dtype=np.int16)) for i, L in enumerate(lengths_s)]


def _accepting(stream_cls=FakeStream):
    def accept(self, sr, x):
        self.pending += len(x)

    def finish(self):
        self.finished = True
    stream_cls.accept_waveform = accept
    stream_cls.input_finished = finish


_accepting()


def _opt(**kw):
    base = dict(packet_ms=20, grid_ms=320, tail_pad_s=0.0, streams=4, decode="batch", max_batch=1024,
                paced=True, duration_s=0.0)
    base.update(kw)
    return SimpleNamespace(**base)


def test_light_load_holds_real_time_and_scores_everything():
    ft = FakeTime()
    out = simulate(_items([3.0, 2.0, 4.0, 1.5, 2.5, 3.5]), FakeRec(ft, 0.005, 0.002), _opt(), ft.clock())
    assert out["drained"] and out["utterances_scored"] == 6
    assert out["lateness_ms"]["p99"] < 400  # one grid (320 ms) plus decode time
    assert out["lateness_ms"]["last_third_mean"] <= out["lateness_ms"]["first_third_mean"] + 50


def test_overload_is_visible_and_grows():
    ft = FakeTime()
    # 4 streams x 0.5 s of compute per 0.32 s chunk: far slower than real time.
    out = simulate(_items([6.0] * 4), FakeRec(ft, 0.0, 0.5), _opt(), ft.clock())
    lat = out["lateness_ms"]
    assert lat["p99"] > 1000 and lat["last_third_mean"] > lat["first_third_mean"]
    assert out["drained"] and out["utterances_scored"] == 4


def test_staggered_streams_refill_mid_timeline():
    ft = FakeTime()
    out = simulate(_items([0.5, 3.0, 0.7, 2.2, 0.4, 1.9, 0.9]), FakeRec(ft, 0.002, 0.001),
                   _opt(streams=2), ft.clock())
    assert out["utterances_scored"] == 7 and out["drained"]


def test_cutoff_drains_and_counts_accepted_audio_only():
    ft = FakeTime()
    out = simulate(_items([30.0] * 3), FakeRec(ft, 0.001, 0.001), _opt(streams=3, duration_s=2.0), ft.clock())
    assert out["drained"] and out["utterances_scored"] == 0  # none fully arrived
    assert abs(out["audio_accepted_s"] - 6.0) < 0.2  # 3 streams x ~2 s
    assert "drain" in out


def test_batching_reduces_cost_against_single():
    a, b = FakeTime(), FakeTime()
    items = _items([2.0] * 8)
    batch = simulate(items, FakeRec(a, 0.02, 0.001), _opt(streams=8, paced=False), a.clock())
    single = simulate(items, FakeRec(b, 0.02, 0.001), _opt(streams=8, paced=False, decode="single"), b.clock())
    assert batch["rtf_cpu"] < single["rtf_cpu"] / 3
