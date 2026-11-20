"""Reference isolation, IPC failures, packetization and stream-instance integrity."""
import hashlib
import io
import json
from pathlib import Path
import struct
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).absolute().parents[1]))
from bridge.worker import serve  # noqa: E402


class Session:
    def get_providers(self):
        return ["CPUExecutionProvider"]


class Stream:
    def __init__(self):
        self.finished = False
        self.t_next = 0
        self.forced = 0
        self.pcm = []

    def accept_waveform(self, sr, samples):
        assert sr == 16000
        self.pcm.extend(samples)

    def input_finished(self):
        self.finished = True


class Recognizer:
    enc = dec = joi = Session()

    def create_stream(self):
        return Stream()

    def is_ready(self, s):
        return s.t_next < len(s.pcm) // 160 + int(s.finished)

    def decode_streams(self, streams):
        for s in streams:
            s.t_next += 1

    def get_result(self, s):
        return str(round(sum(s.pcm) * 32768))


def frame(sequence, kind, items=(), raw=b"", **changes):
    q = dict(run_id="run", worker_id="w0", seq=sequence, kind=kind, items=list(items))
    q.update(changes)
    header = json.dumps(q).encode()
    return struct.pack("<I", len(header)) + header + raw


def item(n, offset=0, start=True, finish=True, uid="u0"):
    return dict(id=uid, offset=offset, samples=n, start=start, finish=finish)


def simulate(requests, rec=None):
    out = io.BytesIO()
    serve(rec or Recognizer(), io.BytesIO(b"".join(requests)), out, "run", "a" * 64, 0.1)
    return [json.loads(line) for line in out.getvalue().splitlines()]


@pytest.mark.parametrize("packet", [160, 320, 592, 113])
def test_packetization_short_tail_and_actual_audio_hash(packet):
    pcm = np.arange(1001, dtype="<i2").tobytes()
    requests = [frame(0, "init")]
    seq = 1
    for offset in range(0, 1001, packet):
        end = min(offset + packet, 1001)
        requests.append(frame(seq, "batch", [item(end-offset, offset, offset == 0, end == 1001)],
                              pcm[offset * 2:end * 2]))
        seq += 1
    requests.append(frame(seq, "shutdown"))
    replies = simulate(requests)
    final = replies[-2]["rows"][0]
    assert final["pcm_sha256"] == hashlib.sha256(pcm).hexdigest()
    assert final["accepted_samples"] == final["processed_samples"] == 1001
    assert final["terminal"] and final["drained"]
    assert final["hyp"] == "500500"
    assert all(r["rows"][0]["processed_samples"] == 0 for r in replies[1:-2])


@pytest.mark.parametrize("change", [dict(run_id="old"), dict(worker_id="w1"),
                                  dict(seq=99), dict(ref="secret"), dict(timings=[1, 2])])
def test_stale_identity_and_reference_fields_refused(change):
    with pytest.raises(ValueError):
        simulate([frame(0, "init", **change)])


def test_duplicate_instance_and_truncated_audio_are_failures():
    batch = frame(1, "batch", [item(1)], b"\x00\x00")
    with pytest.raises(ValueError, match="reused"):
        simulate([frame(0, "init"), batch, frame(2, "batch", [item(1)], b"\x00\x00")])
    with pytest.raises(ValueError, match="truncated"):
        simulate([frame(0, "init"), frame(1, "batch", [item(100)], b"\x00")])


def test_unfinished_shutdown_and_inference_failure_do_not_become_empty_finals():
    with pytest.raises(ValueError, match="unfinished"):
        simulate([frame(0, "init"), frame(1, "batch", [item(1, finish=False)], b"\x00\x00"),
                  frame(2, "shutdown")])

    class Failed(Recognizer):
        def decode_streams(self, streams):
            raise RuntimeError("decoder crashed")

    with pytest.raises(RuntimeError, match="crashed"):
        simulate([frame(0, "init"), frame(1, "batch", [item(1)], b"\x00\x00")], Failed())


def test_future_audio_does_not_change_earlier_observation():
    prefix = [frame(0, "init"), frame(1, "batch", [item(1, finish=False)], b"\x01\x00")]
    def run(tail):
        return simulate(prefix + [frame(2, "batch", [item(1, offset=1, start=False)], tail),
                                  frame(3, "shutdown")])
    a, b = run(b"\x01\x00"), run(b"\xff\x7f")
    assert a[1] == b[1]
    assert a[2]["rows"][0]["hyp"] != b[2]["rows"][0]["hyp"]


@pytest.mark.parametrize("key,value", [("samples", True), ("offset", False), ("id", 1),
                                     ("start", 1), ("finish", 0)])
def test_packet_types_do_not_coerce_bools_and_integers(key, value):
    bad = item(1)
    bad[key] = value
    with pytest.raises(ValueError):
        simulate([frame(0, "init"), frame(1, "batch", [bad], b"\x00\x00")])
