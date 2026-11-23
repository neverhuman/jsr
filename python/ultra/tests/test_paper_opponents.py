"""The open-opponent runners and receipts (docs/paper/opponents): packet feed, traces, shards, guard.

The recognisers themselves are not installed here; they are replaced by small fakes, so these
tests pin the harness behaviour around them.
"""

import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import wave

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
OPPONENTS = REPO / "docs/paper/opponents"
sys.path.insert(0, str(OPPONENTS))

import open_common as common  # noqa: E402
import open_receipt as receipt  # noqa: E402
import open_turn as turn  # noqa: E402
import vosk_stream  # noqa: E402
import whispercpp_stream  # noqa: E402


def staged_wav(path, samples, rate=16000, channels=1):
    """A WAV written the way `paper_dev_audio.py stage` writes the panel."""
    pcm = b"".join(int(i % 2000 - 1000).to_bytes(2, "little", signed=True) for i in range(samples * channels))
    with wave.open(str(path), "wb") as out:
        out.setparams((channels, 2, rate, samples, "NONE", "not compressed"))
        out.writeframes(pcm)
    return pcm


def trace_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_staged_wav_is_read_and_anything_else_is_rejected(tmp_path):
    pcm = staged_wav(tmp_path / "u00000.wav", 5000)
    assert common.read_staged_wav(tmp_path / "u00000.wav") == pcm
    assert common.wav_samples(tmp_path / "u00000.wav") == 5000
    staged_wav(tmp_path / "rate.wav", 100, rate=8000)
    staged_wav(tmp_path / "stereo.wav", 100, channels=2)
    (tmp_path / "short.wav").write_bytes((tmp_path / "u00000.wav").read_bytes()[:-2])
    for name in ("rate.wav", "stereo.wav", "short.wav"):
        with pytest.raises(ValueError):
            common.read_staged_wav(tmp_path / name)
    (tmp_path / "notes.txt").write_text("x")
    with pytest.raises(ValueError):
        common.staged_wavs(tmp_path)


def test_packets_are_20_ms_and_the_last_one_may_be_short():
    pcm = bytes(2 * 1000)
    fed = [(samples, len(packet)) for samples, packet in common.packets(pcm)]
    assert fed == [(320, 640), (640, 640), (960, 640), (1000, 80)]
    assert b"".join(packet for _, packet in common.packets(pcm)) == pcm


def test_balanced_shards_cover_every_item_once_with_near_equal_cost():
    costs = [(i * 7919) % 97 + 3 for i in range(730)]
    shards = common.balanced_shards(costs, 8)
    assert sorted(i for shard in shards for i in shard) == list(range(730))
    assert all(shard == sorted(shard) for shard in shards)
    loads = [sum(costs[i] for i in shard) for shard in shards]
    assert max(loads) - min(loads) <= max(costs)
    assert shards == common.balanced_shards(costs, 8)
    assert common.balanced_shards([5, 1], 1) == [[0, 1]]


def test_trace_events_carry_both_naming_schemes(tmp_path):
    with common.Trace(tmp_path / "trace.jsonl") as trace:
        trace.event("u00000", 320, 0.0125, 0.001, "", False)
        trace.event("u00000", 113608, 1.5, 0.25, "mister quilter", True)
    first, last = trace_rows(tmp_path / "trace.jsonl")
    assert first == {"id": "u00000", "audio_ms": 20, "wall_ms": 12.5, "text": "", "final": False,
                     "audio_s": 0.02, "chunk_wall_ms": 1.0, "cum_wall_ms": 12.5}
    assert last["audio_ms"] == 7100.5 and last["final"] is True and last["cum_wall_ms"] == last["wall_ms"]
    with pytest.raises(FileExistsError):
        common.Trace(tmp_path / "trace.jsonl")
    totals = receipt.trace_summary(tmp_path / "trace.jsonl")
    assert (totals["events"], totals["utterances"]) == (2, 1)
    assert totals["audio_seconds"] == pytest.approx(7.1005, abs=1e-3) and totals["recogniser_seconds"] == 1.5


def test_worker_outputs_merge_into_one_prediction_file_and_one_trace(tmp_path):
    class Args:
        predictions = tmp_path / "predictions.jsonl"
        trace = tmp_path / "trace.jsonl"

    for index, ident in enumerate(("u00001", "u00000")):
        common.write_predictions(common.worker_path(Args.predictions, index), {ident: f"text {index}"})
        with common.Trace(common.worker_path(Args.trace, index)) as trace:
            trace.event(ident, 320, 0.1, 0.1, f"text {index}", True)
    merged = common.merge_worker_outputs(Args, 2)
    assert merged == {"u00001": "text 0", "u00000": "text 1"}
    assert Args.predictions.read_text().splitlines()[0] == '{"id": "u00000", "hyp": "text 1"}'
    assert [row["id"] for row in trace_rows(Args.trace)] == ["u00001", "u00000"]
    assert sorted(path.name for path in tmp_path.iterdir()) == ["predictions.jsonl", "trace.jsonl"]


class FakeKaldi:
    """Reports a partial word per second of audio and one endpoint at 2 s."""

    def __init__(self, model, rate):
        assert rate == 16000
        self.samples = 0
        self.segment_start = 0

    def AcceptWaveform(self, packet):
        self.samples += len(packet) // 2
        if self.segment_start == 0 and self.samples >= 32000:
            self.endpoint = self.samples
            return True
        return False

    def _words(self, through):
        return " ".join(f"w{second}" for second in range(self.segment_start // 16000, through // 16000))

    def Result(self):
        text = self._words(self.endpoint)
        self.segment_start = self.endpoint
        return json.dumps({"text": text})

    def PartialResult(self):
        return json.dumps({"partial": self._words(self.samples)})

    def FinalResult(self):
        return json.dumps({"text": self._words(self.samples) + " end"})


def test_vosk_runner_reads_a_result_per_packet_and_joins_endpoint_segments(tmp_path):
    staged_wav(tmp_path / "u00007.wav", 3 * 16000 + 100)
    with common.Trace(tmp_path / "trace.jsonl") as trace:
        ident, text = vosk_stream.recognise(None, FakeKaldi, tmp_path / "u00007.wav", trace)
    assert (ident, text) == ("u00007", "w0 w1 w2 end")
    rows = trace_rows(tmp_path / "trace.jsonl")
    assert len(rows) == 151 + 1 and [row["final"] for row in rows].count(True) == 1
    assert rows[49]["audio_ms"] == 1000 and rows[49]["text"] == "w0"
    assert rows[99]["text"] == "w0 w1" and rows[149]["text"] == "w0 w1 w2"
    assert rows[-1] == {**rows[-1], "audio_ms": 3006.25, "text": text, "final": True}
    assert all(a["wall_ms"] <= b["wall_ms"] for a, b in zip(rows, rows[1:]))


def picovoice_chunks(audio):
    """The chunk loop of Picovoice's WhisperCppStreamingEngine.transcribe (engine.py at 272be8f)."""
    chunk_samples = int((3000 / 1000) * 16000)
    min_chunk_samples = int(0.2 * 16000)
    chunks = []
    for start in range(0, len(audio), chunk_samples):
        chunk = audio[start:start + chunk_samples]
        if len(chunk) < min_chunk_samples:
            chunk = np.pad(chunk, (0, min_chunk_samples - len(chunk)))
        chunks.append(chunk)
    return chunks


class Segment:
    def __init__(self, text):
        self.text = text


@pytest.mark.parametrize("samples", [48000, 48000 + 320, 2 * 48000 + 1000, 3 * 48000, 2500, 113613])
def test_whisper_runner_decodes_the_same_chunks_as_the_picovoice_engine(tmp_path, samples):
    pcm = staged_wav(tmp_path / "u00003.wav", samples)
    seen = []

    class FakeModel:
        def transcribe(self, chunk):
            seen.append(chunk)
            return [Segment(f" c{len(seen)}. [BLANK_AUDIO]")]

    with common.Trace(tmp_path / "trace.jsonl") as trace:
        ident, text = whispercpp_stream.recognise(FakeModel(), np, tmp_path / "u00003.wav", trace)
    audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
    expected = picovoice_chunks(audio)
    assert len(seen) == len(expected) == whispercpp_stream.chunk_count(samples)
    assert all(a.dtype == np.float32 and np.array_equal(a, b) for a, b in zip(seen, expected, strict=True))
    assert text == " ".join(f"c{i}." for i in range(1, len(expected) + 1))
    rows = trace_rows(tmp_path / "trace.jsonl")
    assert len(rows) == len(expected) and rows[-1]["final"] and rows[-1]["text"] == text
    assert [row["final"] for row in rows[:-1]] == [False] * (len(rows) - 1)
    assert [row["audio_ms"] for row in rows[:-1]] == [3000 * i for i in range(1, len(rows))]


def fake_topology(root, cpus=16):
    """Eight cores with two threads each (cpu N and N+8), four cores per L3."""
    for cpu in range(cpus):
        core = cpu % 8
        (root / f"cpu{cpu}/topology").mkdir(parents=True)
        (root / f"cpu{cpu}/cache/index3").mkdir(parents=True)
        (root / f"cpu{cpu}/topology/thread_siblings_list").write_text(f"{core},{core + 8}\n")
        first = core - core % 4
        (root / f"cpu{cpu}/cache/index3/shared_cpu_list").write_text(f"{first}-{first + 3},{first + 8}-{first + 11}\n")


def snapshots(busy, cpus=16, ticks=10000):
    before = {"utc": "a", "monotonic_s": 0.0, "loadavg": ["0", "0", "0"],
              "cpus": {str(cpu): [0] * 8 for cpu in range(cpus)}}
    after = {"utc": "b", "monotonic_s": 100.0, "loadavg": ["1", "0", "0"], "cpus": {}}
    for cpu in range(cpus):
        used = round(ticks * busy.get(cpu, 0.0))
        after["cpus"][str(cpu)] = [used, 0, 0, ticks - used, 0, 0, 0, 0]
    return before, after


def test_guard_passes_an_arm_alone_on_its_ccx_and_names_other_active_cores(tmp_path):
    fake_topology(tmp_path)
    before, after = snapshots({1: 1.0, 9: 0.004, 2: 0.01, 4: 0.97, 5: 0.99, 13: 0.2})
    result = receipt.guard(before, after, [1], 0.01, 0.02, root=tmp_path)
    assert result["verdict"] == "clean" and result["reasons"] == []
    assert result["smt_siblings_busy"] == {"9": 0.004}
    assert set(result["ccx_neighbours_busy"]) == {"0", "2", "3", "8", "10", "11"}
    assert result["other_active_cores"] == [4, 5]
    assert result["other_busy_cpus"] == {"4": 0.97, "5": 0.99, "13": 0.2}
    assert result["service_busy"] == {"1": 1.0} and result["seconds"] == 100.0


@pytest.mark.parametrize("busy, reason", [({9: 0.03}, "SMT sibling cpu9"), ({3: 0.5}, "CCX neighbour cpu3"),
                                          ({11: 0.05}, "CCX neighbour cpu11")])
def test_guard_rejects_a_busy_sibling_or_ccx_neighbour(tmp_path, busy, reason):
    fake_topology(tmp_path)
    before, after = snapshots({1: 1.0, **busy})
    result = receipt.guard(before, after, [1], 0.01, 0.02, root=tmp_path)
    assert result["verdict"] == "contaminated" and reason in result["reasons"][0]


def test_guard_rejects_a_busy_host_when_asked_to(tmp_path):
    fake_topology(tmp_path)
    before, after = snapshots({1: 1.0, 4: 0.97, 5: 0.99, 6: 0.8})
    assert receipt.guard(before, after, [1], 0.01, 0.02, root=tmp_path)["verdict"] == "clean"
    assert receipt.guard(before, after, [1], 0.01, 0.02, other_limit=3, root=tmp_path)["verdict"] == "clean"
    busy = receipt.guard(before, after, [1], 0.01, 0.02, other_limit=2, root=tmp_path)
    assert busy["verdict"] == "contaminated" and busy["reasons"] == ["3 other cores were active on the host (limit 2)"]
    assert busy["limits"]["other_active_cores"] == 2


def test_guard_covers_both_ccxs_of_an_eight_core_arm(tmp_path):
    fake_topology(tmp_path)
    before, after = snapshots({cpu: 1.0 for cpu in range(8)} | {12: 0.5})
    result = receipt.guard(before, after, list(range(8)), 0.01, 0.02, root=tmp_path)
    assert sorted(result["smt_siblings_busy"]) == sorted(str(cpu) for cpu in range(8, 16))
    assert result["ccx_neighbours_busy"] == {} and "cpu12" in result["reasons"][0]


def test_active_cores_count_a_core_once_and_cpu_lists_expand(tmp_path):
    fake_topology(tmp_path)
    before, after = snapshots({0: 0.9, 8: 0.8, 3: 0.6, 5: 0.49})
    busy = receipt.busy_fractions(before, after)
    assert receipt.active_cores(busy, root=tmp_path) == [0, 3]
    assert receipt.active_cores(busy, exclude=[0, 8], root=tmp_path) == [3]
    assert receipt.parse_cpus("8-15,4") == [4, 8, 9, 10, 11, 12, 13, 14, 15]


def test_turn_plan_parsing_and_turn_rules(tmp_path):
    jobs, anchors = turn.read_plan("anchor ours 3.5 0.5\n5 full a 1 4 0 17\n5 full a 2 24 1 17  # second\n\n"
                                   "12 p73 b 1 4 0 4.5\nanchor ours-s8 1 0.3 12,20\n")
    assert [(job["entry"], job["panel"], job["arm"], job["round"], job["est_s"]) for job in jobs] == [
        ("5", "full", "a", 1, 1020.0), ("5", "full", "a", 2, 1020.0), ("12", "p73", "b", 1, 270.0)]
    assert [job["entry"] for job in turn.read_plan("6b full x 1 4 0 3\n")[0]] == ["6b"]
    with pytest.raises(ValueError):
        turn.read_plan("6-b full x 1 4 0 3\n")
    assert turn.turn_has_come("6b", "DONE 1 t a\nDONE 2 t a\nDONE 3 t a\nDONE 4 t a\nDONE 5 t a\n")
    assert not turn.turn_has_come("7", "DONE 6 t a\n", "6 | a | x\n6b | b | x\n7 | c | x\n")
    assert turn.anchor_for(anchors, 5) == {"arm": "ours", "est_s": {"full": 210.0, "p73": 30.0}}
    assert turn.anchor_for(anchors, 12)["arm"] == turn.anchor_for(anchors, 20)["arm"] == "ours-s8"
    assert turn.anchor_for({}, 5) is None
    assert turn.anchor_service("ours", "8-15") == "8" and turn.anchor_service("ours", "24") == "24"
    assert turn.anchor_service("ours-s8", "8-15") == "8-15" and turn.anchor_service("x-w8", "0,2-3") == "0,2-3"
    for bad in ("5 full a 1 4 0", "anchor ours 3"):
        with pytest.raises(ValueError):
            turn.read_plan(bad)
    done = "DONE 1 t energy\nSKIP 2 t runtime-lane\nSKIP 2b t lead (re-queued)\nDONE 3 t bench-open\nnote 4\n"
    assert turn.finished_entries(done) == {"1", "2", "2b", "3"} and turn.finished_entries(done, "bench-open") == {"3"}
    assert turn.turn_has_come(4, done) and not turn.turn_has_come(5, done) and turn.turn_has_come(1, "")
    order = "# header\n1  | energy | x\n2  | rt | x\n2b | energy | x\n3  | rt | x\n4  | rt | x\n4b | energy | x\n5  | bo | x\n"
    assert turn.order_ids(order) == ["1", "2", "2b", "3", "4", "4b", "5"]
    assert not turn.turn_has_come(5, done + "DONE 4 t rt\n", order)  # 4b, listed before 5, is not done yet
    assert turn.turn_has_come(5, done + "DONE 4 t rt\nSKIP 4b t energy\n", order)
    assert turn.turn_has_come(4, done, order) and not turn.turn_has_come(3, "DONE 1 t e\n", order)
    assert turn.fits(jobs[0], 0.0, False) and turn.fits(jobs[0], 5000.0, False)
    assert turn.fits(jobs[2], 1020.0, True) and not turn.fits(jobs[1], 1020.0, True)
    assert turn.cpu_count("4") == 1 and turn.cpu_count("8-15") == 8 and turn.cpu_count("0,2-3") == 3
    assert turn.receipt_dir(tmp_path, "p73", "8-15", 2, "x") == tmp_path / "8c-p73/cold/round-2/x"
    assert turn.receipt_dir(tmp_path, "full", "4", 1, "x") == tmp_path / "1c/cold/round-1/x"
    for arm, wall, verdict in (("t8", 300.0, "clean"), ("w8", 200.0, "clean"), ("z8", 100.0, "contaminated")):
        run = turn.receipt_dir(tmp_path, "p73", "8-15", 1, arm)
        run.mkdir(parents=True)
        (run / "measurement.json").write_text(json.dumps({"wall_seconds": wall}))
        (run / "host.cpu.json").write_text(json.dumps({"verdict": verdict}))
    job = {"panel": "p73", "service": "8-15"}
    assert turn.resolve(tmp_path, {**job, "arm": "t8|w8"}) == "w8"
    assert turn.resolve(tmp_path, {**job, "arm": "t8|z8"}) == "t8"  # a busy-host round does not count
    assert turn.resolve(tmp_path, {**job, "arm": "q8|r8"}) == "q8"  # nothing measured yet: the first
    assert turn.next_round(tmp_path, "p73", "8-15", "t8") == 2 and turn.next_round(tmp_path, "full", "4", "t8") == 1
    assert turn.read_requirements("require 14b models/x/w.bin\n14b full x 1 4 0 3\nrequire 14b y  # z\n") == {
        "14b": ["models/x/w.bin", "y"]}
    assert [job["entry"] for job in turn.read_plan("require 14b m\n14br full x 1 4 0 3\n")[0]] == ["14br"]
    with pytest.raises(ValueError):
        turn.read_requirements("require 14b\n")
    requeued = ("14 | bench-open | w\n14b | bench-open | x\n7r | runtime-lane | y\n14br | bench-open | x  # re-queued\n"
                "14brr | someone | x\n15 | bench-cross | z\n")
    assert [item[:2] for item in turn.order_entries(requeued)] == [
        ("14", "bench-open"), ("14b", "bench-open"), ("7r", "runtime-lane"), ("14br", "bench-open"),
        ("14brr", "someone"), ("15", "bench-cross")]
    assert turn.family_ids("14b", requeued) == ["14b", "14br"] and turn.family_ids("14", requeued) == ["14"]
    skipped = "DONE 14 t bench-open\nSKIP 14b t bench-open (not staged; re-queued as 14br after 7r)\n"
    assert turn.finished_entries(skipped, "bench-open") == {"14", "14b"}
    assert turn.finished_entries(skipped, kinds=("DONE",)) == {"14"}
    assert turn.active_id("14b", requeued, skipped) == "14br" and turn.active_id("14", requeued, skipped) is None
    assert not turn.turn_has_come("14br", skipped, requeued) and turn.turn_has_come("14br", skipped + "DONE 7r t r\n", requeued)


FAKE_RUN = """
case "$3" in *-*|*,*) group=8c ;; *) group=1c ;; esac
[[ $OPEN_PANEL == full ]] || group="$group-$OPEN_PANEL"
receipt="$OPEN_RUNS/$group/cold/round-$2/$1"
mkdir -p "$receipt"
wall=0.3; [[ $1 == b ]] && wall=1.0; [[ $1 == c ]] && wall=0.5
printf '{"wall_seconds": %s}\\n' "$wall" > "$receipt/measurement.json"
printf '{"verdict": "clean"}\\n' > "$receipt/host.cpu.json"
owner=$(cut -d' ' -f1-2 "$PAPER_OPPONENTS_ROOT/TIMING.lock/owner")
printf '%s %s %s %s %s %s\\n' "$1" "$2" "$3" "$4" "$OPEN_PANEL" "$owner" >> "$FAKE_LOG"
sleep 0.3
if [[ $1 == flaky && ! -e $PAPER_OPPONENTS_ROOT/flaky.seen ]]; then
    : > "$PAPER_OPPONENTS_ROOT/flaky.seen"
    exit 3
fi
"""


def test_turn_runner_holds_the_lock_only_for_its_own_entries_with_an_anchor_per_hold(tmp_path):
    host = tmp_path / "opponents"
    host.mkdir()
    (host / "TIMING.done").write_text("DONE 1 t energy\n")
    (host / "TIMING.order").write_text("1 | energy | a\n2 | runtime-lane | b\n3 | bench-open | c\n4 | bench-cross | d\n"
                                        "4b | energy | e\n5 | bench-open | f\n")
    fake = tmp_path / "fake-run.sh"
    fake.write_text(FAKE_RUN)
    plan = tmp_path / "plan.txt"
    plan.write_text("anchor anc 0.001 0.001\n3 full a 1 4 0 0.004\n3 full a 2 24 1 0.004\n3 full flaky 3 28 2 0.004\n"
                    "5 p73 b 1 8-15 3 0.004\n5 p73 c 1 8-15 3 0.004\n5 p73 b|c 2 8-15 3 0.004\nanchor anc-s8 0.001 0.001 5\n")
    env = {**os.environ, "PAPER_OPPONENTS_ROOT": str(host), "OPEN_RUNS": str(tmp_path / "runs"),
           "OPEN_RUN_SCRIPT": str(fake), "OPEN_POLL_S": "0.1", "OPEN_HOLD_S": "0.6", "OPEN_HOLD_MARGIN_S": "0.1",
           "FAKE_LOG": str(tmp_path / "calls.log"), "OPEN_AGENT": "bench-open",
           "OPEN_AFTER_HOLD": f'cut -d" " -f1-2 "{host}/TIMING.lock/owner" >> "{tmp_path}/after.log"'}
    command = [sys.executable, str(OPPONENTS / "open_turn.py"), str(plan), str(tmp_path / "logs")]
    runner = subprocess.Popen(command, env=env)
    time.sleep(1.0)
    assert not (tmp_path / "calls.log").exists() and not (host / "TIMING.lock").exists()  # entry 2 is not done yet

    def others():  # entry 2 finishes; later entry 4 is skipped by its owner
        with (host / "TIMING.done").open("a") as out:
            out.write("DONE 2 t runtime-lane\n")
        while "DONE 3" not in (host / "TIMING.done").read_text():
            time.sleep(0.05)
        with (host / "TIMING.done").open("a") as out:
            out.write("SKIP 4 t bench-cross\n")
        time.sleep(0.5)
        assert "DONE 5" not in (host / "TIMING.done").read_text()  # 4b still blocks entry 5
        with (host / "TIMING.done").open("a") as out:
            out.write("DONE 4b t energy\n")

    helper = threading.Thread(target=others)
    helper.start()
    assert runner.wait(timeout=90) == 0
    helper.join()
    assert (tmp_path / "calls.log").read_text().splitlines() == [
        "anc 1 4 0 full bench-open 3", "a 1 4 0 full bench-open 3",
        "anc 2 24 1 full bench-open 3", "a 2 24 1 full bench-open 3",
        "anc 3 28 2 full bench-open 3", "flaky 3 28 2 full bench-open 3",
        "anc 4 28 2 full bench-open 3", "flaky 3 28 2 full bench-open 3",
        "anc-s8 1 8-15 3 p73 bench-open 5", "b 1 8-15 3 p73 bench-open 5",
        "anc-s8 2 8-15 3 p73 bench-open 5", "c 1 8-15 3 p73 bench-open 5",
        "anc-s8 3 8-15 3 p73 bench-open 5", "c 2 8-15 3 p73 bench-open 5"]
    done = (host / "TIMING.done").read_text().splitlines()
    assert [line.split()[:2] + line.split()[-1:] for line in done] == [
        ["DONE", "1", "energy"], ["DONE", "2", "runtime-lane"], ["DONE", "3", "bench-open"],
        ["SKIP", "4", "bench-cross"], ["DONE", "4b", "energy"], ["DONE", "5", "bench-open"]]
    assert not (host / "TIMING.lock").exists()
    state = json.loads((tmp_path / "logs/turn.json").read_text())
    assert len(state["holds"]) == 7 and all("released_utc" in hold for hold in state["holds"])
    assert (tmp_path / "after.log").read_text().splitlines() == ["bench-open 3"] * 4 + ["bench-open 5"] * 3
    assert all(hold["after_hold_exit"] == 0 for hold in state["holds"])
    assert [job["status"] for job in state["jobs"] if not job.get("anchor")] == [
        "done", "done", "not-alone", "done", "done", "done", "done"]
    restart = subprocess.run(command, env=env, timeout=30)  # a restarted runner skips what it already did
    assert restart.returncode == 0 and len((host / "TIMING.done").read_text().splitlines()) == 6
    assert len((tmp_path / "calls.log").read_text().splitlines()) == 14


def test_turn_runner_skips_and_requeues_an_entry_whose_model_is_not_staged(tmp_path):
    host = tmp_path / "opponents"
    host.mkdir()
    (host / "TIMING.done").write_text("DONE 1 t energy\n")
    (host / "TIMING.order").write_text("1 | energy | a\n2 | bench-open | ours, final\n2b | bench-open | ours, other\n"
                                        "3 | runtime-lane | b\n4 | bench-open | c\n")
    fake = tmp_path / "fake-run.sh"
    fake.write_text(FAKE_RUN)
    plan = tmp_path / "plan.txt"
    plan.write_text("anchor anc 0.001 0.001\nrequire 2 staged/model.bin\n2 full m 1 4 0 0.004\n"
                    "require 2b never/staged.bin\n2b full n 1 4 0 0.004\n4 full z 1 4 0 0.004\n")
    env = {**os.environ, "PAPER_OPPONENTS_ROOT": str(host), "OPEN_RUNS": str(tmp_path / "runs"),
           "OPEN_RUN_SCRIPT": str(fake), "OPEN_POLL_S": "0.1", "OPEN_HOLD_S": "0.6", "OPEN_HOLD_MARGIN_S": "0.1",
           "FAKE_LOG": str(tmp_path / "calls.log"), "OPEN_AGENT": "bench-open", "OPEN_REQUEUE_MAX": "1",
           "OPEN_PREPARE": f'cut -d" " -f1-2 "{host}/TIMING.lock/owner" >> "{tmp_path}/prepare.log"'}
    command = [sys.executable, str(OPPONENTS / "open_turn.py"), str(plan), str(tmp_path / "logs")]
    runner = subprocess.Popen(command, env=env)

    def owner():  # runtime-lane takes entry 3 once both of bench-open's entries before it are skipped
        while "SKIP 2b" not in (host / "TIMING.done").read_text():
            time.sleep(0.05)
        (host / "staged").mkdir()
        (host / "staged/model.bin").write_text("x")
        with (host / "TIMING.done").open("a") as out:
            out.write("DONE 3 t runtime-lane\n")

    helper = threading.Thread(target=owner)
    helper.start()
    assert runner.wait(timeout=90) == 0
    helper.join()
    assert (tmp_path / "calls.log").read_text().splitlines() == [
        "anc 1 4 0 full bench-open 2r", "m 1 4 0 full bench-open 2r", "anc 2 4 0 full bench-open 4",
        "z 1 4 0 full bench-open 4"]
    assert (tmp_path / "prepare.log").read_text().splitlines() == [
        "bench-open 2", "bench-open 2b", "bench-open 2r", "bench-open 2br"]
    order = (host / "TIMING.order").read_text().splitlines()
    assert [line.split("  #")[0] for line in order] == [
        "1 | energy | a", "2 | bench-open | ours, final", "2b | bench-open | ours, other", "3 | runtime-lane | b",
        "2r | bench-open | ours, final", "2br | bench-open | ours, other", "4 | bench-open | c"]
    assert "# re-queued by bench-open" in order[4] and "after SKIP 2: not staged at its turn: staged/model.bin" in order[4]
    done = [line.split(" ", 4) for line in (host / "TIMING.done").read_text().splitlines()]
    assert [fields[:2] for fields in done] == [
        ["DONE", "1"], ["SKIP", "2"], ["SKIP", "2b"], ["DONE", "3"], ["DONE", "2r"], ["SKIP", "2br"], ["DONE", "4"]]
    assert done[1][4] == "(not staged at its turn: staged/model.bin; re-queued as 2r after 3)"
    assert done[5][4] == "(not staged at its turn: never/staged.bin)"  # OPEN_REQUEUE_MAX=1: skipped, not re-queued
    assert len(list(host.glob("TIMING.order.bak-bench-open-*"))) >= 1 and not (host / "TIMING.lock").exists()
    state = json.loads((tmp_path / "logs/turn.json").read_text())
    assert [job["status"] for job in state["jobs"] if not job.get("anchor")] == ["done", "skipped", "done"]
    restart = subprocess.run(command, env=env, timeout=30)  # nothing left: neither 2 nor 2b runs again
    assert restart.returncode == 0 and len((tmp_path / "calls.log").read_text().splitlines()) == 4


ARM_SCRIPT = """
set -euo pipefail
data=$1 wav=/panel/wav run=/run
source docs/paper/opponents/open-arms.sh
open_arm "$2"
printf '%s\\n' "${cmd[@]}"
printf 'FILES %s\\n' "${#files[@]}"
"""


def open_arm(tmp_path, arm):
    return subprocess.run(["bash", "-c", ARM_SCRIPT, "arm", str(tmp_path), arm], cwd=REPO,
                          capture_output=True, text=True)


def test_arm_definitions_build_the_documented_commands(tmp_path):
    words = open_arm(tmp_path, "vosk-large-w8").stdout.split("\n")
    assert words[0].endswith("opponents/vosk/venv/bin/python") and words[1] == "-I"
    assert words[2] == str(OPPONENTS / "vosk_stream.py")
    assert words[3].endswith("vosk-model-en-us-0.22")
    assert words[4:9] == ["/panel/wav", "/run/predictions.jsonl", "/run/trace.jsonl", "--workers", "8"]
    small = open_arm(tmp_path, "vosk-small").stdout.split("\n")
    assert small[3].endswith("vosk-model-small-en-us-0.15") and small[6] == "/run/trace.jsonl" and "--workers" not in small
    base = open_arm(tmp_path, "whispercpp-stream-base-t8").stdout.split("\n")
    assert base[3].endswith("models/ggml-base.en.bin") and base[7:9] == ["--threads", "8"]
    assert open_arm(tmp_path, "whispercpp-stream-tiny").stdout.split("\n")[3].endswith("models/ggml-tiny.en.bin")
    parakeet = open_arm(tmp_path, "parakeet-v3-onnxasr-int8-w8").stdout.split("\n")
    assert parakeet[2] == str(OPPONENTS / "onnxasr_whole.py") and parakeet[7:9] == ["--workers", "8"]
    anchor = open_arm(tmp_path, "jsr-d2e-16457-int8").stdout.split("\n")
    assert anchor[1] == "r1-engine-int8" and anchor[3:5] == ["/panel/wav", "/run/predictions.jsonl"]
    workers = open_arm(tmp_path, "jsr-d2e-16457-int8-w8").stdout.split("\n")
    assert workers[2] == str(OPPONENTS / "open_workers.py") and workers[3:7] == ["8", "/panel/wav", "/run/predictions.jsonl", "--"]
    assert workers[8] == "r1-engine-int8"
    zipformer = open_arm(tmp_path, "r0-zipformer-int8").stdout.split("\n")
    assert zipformer[0].endswith("/env") and zipformer[1].startswith("ULTRA_SHERPA_LIB=")
    assert zipformer[3] == "zipformer-int8" and zipformer[5:7] == ["/panel/wav", "/run/predictions.jsonl"]
    assert "zipformer-int8" in open_arm(tmp_path, "r0-zipformer-int8-w8").stdout.split("--\n")[1]
    a1 = open_arm(tmp_path, "jsr-a1-30000-ema-int8").stdout.split("\n")
    assert a1[0].endswith("ultra-cold-wav-457a4db7") and a1[2].endswith("opponents/models/A1-30000-ema/int8")
    for arm, folder in (("jsr-a1-49000-ema-int8", "A1-49000-ema"), ("jsr-f1-60000-ema-int8", "F1-60000-ema")):
        final = open_arm(tmp_path, arm).stdout.split("\n")
        assert final[0].endswith("ultra-cold-wav-457a4db7") and final[2].endswith(f"opponents/models/{folder}/int8")
    ours = open_arm(tmp_path, "jsr-h22ctl-79800-ema-int8").stdout.split("\n")
    assert ours[0].endswith("opponents/bench-open/anchor/ultra-cold-wav-457a4db7") and ours[1] == "r1-engine-int8"
    assert ours[2].endswith("opponents/models/H22ctl-79800-ema/int8") and ours[3] == "/panel/wav"
    shared = open_arm(tmp_path, "jsr-h22ctl-79800-ema-int8-s8").stdout.split("\n")
    assert shared[0].endswith("bench-open/anchor/ultra-cold-wav-workers") and shared[1:4] == ["--workers", "8", "r1-engine-int8"]
    assert shared[4].endswith("models/H22ctl-79800-ema/int8") and shared[5:7] == ["/panel/wav", "/run/predictions.jsonl"]
    for bad in ("vosk-small-t8", "jsr-d2e-16457-int8-t8", "r0-zipformer-int8-t8", "nonesuch", "vosk-small-s8"):
        assert open_arm(tmp_path, bad).returncode == 2


def test_run_cold_dispatches_the_open_arms_to_their_definitions():
    script = (REPO / "docs/paper/run-cold.sh").read_text()
    case = next(line for line in script.splitlines() if "parakeet-v3-onnxasr-int8" in line)
    assert set(case.strip().rstrip(")").split("|")) == {
        "vosk-large", "vosk-small", "whispercpp-stream-base", "whispercpp-stream-tiny",
        "parakeet-v3-onnxasr-int8", "jsr-h22ctl-79800-ema-int8", "jsr-a1-30000-ema-int8", "jsr-a1-49000-ema-int8",
        "jsr-f1-60000-ema-int8", "jsr-d2e-16457-int8", "r0-zipformer-int8"}
    assert 'source "$PWD/docs/paper/opponents/open-arms.sh"' in script


FAKE_RECOGNISER = """
import json, os, sys
kind, bundle, wav_dir, output = sys.argv[1:]
cpus = sorted(os.sched_getaffinity(0))
assert len(cpus) == 1
with open(output, "x") as out:
    for name in sorted(os.listdir(wav_dir)):
        assert not os.path.islink(os.path.join(wav_dir, name))
        out.write(json.dumps({"hyp": f"{kind} {cpus[0]}", "id": name[:-4]}) + "\\n")
"""


def test_worker_launcher_pins_each_shard_and_merges_the_parts(tmp_path):
    cpus = sorted(os.sched_getaffinity(0))
    workers = min(3, len(cpus))
    wavs = tmp_path / "wav"
    wavs.mkdir()
    for index in range(7):
        staged_wav(wavs / f"u{index:05d}.wav", 400 * (index + 1))
    fake = tmp_path / "fake.py"
    fake.write_text(FAKE_RECOGNISER)
    predictions = tmp_path / "out" / "predictions.jsonl"
    predictions.parent.mkdir()
    subprocess.run([sys.executable, str(OPPONENTS / "open_workers.py"), str(workers), str(wavs), str(predictions),
                    "--", sys.executable, str(fake), "kind", "bundle"], check=True)
    rows = [json.loads(line) for line in predictions.read_text().splitlines()]
    assert [row["id"] for row in rows] == [f"u{index:05d}" for index in range(7)]
    assert {row["hyp"] for row in rows} == {f"kind {cpu}" for cpu in cpus[:workers]}
    assert [path.name for path in predictions.parent.iterdir()] == ["predictions.jsonl"]


WORKLOAD = json.loads((REPO / "docs/paper/evidence.json").read_text())["workloads"]["cold-panel"]
INDEX = [{"id": "u00000", "uid": "1-1-0", "speaker": "1", "dataset": "dev-clean", "ref": "A B", "duration_s": 1.0},
         {"id": "u00001", "uid": "2-1-0", "speaker": "2", "dataset": "dev-other", "ref": "C", "duration_s": 2.0}]
P73_DIGEST = "12936656" + "0" * 56
P73_SECONDS = 597.089875
SCORE = {"dev-clean": {"wer": 0.05, "utterances": 365}, "dev-other": {"wer": 0.15, "utterances": 365}}


def fake_repetition(runs, group, arm, round_id, cpu_s, wall_s, service, hyps, rss=10**8, verdict="clean", wer=0.1):
    """A finished repetition as open-run.sh leaves it (only the fields the summary reads are real)."""
    run = runs / group / "cold" / f"round-{round_id}" / arm
    run.mkdir(parents=True)
    p73 = group.endswith("-p73")
    texts = {"arm": arm, "round": str(round_id), "host": "xbabe0", "started": f"2026-11-22T21:0{round_id}:00Z",
             "finished": f"2026-11-22T22:0{round_id}:00Z", "command": "cmd", "config.sha256": "0  /x",
             "runtime-libraries.sha256": "0  /lib", "host.before": "0.5 0.5 0.5 1/1 1\nschedutil\n1",
             "host.after": "1.0 0.5 0.5 1/1 1", "service.stderr": "", "artifact.bytes": "1000000",
             "trace.sha256": "0  trace.jsonl", "panel": "p73" if p73 else "full", "harness.commit": "c0e8fc1",
             "audio.seconds": str(P73_SECONDS if p73 else WORKLOAD["audio_seconds"]),
             "workload.sha256": f"{P73_DIGEST if p73 else WORKLOAD['digest']}  /panel/audio.json",
             "host.freq.tsv": "1\t4000000\t0.5 0.5 0.5\n2\t4200000\t0.6 0.5 0.5"}
    for name, text in texts.items():
        (run / name).write_text(text + "\n")
    placement = {"effective_service_cpus_at_spawn": service}
    (run / "measurement.json").write_text(json.dumps({
        "cpu_seconds_user": cpu_s, "cpu_seconds_sys": 0.0, "wall_seconds": wall_s, "peak_rss_bytes": rss,
        "max_threads": 1, "max_processes": 1, "exit_code": 0, "placement": placement}))
    (run / "host.cpu.json").write_text(json.dumps({
        "verdict": verdict, "other_active_cores": [31], "smt_siblings_busy": {"68": 0.001},
        "loadavg_before": ["0.5"], "loadavg_after": ["1.0"]}))
    (run / "host.gate.json").write_text("{}")
    common.write_predictions(run / "predictions.jsonl", hyps)
    if not p73:
        (run / "score.json").write_text(json.dumps({"system_a": {
            "utterances": WORKLOAD["utterances"], "pooled_wer": wer + round_id / 1000, "datasets": SCORE}}))
        with (run / "scores.jsonl").open("x") as out:
            for reference in INDEX:
                row = {key: value for key, value in reference.items() if key not in {"id", "uid"}}
                out.write(json.dumps({**row, "id": reference["uid"], "hyp": hyps[reference["id"]]}) + "\n")
    return run


def fake_wer_pass(root, arm, hyps):
    run = root / arm
    run.mkdir(parents=True)
    for name, text in {"command": "python runner --workers 24", "host": "xbabe3", "started": "t0", "finished": "t1",
                       "config.sha256": "0  /x", "workload.sha256": f"{WORKLOAD['digest']}  /panel/audio.json"}.items():
        (run / name).write_text(text + "\n")
    common.write_predictions(run / "predictions.jsonl", hyps)
    (run / "scores.jsonl").write_text("scorer input\n")
    (run / "score.json").write_text(json.dumps({"system_a": {
        "utterances": WORKLOAD["utterances"], "pooled_wer": 0.2, "datasets": SCORE}}))
    return run


def summary(*args):
    return subprocess.run([sys.executable, str(OPPONENTS / "open_summary.py"), *map(str, args)],
                          capture_output=True, text=True, check=True).stdout


def test_receipts_collect_compactly_expand_exactly_and_export_rows(tmp_path):
    runs, raw, panel = tmp_path / "runs", tmp_path / "raw", tmp_path / "panel"
    panel.mkdir()
    (panel / "index.json").write_text(json.dumps(INDEX))
    same = {"u00000": "a b", "u00001": "c"}
    other = {"u00000": "a", "u00001": "c"}
    audio = WORKLOAD["audio_seconds"]
    lanes = ((1, [4]), (2, [24]), (3, [28]))
    for (round_id, service), cpu in zip(lanes, (0.20, 0.10, 0.15)):  # full panel, cost and WER together
        fake_repetition(runs, "1c", "vosk-small", round_id, cpu * audio, cpu * audio, service, same, rss=round_id * 10**8)
    for round_id, service in lanes:  # panel73 timing; WER from the full-panel pass
        fake_repetition(runs, "1c-p73", "whispercpp-stream-tiny", round_id, 1.2 * P73_SECONDS, 1.2 * P73_SECONDS,
                        service, same if round_id < 3 else other)
    for arm, wall, rounds in (("whispercpp-stream-tiny-t8", 0.30, (1, 2, 3)), ("whispercpp-stream-tiny-w8", 0.20, (1, 2, 3)),
                              ("vosk-small-w8", 0.02, (1, 2, 3)), ("parakeet-v3-onnxasr-int8-t8", 0.05, (1, 2, 3)),
                              ("parakeet-v3-onnxasr-int8-w8", 0.04, (1,))):
        for round_id in rounds:
            fake_repetition(runs, "8c-p73", arm, round_id, 1.5 * P73_SECONDS, wall * P73_SECONDS, list(range(8, 16)), same)
    for round_id, (cpu, service) in enumerate(((0.035, [4]), (0.036, [24]), (0.0355, [28]), (0.040, [4])), 1):
        fake_repetition(runs, "1c", "jsr-h22ctl-79800-ema-int8", round_id, cpu * audio, cpu * audio, service, same,
                        wer=0.08)
    for round_id, service in ((1, [4]), (2, [24]), (3, [28])):
        fake_repetition(runs, "1c-p73", "jsr-h22ctl-79800-ema-int8", round_id, 0.036 * P73_SECONDS,
                        0.036 * P73_SECONDS, service, same)
    fake_repetition(runs, "1c", "vosk-large", 1, 1.0, 1.0, [4], same, verdict="contaminated")
    summary("collect", runs, raw)
    summary("collect-wer", fake_wer_pass(tmp_path / "pass", "whispercpp-stream-tiny", same), raw, "whispercpp-stream-tiny")
    small, tiny = raw / "vosk-small", raw / "whispercpp-stream-tiny"
    assert (small / "round-1/predictions.jsonl").is_file() and not (small / "round-2/predictions.jsonl").exists()
    assert (small / "round-2/predictions.jsonl.ref").read_text().split()[0] == "round-1"
    assert (tiny / "p73-round-2/predictions.jsonl.ref").read_text().split()[0] == "p73-round-1"
    assert (tiny / "p73-round-3/predictions.jsonl").is_file() and (tiny / "wer-full/score.json").is_file()
    assert (tiny / "wer-full/scores.jsonl.sha256").read_text().strip() == receipt.hashlib.sha256(b"scorer input\n").hexdigest()
    assert not list(raw.rglob("scores.jsonl")) and not (tiny / "p73-round-1/scores.jsonl.sha256").exists()
    assert sorted(path.name for path in (raw / "jsr-h22ctl-79800-ema-int8").iterdir()) == [
        "p73-round-1", "p73-round-2", "p73-round-3", "round-1", "round-2", "round-3", "round-4"]
    summary("expand", raw, panel, tmp_path / "full")
    for original in runs.rglob("predictions.jsonl"):
        rebuilt = tmp_path / "full" / original.relative_to(runs)
        assert rebuilt.read_bytes() == original.read_bytes()
        if (original.parent / "scores.jsonl").is_file():
            assert (rebuilt.parent / "scores.jsonl").read_bytes() == (original.parent / "scores.jsonl").read_bytes()
    rows = {row["id"]: row for row in map(json.loads, summary("rows", raw, "2026-11-22 xbabe0 test").splitlines())}
    assert set(rows) == {"vosk-small-zen2-t1", "vosk-small-zen2-t8", "whispercpp-stream-tiny-zen2-t1",
                         "whispercpp-stream-tiny-zen2-t8"}  # no parakeet: no full-panel WER source yet
    one = rows["vosk-small-zen2-t1"]
    assert one["cpu_s_per_audio_s"] == pytest.approx(0.15) and one["cpu_reps"] == pytest.approx([0.20, 0.10, 0.15])
    assert one["wer_pct"] == pytest.approx(10.3) and one["peak_rss_bytes"] == 3 * 10**8
    assert one["threads"] == 1 and one["parallel"] == "intra_op" and one["class"] == "streaming"
    assert one["arm"] == "vosk-small" and one["configuration"].startswith("vosk-model-small-en-us-0.15")
    assert one["repeated_hypotheses_identical"] and one["workload_digest"] == WORKLOAD["digest"]
    assert one["timing_panel"] == "full" and one["wer_source"].startswith("the timed repetitions")
    assert "CPUs 4; 24; 28" in one["hardware"] and one["session"] == "2026-11-22 xbabe0 test"
    assert one["n_busy_cores_reps"] == [2, 2, 2] and one["service_mhz"] == [4100, 4100, 4100]
    assert one["recogniser_s_per_audio_s"] is None and len(one["repetition_windows"]) == 3
    assert one["harness_commits"] == ["c0e8fc1"]  # each receipt's deployed harness commit reaches the row
    slow = rows["whispercpp-stream-tiny-zen2-t1"]
    assert slow["timing_panel"] == "p73" and slow["workload_digest"] == WORKLOAD["digest"]
    assert slow["timing_workload_digest"] == P73_DIGEST and slow["timing_audio_seconds"] == pytest.approx(P73_SECONDS)
    assert slow["cpu_s_per_audio_s"] == pytest.approx(1.2) and slow["wer_pct"] == pytest.approx(20.0)
    assert "full-panel pass on xbabe3" in slow["wer_source"]
    assert slow["shared_hypotheses_equal_full_pass"] == ["2/2", "2/2", "1/2"]
    assert not slow["repeated_hypotheses_identical"]
    eight = rows["whispercpp-stream-tiny-zen2-t8"]
    assert eight["parallel"] == "processes" and eight["wall_s_per_audio_s"] == pytest.approx(0.20)
    assert [(alt["parallel"], alt["reps"]) for alt in eight["alternatives"]] == [("intra_op", 3)]
    assert eight["wer_pct"] == pytest.approx(20.0)
    shared = rows["vosk-small-zen2-t8"]
    assert shared["parallel"] == "workers" and shared["timing_panel"] == "p73"
    assert shared["wer_pct"] == pytest.approx(10.3) and "one-core full-panel repetitions" in shared["wer_source"]
    assert shared["shared_hypotheses_equal_full_pass"] == ["2/2", "2/2", "2/2"]
    for round_id, service in lanes:  # now Parakeet has full-panel repetitions to take its WER from
        fake_repetition(runs, "1c", "parakeet-v3-onnxasr-int8", round_id, 0.15 * audio, 0.15 * audio, service, same)
    summary("collect", runs, raw)
    rows = {row["id"]: row for row in map(json.loads, summary("rows", raw, "session").splitlines())}
    parakeet = rows["parakeet-v3-onnxasr-int8-zen2-t8"]
    assert parakeet["parallel"] == "intra_op" and parakeet["class"] == "whole_utterance"
    assert [(alt["parallel"], alt["reps"]) for alt in parakeet["alternatives"]] == [("processes", 1)]
    anchors = {row["id"]: row for row in map(json.loads, summary("anchors", raw, "session").splitlines())}
    assert set(anchors) == {"h22ctl-79800-ema-int8-zen2-t1-anchor-bench-open-full",
                            "h22ctl-79800-ema-int8-zen2-t1-anchor-bench-open-p73"}
    full = anchors["h22ctl-79800-ema-int8-zen2-t1-anchor-bench-open-full"]
    assert full["anchor"] is True and full["role"] == "ours" and full["variant"] == "h22ctl-79800-ema-int8"
    assert full["cpu_reps"] == pytest.approx([0.035, 0.036, 0.0355, 0.040]) and len(full["anchor_holds"]) == 4
    assert full["cpu_s_per_audio_s"] == pytest.approx(0.03575) and full["wer_pct"] == pytest.approx(8.2, abs=0.11)
    short = anchors["h22ctl-79800-ema-int8-zen2-t1-anchor-bench-open-p73"]
    assert short["timing_panel"] == "p73" and short["timing_workload_digest"] == P73_DIGEST
    assert short["wer_pct"] == full["wer_pct"] and "full-panel anchor repetitions" in short["wer_source"]
    sys.path.insert(0, str(REPO / "python/ultra"))
    from tools.paper_data import schema  # the paper's own row validator accepts every row as written
    for row in list(rows.values()) + list(anchors.values()):
        schema.check_row(dict(row), row["id"])
    table = summary("table", raw)
    assert "| vosk-small | full | 2 | 24 | 0.1000 |" in table and "| vosk-small-w8 | p73 | 3 | 1.5000 |" in table


def test_arm_names_map_to_threads_and_parallel_modes():
    import open_summary
    assert open_summary.split_arm("vosk-small") == ("vosk-small", 1, "intra_op")
    assert open_summary.split_arm("vosk-small-w8") == ("vosk-small", 8, "workers")  # one process, one shared model
    assert open_summary.split_arm("parakeet-v3-onnxasr-int8-w8") == ("parakeet-v3-onnxasr-int8", 8, "processes")
    assert open_summary.split_arm("parakeet-v3-onnxasr-int8-t8") == ("parakeet-v3-onnxasr-int8", 8, "intra_op")
    assert open_summary.split_arm("jsr-h22ctl-79800-ema-int8-s8") == ("jsr-h22ctl-79800-ema-int8", 8, "workers")


def test_loaded_cross_checks_never_become_rows(tmp_path):
    runs, raw = tmp_path / "runs", tmp_path / "raw"
    audio = WORKLOAD["audio_seconds"]
    for round_id, service in ((1, [4]), (2, [24]), (3, [28])):
        fake_repetition(runs, "1c", "vosk-small", round_id, 0.3 * audio, 0.3 * audio, service, {"u00000": "a", "u00001": "c"})
    for legacy in runs.rglob("panel"):  # receipts written before protocol v3 have no panel file
        if legacy.is_file():
            legacy.unlink()
    for legacy in runs.rglob("audio.seconds"):
        legacy.unlink()
    summary("collect", runs, raw / "loaded-2118Z")
    assert summary("rows", raw, "session") == ""
    assert "| vosk-small | full | 3 | 0.3000 |" in summary("table", raw / "loaded-2118Z")
