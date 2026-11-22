"""The open-opponent runners and receipts (docs/paper/opponents): packet feed, traces, shards, guard.

The recognisers themselves are not installed here; they are replaced by small fakes, so these
tests pin the harness behaviour around them.
"""

import json
import os
from pathlib import Path
import subprocess
import sys
import wave

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
OPPONENTS = REPO / "docs/paper/opponents"
sys.path.insert(0, str(OPPONENTS))

import open_common as common  # noqa: E402
import open_receipt as receipt  # noqa: E402
import open_schedule as schedule  # noqa: E402
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


def test_plan_runs_phase_by_phase_and_one_job_per_cpu_set():
    jobs = schedule.read_plan("p1 a 1 4 0\np1 b 1 4 0\np1 a 2 24 1  # second lane\n\np2 c 1 8-15 3\n")

    def ready():
        return [(job["arm"], job["round"]) for job in schedule.next_jobs(jobs)]

    assert ready() == [("a", 1), ("a", 2)]
    jobs[0]["status"] = "running"
    assert ready() == [("a", 2)]
    jobs[0]["status"], jobs[2]["status"] = "done", "running"
    assert ready() == [("b", 1)]
    jobs[1]["status"], jobs[2]["status"] = "failed", "done"
    assert ready() == [("c", 1)]
    with pytest.raises(ValueError):
        schedule.read_plan("p1 a 1 4")


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
    for bad in ("vosk-small-t8", "jsr-d2e-16457-int8-t8", "r0-zipformer-int8-t8", "nonesuch"):
        assert open_arm(tmp_path, bad).returncode == 2


def test_run_cold_dispatches_the_open_arms_to_their_definitions():
    script = (REPO / "docs/paper/run-cold.sh").read_text()
    case = next(line for line in script.splitlines() if "parakeet-v3-onnxasr-int8" in line)
    assert set(case.strip().rstrip(")").split("|")) == {
        "vosk-large", "vosk-small", "whispercpp-stream-base", "whispercpp-stream-tiny",
        "parakeet-v3-onnxasr-int8", "jsr-d2e-16457-int8", "r0-zipformer-int8"}
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


def fake_repetition(runs, group, arm, round_id, cpu_s, wall_s, service, hyps, rss=10**8, verdict="clean"):
    """A finished repetition as open-run.sh leaves it (only the fields the summary reads are real)."""
    run = runs / group / "cold" / f"round-{round_id}" / arm
    run.mkdir(parents=True)
    texts = {"arm": arm, "round": str(round_id), "host": "xbabe0", "started": f"2026-11-22T21:0{round_id}:00Z",
             "finished": f"2026-11-22T22:0{round_id}:00Z", "command": "cmd", "config.sha256": "0  /x",
             "runtime-libraries.sha256": "0  /lib", "host.before": "0.5 0.5 0.5 1/1 1\nschedutil\n1",
             "host.after": "1.0 0.5 0.5 1/1 1", "service.stderr": "", "artifact.bytes": "1000000",
             "trace.sha256": "0  trace.jsonl", "workload.sha256": f"{WORKLOAD['digest']}  /panel/audio.json",
             "host.freq.tsv": "1\t4000000\t0.5 0.5 0.5\n2\t4200000\t0.6 0.5 0.5"}
    for name, text in texts.items():
        (run / name).write_text(text + "\n")
    placement = {"effective_service_cpus_at_spawn": service}
    (run / "measurement.json").write_text(json.dumps({
        "cpu_seconds_user": cpu_s, "cpu_seconds_sys": 0.0, "wall_seconds": wall_s, "peak_rss_bytes": rss,
        "max_threads": 1, "max_processes": 1, "exit_code": 0, "placement": placement}))
    dataset = {"wer": 0.05 + round_id / 1000, "utterances": 365}
    (run / "score.json").write_text(json.dumps({"system_a": {
        "utterances": WORKLOAD["utterances"], "pooled_wer": 0.1 + round_id / 1000,
        "datasets": {"dev-clean": dataset, "dev-other": dataset}}}))
    (run / "host.cpu.json").write_text(json.dumps({
        "verdict": verdict, "other_active_cores": [8, 9], "smt_siblings_busy": {"68": 0.001},
        "loadavg_before": ["0.5"], "loadavg_after": ["1.0"]}))
    (run / "host.gate.json").write_text("{}")
    common.write_predictions(run / "predictions.jsonl", hyps)
    with (run / "scores.jsonl").open("x") as out:
        for reference in INDEX:
            row = {key: value for key, value in reference.items() if key not in {"id", "uid"}}
            out.write(json.dumps({**row, "id": reference["uid"], "hyp": hyps[reference["id"]]}) + "\n")
    return run


def summary(*args):
    return subprocess.run([sys.executable, str(OPPONENTS / "open_summary.py"), *map(str, args)],
                          capture_output=True, text=True, check=True).stdout


def test_receipts_collect_compactly_expand_exactly_and_export_the_median_row(tmp_path):
    runs, raw, panel = tmp_path / "runs", tmp_path / "raw", tmp_path / "panel"
    panel.mkdir()
    (panel / "index.json").write_text(json.dumps(INDEX))
    same = {"u00000": "a b", "u00001": "c"}
    audio = WORKLOAD["audio_seconds"]
    for round_id, cpu, service in ((1, 0.20, [4]), (2, 0.10, [24]), (3, 0.15, [28])):
        fake_repetition(runs, "1c", "vosk-small", round_id, cpu * audio, cpu * audio, service, same, rss=round_id * 10**8)
    for round_id in (1, 2, 3):
        hyps = same if round_id < 3 else {"u00000": "a", "u00001": "c"}
        fake_repetition(runs, "8c", "vosk-small-w8", round_id, 0.2 * audio, 0.03 * audio, list(range(8, 16)), hyps)
    for arm, wall in (("whispercpp-stream-tiny-t8", 0.30), ("whispercpp-stream-tiny-w8", 0.20)):
        for round_id in (1, 2, 3):
            fake_repetition(runs, "8c", arm, round_id, 1.5 * audio, wall * audio, list(range(8, 16)), same)
    for round_id, service in ((1, [4]), (2, [24]), (3, [28])):
        fake_repetition(runs, "1c", "jsr-d2e-16457-int8", round_id, 0.035 * audio, 0.035 * audio, service, same)
    fake_repetition(runs, "1c", "vosk-large", 1, 1.0, 1.0, [4], same, verdict="contaminated")
    summary("collect", runs, raw)
    assert not (raw / "vosk-large").exists()
    small = raw / "vosk-small"
    assert (small / "round-1/predictions.jsonl").is_file() and not (small / "round-2/predictions.jsonl").exists()
    assert (small / "round-2/predictions.jsonl.ref").read_text().split()[0] == "round-1"
    assert (raw / "vosk-small-w8/round-3/predictions.jsonl").is_file()
    assert not list(raw.rglob("scores.jsonl")) and len(list(raw.rglob("scores.jsonl.sha256"))) == 15
    summary("expand", raw, panel, tmp_path / "full")
    for original in runs.rglob("scores.jsonl"):
        rebuilt = tmp_path / "full" / original.relative_to(runs)
        if "vosk-large" not in str(original):
            assert rebuilt.read_bytes() == original.read_bytes()
            assert (rebuilt.parent / "predictions.jsonl").read_bytes() == (original.parent / "predictions.jsonl").read_bytes()
    rows = {row["id"]: row for row in map(json.loads, summary("rows", raw, "2026-11-22 xbabe0 test").splitlines())}
    assert set(rows) == {"vosk-small-zen2-t1", "vosk-small-zen2-t8", "whispercpp-stream-tiny-zen2-t8"}
    one = rows["vosk-small-zen2-t1"]
    assert one["cpu_s_per_audio_s"] == pytest.approx(0.15) and one["cpu_reps"] == pytest.approx([0.20, 0.10, 0.15])
    assert one["wer_pct"] == pytest.approx(10.3) and one["peak_rss_bytes"] == 3 * 10**8
    assert one["threads"] == 1 and one["parallel"] == "intra_op" and one["class"] == "streaming"
    assert one["arm"] == "vosk-small" and one["configuration"].startswith("vosk-model-small-en-us-0.15")
    assert one["repeated_hypotheses_identical"] and one["workload_digest"] == WORKLOAD["digest"]
    assert "CPUs 4; 24; 28" in one["hardware"] and one["session"] == "2026-11-22 xbabe0 test"
    assert one["n_busy_cores"] == [3, 3, 3] and one["service_mhz"] == [4100, 4100, 4100]
    assert one["recogniser_s_per_audio_s"] is None
    assert not rows["vosk-small-zen2-t8"]["repeated_hypotheses_identical"]
    tiny = rows["whispercpp-stream-tiny-zen2-t8"]
    assert tiny["parallel"] == "workers" and tiny["wall_s_per_audio_s"] == pytest.approx(0.20)
    assert [other["parallel"] for other in tiny["alternatives"]] == ["intra_op"]
    anchors = [json.loads(line) for line in summary("anchors", raw, "session").splitlines()]
    assert [(row["id"], row["role"]) for row in anchors] == [("jsr-d2e-16457-int8-zen2-t1", "ours")]
    assert anchors[0]["cpu_s_per_audio_s"] == pytest.approx(0.035)
    table = summary("table", raw)
    assert "| vosk-small | 2 | 24 | 0.1000 |" in table and "| vosk-small-w8 | 3 | 0.2000 |" in table
