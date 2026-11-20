"""Nemotron adapter/merger regressions (codex-review N1)."""
import json
import os
import sys
import wave

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "adapters"))
from nemotron_adapter import write_wavs  # noqa: E402
from nemotron_merge import MergeError, merge  # noqa: E402


def _read(p):
    with wave.open(p) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)


def test_equal_length_different_pcm_is_rewritten(tmp_path):
    a = np.arange(1600, dtype=np.int16)
    b = a[::-1].copy()  # same length, different content
    (pa,), ida = write_wavs([a], str(tmp_path))
    assert np.array_equal(_read(pa), a)
    (pb,), idb = write_wavs([b], str(tmp_path))
    assert np.array_equal(_read(pb), b)
    assert ida != idb


def test_stale_extra_files_removed_and_identity_stable(tmp_path):
    a, b = np.zeros(160, np.int16), np.ones(320, np.int16)
    _, id1 = write_wavs([a, b], str(tmp_path))
    _, id2 = write_wavs([a, b], str(tmp_path))
    assert id1 == id2
    write_wavs([a], str(tmp_path))
    assert sorted(f for f in os.listdir(tmp_path) if f.endswith(".wav")) == ["u00000.wav"]


def _line(tmp_path, shard, uids, tag="t"):
    i, k = shard.split("/")
    p = tmp_path / f"s{i}of{k}_{tag}.utts.jsonl"
    p.write_text("".join(json.dumps({"id": u, "speaker": u.split("-")[0], "dataset": "x", "ref": "A B",
                                     "hyp": "A B", "duration_s": 1.0}) + "\n" for u in uids))
    return dict(run_id=f"x_{shard}_{tag}", tag=tag, split="x", chunk_ms=160, concurrency=1, threads=1,
                shard=shard, runtime_commit="r", harness_commit="h", argv=["--model", "m"],
                partition="DEV", honesty_label="unknown overlap", utterances=len(uids), audio_seconds=1.0 * len(uids),
                wer={"wer": 0, "sub": 0, "del": 0, "ins": 0, "ref_words": 2 * len(uids)},
                truncation_probe={"utts_last_ref_word_missing_from_hyp_tail": 0},
                process_cpu_seconds=1.0, process_wall_seconds=1.0, peak_rss_mib=1.0,
                bench_run={}, affinity=[16], gpu_pids_seen=[], per_utterance=str(p))


def test_wrong_shard_set_is_not_complete(tmp_path):
    with pytest.raises(MergeError):
        merge([_line(tmp_path, "0/2", ["1-1-1"]), _line(tmp_path, "3/2", ["2-1-1"])])
    out, _ = merge([_line(tmp_path, "0/2", ["1-1-1"])])
    assert out[0]["complete"] is False and out[0]["missing_shards"] == ["1/2"]


def test_duplicate_shard_and_overlapping_ids_rejected(tmp_path):
    with pytest.raises(MergeError):
        merge([_line(tmp_path, "0/2", ["1-1-1"]), _line(tmp_path, "0/2", ["1-1-2"], tag="t")])
    with pytest.raises(MergeError):
        merge([_line(tmp_path, "0/2", ["1-1-1"]), _line(tmp_path, "1/2", ["1-1-1"])])


def test_complete_group_pools_and_exports_score_records(tmp_path):
    out, recs = merge([_line(tmp_path, "0/2", ["1-1-1"]), _line(tmp_path, "1/2", ["2-1-1"])])
    assert out[0]["complete"] and out[0]["utterances"] == 2
    (rows,) = recs.values()
    assert [r["speaker"] for r in rows] == ["1", "2"]
    assert set(rows[0]) == {"id", "speaker", "dataset", "ref", "hyp", "duration_s"}
