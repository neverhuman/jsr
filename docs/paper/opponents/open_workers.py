"""Run N copies of a WAV-directory recogniser side by side, one balanced shard and one CPU each.

    open_workers.py N WAV_DIR PREDICTIONS_JSONL -- COMMAND...

Used for recognisers that are single-threaded per stream and take a staged WAV directory (our
engine's `ultra-cold-wav`). Worker K runs `COMMAND <shard K directory> <predictions part K>`
pinned to the K-th CPU of the affinity mask this launcher was started with. Shards hold hard
links to the staged WAVs and are balanced by audio length. The parts are merged into
PREDICTIONS_JSONL; the launcher fails if any worker fails.
"""

import functools
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import open_common as common  # noqa: E402


def main():
    if len(sys.argv) < 6 or sys.argv[4] != "--":
        sys.exit(__doc__)
    workers, wav_dir, predictions, command = int(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[5:]
    paths = common.staged_wavs(wav_dir)
    shards = common.balanced_shards([common.wav_samples(path) for path in paths], workers)
    cpus = sorted(os.sched_getaffinity(0))
    if len(cpus) < workers:
        sys.exit(f"{workers} workers need {workers} CPUs in the affinity mask, have {cpus}")
    staging = predictions.with_name(predictions.name + ".shards")
    staging.mkdir()
    children = []
    for index, shard in enumerate(shards):
        directory = staging / f"w{index}"
        directory.mkdir()
        for item in shard:
            os.link(paths[item], directory / paths[item].name)
        pin = functools.partial(os.sched_setaffinity, 0, {cpus[index]})
        children.append(subprocess.Popen([*command, str(directory), str(common.worker_path(predictions, index))],
                                         preexec_fn=pin))
    failed = sum(child.wait() != 0 for child in children)
    if failed:
        sys.exit(f"{failed} worker(s) failed")
    hypotheses = {}
    for index in range(workers):
        part = common.worker_path(predictions, index)
        hypotheses.update(common.read_predictions(part))
        part.unlink()
        for link in (staging / f"w{index}").iterdir():
            link.unlink()
        (staging / f"w{index}").rmdir()
    staging.rmdir()
    if len(hypotheses) != len(paths):
        sys.exit("workers returned a different utterance population")
    common.write_predictions(predictions, hypotheses)


if __name__ == "__main__":
    main()
