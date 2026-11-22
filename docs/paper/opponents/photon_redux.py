"""Cold whole-utterance service around Photon for moondream/parakeet-redux.

    python photon_redux.py WAV_DIR PREDICTIONS.jsonl THREADS

One model load per process, the staged u%05d.wav files in name order, one utterance at a
time through the model card's own call (`speech.transcribe(audio=path)`), one final
hypothesis per file as `{"id", "hyp"}`. THREADS sizes Photon's torch and native kernel
pools (`cpu_threads`); the caller pins the cores. Weights come from the Hugging Face cache
under HF_HOME; timed runs set HF_HUB_OFFLINE=1 so nothing is fetched inside the meter.
"""

import argparse
import json
import os
from pathlib import Path
import re
import sys


def staged(wav_dir):
    wavs = sorted(path for path in wav_dir.iterdir())
    if not wavs or any(not re.fullmatch(r"u\d{5}\.wav", path.name) or not path.is_file() for path in wavs):
        raise ValueError(f"unexpected staged input in {wav_dir}")
    return wavs


def runtime_facts(threads):
    """Versions and the kernel paths Photon selected, for the receipt."""
    from importlib.metadata import version

    facts = {"runtime": "photon", "threads": threads}
    for package in ("moondream", "kestrel", "kestrel-kernels", "kestrel-native", "torch"):
        try:
            facts[package] = version(package)
        except Exception:  # noqa: BLE001 - a missing optional package is itself a fact
            facts[package] = None
    try:
        from kestrel_kernels import ternary
        from kestrel_kernels.cpu import conformer

        facts["ternary_gemm_isa"] = ternary.ternary_gemm_isa()
        facts["ternary_worker_threads"] = ternary.worker_threads()
        facts["conformer_isa"] = conformer.active_isa()
        facts["pool_placement"] = conformer.pool_placement()
    except Exception as error:  # noqa: BLE001 - introspection must never fail the run
        facts["kernel_introspection_error"] = repr(error)
    try:
        import torch

        facts["torch_threads"] = torch.get_num_threads()
    except Exception as error:  # noqa: BLE001
        facts["torch_introspection_error"] = repr(error)
    return facts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wav_dir", type=Path)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("threads", type=int)
    parser.add_argument("--model", default="moondream/parakeet-redux")
    args = parser.parse_args()
    if args.threads < 1:
        raise ValueError("THREADS must be at least 1")
    wavs = staged(args.wav_dir)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = str(args.threads)

    import moondream as md

    with args.predictions.open("x", encoding="utf-8") as out, \
            md.photon(args.model, device="cpu", cpu_threads=args.threads) as speech:
        for wav in wavs:
            result = speech.transcribe(audio=str(wav))
            json.dump({"id": wav.stem, "hyp": result["text"]}, out, ensure_ascii=False)
            out.write("\n")
        print(json.dumps(runtime_facts(args.threads)), file=sys.stderr)


if __name__ == "__main__":
    main()
