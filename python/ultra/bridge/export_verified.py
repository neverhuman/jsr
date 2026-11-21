"""Re-export through the delivered A2 API, retaining verifiable export-source provenance.

This does not reconstruct the source used for the original training run.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import runpy
import subprocess
import sys

REPO = Path(__file__).absolute().parents[3]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git(*args):
    return subprocess.check_output(["rtk", "proxy", "git", *args], cwd=REPO)


def main():
    job = json.loads(Path(sys.argv[1]).read_text())
    out = Path(job["out"])
    if out.exists():
        raise FileExistsError("never overwrite an exported bundle")
    export_format = job.get("export_format", "onnx")
    if export_format not in ("onnx", "fused-fp32"):
        raise ValueError("unsupported export_format")
    source_head = git("rev-parse", "HEAD").decode().strip()
    sources = {}
    source_root = REPO / ("python/ultra" if export_format == "fused-fp32" else "python/ultra/train")
    for path in sorted(source_root.rglob("*.py")):
        rel = str(path.relative_to(REPO))
        actual = path.read_bytes()
        if actual != git("show", source_head + ":" + rel):
            raise ValueError("export source differs from immutable revision: " + rel)
        sources[rel] = hashlib.sha256(actual).hexdigest()
    sys.path.insert(0, str(REPO / "python/ultra"))
    if export_format == "fused-fp32":
        converter = runpy.run_path(str(REPO / "python/ultra/tools/r1_to_safetensors.py"),
                                   run_name="verified_fused_converter")
        manifest = converter["convert"](job["checkpoint"], str(out), job["source_bundle"])
        revision = manifest["converter"]["commit"]
        checkpoint_sha256 = manifest["source"]["checkpoint_sha256"]
        manifest_name = "arch.json"
    else:
        from train.export import export_bundle

        manifest = export_bundle(job["checkpoint"], str(out), int8=True)
        revision = manifest["model"]["exporter_commit"]
        checkpoint_sha256 = manifest["model"]["checkpoint_sha256"]
        manifest_name = "manifest.json"
    for rel, expected in sources.items():
        committed = hashlib.sha256(git("show", revision + ":" + rel)).hexdigest()
        if digest(REPO / rel) != expected or committed != expected:
            raise ValueError("export source changed during execution: " + rel)
    for entry in manifest["files"].values():
        if digest(out / entry["path"]) != entry["sha256"]:
            raise ValueError("export payload changed: " + entry["path"])
    import numpy
    import onnx
    import onnxruntime
    import torch

    provenance = dict(schema="ultra.export-source-provenance/v1", scope="export only",
        export_format=export_format, manifest_file=manifest_name,
        source_head_at_start=source_head, exporter_commit=revision, source_sha256=sources,
        verification="source bytes captured before import; unchanged afterward; match recorded immutable commit",
        checkpoint_sha256=checkpoint_sha256,
        manifest_sha256=digest(out / manifest_name),
        orchestrator_sha256=digest(__file__),
        software=dict(numpy=numpy.__version__, onnx=onnx.__version__,
                      onnxruntime=onnxruntime.__version__, torch=torch.__version__),
        training_source="not reconstructed; original launch provenance remains with trainer records")
    (out / "source-provenance.json").write_text(json.dumps(provenance, indent=2))
    print(json.dumps(dict(bundle=str(out), manifest_sha256=provenance["manifest_sha256"],
                          provenance_sha256=digest(out / "source-provenance.json"))))


if __name__ == "__main__":
    main()
