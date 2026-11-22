"""Incremental remote train.jsonl tailer over persistent SSH ControlMaster."""

from __future__ import annotations

import asyncio
import json
import re

REMOTE_TAIL_PY = r"""
import json, os, re, subprocess
offs = json.loads(os.environ.get("DASH_OFFS", "{}"))
ps = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
found = {}
for line in ps.splitlines():
    if "train.trainer" not in line or "--run-dir" not in line:
        continue
    m = re.search(r"--run-dir\s+(\S+)", line)
    if not m:
        continue
    rdir = m.group(1)
    ts = re.search(r"total_steps=(\d+)", line)
    found[rdir] = int(ts.group(1)) if ts else None
out = []
for rdir, total in found.items():
    p = os.path.join(rdir, "train.jsonl")
    if not os.path.isfile(p):
        continue
    sz = os.path.getsize(p)
    off = int(offs.get(p, 0))
    if sz < off:
        off = 0
    lines = []
    new_off = off
    if sz > off:
        with open(p, "rb") as fh:
            fh.seek(off)
            chunk = fh.read()
        cut = chunk.rfind(b"\n")
        if cut >= 0:
            new_off = off + cut + 1
            lines = chunk[:cut].decode("utf-8", "replace").splitlines()
    out.append({"run": os.path.basename(rdir.rstrip("/")), "path": p, "total": total, "off": new_off, "lines": lines})
print(json.dumps(out, separators=(",", ":")))
"""


def parse_remote_payload(raw: str) -> list[dict]:
    """Parse JSON list emitted by `REMOTE_TAIL_PY`, returning [] on malformed output."""
    raw = raw.strip()
    if not raw:
        return []
    try:
        data = json.loads(raw)
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []


def infer_total_steps_from_cmd(cmdline: str) -> int | None:
    m = re.search(r"total_steps=(\d+)", cmdline)
    return int(m.group(1)) if m else None


async def tail_remote_node(node: str, ssh_opts: list[str], offsets: dict[str, int]) -> list[dict]:
    """Fetch only new train.jsonl bytes for every live train.trainer process on `node`."""
    node_offs = {k.split(":", 1)[1]: v for k, v in offsets.items() if k.startswith(f"{node}:")}
    env_json = json.dumps(node_offs, separators=(",", ":"))
    remote_cmd = f"DASH_OFFS={json.dumps(env_json)} python3 -"
    cmd = ["ssh", *ssh_opts, node, remote_cmd]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        out, _ = await asyncio.wait_for(proc.communicate(REMOTE_TAIL_PY.encode()), timeout=8)
    except (asyncio.TimeoutError, OSError):
        return []
    return parse_remote_payload(out.decode("utf-8", "replace"))
