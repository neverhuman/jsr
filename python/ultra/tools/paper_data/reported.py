"""Numbers published by others (data/reported.json): macros for the "reported elsewhere" table only.

They are records with the label `reported`. No reported value can enter an evidence row, cold_dev.csv
or a figure: this module writes nothing but metrics.tex macros.
"""

from __future__ import annotations

from pathlib import Path
import re

from .common import DATA, PaperDataError, read_json
from .records import Records


def add_reported(root: Path, rec: Records) -> None:
    rel = DATA / "reported.json"
    if not (root / rel).is_file():
        return
    doc = read_json(root, str(rel))
    for rid in sorted(doc["values"]):
        v = doc["values"][rid]
        src = doc["sources"].get(v["source"])
        if src is None:
            raise PaperDataError(f"{rel}: {rid} names the unknown source {v['source']!r}")
        if not rid.startswith("Rep"):
            raise PaperDataError(f"{rel}: {rid}: a reported value's macro name starts with Rep")
        if not re.search(r"(?<![\d.])" + re.escape(v["text"]) + r"(?!\d)", v["quote"].replace(",", "")):
            raise PaperDataError(f"{rel}: {rid}: {v['text']} does not appear in its quoted source line")
        rec.add(rid, v["text"], raw=float(v["text"]), unit=v["unit"], label="reported",
                configuration=f"published by the source, not measured here: {v['what']}",
                workload="as published by the source", workload_digest="n/a (published)",
                source=(f"{src['title']}, {src['url']} (revision {src['revision']}; retrieved {src['retrieved']}; "
                        f"sha256 of the fetched file {src['sha256'][:12]}); bibliography key {src['cite']}"),
                quote=v["quote"])
    extra = doc.get("derived", {})
    for rid in sorted(extra):
        e = extra[rid]
        base = doc["values"][e["from"]]
        value = float(base["text"]) * float(e["factor"])
        rec.add(rid, f"{value:.{int(e['decimals'])}f}", raw=value, unit=e["unit"], label="reported",
                configuration=f"unit conversion of {e['from']} (x {e['factor']}): {e['what']}",
                workload="as published by the source", workload_digest="n/a (published)",
                source=f"docs/paper/data/reported.json values.{e['from']}", quote=base["quote"])
