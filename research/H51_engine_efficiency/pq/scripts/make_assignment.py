"""Build a joint-run layer regex from a per-layer sensitivity table.

usage: python -I make_assignment.py <table.json> <max_dWER_pp> [--by distortion --max-rel-mse X]
prints: the regex (for engine_sens --layers group:<re>) and the covered INT8-byte share.
"""
from __future__ import annotations

import json
import re
import sys

tab = json.load(open(sys.argv[1]))
th = float(sys.argv[2])
by_dist = "--by" in sys.argv and sys.argv[sys.argv.index("--by") + 1] == "distortion"
max_mse = float(sys.argv[sys.argv.index("--max-rel-mse") + 1]) if "--max-rel-mse" in sys.argv else None
sel = []
for x in tab:
    if by_dist:
        if x["rel_mse_pq_exact"] is not None and x["rel_mse_pq_exact"] <= max_mse:
            sel.append(x)
    elif x["d_pooled_pp"] <= th:
        sel.append(x)
tot = sum(x["int8_bytes"] for x in tab)
cov = sum(x["int8_bytes"] for x in sel)
rx = "^(" + "|".join(re.escape(x["layer"]) for x in sel) + ")$"
print(json.dumps({"layers": len(sel), "of": len(tab), "int8_byte_share": round(cov / tot, 4),
                  "sum_single_dWER_pp": round(sum(x["d_pooled_pp"] for x in sel), 3)}))
print(rx)
