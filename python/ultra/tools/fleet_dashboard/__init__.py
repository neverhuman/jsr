"""Real-time fleet dashboard: training loss / WER / throughput per run, GPU and CPU per node.

Run:  just dashboard            (serves http://127.0.0.1:8765, WebSocket at /ws)
View: ssh -L 8765:127.0.0.1:8765 <this-host>   then open http://localhost:8765
"""

from tools.fleet_dashboard.parse import parse_line, RunState  # noqa: F401
