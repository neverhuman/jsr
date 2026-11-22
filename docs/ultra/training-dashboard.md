# Live fleet training dashboard

From the repository root, start the read-only dashboard with the trainer
environment created by `just venv`. A single local probe works without fleet
SSH access.

```sh
PYTHONPATH=python/ultra .venv/bin/python -m tools.training_dashboard
```

For a local run, pass `--hosts "$(hostname -s)" --run-dir /path/to/run`.

Open [the local dashboard](http://127.0.0.1:8766/). From another machine,
forward the port and open the same address in your browser.

```sh
ssh -N -L 8766:127.0.0.1:8766 xbabe3
```

The standalone [HTML page](../../training-dashboard.html) contains its scripts,
styles, and charts. To open the file directly, append
`?ws=ws://127.0.0.1:8766/ws` to its file URL and start the server with
`--allow-origin null`. A reverse proxy can serve the page and WebSocket
together. Supply its exact browser origin with `--allow-origin`.

## Fleet and piped input

The default host list is xbabe0 through xbabe9. Override it with
`--hosts xbabe1 xbabe3 xbabe5`. The server runs a small Python probe directly
on the local host and streams the same probe through SSH to the other hosts.
The SSH aliases use existing key-based access. Probes discover active training
processes and read their local log files. No node needs a dashboard deployment.

To pipe collector JSON lines into the same WebSocket service, select a node
with `--stdin`. This example streams xbabe5 into an otherwise independent
dashboard on port 8767.

```sh
ssh -T xbabe5 python3 -u - < python/ultra/tools/training_dashboard_probe.py |
  PYTHONPATH=python/ultra .venv/bin/python -m tools.training_dashboard \
    --hosts xbabe5 --stdin xbabe5 --port 8767
```

The HTTP server defaults to loopback. The port and address can be set with
`--port` and `--bind`. Use `--local-host` to map a fleet alias to the machine
running the server, or `--run-dir` to add a local log directory explicitly.

## Reading the metrics

- **Audio/s** is the block's audio seconds divided by elapsed wall seconds.
  Consecutive trainer timestamps include time spent evaluating when an
  evaluation resets the block timer. The fleet rate sums the latest fresh
  blocks across active runs. A node can contribute more than one run.
- **WER** updates when the trainer writes an evaluation. The fleet's best
  retained rows are grouped by DEV split, decoder, evaluation population and
  reference-word count. The display identifies the run and step. The values
  come from trainer evaluations.
- **Loss** is the latest logged training objective. The fleet median is a
  summary across fresh runs. Use each run's history to compare its trend.
- **DDP** workers follow the fresh step, loss, and WER from their live rank 0.
  Select a worker to see its local GPU telemetry alongside the shared run.
  The fleet counts the shared run's throughput once. A live worker whose
  rank 0 has stopped reporting progress shows as stale.
- **Stale** means the probe is connected and an active trainer's block is
  overdue. **Offline** means the probe connection is down. Its retained
  history is still visible and its rate is excluded from the fleet sum.

The WebSocket sends an initial snapshot, node updates, and clock ticks. A
connected browser can switch metrics and select runs without downloading
external assets. Rosters and training histories reflect the probe's bounded
recent data.
