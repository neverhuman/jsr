"""Hardware CPU inference wattage and energy profiler (AMD Zen 2 MSR CorWatt + RAPL PkgWatt).

Implements two complementary protocols:
  1. Single-core MSR energy (`CorWatt` via `turbostat` on one pinned physical core with
     pre/post idle bracketing), completely isolated from background activity on other cores.
  2. Multi-core saturated package delta (`PkgWatt` + `/sys/class/powercap/intel-rapl:0/energy_uj`
     across 8 pinned physical cores with pre/post idle bracketing), capturing L3/uncore/DRAM power.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time


RAPL_PKG = Path("/sys/class/powercap/intel-rapl:0/energy_uj")
RAPL_MAX = Path("/sys/class/powercap/intel-rapl:0/max_energy_range_uj")


def parse_turbostat_rows(text: str, target_cpus: set[int] | None = None) -> list[dict]:
    """Parse `turbostat --quiet --show Core,CPU,Busy%,Bzy_MHz,CorWatt,PkgWatt` output."""
    rows: list[dict] = []
    headers: list[str] | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if parts[0] == "Core" and "CorWatt" in parts:
            headers = parts
            continue
        if headers is None or len(parts) < 5:
            continue
        if parts[0] == "-" and parts[1] == "-":
            # Aggregate summary row (present on multi-core runs)
            rec = {"summary": True}
            for k, v in zip(headers, parts):
                if k in ("Busy%", "Bzy_MHz", "CorWatt", "PkgWatt"):
                    try:
                        rec[k] = float(v)
                    except ValueError:
                        pass
            rows.append(rec)
            continue
        try:
            core = int(parts[0])
            cpu = int(parts[1])
        except ValueError:
            continue
        if target_cpus is not None and cpu not in target_cpus:
            continue
        rec = {"summary": False, "Core": core, "CPU": cpu}
        for k, v in zip(headers[2:], parts[2:]):
            try:
                rec[k] = float(v)
            except ValueError:
                pass
        if "CorWatt" in rec:
            rows.append(rec)
    return rows


def rapl_delta_joules(before_uj: int, after_uj: int, max_range_uj: int | None = None) -> float:
    delta = after_uj - before_uj
    if delta < 0 and max_range_uj:
        delta += max_range_uj
    return max(0.0, delta / 1_000_000.0)


def summarize_core_power(
    *,
    name: str,
    mode: str,
    audio_seconds: float,
    wall_seconds: float,
    pre_idle_watts: list[float],
    active_watts: list[float],
    post_idle_watts: list[float],
    active_mhz: list[float],
    busy_pcts: list[float],
    pkg_watts: list[float] | None = None,
) -> dict:
    idle_all = pre_idle_watts + post_idle_watts
    idle_w = statistics.median(idle_all) if idle_all else 0.0
    pre_w = statistics.median(pre_idle_watts) if pre_idle_watts else None
    post_w = statistics.median(post_idle_watts) if post_idle_watts else None
    # Filter to genuinely active samples (Busy% >= 50%) when available to exclude ramp/tail edges
    if active_watts and busy_pcts and len(active_watts) == len(busy_pcts):
        steady = [w for w, b in zip(active_watts, busy_pcts) if b >= 50.0]
        cor_w = statistics.mean(steady) if steady else statistics.mean(active_watts)
    else:
        cor_w = statistics.mean(active_watts) if active_watts else 0.0
    dyn_w = max(0.0, cor_w - idle_w)
    cpu_s_per_s = wall_seconds / audio_seconds if audio_seconds > 0 else 0.0
    rtfx = 1.0 / cpu_s_per_s if cpu_s_per_s > 0 else 0.0
    audio_hours = audio_seconds / 3600.0
    total_core_j = cor_w * wall_seconds
    dyn_core_j = dyn_w * wall_seconds
    wh_per_audio_h = (total_core_j / 3600.0) / audio_hours if audio_hours > 0 else 0.0
    dyn_wh_per_audio_h = (dyn_core_j / 3600.0) / audio_hours if audio_hours > 0 else 0.0
    mw_per_stream = cor_w * cpu_s_per_s * 1000.0
    dyn_mw_per_stream = dyn_w * cpu_s_per_s * 1000.0
    streams_per_watt = (1.0 / (cor_w * cpu_s_per_s)) if (cor_w * cpu_s_per_s) > 0 else 0.0
    return {
        "system": name,
        "mode": mode,
        "audio_seconds": round(audio_seconds, 2),
        "wall_seconds": round(wall_seconds, 3),
        "cpu_s_per_audio_s": round(cpu_s_per_s, 5),
        "rtfx_1core": round(rtfx, 2),
        "idle_pre_w": round(pre_w, 3) if pre_w is not None else None,
        "idle_post_w": round(post_w, 3) if post_w is not None else None,
        "idle_core_w": round(idle_w, 3),
        "active_core_w": round(cor_w, 3),
        "dynamic_core_w": round(dyn_w, 3),
        "avg_busy_mhz": round(statistics.mean(active_mhz), 1) if active_mhz else None,
        "avg_pkg_w": round(statistics.mean(pkg_watts), 2) if pkg_watts else None,
        "core_joules_per_audio_h": round(total_core_j / audio_hours, 1) if audio_hours > 0 else 0.0,
        "core_wh_per_audio_h": round(wh_per_audio_h, 4),
        "dynamic_wh_per_audio_h": round(dyn_wh_per_audio_h, 4),
        "mw_per_live_stream": round(mw_per_stream, 1),
        "dynamic_mw_per_live_stream": round(dyn_mw_per_stream, 1),
        "streams_per_core_watt": round(streams_per_watt, 2),
        "samples_active": len(active_watts),
    }


def summarize_pkg8_power(
    *,
    name: str,
    mode: str,
    audio_seconds: float,
    wall_seconds: float,
    pre_pkg_w: float,
    post_pkg_w: float,
    run_pkg_j: float,
    n_cores: int = 8,
) -> dict:
    base_pkg_w = 0.5 * (pre_pkg_w + post_pkg_w)
    run_pkg_w = run_pkg_j / wall_seconds if wall_seconds > 0 else 0.0
    delta_pkg_w = max(0.0, run_pkg_w - base_pkg_w)
    delta_pkg_j = delta_pkg_w * wall_seconds
    audio_hours = audio_seconds / 3600.0
    wall_rtfx = audio_seconds / wall_seconds if wall_seconds > 0 else 0.0
    delta_wh_per_audio_h = (delta_pkg_j / 3600.0) / audio_hours if audio_hours > 0 else 0.0
    # Equivalent incremental socket mW per 1x real-time stream
    delta_mw_per_stream = (delta_pkg_w / wall_rtfx) * 1000.0 if wall_rtfx > 0 else 0.0
    return {
        "system": name,
        "mode": mode,
        "cores": n_cores,
        "audio_seconds": round(audio_seconds, 2),
        "wall_seconds": round(wall_seconds, 3),
        "wall_rtfx_8core": round(wall_rtfx, 2),
        "per_core_rtfx": round(wall_rtfx / n_cores, 2),
        "pre_pkg_w": round(pre_pkg_w, 2),
        "post_pkg_w": round(post_pkg_w, 2),
        "baseline_pkg_w": round(base_pkg_w, 2),
        "run_pkg_w": round(run_pkg_w, 2),
        "delta_pkg_w": round(delta_pkg_w, 2),
        "delta_pkg_w_per_core": round(delta_pkg_w / n_cores, 2),
        "delta_pkg_wh_per_audio_h": round(delta_wh_per_audio_h, 4),
        "delta_pkg_mw_per_stream": round(delta_mw_per_stream, 1),
    }


def read_rapl_uj() -> tuple[int, int | None]:
    out = subprocess.check_output(["sudo", "-n", "cat", str(RAPL_PKG)], text=True).strip()
    max_r = None
    try:
        max_r = int(subprocess.check_output(["sudo", "-n", "cat", str(RAPL_MAX)], text=True).strip())
    except subprocess.CalledProcessError:
        pass
    return int(out), max_r


def sample_idle_core(cpu: int, duration_s: float = 4.0, interval_s: float = 0.5) -> tuple[list[float], float]:
    iters = max(2, int(round(duration_s / interval_s)))
    proc = subprocess.run(
        [
            "sudo", "-n", "turbostat", "--quiet",
            "--cpu", str(cpu),
            "--show", "Core,CPU,Busy%,Bzy_MHz,CorWatt,PkgWatt",
            "--interval", str(interval_s),
            "--num_iterations", str(iters),
        ],
        capture_output=True, text=True, check=True,
    )
    rows = [r for r in parse_turbostat_rows(proc.stdout, {cpu}) if not r["summary"]]
    cor = [r["CorWatt"] for r in rows if "CorWatt" in r]
    pkg = [r["PkgWatt"] for r in rows if "PkgWatt" in r]
    return cor, (statistics.median(pkg) if pkg else 0.0)


def measure_single_core_cmd(
    name: str,
    mode: str,
    cmd: list[str],
    env_extra: dict[str, str],
    cpu: int,
    audio_seconds: float,
    interval_s: float = 0.5,
) -> dict:
    import os

    pre_cor, _ = sample_idle_core(cpu, duration_s=3.5, interval_s=interval_s)
    ts = subprocess.Popen(
        [
            "sudo", "-n", "turbostat", "--quiet",
            "--cpu", str(cpu),
            "--show", "Core,CPU,Busy%,Bzy_MHz,CorWatt,PkgWatt",
            "--interval", str(interval_s),
        ],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
    )
    env = os.environ.copy()
    env.update(env_extra)
    t0 = time.monotonic()
    rc = subprocess.run(["taskset", "-c", str(cpu), *cmd], env=env,
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode
    wall_s = time.monotonic() - t0
    ts.terminate()
    try:
        out, _ = ts.communicate(timeout=3)
    except subprocess.TimeoutExpired:
        ts.kill()
        out, _ = ts.communicate()
    if rc != 0:
        raise RuntimeError(f"{name} exited with code {rc}")
    post_cor, _ = sample_idle_core(cpu, duration_s=3.5, interval_s=interval_s)
    rows = [r for r in parse_turbostat_rows(out, {cpu}) if not r["summary"]]
    return summarize_core_power(
        name=name,
        mode=mode,
        audio_seconds=audio_seconds,
        wall_seconds=wall_s,
        pre_idle_watts=pre_cor,
        active_watts=[r["CorWatt"] for r in rows if "CorWatt" in r],
        post_idle_watts=post_cor,
        active_mhz=[r["Bzy_MHz"] for r in rows if "Bzy_MHz" in r],
        busy_pcts=[r.get("Busy%", 100.0) for r in rows if "CorWatt" in r],
        pkg_watts=[r["PkgWatt"] for r in rows if "PkgWatt" in r],
    )


def sample_idle_pkg_w(duration_s: float = 5.0) -> float:
    e0, max_r = read_rapl_uj()
    t0 = time.monotonic()
    time.sleep(duration_s)
    e1, _ = read_rapl_uj()
    dt = time.monotonic() - t0
    return rapl_delta_joules(e0, e1, max_r) / dt


def measure_pkg8_cmd(
    name: str,
    mode: str,
    shell_cmd: str,
    cpus: str,
    audio_seconds: float,
    n_cores: int = 8,
) -> dict:
    pre_w = sample_idle_pkg_w(4.0)
    e0, max_r = read_rapl_uj()
    t0 = time.monotonic()
    rc = subprocess.run(["bash", "-c", f"taskset -c {cpus} {shell_cmd}"],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode
    wall_s = time.monotonic() - t0
    e1, _ = read_rapl_uj()
    if rc != 0:
        raise RuntimeError(f"{name} (8-core) exited with code {rc}")
    post_w = sample_idle_pkg_w(4.0)
    run_j = rapl_delta_joules(e0, e1, max_r)
    return summarize_pkg8_power(
        name=name,
        mode=mode,
        audio_seconds=audio_seconds,
        wall_seconds=wall_s,
        pre_pkg_w=pre_w,
        post_pkg_w=post_w,
        run_pkg_j=run_j,
        n_cores=n_cores,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timestamps", type=Path, required=True)
    parser.add_argument("--wav-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cpu", type=int, default=44)
    parser.add_argument("--cpus8", default="56-63")
    parser.add_argument("--skip-pkg8", action="store_true")
    args = parser.parse_args()

    ts_data = json.loads(args.timestamps.read_text(encoding="utf-8"))
    audio_s = sum(float(r["duration_s"]) for r in ts_data)
    ts_sha = hashlib.sha256(args.timestamps.read_bytes()).hexdigest()

    data = Path("/mnt/xbabe3-external/jsr")
    sdk = Path("/home/ubuntu/jsr/target/paper-smoke")
    cold_wav = Path("/mnt/xbabe3-external/jsr/cargo-target/main/release/ultra-cold-wav")
    ms_bin = sdk / "moonshine-stream-trace"
    nemo = data / "ext/NeMo-Speech.cpp/build/cpu-asr/bin/nemo-speech"
    nemo_model = data / "models/nemotron-0.6b/nemotron-speech-streaming-en-0.6b.q8_0.gguf"
    nemo35 = sdk / "nemo-speech-0.3.0-linux-x86_64-cpu/bin/nemo-speech"
    nemo35_model = sdk / "nemo35-verified.gguf"
    opp_py = Path(__file__).with_name("opponents_bench.py")
    venv_py = Path("/home/ubuntu/jsr/.venv/bin/python")
    bench_py = sdk / ".venv-bench/bin/python"
    tmp = args.out.parent / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    env1 = {"OMP_NUM_THREADS": "1", "OMP_THREAD_LIMIT": "1", "MKL_NUM_THREADS": "1",
            "RAYON_NUM_THREADS": "1", "HF_HUB_OFFLINE": "1"}

    def clean_dir(p: Path) -> Path:
        import shutil
        if p.exists():
            shutil.rmtree(p)
        return p

    arms = [
        ("jsr-d2e-16457-int8", "streaming", lambda: (
            (tmp / "d2e.jsonl").unlink(missing_ok=True),
            [str(cold_wav), "r1-engine-int8",
             str(data / "runs/dashboard-asr/D2e-00016457-yqcxmp56/int8"),
             str(args.wav_dir), str(tmp / "d2e.jsonl")],
        )[1]),
        ("jsr-d2c-10193-int8", "streaming", lambda: (
            (tmp / "d2c.jsonl").unlink(missing_ok=True),
            [str(cold_wav), "r1-engine-int8",
             str(data / "runs/dashboard-asr/ckpt-00010193/int8"),
             str(args.wav_dir), str(tmp / "d2c.jsonl")],
        )[1]),
        ("jsr-d2mr-50k-int8", "streaming", lambda: (
            (tmp / "d2mr.jsonl").unlink(missing_ok=True),
            [str(cold_wav), "r1-engine-int8",
             str(data / "runs/engine/d2mr-50000/weights-int8prep"),
             str(args.wav_dir), str(tmp / "d2mr.jsonl")],
        )[1]),
        ("r0-zipformer-int8", "streaming", lambda: (
            (tmp / "r0.jsonl").unlink(missing_ok=True),
            [str(cold_wav), "zipformer-int8",
             str(data / "bundles/r0-zipformer-int8"),
             str(args.wav_dir), str(tmp / "r0.jsonl")],
        )[1]),
        ("moonshine-tiny", "streaming", lambda: [
            str(ms_bin), "tiny", str(sdk / "moonshine-tiny"),
            str(args.wav_dir), str(clean_dir(tmp / "ms-tiny")),
        ]),
        ("moonshine-small", "streaming", lambda: [
            str(ms_bin), "small", str(sdk / "moonshine-small"),
            str(args.wav_dir), str(clean_dir(tmp / "ms-small")),
        ]),
        ("moonshine-medium", "streaming", lambda: [
            str(ms_bin), "medium", str(sdk / "moonshine-medium"),
            str(args.wav_dir), str(clean_dir(tmp / "ms-medium")),
        ]),
        ("nemotron-en-560", "streaming", lambda: (
            clean_dir(tmp / "nemo560").mkdir(parents=True),
            [str(nemo), "--json", "bench", "asr", str(args.wav_dir),
             "--model", str(nemo_model), "--device", "cpu", "--mode", "stream",
             "--asr.backend.threads", "1", "--asr.streaming.rnnt_right_context", "6",
             "--concurrency", "1", "--repetitions", "1", "--warmup", "0",
             "--save", str(tmp / "nemo560")],
        )[1]),
        ("nemotron-en-160", "streaming", lambda: (
            clean_dir(tmp / "nemo160").mkdir(parents=True),
            [str(nemo), "--json", "bench", "asr", str(args.wav_dir),
             "--model", str(nemo_model), "--device", "cpu", "--mode", "stream",
             "--asr.backend.threads", "1", "--asr.streaming.rnnt_right_context", "1",
             "--concurrency", "1", "--repetitions", "1", "--warmup", "0",
             "--save", str(tmp / "nemo160")],
        )[1]),
        ("nemotron-35", "streaming", lambda: (
            clean_dir(tmp / "nemo35").mkdir(parents=True),
            [str(nemo35), "--json", "bench", "asr", str(args.wav_dir),
             "--model", str(nemo35_model), "--device", "cpu", "--mode", "stream",
             "--language", "en-US", "--chunk-ms", "80",
             "--asr.backend.threads", "1", "--asr.streaming.rnnt_right_context", "0",
             "--concurrency", "1", "--repetitions", "1", "--warmup", "0",
             "--save", str(tmp / "nemo35")],
        )[1]),
        ("parakeet-redux-photon", "whole_utterance", lambda: [
            str(bench_py), str(opp_py), "parakeet-redux",
            "--wav-dir", str(args.wav_dir),
            "--output-dir", str(clean_dir(tmp / "rdx")), "--threads", "1",
        ]),
        ("parakeet-tdt-v3-int8", "whole_utterance", lambda: [
            str(venv_py), str(opp_py), "parakeet-tdt-v3",
            "--model-dir", str(sdk / "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"),
            "--wav-dir", str(args.wav_dir),
            "--output-dir", str(clean_dir(tmp / "tdt")), "--threads", "1",
        ]),
    ]

    result = {
        "schema": "jsr.power-bench/v1",
        "hardware": "AMD Ryzen Threadripper PRO 3995WX (Zen 2, 64-core)",
        "pinned_core": args.cpu,
        "workload": f"dev-clean first {len(ts_data)} utterances ({audio_s:.1f} s)",
        "timestamps_sha256": ts_sha,
        "core_power": [],
        "pkg8_power": [],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for name, mode, make_cmd in arms:
        print(f"[power] core-{args.cpu}: {name} ...", flush=True)
        cmd = make_cmd()
        row = measure_single_core_cmd(name, mode, cmd, env1, args.cpu, audio_s)
        result["core_power"].append(row)
        print(f"  -> {name:24s} | CorWatt={row['active_core_w']:5.2f}W (dyn {row['dynamic_core_w']:4.2f}W) "
              f"| RTFx={row['rtfx_1core']:6.2f}x | {row['mw_per_live_stream']:7.1f} mW/stream "
              f"| {row['core_wh_per_audio_h']:.3f} Wh/audio-h", flush=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    if not args.skip_pkg8:
        pkg8_arms = [
            # 8 independent single-thread JSR workers in parallel (8 * audio_s processed!)
            ("jsr-d2e-16457-int8-8w", "streaming", audio_s * 8,
             f"bash -c 'for i in $(seq 0 7); do rm -f {tmp}/p8_d2e_$i.jsonl; "
             f"OMP_NUM_THREADS=1 taskset -c $((56+i)) {cold_wav} r1-engine-int8 "
             f"{data}/runs/dashboard-asr/D2e-00016457-yqcxmp56/int8 {args.wav_dir} {tmp}/p8_d2e_$i.jsonl & done; wait'"),
            ("r0-zipformer-int8-8w", "streaming", audio_s * 8,
             f"bash -c 'for i in $(seq 0 7); do rm -f {tmp}/p8_r0_$i.jsonl; "
             f"OMP_NUM_THREADS=1 taskset -c $((56+i)) {cold_wav} zipformer-int8 "
             f"{data}/bundles/r0-zipformer-int8 {args.wav_dir} {tmp}/p8_r0_$i.jsonl & done; wait'"),
            ("parakeet-redux-photon-8t", "whole_utterance", audio_s,
             f"env OMP_NUM_THREADS=8 RAYON_NUM_THREADS=8 MKL_NUM_THREADS=8 HF_HUB_OFFLINE=1 "
             f"{bench_py} {opp_py} parakeet-redux --wav-dir {args.wav_dir} --output-dir {tmp}/p8_rdx --threads 8"),
            ("parakeet-tdt-v3-int8-8t", "whole_utterance", audio_s,
             f"env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 "
             f"{venv_py} {opp_py} parakeet-tdt-v3 --model-dir {sdk}/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8 "
             f"--wav-dir {args.wav_dir} --output-dir {tmp}/p8_tdt --threads 8"),
            ("nemotron-en-160-8t", "streaming", audio_s,
             f"bash -c 'rm -rf {tmp}/p8_nemo160 && mkdir -p {tmp}/p8_nemo160 && "
             f"OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 {nemo} --json bench asr {args.wav_dir} "
             f"--model {nemo_model} --device cpu --mode stream --asr.backend.threads 8 "
             f"--asr.streaming.rnnt_right_context 1 --concurrency 1 --repetitions 1 --warmup 0 --save {tmp}/p8_nemo160'"),
        ]
        for name, mode, aud, scmd in pkg8_arms:
            print(f"[power] pkg8 ({args.cpus8}): {name} ...", flush=True)
            row = measure_pkg8_cmd(name, mode, scmd, args.cpus8, aud, n_cores=8)
            result["pkg8_power"].append(row)
            print(f"  -> {name:24s} | dPkgWatt={row['delta_pkg_w']:6.2f}W ({row['delta_pkg_w_per_core']:4.2f}W/core) "
                  f"| 8c RTFx={row['wall_rtfx_8core']:6.2f}x | {row['delta_pkg_mw_per_stream']:7.1f} mW/stream "
                  f"| {row['delta_pkg_wh_per_audio_h']:.3f} Wh/audio-h", flush=True)
            args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    sys.exit(main())
