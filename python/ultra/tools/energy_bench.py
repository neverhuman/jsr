"""Energy per audio-second (= watts per live stream) for a command on pinned cores.

A read-only root helper samples the AMD per-core energy register (MSR 0xC001029A)
and the package energy register (MSR 0xC001029B, plus the powercap file) once a
second while an unprivileged command runs on pinned cores, with an idle window
before and after. Standard library only. Protocol: research/H39_energy/README.md.
Reports over the receipts (fits, tables, evidence lines): energy_report.py.

    energy_bench.py run --arm NAME --cpus 16 --observer-cpu 31 --audio-seconds S
        --out receipt.json -- COMMAND [ARGS...]     ({rep}, {worker}, {cpu} are substituted)
    energy_bench.py spin --seconds 30               (busy loop, the validation workload)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import select
import shutil
import statistics
import struct
import subprocess
import sys
import threading
import time
import wave

SCHEMA = "jsr.energy-bench/v1"
MSR_RAPL_UNIT = 0xC0010299
MSR_CORE_ENERGY = 0xC001029A
MSR_PKG_ENERGY = 0xC001029B
COUNTER_MODULUS = 1 << 32
POWERCAP_PKG = Path("/sys/class/powercap/intel-rapl:0")
CPU_SYSFS = Path("/sys/devices/system/cpu")
ENERGY_METHOD = "amd-msr-core-energy-0xC001029A+package-0xC001029B, idle-subtracted, exact phase boundaries"
ENERGY_METHOD_PACKAGE_ONLY = "rapl-powercap-package-only, idle-subtracted"


# ----------------------------------------------------------------------------
# Pure arithmetic (unit-tested in python/ultra/tests/test_energy_bench.py)
# ----------------------------------------------------------------------------

def parse_cpu_list(text):
    """'16-19,31' -> [16, 17, 18, 19, 31]; order kept, duplicates refused."""
    cpus = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        first, _, last = part.partition("-")
        low, high = int(first), int(last or first)
        if high < low:
            raise ValueError(f"descending CPU range {part!r}")
        cpus.extend(range(low, high + 1))
    if not cpus or len(set(cpus)) != len(cpus) or min(cpus) < 0:
        raise ValueError(f"invalid CPU list {text!r}")
    return cpus


def energy_unit_joules(unit_msr):
    """Joules per count from the RAPL unit register (bits 12:8 are the exponent)."""
    return 1.0 / (1 << ((unit_msr >> 8) & 0x1F))


def wrap_delta(previous, current, modulus):
    """Counts added between two readings of a counter that wraps at `modulus`."""
    if modulus <= 0 or not 0 <= previous <= modulus or not 0 <= current <= modulus:
        raise ValueError("counter reading outside its range")
    return (current - previous) % modulus


def accumulate(readings, modulus, scale):
    """Cumulative energy (first reading = 0) from consecutive readings of a wrapping counter.

    Correct as long as the counter wraps at most once between two readings.
    """
    total = 0
    out = [0.0]
    for previous, current in zip(readings, readings[1:]):
        total += wrap_delta(previous, current, modulus)
        out.append(total * scale)
    return out


def summarise(values):
    """Median with min, max and spread for a handful of repetitions."""
    values = [float(v) for v in values if v is not None]
    if not values:
        return None
    median = statistics.median(values)
    spread = max(values) - min(values)
    return {
        "median": median,
        "min": min(values),
        "max": max(values),
        "spread": spread,
        "spread_pct": 100.0 * spread / abs(median) if median else None,
        "stdev": statistics.stdev(values) if len(values) > 1 else None,
        "n": len(values),
    }


def view_metrics(run_joules, run_seconds, idle_joules, idle_seconds, audio_seconds, cpu_seconds):
    """Idle-subtracted power and energy for one view (core or package) of one repetition."""
    run_watts = run_joules / run_seconds
    idle_watts = idle_joules / idle_seconds
    net_watts = run_watts - idle_watts
    net_joules = net_watts * run_seconds
    return {
        "run_watts": run_watts,
        "idle_watts": idle_watts,
        "net_watts": net_watts,
        "net_joules": net_joules,
        "joules_per_audio_s": net_joules / audio_seconds if audio_seconds else None,
        "gross_joules_per_audio_s": run_joules / audio_seconds if audio_seconds else None,
        "joules_per_cpu_s": net_joules / cpu_seconds if cpu_seconds else None,
    }


class Trace:
    """Counter samples from the helper, turned into cumulative joules per counter."""

    def __init__(self, header, samples):
        if len(samples) < 2:
            raise ValueError("need at least two samples")
        self.times = [s["t"] for s in samples]
        if any(b <= a for a, b in zip(self.times, self.times[1:])):
            raise ValueError("sample times must increase")
        unit = header["energy_unit_j"]
        self.core_cpus = [int(c) for c in header["cpus"]]
        self.core = {
            cpu: accumulate([s["core"][str(cpu)] for s in samples], COUNTER_MODULUS, unit)
            for cpu in self.core_cpus
        }
        self.package = None
        if samples[0].get("pkg") is not None:
            self.package = accumulate([s["pkg"] for s in samples], COUNTER_MODULUS, unit)
        self.package_sysfs = None
        if samples[0].get("pkg_uj") is not None:
            self.package_sysfs = accumulate([s["pkg_uj"] for s in samples], header["pkg_max_uj"], 1e-6)
        self.marks = {s["label"]: i for i, s in enumerate(samples) if s.get("label")}
        self.snapshots = {s["label"]: s["all"] for s in samples if s.get("label") and s.get("all")}
        self.unit = unit

    def core_sum(self):
        return [sum(self.core[cpu][i] for cpu in self.core_cpus) for i in range(len(self.times))]

    def span(self, series, first, last):
        """(joules, seconds) between two sample indices: an exact counter difference."""
        return series[last] - series[first], self.times[last] - self.times[first]

    def trimmed(self, first, last, trim):
        """Sample indices that leave `trim` seconds out at both ends of [first, last]."""
        low, high = self.times[first] + trim, self.times[last] - trim
        inside = [i for i in range(first, last + 1) if low <= self.times[i] <= high]
        if len(inside) < 2:
            raise ValueError("nothing left of the run after trimming")
        return inside[0], inside[-1]

    def per_second_watts(self, series, first, last, shortest=0.3):
        """Watts of each sampling interval in [first, last]; slivers next to a mark are left out."""
        return [
            (series[i + 1] - series[i]) / (self.times[i + 1] - self.times[i])
            for i in range(first, last) if self.times[i + 1] - self.times[i] >= shortest
        ]

    def other_cores_joules(self, start_label, end_label, exclude):
        """Energy of every physical core outside `exclude` between two boundary snapshots."""
        before, after = self.snapshots.get(start_label), self.snapshots.get(end_label)
        if not before or not after:
            return None
        counts = sum(
            wrap_delta(before[cpu], after[cpu], COUNTER_MODULUS)
            for cpu in before if int(cpu) not in exclude
        )
        return counts * self.unit


def busy_fraction(before, after, cpus):
    """Share of the interval the given CPUs spent busy, from two /proc/stat snapshots."""
    busy = sum(after[c][0] - before[c][0] for c in cpus)
    total = sum(after[c][1] - before[c][1] for c in cpus)
    return busy / total if total else 0.0


def flags_for(rep, thresholds, pinned_cores, busy_workers):
    """The distrust rules of the protocol note, evaluated for one repetition.

    `busy_workers` is the number of workers expected to be on-CPU for the whole run
    (0 for paced runs, where waiting is the point).
    """
    flags = []
    if rep["busy"]["sibling_busy_fraction_run"] > thresholds["sibling_busy"]:
        flags.append("sibling_busy")
    other = rep.get("other_cores")
    if other:
        levels = [other["run_watts"], other.get("idle_before_watts", other["idle_watts"]),
                  other.get("idle_after_watts", other["idle_watts"])]
        if max(levels) - min(levels) > thresholds["foreign_watts"]:
            flags.append("foreign_load_changed")
    core, package = rep.get("core"), rep.get("package")
    if core and abs(core["idle_before_watts"] - core["idle_after_watts"]) > thresholds["idle_drift_core"] * pinned_cores:
        flags.append("idle_drift_core")
    if package and abs(package["idle_before_watts"] - package["idle_after_watts"]) > thresholds["idle_drift_package"]:
        flags.append("idle_drift_package")
    if busy_workers and rep["cpu_seconds"] < thresholds["on_cpu"] * rep["wall_seconds"] * busy_workers:
        flags.append("worker_not_on_cpu")
    if any(code != 0 for code in rep["exit_codes"]):
        flags.append("nonzero_exit")
    return flags


CORE_DIRTY = frozenset({"sibling_busy", "idle_drift_core", "worker_not_on_cpu", "nonzero_exit"})
PACKAGE_DIRTY = CORE_DIRTY | {"foreign_load_changed", "idle_drift_package"}
DEFAULT_THRESHOLDS = {
    "sibling_busy": 0.01,        # share of the run an SMT sibling of a pinned core may be busy
    "foreign_watts": 1.0,        # other cores' summed watts may move this much across idle, run, idle
    "idle_drift_core": 0.2,      # watts per pinned core between the two idle windows
    "idle_drift_package": 2.0,   # watts between the two idle windows
    "on_cpu": 0.98,              # CPU seconds / (wall seconds x busy workers)
    "rep_spread_pct": 5.0,       # min-to-max of joules per audio-second over the clean repetitions
}


def headline(reps, thresholds):
    """Headline numbers over the repetitions that are clean for each view (all of them if none is)."""
    def clean(view):
        dirty = CORE_DIRTY if view == "core" else PACKAGE_DIRTY
        return [rep for rep in reps if view in rep and not dirty & set(rep["flags"])]

    def over(view, key):
        return summarise([rep[view][key] for rep in clean(view) or [r for r in reps if view in r]])

    def plain(key):
        return summarise([rep[key] for rep in clean("core") or reps])

    def robust(key):
        """Package figure from the median second of run and idle, over every core-clean repetition."""
        usable = [rep for rep in clean("core") or reps if "package" in rep]
        return summarise([rep["package"].get(key) for rep in usable])

    summary = {
        "joules_per_audio_s": over("core", "joules_per_audio_s"),
        "joules_per_audio_s_package": over("package", "joules_per_audio_s"),
        "gross_joules_per_audio_s": over("core", "gross_joules_per_audio_s"),
        "watts_per_busy_core": over("core", "joules_per_cpu_s"),
        "watts_per_busy_core_package": over("package", "joules_per_cpu_s"),
        "net_watts": over("core", "net_watts"),
        "net_watts_package": over("package", "net_watts"),
        "run_watts": over("core", "run_watts"),
        "run_watts_package": over("package", "run_watts"),
        "idle_watts_cores": over("core", "idle_watts"),
        "idle_watts": over("package", "idle_watts"),
        "joules_per_audio_s_package_robust": robust("joules_per_audio_s_median"),
        "watts_per_busy_core_package_robust": robust("joules_per_cpu_s_median"),
        "net_watts_package_robust": robust("net_watts_median"),
        "cpu_s_per_audio_s": plain("cpu_s_per_audio_s"),
        "wall_seconds": plain("wall_seconds"),
        "mean_mhz": plain("mean_mhz"),
    }

    def median(name):
        return summary[name]["median"] if summary[name] else None

    spread = summary["joules_per_audio_s"]
    flags = {flag for rep in reps for flag in rep["flags"]}
    for view in ("core", "package"):
        if any(view in rep for rep in reps) and not clean(view):
            flags.add(f"no_clean_{view}_rep")
    return {
        "joules_per_audio_s": median("joules_per_audio_s"),
        "watts_per_stream": median("joules_per_audio_s"),
        "joules_per_audio_s_package": median("joules_per_audio_s_package"),
        "watts_per_stream_package": median("joules_per_audio_s_package"),
        "joules_per_audio_s_package_robust": median("joules_per_audio_s_package_robust"),
        "watts_per_busy_core": median("watts_per_busy_core"),
        "watts_per_busy_core_package": median("watts_per_busy_core_package"),
        "watts_per_busy_core_package_robust": median("watts_per_busy_core_package_robust"),
        "idle_watts": median("idle_watts"),
        "cpu_s_per_audio_s": median("cpu_s_per_audio_s"),
        "repetitions": len(reps),
        "clean_repetitions": {"core": len(clean("core")), "package": len(clean("package"))},
        "repetitions_disagree": bool(spread and spread["spread_pct"] is not None
                                     and spread["spread_pct"] > thresholds["rep_spread_pct"]),
        "flags": sorted(flags),
        "summary": summary,
    }


# ----------------------------------------------------------------------------
# Host facts (sysfs and procfs, no privileges)
# ----------------------------------------------------------------------------

def read_text(path, default=None):
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return default


def thread_siblings(cpu):
    return parse_cpu_list(read_text(CPU_SYSFS / f"cpu{cpu}/topology/thread_siblings_list"))


def physical_cores():
    """{lowest thread of the core: [threads]} for every online CPU."""
    cores = {}
    for cpu in parse_cpu_list(read_text(CPU_SYSFS / "online")):
        threads = thread_siblings(cpu)
        cores[min(threads)] = threads
    return cores


def read_proc_stat():
    """{cpu: (busy jiffies, all jiffies)} from /proc/stat."""
    out = {}
    for line in Path("/proc/stat").read_text(encoding="ascii").splitlines():
        name, *fields = line.split()
        if name.startswith("cpu") and name[3:].isdigit():
            user, nice, system, idle, iowait, irq, softirq, steal = (int(v) for v in fields[:8])
            busy = user + nice + system + irq + softirq + steal
            out[int(name[3:])] = (busy, busy + idle + iowait)
    return out


def cpu_identity():
    info = {}
    for line in Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace").splitlines():
        key, _, value = line.partition(":")
        key = key.strip()
        if key in ("vendor_id", "model name", "microcode") and key not in info:
            info[key] = value.strip()
        if len(info) == 3:
            break
    return info


def host_facts(cpus):
    identity = cpu_identity()
    first = cpus[0]
    idle_states = []
    for state in sorted((CPU_SYSFS / f"cpu{first}/cpuidle").glob("state*")):
        idle_states.append(read_text(state / "name"))
    return {
        "hostname": platform.node(),
        "kernel": platform.release(),
        "cpu_model": identity.get("model name"),
        "cpu_vendor": identity.get("vendor_id"),
        "microcode": identity.get("microcode"),
        "governor": {str(c): read_text(CPU_SYSFS / f"cpu{c}/cpufreq/scaling_governor") for c in cpus},
        "cpufreq_driver": read_text(CPU_SYSFS / f"cpu{first}/cpufreq/scaling_driver"),
        "boost": read_text(CPU_SYSFS / "cpufreq/boost"),
        "smt_active": read_text(CPU_SYSFS / "smt/active"),
        "idle_states": idle_states,
        "online_cpus": read_text(CPU_SYSFS / "online"),
    }


def sha256_file(path):
    with open(path, "rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def sha256_path(path):
    """A file's SHA-256, or for a directory the SHA-256 of its sorted 'hash  relative-path' lines."""
    path = Path(path)
    if path.is_file():
        return sha256_file(path)
    lines = [
        f"{sha256_file(item)}  {item.relative_to(path).as_posix()}\n"
        for item in sorted(path.rglob("*")) if item.is_file()
    ]
    if not lines:
        raise ValueError(f"nothing to hash under {path}")
    return hashlib.sha256("".join(lines).encode()).hexdigest()


def wav_dir_seconds(directory):
    """(audio seconds, file count) of every .wav directly inside a directory."""
    seconds, count = 0.0, 0
    for path in sorted(Path(directory).iterdir()):
        if path.suffix == ".wav" and path.is_file():
            with wave.open(str(path), "rb") as source:
                seconds += source.getnframes() / source.getframerate()
            count += 1
    if not count:
        raise ValueError(f"no .wav files in {directory}")
    return seconds, count


# ----------------------------------------------------------------------------
# Root helper: opens counters read-only and prints samples. Nothing else.
# ----------------------------------------------------------------------------

def read_msr(fd, register):
    return struct.unpack("<Q", os.pread(fd, 8, register))[0]


def sampler_main(args):
    if args.pin_cpu is not None:
        os.sched_setaffinity(0, {args.pin_cpu})
    cpus = parse_cpu_list(args.cpus)
    every = sorted(physical_cores()) if args.all_cores else []
    fds = {}
    unit_j = None
    msr_ok = cpu_identity().get("vendor_id") == "AuthenticAMD" and os.path.exists(f"/dev/cpu/{cpus[0]}/msr")
    if msr_ok:
        for cpu in sorted(set(cpus) | set(every)):
            fds[cpu] = os.open(f"/dev/cpu/{cpu}/msr", os.O_RDONLY)
        unit_j = energy_unit_joules(read_msr(fds[cpus[0]], MSR_RAPL_UNIT))
    pkg_file = None
    pkg_max = None
    if (POWERCAP_PKG / "energy_uj").exists():
        pkg_file = open(POWERCAP_PKG / "energy_uj", "rb", buffering=0)  # held open for the whole run
        pkg_max = int(read_text(POWERCAP_PKG / "max_energy_range_uj"))
    if not msr_ok and pkg_file is None:
        print("energy_bench sample: no readable energy counter on this host", file=sys.stderr)
        return 2
    out = sys.stdout
    out.write(json.dumps({
        "type": "header", "energy_unit_j": unit_j, "pkg_max_uj": pkg_max,
        "cpus": cpus if msr_ok else [], "all_cores": every if msr_ok else [],
        "msr": msr_ok, "powercap": pkg_file is not None, "interval": args.interval,
    }) + "\n")
    out.flush()

    def sample(label):
        record = {"t": 0.0, "label": label}
        before = time.monotonic()
        if msr_ok:
            record["pkg"] = read_msr(fds[cpus[0]], MSR_PKG_ENERGY) & 0xFFFFFFFF
            record["core"] = {str(c): read_msr(fds[c], MSR_CORE_ENERGY) & 0xFFFFFFFF for c in cpus}
        if pkg_file is not None:
            pkg_file.seek(0)
            record["pkg_uj"] = int(pkg_file.read())
        record["t"] = (before + time.monotonic()) / 2
        record["wall"] = time.time()
        if label and msr_ok and every:
            record["all"] = {str(c): read_msr(fds[c], MSR_CORE_ENERGY) & 0xFFFFFFFF for c in every}
        out.write(json.dumps(record, separators=(",", ":")) + "\n")
        out.flush()

    start = time.monotonic()
    deadline = start + args.max_seconds
    tick = 0
    buffered = b""
    sample("start")
    while time.monotonic() < deadline:
        tick += 1
        due = start + tick * args.interval
        while True:
            wait = due - time.monotonic()
            if wait <= 0:
                break
            ready, _, _ = select.select([sys.stdin.buffer], [], [], wait)
            if not ready:
                break
            chunk = os.read(sys.stdin.fileno(), 4096)
            if not chunk:
                sample("stop")
                return 0
            buffered += chunk
            while b"\n" in buffered:
                line, buffered = buffered.split(b"\n", 1)
                words = line.decode("ascii", "replace").split()
                if words[:1] == ["mark"] and len(words) == 2:
                    sample(words[1])
                elif words[:1] == ["stop"]:
                    sample("stop")
                    return 0
        sample(None)
    return 0


# ----------------------------------------------------------------------------
# Unprivileged side: start the helper, run the command, compute, write the receipt
# ----------------------------------------------------------------------------

class Sampler:
    """The root helper as a child process; marks are exact extra samples."""

    def __init__(self, cpus, observer_cpu, interval, max_seconds, frequency_cpus, sudo=True):
        command = [sys.executable, str(Path(__file__).resolve()), "sample", "--cpus", ",".join(map(str, cpus)),
                   "--interval", str(interval), "--max-seconds", str(max_seconds), "--all-cores"]
        if observer_cpu is not None:
            command += ["--pin-cpu", str(observer_cpu)]
        if sudo and os.geteuid() != 0:
            command = ["sudo", "-n", *command]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        self.header = json.loads(self.process.stdout.readline() or b"null")
        if not self.header:
            raise RuntimeError("the counter helper did not start (is `sudo -n` allowed?)")
        self.samples = []
        self.frequency = []
        self.frequency_cpus = frequency_cpus
        self.arrived = threading.Condition()
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self):
        for line in self.process.stdout:
            record = json.loads(line)
            mhz = []
            for cpu in self.frequency_cpus:
                value = read_text(CPU_SYSFS / f"cpu{cpu}/cpufreq/scaling_cur_freq")
                if value:
                    mhz.append(int(value) / 1000.0)
            with self.arrived:
                self.samples.append(record)
                self.frequency.append((record["t"], statistics.fmean(mhz) if mhz else None))
                self.arrived.notify_all()

    def mark(self, label, timeout=20.0):
        self.process.stdin.write(f"mark {label}\n".encode())
        self.process.stdin.flush()
        with self.arrived:
            if not self.arrived.wait_for(lambda: any(s.get("label") == label for s in self.samples), timeout):
                raise RuntimeError(f"the counter helper did not answer mark {label}")

    def stop(self):
        try:
            self.process.stdin.write(b"stop\n")
            self.process.stdin.close()
        except OSError:
            pass
        self.process.wait(timeout=30)
        self.reader.join(timeout=30)


def substitute(argv, rep, worker, cpu):
    return [a.replace("{rep}", str(rep)).replace("{worker}", str(worker)).replace("{cpu}", str(cpu)) for a in argv]


def launch(argv, cpu_set, env, log_dir, rep, worker):
    stdout = stderr = subprocess.DEVNULL
    handles = []
    if log_dir:
        for name in ("stdout", "stderr"):
            handles.append(open(Path(log_dir) / f"rep{rep}-w{worker}.{name}", "wb"))
        stdout, stderr = handles
    # The child inherits the calling thread's affinity, so no Python runs between fork and exec
    # (the sample reader thread is alive here, which rules out preexec_fn).
    previous = os.sched_getaffinity(0)
    os.sched_setaffinity(0, cpu_set)
    try:
        return subprocess.Popen(argv, env=env, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr)
    finally:
        os.sched_setaffinity(0, previous)
        for handle in handles:
            handle.close()


def rounded(value, digits=6):
    if isinstance(value, float):
        return float(f"{value:.{digits}g}") if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: rounded(v, digits) for k, v in value.items()}
    if isinstance(value, list):
        return [rounded(v, digits) for v in value]
    return value


def view_for_rep(trace, series, marks, rep, last_rep, audio_seconds, cpu_seconds, trim, audio_rate, workers):
    """One view of one repetition: run window, the idle windows on both sides, optional trimming."""
    run_first, run_last = marks[f"run{rep}_start"], marks[f"run{rep}_end"]
    before = trace.span(series, marks[f"idle{rep}_start"], run_first)
    after_end = f"idle{rep + 1}_end" if rep == last_rep else f"run{rep + 1}_start"
    after = trace.span(series, marks[f"idle{rep + 1}_start"], marks[after_end])
    if trim:
        run_first, run_last = trace.trimmed(run_first, run_last, trim)
    run_joules, run_seconds = trace.span(series, run_first, run_last)
    if audio_rate:
        audio_seconds = audio_rate * workers * run_seconds
        cpu_seconds = None
    metrics = view_metrics(run_joules, run_seconds, before[0] + after[0], before[1] + after[1],
                           audio_seconds, cpu_seconds)
    watts = trace.per_second_watts(series, run_first, run_last)
    idle_watts = (trace.per_second_watts(series, marks[f"idle{rep}_start"], marks[f"run{rep}_start"])
                  + trace.per_second_watts(series, marks[f"idle{rep + 1}_start"], marks[after_end]))
    # Bursts from other work only ever add power, so the median second is close to the undisturbed level.
    run_median = statistics.median(watts) if watts else None
    idle_median = statistics.median(idle_watts) if idle_watts else None
    robust_net = run_median - idle_median if watts and idle_watts else None
    metrics.update({
        "idle_before_watts": before[0] / before[1],
        "idle_after_watts": after[0] / after[1],
        "run_window_seconds": run_seconds,
        "per_second_watts_stdev": statistics.pstdev(watts) if len(watts) > 1 else None,
        "run_watts_median": run_median,
        "idle_watts_median": idle_median,
        "net_watts_median": robust_net,
        "joules_per_audio_s_median": (robust_net * run_seconds / audio_seconds
                                      if robust_net is not None and audio_seconds else None),
        "joules_per_cpu_s_median": (robust_net * run_seconds / cpu_seconds
                                    if robust_net is not None and cpu_seconds else None),
    })
    return metrics


def run_main(args):
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise SystemExit("energy_bench run: no command given after --")
    cpus = parse_cpu_list(args.cpus)
    cores = physical_cores()
    core_of = {thread: first for first, threads in cores.items() for thread in threads}
    measured = sorted({core_of[c] for c in cpus})
    siblings = sorted({t for c in measured for t in cores[c]} - set(cpus))
    if len(measured) != len(cpus):
        raise SystemExit("energy_bench run: --cpus names two threads of one physical core")
    if args.observer_cpu is not None:
        if core_of[args.observer_cpu] in measured:
            raise SystemExit("energy_bench run: the observer CPU shares a core with the workload")
        os.sched_setaffinity(0, {args.observer_cpu})
    binary = shutil.which(command[0]) or command[0]
    hashes = {str(binary): sha256_path(binary)}
    for extra in args.hash or []:
        hashes[str(extra)] = sha256_path(extra)
    audio_per_worker = args.audio_seconds
    audio_files = None
    if args.wav_dir:
        audio_per_worker, audio_files = wav_dir_seconds(args.wav_dir)
    audio_digest = sha256_file(args.audio_manifest) if args.audio_manifest else None
    env = dict(os.environ)
    extra_env = {}
    for item in args.env or []:
        key, _, value = item.partition("=")
        env[key] = extra_env[key] = value
    if args.log_dir:
        Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    facts = host_facts(cpus)
    budget = 600 + args.reps * (args.max_run_seconds + args.idle_seconds + args.settle_seconds) + args.idle_seconds
    worker_cpus = cpus[:min(args.workers, len(cpus))]  # frequency is read where the workers run
    sampler = Sampler(measured, args.observer_cpu, args.interval, budget, worker_cpus, sudo=not args.no_sudo)
    stats = {}
    reps = []

    def mark(label):
        sampler.mark(label)
        stats[label] = read_proc_stat()

    try:
        time.sleep(args.settle_seconds)
        for rep in range(1, args.reps + 1):
            mark(f"idle{rep}_start")
            time.sleep(args.idle_seconds)
            load_before = read_text("/proc/loadavg")
            started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            mark(f"run{rep}_start")
            began = time.monotonic()
            children = []
            for worker in range(args.workers):
                cpu = cpus[worker % len(cpus)]
                children.append(launch(substitute(command, rep, worker, cpu), {cpu}, env, args.log_dir, rep, worker))
            cpu_seconds, exit_codes, max_rss = 0.0, [], 0
            for child in children:
                _, status, usage = os.wait4(child.pid, 0)
                child.returncode = os.waitstatus_to_exitcode(status)
                exit_codes.append(child.returncode)
                cpu_seconds += usage.ru_utime + usage.ru_stime
                max_rss = max(max_rss, usage.ru_maxrss * 1024)
            wall = time.monotonic() - began
            mark(f"run{rep}_end")
            reps.append({
                "rep": rep, "started_utc": started, "wall_seconds": wall, "cpu_seconds": cpu_seconds,
                "exit_codes": exit_codes, "peak_rss_bytes": max_rss,
                "loadavg_before": load_before, "loadavg_after": read_text("/proc/loadavg"),
            })
            time.sleep(args.settle_seconds)
        mark(f"idle{args.reps + 1}_start")
        time.sleep(args.idle_seconds)
        mark(f"idle{args.reps + 1}_end")
    finally:
        sampler.stop()

    trace = Trace(sampler.header, sampler.samples) if sampler.header["msr"] else None
    package_trace = trace
    if trace is None:
        header = dict(sampler.header, energy_unit_j=1.0, cpus=[])
        package_trace = Trace(header, sampler.samples)
    marks = package_trace.marks
    package_series = package_trace.package if package_trace.package is not None else package_trace.package_sysfs
    core_series = trace.core_sum() if trace else None
    thresholds = {
        "sibling_busy": args.max_sibling_busy, "foreign_watts": args.max_foreign_watts,
        "idle_drift_core": args.max_idle_drift_core, "idle_drift_package": args.max_idle_drift_package,
        "on_cpu": args.min_on_cpu, "rep_spread_pct": args.max_rep_spread_pct,
    }
    elsewhere = sorted(set(stats["idle1_start"]) - set(cpus) - set(siblings) - {args.observer_cpu})
    busy_workers = 0 if args.audio_rate or args.multi_threaded else args.workers
    total_audio = audio_per_worker * args.workers if audio_per_worker else None
    for rep in reps:
        number = rep["rep"]
        common = (marks, number, args.reps, total_audio, rep["cpu_seconds"], args.trim_seconds,
                  args.audio_rate, args.workers)
        if core_series is not None:
            rep["core"] = view_for_rep(trace, core_series, *common)
            rep["net_watts_by_core"] = {
                str(cpu): view_for_rep(trace, trace.core[cpu], *common)["net_watts"] for cpu in measured
            }
        if package_series is not None:
            rep["package"] = view_for_rep(package_trace, package_series, *common)
        if trace and trace.package_sysfs is not None and trace.package is not None:
            sysfs = trace.span(trace.package_sysfs, marks[f"run{number}_start"], marks[f"run{number}_end"])
            msr = trace.span(trace.package, marks[f"run{number}_start"], marks[f"run{number}_end"])
            rep["package_sysfs_over_msr"] = sysfs[0] / msr[0] if msr[0] else None
        run_start, run_end = f"run{number}_start", f"run{number}_end"
        idle_a, idle_b = f"idle{number}_start", f"idle{number + 1}_start"
        idle_b_end = f"idle{number + 1}_end" if number == args.reps else f"run{number + 1}_start"
        if trace:
            run_other = trace.other_cores_joules(run_start, run_end, set(measured))
            idle_other = [trace.other_cores_joules(idle_a, run_start, set(measured)),
                          trace.other_cores_joules(idle_b, idle_b_end, set(measured))]
            times = trace.times
            if run_other is not None and None not in idle_other:
                idle_times = [times[marks[run_start]] - times[marks[idle_a]],
                              times[marks[idle_b_end]] - times[marks[idle_b]]]
                rep["other_cores"] = {
                    "run_watts": run_other / (times[marks[run_end]] - times[marks[run_start]]),
                    "idle_watts": sum(idle_other) / sum(idle_times),
                    "idle_before_watts": idle_other[0] / idle_times[0],
                    "idle_after_watts": idle_other[1] / idle_times[1],
                }
        in_run = [mhz for t, mhz in sampler.frequency
                  if mhz and package_trace.times[marks[run_start]] <= t <= package_trace.times[marks[run_end]]]
        rep["mean_mhz"] = statistics.fmean(in_run) if in_run else None
        rep["busy"] = {
            "sibling_busy_fraction_run": busy_fraction(stats[run_start], stats[run_end], siblings) if siblings else 0.0,
            "pinned_busy_fraction_run": busy_fraction(stats[run_start], stats[run_end], cpus),
            "pinned_busy_fraction_idle": busy_fraction(stats[idle_a], stats[run_start], cpus),
            "elsewhere_busy_cpus_run": busy_fraction(stats[run_start], stats[run_end], elsewhere) * len(elsewhere),
            "elsewhere_busy_cpus_idle": busy_fraction(stats[idle_a], stats[run_start], elsewhere) * len(elsewhere),
        }
        audio = total_audio
        if args.audio_rate:
            audio = args.audio_rate * args.workers * rep["wall_seconds"]
        rep["audio_seconds"] = audio
        rep["cpu_s_per_audio_s"] = rep["cpu_seconds"] / audio if audio else None
        rep["flags"] = flags_for(rep, thresholds, len(measured), busy_workers)

    receipt = {
        "schema": SCHEMA,
        "arm": args.arm,
        "note": args.note,
        "machine": facts["hostname"],
        "energy_method": ENERGY_METHOD if trace else ENERGY_METHOD_PACKAGE_ONLY,
        **headline(reps, thresholds),
        "reps": reps,
        "command": command,
        "env": extra_env,
        "cpus": cpus,
        "measured_cores": measured,
        "smt_siblings_kept_idle": siblings,
        "observer_cpu": args.observer_cpu,
        "workers": args.workers,
        "audio_seconds_per_worker": audio_per_worker,
        "audio_files_per_worker": audio_files,
        "audio_rate_per_worker": args.audio_rate,
        "audio_manifest_sha256": audio_digest,
        "trim_seconds": args.trim_seconds,
        "idle_seconds": args.idle_seconds,
        "settle_seconds": args.settle_seconds,
        "interval_seconds": args.interval,
        "samples": len(sampler.samples),
        "energy_unit_j": sampler.header["energy_unit_j"],
        "thresholds": thresholds,
        "busy_workers": busy_workers if reps else 0,
        "host": facts,
        "sha256": hashes,
        "tool_sha256": sha256_file(Path(__file__).resolve()),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rounded(receipt), indent=1) + "\n", encoding="utf-8")
    if args.series_out:
        origin = package_trace.times[0]
        series = {
            "t": [moment - origin for moment in package_trace.times],
            "labels": {label: index for label, index in marks.items()},
            "core_joules": core_series,
            "core_joules_by_cpu": {str(cpu): trace.core[cpu] for cpu in measured} if trace else None,
            "package_joules": package_series,
            "mhz": [mhz for _, mhz in sampler.frequency],
        }
        Path(args.series_out).write_text(json.dumps(rounded(series, 9)) + "\n", encoding="utf-8")
    print(one_line(rounded(receipt)))
    return 1 if "nonzero_exit" in receipt["flags"] else 0


def one_line(receipt):
    def show(value, unit=""):
        return "n/a" if value is None else f"{value:.4g}{unit}"

    return (f"{receipt['arm']}: {show(receipt['joules_per_audio_s'])} J/audio-s core, "
            f"{show(receipt['joules_per_audio_s_package'])} package; "
            f"{show(receipt['watts_per_busy_core'], ' W')}/busy core core, "
            f"{show(receipt['watts_per_busy_core_package'], ' W')} package; "
            f"idle package {show(receipt['idle_watts'], ' W')}; "
            f"cpu-s/audio-s {show(receipt['cpu_s_per_audio_s'])}; flags {receipt['flags'] or 'none'}")


# ----------------------------------------------------------------------------
# Validation workload
# ----------------------------------------------------------------------------

def spin_main(args):
    """A pure busy loop for a fixed time (integer arithmetic, no I/O, no SIMD)."""
    end = time.monotonic() + args.seconds
    value = 1
    while time.monotonic() < end:
        for _ in range(20000):
            value = (value * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
    return 0


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)

    run = sub.add_parser("run", help="measure a command with idle windows before and after")
    run.add_argument("--arm", required=True)
    run.add_argument("--note", default=None)
    run.add_argument("--cpus", required=True, help="one hardware thread per physical core, e.g. 16 or 16-23")
    run.add_argument("--observer-cpu", type=int, default=None, help="CPU for this tool and its counter helper")
    run.add_argument("--workers", type=int, default=1, help="copies of the command; worker i runs on cpus[i mod n]")
    run.add_argument("--reps", type=int, default=3)
    run.add_argument("--idle-seconds", type=float, default=60.0)
    run.add_argument("--settle-seconds", type=float, default=5.0)
    run.add_argument("--interval", type=float, default=1.0)
    run.add_argument("--max-run-seconds", type=float, default=900.0, help="only sizes the helper's lifetime")
    run.add_argument("--audio-seconds", type=float, default=None, help="audio each worker processes per repetition")
    run.add_argument("--wav-dir", default=None, help="take the audio seconds from the WAV headers in this directory")
    run.add_argument("--audio-rate", type=float, default=None,
                     help="audio seconds per wall second per worker (1.0 for streams paced at 1x)")
    run.add_argument("--audio-manifest", default=None, help="file whose SHA-256 identifies the audio set")
    run.add_argument("--trim-seconds", type=float, default=0.0, help="leave this much out at both ends of the run")
    run.add_argument("--hash", action="append", help="model file or directory to hash into the receipt")
    run.add_argument("--env", action="append", help="KEY=VALUE for the command")
    run.add_argument("--log-dir", default=None)
    run.add_argument("--series-out", default=None, help="per-second cumulative joules (large, not for git)")
    run.add_argument("--out", required=True)
    run.add_argument("--no-sudo", action="store_true", help="the caller is already allowed to read the counters")
    run.add_argument("--multi-threaded", action="store_true",
                     help="the command is not one busy thread per worker; skip the on-CPU check")
    run.add_argument("--max-sibling-busy", type=float, default=DEFAULT_THRESHOLDS["sibling_busy"])
    run.add_argument("--max-foreign-watts", type=float, default=DEFAULT_THRESHOLDS["foreign_watts"])
    run.add_argument("--max-idle-drift-core", type=float, default=DEFAULT_THRESHOLDS["idle_drift_core"])
    run.add_argument("--max-idle-drift-package", type=float, default=DEFAULT_THRESHOLDS["idle_drift_package"])
    run.add_argument("--min-on-cpu", type=float, default=DEFAULT_THRESHOLDS["on_cpu"])
    run.add_argument("--max-rep-spread-pct", type=float, default=DEFAULT_THRESHOLDS["rep_spread_pct"])
    run.add_argument("command", nargs=argparse.REMAINDER)
    run.set_defaults(handler=run_main)

    sample = sub.add_parser("sample", help="(root helper) print counter samples; reads only")
    sample.add_argument("--cpus", required=True)
    sample.add_argument("--interval", type=float, default=1.0)
    sample.add_argument("--max-seconds", type=float, default=7200.0)
    sample.add_argument("--pin-cpu", type=int, default=None)
    sample.add_argument("--all-cores", action="store_true")
    sample.set_defaults(handler=sampler_main)

    spin = sub.add_parser("spin", help="busy loop for a fixed time")
    spin.add_argument("--seconds", type=float, required=True)
    spin.set_defaults(handler=spin_main)

    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
