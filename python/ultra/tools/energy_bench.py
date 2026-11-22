"""Energy per audio-second (= watts per live stream) for a command on pinned cores.

A read-only root helper samples the AMD per-core energy register (MSR 0xC001029A)
and the package energy register (MSR 0xC001029B, plus the powercap file) once a
second while an unprivileged command runs on pinned cores, with an idle window
before and after. Standard library only. Protocol: research/H39_energy/README.md.

    energy_bench.py run --arm NAME --cpus 16 --observer-cpu 31 --audio-seconds S
        --out receipt.json -- COMMAND [ARGS...]     ({rep}, {worker}, {cpu} are substituted)
    energy_bench.py spin --seconds 30               (busy loop, the validation workload)
    energy_bench.py fit RECEIPT...                  (watts against busy cores, least squares)
    energy_bench.py table --ours ARM RECEIPT...     (one Markdown table)
    energy_bench.py sidecar --row ID RECEIPT        (energy lines for the paper evidence file)
    energy_bench.py uplift RECEIPT...               (package watts per core-register watt)
    energy_bench.py resummarise RECEIPT             (re-apply the distrust rules to stored repetitions)
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


def linear_fit(xs, ys):
    """Least-squares line with R^2 and the largest residual relative to the fitted value."""
    if len(xs) != len(ys) or len(xs) < 2 or len(set(xs)) < 2:
        raise ValueError("a line needs at least two distinct x values")
    mean_x, mean_y = statistics.fmean(xs), statistics.fmean(ys)
    sxx = sum((x - mean_x) ** 2 for x in xs)
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / sxx
    intercept = mean_y - slope * mean_x
    fitted = [intercept + slope * x for x in xs]
    ss_res = sum((y - f) ** 2 for y, f in zip(ys, fitted))
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    return {
        "slope": slope,
        "intercept": intercept,
        "r2": 1.0 - ss_res / ss_tot if ss_tot else 1.0,
        "max_rel_residual": max(abs(y - f) / abs(f) if f else math.inf for y, f in zip(ys, fitted)),
    }


def least_squares(rows, ys):
    """Ordinary least squares with an intercept: ys ~ a + sum(b_i * rows[.][i]).

    Returns {"intercept", "coefficients", "r2", "residual_stdev", "residuals", "n"}.
    """
    if not rows or len(rows) != len(ys) or len(rows) <= len(rows[0]) + 1:
        raise ValueError("not enough points for the fit")
    design = [[1.0, *row] for row in rows]
    size = len(design[0])
    normal = [[sum(r[i] * r[j] for r in design) for j in range(size)] + [sum(r[i] * y for r, y in zip(design, ys))]
              for i in range(size)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda r: abs(normal[r][column]))
        if abs(normal[pivot][column]) < 1e-12:
            raise ValueError("the regressors do not vary independently")
        normal[column], normal[pivot] = normal[pivot], normal[column]
        for other in range(size):
            if other != column:
                factor = normal[other][column] / normal[column][column]
                normal[other] = [a - factor * b for a, b in zip(normal[other], normal[column])]
    beta = [normal[i][size] / normal[i][i] for i in range(size)]
    residuals = [y - sum(b * x for b, x in zip(beta, r)) for r, y in zip(design, ys)]
    mean_y = statistics.fmean(ys)
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    ss_res = sum(e * e for e in residuals)
    return {
        "intercept": beta[0],
        "coefficients": beta[1:],
        "r2": 1.0 - ss_res / ss_tot if ss_tot else 1.0,
        "residual_stdev": math.sqrt(ss_res / (len(ys) - size)),
        "residuals": residuals,
        "n": len(ys),
    }


def phase_points(receipt):
    """(label, pinned-core watts, other-core watts, package watts) for every phase of a receipt.

    The idle window between two repetitions is counted once.
    """
    points = []
    reps = receipt["reps"]
    for index, rep in enumerate(reps):
        core, other, package = rep.get("core"), rep.get("other_cores"), rep.get("package")
        if not (core and other and package):
            continue
        points.append((f"{receipt['arm']}:run{rep['rep']}", core["run_watts"], other["run_watts"], package["run_watts"]))
        if "idle_before_watts" not in other:  # older receipts kept only the two idle windows together
            points.append((f"{receipt['arm']}:idle{rep['rep']}", core["idle_watts"], other["idle_watts"],
                           package["idle_watts"]))
            continue
        points.append((f"{receipt['arm']}:idle{rep['rep']}", core["idle_before_watts"],
                       other["idle_before_watts"], package["idle_before_watts"]))
        if index == len(reps) - 1:
            points.append((f"{receipt['arm']}:idle{rep['rep'] + 1}", core["idle_after_watts"],
                           other["idle_after_watts"], package["idle_after_watts"]))
    return points


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

    def per_second_watts(self, series, first, last):
        return [
            (series[i + 1] - series[i]) / (self.times[i + 1] - self.times[i])
            for i in range(first, last)
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
        "watts_per_busy_core": median("watts_per_busy_core"),
        "watts_per_busy_core_package": median("watts_per_busy_core_package"),
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
    metrics.update({
        "idle_before_watts": before[0] / before[1],
        "idle_after_watts": after[0] / after[1],
        "run_window_seconds": run_seconds,
        "per_second_watts_stdev": statistics.pstdev(watts) if len(watts) > 1 else None,
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
    sampler = Sampler(measured, args.observer_cpu, args.interval, budget, cpus, sudo=not args.no_sudo)
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
        series = {
            "t": package_trace.times,
            "labels": {label: index for label, index in marks.items()},
            "core_joules": core_series,
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
# Validation workload and reporting
# ----------------------------------------------------------------------------

def spin_main(args):
    """A pure busy loop for a fixed time (integer arithmetic, no I/O, no SIMD)."""
    end = time.monotonic() + args.seconds
    value = 1
    while time.monotonic() < end:
        for _ in range(20000):
            value = (value * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
    return 0


def load_receipts(paths):
    receipts = [json.loads(Path(p).read_text(encoding="utf-8")) for p in paths]
    for receipt in receipts:
        if receipt.get("schema") != SCHEMA:
            raise SystemExit(f"energy_bench: {receipt.get('arm')} is not a {SCHEMA} receipt")
    return receipts


def fit_main(args):
    """Net watts against the number of busy workers, per view, over several receipts."""
    receipts = sorted(load_receipts(args.receipts), key=lambda r: r["workers"])
    xs = [r["workers"] for r in receipts]
    result = {"workers": xs, "arms": [r["arm"] for r in receipts]}
    def worker_core_watts(receipt):
        """Median net watts of the cores that host workers (the receipt may have measured more)."""
        reps = receipt.get("reps") or []
        clean = [rep for rep in reps if not CORE_DIRTY & set(rep["flags"])] or reps
        if not clean or not all("net_watts_by_core" in rep for rep in clean):
            return receipt["summary"]["net_watts"]["median"]
        hosts = [str(c) for c in receipt["cpus"][:receipt["workers"]]]
        return statistics.median(sum(rep["net_watts_by_core"][c] for c in hosts) for rep in clean)

    core_watts = [worker_core_watts(r) for r in receipts]
    package_watts = [r["summary"]["net_watts_package"]["median"] for r in receipts]
    result["core"] = dict(linear_fit(xs, core_watts), net_watts=core_watts)
    result["package"] = dict(linear_fit(xs, package_watts), net_watts=package_watts)
    result["clean_repetitions"] = [r.get("clean_repetitions") for r in receipts]
    per_audio = [r["joules_per_audio_s"] for r in receipts]
    if None not in per_audio:
        low, high = min(per_audio), max(per_audio)
        result["joules_per_audio_s"] = per_audio
        result["joules_per_audio_s_range_pct"] = 100.0 * (high - low) / statistics.median(per_audio)
    if result["core"]["slope"]:
        result["package_over_core_slope"] = result["package"]["slope"] / result["core"]["slope"]
    text = json.dumps(rounded(result), indent=1)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


def uplift_main(args):
    """Package watts against core-register watts over every phase of the given receipts.

    package = floor + k_pinned * (pinned cores) + k_other * (all other cores). k is the number
    of package watts that appear per watt the core registers report: the part a per-core figure
    leaves out is (k - 1) per core-register watt.
    """
    receipts = load_receipts(args.receipts)
    points = [point for receipt in receipts for point in phase_points(receipt)]
    both = least_squares([[mine, other] for _, mine, other, _ in points], [pkg for *_, pkg in points])
    one = least_squares([[mine + other] for _, mine, other, _ in points], [pkg for *_, pkg in points])
    result = {
        "arms": [r["arm"] for r in receipts],
        "phases": len(points),
        "all_cores": {"floor_watts": one["intercept"], "package_watts_per_core_watt": one["coefficients"][0],
                      "r2": one["r2"], "residual_stdev_watts": one["residual_stdev"]},
        "split": {"floor_watts": both["intercept"], "pinned_cores": both["coefficients"][0],
                  "other_cores": both["coefficients"][1], "r2": both["r2"],
                  "residual_stdev_watts": both["residual_stdev"]},
        "core_watts_range": [min(m + o for _, m, o, _ in points), max(m + o for _, m, o, _ in points)],
    }
    if args.points:
        result["points"] = [
            {"phase": label, "pinned": mine, "other": other, "package": pkg, "residual": error}
            for (label, mine, other, pkg), error in zip(points, one["residuals"])
        ]
    text = json.dumps(rounded(result), indent=1)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


def resummarise_main(args):
    """Re-evaluate the distrust flags and the headline of a receipt with the current rules."""
    path = Path(args.receipt)
    receipt = load_receipts([path])[0]
    thresholds = dict(DEFAULT_THRESHOLDS)
    busy_workers = receipt.get("busy_workers")
    if busy_workers is None:
        paced_or_threaded = receipt.get("audio_rate_per_worker") or receipt["command"][:1] == ["sleep"]
        busy_workers = 0 if paced_or_threaded else receipt["workers"]
    for rep in receipt["reps"]:
        rep["flags"] = flags_for(rep, thresholds, len(receipt["measured_cores"]), busy_workers)
    previous_tool = receipt.get("resummarised", {}).get("measured_with_tool_sha256", receipt["tool_sha256"])
    receipt.update(headline(receipt["reps"], thresholds))
    receipt["thresholds"] = thresholds
    receipt["busy_workers"] = busy_workers
    receipt["resummarised"] = {
        "measured_with_tool_sha256": previous_tool,
        "rules_from_tool_sha256": sha256_file(Path(__file__).resolve()),
        "note": "flags and headline recomputed from the stored repetitions; no measured value changed",
    }
    Path(args.out or path).write_text(json.dumps(rounded(receipt), indent=1) + "\n", encoding="utf-8")
    print(one_line(rounded(receipt)))
    return 0


def table_main(args):
    receipts = load_receipts(args.receipts)
    ours = next((r for r in receipts if r["arm"] == args.ours), None)
    if ours is None:
        raise SystemExit(f"energy_bench table: no receipt for --ours {args.ours}")

    def cell(value, digits=3):
        return "n/a" if value is None else f"{value:.{digits}g}"

    lines = [
        "| System | CPU-s per audio-s | J per audio-s = W per live stream (core) | same, package view "
        "| W per busy core (core / package) | Ratio to ours (core) | Reps (spread) | Flags |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for r in sorted(receipts, key=lambda r: r["joules_per_audio_s"] or math.inf):
        ratio = r["joules_per_audio_s"] / ours["joules_per_audio_s"] if r["joules_per_audio_s"] else None
        spread = r["summary"]["joules_per_audio_s"]
        lines.append(
            f"| {r['arm']} | {cell(r['cpu_s_per_audio_s'])} | {cell(r['joules_per_audio_s'])} "
            f"| {cell(r['joules_per_audio_s_package'])} "
            f"| {cell(r['watts_per_busy_core'])} / {cell(r['watts_per_busy_core_package'])} "
            f"| {cell(ratio)}x | {r['repetitions']} ({cell(spread['spread_pct'] if spread else None, 2)}%) "
            f"| {', '.join(r['flags']) or 'none'} |"
        )
    text = "\n".join(lines)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


def sidecar_lines(receipt, row, receipt_sha256, session=None):
    """Energy lines for the paper evidence file: one per (row, method), joined to a cost row by `row`.

    core_msr_active   core register while running, idle floor included (runtime-lane's CorWatt basis)
    core_msr_dynamic  core register above idle
    pkg_rapl_delta    package counter above idle (adds the shared uncore power a busy core causes)
    """
    methods = (
        ("core_msr_active", "core", "gross_joules_per_audio_s", "watts_per_busy_core_active"),
        ("core_msr_dynamic", "core", "joules_per_audio_s", "watts_per_busy_core"),
        ("pkg_rapl_delta", "package", "joules_per_audio_s", "watts_per_busy_core_package"),
    )
    lines = []
    for method, view, key, _ in methods:
        reps = [rep[view][key] for rep in receipt["reps"] if view in rep]
        if not reps or None in reps:
            continue
        per_core = [
            (rep[view]["run_watts"] if method == "core_msr_active" else rep[view]["net_watts"])
            * rep[view]["run_window_seconds"] / rep["cpu_seconds"]
            for rep in receipt["reps"] if rep["cpu_seconds"]
        ]
        value = statistics.median(reps)
        lines.append({
            "kind": "energy",
            "row": row,
            "energy_method": method,
            "joules_per_audio_s": value,
            "watts_per_stream": value,
            "energy_reps": reps,
            "energy_idle_w": statistics.median(rep[view]["idle_watts"] for rep in receipt["reps"]),
            "energy_active_w": statistics.median(rep[view]["run_watts"] for rep in receipt["reps"]),
            "watts_per_busy_core": statistics.median(per_core) if per_core else None,
            "energy_cpu_s_per_audio_s": receipt["cpu_s_per_audio_s"],
            "energy_workload_digest": receipt["audio_manifest_sha256"],
            "energy_receipt_sha256": receipt_sha256,
            "energy_host": receipt["machine"],
            "energy_tool": SCHEMA,
            "energy_flags": receipt["flags"],
            "session": session,
        })
    return lines


def sidecar_main(args):
    path = Path(args.receipt)
    receipt = load_receipts([path])[0]
    for line in sidecar_lines(receipt, args.row, sha256_file(path), args.session):
        print(json.dumps(rounded(line), separators=(", ", ": ")))
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

    fit = sub.add_parser("fit", help="watts against busy workers over several receipts")
    fit.add_argument("receipts", nargs="+")
    fit.add_argument("--out", default=None)
    fit.set_defaults(handler=fit_main)

    sidecar = sub.add_parser("sidecar", help="energy lines for docs/paper/data/evidence, one per method")
    sidecar.add_argument("receipt")
    sidecar.add_argument("--row", required=True, help="id of the cost row these lines belong to")
    sidecar.add_argument("--session", default=None)
    sidecar.set_defaults(handler=sidecar_main)

    uplift = sub.add_parser("uplift", help="package watts per core-register watt over all phases")
    uplift.add_argument("receipts", nargs="+")
    uplift.add_argument("--points", action="store_true", help="include every phase and its residual")
    uplift.add_argument("--out", default=None)
    uplift.set_defaults(handler=uplift_main)

    again = sub.add_parser("resummarise", help="recompute flags and headline from stored repetitions")
    again.add_argument("receipt")
    again.add_argument("--out", default=None)
    again.set_defaults(handler=resummarise_main)

    table = sub.add_parser("table", help="one Markdown table over several receipts")
    table.add_argument("receipts", nargs="+")
    table.add_argument("--ours", required=True, help="arm name the ratio column divides by")
    table.add_argument("--out", default=None)
    table.set_defaults(handler=table_main)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
