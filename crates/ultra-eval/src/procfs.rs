//! Linux process-tree resource accounting (v1.0-B §12.3; v1 §11.3 "Cost", "Memory").
//!
//! CPU: `/proc/<pid>/stat` utime+stime already aggregate every thread of a process,
//! live and exited, so `/proc/<pid>/task/*` is enumerated only for the observed thread
//! count (summing task times as well would double count). cutime+cstime add children
//! the process has reaped. A sample sums utime+stime+cutime+cstime over the live tree
//! (root plus all descendants by ppid walk). A process that vanishes between samples
//! was reaped either by a live tree member (its time is now in that member's cutime)
//! or by an outsider; in the second case its last observed time is kept as a lower
//! bound. Processes are keyed by (pid, starttime) to survive pid reuse. The cumulative
//! CPU reported is the running maximum over samples.
//!
//! Memory: peak RSS is the maximum over samples of the summed `VmHWM` of live tree
//! members (exact for one process; a conservative upper bound for several, whose
//! peaks need not coincide). PSS is the maximum over samples of summed `Pss` from
//! `smaps_rollup`, so a spike between samples can be missed. Threads: maximum over
//! samples of summed task counts. Idle waiting costs no CPU ticks and so is excluded.
//!
//! For exact totals of a command we spawn, see [`measure_command`]: it becomes a child
//! subreaper so orphans re-parent to it, waits for the whole tree, and takes CPU from
//! `getrusage(RUSAGE_CHILDREN)` (microsecond resolution, every reaped descendant).

use std::collections::{BTreeMap, BTreeSet};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::EvalError;

/// Fields of `/proc/<pid>/stat` used here (times in clock ticks).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProcStat {
    pub pid: i32,
    pub ppid: i32,
    pub utime: u64,
    pub stime: u64,
    pub cutime: u64,
    pub cstime: u64,
    pub num_threads: u64,
    pub starttime: u64,
}

/// Parse one `/proc/<pid>/stat` line. The command name may contain spaces and
/// parentheses, so fields are read after the LAST `)`.
pub fn parse_stat(line: &str) -> Option<ProcStat> {
    let open = line.find('(')?;
    let close = line.rfind(')')?;
    let pid = line[..open].trim().parse().ok()?;
    let f: Vec<&str> = line.get(close + 1..)?.split_whitespace().collect();
    // f[0] is field 3 (state); field k is f[k - 3].
    let num = |k: usize| -> Option<u64> {
        let s = f.get(k - 3)?;
        s.parse::<u64>()
            .ok()
            .or_else(|| s.parse::<i64>().ok().map(|v| v.max(0) as u64))
    };
    Some(ProcStat {
        pid,
        ppid: f.get(1)?.parse().ok()?,
        utime: num(14)?,
        stime: num(15)?,
        cutime: num(16)?,
        cstime: num(17)?,
        num_threads: num(20)?,
        starttime: num(22)?,
    })
}

pub fn read_stat(pid: i32) -> Option<ProcStat> {
    parse_stat(&std::fs::read_to_string(format!("/proc/{pid}/stat")).ok()?)
}

/// A `key:   N kB` line from a /proc text file, in bytes.
pub fn parse_kb_field(text: &str, key: &str) -> Option<u64> {
    text.lines().find_map(|l| {
        let rest = l.strip_prefix(key)?.strip_prefix(':')?;
        let kb: u64 = rest.split_whitespace().next()?.parse().ok()?;
        Some(kb * 1024)
    })
}

/// `VmHWM`, `VmRSS`, ... from `/proc/<pid>/status`, in bytes.
pub fn read_status_bytes(pid: i32, key: &str) -> Option<u64> {
    parse_kb_field(
        &std::fs::read_to_string(format!("/proc/{pid}/status")).ok()?,
        key,
    )
}

/// `Pss` from `/proc/<pid>/smaps_rollup`, in bytes (None when unreadable).
pub fn read_pss_bytes(pid: i32) -> Option<u64> {
    parse_kb_field(
        &std::fs::read_to_string(format!("/proc/{pid}/smaps_rollup")).ok()?,
        "Pss",
    )
}

/// Number of entries in `/proc/<pid>/task` (observed threads).
pub fn task_count(pid: i32) -> Option<u64> {
    Some(
        std::fs::read_dir(format!("/proc/{pid}/task"))
            .ok()?
            .filter(|e| e.is_ok())
            .count() as u64,
    )
}

/// Clock ticks per second (`sysconf(_SC_CLK_TCK)`).
pub fn clk_tck() -> u64 {
    // SAFETY: sysconf has no preconditions and only reads a configuration value.
    let v = unsafe { libc::sysconf(libc::_SC_CLK_TCK) };
    if v > 0 {
        v as u64
    } else {
        100
    }
}

/// Stats of every readable process.
pub fn all_stats() -> Vec<ProcStat> {
    let Ok(rd) = std::fs::read_dir("/proc") else {
        return Vec::new();
    };
    rd.filter_map(|e| e.ok()?.file_name().to_str()?.parse::<i32>().ok())
        .filter_map(read_stat)
        .collect()
}

/// `root` and all its descendants (by ppid), from one scan of `stats`.
pub fn tree_of(root: i32, stats: &[ProcStat]) -> BTreeSet<i32> {
    let mut kids: BTreeMap<i32, Vec<i32>> = BTreeMap::new();
    for s in stats {
        kids.entry(s.ppid).or_default().push(s.pid);
    }
    let mut out = BTreeSet::new();
    let mut stack = vec![root];
    while let Some(p) = stack.pop() {
        if out.insert(p) {
            if let Some(k) = kids.get(&p) {
                stack.extend(k);
            }
        }
    }
    out
}

/// One sample of the tree.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct TreeSample {
    pub cpu_seconds_user: f64,
    pub cpu_seconds_sys: f64,
    pub rss_bytes: u64,
    pub hwm_bytes: u64,
    pub pss_bytes: u64,
    pub threads: u64,
    pub processes: u64,
}

/// Peak/cumulative summary over all samples.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct ResourceSummary {
    pub cpu_seconds_user: f64,
    pub cpu_seconds_sys: f64,
    pub peak_rss_bytes: u64,
    pub peak_pss_bytes: u64,
    pub max_threads: u64,
    #[serde(default)]
    pub max_processes: u64,
    pub samples: u64,
}

#[derive(Debug, Clone, Copy)]
struct Last {
    ppid: i32,
    user: u64,
    sys: u64,
}

/// Repeated sampler of one process tree.
#[derive(Debug)]
pub struct TreeSampler {
    root: i32,
    include_root: bool,
    tck: f64,
    seen: BTreeMap<(i32, u64), Last>,
    lost_user: u64,
    lost_sys: u64,
    summary: ResourceSummary,
}

impl TreeSampler {
    /// Account `root` and all of its descendants.
    pub fn new(root: i32) -> Self {
        Self::build(root, true)
    }

    /// Account only the descendants of `root` (e.g. the measuring process itself).
    pub fn descendants_of(root: i32) -> Self {
        Self::build(root, false)
    }

    fn build(root: i32, include_root: bool) -> Self {
        TreeSampler {
            root,
            include_root,
            tck: clk_tck() as f64,
            seen: BTreeMap::new(),
            lost_user: 0,
            lost_sys: 0,
            summary: ResourceSummary::default(),
        }
    }

    /// Take one sample and fold it into the summary.
    pub fn sample(&mut self) -> TreeSample {
        let stats = all_stats();
        let mut tree = tree_of(self.root, &stats);
        if !self.include_root {
            tree.remove(&self.root);
        }
        let mut s = TreeSample::default();
        let mut now: BTreeMap<(i32, u64), Last> = BTreeMap::new();
        let (mut user, mut sys) = (0u64, 0u64);
        for st in stats.iter().filter(|st| tree.contains(&st.pid)) {
            let l = Last {
                ppid: st.ppid,
                user: st.utime + st.cutime,
                sys: st.stime + st.cstime,
            };
            user += l.user;
            sys += l.sys;
            now.insert((st.pid, st.starttime), l);
            s.processes += 1;
            s.threads += task_count(st.pid).unwrap_or(st.num_threads);
            s.rss_bytes += read_status_bytes(st.pid, "VmRSS").unwrap_or(0);
            s.hwm_bytes += read_status_bytes(st.pid, "VmHWM").unwrap_or(0);
            s.pss_bytes += read_pss_bytes(st.pid).unwrap_or(0);
        }
        for (key, last) in &self.seen {
            if !now.contains_key(key) && !tree.contains(&last.ppid) {
                // Reaped outside the accounted tree: keep the last observation.
                self.lost_user += last.user;
                self.lost_sys += last.sys;
            }
        }
        self.seen = now;
        s.cpu_seconds_user = (user + self.lost_user) as f64 / self.tck;
        s.cpu_seconds_sys = (sys + self.lost_sys) as f64 / self.tck;
        let m = &mut self.summary;
        m.cpu_seconds_user = m.cpu_seconds_user.max(s.cpu_seconds_user);
        m.cpu_seconds_sys = m.cpu_seconds_sys.max(s.cpu_seconds_sys);
        m.peak_rss_bytes = m.peak_rss_bytes.max(s.hwm_bytes.max(s.rss_bytes));
        m.peak_pss_bytes = m.peak_pss_bytes.max(s.pss_bytes);
        m.max_threads = m.max_threads.max(s.threads);
        m.max_processes = m.max_processes.max(s.processes);
        m.samples += 1;
        s
    }

    pub fn summary(&self) -> ResourceSummary {
        self.summary
    }
}

/// Result of [`measure_command`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Measurement {
    pub cpu_seconds_user: f64,
    pub cpu_seconds_sys: f64,
    pub wall_seconds: f64,
    pub peak_rss_bytes: u64,
    pub peak_pss_bytes: u64,
    pub max_threads: u64,
    pub exit_code: Option<i32>,
    pub samples: u64,
    /// The sampled (tick-resolution) CPU total, for cross-checking getrusage.
    pub sampled_cpu_seconds: f64,
    pub clk_tck: u64,
    #[serde(default)]
    pub max_processes: Option<u64>,
    #[serde(default)]
    pub observer_cpu_seconds: Option<f64>,
    #[serde(default)]
    pub interval_ms: Option<u64>,
    #[serde(default)]
    pub placement: Option<crate::affinity::PlacementReport>,
}

fn sleep_until(deadline: Instant) {
    loop {
        let now = Instant::now();
        if now >= deadline {
            return;
        }
        std::thread::sleep(deadline - now);
    }
}

/// Run `program args...`, sample its whole process tree every `interval`, wait for
/// every descendant (orphans included), and return exact CPU totals. The child's
/// stdout is redirected to this process's stderr so stdout stays machine-readable.
///
/// Makes the calling process a child subreaper and reaps with `waitpid(-1)`: call it
/// from a process that has no other children (the `ultra-measure` binary).
pub fn measure_command(
    program: &str,
    args: &[String],
    interval: Duration,
) -> Result<Measurement, EvalError> {
    measure_command_with_placement(program, args, interval, None)
}

/// Optional disjoint observer/service placement; callers must still isolate this process.
pub fn measure_command_with_placement(
    program: &str,
    args: &[String],
    interval: Duration,
    placement: Option<&crate::affinity::PlacementSpec>,
) -> Result<Measurement, EvalError> {
    let interval_ms: u64 = interval
        .as_millis()
        .try_into()
        .map_err(|_| EvalError::Input("measurement interval too large".into()))?;
    if interval_ms == 0 || Duration::from_millis(interval_ms) != interval {
        return Err(EvalError::Input(
            "measurement interval must be whole positive milliseconds".into(),
        ));
    }
    let observer_cpu_before = observer_cpu_s()?;
    let binding = placement
        .map(crate::affinity::PlacementSpec::bind)
        .transpose()?;
    // SAFETY: prctl(PR_SET_CHILD_SUBREAPER, 1) only sets a flag on this process.
    if unsafe { libc::prctl(libc::PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) } != 0 {
        return Err(std::io::Error::last_os_error().into());
    }
    let me = std::process::id() as i32;
    let start = Instant::now();
    let mut command = std::process::Command::new(program);
    command
        .args(args)
        .stdout(std::process::Stdio::from(std::io::stderr()));
    let child = match &binding {
        Some(bound) => bound.spawn(&mut command)?,
        None => command.spawn()?,
    };
    let child_pid = child.id() as i32;
    let mut sampler = TreeSampler::descendants_of(me);
    let mut exit_code = None;
    let mut root_done = false;
    let mut next = start;
    loop {
        sampler.sample();
        let mut no_children = false;
        loop {
            let mut status: libc::c_int = 0;
            // SAFETY: status is a valid out-pointer; WNOHANG never blocks.
            let r = unsafe { libc::waitpid(-1, &mut status, libc::WNOHANG) };
            if r == child_pid {
                root_done = true;
                exit_code = if libc::WIFEXITED(status) {
                    Some(libc::WEXITSTATUS(status))
                } else {
                    None
                };
            } else if r == 0 {
                break;
            } else if r < 0 {
                match std::io::Error::last_os_error().raw_os_error() {
                    Some(libc::EINTR) => continue,
                    Some(libc::ECHILD) => {
                        no_children = true;
                        break;
                    }
                    _ => return Err(std::io::Error::last_os_error().into()),
                }
            }
        }
        if root_done && no_children {
            break;
        }
        next += interval;
        sleep_until(next);
    }
    drop(child);
    // Wall time runs until the last descendant (orphans included) is reaped.
    let wall = start.elapsed();
    let ru = nix::sys::resource::getrusage(nix::sys::resource::UsageWho::RUSAGE_CHILDREN)
        .map_err(|e| EvalError::Io(format!("getrusage failed: {e}")))?;
    let tv = |t: nix::sys::time::TimeVal| t.tv_sec() as f64 + t.tv_usec() as f64 / 1e6;
    let s = sampler.summary();
    Ok(Measurement {
        cpu_seconds_user: tv(ru.user_time()),
        cpu_seconds_sys: tv(ru.system_time()),
        wall_seconds: wall.as_secs_f64(),
        peak_rss_bytes: s.peak_rss_bytes.max(ru.max_rss().max(0) as u64 * 1024),
        peak_pss_bytes: s.peak_pss_bytes,
        max_threads: s.max_threads,
        exit_code,
        samples: s.samples,
        sampled_cpu_seconds: s.cpu_seconds_user + s.cpu_seconds_sys,
        clk_tck: clk_tck(),
        max_processes: Some(s.max_processes),
        observer_cpu_seconds: Some((observer_cpu_s()? - observer_cpu_before).max(0.0)),
        interval_ms: Some(interval_ms),
        placement: binding
            .as_ref()
            .map(crate::affinity::ObserverBinding::report)
            .transpose()?,
    })
}

fn observer_cpu_s() -> Result<f64, EvalError> {
    let r = nix::sys::resource::getrusage(nix::sys::resource::UsageWho::RUSAGE_SELF)
        .map_err(|e| EvalError::Io(format!("observer getrusage: {e}")))?;
    let seconds = |t: nix::sys::time::TimeVal| t.tv_sec() as f64 + t.tv_usec() as f64 / 1e6;
    Ok(seconds(r.user_time()) + seconds(r.system_time()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_stat_with_hostile_comm() {
        let line = "4242 (a b) (c)) S 7 4242 4242 0 -1 4194304 100 0 0 0 \
                    11 22 33 44 20 0 5 0 99999 1000 10 18446744073709551615";
        let s = parse_stat(line).unwrap();
        assert_eq!(s.pid, 4242);
        assert_eq!(s.ppid, 7);
        assert_eq!((s.utime, s.stime, s.cutime, s.cstime), (11, 22, 33, 44));
        assert_eq!(s.num_threads, 5);
        assert_eq!(s.starttime, 99999);
        assert!(parse_stat("garbage").is_none());
    }

    #[test]
    fn parses_kb_fields() {
        let t = "Name:\tx\nVmHWM:\t  2048 kB\nVmRSS:\t1024 kB\n";
        assert_eq!(parse_kb_field(t, "VmHWM"), Some(2048 * 1024));
        assert_eq!(parse_kb_field(t, "VmRSS"), Some(1024 * 1024));
        assert_eq!(parse_kb_field(t, "Pss"), None);
    }

    #[test]
    fn self_is_readable() {
        let me = std::process::id() as i32;
        let s = read_stat(me).unwrap();
        assert_eq!(s.pid, me);
        assert!(read_status_bytes(me, "VmHWM").unwrap() > 0);
        assert!(task_count(me).unwrap() >= 1);
        assert!(clk_tck() >= 1);
    }

    #[test]
    fn sampler_sees_a_burning_child_and_its_threads() {
        let mut child = std::process::Command::new("sh")
            .args(["-c", "i=0; while [ $i -lt 400000 ]; do i=$((i+1)); done"])
            .spawn()
            .unwrap();
        let pid = child.id() as i32;
        let mut t = TreeSampler::new(pid);
        let start = Instant::now();
        while child.try_wait().unwrap().is_none() && start.elapsed() < Duration::from_secs(30) {
            t.sample();
            std::thread::sleep(Duration::from_millis(50));
        }
        let s = t.summary();
        assert!(s.samples >= 1);
        assert!(s.max_threads >= 1);
        assert!(s.peak_rss_bytes > 0);
        if start.elapsed() > Duration::from_millis(300) {
            assert!(s.cpu_seconds_user + s.cpu_seconds_sys > 0.0, "{s:?}");
        }
    }

    #[test]
    fn sampler_of_sleeping_child_reports_no_cpu() {
        let mut child = std::process::Command::new("sleep")
            .arg("0.5")
            .spawn()
            .unwrap();
        let mut t = TreeSampler::new(child.id() as i32);
        for _ in 0..5 {
            t.sample();
            std::thread::sleep(Duration::from_millis(50));
        }
        child.wait().unwrap();
        let s = t.summary();
        assert!(s.cpu_seconds_user + s.cpu_seconds_sys < 0.05, "{s:?}");
    }
}
