//! Linux measurement placement. The observer and service use disjoint L3/SMT domains.
//! Service affinity is set and checked before exec; later worker drift is not monitored.

use std::collections::BTreeSet;
use std::io;
use std::marker::PhantomData;
use std::os::unix::process::CommandExt;
use std::process::{Child, Command};
use std::rc::Rc;

use nix::sched::{sched_getaffinity, sched_setaffinity, CpuSet};
use nix::unistd::Pid;
use serde::{Deserialize, Serialize};

use crate::EvalError;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlacementSpec {
    pub service_cpus: String,
    pub observer_cpus: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlacementReport {
    pub configured_service_cpus: Vec<usize>,
    pub configured_observer_cpus: Vec<usize>,
    pub effective_service_cpus_at_spawn: Vec<usize>,
    pub effective_observer_cpus: Vec<usize>,
    pub separate_l3_and_smt: bool,
    pub verification_scope: String,
}

pub fn parse_cpus(text: &str) -> Result<BTreeSet<usize>, EvalError> {
    let mut cpus = BTreeSet::new();
    let bad = || EvalError::Input(format!("invalid CPU list {text:?}"));
    for part in text.split(',') {
        let part = part.trim();
        let (lo, hi) = match part.split_once('-') {
            Some((a, b)) => (
                a.trim().parse().map_err(|_| bad())?,
                b.trim().parse().map_err(|_| bad())?,
            ),
            None => {
                let cpu: usize = part.parse().map_err(|_| bad())?;
                (cpu, cpu)
            }
        };
        if lo > hi || hi >= libc::CPU_SETSIZE as usize {
            return Err(bad());
        }
        cpus.extend(lo..=hi);
    }
    if cpus.is_empty() {
        return Err(bad());
    }
    Ok(cpus)
}

impl PlacementSpec {
    /// Fail closed on unknown CPUs or unavailable topology. No affinity is changed here.
    pub fn validate(&self) -> Result<(BTreeSet<usize>, BTreeSet<usize>), EvalError> {
        let service = parse_cpus(&self.service_cpus)?;
        let observer = parse_cpus(&self.observer_cpus)?;
        let online = parse_cpus(std::fs::read_to_string("/sys/devices/system/cpu/online")?.trim())?;
        if !service.is_subset(&online) || !observer.is_subset(&online) {
            return Err(EvalError::Input(
                "placement includes offline/nonexistent CPUs".into(),
            ));
        }
        let mut domains = service.clone();
        for cpu in &service {
            let root = format!("/sys/devices/system/cpu/cpu{cpu}");
            let level = std::fs::read_to_string(format!("{root}/cache/index3/level"))?;
            if level.trim() != "3" {
                return Err(EvalError::Input(
                    "cannot verify L3 placement on this host".into(),
                ));
            }
            for relative in [
                "topology/thread_siblings_list",
                "cache/index3/shared_cpu_list",
            ] {
                domains.extend(parse_cpus(
                    std::fs::read_to_string(format!("{root}/{relative}"))?.trim(),
                )?);
            }
        }
        if !domains.is_disjoint(&observer) {
            return Err(EvalError::Input(
                "observer overlaps service CPU, SMT sibling or L3 domain".into(),
            ));
        }
        Ok((service, observer))
    }

    pub(crate) fn bind(&self) -> Result<ObserverBinding, EvalError> {
        let (service, observer) = self.validate()?;
        let original = current_mask()?;
        let mask = make_mask(&observer)?;
        if let Err(e) = bind_exact(&mask) {
            // A failed exact check may follow a successful, partially intersected set.
            let _ = bind_exact(&original);
            return Err(e.into());
        }
        Ok(ObserverBinding {
            original,
            service: make_mask(&service)?,
            observer: mask,
            report: PlacementReport {
                configured_service_cpus: service.iter().copied().collect(),
                configured_observer_cpus: observer.iter().copied().collect(),
                effective_service_cpus_at_spawn: service.into_iter().collect(),
                effective_observer_cpus: observer.into_iter().collect(),
                separate_l3_and_smt: true,
                verification_scope: "service mask checked in child before exec; observer checked at bind and after drain; later service/thread affinity drift not monitored".into(),
            },
            _same_thread: PhantomData,
        })
    }
}

fn make_mask(cpus: &BTreeSet<usize>) -> io::Result<CpuSet> {
    let mut mask = CpuSet::new();
    for &cpu in cpus {
        mask.set(cpu)?;
    }
    Ok(mask)
}

fn current_mask() -> io::Result<CpuSet> {
    sched_getaffinity(Pid::from_raw(0)).map_err(Into::into)
}

// Only syscalls, fixed stack storage and raw OS errors: also safe to call after fork.
fn bind_exact(mask: &CpuSet) -> io::Result<()> {
    sched_setaffinity(Pid::from_raw(0), mask)?;
    if current_mask()? != *mask {
        return Err(io::Error::from_raw_os_error(libc::EINVAL));
    }
    Ok(())
}

pub(crate) struct ObserverBinding {
    original: CpuSet,
    service: CpuSet,
    observer: CpuSet,
    report: PlacementReport,
    // Affinity belongs to the calling thread; never restore it on another thread.
    _same_thread: PhantomData<Rc<()>>,
}

impl ObserverBinding {
    pub(crate) fn spawn(&self, command: &mut Command) -> io::Result<Child> {
        let service = self.service;
        // SAFETY: the closure uses only sched_* syscalls, fixed stack storage and raw
        // OS errors; it allocates nothing, takes no locks and checks the mask before exec.
        unsafe {
            command.pre_exec(move || bind_exact(&service));
        }
        command.spawn()
    }

    pub(crate) fn report(&self) -> Result<PlacementReport, EvalError> {
        if current_mask()? != self.observer {
            return Err(EvalError::Input(
                "observer affinity changed during measurement".into(),
            ));
        }
        Ok(self.report.clone())
    }
}

impl Drop for ObserverBinding {
    fn drop(&mut self) {
        let _ = bind_exact(&self.original);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_lists_are_bounded_and_strict() {
        assert_eq!(
            parse_cpus("1,3-5,3")
                .unwrap()
                .into_iter()
                .collect::<Vec<_>>(),
            vec![1, 3, 4, 5]
        );
        for bad in ["", "1,", "2-1", "-1", "1,,2", "999999", "0-999999"] {
            assert!(parse_cpus(bad).is_err(), "{bad}");
        }
    }
}
