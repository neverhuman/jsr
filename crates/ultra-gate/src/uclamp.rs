//! Linux opt-in per-thread utilisation clamp (`--uclamp-max`) via `sched_setattr`.
//!
//! Lowers only the calling thread's `sched_util_max` ceiling (`0..=1024`) without modifying any
//! machine-wide CPU governor or frequency setting (compliant with O18). On non-Linux targets,
//! [`set_thread_uclamp_max`] validates the range and succeeds as a safe no-op.

use std::fmt;

/// Maximum Linux scheduler utilisation scale (`1024` = unclamped 100%).
pub const UCLAMP_MAX_LIMIT: u32 = 1024;

/// Typed errors returned by [`set_thread_uclamp_max`], [`get_thread_uclamp_max`], and
/// [`parse_uclamp_max`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UclampError {
    /// Requested utilisation clamp exceeded `0..=1024`.
    OutOfRange { value: u32, max: u32 },
    /// Failed to parse a CLI string into a valid `u32`.
    InvalidFormat(String),
    /// Kernel does not support `CONFIG_UCLAMP_TASK` or `sched_setattr` (`EOPNOTSUPP`/`ENOSYS`).
    Unsupported { errno: i32 },
    /// Kernel denied changing utilisation clamp (`EPERM`/`EACCES`).
    PermissionDenied { errno: i32 },
    /// Any other OS error returned by `sched_setattr`/`sched_getattr`.
    Syscall { errno: i32, message: String },
}

impl fmt::Display for UclampError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutOfRange { value, max } => {
                write!(f, "uclamp_max {value} out of range 0..={max}")
            }
            Self::InvalidFormat(s) => write!(f, "invalid --uclamp-max value: {s}"),
            Self::Unsupported { errno } => write!(
                f,
                "kernel does not support per-thread uclamp (errno {errno})"
            ),
            Self::PermissionDenied { errno } => write!(
                f,
                "permission denied setting per-thread uclamp (errno {errno})"
            ),
            Self::Syscall { errno, message } => {
                write!(f, "sched_setattr/getattr failed (errno {errno}): {message}")
            }
        }
    }
}

impl std::error::Error for UclampError {}

/// Parse and range-check a `--uclamp-max` CLI argument (`0..=1024`).
pub fn parse_uclamp_max(raw: &str) -> Result<u32, UclampError> {
    let val: u32 = raw
        .trim()
        .parse()
        .map_err(|_| UclampError::InvalidFormat(raw.to_owned()))?;
    validate_uclamp(val)?;
    Ok(val)
}

#[inline]
fn validate_uclamp(util_max: u32) -> Result<(), UclampError> {
    if util_max > UCLAMP_MAX_LIMIT {
        return Err(UclampError::OutOfRange {
            value: util_max,
            max: UCLAMP_MAX_LIMIT,
        });
    }
    Ok(())
}

#[cfg(target_os = "linux")]
mod imp {
    use super::{UclampError, UCLAMP_MAX_LIMIT};

    const SCHED_FLAG_KEEP_POLICY: u64 = 0x08;
    const SCHED_FLAG_KEEP_PARAMS: u64 = 0x10;
    const SCHED_FLAG_UTIL_CLAMP_MAX: u64 = 0x40;

    /// Exact ABI layout of Linux `struct sched_attr` (`linux/sched/types.h`, 56 bytes).
    #[repr(C)]
    #[derive(Debug, Clone, Copy, Default)]
    struct SchedAttr {
        size: u32,
        sched_policy: u32,
        sched_flags: u64,
        sched_nice: i32,
        sched_priority: u32,
        sched_runtime: u64,
        sched_deadline: u64,
        sched_period: u64,
        sched_util_min: u32,
        sched_util_max: u32,
    }

    const _: () = assert!(std::mem::size_of::<SchedAttr>() == 56);

    fn map_os_error(err: std::io::Error) -> UclampError {
        let errno = err.raw_os_error().unwrap_or(0);
        match errno {
            libc::EOPNOTSUPP | libc::ENOSYS => UclampError::Unsupported { errno },
            libc::EPERM | libc::EACCES => UclampError::PermissionDenied { errno },
            _ => UclampError::Syscall {
                errno,
                message: err.to_string(),
            },
        }
    }

    pub(super) fn set_uclamp_max(util_max: u32) -> Result<(), UclampError> {
        let attr = SchedAttr {
            size: std::mem::size_of::<SchedAttr>() as u32,
            sched_flags: SCHED_FLAG_KEEP_POLICY
                | SCHED_FLAG_KEEP_PARAMS
                | SCHED_FLAG_UTIL_CLAMP_MAX,
            sched_util_max: util_max.min(UCLAMP_MAX_LIMIT),
            ..SchedAttr::default()
        };
        // SAFETY: `attr` is a valid, properly sized 56-byte `sched_attr` struct passed by pointer
        // to the Linux `sched_setattr(pid=0, &attr, flags=0)` syscall for the current thread only.
        let rc = unsafe {
            libc::syscall(
                libc::SYS_sched_setattr,
                0 as libc::pid_t,
                &attr as *const SchedAttr,
                0 as libc::c_uint,
            )
        };
        if rc == 0 {
            Ok(())
        } else {
            Err(map_os_error(std::io::Error::last_os_error()))
        }
    }

    pub(super) fn get_uclamp_max() -> Result<Option<u32>, UclampError> {
        let mut attr = SchedAttr::default();
        let sz = std::mem::size_of::<SchedAttr>() as libc::c_uint;
        // SAFETY: `attr` is a valid writable 56-byte `sched_attr` buffer of length `sz` passed to
        // Linux `sched_getattr(pid=0, &mut attr, sz, flags=0)` for the current thread only.
        let rc = unsafe {
            libc::syscall(
                libc::SYS_sched_getattr,
                0 as libc::pid_t,
                &mut attr as *mut SchedAttr,
                sz,
                0 as libc::c_uint,
            )
        };
        if rc == 0 {
            Ok(Some(attr.sched_util_max))
        } else {
            Err(map_os_error(std::io::Error::last_os_error()))
        }
    }
}

#[cfg(not(target_os = "linux"))]
mod imp {
    use super::UclampError;

    pub(super) fn set_uclamp_max(_util_max: u32) -> Result<(), UclampError> {
        Ok(())
    }

    pub(super) fn get_uclamp_max() -> Result<Option<u32>, UclampError> {
        Ok(None)
    }
}

/// Apply an opt-in per-thread utilisation ceiling (`0..=1024`) to the calling thread.
///
/// On Linux this invokes `sched_setattr` with `SCHED_FLAG_UTIL_CLAMP_MAX` (keeping the existing
/// policy and priority). On non-Linux targets it validates `util_max <= 1024` and is a safe no-op.
pub fn set_thread_uclamp_max(util_max: u32) -> Result<(), UclampError> {
    validate_uclamp(util_max)?;
    imp::set_uclamp_max(util_max)
}

/// Read the calling thread's current `sched_util_max` ceiling (`Ok(None)` on non-Linux).
pub fn get_thread_uclamp_max() -> Result<Option<u32>, UclampError> {
    imp::get_uclamp_max()
}
