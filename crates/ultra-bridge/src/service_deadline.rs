//! Native calls cannot be interrupted safely. Expiry ends the isolated service process;
//! the outside measurement driver records a failed run, never a successful empty result.

use std::sync::mpsc::{self, RecvTimeoutError, Sender};
use std::thread::JoinHandle;
use std::time::Duration;

pub(crate) struct ServiceDeadline {
    cancel: Sender<()>,
    thread: Option<JoinHandle<()>>,
}

impl ServiceDeadline {
    pub(crate) fn start(timeout: Duration) -> std::io::Result<Self> {
        if std::time::Instant::now().checked_add(timeout).is_none() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "native deadline exceeds the supported monotonic clock range",
            ));
        }
        let (cancel, receiver) = mpsc::channel();
        let thread = std::thread::Builder::new()
            .name("service-deadline".into())
            .spawn(move || {
                if receiver.recv_timeout(timeout) == Err(RecvTimeoutError::Timeout) {
                    eprintln!("native inference service exceeded its wall-clock deadline");
                    std::process::exit(124);
                }
            })?;
        Ok(Self {
            cancel,
            thread: Some(thread),
        })
    }
}

impl Drop for ServiceDeadline {
    fn drop(&mut self) {
        let _ = self.cancel.send(());
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_deadline_is_scoped_and_fatal() {
        assert!(ServiceDeadline::start(Duration::MAX).is_err());
        if let Ok(mode) = std::env::var("ULTRA_TEST_NATIVE_DEADLINE_CHILD") {
            let guard = ServiceDeadline::start(Duration::from_millis(100)).expect("deadline");
            if mode == "cancel" {
                drop(guard);
                std::thread::sleep(Duration::from_millis(300));
            } else {
                std::thread::sleep(Duration::from_secs(10));
            }
            return;
        }
        for cancelled in [false, true] {
            let mut command =
                std::process::Command::new(std::env::current_exe().expect("test exe"));
            command.args([
                "--exact",
                "service_deadline::tests::native_deadline_is_scoped_and_fatal",
            ]);
            command.env(
                "ULTRA_TEST_NATIVE_DEADLINE_CHILD",
                if cancelled { "cancel" } else { "expire" },
            );
            let output = command.output().expect("deadline child");
            assert_eq!(output.status.code(), Some(if cancelled { 0 } else { 124 }));
        }
    }
}
