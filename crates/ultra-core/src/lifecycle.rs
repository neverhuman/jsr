//! Stream lifecycle (§3.5): ACCEPTING -> DRAINING -> FINISHED, with CANCELLED and FAILED terminal.

use serde::{Deserialize, Serialize};

use crate::CoreError;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StreamState {
    Accepting,
    Draining,
    Finished,
    Cancelled,
    Failed,
}

impl StreamState {
    pub fn is_terminal(self) -> bool {
        matches!(self, Self::Finished | Self::Cancelled | Self::Failed)
    }
}

/// The lifecycle state machine. `finish_input` and `cancel` are idempotent.
#[derive(Debug, Clone)]
pub struct Lifecycle {
    state: StreamState,
    flushes_started: u32,
}

impl Default for Lifecycle {
    fn default() -> Self {
        Self {
            state: StreamState::Accepting,
            flushes_started: 0,
        }
    }
}

impl Lifecycle {
    pub fn state(&self) -> StreamState {
        self.state
    }

    /// Number of flushes ever started (must never exceed 1).
    pub fn flushes_started(&self) -> u32 {
        self.flushes_started
    }

    /// May audio be pushed now?
    pub fn check_push(&self) -> Result<(), CoreError> {
        match self.state {
            StreamState::Accepting => Ok(()),
            _ => Err(CoreError::Lifecycle(
                "push after finish_input or terminal state",
            )),
        }
    }

    /// Idempotent; starts exactly one flush. Returns true if this call started it.
    pub fn finish_input(&mut self) -> Result<bool, CoreError> {
        match self.state {
            StreamState::Accepting => {
                self.state = StreamState::Draining;
                self.flushes_started += 1;
                Ok(true)
            }
            StreamState::Draining | StreamState::Finished => Ok(false),
            StreamState::Cancelled | StreamState::Failed => Err(CoreError::Lifecycle(
                "finish_input on a cancelled or failed stream",
            )),
        }
    }

    /// The flush completed.
    pub fn complete(&mut self) -> Result<(), CoreError> {
        match self.state {
            StreamState::Draining => {
                self.state = StreamState::Finished;
                Ok(())
            }
            StreamState::Finished => Ok(()),
            _ => Err(CoreError::Lifecycle("complete without a flush")),
        }
    }

    /// Idempotent cancel; a finished stream stays finished.
    pub fn cancel(&mut self) {
        if !self.state.is_terminal() {
            self.state = StreamState::Cancelled;
        }
    }

    /// A failure is never reported as completion.
    pub fn fail(&mut self) {
        if !matches!(self.state, StreamState::Finished | StreamState::Cancelled) {
            self.state = StreamState::Failed;
        }
    }
}
