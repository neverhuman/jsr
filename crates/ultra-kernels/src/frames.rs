//! A batch of A7 activation vectors (rule N1: every byte <= 127).

use crate::KernelError;
use ultra_core::numeric::check_a7;

/// `count` frames of `din` A7 activations, frame-major (`data[f * din + j]`).
#[derive(Debug, Clone, Copy)]
pub struct Frames<'a> {
    data: &'a [u8],
    din: usize,
    count: usize,
}

impl<'a> Frames<'a> {
    /// Validate shape and the 7-bit rule.
    pub fn new(data: &'a [u8], din: usize) -> Result<Self, KernelError> {
        if din == 0 || data.is_empty() || !data.len().is_multiple_of(din) {
            return Err(KernelError::Shape(
                "frames: len must be a non-zero multiple of din",
            ));
        }
        check_a7(data)?;
        Ok(Self {
            data,
            din,
            count: data.len() / din,
        })
    }

    /// Activation width.
    pub fn din(&self) -> usize {
        self.din
    }

    /// Number of frames `F`.
    pub fn count(&self) -> usize {
        self.count
    }

    /// Frame `f` (panics if `f >= count`).
    pub fn frame(&self, f: usize) -> &'a [u8] {
        &self.data[f * self.din..(f + 1) * self.din]
    }
}
