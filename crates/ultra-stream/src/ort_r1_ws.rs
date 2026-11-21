//! Reusable contiguous scratch workspace (`BatchWorkspace`) and finite-output check for
//! [`crate::ort_r1::OrtTransducerBackend`].

use ultra_gp::GpError;

use crate::api::StreamError;

#[derive(Debug, Clone)]
pub(crate) enum StateBuf {
    F32(Vec<f32>),
    I64(Vec<i64>),
}

impl StateBuf {
    pub(crate) fn retained_bytes(&self) -> usize {
        match self {
            Self::F32(v) => v.capacity() * 4,
            Self::I64(v) => v.capacity() * 8,
        }
    }
}

/// Reusable contiguous scratch buffers eliminating per-chunk heap allocations across streams.
#[derive(Default)]
pub struct BatchWorkspace {
    pub(crate) feats_in: Vec<f32>,
    pub(crate) t0_in: Vec<i64>,
    pub(crate) state_ins: Vec<StateBuf>,
    pub(crate) state_outs: Vec<StateBuf>,
    pub(crate) state_shapes: Vec<Vec<i64>>,
    pub(crate) enc_out: Vec<f32>,
    pub(crate) dec_ctx: Vec<i64>,
    pub(crate) dec_out: Vec<f32>,
    pub(crate) joi_enc: Vec<f32>,
    pub(crate) joi_logits: Vec<f32>,
}

impl BatchWorkspace {
    pub fn new() -> Self {
        Self::default()
    }

    /// Total heap bytes retained across workspace scratch buffers and shape vectors.
    pub fn retained_bytes(&self) -> usize {
        let states_b: usize = self
            .state_ins
            .iter()
            .chain(&self.state_outs)
            .map(StateBuf::retained_bytes)
            .sum();
        let shapes_b: usize = self.state_shapes.capacity() * std::mem::size_of::<Vec<i64>>()
            + self
                .state_shapes
                .iter()
                .map(|v| v.capacity() * 8)
                .sum::<usize>()
            + (self.state_ins.capacity() + self.state_outs.capacity())
                * std::mem::size_of::<StateBuf>();
        states_b
            + shapes_b
            + (self.feats_in.capacity()
                + self.enc_out.capacity()
                + self.dec_out.capacity()
                + self.joi_enc.capacity()
                + self.joi_logits.capacity())
                * 4
            + (self.t0_in.capacity() + self.dec_ctx.capacity()) * 8
    }
}

pub(crate) fn ensure_all_finite(slice: &[f32], label: &'static str) -> Result<(), StreamError> {
    if slice.iter().all(|v| v.is_finite()) {
        Ok(())
    } else {
        Err(GpError::Output(format!("non-finite {label} tensor entry")).into())
    }
}
