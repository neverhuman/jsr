//! ULTRA core: the integer numeric contract (§5), the output event contract (§3),
//! the stream lifecycle and receipts (Appendix D).

pub mod events;
pub mod hash;
pub mod lifecycle;
pub mod numeric;
pub mod receipt;
pub mod repair;
pub mod ternary;

/// Errors from the core contracts. Every refusal names the rule it enforces.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum CoreError {
    #[error("requantisation shift {0} outside 1..=31")]
    BadShift(u32),
    #[error("shape mismatch in {0}")]
    Shape(&'static str),
    #[error("row overflow proof failed: fan_in {fan_in} x 127 x {max_abs_w} >= 2^31")]
    Overflow { fan_in: usize, max_abs_w: u32 },
    #[error("weight out of range (rule N1)")]
    WeightRange,
    #[error("activation above 127 (rule N1)")]
    ActivationRange,
    #[error("ternary code 3 refused at load")]
    BadTernaryCode,
    #[error("output contract violation: {0}")]
    Contract(&'static str),
    #[error("lifecycle violation: {0}")]
    Lifecycle(&'static str),
    #[error("receipt refused as evidence: {0}")]
    Receipt(&'static str),
}
