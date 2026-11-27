//! Confidence-gated wide commit search (research H294, isosyll v4.2.0 final pass).
//!
//! The commit pass first runs its normal beam. When the fused-beam top-1 minus top-2 score is
//! below a frozen margin, the same encoder projection is searched again at a wider beam and the
//! wider list goes to hypothesis selection. A one-candidate beam never widens. Default off.

use crate::{Engine, EngineError, EngineStream};

/// Frozen wide-search gate for the commit pass.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CommitWideGate {
    /// Beam width of the second search (`2..=64`).
    pub width: usize,
    /// Widen only when the first beam's top-1 minus top-2 fused score is strictly below this.
    pub margin: f32,
}

impl CommitWideGate {
    pub fn new(width: usize, margin: f32) -> Result<Self, EngineError> {
        if !(2..=64).contains(&width) {
            return Err(EngineError::Shape(format!(
                "wide gate width {width} not in 2..=64"
            )));
        }
        if !margin.is_finite() || margin < 0.0 {
            return Err(EngineError::Shape(
                "wide gate margin must be finite and >= 0".into(),
            ));
        }
        Ok(Self { width, margin })
    }

    /// Whether a first-pass top-1 minus top-2 gap routes to the wide search.
    pub fn routes(&self, gap: Option<f32>) -> bool {
        gap.is_some_and(|g| g < self.margin)
    }
}

/// Top-1 minus top-2 fused beam score, or `None` for a beam with fewer than two hypotheses.
pub(crate) fn beam_margin(stream: &EngineStream) -> Option<f32> {
    let beam = stream.dec.beam.as_ref()?;
    top_two_gap(beam.hyps.iter().map(|h| h.score))
}

pub(crate) fn top_two_gap(scores: impl IntoIterator<Item = f32>) -> Option<f32> {
    let (mut best, mut second) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
    for s in scores {
        if s > best {
            second = best;
            best = s;
        } else if s > second {
            second = s;
        }
    }
    second.is_finite().then_some(best - second)
}

/// Engine for the second search: the commit engine with only the beam width changed.
pub(crate) fn wide_engine(eng: &Engine, gate: CommitWideGate) -> Engine {
    eng.clone().with_beam(gate.width)
}

#[cfg(test)]
mod tests {
    use super::{top_two_gap, CommitWideGate};

    #[test]
    fn gap_needs_two_finite_scores() {
        assert_eq!(top_two_gap([]), None);
        assert_eq!(top_two_gap([-1.0]), None);
        assert_eq!(top_two_gap([-3.0, -1.0, -2.5]), Some(1.5));
        assert_eq!(top_two_gap([-1.0, -1.0]), Some(0.0));
    }

    #[test]
    fn gate_is_strict_and_validated() {
        let g = CommitWideGate::new(48, 0.5).expect("gate");
        assert!(g.routes(Some(0.49)));
        assert!(!g.routes(Some(0.5)));
        assert!(!g.routes(None));
        assert!(CommitWideGate::new(1, 0.5).is_err());
        assert!(CommitWideGate::new(65, 0.5).is_err());
        assert!(CommitWideGate::new(48, f32::NAN).is_err());
        assert!(CommitWideGate::new(48, -0.1).is_err());
    }
}
