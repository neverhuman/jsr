//! Exact `f64` RNN-T forward recursions (full-sum + 1-symbol-per-frame synchronous).

#[inline]
pub(super) fn logaddexp(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        return b;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    let m = a.max(b);
    m + (-(a - b).abs()).exp().ln_1p()
}

pub(super) struct ForwardDp {
    full: Vec<f64>,
    prev_blank: Vec<f64>,
    sync: Vec<f64>,
    next_sync: Vec<f64>,
    started: bool,
}

impl ForwardDp {
    pub(super) fn new(n_tokens: usize) -> Self {
        let cols = n_tokens + 1;
        let mut sync = vec![f64::NEG_INFINITY; cols];
        sync[0] = 0.0;
        Self {
            full: vec![f64::NEG_INFINITY; cols],
            prev_blank: vec![0.0; cols],
            sync,
            next_sync: vec![f64::NEG_INFINITY; cols],
            started: false,
        }
    }

    pub(super) fn step(&mut self, b_row: &[f64], y_row: &[f64]) {
        let u_len = y_row.len();
        if !self.started {
            self.started = true;
            self.full[0] = 0.0;
            for u in 1..=u_len {
                self.full[u] = self.full[u - 1] + y_row[u - 1];
            }
        } else {
            self.full[0] += self.prev_blank[0];
            for u in 1..=u_len {
                self.full[u] = logaddexp(
                    self.full[u] + self.prev_blank[u],
                    self.full[u - 1] + y_row[u - 1],
                );
            }
        }
        self.next_sync[0] = self.sync[0] + b_row[0];
        for u in 1..=u_len {
            self.next_sync[u] = logaddexp(self.sync[u] + b_row[u], self.sync[u - 1] + y_row[u - 1]);
        }
        std::mem::swap(&mut self.sync, &mut self.next_sync);
        self.prev_blank.copy_from_slice(b_row);
    }

    pub(super) fn finish(&self, n_tokens: usize) -> (f64, f64) {
        if !self.started {
            return if n_tokens == 0 {
                (0.0, 0.0)
            } else {
                (f64::NEG_INFINITY, f64::NEG_INFINITY)
            };
        }
        (
            self.full[n_tokens] + self.prev_blank[n_tokens],
            self.sync[n_tokens],
        )
    }
}

/// Pure `(rnnt_full_logp, rnnt_sync_logp)` forward recursion over flat log-probability tables:
/// `logp_blank` (`[n_frames][n_tokens + 1]`) and `logp_tok` (`[n_frames][n_tokens]`).
pub fn rnnt_forward_from_log_probs(
    logp_blank: &[f64],
    logp_tok: &[f64],
    n_frames: usize,
    n_tokens: usize,
) -> (f64, f64) {
    let cols = n_tokens + 1;
    assert_eq!(logp_blank.len(), n_frames * cols);
    assert_eq!(logp_tok.len(), n_frames * n_tokens);
    let mut dp = ForwardDp::new(n_tokens);
    for t in 0..n_frames {
        dp.step(
            &logp_blank[t * cols..(t + 1) * cols],
            &logp_tok[t * n_tokens..(t + 1) * n_tokens],
        );
    }
    dp.finish(n_tokens)
}
