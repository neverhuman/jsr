//! Least-squares fit of the cost law `y = a + c F` (y in ps per weight per pass).

/// Result of fitting one (kernel, size, threads) series.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Fit {
    /// Intercept: fixed cost, ps per weight per pass.
    pub a: f64,
    /// Slope: marginal cost, ps per multiply-add.
    pub c: f64,
    /// Coefficient of determination.
    pub r2: f64,
    /// Largest |y - fit| / y over the points.
    pub max_rel_err: f64,
}

/// Ordinary least squares over `(F, y)` points; `None` with fewer than two distinct F.
pub fn fit_line(points: &[(f64, f64)]) -> Option<Fit> {
    let n = points.len() as f64;
    let mx = points.iter().map(|p| p.0).sum::<f64>() / n;
    let my = points.iter().map(|p| p.1).sum::<f64>() / n;
    let sxx: f64 = points.iter().map(|p| (p.0 - mx).powi(2)).sum();
    if points.len() < 2 || sxx <= 0.0 {
        return None;
    }
    let sxy: f64 = points.iter().map(|p| (p.0 - mx) * (p.1 - my)).sum();
    let c = sxy / sxx;
    let a = my - c * mx;
    let ss_res: f64 = points.iter().map(|p| (p.1 - a - c * p.0).powi(2)).sum();
    let ss_tot: f64 = points.iter().map(|p| (p.1 - my).powi(2)).sum();
    let r2 = if ss_tot > 0.0 {
        1.0 - ss_res / ss_tot
    } else {
        1.0
    };
    let max_rel_err = points
        .iter()
        .map(|p| ((p.1 - a - c * p.0) / p.1).abs())
        .fold(0.0, f64::max);
    Some(Fit {
        a,
        c,
        r2,
        max_rel_err,
    })
}

/// Cache residency of a weight image on H0 (Zen 2: 512 KiB L2 per core, 16 MiB L3 per
/// CCX; one thread sees only its own CCX's L3).
pub fn residency(weight_bytes: u64) -> &'static str {
    if weight_bytes <= 512 << 10 {
        "in_l2"
    } else if weight_bytes <= 16 << 20 {
        "in_l3_ccx"
    } else {
        "out_of_cache"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_line_is_recovered() {
        let pts: Vec<(f64, f64)> = [1.0, 2.0, 4.0, 16.0]
            .iter()
            .map(|&f| (f, 19.4 + 11.0 * f))
            .collect();
        let fit = fit_line(&pts).unwrap();
        assert!((fit.a - 19.4).abs() < 1e-9 && (fit.c - 11.0).abs() < 1e-9);
        assert!((fit.r2 - 1.0).abs() < 1e-12 && fit.max_rel_err < 1e-12);
    }

    #[test]
    fn appendix_a_tern_1536m_fit_reproduces() {
        // Raw Appendix A rows: ms per pass for tern2_vnni at 1536M.
        let ms = [
            46.527, 65.645, 87.957, 104.478, 134.738, 170.887, 245.636, 313.304,
        ];
        let fs = [1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0];
        let w = 1536.0 * 1048576.0;
        let pts: Vec<(f64, f64)> = fs.iter().zip(ms).map(|(&f, m)| (f, m * 1e9 / w)).collect();
        let fit = fit_line(&pts).unwrap();
        assert!((fit.a - 19.36).abs() < 0.02, "{fit:?}");
        assert!((fit.c - 10.98).abs() < 0.02, "{fit:?}");
        assert!((fit.r2 - 0.9993).abs() < 0.0005, "{fit:?}");
    }

    #[test]
    fn degenerate_inputs() {
        assert!(fit_line(&[(1.0, 2.0)]).is_none());
        assert!(fit_line(&[(1.0, 2.0), (1.0, 3.0)]).is_none());
        assert_eq!(residency(256 << 10), "in_l2");
        assert_eq!(residency(16 << 20), "in_l3_ccx");
        assert_eq!(residency(1 << 30), "out_of_cache");
    }
}

/// One fitted series of a sweep.
#[derive(Debug, Clone, PartialEq)]
pub struct Series {
    /// Kernel name as printed by `wstream` (e.g. `t1_avx2`).
    pub kernel: String,
    /// Weight count name (e.g. `1536M`).
    pub size: String,
    /// Threads used.
    pub threads: usize,
    /// Bytes of the weight image.
    pub weight_bytes: u64,
    /// The fit.
    pub fit: Fit,
}

impl Series {
    /// Whether this is an int8 (control) kernel.
    pub fn is_int8(&self) -> bool {
        self.kernel.starts_with("int8")
    }
}

/// W01 pass rule (spec §12): R² >= 0.98 and ternary fixed term <= 0.4 x int8, out of
/// cache, single thread. Returns the verdict word and one explanation line per check.
pub fn w01_verdict(series: &[Series]) -> (&'static str, Vec<String>) {
    let mut lines = Vec::new();
    let ooc: Vec<&Series> = series
        .iter()
        .filter(|s| s.threads == 1 && residency_of_size(s, series) == "out_of_cache")
        .collect();
    let mut sizes: Vec<&str> = ooc.iter().map(|s| s.size.as_str()).collect();
    sizes.dedup();
    let (mut any_compared, mut r2_ok, mut lift) = (false, true, true);
    for size in sizes {
        let Some(ctrl) = ooc.iter().find(|s| s.size == size && s.is_int8()) else {
            lines.push(format!("{size}: no int8 control series; not compared"));
            continue;
        };
        let mut best: Option<f64> = None;
        for t in ooc.iter().filter(|s| s.size == size && !s.is_int8()) {
            any_compared = true;
            let ratio = t.fit.a / ctrl.fit.a;
            let good = t.fit.r2 >= 0.98 && ctrl.fit.r2 >= 0.98;
            r2_ok &= good;
            best = Some(best.map_or(ratio, |b: f64| b.min(ratio)));
            lines.push(format!(
                "{size}: {} a={:.2} vs {} a={:.2} -> ratio {:.3} (R2 {:.4} / {:.4})",
                t.kernel, t.fit.a, ctrl.kernel, ctrl.fit.a, ratio, t.fit.r2, ctrl.fit.r2
            ));
        }
        lift &= best.is_some_and(|b| b <= 0.4);
    }
    let verdict = match (any_compared && r2_ok, lift) {
        (false, _) => "INCONCLUSIVE",
        (true, true) => "PASS",
        (true, false) => "NO_LIFT",
    };
    (verdict, lines)
}

fn residency_of_size(s: &Series, all: &[Series]) -> &'static str {
    // A size is out of cache when its smallest image (packed ternary) is.
    let min_bytes = all
        .iter()
        .filter(|o| o.size == s.size)
        .map(|o| o.weight_bytes)
        .min()
        .unwrap_or(s.weight_bytes);
    residency(min_bytes)
}

#[cfg(test)]
mod verdict_tests {
    use super::*;

    fn s(kernel: &str, size: &str, bytes: u64, a: f64, r2: f64) -> Series {
        Series {
            kernel: kernel.into(),
            size: size.into(),
            threads: 1,
            weight_bytes: bytes,
            fit: Fit {
                a,
                c: 10.0,
                r2,
                max_rel_err: 0.0,
            },
        }
    }

    #[test]
    fn appendix_a_numbers_pass() {
        let v = [
            s("int8_vnni", "1536M", 1536 << 20, 96.38, 0.99),
            s("tern2_vnni", "1536M", 384 << 20, 19.36, 0.9993),
            s("int8_vnni", "1M", 1 << 20, 5.36, 0.9955),
        ];
        assert_eq!(w01_verdict(&v).0, "PASS");
    }

    #[test]
    fn appendix_a_r2_fails_rule_and_no_lift_detected() {
        let v = [
            s("int8_vnni", "1536M", 1536 << 20, 96.38, 0.9667),
            s("tern2_vnni", "1536M", 384 << 20, 19.36, 0.9993),
        ];
        assert_eq!(w01_verdict(&v).0, "INCONCLUSIVE");
        let v = [
            s("int8_avx2", "1536M", 1536 << 20, 50.0, 0.99),
            s("t1_avx2", "1536M", 384 << 20, 30.0, 0.99),
        ];
        assert_eq!(w01_verdict(&v).0, "NO_LIFT");
        assert_eq!(w01_verdict(&v[..1]).0, "INCONCLUSIVE");
    }
}
