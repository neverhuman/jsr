//! Exact Student-t paired log-ratio statistics and Holm-Bonferroni step-down correction
//! for `FREEZE.md §4.3 / §5` ABAB timing blocks (`N10`).

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbabPairedComparison {
    pub baseline: String,
    pub candidate: String,
    pub blocks_paired: usize,
    pub log_ratios: Vec<f64>,
    pub mean_log_ratio: f64,
    pub sd_log_ratio: f64,
    pub t_stat: f64,
    pub df: usize,
    /// Exact one-sided Student-t p-value for `H1: mean(log(cand / base)) < 0` (`cand < base`).
    pub p_value_one_sided: f64,
    /// Holm-Bonferroni adjusted one-sided p-value across all comparisons in this report.
    pub holm_p_value: f64,
    pub holm_reject_alpha_0_05: bool,
    pub median_ratio_cand_over_base: f64,
    pub ratio_ci95_bootstrap: [f64; 2],
    pub geometric_mean_ratio: f64,
    pub geometric_mean_ratio_ci95_t: [f64; 2],
    pub median_diff_rtf_cpu: f64,
    pub min_diff_rtf_cpu: f64,
    pub max_diff_rtf_cpu: f64,
    pub bootstrap_replicates: usize,
    pub seed: u64,
}

fn median_sorted(sorted: &[f64]) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return 0.0;
    }
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        0.5 * (sorted[n / 2 - 1] + sorted[n / 2])
    }
}

pub fn summarize_values(vals: &[f64]) -> (f64, f64, f64) {
    if vals.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let mut s = vals.to_vec();
    s.sort_by(f64::total_cmp);
    (median_sorted(&s), s[0], s[s.len() - 1])
}

fn ln_gamma(z: f64) -> f64 {
    const COEFFS: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if z < 0.5 {
        return std::f64::consts::PI.ln()
            - (std::f64::consts::PI * z).sin().ln()
            - ln_gamma(1.0 - z);
    }
    let z = z - 1.0;
    let mut x = COEFFS[0];
    for (i, &c) in COEFFS.iter().enumerate().skip(1) {
        x += c / (z + i as f64);
    }
    let t = z + 7.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (z + 0.5) * t.ln() - t + x.ln()
}

fn betacf(a: f64, b: f64, x: f64) -> f64 {
    const FPMIN: f64 = 1e-300;
    const EPS: f64 = 3e-14;
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < FPMIN {
        d = FPMIN;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..=200 {
        let mf = m as f64;
        let m2 = 2.0 * mf;
        let aa = mf * (b - mf) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        h *= d * c;
        let aa = -(a + mf) * (qab + mf) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < EPS {
            break;
        }
    }
    h
}

fn regularized_incomplete_beta(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let bt = (ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * x.ln() + b * (1.0 - x).ln()).exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        bt * betacf(a, b, x) / a
    } else {
        1.0 - bt * betacf(b, a, 1.0 - x) / b
    }
}

/// Exact Student-t CDF `P(T <= t)` for `df >= 1`.
pub fn student_t_cdf(t: f64, df: usize) -> f64 {
    if df == 0 || t.is_nan() {
        return 1.0;
    }
    if t.is_infinite() {
        return if t.is_sign_negative() { 0.0 } else { 1.0 };
    }
    if t == 0.0 {
        return 0.5;
    }
    let v = df as f64;
    let x = v / (v + t * t);
    let ib = regularized_incomplete_beta(0.5 * v, 0.5, x);
    if t < 0.0 {
        (0.5 * ib).clamp(0.0, 1.0)
    } else {
        (1.0 - 0.5 * ib).clamp(0.0, 1.0)
    }
}

/// Exact two-sided 95% Student-t critical value `t_{0.975, df}` via bisection.
pub fn student_t_critical_975(df: usize) -> f64 {
    if df == 0 {
        return 0.0;
    }
    let (mut lo, mut hi) = (0.0f64, 20.0f64);
    while student_t_cdf(hi, df) < 0.975 {
        hi *= 2.0;
    }
    for _ in 0..80 {
        let mid = 0.5 * (lo + hi);
        if student_t_cdf(mid, df) < 0.975 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Apply Holm-Bonferroni step-down adjustment in place over all comparisons (`FREEZE.md §5`).
pub fn apply_holm_correction(comparisons: &mut [AbabPairedComparison], alpha: f64) {
    let m = comparisons.len();
    if m == 0 {
        return;
    }
    let mut order: Vec<usize> = (0..m).collect();
    order.sort_by(|&a, &b| {
        comparisons[a]
            .p_value_one_sided
            .total_cmp(&comparisons[b].p_value_one_sided)
    });
    let mut running_max = 0.0f64;
    for (rank, &idx) in order.iter().enumerate() {
        let adj = (comparisons[idx].p_value_one_sided * ((m - rank) as f64)).min(1.0);
        running_max = running_max.max(adj);
        comparisons[idx].holm_p_value = running_max;
        comparisons[idx].holm_reject_alpha_0_05 = running_max <= alpha;
    }
}

/// Paired ABAB comparison (`N10` / `FREEZE.md §5`): one-sided paired t-test on `log(cand / base)`
/// (`H1: cand < base`) plus median ratio with 95% bootstrap CI and geometric-mean ratio t-CI.
pub fn paired_block_comparison(
    baseline_name: &str,
    cand_name: &str,
    base_rtfs: &[f64],
    cand_rtfs: &[f64],
    replicates: usize,
    seed: u64,
) -> AbabPairedComparison {
    let n = base_rtfs.len().min(cand_rtfs.len());
    let diffs: Vec<f64> = (0..n).map(|i| cand_rtfs[i] - base_rtfs[i]).collect();
    let ratios: Vec<f64> = (0..n)
        .map(|i| {
            if base_rtfs[i] > 0.0 {
                cand_rtfs[i] / base_rtfs[i]
            } else {
                1.0
            }
        })
        .collect();
    let log_ratios: Vec<f64> = ratios.iter().map(|&r| r.max(1e-12).ln()).collect();
    let (med_d, min_d, max_d) = summarize_values(&diffs);
    let (med_ratio, _, _) = summarize_values(&ratios);

    let df = n.saturating_sub(1);
    let mean_log = if n > 0 {
        log_ratios.iter().sum::<f64>() / (n as f64)
    } else {
        0.0
    };
    let var_log = if n > 1 {
        log_ratios
            .iter()
            .map(|&x| (x - mean_log) * (x - mean_log))
            .sum::<f64>()
            / (df as f64)
    } else {
        0.0
    };
    let sd_log = var_log.sqrt();
    let se_log = if n > 0 {
        sd_log / (n as f64).sqrt()
    } else {
        0.0
    };
    let (t_stat, p_one_sided) = if df == 0 {
        (0.0, 1.0)
    } else if se_log > 0.0 {
        let t = mean_log / se_log;
        (t, student_t_cdf(t, df))
    } else if mean_log < 0.0 {
        (-1e9, 0.0)
    } else if mean_log > 0.0 {
        (1e9, 1.0)
    } else {
        (0.0, 0.5)
    };
    let t_crit = student_t_critical_975(df);
    let geom_mean = mean_log.exp();
    let geom_ci = [
        (mean_log - t_crit * se_log).exp(),
        (mean_log + t_crit * se_log).exp(),
    ];

    let mut state = seed;
    let mut next_idx = |len: usize| -> usize {
        state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        ((z ^ (z >> 31)) as usize) % len
    };
    let reps = replicates.max(1);
    let mut boot_ratios = Vec::with_capacity(reps);
    let mut sample = vec![0.0f64; n];
    if n > 0 {
        for _ in 0..reps {
            for dst in &mut sample {
                *dst = ratios[next_idx(n)];
            }
            sample.sort_by(f64::total_cmp);
            boot_ratios.push(median_sorted(&sample));
        }
        boot_ratios.sort_by(f64::total_cmp);
    }
    let ratio_lo = boot_ratios
        .get(((boot_ratios.len() as f64) * 0.025) as usize)
        .copied()
        .unwrap_or(med_ratio);
    let ratio_hi = if boot_ratios.is_empty() {
        med_ratio
    } else {
        boot_ratios[(((boot_ratios.len() as f64) * 0.975) as usize).min(boot_ratios.len() - 1)]
    };

    AbabPairedComparison {
        baseline: baseline_name.to_owned(),
        candidate: cand_name.to_owned(),
        blocks_paired: n,
        log_ratios,
        mean_log_ratio: mean_log,
        sd_log_ratio: sd_log,
        t_stat,
        df,
        p_value_one_sided: p_one_sided,
        holm_p_value: p_one_sided,
        holm_reject_alpha_0_05: p_one_sided <= 0.05,
        median_ratio_cand_over_base: med_ratio,
        ratio_ci95_bootstrap: [ratio_lo, ratio_hi],
        geometric_mean_ratio: geom_mean,
        geometric_mean_ratio_ci95_t: geom_ci,
        median_diff_rtf_cpu: med_d,
        min_diff_rtf_cpu: min_d,
        max_diff_rtf_cpu: max_d,
        bootstrap_replicates: reps,
        seed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn student_t_cdf_and_paired_log_ratio_test_match_exact_tables() {
        let p_c1 = student_t_cdf(-1.0, 1);
        assert!((p_c1 - 0.25).abs() < 1e-10, "df=1 t=-1: {p_c1}");
        let p_df2 = student_t_cdf(-1.0, 2);
        let expect_df2 = 0.5 * (1.0 - 1.0 / 3.0_f64.sqrt());
        assert!((p_df2 - expect_df2).abs() < 1e-10, "df=2 t=-1: {p_df2}");
        assert!((student_t_cdf(-1.894_578_6, 7) - 0.05).abs() < 1e-6);
        assert!((student_t_cdf(-2.364_624_3, 7) - 0.025).abs() < 1e-6);
        assert!((student_t_cdf(-2.997_951_6, 7) - 0.01).abs() < 1e-6);
        assert!((student_t_critical_975(7) - 2.364_624_3).abs() < 1e-5);

        let base = [0.080, 0.078, 0.082, 0.079, 0.081, 0.080, 0.079, 0.081];
        let fast = [0.041, 0.040, 0.042, 0.041, 0.040, 0.042, 0.040, 0.041];
        let slow = [0.085, 0.083, 0.084, 0.086, 0.085, 0.084, 0.083, 0.085];
        let mut cmps = vec![
            paired_block_comparison("zipformer-int8", "r1-mr", &base, &fast, 2_000, 42),
            paired_block_comparison("zipformer-int8", "r1-slow", &base, &slow, 2_000, 42),
        ];
        apply_holm_correction(&mut cmps, 0.05);
        assert_eq!(cmps[0].df, 7);
        assert!(cmps[0].t_stat < -30.0);
        assert!(cmps[0].p_value_one_sided < 1e-8);
        assert!(cmps[0].holm_reject_alpha_0_05);
        assert!(!cmps[1].holm_reject_alpha_0_05);
        assert!(cmps[0].ratio_ci95_bootstrap[1] < 0.6);
    }
}
