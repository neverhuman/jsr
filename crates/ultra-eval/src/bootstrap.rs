//! Paired cluster bootstrap (v1.0-B §30.4, §30.6; v1 §11.6) and multiplicity helpers.
//!
//! Each [`PairedUnit`] carries a numerator for system A (control) and system B
//! (treatment) over a shared denominator: edit counts over reference words for WER,
//! or deadline successes over reference words for CQ. Units are aggregated into
//! clusters (speaker/session/recording) within dataset strata. Every replicate
//! resamples, within each stratum, the original number of clusters with replacement,
//! the SAME clusters for both systems, then recomputes each dataset ratio and either
//! the frozen-weight macro average or the pooled ratio. The statistic is
//! Delta = B - A (treatment minus control). Bounds are percentile bounds
//! (nearest-rank). Linked clusters across datasets are not modelled here: cluster ids
//! are scoped to their dataset.
//!
//! The RNG is a local SplitMix64 so results are bit-reproducible from the seed.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::corpus::MacroWeights;
use crate::delay::quantile_nearest_rank;
use crate::EvalError;

/// Floor on replicates (v1 §11.6). v1.0-B §30.4 asks for
/// [`recommended_replicates`] for a formal one-sided bound.
pub const MIN_REPLICATES: usize = 5_000;

/// `max(10_000, ceil(200 / alpha_eff))` replicates (v1.0-B §30.4).
pub fn recommended_replicates(alpha_eff: f64) -> usize {
    let tail = (200.0 / alpha_eff).ceil();
    if tail.is_finite() && tail > 10_000.0 {
        tail as usize
    } else {
        10_000
    }
}

/// SplitMix64 (Steele, Lea, Flood 2014; constants from Vigna's reference).
#[derive(Debug, Clone)]
pub struct SplitMix64(u64);

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        SplitMix64(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Unbiased integer in `0..n` (Lemire's multiply-shift with rejection). `n > 0`.
    pub fn next_below(&mut self, n: u64) -> u64 {
        assert!(n > 0, "next_below(0)");
        let threshold = n.wrapping_neg() % n;
        loop {
            let m = u128::from(self.next_u64()) * u128::from(n);
            if (m as u64) >= threshold {
                return (m >> 64) as u64;
            }
        }
    }
}

/// One paired observation (usually one utterance).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PairedUnit {
    pub dataset: String,
    pub cluster: String,
    pub num_a: u64,
    pub num_b: u64,
    pub den: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BootstrapConfig {
    pub replicates: usize,
    pub seed: u64,
    /// One-sided level of each reported bound (already multiplicity-adjusted).
    pub alpha: f64,
}

/// Support of one stratum.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StratumSupport {
    pub dataset: String,
    pub clusters: u64,
    pub den: u64,
    /// Fewer than two clusters: no empirical within-stratum variance.
    pub sparse: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BootstrapResult {
    pub observed_a: f64,
    pub observed_b: f64,
    /// B - A.
    pub delta: f64,
    /// (B - A) / A; `None` when A is 0.
    pub relative_delta: Option<f64>,
    /// One-sided lower bound at `alpha` (alpha quantile of replicate deltas).
    pub lower: f64,
    /// One-sided upper bound at `alpha` (1 - alpha quantile).
    pub upper: f64,
    pub replicates: usize,
    pub seed: u64,
    pub alpha: f64,
    pub strata: Vec<StratumSupport>,
    /// True when any stratum is sparse: label the interval limited/INCONCLUSIVE.
    pub limited: bool,
    #[serde(skip)]
    sorted_deltas: Vec<f64>,
}

impl BootstrapResult {
    /// Fraction of replicate deltas >= `margin`, with the +1 correction
    /// `(1 + #{d* >= margin}) / (B + 1)`: a one-sided bootstrap tail fraction for the
    /// claim "Delta < margin".
    pub fn tail_at_least(&self, margin: f64) -> f64 {
        let idx = self.sorted_deltas.partition_point(|d| *d < margin);
        let k = self.sorted_deltas.len() - idx;
        (1 + k) as f64 / (self.sorted_deltas.len() + 1) as f64
    }

    /// Replicate deltas, ascending.
    pub fn sorted_deltas(&self) -> &[f64] {
        &self.sorted_deltas
    }
}

type Cluster = (u64, u64, u64); // (num_a, num_b, den)

fn statistic(
    sums: &BTreeMap<&str, Cluster>,
    weights: Option<&MacroWeights>,
) -> Result<(f64, f64), EvalError> {
    match weights {
        None => {
            let (a, b, d) = sums
                .values()
                .fold((0, 0, 0), |acc, c| (acc.0 + c.0, acc.1 + c.1, acc.2 + c.2));
            Ok((a as f64 / d as f64, b as f64 / d as f64))
        }
        Some(w) => {
            let ra = sums
                .iter()
                .map(|(k, c)| ((*k).to_owned(), c.0 as f64 / c.2 as f64))
                .collect();
            let rb = sums
                .iter()
                .map(|(k, c)| ((*k).to_owned(), c.1 as f64 / c.2 as f64))
                .collect();
            Ok((w.apply(&ra)?, w.apply(&rb)?))
        }
    }
}

/// Paired cluster bootstrap of Delta = B - A. With `weights` the statistic is the
/// frozen-weight macro average of per-dataset ratios; without, the pooled ratio.
pub fn paired_cluster_bootstrap(
    units: &[PairedUnit],
    weights: Option<&MacroWeights>,
    cfg: &BootstrapConfig,
) -> Result<BootstrapResult, EvalError> {
    if cfg.replicates < MIN_REPLICATES {
        return Err(EvalError::Bootstrap(format!(
            "{} replicates < minimum {MIN_REPLICATES}",
            cfg.replicates
        )));
    }
    if !(cfg.alpha > 0.0 && cfg.alpha < 0.5) {
        return Err(EvalError::Bootstrap(format!(
            "alpha {} outside (0, 0.5)",
            cfg.alpha
        )));
    }
    let mut strata: BTreeMap<&str, BTreeMap<&str, Cluster>> = BTreeMap::new();
    for u in units {
        let c = strata
            .entry(u.dataset.as_str())
            .or_default()
            .entry(u.cluster.as_str())
            .or_default();
        c.0 += u.num_a;
        c.1 += u.num_b;
        c.2 += u.den;
    }
    if strata.is_empty() {
        return Err(EvalError::Bootstrap("no units".into()));
    }
    let mut lists: Vec<(&str, Vec<Cluster>)> = Vec::new();
    let mut support = Vec::new();
    let mut observed: BTreeMap<&str, Cluster> = BTreeMap::new();
    for (ds, clusters) in &strata {
        if let Some((c, _)) = clusters.iter().find(|(_, c)| c.2 == 0) {
            return Err(EvalError::Bootstrap(format!(
                "cluster {ds}/{c} has a zero denominator; score silence separately"
            )));
        }
        let v: Vec<Cluster> = clusters.values().copied().collect();
        let tot = v
            .iter()
            .fold((0, 0, 0), |a, c| (a.0 + c.0, a.1 + c.1, a.2 + c.2));
        support.push(StratumSupport {
            dataset: (*ds).to_owned(),
            clusters: v.len() as u64,
            den: tot.2,
            sparse: v.len() < 2,
        });
        observed.insert(ds, tot);
        lists.push((ds, v));
    }
    let (oa, ob) = statistic(&observed, weights)?;
    let mut rng = SplitMix64::new(cfg.seed);
    let mut deltas = Vec::with_capacity(cfg.replicates);
    let mut sums: BTreeMap<&str, Cluster> = BTreeMap::new();
    for _ in 0..cfg.replicates {
        sums.clear();
        for (ds, v) in &lists {
            let n = v.len() as u64;
            let mut acc = (0, 0, 0);
            for _ in 0..n {
                let c = v[rng.next_below(n) as usize];
                acc = (acc.0 + c.0, acc.1 + c.1, acc.2 + c.2);
            }
            sums.insert(ds, acc);
        }
        let (a, b) = statistic(&sums, weights)?;
        deltas.push(b - a);
    }
    deltas.sort_by(f64::total_cmp);
    let q = |p: f64| quantile_nearest_rank(&deltas, p).unwrap_or(f64::NAN);
    let limited = support.iter().any(|s| s.sparse);
    Ok(BootstrapResult {
        observed_a: oa,
        observed_b: ob,
        delta: ob - oa,
        relative_delta: (oa != 0.0).then(|| (ob - oa) / oa),
        lower: q(cfg.alpha),
        upper: q(1.0 - cfg.alpha),
        replicates: cfg.replicates,
        seed: cfg.seed,
        alpha: cfg.alpha,
        strata: support,
        limited,
        sorted_deltas: deltas,
    })
}

/// Bonferroni per-comparison level `alpha / m` (v1.0-B §30.6 default).
pub fn bonferroni_alpha(alpha: f64, m: usize) -> f64 {
    alpha / m.max(1) as f64
}

/// Holm step-down adjusted p-values, in input order (monotone, capped at 1).
pub fn holm_adjust(p: &[f64]) -> Vec<f64> {
    let m = p.len();
    let mut idx: Vec<usize> = (0..m).collect();
    idx.sort_by(|&a, &b| p[a].total_cmp(&p[b]).then(a.cmp(&b)));
    let mut out = vec![0.0; m];
    let mut running: f64 = 0.0;
    for (rank, &i) in idx.iter().enumerate() {
        let adj = ((m - rank) as f64 * p[i]).min(1.0);
        running = running.max(adj);
        out[i] = running;
    }
    out
}

/// Holm rejections at family level `alpha`, in input order.
pub fn holm_reject(p: &[f64], alpha: f64) -> Vec<bool> {
    holm_adjust(p).into_iter().map(|q| q <= alpha).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(seed: u64) -> BootstrapConfig {
        BootstrapConfig {
            replicates: MIN_REPLICATES,
            seed,
            alpha: 0.05,
        }
    }

    fn unit(ds: &str, cl: usize, a: u64, b: u64, den: u64) -> PairedUnit {
        PairedUnit {
            dataset: ds.into(),
            cluster: format!("spk{cl}"),
            num_a: a,
            num_b: b,
            den,
        }
    }

    #[test]
    fn splitmix_reference_vector() {
        let mut r = SplitMix64::new(0);
        assert_eq!(r.next_u64(), 0xE220_A839_7B1D_CDAF);
        let mut r = SplitMix64::new(1_234_567);
        assert_eq!(r.next_u64(), 6_457_827_717_110_365_317);
        assert_eq!(r.next_u64(), 3_203_168_211_198_807_973);
    }

    #[test]
    fn next_below_is_in_range_and_roughly_uniform() {
        let mut r = SplitMix64::new(7);
        let mut hist = [0u32; 7];
        for _ in 0..70_000 {
            hist[r.next_below(7) as usize] += 1;
        }
        for h in hist {
            assert!((9_000..11_000).contains(&h), "{hist:?}");
        }
    }

    #[test]
    fn identical_systems_give_zero_interval() {
        let u: Vec<_> = (0..20)
            .map(|i| unit("d", i, i as u64 % 5, i as u64 % 5, 50))
            .collect();
        let r = paired_cluster_bootstrap(&u, None, &cfg(1)).unwrap();
        assert_eq!((r.delta, r.lower, r.upper), (0.0, 0.0, 0.0));
        assert!(!r.limited);
    }

    #[test]
    fn known_improvement_is_detected_within_strata_and_macro() {
        // Treatment removes 2 of ~8 errors per speaker in both datasets.
        let mut u = Vec::new();
        let mut rng = SplitMix64::new(99);
        for ds in ["a", "b"] {
            for i in 0..30 {
                let base = 5 + rng.next_below(7);
                u.push(unit(ds, i, base, base - 2, 100));
            }
        }
        let w = MacroWeights::new([("a".into(), 0.5), ("b".into(), 0.5)].into()).unwrap();
        let r = paired_cluster_bootstrap(&u, Some(&w), &cfg(5)).unwrap();
        assert!((r.delta + 0.02).abs() < 1e-12, "delta {}", r.delta);
        assert!(r.upper < 0.0, "upper {}", r.upper);
        assert!(r.lower <= r.delta && r.delta <= r.upper);
        assert!(r.tail_at_least(0.0) < 0.001);
        assert_eq!(r.strata.len(), 2);
        assert_eq!(r.strata[0].clusters, 30);
        let pooled = paired_cluster_bootstrap(&u, None, &cfg(5)).unwrap();
        assert!(pooled.upper < 0.0);
    }

    #[test]
    fn null_effect_interval_straddles_zero() {
        // Errors are swapped between speaker pairs: same total, real per-speaker noise.
        let mut u = Vec::new();
        for i in 0..40 {
            let (a, b) = if i % 2 == 0 { (3, 9) } else { (9, 3) };
            u.push(unit("d", i, a, b, 100));
        }
        let r = paired_cluster_bootstrap(&u, None, &cfg(11)).unwrap();
        assert_eq!(r.delta, 0.0);
        assert!(r.lower < 0.0 && r.upper > 0.0, "{} {}", r.lower, r.upper);
        assert!(r.tail_at_least(0.0) > 0.3);
    }

    #[test]
    fn deterministic_and_seed_sensitive() {
        let u: Vec<_> = (0..25)
            .map(|i| unit("d", i, (i % 7) as u64, (i % 4) as u64, 40))
            .collect();
        let a = paired_cluster_bootstrap(&u, None, &cfg(3)).unwrap();
        let b = paired_cluster_bootstrap(&u, None, &cfg(3)).unwrap();
        let c = paired_cluster_bootstrap(&u, None, &cfg(4)).unwrap();
        assert_eq!(a, b);
        assert_ne!(a.sorted_deltas(), c.sorted_deltas());
    }

    #[test]
    fn utterances_of_one_speaker_move_together() {
        // One speaker split over many utterances is still one cluster.
        let mut u: Vec<_> = (0..10)
            .map(|j| PairedUnit {
                dataset: "d".into(),
                cluster: "only".into(),
                num_a: j,
                num_b: 0,
                den: 10,
            })
            .collect();
        let r = paired_cluster_bootstrap(&u, None, &cfg(1)).unwrap();
        assert!(r.limited && r.strata[0].sparse);
        assert_eq!(r.lower, r.upper); // a single cluster has no variance
        u.push(unit("d", 1, 0, 0, 0));
        assert!(paired_cluster_bootstrap(&u, None, &cfg(1)).is_err());
    }

    #[test]
    fn refuses_too_few_replicates_and_bad_alpha() {
        let u = vec![unit("d", 0, 1, 1, 1)];
        let mut c = cfg(1);
        c.replicates = 100;
        assert!(paired_cluster_bootstrap(&u, None, &c).is_err());
        let mut c = cfg(1);
        c.alpha = 0.7;
        assert!(paired_cluster_bootstrap(&u, None, &c).is_err());
        assert!(paired_cluster_bootstrap(&[], None, &cfg(1)).is_err());
    }

    #[test]
    fn replicate_recommendation() {
        assert_eq!(recommended_replicates(0.05), 10_000);
        assert_eq!(recommended_replicates(0.0125), 16_000);
        assert_eq!(bonferroni_alpha(0.05, 4), 0.0125);
    }

    #[test]
    fn holm_known_answer() {
        let p = [0.01, 0.04, 0.03, 0.005];
        let adj = holm_adjust(&p);
        let want = [0.03, 0.06, 0.06, 0.02];
        for (a, w) in adj.iter().zip(want) {
            assert!((a - w).abs() < 1e-12, "{adj:?}");
        }
        assert_eq!(holm_reject(&p, 0.05), vec![true, false, false, true]);
        // Step-down monotonicity: 0.8 * 2 caps at 1, so 0.9 is lifted to 1 too.
        assert_eq!(holm_adjust(&[0.9, 0.8]), vec![1.0, 1.0]);
    }
}
