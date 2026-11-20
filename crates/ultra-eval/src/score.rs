//! Corpus scoring of `{id, speaker, dataset, ref, hyp[, duration_s]}` JSONL records and
//! paired comparison of two systems (the library behind `ultra-score`).

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::bootstrap::{paired_cluster_bootstrap, BootstrapConfig, BootstrapResult, PairedUnit};
use crate::corpus::{macro_wer, pool, pooled_wer, DatasetScore, MacroWeights, UtteranceResult};
use crate::normalise::{normaliser_sha256, NORMALISER_ID};
use crate::wer::align_text;
use crate::EvalError;

/// Report schema identity.
pub const SCORE_SCHEMA: &str = "ultra.score-report/v1";

/// One input line.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScoreRecord {
    pub id: String,
    pub speaker: String,
    pub dataset: String,
    #[serde(rename = "ref")]
    pub reference: String,
    pub hyp: String,
    #[serde(default)]
    pub duration_s: Option<f64>,
}

/// Parse JSONL text; blank lines are skipped; ids must be unique.
pub fn parse_jsonl(text: &str) -> Result<Vec<ScoreRecord>, EvalError> {
    let mut out = Vec::new();
    let mut ids = BTreeSet::new();
    for (n, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let r: ScoreRecord = serde_json::from_str(line)
            .map_err(|e| EvalError::Input(format!("line {}: {e}", n + 1)))?;
        if !ids.insert(r.id.clone()) {
            return Err(EvalError::Input(format!("duplicate id {}", r.id)));
        }
        out.push(r);
    }
    Ok(out)
}

/// Normalise and align every record.
pub fn score_records(recs: &[ScoreRecord]) -> Vec<UtteranceResult> {
    recs.iter()
        .map(|r| UtteranceResult {
            id: r.id.clone(),
            dataset: r.dataset.clone(),
            speaker: r.speaker.clone(),
            counts: align_text(&r.reference, &r.hyp).counts,
            duration_s: r.duration_s,
        })
        .collect()
}

/// Scores of one system.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SystemReport {
    pub utterances: u64,
    pub datasets: BTreeMap<String, DatasetScore>,
    pub pooled_wer: Option<f64>,
    pub macro_wer: Option<f64>,
}

fn system_report(utts: &[UtteranceResult], w: &MacroWeights) -> SystemReport {
    let datasets = pool(utts);
    SystemReport {
        utterances: utts.len() as u64,
        pooled_wer: pooled_wer(&datasets),
        macro_wer: macro_wer(&datasets, w).ok(),
        datasets,
    }
}

/// Paired comparison B - A.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Comparison {
    pub macro_delta: BootstrapResult,
    pub pooled_delta: BootstrapResult,
    pub per_dataset: BTreeMap<String, BootstrapResult>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScoreReport {
    pub schema: String,
    pub normaliser_id: String,
    pub normaliser_sha256: String,
    pub weights: BTreeMap<String, f64>,
    pub weights_sha256: String,
    pub system_a: SystemReport,
    pub system_b: Option<SystemReport>,
    pub comparison: Option<Comparison>,
}

/// Datasets with at least one reference word (those a macro WER can weight).
pub fn scorable_datasets(recs: &[ScoreRecord]) -> BTreeSet<String> {
    score_records(recs)
        .into_iter()
        .filter(|u| u.counts.ref_words > 0)
        .map(|u| u.dataset)
        .collect()
}

/// Score system A, and if given, system B paired with A by id. With B, references,
/// speakers and datasets must agree record by record.
pub fn score_report(
    a: &[ScoreRecord],
    b: Option<&[ScoreRecord]>,
    weights: &MacroWeights,
    cfg: &BootstrapConfig,
) -> Result<ScoreReport, EvalError> {
    let ua = score_records(a);
    let mut report = ScoreReport {
        schema: SCORE_SCHEMA.into(),
        normaliser_id: NORMALISER_ID.into(),
        normaliser_sha256: normaliser_sha256().into(),
        weights: weights.weights().clone(),
        weights_sha256: weights.sha256(),
        system_a: system_report(&ua, weights),
        system_b: None,
        comparison: None,
    };
    let Some(b) = b else {
        return Ok(report);
    };
    let by_id: BTreeMap<&str, &ScoreRecord> = b.iter().map(|r| (r.id.as_str(), r)).collect();
    if by_id.len() != a.len() {
        return Err(EvalError::Input(format!(
            "system B has {} records, A has {}",
            by_id.len(),
            a.len()
        )));
    }
    let mut b_sorted = Vec::with_capacity(a.len());
    for ra in a {
        let rb = by_id
            .get(ra.id.as_str())
            .ok_or_else(|| EvalError::Input(format!("id {} missing from system B", ra.id)))?;
        if rb.reference != ra.reference || rb.speaker != ra.speaker || rb.dataset != ra.dataset {
            return Err(EvalError::Input(format!(
                "id {}: ref/speaker/dataset differ between systems",
                ra.id
            )));
        }
        b_sorted.push((*rb).clone());
    }
    let ub = score_records(&b_sorted);
    let units: Vec<PairedUnit> = ua
        .iter()
        .zip(&ub)
        .filter(|(x, _)| x.counts.ref_words > 0)
        .map(|(x, y)| PairedUnit {
            dataset: x.dataset.clone(),
            cluster: x.speaker.clone(),
            num_a: x.counts.errors(),
            num_b: y.counts.errors(),
            den: x.counts.ref_words,
        })
        .collect();
    let mut per_dataset = BTreeMap::new();
    for ds in weights.weights().keys() {
        let sub: Vec<PairedUnit> = units.iter().filter(|u| &u.dataset == ds).cloned().collect();
        per_dataset.insert(ds.clone(), paired_cluster_bootstrap(&sub, None, cfg)?);
    }
    report.comparison = Some(Comparison {
        macro_delta: paired_cluster_bootstrap(&units, Some(weights), cfg)?,
        pooled_delta: paired_cluster_bootstrap(&units, None, cfg)?,
        per_dataset,
    });
    report.system_b = Some(system_report(&ub, weights));
    Ok(report)
}

fn pct(x: Option<f64>) -> String {
    x.map_or_else(|| "n/a".into(), |v| format!("{:.2}", 100.0 * v))
}

/// A short markdown summary of a report.
pub fn markdown(r: &ScoreReport) -> String {
    let mut s = format!(
        "normaliser `{}` (`{}`), weights `{}`\n\n",
        r.normaliser_id,
        &r.normaliser_sha256[..12],
        &r.weights_sha256[..12]
    );
    s.push_str("| system | dataset | WER % | S | D | I | ref words | utts | silence ins/h |\n");
    s.push_str("|---|---|---:|---:|---:|---:|---:|---:|---:|\n");
    let mut sys = vec![("A", &r.system_a)];
    if let Some(b) = &r.system_b {
        sys.push(("B", b));
    }
    for (name, rep) in &sys {
        for (ds, d) in &rep.datasets {
            let c = d.counts;
            s.push_str(&format!(
                "| {name} | {ds} | {} | {} | {} | {} | {} | {} | {} |\n",
                pct(d.wer),
                c.substitutions,
                c.deletions,
                c.insertions,
                c.ref_words,
                d.utterances,
                d.silence
                    .insertions_per_hour
                    .map_or_else(|| "n/a".into(), |v| format!("{v:.1}")),
            ));
        }
        s.push_str(&format!(
            "| {name} | **macro** | {} | | | | | | |\n| {name} | **pooled** | {} | | | | | | |\n",
            pct(rep.macro_wer),
            pct(rep.pooled_wer)
        ));
    }
    if let Some(c) = &r.comparison {
        let m = &c.macro_delta;
        s.push_str(&format!(
            "\nDelta macro WER (B - A): {:+.3} points ({}), one-sided {:.0}% bounds [{:+.3}, {:+.3}], {} replicates, seed {}{}\n",
            100.0 * m.delta,
            m.relative_delta
                .map_or_else(|| "n/a relative".into(), |v| format!("{:+.1}% relative", 100.0 * v)),
            100.0 * (1.0 - m.alpha),
            100.0 * m.lower,
            100.0 * m.upper,
            m.replicates,
            m.seed,
            if m.limited { " (LIMITED: sparse stratum)" } else { "" }
        ));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bootstrap::MIN_REPLICATES;

    fn rec(id: &str, spk: &str, ds: &str, r: &str, h: &str) -> String {
        serde_json::json!({"id": id, "speaker": spk, "dataset": ds, "ref": r, "hyp": h}).to_string()
    }

    fn cfg() -> BootstrapConfig {
        BootstrapConfig {
            replicates: MIN_REPLICATES,
            seed: 1,
            alpha: 0.05,
        }
    }

    #[test]
    fn single_system_report_and_markdown() {
        let text = [
            rec("1", "s1", "x", "Hello world", "hello word"),
            rec("2", "s2", "x", "It's fine", "it is fine"),
            rec("3", "s1", "y", "a b c d", "a b c d e"),
            String::new(),
        ]
        .join("\n");
        let recs = parse_jsonl(&text).unwrap();
        let w = MacroWeights::equal(scorable_datasets(&recs)).unwrap();
        let r = score_report(&recs, None, &w, &cfg()).unwrap();
        assert_eq!(r.system_a.datasets["x"].counts.substitutions, 1);
        assert_eq!(r.system_a.datasets["x"].counts.ref_words, 5);
        assert_eq!(r.system_a.datasets["y"].counts.insertions, 1);
        assert!((r.system_a.macro_wer.unwrap() - (0.2 + 0.25) / 2.0).abs() < 1e-12);
        let md = markdown(&r);
        assert!(md.contains("| A | x | 20.00 | 1 | 0 | 0 | 5 | 2 |"), "{md}");
        assert_eq!(r.normaliser_id, NORMALISER_ID);
    }

    #[test]
    fn duplicate_ids_and_mismatched_refs_are_refused() {
        let dup = [rec("1", "s", "x", "a", "a"), rec("1", "s", "x", "b", "b")].join("\n");
        assert!(parse_jsonl(&dup).is_err());
        let a = parse_jsonl(&rec("1", "s", "x", "a b", "a b")).unwrap();
        let b = parse_jsonl(&rec("1", "s", "x", "a c", "a c")).unwrap();
        let w = MacroWeights::equal(["x".to_string()]).unwrap();
        assert!(score_report(&a, Some(&b), &w, &cfg()).is_err());
    }

    #[test]
    fn paired_comparison_detects_better_system() {
        let mut a = Vec::new();
        let mut b = Vec::new();
        for i in 0..30 {
            let spk = format!("s{i}");
            let r = "one two three four five six seven eight nine ten";
            a.push(rec(
                &i.to_string(),
                &spk,
                "x",
                r,
                "one two tree four fife six seven ate nine ten",
            ));
            b.push(rec(
                &i.to_string(),
                &spk,
                "x",
                r,
                "one two three four five six seven ate nine ten",
            ));
        }
        let a = parse_jsonl(&a.join("\n")).unwrap();
        let b = parse_jsonl(&b.join("\n")).unwrap();
        let w = MacroWeights::equal(["x".to_string()]).unwrap();
        let r = score_report(&a, Some(&b), &w, &cfg()).unwrap();
        let c = r.comparison.unwrap();
        assert!((c.macro_delta.delta + 0.2).abs() < 1e-12);
        assert!(c.macro_delta.upper < 0.0);
        assert_eq!(c.per_dataset["x"].delta, c.pooled_delta.delta);
    }
}
