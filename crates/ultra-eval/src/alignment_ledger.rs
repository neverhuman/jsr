//! Word-level canonical scoring evidence. Input annotations index normalized words.

use std::collections::BTreeMap;
use std::io::Write;

use serde::{Deserialize, Serialize};

use crate::normalise::{normalise_words, normaliser_sha256, NORMALISER_ID};
use crate::score::{ScoreRecord, SCORE_SCHEMA};
use crate::wer::{align, align_with, AlignOp, Alignment, EditCounts, TieBreak};
use crate::EvalError;

/// Run identities supplied by the caller from the prediction run's receipt.
#[derive(Debug, Serialize, Deserialize)]
pub struct RunIdentity {
    pub audio_manifest_sha256: String,
    pub artifact_sha256: String,
    pub runtime_sha256: String,
    pub decoder: String,
    pub chunk_policy: String,
}

impl RunIdentity {
    pub fn validate(&self) -> Result<(), EvalError> {
        let is_digest = ultra_core::receipt::is_sha256_hex;
        if !is_digest(&self.audio_manifest_sha256)
            || !is_digest(&self.artifact_sha256)
            || !is_digest(&self.runtime_sha256)
            || self.decoder.trim().is_empty()
            || self.chunk_policy.trim().is_empty()
        {
            return Err(EvalError::Input("invalid alignment run identity".into()));
        }
        Ok(())
    }
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct WordEvidence {
    /// Half-open interval in seconds on the clock named by the record.
    time_s: Option<[f64; 2]>,
    /// Half-open interval in the corresponding sequence of model tokens.
    token_span: Option<[usize; 2]>,
    train_frequency: Option<u64>,
    function_word: Option<bool>,
    phones: Option<String>,
    morphology: Option<String>,
}

#[derive(Debug, Deserialize)]
struct TimedWord {
    w: String,
    start_s: f64,
    end_s: f64,
}

#[derive(Debug, Deserialize)]
struct LedgerInput {
    #[serde(flatten)]
    score: ScoreRecord,
    ref_words: Option<Vec<WordEvidence>>,
    hyp_words: Option<Vec<WordEvidence>>,
    /// Source word timing before canonical normalization.
    reference_timing: Option<Vec<TimedWord>>,
    timing_clock: Option<String>,
    condition: Option<String>,
    session: Option<String>,
    /// Hypotheses in decoder rank order, including the selected hypothesis.
    candidates: Option<Vec<String>>,
}

struct PreparedRecord {
    input: LedgerInput,
    reference: Vec<String>,
    hypothesis: Vec<String>,
    alignment: Alignment,
    candidate_ranks: Option<Vec<Option<usize>>>,
}

/// Validated annotations and normalized texts from one score input.
pub struct LedgerRecords(Vec<PreparedRecord>);

impl LedgerRecords {
    pub fn parse(jsonl: &str) -> Result<Self, EvalError> {
        jsonl
            .lines()
            .enumerate()
            .filter(|(_, line)| !line.trim().is_empty())
            .map(|(index, line)| {
                let input = serde_json::from_str(line)
                    .map_err(|e| EvalError::Input(format!("line {}: {e}", index + 1)))?;
                PreparedRecord::new(input)
            })
            .collect::<Result<Vec<_>, _>>()
            .map(Self)
    }
}

impl PreparedRecord {
    fn new(mut input: LedgerInput) -> Result<Self, EvalError> {
        let reference = normalise_words(&input.score.reference);
        let hypothesis = normalise_words(&input.score.hyp);
        let invalid = |detail| EvalError::Input(format!("{}: {detail}", input.score.id));
        if input.score.duration_s.is_some_and(|s| s < 0.0) {
            return Err(invalid("negative duration"));
        }
        for (annotations, words) in [
            (&input.ref_words, &reference),
            (&input.hyp_words, &hypothesis),
        ] {
            let Some(annotations) = annotations else {
                continue;
            };
            if annotations.len() != words.len() {
                return Err(invalid("word evidence length differs from normalized text"));
            }
            for annotation in annotations {
                if annotation.time_s.is_some_and(|[start, end]| {
                    start < 0.0
                        || end < start
                        || input.timing_clock.as_deref().is_none_or(str::is_empty)
                }) || annotation
                    .token_span
                    .is_some_and(|[start, end]| end <= start)
                {
                    return Err(invalid("invalid timing or token span"));
                }
            }
        }
        if let Some(timing) = input.reference_timing.take() {
            let timed = timing.into_iter().flat_map(|word| {
                let span = [word.start_s, word.end_s];
                normalise_words(&word.w).into_iter().map(move |w| (w, span))
            });
            let annotations = input.ref_words.get_or_insert_with(|| {
                (0..reference.len())
                    .map(|_| WordEvidence::default())
                    .collect()
            });
            let mut count = 0;
            for (index, (word, [start, end])) in timed.enumerate() {
                if reference.get(index) != Some(&word) {
                    return Err(invalid(
                        "reference timing words differ from normalized text",
                    ));
                }
                let span = [start, end];
                if start < 0.0
                    || end < start
                    || input.timing_clock.as_deref().is_none_or(str::is_empty)
                    || annotations[index]
                        .time_s
                        .is_some_and(|previous| previous != span)
                {
                    return Err(invalid("invalid or conflicting reference timing span"));
                }
                annotations[index].time_s = Some(span);
                count += 1;
            }
            if count != reference.len() {
                return Err(invalid(
                    "reference timing words differ from normalized text",
                ));
            }
        }
        let candidate_ranks = if let Some(candidates) = &input.candidates {
            if candidates
                .first()
                .is_none_or(|first| normalise_words(first) != hypothesis)
            {
                return Err(invalid(
                    "first candidate must match the selected hypothesis",
                ));
            }
            let mut ranks = vec![None; reference.len()];
            for (i, candidate) in candidates.iter().enumerate() {
                let words = normalise_words(candidate);
                for op in align(&reference, &words).ops {
                    if let AlignOp::Match { r, .. } = op {
                        ranks[r].get_or_insert(i + 1);
                    }
                }
            }
            Some(ranks)
        } else {
            None
        };
        let alignment = align(&reference, &hypothesis);
        Ok(Self {
            input,
            reference,
            hypothesis,
            alignment,
            candidate_ranks,
        })
    }
}

#[derive(Serialize)]
struct Word<'a> {
    index: usize,
    word: &'a str,
    evidence: Option<&'a WordEvidence>,
}

fn word<'a>(
    index: Option<usize>,
    words: &'a [String],
    evidence: &'a Option<Vec<WordEvidence>>,
) -> Option<Word<'a>> {
    index.map(|index| Word {
        index,
        word: &words[index],
        evidence: evidence.as_ref().map(|items| &items[index]),
    })
}

#[derive(Serialize)]
struct Row<'a> {
    kind: &'static str,
    system: &'a str,
    id: &'a str,
    split: &'a str,
    speaker: &'a str,
    session: Option<&'a str>,
    condition: Option<&'a str>,
    duration_s: Option<f64>,
    timing_clock: Option<&'a str>,
    reference_words: usize,
    op: &'static str,
    /// Number of reference words consumed before this operation, including for insertions.
    reference_boundary: usize,
    reference: Option<Word<'a>>,
    hypothesis: Option<Word<'a>>,
    candidates_examined: Option<usize>,
    first_correct_candidate_rank: Option<usize>,
}

#[derive(Default)]
struct Slices<'a>(BTreeMap<(&'a str, &'static str, String), EditCounts>);

impl<'a> Slices<'a> {
    fn add(&mut self, row: &Row<'a>) {
        let counts = EditCounts {
            ref_words: u64::from(row.reference.is_some()),
            hyp_words: u64::from(row.hypothesis.is_some()),
            matches: u64::from(row.op == "match"),
            substitutions: u64::from(row.op == "sub"),
            deletions: u64::from(row.op == "del"),
            insertions: u64::from(row.op == "ins"),
        };
        let mut add = |dimension, bucket: &str| {
            self.0
                .entry((row.split, dimension, bucket.to_owned()))
                .or_default()
                .add(&counts);
        };
        add("speaker", row.speaker);
        add("condition", row.condition.unwrap_or("unknown"));
        add(
            "duration_s",
            match row.duration_s {
                Some(s) if s < 3.0 => "[0,3)",
                Some(s) if s < 10.0 => "[3,10)",
                Some(_) => "[10,inf)",
                None => "unknown",
            },
        );
        add(
            "word_position",
            match (
                row.reference.is_some(),
                row.reference_boundary,
                row.reference_words,
            ) {
                (_, _, 0) => "no_reference",
                (false, 0, _) => "leading_insertion",
                (false, boundary, len) if boundary == len => "trailing_insertion",
                (false, _, _) => "interior_insertion",
                (true, _, 1) => "single",
                (true, 0, _) => "first",
                (true, i, len) if i + 1 == len => "last",
                (true, _, _) => "interior",
            },
        );
        let reference = row.reference.as_ref().and_then(|w| w.evidence);
        let hypothesis = row.hypothesis.as_ref().and_then(|w| w.evidence);
        add(
            "train_frequency",
            match reference.and_then(|w| w.train_frequency) {
                Some(0) => "0",
                Some(1..=9) => "1-9",
                Some(10..=99) => "10-99",
                Some(100..) => "100+",
                None => "unknown",
            },
        );
        add(
            "function_word",
            match reference.and_then(|w| w.function_word) {
                Some(true) => "yes",
                Some(false) => "no",
                None => "unknown",
            },
        );
        add(
            "reference_phones",
            reference
                .and_then(|w| w.phones.as_deref())
                .unwrap_or("unknown"),
        );
        add(
            "reference_morphology",
            reference
                .and_then(|w| w.morphology.as_deref())
                .unwrap_or("unknown"),
        );
        if row.op == "sub" {
            if let (Some(r), Some(h)) = (reference, hypothesis) {
                if let (Some(a), Some(b)) = (&r.phones, &h.phones) {
                    add("phonetic_confusion", &format!("{a} → {b}"));
                }
                if let (Some(a), Some(b)) = (&r.morphology, &h.morphology) {
                    add("morphology_confusion", &format!("{a} → {b}"));
                }
                if let (Some(a), Some(b)) = (r.function_word, h.function_word) {
                    add("function_word_confusion", &format!("{a} → {b}"));
                }
            }
        }
    }
}

fn line(out: &mut impl Write, record: &impl Serialize) -> Result<(), EvalError> {
    serde_json::to_writer(&mut *out, record).map_err(|e| EvalError::Io(e.to_string()))?;
    out.write_all(b"\n")?;
    Ok(())
}

/// Write the run receipt, canonical alignment rows including matches, and per-split
/// sensitivity counts from the alternate equal-cost alignment.
pub fn write_system(
    out: &mut impl Write,
    records: &LedgerRecords,
    system: &str,
    run: Option<&RunIdentity>,
    input_sha256: &str,
    scorer_sha256: &str,
) -> Result<(), EvalError> {
    line(
        out,
        &serde_json::json!({
            "kind": "receipt",
            "schema": "ultra.alignment-ledger/v1",
            "system": system,
            "score_input_sha256": input_sha256,
            "run": run,
            "scorer": {
                "schema": SCORE_SCHEMA,
                "executable_sha256": scorer_sha256,
                "normaliser_id": NORMALISER_ID,
                "normaliser_sha256": normaliser_sha256(),
                "tie_break": "diagonal,deletion,insertion"
            }
        }),
    )?;
    let mut sensitivity = BTreeMap::<&str, (EditCounts, EditCounts)>::new();
    let mut slices = Slices::default();
    for record in &records.0 {
        let input = &record.input;
        let counts = sensitivity.entry(&input.score.dataset).or_default();
        counts.0.add(&record.alignment.counts);
        counts
            .1
            .add(&align_with(&record.reference, &record.hypothesis, TieBreak::Earliest).counts);
        let mut reference_boundary = 0;
        for &op in &record.alignment.ops {
            let (operation, r, h) = match op {
                AlignOp::Match { r, h } => ("match", Some(r), Some(h)),
                AlignOp::Sub { r, h } => ("sub", Some(r), Some(h)),
                AlignOp::Del { r } => ("del", Some(r), None),
                AlignOp::Ins { h } => ("ins", None, Some(h)),
            };
            let row = Row {
                kind: "alignment",
                system,
                id: &input.score.id,
                split: &input.score.dataset,
                speaker: &input.score.speaker,
                session: input.session.as_deref(),
                condition: input.condition.as_deref(),
                duration_s: input.score.duration_s,
                timing_clock: input.timing_clock.as_deref(),
                reference_words: record.reference.len(),
                op: operation,
                reference_boundary,
                reference: word(r, &record.reference, &input.ref_words),
                hypothesis: word(h, &record.hypothesis, &input.hyp_words),
                candidates_examined: input.candidates.as_ref().map(Vec::len),
                first_correct_candidate_rank: r
                    .and_then(|i| record.candidate_ranks.as_ref().and_then(|ranks| ranks[i])),
            };
            slices.add(&row);
            line(out, &row)?;
            reference_boundary += usize::from(r.is_some());
        }
    }
    for (split, (canonical, alternate)) in sensitivity {
        line(
            out,
            &serde_json::json!({
                "kind": "alignment_sensitivity",
                "system": system,
                "split": split,
                "canonical": canonical,
                "alternate": alternate,
                "alternate_tie_break": "deletion,insertion,diagonal"
            }),
        )?;
    }
    for ((split, dimension, bucket), counts) in slices.0 {
        line(
            out,
            &serde_json::json!({
                "kind": "slice",
                "system": system,
                "split": split,
                "dimension": dimension,
                "bucket": bucket,
                "counts": counts
            }),
        )?;
    }
    Ok(())
}

fn reference_ops(record: &PreparedRecord) -> impl Iterator<Item = &'static str> + '_ {
    record.alignment.ops.iter().filter_map(|op| match op {
        AlignOp::Match { .. } => Some("match"),
        AlignOp::Sub { .. } => Some("sub"),
        AlignOp::Del { .. } => Some("del"),
        AlignOp::Ins { .. } => None,
    })
}

/// Compare the two canonical operations at every reference position. Insertions
/// have no reference position and carry their boundary in the word ledger.
/// Call after the paired report has confirmed the inputs represent the same clips.
pub fn write_paired_transitions(
    out: &mut impl Write,
    a: &LedgerRecords,
    b: &LedgerRecords,
) -> Result<(), EvalError> {
    let by_id: BTreeMap<_, _> = b.0.iter().map(|r| (r.input.score.id.as_str(), r)).collect();
    let mut transitions = BTreeMap::<_, u64>::new();
    for left in &a.0 {
        let score = &left.input.score;
        let right = by_id
            .get(score.id.as_str())
            .ok_or_else(|| EvalError::Input(format!("id {} missing from system B", score.id)))?;
        for (op_a, op_b) in reference_ops(left).zip(reference_ops(right)) {
            *transitions
                .entry((score.dataset.as_str(), op_a, op_b))
                .or_default() += 1;
        }
    }
    for ((split, op_a, op_b), ref_words) in transitions {
        line(
            out,
            &serde_json::json!({
                "kind": "paired_transition",
                "split": split,
                "op_a": op_a,
                "op_b": op_b,
                "ref_words": ref_words
            }),
        )?;
    }
    Ok(())
}
