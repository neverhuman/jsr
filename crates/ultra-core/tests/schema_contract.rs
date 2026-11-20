//! Contract drift: serialised ultra-core values must validate against the
//! hand-maintained schemas in `contracts/events/`, and the schemas must refuse
//! what the Rust types refuse.
//!
//! The validator below implements only the JSON Schema keywords these schemas
//! use. Any other keyword fails the test, so a schema edit cannot silently
//! outgrow the check.

use std::path::PathBuf;

use serde_json::{json, Map, Value};
use ultra_core::events::{
    Body, Emitter, ProcessingEvidence, TranscriptEvent, EVENT_SCHEMA, MAX_PAYLOAD_BYTES,
};
use ultra_core::receipt::{ChunkMs, Hashes, HonestyLabel, Partition, Receipt, RECEIPT_SCHEMA};

const ANNOTATIONS: &[&str] = &["$schema", "$id", "title", "description"];
const KEYWORDS: &[&str] = &[
    "type",
    "required",
    "properties",
    "additionalProperties",
    "const",
    "enum",
    "minimum",
    "maxLength",
    "items",
    "minItems",
    "maxItems",
    "allOf",
    "if",
    "then",
];

fn load(name: &str) -> Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../contracts/events")
        .join(name);
    let text = std::fs::read_to_string(&path).expect("schema file readable");
    serde_json::from_str(&text).expect("schema is JSON")
}

fn type_ok(want: &str, v: &Value) -> bool {
    match want {
        "null" => v.is_null(),
        "boolean" => v.is_boolean(),
        "string" => v.is_string(),
        "object" => v.is_object(),
        "array" => v.is_array(),
        "number" => v.is_number(),
        "integer" => v.is_u64() || v.is_i64(),
        other => panic!("unsupported schema type {other}"),
    }
}

/// Collect every violation of `schema` by `v` into `errs`, prefixed by `at`.
fn check(schema: &Value, v: &Value, at: &str, errs: &mut Vec<String>) {
    let s = schema.as_object().expect("schema node is an object");
    for k in s.keys() {
        assert!(
            ANNOTATIONS.contains(&k.as_str()) || KEYWORDS.contains(&k.as_str()),
            "schema keyword `{k}` at {at} is not supported by this check; extend schema_contract.rs"
        );
    }
    if let Some(t) = s.get("type") {
        let ok = match t {
            Value::String(one) => type_ok(one, v),
            Value::Array(any) => any.iter().any(|x| type_ok(x.as_str().unwrap_or(""), v)),
            _ => panic!("bad type keyword at {at}"),
        };
        if !ok {
            errs.push(format!("{at}: {v} is not of type {t}"));
            return;
        }
    }
    if let Some(c) = s.get("const") {
        if c != v {
            errs.push(format!("{at}: {v} != const {c}"));
        }
    }
    if let Some(Value::Array(opts)) = s.get("enum") {
        if !opts.contains(v) {
            errs.push(format!("{at}: {v} not in enum"));
        }
    }
    if let (Some(min), Some(n)) = (s.get("minimum").and_then(Value::as_f64), v.as_f64()) {
        if n < min {
            errs.push(format!("{at}: {n} < minimum {min}"));
        }
    }
    if let (Some(max), Some(text)) = (s.get("maxLength").and_then(Value::as_u64), v.as_str()) {
        if text.chars().count() as u64 > max {
            errs.push(format!("{at}: string longer than {max}"));
        }
    }
    if let Some(items) = v.as_array() {
        if let Some(min) = s.get("minItems").and_then(Value::as_u64) {
            if (items.len() as u64) < min {
                errs.push(format!("{at}: fewer than {min} items"));
            }
        }
        if let Some(max) = s.get("maxItems").and_then(Value::as_u64) {
            if items.len() as u64 > max {
                errs.push(format!("{at}: more than {max} items"));
            }
        }
        if let Some(item_schema) = s.get("items") {
            for (i, item) in items.iter().enumerate() {
                check(item_schema, item, &format!("{at}[{i}]"), errs);
            }
        }
    }
    if let Some(obj) = v.as_object() {
        check_object(s, obj, at, errs);
    }
    if let Some(Value::Array(all)) = s.get("allOf") {
        for sub in all {
            check(sub, v, at, errs);
        }
    }
    if let Some(cond) = s.get("if") {
        let mut probe = Vec::new();
        check(cond, v, at, &mut probe);
        if probe.is_empty() {
            if let Some(then) = s.get("then") {
                check(then, v, at, errs);
            }
        }
    }
}

fn check_object(
    s: &Map<String, Value>,
    obj: &Map<String, Value>,
    at: &str,
    errs: &mut Vec<String>,
) {
    if let Some(Value::Array(req)) = s.get("required") {
        for r in req {
            let key = r.as_str().expect("required entries are strings");
            if !obj.contains_key(key) {
                errs.push(format!("{at}: missing required `{key}`"));
            }
        }
    }
    let props = s.get("properties").and_then(Value::as_object);
    for (k, val) in obj {
        match props.and_then(|p| p.get(k)) {
            Some(sub) => check(sub, val, &format!("{at}.{k}"), errs),
            None if s.get("additionalProperties") == Some(&Value::Bool(false)) => {
                errs.push(format!("{at}: unexpected property `{k}`"));
            }
            None => {}
        }
    }
}

fn violations(schema: &Value, v: &Value) -> Vec<String> {
    let mut errs = Vec::new();
    check(schema, v, "$", &mut errs);
    errs
}

const DONE: ProcessingEvidence = ProcessingEvidence {
    accepted_through_sample: 48_000,
    required_processed_through_sample: 48_000,
    flushed: true,
};

/// One event of every kind, produced the way a backend would (Emitter), plus a correction.
fn every_event() -> Vec<TranscriptEvent> {
    let mut e = Emitter::new("stream-1", "zipformer-int8@test");
    let mut v = vec![
        e.preview(1, "hel", 16_000).expect("preview"),
        e.commit(2, "hello ", 6, 32_000).expect("commit"),
        e.finalize(3, "world", 48_000, DONE).expect("final"),
        e.preview(4, "par", 56_000).expect("preview"),
        e.incomplete(5, "deadline miss", 64_000, 60_000, false)
            .expect("incomplete"),
        e.preview(6, "x", 70_000).expect("preview"),
        e.gap(7, Some((70_000, 72_000)), "packet loss"),
        e.gap(8, None, "unknown extent"),
        e.status(9, "shed", "step 1"),
    ];
    v.push(TranscriptEvent {
        schema: EVENT_SCHEMA.into(),
        stream_id: "stream-1".into(),
        event_sequence: 9,
        state_epoch: 3,
        observed_monotonic_ns: 10,
        model_config_id: "zipformer-int8@test".into(),
        body: Body::CorrectionSuggested {
            segment_id: 0,
            final_revision: 3,
            final_sha256: ultra_core::receipt::sha256_hex(b"hello world"),
            byte_start: 0,
            byte_end: 5,
            alternative: "hullo".into(),
        },
    });
    v
}

fn receipt(measured: bool) -> Receipt {
    let m = |x: f64| measured.then_some(x);
    Receipt {
        schema: RECEIPT_SCHEMA.into(),
        bundle: "models/sha256-00".into(),
        source_commit: "b9c1335".into(),
        hardware: "H0".into(),
        target: "avx2".into(),
        configured_threads: 1,
        observed_threads: measured.then_some(1),
        cpu_only_verified: true,
        audio_seconds: 12.5,
        process_tree_cpu_seconds: m(1.25),
        peak_rss_bytes: measured.then_some(10 << 20),
        events: measured.then_some(40),
        wakeups: measured.then_some(80),
        cap_hits: measured.then_some(0),
        chunk_ms: ChunkMs {
            p50: m(1.0),
            p95: m(2.0),
            p99: m(3.0),
        },
        status_events: vec!["shed".into()],
        gaps: vec![(100, 200)],
        hashes: Hashes {
            events: "e".repeat(64),
            preview: "a".repeat(64),
            commit: "c".repeat(64),
            final_: "f".repeat(64),
        },
        label: HonestyLabel::InCorpus,
        partition: Partition::PublicTest,
        completion_status: "complete".into(),
    }
}

#[test]
fn every_event_kind_validates_against_the_schema_and_the_ledger() {
    let schema = load("transcript-event.v1.schema.json");
    let mut ledger = ultra_core::events::Ledger::new();
    for ev in every_event() {
        let v = serde_json::to_value(&ev).expect("serialises");
        let errs = violations(&schema, &v);
        assert!(errs.is_empty(), "{v}: {errs:?}");
        ledger
            .apply(&ev)
            .unwrap_or_else(|e| panic!("ledger refused {v}: {e}"));
        let back: TranscriptEvent = serde_json::from_value(v).expect("round-trips");
        assert_eq!(back, ev);
    }
}

#[test]
fn schema_event_types_match_the_rust_enum() {
    let schema = load("transcript-event.v1.schema.json");
    let listed: Vec<&str> = schema["properties"]["event_type"]["enum"]
        .as_array()
        .expect("event_type enum")
        .iter()
        .filter_map(Value::as_str)
        .collect();
    let mut emitted: Vec<String> = Vec::new();
    for ev in every_event() {
        let t = serde_json::to_value(&ev).expect("serialises")["event_type"]
            .as_str()
            .expect("tag")
            .to_owned();
        if !emitted.contains(&t) {
            emitted.push(t);
        }
    }
    let mut a: Vec<String> = listed.iter().map(|s| s.to_string()).collect();
    a.sort();
    emitted.sort();
    assert_eq!(
        a, emitted,
        "every event_type in the schema is produced by Rust and vice versa"
    );
}

#[test]
fn schema_refuses_what_the_contract_refuses() {
    let schema = load("transcript-event.v1.schema.json");
    let good = serde_json::to_value(&every_event()[1]).expect("commit");

    let mut unknown_type = good.clone();
    unknown_type["event_type"] = json!("Rewrite");
    assert!(!violations(&schema, &unknown_type).is_empty());

    let mut missing_snapshot = good.clone();
    missing_snapshot
        .as_object_mut()
        .expect("object")
        .remove("snapshot_utf8");
    assert!(!violations(&schema, &missing_snapshot).is_empty());

    let mut extra = good.clone();
    extra["confidence"] = json!(0.9);
    assert!(!violations(&schema, &extra).is_empty());

    let mut negative = good.clone();
    negative["segment_id"] = json!(-1);
    assert!(!violations(&schema, &negative).is_empty());

    let mut final_without_evidence = serde_json::to_value(&every_event()[2]).expect("final");
    final_without_evidence
        .as_object_mut()
        .expect("object")
        .remove("flushed");
    assert!(!violations(&schema, &final_without_evidence).is_empty());

    let mut big = good;
    big["snapshot_utf8"] = json!("x".repeat(MAX_PAYLOAD_BYTES + 1));
    assert!(!violations(&schema, &big).is_empty());
}

#[test]
fn receipts_validate_against_the_schema() {
    let schema = load("receipt.v1.schema.json");
    for measured in [true, false] {
        let v = serde_json::to_value(receipt(measured)).expect("serialises");
        let errs = violations(&schema, &v);
        assert!(errs.is_empty(), "measured={measured}: {errs:?}");
    }
}

#[test]
fn receipt_schema_required_matches_rust_non_optional_fields() {
    let schema = load("receipt.v1.schema.json");
    let full = serde_json::to_value(receipt(true)).expect("serialises");
    let required: Vec<&str> = schema["required"]
        .as_array()
        .expect("required list")
        .iter()
        .filter_map(Value::as_str)
        .collect();
    let obj = full.as_object().expect("object");

    // Keeping only the required keys must still deserialise: nothing else is mandatory in Rust.
    let minimal: Map<String, Value> = obj
        .iter()
        .filter(|(k, _)| required.contains(&k.as_str()))
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    let minimal = Value::Object(minimal);
    assert!(violations(&schema, &minimal).is_empty());
    serde_json::from_value::<Receipt>(minimal.clone()).expect("required keys suffice for Rust");

    // Dropping any required key must fail both the schema and Rust.
    for key in &required {
        let mut v = minimal.clone();
        v.as_object_mut().expect("object").remove(*key);
        assert!(
            !violations(&schema, &v).is_empty(),
            "schema accepted missing {key}"
        );
        assert!(
            serde_json::from_value::<Receipt>(v).is_err(),
            "Rust accepted missing {key}; drop it from the schema's required list"
        );
    }
}

#[test]
fn receipt_schema_refuses_bad_values() {
    let schema = load("receipt.v1.schema.json");
    let good = serde_json::to_value(receipt(true)).expect("serialises");

    let mut wrong_schema = good.clone();
    wrong_schema["schema"] = json!("ultra.receipt/v0");
    assert!(!violations(&schema, &wrong_schema).is_empty());

    let mut bad_partition = good.clone();
    bad_partition["partition"] = json!("TEST");
    assert!(!violations(&schema, &bad_partition).is_empty());

    let mut bad_gap = good.clone();
    bad_gap["gaps"] = json!([[1, 2, 3]]);
    assert!(!violations(&schema, &bad_gap).is_empty());

    let mut extra = good;
    extra["wer"] = json!(0.05);
    assert!(!violations(&schema, &extra).is_empty());
}
