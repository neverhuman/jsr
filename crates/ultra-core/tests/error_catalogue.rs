//! Every `CoreError` variant has an entry in `agent/exceptions.toml` and a
//! section in `docs/errors.md`. The exhaustive match makes a new variant a
//! compile error here until it is catalogued.

use std::path::PathBuf;

use ultra_core::CoreError;

fn name(e: &CoreError) -> &'static str {
    match e {
        CoreError::BadShift(_) => "BadShift",
        CoreError::Shape(_) => "Shape",
        CoreError::Overflow { .. } => "Overflow",
        CoreError::WeightRange => "WeightRange",
        CoreError::ActivationRange => "ActivationRange",
        CoreError::BadTernaryCode => "BadTernaryCode",
        CoreError::Contract(_) => "Contract",
        CoreError::Lifecycle(_) => "Lifecycle",
        CoreError::Receipt(_) => "Receipt",
    }
}

fn every_variant() -> Vec<CoreError> {
    vec![
        CoreError::BadShift(0),
        CoreError::Shape("x"),
        CoreError::Overflow {
            fan_in: 1,
            max_abs_w: 1,
        },
        CoreError::WeightRange,
        CoreError::ActivationRange,
        CoreError::BadTernaryCode,
        CoreError::Contract("x"),
        CoreError::Lifecycle("x"),
        CoreError::Receipt("x"),
    ]
}

/// Fields every catalogue entry must carry so a failure tells the next agent
/// why it happened, how it is usually fixed, and what to rerun.
const REQUIRED_FIELDS: &[&str] = &[
    "purpose",
    "reason",
    "common_fixes",
    "docs_url",
    "repair_hint",
];

fn repo_file(rel: &str) -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(rel);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {rel}: {e}"))
}

#[test]
fn every_error_kind_is_catalogued() {
    let catalogue = repo_file("agent/exceptions.toml");
    let docs = repo_file("docs/errors.md");
    let entries = catalogue.matches("[[error_kind]]").count();
    assert_eq!(
        entries,
        every_variant().len(),
        "stale entries in agent/exceptions.toml"
    );
    for e in every_variant() {
        let n = name(&e);
        assert!(
            catalogue.contains(&format!("name = \"{n}\"")),
            "agent/exceptions.toml lacks {n}"
        );
        assert!(
            catalogue.contains(&format!("docs/errors.md#{}", n.to_lowercase())),
            "agent/exceptions.toml entry {n} lacks its docs_url"
        );
        assert!(
            docs.contains(&format!("### {n}")),
            "docs/errors.md lacks ### {n}"
        );
    }
}

#[test]
fn every_catalogue_entry_carries_the_repair_fields() {
    let catalogue = repo_file("agent/exceptions.toml");
    let blocks: Vec<&str> = catalogue.split("[[error_kind]]").skip(1).collect();
    assert!(!blocks.is_empty());
    for block in blocks {
        let entry = block
            .lines()
            .find(|l| l.starts_with("name = "))
            .unwrap_or("<unnamed>");
        for field in REQUIRED_FIELDS {
            assert!(
                block.lines().any(|l| l.starts_with(&format!("{field} = "))),
                "{entry} lacks {field}"
            );
        }
        assert!(
            block.contains("common_fixes = [\n  \""),
            "{entry}: common_fixes must list at least one fix"
        );
    }
}

/// `CoreError::repair()` (generated into src/repair.rs) must agree with agent/exceptions.toml.
#[test]
fn repair_hints_match_catalogue() {
    let toml = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../agent/exceptions.toml"
    ))
    .expect("catalogue readable");
    let all = [
        ultra_core::CoreError::BadShift(0),
        ultra_core::CoreError::Shape("x"),
        ultra_core::CoreError::Overflow {
            fan_in: 1,
            max_abs_w: 1,
        },
        ultra_core::CoreError::WeightRange,
        ultra_core::CoreError::ActivationRange,
        ultra_core::CoreError::BadTernaryCode,
        ultra_core::CoreError::Contract("x"),
        ultra_core::CoreError::Lifecycle("x"),
        ultra_core::CoreError::Receipt("x"),
    ];
    for e in all {
        let entry = toml
            .split("[[error_kind]]")
            .find(|b| b.contains(&format!("name = \"{}\"", e.kind())))
            .unwrap_or_else(|| panic!("{} not catalogued", e.kind()));
        let h = e.repair();
        for (field, value) in [
            ("purpose", h.purpose),
            ("reason", h.reason),
            ("docs_url", h.docs_url),
            ("repair_hint", h.repair_hint),
        ] {
            assert!(
                entry.contains(&format!("{field} = \"{value}\"")),
                "{}: {field} drifted from the catalogue",
                e.kind()
            );
        }
        for fix in h.common_fixes {
            assert!(
                entry.contains(&format!("\"{fix}\"")),
                "{}: fix drifted: {fix}",
                e.kind()
            );
        }
    }
}
