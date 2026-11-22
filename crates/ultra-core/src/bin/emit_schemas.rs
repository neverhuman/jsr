//! Write the event and receipt JSON Schemas from the crate source into `contracts/events/`.
//!
//! The files under `crates/ultra-core/src/schemas/` are the source. The copies under
//! `contracts/events/` are generated. `cargo run -p ultra-core --bin emit-schemas --locked`
//! rewrites them. `--check` fails if a generated file has drifted.

use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

use ultra_core::repair::RepairHint;

/// Tell the next agent where the schema source is and which command rewrites the copy.
fn fail(code: u8, msg: &str) -> ExitCode {
    let hint = RepairHint {
        purpose: "keep contracts/events identical to crates/ultra-core/src/schemas",
        reason: "the generated schema was hand-edited, or emit-schemas could not read or write it",
        common_fixes: &[
            "edit crates/ultra-core/src/schemas, not contracts/events",
            "run cargo run -p ultra-core --bin emit-schemas --locked",
        ],
        docs_url: "docs/generated-zones.md",
        repair_hint: "cargo test -p ultra-core --test schema_contract",
    };
    eprintln!("emit-schemas: {msg}");
    eprintln!("purpose: {}", hint.purpose);
    eprintln!("reason: {}", hint.reason);
    for fix in hint.common_fixes {
        eprintln!("common_fixes: {fix}");
    }
    eprintln!("docs_url: {}", hint.docs_url);
    eprintln!("repair_hint: {}", hint.repair_hint);
    ExitCode::from(code)
}

const RECEIPT: &str = include_str!("../schemas/receipt.v1.schema.json");
const EVENT: &str = include_str!("../schemas/transcript-event.v1.schema.json");

fn main() -> ExitCode {
    let mut check = false;
    let mut out: Option<PathBuf> = None;
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--check" => check = true,
            "--out" => match args.next() {
                Some(path) => out = Some(PathBuf::from(path)),
                None => return fail(2, "--out needs a directory"),
            },
            other => return fail(2, &format!("unknown argument {other}")),
        }
    }
    let dest = out.unwrap_or_else(|| {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../contracts/events")
    });
    for (name, body) in [
        ("receipt.v1.schema.json", RECEIPT),
        ("transcript-event.v1.schema.json", EVENT),
    ] {
        let path = dest.join(name);
        if check {
            match fs::read_to_string(&path) {
                Ok(disk) if disk == body => {}
                Ok(_) => {
                    return fail(
                        1,
                        &format!("{path:?} drifted from crates/ultra-core/src/schemas/{name}"),
                    );
                }
                Err(e) => return fail(1, &format!("reading {path:?}: {e}")),
            }
        } else if let Err(e) = fs::write(&path, body) {
            return fail(1, &format!("writing {path:?}: {e}"));
        }
    }
    ExitCode::SUCCESS
}
