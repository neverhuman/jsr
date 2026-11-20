//! `prepare CACHE OUT DATASET COUNT | run JOB.json | service PLAN.json | score RUN CACHE`.

use std::path::Path;
use ultra_bridge::{nemotron, nemotron_service, records, require, run, service, Result};

fn main() {
    if let Err(e) = execute() {
        eprintln!("ultra-bridge: {e}");
        std::process::exit(1);
    }
}

fn execute() -> Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    match a.first().map(String::as_str) {
        Some("prepare") if a.len() == 5 => records::prepare(
            Path::new(&a[1]),
            Path::new(&a[2]),
            a[3].clone(),
            a[4].parse()?,
        ),
        Some("run") if a.len() == 2 => run::execute(Path::new(&a[1])),
        Some("run-nemotron") if a.len() == 2 => nemotron::execute(Path::new(&a[1])),
        Some("service-nemotron") if a.len() == 2 => {
            nemotron_service::execute(&records::read_json(Path::new(&a[1]))?)
        }
        Some("service") if a.len() == 2 => service::execute(&records::read_json(Path::new(&a[1]))?),
        Some("score") if a.len() == 3 => run::score(Path::new(&a[1]), Path::new(&a[2])),
        _ => require(
            false,
            "usage: ultra-bridge prepare CACHE OUT DATASET COUNT | run JOB.json | score RUN CACHE",
        ),
    }
}
