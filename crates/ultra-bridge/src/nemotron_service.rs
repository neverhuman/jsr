//! Reference-blind WAV preparation, process execution, strict file collection and drain.

use std::collections::{BTreeMap, BTreeSet};
use std::time::Instant;

use ultra_core::events::{Emitter, ProcessingEvidence};

use crate::nemotron::{run_child, Plan};
use crate::records::{
    hash_file, json_line, read_json, validate_outcomes, write_json, Outcome, Workload, SCHEMA,
};
use crate::{require, Result};

/// Refuse missing, extra, symlinked or non-UTF8 hypothesis files; never invent empty output.
pub fn collect_texts(
    dir: &std::path::Path,
    expected: &BTreeSet<String>,
) -> Result<BTreeMap<String, String>> {
    let mut texts = BTreeMap::new();
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        require(entry.file_type()?.is_file(), "non-file CLI output")?;
        if path.extension().is_some_and(|x| x == "txt") {
            let id = path
                .file_stem()
                .and_then(|x| x.to_str())
                .ok_or("malformed output filename")?;
            require(expected.contains(id), "unexpected hypothesis file")?;
            let text = String::from_utf8(std::fs::read(&path)?)?;
            require(
                text.len() <= ultra_core::events::MAX_PAYLOAD_BYTES,
                "oversized file hypothesis",
            )?;
            require(
                texts
                    .insert(id.to_owned(), text.trim().to_owned())
                    .is_none(),
                "duplicate hypothesis",
            )?;
        }
    }
    require(
        texts.keys().cloned().collect::<BTreeSet<_>>() == *expected,
        "expected hypothesis file absent",
    )?;
    Ok(texts)
}

pub fn execute(plan: &Plan) -> Result<()> {
    let j = &plan.job;
    std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(j.out.join("service.started"))?;
    require(
        plan.schema == SCHEMA
            && hash_file(&std::env::current_exe()?)? == plan.executable_sha256
            && hash_file(&j.inputs)? == plan.workload_sha256
            && hash_file(&j.binary)? == plan.binary_sha256
            && hash_file(&j.model)? == plan.model_sha256
            && hash_file(&j.candidate)? == plan.candidate_sha256,
        "file-control identity changed",
    )?;
    for (p, h) in &plan.libraries {
        require(
            hash_file(std::path::Path::new(p))? == *h,
            "native dependency changed",
        )?;
    }
    let work: Workload = read_json(&j.inputs)?;
    work.validate()?;
    let root = j.inputs.parent().ok_or("audio directory")?;
    let started = Instant::now();
    let wav_dir = j.out.join("wav");
    std::fs::create_dir(&wav_dir)?;
    // Outputs start empty, including metadata. The CLI cannot reuse a previous result.
    std::fs::create_dir(j.out.join("hyp"))?;
    let mut wav_hashes = BTreeMap::new();
    for i in &work.inputs {
        let path = root.join(&i.path);
        let raw = std::fs::read(path)?;
        require(
            raw.len() as u64 == i.samples * 2 && ultra_core::receipt::sha256_hex(&raw) == i.sha256,
            "wrong PCM input",
        )?;
        let pcm: Vec<_> = raw
            .chunks_exact(2)
            .map(|s| i16::from_le_bytes([s[0], s[1]]))
            .collect();
        let wav = wav_dir.join(format!("{}.wav", i.id));
        ultra_frontend::wav::write_wav_pcm16(&wav, 16000, &pcm)?;
        wav_hashes.insert(i.id.clone(), hash_file(&wav)?);
    }
    let argv = crate::nemotron::cli_args(plan);
    write_json(&j.out.join("effective-cli.json"), &argv)?;
    let cpu_check = run_child(plan)?;
    let bench: serde_json::Value = read_json(&j.out.join("cli.json"))?;
    let runs = bench["runs"].as_array().ok_or("CLI run metrics absent")?;
    require(runs.len() == 1, "unexpected CLI repetitions")?;
    let chunk_ms: ultra_core::receipt::ChunkMs =
        serde_json::from_value(runs[0]["metrics"]["chunk_latency_ms"].clone())?;
    require(
        [chunk_ms.p50, chunk_ms.p95, chunk_ms.p99]
            .iter()
            .all(|v| v.is_some_and(|x| x.is_finite() && x >= 0.0)),
        "missing/nonfinite CLI timing",
    )?;
    let expected: BTreeSet<_> = work.inputs.iter().map(|i| i.id.clone()).collect();
    let texts = collect_texts(&j.out.join("hyp"), &expected)?;
    let mut output = std::fs::File::create(j.out.join("outcomes.jsonl"))?;
    let mut events = std::fs::File::create(j.out.join("events.jsonl"))?;
    let mut finals = std::fs::File::create(j.out.join("final.jsonl"))?;
    std::fs::File::create(j.out.join("preview.jsonl"))?;
    std::fs::File::create(j.out.join("commit.jsonl"))?;
    let mut outcomes = Vec::new();
    for i in &work.inputs {
        require(
            hash_file(&wav_dir.join(format!("{}.wav", i.id)))? == wav_hashes[&i.id],
            "WAV changed during inference",
        )?;
        let text = texts.get(&i.id).ok_or("missing actual output")?;
        let observed_s = started.elapsed().as_secs_f64();
        let outcome = Outcome {
            run_id: plan.run_id.clone(),
            worker_id: "w0".into(),
            input_id: i.id.clone(),
            source_index: i.index,
            pcm_sha256: i.sha256.clone(),
            accepted_samples: i.samples,
            processed_samples: i.samples,
            final_text: text.clone(),
            forced_advances: None,
            source_start_s: 0.0,
            final_observed_s: observed_s,
        };
        let event = Emitter::new(&i.id, &plan.model_sha256).finalize(
            started.elapsed().as_nanos().try_into()?,
            text,
            i.samples,
            ProcessingEvidence {
                accepted_through_sample: i.samples,
                required_processed_through_sample: i.samples,
                flushed: true,
            },
        )?;
        json_line(&mut output, &outcome)?;
        json_line(
            &mut events,
            &serde_json::json!({"run_id": plan.run_id, "worker_id": "w0", "input_id": i.id,
            "scope": "file-mode collection observation; not live latency", "timed": ultra_eval::stream::TimedEvent { t_s: observed_s, event }}),
        )?;
        json_line(
            &mut finals,
            &serde_json::json!({"input_id": i.id, "text": text}),
        )?;
        outcomes.push(outcome);
    }
    validate_outcomes(&plan.run_id, &work, &outcomes)?;
    write_json(&j.out.join("wav-checksums.json"), &wav_hashes)?;
    write_json(
        &j.out.join("service.json"),
        &serde_json::json!({"schema": SCHEMA, "run_id": plan.run_id,
        "complete": true, "expected": expected.len(), "completed": outcomes.len(), "unfinished_samples": 0,
        "cpu_check": cpu_check, "chunk_ms": chunk_ms, "cold_load": {"load_ms": bench["load_ms"], "warmup_ms": bench["warmup_ms"]},
        "mode": "completed files; internal cache-aware chunks; no live timing or pacing claim",
        "service_wall_s": started.elapsed().as_secs_f64()}),
    )
}
