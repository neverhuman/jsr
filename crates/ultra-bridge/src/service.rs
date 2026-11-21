//! Measured service: audio loading, capture pacing, IPC/native work, collection and drain.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::time::{Duration, Instant};

use ultra_core::events::{Emitter, ProcessingEvidence};

use crate::engine::Engine;
use crate::protocol::Packet;
use crate::records::{
    hash_file, json_line, read_json, validate_outcomes, write_json, Input, Observation, Outcome,
    Plan, Workload, SCHEMA,
};
use crate::{require, Result};

struct Slot {
    input: Input,
    file: File,
    cursor: u64,
    started: Instant,
    source_start_s: f64,
    emitter: Emitter,
    last_hyp: String,
}

/// Input and result integrity are independent of whether a hypothesis happens to be empty.
pub fn validate_batch(packets: &[Packet], rows: &[Observation]) -> Result<()> {
    require(
        packets.len() == rows.len(),
        "missing/extra worker observation",
    )?;
    let mut seen = BTreeSet::new();
    for r in rows {
        let p = packets
            .iter()
            .find(|p| p.id == r.id)
            .ok_or("unknown observation ID")?;
        let accepted = p.offset + p.pcm.len() as u64;
        require(
            seen.insert(&r.id)
                && r.accepted_samples == accepted
                && r.processed_samples <= accepted
                && r.terminal == p.finish
                && r.drained == p.finish
                && (r.pcm_sha256.is_some() == p.finish)
                && (!p.finish || r.processed_samples == accepted),
            "invalid processing/terminal evidence",
        )?;
    }
    Ok(())
}

pub fn execute(plan: &Plan) -> Result<()> {
    let j = &plan.job;
    j.validate()?;
    std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(j.out.join("service.started"))?;
    require(plan.schema == SCHEMA, "wrong plan schema")?;
    require(
        hash_file(&j.inputs)? == plan.workload_sha256
            && hash_file(&j.bundle_manifest_path())? == plan.bundle_manifest_sha256
            && hash_file(&std::env::current_exe()?)? == plan.executable_sha256,
        "run identity changed",
    )?;
    if j.engine == "r1-bridge" {
        require(
            hash_file(&j.worker)? == plan.worker_sha256
                && hash_file(
                    &j.worker
                        .parent()
                        .ok_or("worker parent")?
                        .join("../train/bundle_infer.py"),
                )? == plan.consumer_sha256,
            "worker/consumer source changed",
        )?;
    }
    let work: Workload = read_json(&j.inputs)?;
    work.validate()?;
    let root = j.inputs.parent().ok_or("missing workload directory")?;
    let _native_deadline = if j.engine != "r1-bridge" {
        Some(crate::service_deadline::ServiceDeadline::start(
            Duration::from_secs(j.timeout_s),
        )?)
    } else {
        None
    };
    let service_start = Instant::now();
    let (mut engine, info) = Engine::load(plan)?;
    let cold_snapshot = ultra_eval::procfs::TreeSampler::new(std::process::id() as i32).sample();
    let cold_tree_cpu_s = cold_snapshot.cpu_seconds_user + cold_snapshot.cpu_seconds_sys;
    write_json(&j.out.join("load.json"), &info)?;
    let mut events = File::create(j.out.join("events.jsonl"))?;
    let mut previews = File::create(j.out.join("preview.jsonl"))?;
    let mut commits = File::create(j.out.join("commit.jsonl"))?;
    let mut finals = File::create(j.out.join("final.jsonl"))?;
    let mut observations = File::create(j.out.join("observations.jsonl"))?;
    let mut terminals = File::create(j.out.join("outcomes.jsonl"))?;
    let mut slots = BTreeMap::<String, Slot>::new();
    let mut remaining = work.inputs.iter();
    let mut outcomes = Vec::new();
    let packet_samples = u64::from(j.packet_ms) * 16;
    let mut packet_count = 0;
    let mut event_count = 0;
    let mut preview_views = Vec::new();
    let mut batch_costs_ms = Vec::new();
    loop {
        while slots.len() < j.streams {
            let Some(input) = remaining.next() else {
                break;
            };
            let path = root.join(&input.path);
            require(
                hash_file(&path)? == input.sha256
                    && std::fs::metadata(&path)?.len() == input.samples * 2,
                "audio identity mismatch",
            )?;
            let now = Instant::now();
            slots.insert(
                input.id.clone(),
                Slot {
                    input: input.clone(),
                    file: File::open(path)?,
                    cursor: 0,
                    started: now,
                    source_start_s: now.duration_since(service_start).as_secs_f64(),
                    emitter: Emitter::new(&input.id, &plan.bundle_manifest_sha256),
                    last_hyp: String::new(),
                },
            );
        }
        if slots.is_empty() {
            break;
        }
        let due = |s: &Slot| {
            s.started
                + Duration::from_secs_f64(
                    (s.cursor + packet_samples).min(s.input.samples) as f64 / 16000.0,
                )
        };
        if j.paced {
            let earliest = slots.values().map(due).min().ok_or("empty schedule")?;
            if let Some(wait) = earliest.checked_duration_since(Instant::now()) {
                std::thread::sleep(wait);
            }
        }
        let release = Instant::now();
        let mut packets = Vec::new();
        for (id, s) in &mut slots {
            if j.paced && due(s) > release {
                continue;
            }
            let count = packet_samples.min(s.input.samples - s.cursor) as usize;
            let mut raw = vec![0u8; count * 2];
            s.file.seek(SeekFrom::Start(s.cursor * 2))?;
            s.file.read_exact(&mut raw)?;
            packets.push(Packet {
                id: id.clone(),
                offset: s.cursor,
                pcm: raw
                    .chunks_exact(2)
                    .map(|v| i16::from_le_bytes([v[0], v[1]]))
                    .collect(),
                start: s.cursor == 0,
                finish: s.cursor + count as u64 == s.input.samples,
            });
        }
        require(!packets.is_empty(), "scheduler made no progress")?;
        packet_count += packets.len();
        let batch_start = Instant::now();
        let rows = engine.step(&packets)?;
        let batch_ms = batch_start.elapsed().as_secs_f64() * 1000.0;
        batch_costs_ms.extend(engine.take_model_batch_ms());
        validate_batch(&packets, &rows)?;
        let now = Instant::now();
        let observed_s = now.duration_since(service_start).as_secs_f64();
        for row in rows {
            let s = slots.get_mut(&row.id).ok_or("lost stream")?;
            s.cursor = row.accepted_samples;
            let age = if j.paced {
                (now.duration_since(s.started).as_secs_f64()
                    - row.processed_samples as f64 / 16000.0)
                    .max(0.0)
            } else {
                0.0
            };
            json_line(
                &mut observations,
                &serde_json::json!({"run_id": plan.run_id, "worker_id": "w0",
                "observed_s": observed_s, "source_start_s": s.source_start_s,
                "backlog_age_upper_bound_s": (!j.guarded).then_some(age), "processing_watermark": "eof_drain_only",
                "service_batch_ms": (!j.guarded).then_some(batch_ms),
                "observation": row}),
            )?;
            let ns = now.duration_since(service_start).as_nanos().try_into()?;
            let event = if row.terminal {
                Some(s.emitter.finalize(
                    ns,
                    &row.hyp,
                    s.cursor,
                    ProcessingEvidence {
                        accepted_through_sample: s.cursor,
                        required_processed_through_sample: row.processed_samples,
                        flushed: row.drained,
                    },
                )?)
            } else if row.hyp != s.last_hyp {
                Some(s.emitter.preview(ns, &row.hyp, s.cursor)?)
            } else {
                None
            };
            if let Some(event) = event {
                event_count += 1;
                let view = serde_json::json!({"input_id": row.id, "text": row.hyp});
                preview_views.push((s.input.index, view));
                json_line(
                    &mut events,
                    &serde_json::json!({"run_id": plan.run_id, "worker_id": "w0",
                    "input_id": row.id, "source_start_s": s.source_start_s,
                    "timed": ultra_eval::stream::TimedEvent { t_s: observed_s, event }}),
                )?;
            }
            s.last_hyp.clone_from(&row.hyp);
            if row.terminal {
                let digest = row.pcm_sha256.ok_or("missing actual audio digest")?;
                require(digest == s.input.sha256, "worker consumed wrong audio")?;
                let out = Outcome {
                    run_id: plan.run_id.clone(),
                    worker_id: "w0".into(),
                    input_id: row.id.clone(),
                    source_index: s.input.index,
                    pcm_sha256: digest,
                    accepted_samples: s.cursor,
                    processed_samples: row.processed_samples,
                    final_text: row.hyp,
                    forced_advances: row.forced_advances,
                    source_start_s: s.source_start_s,
                    final_observed_s: observed_s,
                };
                outcomes.push(out);
                slots.remove(&row.id);
            }
        }
    }
    engine.shutdown()?;
    let wakeups = engine.wakeups();
    drop(engine);
    validate_outcomes(&plan.run_id, &work, &outcomes)?;
    outcomes.sort_by_key(|r| r.source_index);
    // Stable source order, preserving revisions within each stream. Raw timed
    // events above retain actual observation order and have integrity hashes.
    preview_views.sort_by_key(|r| r.0);
    for (_, view) in preview_views {
        json_line(&mut previews, &view)?;
    }
    for r in &outcomes {
        json_line(&mut terminals, r)?;
        let view = serde_json::json!({"input_id": r.input_id, "text": r.final_text});
        json_line(&mut commits, &view)?;
        json_line(&mut finals, &view)?;
    }
    batch_costs_ms.sort_by(f64::total_cmp);
    let q = |p| ultra_eval::delay::quantile_nearest_rank(&batch_costs_ms, p);
    let forced = outcomes
        .iter()
        .map(|r| r.forced_advances)
        .collect::<Option<Vec<_>>>()
        .map(|v| v.into_iter().sum::<u64>());
    write_json(
        &j.out.join("service.json"),
        &serde_json::json!({"schema": SCHEMA, "run_id": plan.run_id,
        "complete": true, "expected": work.inputs.len(), "completed": outcomes.len(),
        "accepted_samples": outcomes.iter().map(|r| r.accepted_samples).sum::<u64>(),
        "processed_samples": outcomes.iter().map(|r| r.processed_samples).sum::<u64>(),
        "unfinished_samples": 0, "packets": packet_count, "events": event_count, "wakeups": wakeups,
        "forced_advances": forced, "cold_tree_cpu_s": cold_tree_cpu_s,
        "cpu_snapshot_resolution_s": 1.0 / ultra_eval::procfs::clk_tck() as f64,
        "chunk_ms": {"p50": (!j.guarded).then(|| q(0.5)).flatten(),
            "p95": (!j.guarded).then(|| q(0.95)).flatten(), "p99": (!j.guarded).then(|| q(0.99)).flatten()},
        "chunk_timing_scope": "model batch wall time; R1 encoder plus greedy, R0 native decode call; variable batch occupancy; excludes IPC/no-op packets",
        "service_wall_s": service_start.elapsed().as_secs_f64(), "endpoint": "controlled_eof"}),
    )
}
