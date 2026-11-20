use ultra_bridge::protocol::Packet;
use ultra_bridge::records::{Input, Observation, Workload, SCHEMA};
use ultra_bridge::service::validate_batch;

fn packet() -> Packet {
    Packet {
        id: "u00000".into(),
        offset: 10,
        pcm: vec![0; 320],
        start: false,
        finish: true,
    }
}

fn observation() -> Observation {
    Observation {
        id: "u00000".into(),
        accepted_samples: 330,
        processed_samples: 330,
        terminal: true,
        drained: true,
        hyp: String::new(),
        forced_advances: Some(0),
        pcm_sha256: Some("0".repeat(64)),
    }
}

#[test]
fn empty_transcript_is_valid_only_with_complete_explicit_evidence() {
    validate_batch(&[packet()], &[observation()]).expect("explicit drained empty output");
    assert!(validate_batch(&[packet()], &[]).is_err());
    let mut r = observation();
    r.processed_samples = 10;
    assert!(validate_batch(&[packet()], &[r]).is_err());
    let mut r = observation();
    r.drained = false;
    assert!(validate_batch(&[packet()], &[r]).is_err());
    let mut r = observation();
    r.id = "stale".into();
    assert!(validate_batch(&[packet()], &[r]).is_err());
    assert!(validate_batch(&[packet()], &[observation(), observation()]).is_err());
}

#[test]
fn premature_terminal_impossible_watermark_and_missing_digest_fail() {
    let mut p = packet();
    p.finish = false;
    assert!(validate_batch(&[p], &[observation()]).is_err());
    let mut r = observation();
    r.processed_samples = 331;
    assert!(validate_batch(&[packet()], &[r]).is_err());
    let mut r = observation();
    r.pcm_sha256 = None;
    assert!(validate_batch(&[packet()], &[r]).is_err());
}

#[test]
fn workload_refuses_duplicate_sources_traversal_and_sealed_partition() {
    let input = Input {
        id: "u00000".into(),
        index: 0,
        path: "u0.s16le".into(),
        samples: 10,
        sha256: "0".repeat(64),
    };
    let mut w = Workload {
        schema: SCHEMA.into(),
        dataset: "dev-clean".into(),
        sample_rate: 16000,
        source_cache_sha256: "1".repeat(64),
        inputs: vec![input.clone()],
    };
    w.validate().expect("valid DEV workload");
    w.inputs.push(input);
    assert!(w.validate().is_err());
    w.inputs.pop();
    w.inputs[0].path = "../refs".into();
    assert!(w.validate().is_err());
    w.inputs[0].path = "audio.s16le".into();
    w.dataset = "test-clean".into();
    assert!(w.validate().is_err());
}
