//! Native R1 service admission, mixed partial chunks and irreversible failures.

use std::path::{Path, PathBuf};

use ultra_bridge::engine::Engine;
use ultra_bridge::protocol::Packet;
use ultra_bridge::records::{hash_file, Job, Plan, SCHEMA};
use ultra_bridge::service::validate_batch;

fn plan(bundle: &Path) -> Plan {
    Plan {
        schema: SCHEMA.into(),
        run_id: "native-r1-fixture".into(),
        job: Job {
            inputs: PathBuf::new(),
            bundle: bundle.into(),
            engine: "r1-native".into(),
            precision: "fp32".into(),
            out: PathBuf::new(),
            streams: 32,
            packet_ms: 20,
            threads: 1,
            paced: false,
            guarded: true,
            python: PathBuf::new(),
            worker: PathBuf::new(),
            measure: PathBuf::new(),
            timeout_s: 60,
            host_window: "native contract fixture".into(),
        },
        workload_sha256: "a".repeat(64),
        bundle_manifest_sha256: hash_file(&bundle.join("manifest.json")).expect("manifest hash"),
        executable_sha256: "a".repeat(64),
        worker_sha256: "a".repeat(64),
        consumer_sha256: "a".repeat(64),
        source_commit: "a".repeat(40),
        source_dirty: true,
        host: serde_json::Value::Null,
    }
}

#[test]
fn bad_native_r1_manifest_is_rejected_without_native_loading() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/bridge-tests")
        .join(format!("r1-admission-{}", std::process::id()));
    std::fs::create_dir_all(&root).expect("fixture directory");
    let manifest = serde_json::json!({
        "schema":"ultra.r1-export/v1", "model":{"family":"unsupported-fixture", "exporter_commit":"x"},
        "files":{}, "features":{"sample_rate":16000,"frame_length":400,"frame_shift":160,
        "n_fft":512,"n_mels":80,"log_floor":0.000001,"cmvn":{"mean":[],"istd":[]}},
        "streaming":{"chunk_frames":4,"chunk_ms":160,"left_context_frames":128,"states":[],
        "encoder_output":{"name":"enc","shape":["B","n",320]}},
        "decoder":{"blank_id":0,"vocab_size":4097,"ctx_init":[0,0],"max_symbols":8}
    });
    std::fs::write(root.join("manifest.json"), manifest.to_string()).expect("manifest");
    let mut job = plan(&root);
    job.job.validate().expect("native needs no Python paths");
    let err = Engine::load(&job)
        .err()
        .expect("unsupported family rejected");
    assert!(err.to_string().contains("unsupported-fixture"));
    job.bundle_manifest_sha256 = "0".repeat(64);
    let err = Engine::load(&job)
        .err()
        .expect("manifest identity rejected");
    assert!(err.to_string().contains("manifest identity changed"));
    job.job.engine = "r1-bridge".into();
    assert!(
        job.job.validate().is_err(),
        "Python bridge requires its worker"
    );
    std::fs::remove_dir_all(root).expect("remove fixture");
}

#[test]
fn native_r1_batches_partial_chunks_and_failure_cannot_be_retried() {
    let data = PathBuf::from(
        std::env::var("JSR_DATA").unwrap_or_else(|_| "/mnt/xbabe3-external/jsr".into()),
    );
    let bundle = data.join("bundles/r1-d1-reviewed-20261120-v2");
    if !bundle.join("manifest.json").is_file() {
        assert!(
            std::env::var_os("ULTRA_REQUIRE_NATIVE_TESTS").is_none(),
            "required R1 bundle missing"
        );
        eprintln!("SKIPPED native R1 batch fixture: retained bundle missing");
        return;
    }
    let (mut engine, info) = Engine::load(&plan(&bundle)).expect("native R1 load");
    assert_eq!(info["python_runtime_used"], false);
    assert_eq!(info["cpu_only_verified"], true);
    for batch in [1, 3, 5, 16, 32] {
        let first: Vec<_> = (0..batch)
            .map(|i| Packet {
                id: format!("b{batch}-{i}"),
                offset: 0,
                pcm: vec![0; 320],
                start: true,
                finish: false,
            })
            .collect();
        let rows = engine.step(&first).expect("subframe input");
        validate_batch(&first, &rows).expect("subframe observations");
        assert!(rows.iter().all(|r| r.processed_samples == 0 && !r.terminal));
        let middle: Vec<_> = (0..batch)
            .map(|i| Packet {
                id: format!("b{batch}-{i}"),
                offset: 320,
                pcm: vec![0; 2000],
                start: false,
                finish: false,
            })
            .collect();
        let before = engine.wakeups();
        let rows = engine.step(&middle).expect("full compatible batch");
        validate_batch(&middle, &rows).expect("full-batch observations");
        assert_eq!(
            engine.wakeups(),
            before + 1,
            "exact requested batch is decoded"
        );
        // Complete feature frames lead to 1, 2, 3 or 4 final encoder frames in one batch.
        let final_packets: Vec<_> = (0..batch)
            .map(|i| Packet {
                id: format!("b{batch}-{i}"),
                offset: 2320,
                pcm: vec![0; 640 + (i % 4) * 640],
                start: false,
                finish: true,
            })
            .collect();
        let rows = engine.step(&final_packets).expect("mixed final chunks");
        validate_batch(&final_packets, &rows).expect("complete observations");
        for (packet, row) in final_packets.iter().zip(&rows) {
            let raw: Vec<_> = vec![0_i16; 2320 + packet.pcm.len()]
                .into_iter()
                .flat_map(i16::to_le_bytes)
                .collect();
            assert_eq!(
                row.pcm_sha256.as_deref(),
                Some(ultra_core::receipt::sha256_hex(&raw).as_str())
            );
            assert!(row.forced_advances.is_some());
        }
        engine.shutdown().expect("all streams drained");
    }
    let bad = Packet {
        id: "absent".into(),
        offset: 0,
        pcm: vec![0; 320],
        start: false,
        finish: true,
    };
    assert!(engine.step(&[bad]).is_err());
    let good = Packet {
        id: "new".into(),
        offset: 0,
        pcm: vec![0; 320],
        start: true,
        finish: true,
    };
    let err = engine
        .step(&[good])
        .expect_err("failed service cannot recover into Final");
    assert!(err.to_string().contains("already failed"));
    assert!(engine.shutdown().is_err());
}
