//! Native inference must reject unbound payloads before loading a native library.
use std::path::PathBuf;
use ultra_bridge::engine::Engine;
use ultra_bridge::records::{hash_file, Job, Plan, SCHEMA};

struct Fixture(PathBuf);
impl Fixture {
    fn new() -> Self {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/bridge-tests")
            .join(format!("native-{}", std::process::id()));
        std::fs::create_dir_all(root.join("model")).expect("fixture directory");
        Self(root)
    }

    fn plan(&self, digest: &str) -> Plan {
        let name = "model/encoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx";
        std::fs::write(self.0.join(name), b"abcd").expect("payload");
        let manifest = serde_json::json!({
            "schema":"ultra.bundle/v1", "bundle_id":"r0-fixture", "candidate_id":"r0-fixture",
            "precision":"int8", "required_isa":[], "packed_weight_bytes":4,
            "expanded_weight_bytes":4, "frontend":{"sample_rate_hz":16000,
            "feature_dim":80,"kind":"logmel80"}, "limits":{"max_concurrent_streams":1,
            "max_service_rss_bytes":1000000,"supported_chunk_ms":[320]},
            "normalizer_id":"ultra.en-norm.v1", "license_code":"Apache-2.0",
            "license_weights":"Apache-2.0", "provenance_summary":"synthetic negative fixture",
            "entries":[{"path":name,"bytes":4,"sha256":digest}]
        });
        let manifest_path = self.0.join("manifest.json");
        std::fs::write(&manifest_path, manifest.to_string()).expect("manifest");
        Plan {
            schema: SCHEMA.into(),
            run_id: "native-negative".into(),
            job: Job {
                inputs: self.0.join("unused"),
                bundle: self.0.clone(),
                engine: "r0-native".into(),
                precision: "int8".into(),
                out: self.0.join("unused"),
                streams: 1,
                packet_ms: 20,
                threads: 1,
                paced: false,
                guarded: false,
                python: "unused".into(),
                worker: "unused".into(),
                measure: "unused".into(),
                timeout_s: 1,
                host_window: "negative fixture".into(),
                service_cpus: None,
                observer_cpus: None,
            },
            workload_sha256: "a".repeat(64),
            bundle_manifest_sha256: hash_file(&manifest_path).expect("manifest hash"),
            executable_sha256: "a".repeat(64),
            worker_sha256: "a".repeat(64),
            consumer_sha256: "a".repeat(64),
            source_commit: "a".repeat(40),
            source_dirty: false,
            host: serde_json::Value::Null,
        }
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).expect("remove fixture");
    }
}

#[test]
fn corrupt_and_incomplete_native_bundles_fail_before_native_loading() {
    let fixture = Fixture::new();
    let bad = fixture.plan(&"0".repeat(64));
    let err = Engine::load(&bad).err().expect("corrupt payload rejected");
    assert!(matches!(
        err.downcast_ref::<ultra_model::ModelError>(),
        Some(ultra_model::ModelError::HashMismatch { .. })
    ));

    let digest = ultra_core::receipt::sha256_hex(b"abcd");
    let incomplete = fixture.plan(&digest);
    let err = Engine::load(&incomplete)
        .err()
        .expect("unbound decoder rejected");
    assert!(err
        .to_string()
        .contains("native model absent from verified payloads"));

    let manifest_path = fixture.0.join("manifest.json");
    let mut manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&manifest_path).expect("manifest")).expect("JSON");
    manifest["required_isa"] = serde_json::json!(["unknown-instruction-set"]);
    std::fs::write(&manifest_path, manifest.to_string()).expect("unsupported ISA manifest");
    let mut unsupported = incomplete;
    unsupported.bundle_manifest_sha256 = hash_file(&manifest_path).expect("manifest hash");
    let err = Engine::load(&unsupported).err().expect("ISA rejected");
    assert!(err.to_string().contains("unsupported required native ISA"));
}
