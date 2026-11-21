//! Fused service admission, processing truth and irreversible failures.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use ultra_bridge::engine::Engine;
use ultra_bridge::protocol::Packet;
use ultra_bridge::records::{hash_file, Job, Plan, SCHEMA};
use ultra_bridge::service::validate_batch;

fn plan(bundle: &Path) -> Plan {
    Plan {
        schema: SCHEMA.into(),
        run_id: "fused-fixture".into(),
        job: Job {
            inputs: PathBuf::new(),
            bundle: bundle.into(),
            engine: "r1-fused".into(),
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
            host_window: "fused contract fixture".into(),
            service_cpus: None,
            observer_cpus: None,
        },
        workload_sha256: "a".repeat(64),
        bundle_manifest_sha256: hash_file(&bundle.join("arch.json")).expect("arch hash"),
        executable_sha256: "a".repeat(64),
        worker_sha256: "a".repeat(64),
        consumer_sha256: "a".repeat(64),
        source_commit: "a".repeat(40),
        source_dirty: true,
        host: serde_json::Value::Null,
    }
}

static NEXT: AtomicU64 = AtomicU64::new(0);
struct Fixture(PathBuf);
impl Fixture {
    fn new() -> Self {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/bridge-tests")
            .join(format!(
                "fused-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
        std::fs::create_dir_all(&path).expect("fixture directory");
        let f = Self(path);
        let mut header = serde_json::to_vec(&serde_json::json!({
            "__metadata__":{"checkpoint_sha256":"a".repeat(64)},
            "cmvn.mean":{"dtype":"F32","shape":[80],"data_offsets":[0,320]},
            "cmvn.istd":{"dtype":"F32","shape":[80],"data_offsets":[320,640]}
        }))
        .expect("serialize fixture tensor header");
        while !header.len().is_multiple_of(8) {
            header.push(b' ');
        }
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header);
        for value in [0.0f32, 1.0] {
            for _ in 0..80 {
                bytes.extend(value.to_le_bytes());
            }
        }
        std::fs::write(f.0.join("weights.safetensors"), bytes).expect("fixture tensors");
        for name in ["tokens.txt", "bpe.model", "mel_fbank.f32"] {
            std::fs::write(f.0.join(name), b"fixture").expect("fixture payload");
        }
        let mut files = serde_json::Map::new();
        for (key, name) in [
            ("weights", "weights.safetensors"),
            ("tokens", "tokens.txt"),
            ("bpe", "bpe.model"),
            ("mel_fbank", "mel_fbank.f32"),
        ] {
            files.insert(
                key.into(),
                serde_json::json!({"path":name,"sha256":hash_file(&f.0.join(name)).expect("fixture payload hash")}),
            );
        }
        let arch = serde_json::json!({"schema":"ultra.r1-weights/v1", "files":files,
            "source":{"checkpoint_sha256":"a".repeat(64),"tokenizer_sha256":files["bpe"]["sha256"]},
            "features":{"cmvn":{"mean":vec![0.0;80],"istd":vec![1.0;80]}}});
        f.write_arch(&arch);
        f
    }
    fn write_arch(&self, v: &serde_json::Value) {
        std::fs::write(
            self.0.join("arch.json"),
            serde_json::to_vec(v).expect("serialize fixture arch"),
        )
        .expect("write fixture arch");
    }
    fn arch(&self) -> serde_json::Value {
        serde_json::from_slice(&std::fs::read(self.0.join("arch.json")).expect("read fixture arch"))
            .expect("parse fixture arch")
    }
    fn error(&self) -> String {
        Engine::load(&plan(&self.0))
            .err()
            .expect("invalid bundle")
            .to_string()
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).expect("remove fixture");
    }
}

#[test]
fn admission_checks_bpe_paths_cmvn_and_effective_settings() {
    let f = Fixture::new();
    std::fs::write(f.0.join("bpe.model"), b"tampered").unwrap();
    assert!(f.error().contains("payload hash mismatch: bpe"));
    std::fs::write(f.0.join("bpe.model"), b"fixture").unwrap();
    let original = f.arch();
    let mut altered = original.clone();
    altered["features"]["cmvn"]["istd"][0] = 2.0.into();
    f.write_arch(&altered);
    assert!(f.error().contains("CMVN istd differs"));
    altered = original.clone();
    altered["files"]["bpe"]["path"] = "../bpe.model".into();
    f.write_arch(&altered);
    assert!(f.error().contains("invalid fused payload path"));
    f.write_arch(&original);
    let mut p = plan(&f.0);
    p.job.precision = "int8".into();
    assert!(
        p.job.validate().is_ok(),
        "prepared INT8 is an admitted setting"
    );
    p.job.precision = "fp32".into();
    p.job.threads = 2;
    assert!(p.job.validate().is_err());
    p.job.threads = 1;
    assert_eq!(p.job.bundle_manifest_path(), f.0.join("arch.json"));
    p.bundle_manifest_sha256 = "0".repeat(64);
    assert!(Engine::load(&p)
        .err()
        .unwrap()
        .to_string()
        .contains("manifest identity"));
}

#[test]
fn learned_fused_bundle_batches_partial_chunks_and_has_sticky_failures() {
    let data = PathBuf::from(
        std::env::var("JSR_DATA").unwrap_or_else(|_| "/mnt/xbabe3-external/jsr".into()),
    );
    let bundle = data.join("runs/engine/weights-r1-d1-ckpt7800-v3");
    if !bundle.join("arch.json").is_file() {
        assert!(
            std::env::var_os("ULTRA_REQUIRE_NATIVE_TESTS").is_none(),
            "fused fixture missing"
        );
        eprintln!("SKIPPED learned fused fixture: retained engine bundle missing");
        return;
    }
    let (mut engine, info) = Engine::load(&plan(&bundle)).expect("load learned FP32 fused model");
    assert_eq!(info["cpu_only_verified"], true);
    assert_eq!(info["python_runtime_used"], false);
    assert_eq!(info["runtime_weight_quantization"], false);
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
        validate_batch(&first, &rows).unwrap();
        assert!(rows.iter().all(|r| r.processed_samples == 0 && !r.terminal));
        let last: Vec<_> = (0..batch)
            .map(|i| Packet {
                id: format!("b{batch}-{i}"),
                offset: 320,
                pcm: vec![0; 2000 + 640 * (1 + i % 4)],
                start: false,
                finish: true,
            })
            .collect();
        let rows = engine.step(&last).expect("mixed partial final chunks");
        validate_batch(&last, &rows).unwrap();
        assert!(rows
            .iter()
            .all(|r| r.drained && r.terminal && r.pcm_sha256.is_some()));
    }
    engine.shutdown().expect("all streams drained");
    let bad = Packet {
        id: "absent".into(),
        offset: 0,
        pcm: vec![0; 320],
        start: false,
        finish: false,
    };
    assert!(engine.step(&[bad]).is_err());
    assert!(engine.step(&[]).is_err());
    assert!(
        engine.shutdown().is_err(),
        "failed service cannot close successfully"
    );
}

#[test]
fn complete_fused_service_reproduces_eight_retained_learned_texts() {
    let data = PathBuf::from(
        std::env::var("JSR_DATA").unwrap_or_else(|_| "/mnt/xbabe3-external/jsr".into()),
    );
    let bundle = data.join("runs/engine/weights-r1-d1-ckpt7800-v3");
    let inputs = data.join("runs/bridge/m1-dev8/inputs.json");
    let expected_path = data.join("runs/bridge/m2-native-fp32-dev400/final.jsonl");
    if !bundle.join("arch.json").is_file() || !inputs.is_file() || !expected_path.is_file() {
        assert!(
            std::env::var_os("ULTRA_REQUIRE_NATIVE_TESTS").is_none(),
            "fused service fixture missing"
        );
        eprintln!("SKIPPED full fused service fixture: retained bundle/audio/output missing");
        return;
    }
    let out = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/bridge-tests")
        .join(format!(
            "fused-service-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
    std::fs::create_dir_all(&out).unwrap();
    let cleanup = Fixture(out);
    let mut p = plan(&bundle);
    p.workload_sha256 = hash_file(&inputs).unwrap();
    p.executable_sha256 = hash_file(&std::env::current_exe().unwrap()).unwrap();
    p.job.inputs = inputs;
    p.job.out = cleanup.0.clone();
    p.job.streams = 8;
    // Inference gets only bundle, opaque audio inputs and config. Expected output
    // is opened after the service has drained and closed its model.
    ultra_bridge::service::execute(&p).expect("complete audio-only fused service");
    let parse = |s: String| -> Vec<serde_json::Value> {
        s.lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect()
    };
    let actual = parse(std::fs::read_to_string(cleanup.0.join("final.jsonl")).unwrap());
    let expected = parse(std::fs::read_to_string(expected_path).unwrap());
    assert_eq!(actual.len(), 8);
    assert_eq!(
        actual,
        expected[..8],
        "fused service must preserve learned FP32 text"
    );
    let result: serde_json::Value =
        serde_json::from_slice(&std::fs::read(cleanup.0.join("service.json")).unwrap()).unwrap();
    assert_eq!(result["complete"], true);
    assert_eq!(result["unfinished_samples"], 0);
    assert_eq!(result["completed"], 8);
}

fn run_packets(
    engine: &mut Engine,
    audio: &[Vec<i16>],
    batch: usize,
    packet_samples: usize,
) -> Vec<String> {
    let mut texts = vec![String::new(); audio.len()];
    for (group, pcm) in audio.chunks(batch).enumerate() {
        let mut offsets = vec![0; pcm.len()];
        while offsets.iter().zip(pcm).any(|(a, p)| *a < p.len()) {
            let packets: Vec<_> = pcm
                .iter()
                .enumerate()
                .filter_map(|(i, samples)| {
                    let start = offsets[i];
                    if start == samples.len() {
                        return None;
                    }
                    let end = (start + packet_samples).min(samples.len());
                    offsets[i] = end;
                    Some(Packet {
                        id: (group * batch + i).to_string(),
                        offset: start as u64,
                        pcm: samples[start..end].to_vec(),
                        start: start == 0,
                        finish: end == samples.len(),
                    })
                })
                .collect();
            let rows = engine.step(&packets).expect("prepared INT8 packet step");
            validate_batch(&packets, &rows).expect("prepared observation contract");
            for row in rows {
                if row.terminal {
                    assert!(row.drained && row.pcm_sha256.is_some());
                    let index = row.id.parse::<usize>().expect("fixture instance ID");
                    texts[index] = row.hyp;
                } else {
                    assert_eq!(row.processed_samples, 0);
                }
            }
        }
    }
    texts
}

#[test]
fn prepared_int8_refuses_runtime_weight_quantization_and_is_batch_packet_invariant() {
    let data = PathBuf::from(
        std::env::var("JSR_DATA").unwrap_or_else(|_| "/mnt/xbabe3-external/jsr".into()),
    );
    let fp32 = data.join("runs/engine/weights-r1-d1-ckpt7800-v3");
    let prepared = data.join("runs/engine/weights-r1-d1-ckpt7800-v3-int8prep-723da8f");
    let cache = data.join("cache/librispeech_pcm/dev-clean.npz");
    if !fp32.join("arch.json").is_file()
        || !prepared.join("arch.json").is_file()
        || !cache.is_file()
    {
        assert!(
            std::env::var_os("ULTRA_REQUIRE_NATIVE_TESTS").is_none(),
            "prepared INT8 fixture missing"
        );
        eprintln!("SKIPPED prepared INT8 fixture: retained bundle/audio missing");
        return;
    }
    let mut unprepared = plan(&fp32);
    unprepared.job.precision = "int8".into();
    assert!(Engine::load(&unprepared)
        .err()
        .expect("unprepared service refused")
        .to_string()
        .contains("requires prepared"));

    let mut p = plan(&prepared);
    p.job.precision = "int8".into();
    let (mut engine, info) = Engine::load(&p).expect("strict prepared INT8 load");
    let arch: serde_json::Value =
        ultra_bridge::records::read_json(&prepared.join("arch.json")).expect("prepared manifest");
    assert_eq!(info["precision"], "int8");
    assert_eq!(info["int8_source"], "prepared");
    assert_eq!(info["runtime_weight_quantization"], false);
    assert_eq!(
        info["weights_int8_sha256"],
        arch["files"]["weights_int8"]["sha256"]
    );
    assert_eq!(info["int8_fp32_kinds"], arch["int8"]["fp32_kinds"]);
    let sources =
        ultra_frontend::npz::load_npz_audio_only(&cache, 0, Some(8)).expect("audio-only fixture");
    assert_eq!(sources.len(), 8);
    let audio: Vec<_> = (0..32)
        .map(|i| {
            let pcm = &sources[i % 8].pcm;
            pcm[..pcm.len().min(16_000 + i * 317)].to_vec()
        })
        .collect();
    let reference = run_packets(&mut engine, &audio, 1, 320);
    for batch in [3, 5, 16, 32] {
        for packet in [160, 592] {
            assert_eq!(
                run_packets(&mut engine, &audio, batch, packet),
                reference,
                "prepared INT8 drift: batch={batch}, packet_samples={packet}"
            );
        }
    }
    engine.shutdown().expect("all prepared streams drained");
}
