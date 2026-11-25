//! Integration tests for Z1 product-caller CLI wiring (`ultra stream --commit-pass --commit-rms-norm`).

use std::path::Path;

use serde_json::{json, Map, Value};
use ultra_cli::stream_commit::StreamCommitCliOptions;
use ultra_cli::{run_cli_args, usage_text};
use ultra_core::receipt::{sha256_hex, Receipt};
use ultra_engine::commit::CommitConfig;
use ultra_engine::model::Model;
use ultra_engine::Precision;
use ultra_frontend::{rms_normalize, write_wav_pcm16};
use ultra_stream::{
    AsrStream, LiveCommitAsrStream, LiveCommitConfig, NativeEngineBackend, NativeEngineLoadConfig,
    OperatingProfile, OutputPolicy, PollResult, StreamClock, StreamConfig, StreamCoordinator,
};

const VOCAB4: &str = "<blk> 0\n\u{2581}the 1\n\u{2581}cat 2\n\u{2581}sat 3\n";

fn write_staged_r1_model(dir: &Path) {
    let fp32_dir = dir.with_file_name(format!(
        "{}_fp32",
        dir.file_name().expect("file_name").to_string_lossy()
    ));
    std::fs::create_dir_all(&fp32_dir).expect("mkdir");
    let (d, ch, ffn, k, vocab, pred_emb, joiner, f2) = (4, 4, 8, 3, 4, 4, 4, 19);
    let ckpt_sha = "7".repeat(64);
    let ones = |n: usize| vec![1.0f32; n];
    let zeros = |n: usize| vec![0.0f32; n];
    let mut stem_out_w = zeros(d * ch * f2);
    for v in &mut stem_out_w[..ch * f2] {
        *v = 0.25;
    }
    let mut enc_proj_w = zeros(joiner * d);
    enc_proj_w[0] = 2.0;
    let mut joiner_out_w = zeros(vocab * joiner);
    joiner_out_w[joiner] = -4.0; // token 1 ("the") preferred when dim 0 < 0 (quiet)
    joiner_out_w[2 * joiner] = 4.0; // token 2 ("cat") preferred when dim 0 > 0 (RMS-normalised)
    let tensors: Vec<(&str, Vec<usize>, Vec<f32>)> = vec![
        ("cmvn.mean", vec![80], vec![-6.0; 80]),
        ("cmvn.istd", vec![80], ones(80)),
        (
            "encoder.stem.c1.weight",
            vec![ch, 1, 3, 3],
            vec![0.25; ch * 9],
        ),
        ("encoder.stem.c1.bias", vec![ch], zeros(ch)),
        (
            "encoder.stem.c2.weight",
            vec![ch, ch, 3, 3],
            vec![0.25; ch * ch * 9],
        ),
        ("encoder.stem.c2.bias", vec![ch], zeros(ch)),
        ("encoder.stem.out.weight", vec![d, ch * f2], stem_out_w),
        ("encoder.stem.out.bias", vec![d], vec![0.0, 1.5, 0.0, 0.0]),
        ("encoder.blocks.0.ff1.norm.weight", vec![d], ones(d)),
        ("encoder.blocks.0.ff1.norm.bias", vec![d], zeros(d)),
        (
            "encoder.blocks.0.ff1.w1.weight",
            vec![ffn, d],
            zeros(ffn * d),
        ),
        ("encoder.blocks.0.ff1.w1.bias", vec![ffn], zeros(ffn)),
        (
            "encoder.blocks.0.ff1.w2.weight",
            vec![d, ffn],
            zeros(d * ffn),
        ),
        ("encoder.blocks.0.ff1.w2.bias", vec![d], zeros(d)),
        ("encoder.blocks.0.att.norm.weight", vec![d], ones(d)),
        ("encoder.blocks.0.att.norm.bias", vec![d], zeros(d)),
        (
            "encoder.blocks.0.att.qkv.weight",
            vec![3 * d, d],
            zeros(3 * d * d),
        ),
        ("encoder.blocks.0.att.qkv.bias", vec![3 * d], zeros(3 * d)),
        ("encoder.blocks.0.att.out.weight", vec![d, d], zeros(d * d)),
        ("encoder.blocks.0.att.out.bias", vec![d], zeros(d)),
        ("encoder.blocks.0.conv.norm.weight", vec![d], ones(d)),
        ("encoder.blocks.0.conv.norm.bias", vec![d], zeros(d)),
        (
            "encoder.blocks.0.conv.pw1.weight",
            vec![2 * d, d],
            zeros(2 * d * d),
        ),
        ("encoder.blocks.0.conv.pw1.bias", vec![2 * d], zeros(2 * d)),
        (
            "encoder.blocks.0.conv.dw.weight",
            vec![d, 1, k],
            zeros(d * k),
        ),
        ("encoder.blocks.0.conv.dw.bias", vec![d], zeros(d)),
        ("encoder.blocks.0.conv.dnorm.weight", vec![d], ones(d)),
        ("encoder.blocks.0.conv.dnorm.bias", vec![d], zeros(d)),
        ("encoder.blocks.0.conv.pw2.weight", vec![d, d], zeros(d * d)),
        ("encoder.blocks.0.conv.pw2.bias", vec![d], zeros(d)),
        ("encoder.blocks.0.ff2.norm.weight", vec![d], ones(d)),
        ("encoder.blocks.0.ff2.norm.bias", vec![d], zeros(d)),
        (
            "encoder.blocks.0.ff2.w1.weight",
            vec![ffn, d],
            zeros(ffn * d),
        ),
        ("encoder.blocks.0.ff2.w1.bias", vec![ffn], zeros(ffn)),
        (
            "encoder.blocks.0.ff2.w2.weight",
            vec![d, ffn],
            zeros(d * ffn),
        ),
        ("encoder.blocks.0.ff2.w2.bias", vec![d], zeros(d)),
        ("encoder.final_norm.weight", vec![d], ones(d)),
        ("encoder.final_norm.bias", vec![d], zeros(d)),
        ("engine.rope_inv_freq", vec![d / 2], vec![1.0, 0.01]),
        ("joiner.enc_proj.weight", vec![joiner, d], enc_proj_w),
        ("joiner.enc_proj.bias", vec![joiner], zeros(joiner)),
        (
            "predictor.emb.weight",
            vec![vocab, pred_emb],
            zeros(vocab * pred_emb),
        ),
        (
            "predictor.proj.weight",
            vec![joiner, 2 * pred_emb],
            zeros(joiner * 2 * pred_emb),
        ),
        ("predictor.proj.bias", vec![joiner], zeros(joiner)),
        ("joiner.out.weight", vec![vocab, joiner], joiner_out_w),
        ("joiner.out.bias", vec![vocab], vec![-3.0, 0.0, 0.0, -4.0]),
    ];
    let mut hdr = Map::new();
    hdr.insert(
        "__metadata__".into(),
        json!({"checkpoint_sha256": ckpt_sha}),
    );
    let mut payload = Vec::new();
    for (name, shape, data) in &tensors {
        let start = payload.len();
        for &v in data {
            payload.extend_from_slice(&v.to_le_bytes());
        }
        hdr.insert(
            (*name).into(),
            json!({"dtype": "F32", "shape": shape, "data_offsets": [start, payload.len()]}),
        );
    }
    let hdr_bytes = serde_json::to_vec(&Value::Object(hdr)).expect("json");
    let mut st_bytes = Vec::with_capacity(8 + hdr_bytes.len() + payload.len());
    st_bytes.extend_from_slice(&(hdr_bytes.len() as u64).to_le_bytes());
    st_bytes.extend_from_slice(&hdr_bytes);
    st_bytes.extend_from_slice(&payload);

    let tokens_bytes = VOCAB4.as_bytes();
    let fbank_bytes: Vec<u8> = (0..257 * 80).flat_map(|_| 0.01f32.to_le_bytes()).collect();
    let bpe_bytes = b"cli-z1-bpe-v1";
    std::fs::write(fp32_dir.join("weights.safetensors"), &st_bytes).expect("weights");
    std::fs::write(fp32_dir.join("tokens.txt"), tokens_bytes).expect("tokens");
    std::fs::write(fp32_dir.join("mel_fbank.bin"), &fbank_bytes).expect("fbank");
    std::fs::write(fp32_dir.join("bpe.model"), bpe_bytes).expect("bpe");
    let arch = json!({
        "schema": "ultra.r1-weights/v1",
        "model": {
            "family": "dense",
            "config": { "vocab_rnnt": 4, "pred_emb": 4, "joiner": 4, "joiner_act": "tanh" },
            "encoder": {
                "n_mels": 80, "d_model": 4, "n_blocks": 1, "n_heads": 1, "ffn_mult": 2,
                "conv_kernel": 3, "stem_channels": 4, "chunk": 4, "left_context": 4,
                "post_block_norm": false, "family": "dense",
                "lower_blocks": 1, "middle_blocks": 0, "final_blocks": 0
            }
        },
        "source": { "checkpoint_sha256": ckpt_sha },
        "files": {
            "weights": { "path": "weights.safetensors", "sha256": sha256_hex(&st_bytes) },
            "tokens": { "path": "tokens.txt", "sha256": sha256_hex(tokens_bytes) },
            "mel_fbank": { "path": "mel_fbank.bin", "sha256": sha256_hex(&fbank_bytes) },
            "bpe": { "path": "bpe.model", "sha256": sha256_hex(bpe_bytes) }
        },
        "features": {
            "sample_rate": 16000, "frame_length": 400, "frame_shift": 160,
            "n_fft": 512, "n_mels": 80, "log_floor": 1e-6,
            "cmvn": { "mean": vec![-6.0f32; 80], "istd": vec![1.0f32; 80] }
        },
        "decoder": {
            "kind": "stateless", "context": 2, "vocab_size": 4,
            "joiner_activation": "tanh", "blank_id": 0, "max_symbols": 1, "ctx_init": [0, 0]
        }
    });
    std::fs::write(fp32_dir.join("arch.json"), arch.to_string()).expect("arch.json");
    let mut model = Model::load(&fp32_dir, true).expect("load fp32 model");
    model.quantize_int8();
    model
        .write_prepared_int8(dir, &[])
        .expect("write prepared int8");
}

fn parse_commit_opts(flags: &[&str]) -> StreamCommitCliOptions {
    let args: Vec<String> = flags.iter().map(|s| (*s).to_owned()).collect();
    let mut opts = StreamCommitCliOptions::default();
    let mut i = 0;
    while i < args.len() {
        assert!(opts.try_parse_flag(&args, &mut i).expect("parse flag"));
        i += 1;
    }
    opts
}

fn status_prefixed<'a>(receipt: &'a Receipt, prefix: &str) -> Option<&'a str> {
    receipt
        .status_events
        .iter()
        .find_map(|s| s.strip_prefix(prefix))
}

fn run_live_commit_stream(
    backend: &std::sync::Arc<NativeEngineBackend>,
    live_cfg: &LiveCommitConfig,
    pcm: &[i16],
) -> String {
    let scfg = StreamConfig {
        stream_id: "z1-cli-test".into(),
        model_config_id: backend.bundle_id(),
        policy: OutputPolicy::EditableDictation,
        profile: OperatingProfile::Live,
        segment_hard_cap_samples: u64::MAX,
        endpoint_hangover_ms: u64::from(live_cfg.hangover_ms),
        ..StreamConfig::default()
    };
    let clk = StreamClock::real();
    let inner = backend
        .create_live_commit_stream_with_clock(&scfg, live_cfg, clk.clone())
        .expect("create_live_commit_stream_with_clock");
    let mut coord = StreamCoordinator::with_clock(scfg, inner, clk);
    for (k, pkt) in pcm.chunks(320).enumerate() {
        coord.push_pcm((k * 320) as u64, pkt).expect("push");
        while let PollResult::Event(_) = coord.poll_event().expect("poll") {}
    }
    coord.finish_input().expect("finish");
    while let PollResult::Event(_) = coord.poll_event().expect("poll") {}
    coord.ledger().committed_text()
}

#[test]
fn stream_cli_commit_rms_norm_default_off_and_opt_in_propagates_to_policy_and_session() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let model_dir = tmp.path().join("r1_model");
    write_staged_r1_model(&model_dir);

    let quiet: Vec<i16> = (0..8_000)
        .map(|i| (32767.0 * 0.003 * (i as f64 * 0.17).sin()) as i16)
        .collect();
    let (norm_pcm, st) = rms_normalize(&quiet);
    assert!(st.gain > 10.0, "quiet clip must require substantial gain");

    let wav_path = tmp.path().join("quiet.wav");
    write_wav_pcm16(&wav_path, 16_000, &quiet).expect("wav");

    let opts_off = parse_commit_opts(&["--commit-pass"]);
    let opts_on = parse_commit_opts(&["--commit-pass", "--commit-rms-norm"]);
    assert!(!opts_off.commit_rms_norm);
    assert!(opts_on.commit_rms_norm);

    let live_off = opts_off.to_live_commit_config().expect("live_off");
    let live_on = opts_on.to_live_commit_config().expect("live_on");
    assert_eq!(live_off.commit.rms_norm, CommitConfig::default().rms_norm);
    assert!(!live_off.commit.rms_norm);
    assert!(live_on.commit.rms_norm);

    let backend = NativeEngineBackend::load_with_config(
        &model_dir,
        Precision::Int8,
        1,
        true,
        &NativeEngineLoadConfig::default(),
    )
    .expect("load backend");
    let scfg = StreamConfig {
        stream_id: "cli-stream-0".into(),
        model_config_id: backend.bundle_id(),
        policy: OutputPolicy::EditableDictation,
        profile: OperatingProfile::Live,
        segment_hard_cap_samples: u64::MAX,
        endpoint_hangover_ms: u64::from(live_off.hangover_ms),
        ..StreamConfig::default()
    };
    let s_off = LiveCommitAsrStream::new(&backend, &scfg, &live_off, StreamClock::custom(|| 1))
        .expect("stream off");
    let s_on = LiveCommitAsrStream::new(&backend, &scfg, &live_on, StreamClock::custom(|| 1))
        .expect("stream on");

    assert!(!s_off.commit_decode_config().commit_rms_norm);
    assert!(
        !s_off
            .commit_decode_config()
            .canonical_json()
            .contains("commit_rms_norm"),
        "default off must omit commit_rms_norm so policy_sha256 matches v3.1.0"
    );
    assert!(s_on.commit_decode_config().commit_rms_norm);
    assert!(!s_on.provisional_decode_config().commit_rms_norm);

    let exp_prov_sha = s_off.provisional_decode_config().policy_sha256();
    let exp_commit_off_sha = s_off.commit_decode_config().policy_sha256();
    let exp_commit_on_sha = s_on.commit_decode_config().policy_sha256();
    assert_eq!(
        s_on.provisional_decode_config().policy_sha256(),
        exp_prov_sha
    );
    assert_ne!(exp_commit_on_sha, exp_commit_off_sha);

    // Propagates cleanly through NativeEngineBackend::create_live_commit_stream_with_clock:
    // quiet audio changes committed text when rms_norm is on and matches pre-normalised input.
    let committed_off = run_live_commit_stream(&backend, &live_off, &quiet);
    let committed_on = run_live_commit_stream(&backend, &live_on, &quiet);
    let committed_prenorm = run_live_commit_stream(&backend, &live_off, &norm_pcm);
    assert!(!committed_on.is_empty());
    assert_ne!(
        committed_on, committed_off,
        "enabling rms_norm must actively transform quiet audio in the commit worker"
    );
    assert_eq!(committed_on, committed_prenorm);

    // End-to-end `ultra stream --commit-pass [--commit-rms-norm]` receipts.
    let rec_off_path = tmp.path().join("receipt_off.json");
    let rec_on_path = tmp.path().join("receipt_on.json");
    let model_s = model_dir.to_string_lossy().into_owned();
    let wav_s = wav_path.to_string_lossy().into_owned();
    run_cli_args(&[
        "stream".into(),
        "--candidate".into(),
        model_s.clone(),
        "--input".into(),
        wav_s.clone(),
        "--pace".into(),
        "0".into(),
        "--commit-pass".into(),
        "--receipt".into(),
        rec_off_path.to_string_lossy().into_owned(),
    ])
    .expect("cli commit-pass default");
    run_cli_args(&[
        "stream".into(),
        "--candidate".into(),
        model_s,
        "--input".into(),
        wav_s,
        "--pace".into(),
        "0".into(),
        "--commit-pass".into(),
        "--commit-rms-norm".into(),
        "--receipt".into(),
        rec_on_path.to_string_lossy().into_owned(),
    ])
    .expect("cli commit-pass rms-norm");

    let rec_off: Receipt =
        serde_json::from_slice(&std::fs::read(&rec_off_path).expect("read off")).expect("json off");
    let rec_on: Receipt =
        serde_json::from_slice(&std::fs::read(&rec_on_path).expect("read on")).expect("json on");
    assert_eq!(
        status_prefixed(&rec_off, "decode_policy_sha256:"),
        Some(exp_prov_sha.as_str())
    );
    assert_eq!(
        status_prefixed(&rec_on, "decode_policy_sha256:"),
        Some(exp_prov_sha.as_str())
    );
    assert_eq!(
        status_prefixed(&rec_off, "commit_decode_policy_sha256:"),
        Some(exp_commit_off_sha.as_str())
    );
    assert_eq!(
        status_prefixed(&rec_on, "commit_decode_policy_sha256:"),
        Some(exp_commit_on_sha.as_str())
    );
    assert_ne!(
        rec_on.hashes.commit, rec_off.hashes.commit,
        "CLI --commit-rms-norm must actively change committed output on quiet audio"
    );
}

#[test]
fn stream_cli_commit_rms_norm_without_commit_pass_follows_existing_commit_flag_pattern() {
    assert!(usage_text().contains("--commit-rms-norm"));

    let tmp = tempfile::tempdir().expect("tempdir");
    let model_dir = tmp.path().join("r1_model");
    write_staged_r1_model(&model_dir);

    let wav_path = tmp.path().join("quiet.wav");
    let quiet: Vec<i16> = (0..6_400)
        .map(|i| (32767.0 * 0.003 * (i as f64 * 0.17).sin()) as i16)
        .collect();
    write_wav_pcm16(&wav_path, 16_000, &quiet).expect("wav");

    let model_s = model_dir.to_string_lossy().into_owned();
    let wav_s = wav_path.to_string_lossy().into_owned();
    let run_with = |extra: &[&str], name: &str| -> Receipt {
        let rec_path = tmp.path().join(name);
        let mut args = vec![
            "stream".to_owned(),
            "--candidate".to_owned(),
            model_s.clone(),
            "--input".to_owned(),
            wav_s.clone(),
            "--pace".to_owned(),
            "0".to_owned(),
            "--receipt".to_owned(),
            rec_path.to_string_lossy().into_owned(),
        ];
        args.extend(extra.iter().map(|s| (*s).to_owned()));
        run_cli_args(&args).expect(name);
        serde_json::from_slice(&std::fs::read(&rec_path).expect("read receipt")).expect("json")
    };

    let base = run_with(&[], "base.json");
    let other_commit_flags = run_with(&["--commit-beam", "4", "--commit-chunk", "8"], "other.json");
    let rms_only = run_with(&["--commit-rms-norm"], "rms_only.json");

    for r in [&base, &other_commit_flags, &rms_only] {
        assert!(status_prefixed(r, "commit_decode_policy_sha256:").is_none());
        assert_eq!(
            status_prefixed(r, "decode_policy_sha256:"),
            status_prefixed(&base, "decode_policy_sha256:")
        );
        assert_eq!(r.hashes.commit, base.hashes.commit);
        assert_eq!(r.hashes.final_, base.hashes.final_);
    }
}
