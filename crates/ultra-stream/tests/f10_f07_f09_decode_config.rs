//! Integration tests for `F10` (`EffectiveDecodeConfig` + `policy_sha256` + identical LM output
//! across `Engine`, `NativeEngineBackend`, `NativeEngineSlotBank`, and sync/async commit paths),
//! `F07` (`LiveCommitAsrStream` equal-length beam revision), and `F09` (honest commit CPU vs wall
//! and cancelled/stale worker accounting).

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use ultra_core::events::{Body, TranscriptEvent};
use ultra_core::receipt::{HonestyLabel, Partition};
use ultra_engine::commit::{CommitConfig, CommitPassSession};
use ultra_engine::model::{Arch, Block, Linear, Model, Norm};
use ultra_engine::{Engine, Isa, LmConfig, NgramLm, Precision, Workspace};
use ultra_frontend::npz::AudioUtterance;
use ultra_frontend::{GlobalCmvn, TrainerMelConfig};
use ultra_gp::TokenTable;
use ultra_stream::{
    run_engine_packet_streaming_with_tokens, AsrStream, DecodeMode, EngineSpAConfig,
    LiveCommitConfig, NativeEngineBackend, NativeEngineLoadConfig, PollResult, StreamClock,
    StreamConfig, StreamCoordinator, DEFAULT_LATENESS_BOUND_MS,
};

const VOCAB4: &str = "<blk> 0\n\u{2581}the 1\n\u{2581}cat 2\n\u{2581}sat 3\n";

const ARPA_PREFER_CAT: &str = "\
\\data\\
ngram 1=4
ngram 2=3

\\1-grams:
-4.5\t<unk>
-99.0\t<s>\t-0.1
-3.0\t1\t-0.1
-0.05\t2\t-0.05

\\2-grams:
-0.05\t<s> 2
-0.05\t2 2
-3.0\t<s> 1

\\end\\
";

const ARPA_LODR_FLAT: &str = "\
\\data\\
ngram 1=4
ngram 2=2

\\1-grams:
-4.0\t<unk>
-99.0\t<s>\t-0.1
-1.5\t1
-1.5\t2

\\2-grams:
-1.5\t<s> 1
-1.5\t<s> 2

\\end\\
";

fn synthetic_near_tie_model(bias1: f32, bias2: f32) -> Arc<Model> {
    let (d, h, dh, ffn, k, sc, f1, f2) = (8, 2, 4, 16, 3, 4, 39, 19);
    let (vocab, pe, jn) = (4, 4, 8);
    let z = |n, m| Linear::from_fp32(&vec![0.0; n * m], n, m, None);
    let nm = |n| Norm {
        g: vec![1.0; n],
        b: vec![0.0; n],
    };
    let blk = Block {
        ff1_norm: nm(d),
        ff1_w1: z(ffn, d),
        ff1_w2: z(d, ffn),
        att_norm: nm(d),
        qkv: z(3 * d, d),
        att_out: z(d, d),
        conv_norm: nm(d),
        pw1: z(2 * d, d),
        dw_w: vec![0.0; k * d],
        dw_b: vec![0.0; d],
        dnorm: nm(d),
        pw2: z(d, d),
        ff2_norm: nm(d),
        ff2_w1: z(ffn, d),
        ff2_w2: z(d, ffn),
        post_norm: None,
    };
    let jb = vec![-2.0, bias1, bias2, -4.0];
    let tok = TokenTable::from_str_checked(VOCAB4, vocab).expect("tok");
    let arch = Arch {
        n_mels: 80,
        d,
        n_blocks: 1,
        lower: 1,
        mid: 0,
        fin: 0,
        heads: h,
        dh,
        ffn,
        kernel: k,
        stem_ch: sc,
        f1,
        f2,
        chunk: 4,
        left: 4,
        post_block_norm: false,
        vocab,
        pred_emb: pe,
        joiner: jn,
        blank: 0,
        max_symbols: 1,
        ctx_init: vec![0, 0],
        log_floor: 1e-6,
        checkpoint_sha256: "1".repeat(64),
        weights_sha256: "2".repeat(64),
        qat_stem_int8: false,
    };
    Arc::new(Model {
        arch,
        root: PathBuf::new(),
        c1_w: vec![0.0; 9 * sc],
        c1_b: vec![0.0; sc],
        c2: z(sc, 9 * sc),
        stem_out: z(d, f2 * sc),
        blocks: vec![blk],
        final_norm: Some(nm(d)),
        pair: None,
        restore: None,
        enc_proj: z(jn, d),
        emb: vec![0.0; vocab * pe],
        pred_proj: z(jn, 2 * pe),
        pred_cache: None,
        joiner_out: Linear::from_fp32(&vec![0.0; vocab * jn], vocab, jn, Some(&jb)),
        blank_cert_f32: None,
        blank_cert_i8: None,
        rope_inv: vec![1.0; dh / 2],
        mel_cfg: Arc::new(TrainerMelConfig::default_htk(GlobalCmvn::identity()).expect("mel")),
        tokens: Arc::new(tok),
        ctc_head: None,
        assets: Default::default(),
        int8_prepared: None,
    })
}

fn drain(c: &mut StreamCoordinator) -> Vec<TranscriptEvent> {
    let mut out = Vec::new();
    while let PollResult::Event(ev) = c.poll_event().expect("poll") {
        out.push(*ev);
    }
    out
}

#[test]
fn f10_lm_activation_and_effective_decode_config_match_across_engine_native_bank_and_commit() {
    let model = synthetic_near_tie_model(2.0, 1.5);
    let lm_mem =
        NgramLm::from_arpa_str(ARPA_PREFER_CAT, Some(ARPA_LODR_FLAT), 4, false).expect("arpa");
    let lm_path =
        std::env::temp_dir().join(format!("ultra_f10_fixture_{}.ulm", std::process::id()));
    lm_mem.save(&lm_path).expect("save ulm");
    let lm = Arc::new(NgramLm::load(&lm_path).expect("load ulm"));
    let lm_cfg = LmConfig {
        lm_weight: 0.8,
        lodr_weight: 0.6,
        blank_penalty: 0.0,
    };
    let pcm = vec![0i16; 6_400];

    // Control without LM: decodes "the..."
    let eng_no_lm = Engine::new(model.clone(), Isa::Scalar, Precision::F32)
        .expect("eng_no_lm")
        .with_beam(1);
    let no_lm_text = {
        let mut s = eng_no_lm.create_stream();
        let mut ws = Workspace::default();
        for pkt in pcm.chunks(320) {
            s.accept_i16(pkt).expect("accept");
            while eng_no_lm.decode(&mut [&mut s], &mut ws).expect("dec") > 0 {}
        }
        s.input_finished();
        while eng_no_lm.decode(&mut [&mut s], &mut ws).expect("dec") > 0 {}
        eng_no_lm.text(&s)
    };
    assert!(
        no_lm_text.split_whitespace().all(|w| w == "the"),
        "expected acoustic winner 'the', got `{no_lm_text}`"
    );

    // Path 1: Direct `Engine` with LM (width-1 fused greedy LM, T9).
    let engine = Engine::new(model, Isa::Scalar, Precision::F32)
        .expect("eng")
        .with_beam(1)
        .with_beam_max_symbols(1)
        .with_lm(lm.clone(), lm_cfg)
        .expect("with_lm");
    let expected_cfg = engine.effective_decode_config();
    let expected_sha = expected_cfg.policy_sha256();
    assert_eq!(expected_cfg.beam_width, 1);
    assert_eq!(expected_cfg.beam_max_symbols, 1);
    assert_eq!(
        expected_cfg.lm_path.as_deref(),
        Some(lm_path.to_string_lossy().as_ref())
    );
    assert_eq!(expected_cfg.lm_sha256.as_deref().map(str::len), Some(64));
    assert_eq!(expected_cfg.lodr_sha256.as_deref().map(str::len), Some(64));
    assert!((expected_cfg.lm_weight - 0.8).abs() < 1e-6);
    assert!((expected_cfg.lodr_weight - 0.6).abs() < 1e-6);
    assert_ne!(
        expected_sha,
        eng_no_lm.effective_decode_config().policy_sha256()
    );

    let direct_text = {
        let mut s = engine.create_stream();
        let mut ws = Workspace::default();
        for pkt in pcm.chunks(320) {
            s.accept_i16(pkt).expect("accept");
            while engine.decode(&mut [&mut s], &mut ws).expect("dec") > 0 {}
        }
        s.input_finished();
        while engine.decode(&mut [&mut s], &mut ws).expect("dec") > 0 {}
        engine.text(&s)
    };
    assert_ne!(
        direct_text, no_lm_text,
        "activating LM must strictly change top decoded hypothesis"
    );
    assert!(
        direct_text.split_whitespace().all(|w| w == "cat"),
        "LM must steer output to 'cat', got `{direct_text}`"
    );

    // Path 2: `NativeEngineBackend` + `StreamCoordinator`.
    let backend = NativeEngineBackend::from_engine(engine.clone(), "0".repeat(64), "n/a");
    assert_eq!(backend.effective_decode_config(), expected_cfg);
    assert_eq!(
        backend.effective_decode_config().policy_sha256(),
        expected_sha
    );

    let scfg = StreamConfig::default();
    let mut coord = StreamCoordinator::new(
        scfg.clone(),
        backend.create_stream_with_clock(&scfg, StreamClock::custom(|| 1)),
    );
    for (k, pkt) in pcm.chunks(320).enumerate() {
        coord.push_pcm((k * 320) as u64, pkt).expect("push");
        let _ = drain(&mut coord);
    }
    coord.finish_input().expect("finish");
    let _ = drain(&mut coord);
    let native_text = coord.ledger().committed_text();
    assert_eq!(native_text, direct_text);

    // Path 3: `NativeEngineSlotBank` / packet-streaming runner + `EngineSpAConfig` config retention.
    let utts = vec![AudioUtterance {
        index: 0,
        pcm: pcm.clone(),
    }];
    let bank_run =
        run_engine_packet_streaming_with_tokens(&backend, &utts, 1, 20).expect("bank run");
    assert_eq!(bank_run.hyps[0], direct_text);

    let spa_cfg = EngineSpAConfig {
        weights_dir: PathBuf::new(),
        precision: Precision::F32,
        threads: 1,
        npz_path: PathBuf::new(),
        split_name: "synth".into(),
        partition: Partition::Dev,
        honesty_label: HonestyLabel::InCorpus,
        offset: 0,
        limit: Some(1),
        selected_indices: None,
        duration_s: None,
        streams: 1,
        decode: DecodeMode::Single,
        max_batch: 1,
        max_chunks: 1,
        beam: 1,
        decode_config: NativeEngineLoadConfig {
            beam: 1,
            beam_max_symbols: 1,
            lm: Some((lm.clone(), lm_cfg)),
            ..NativeEngineLoadConfig::default()
        },
        packet_ms: 20,
        grid_ms: 320,
        coalesce_ms: 0,
        lateness_bound_ms: DEFAULT_LATENESS_BOUND_MS,
        paced: false,
        source_commit: "test".into(),
        hardware_id: "test".into(),
        completion_status: "exploratory".into(),
    };
    let resolved = spa_cfg.resolved_load_config();
    assert!(
        resolved.lm.is_some(),
        "EngineSpAConfig must preserve compact LM"
    );
    assert_eq!(resolved.beam, 1);

    // Path 4a: Sync `CommitPassSession` with matching chunk=4, tail_pad=0.
    let ccfg = CommitConfig {
        commit_chunk: 4,
        tail_pad_samples: 0,
        beam_size: 1,
        ..CommitConfig::default()
    };
    let mut sync_sess = CommitPassSession::new(engine.clone(), ccfg).expect("sync sess");
    assert_eq!(
        sync_sess.provisional_engine().effective_decode_config(),
        expected_cfg
    );
    assert_eq!(
        sync_sess
            .provisional_engine()
            .effective_decode_config()
            .policy_sha256(),
        expected_sha
    );
    assert!(
        sync_sess.commit_engine().lm.is_some(),
        "commit engine must inherit LM from engine"
    );
    for pkt in pcm.chunks(320) {
        sync_sess.accept_i16(pkt).expect("sync accept");
    }
    let sync_res = sync_sess.commit_endpoint().expect("sync commit");
    assert_eq!(sync_res.provisional, direct_text);
    assert_eq!(sync_res.committed, direct_text);

    // Path 4b: Async `LiveCommitAsrStream` via `NativeEngineBackend`.
    let live_cfg = LiveCommitConfig {
        commit: ccfg,
        hangover_ms: 300,
        preroll_ms: 300,
        deadline_ms: 2_000,
        ..LiveCommitConfig::default()
    };
    let tick = Arc::new(AtomicU64::new(0));
    let t_ref = Arc::clone(&tick);
    let clk = StreamClock::custom(move || t_ref.fetch_add(1_000_000, Ordering::SeqCst) + 1);
    let live_scfg = StreamConfig {
        segment_hard_cap_samples: u64::MAX,
        endpoint_hangover_ms: 300,
        ..StreamConfig::default()
    };
    let live_stream =
        ultra_stream::LiveCommitAsrStream::new(&backend, &live_scfg, &live_cfg, clk.clone())
            .expect("live commit stream");
    let prov_cfg = live_stream.provisional_decode_config();
    assert_eq!(prov_cfg.lm_sha256, expected_cfg.lm_sha256);
    assert_eq!(prov_cfg.beam_width, expected_cfg.beam_width);
    assert_eq!(
        prov_cfg
            .with_endpoint_policy(expected_cfg.endpoint_policy.clone())
            .policy_sha256(),
        expected_sha
    );

    let mut live_coord = StreamCoordinator::with_clock(live_scfg, Box::new(live_stream), clk);
    let mut evs = Vec::new();
    for (k, pkt) in pcm.chunks(320).enumerate() {
        live_coord
            .push_pcm((k * 320) as u64, pkt)
            .expect("live push");
        evs.extend(drain(&mut live_coord));
    }
    live_coord.finish_input().expect("live finish");
    evs.extend(drain(&mut live_coord));
    assert_eq!(live_coord.ledger().committed_text(), direct_text);

    // Verify F09: commit_pass status event separates commit_wall_s/ms and commit_cpu_s/ms.
    let status_json = evs
        .iter()
        .find_map(|e| match &e.body {
            Body::Status { code, detail } if code == "commit_pass" => {
                Some(serde_json::from_str::<serde_json::Value>(detail).expect("json"))
            }
            _ => None,
        })
        .expect("commit_pass status event");
    assert!(status_json["commit_wall_s"].is_number());
    assert!(status_json["commit_cpu_s"].is_number());
    assert!(status_json["commit_wall_ms"].is_number());
    assert!(status_json["commit_cpu_ms"].is_number());
    assert!(status_json["stale_worker_cpu_s"].is_number());
    assert!(status_json["stale_worker_jobs"].is_number());

    let _ = std::fs::remove_file(&lm_path);
}

#[test]
fn f09_live_commit_cancel_drains_and_accounts_worker_cpu_cleanly() {
    let model = synthetic_near_tie_model(2.0, 1.5);
    let eng = Engine::new(model, Isa::Scalar, Precision::F32).expect("eng");
    let backend = NativeEngineBackend::from_engine(eng, "0".repeat(64), "n/a");
    let live_cfg = LiveCommitConfig {
        commit: CommitConfig {
            commit_chunk: 4,
            tail_pad_samples: 0,
            ..CommitConfig::default()
        },
        hangover_ms: 200,
        preroll_ms: 200,
        deadline_ms: 2_000,
        ..LiveCommitConfig::default()
    };
    let scfg = StreamConfig {
        segment_hard_cap_samples: u64::MAX,
        endpoint_hangover_ms: 200,
        ..StreamConfig::default()
    };
    let mut stream =
        ultra_stream::LiveCommitAsrStream::new(&backend, &scfg, &live_cfg, StreamClock::real())
            .expect("stream");
    stream.push_pcm(0, &vec![0i16; 3_200]).expect("push");
    stream.cancel().expect("cancel");
    let m = stream.metrics();
    assert!(m.refiner_charged_cpu_s.is_finite() && m.refiner_charged_cpu_s >= 0.0);
    assert!(stream.stale_worker_cpu_s().is_finite() && stream.stale_worker_cpu_s() >= 0.0);
}

#[test]
fn f07_live_commit_stream_resets_hangover_on_non_growing_activity_and_commits_on_true_silence() {
    // Model whose joiner emits token 1 on the first chunk (arming speech), then pure blank on
    // subsequent chunks once context contains token 1 -> verifies LiveCommitAsrStream uses
    // DecodeProgress cleanly and commits only after genuine hangover silence.
    let mut base = Arc::try_unwrap(synthetic_near_tie_model(2.5, -2.0)).expect("unique");
    let (vocab, pe, jn) = (base.arch.vocab, base.arch.pred_emb, base.arch.joiner);
    let mut emb = vec![0.0f32; vocab * pe];
    emb[pe] = 1.0; // token 1 -> dim 0 = 1.0
    base.emb = emb;
    let mut pw = vec![0.0f32; jn * 2 * pe];
    pw[pe] = 3.0; // slot 1, dim 0 -> joiner hidden dim 0 = +3.0
    base.pred_proj = Linear::from_fp32(&pw, jn, 2 * pe, Some(&vec![0.0; jn]));
    let mut jw = vec![0.0f32; vocab * jn];
    jw[0] = 8.0; // once token 1 is in context, blank strongly dominates
    base.joiner_out = Linear::from_fp32(&jw, vocab, jn, Some(&[0.0, 2.5, -2.0, -4.0]));
    let model = Arc::new(base);

    let eng = Engine::new(model, Isa::Scalar, Precision::F32)
        .expect("eng")
        .with_beam(2);
    let backend = NativeEngineBackend::from_engine(eng, "0".repeat(64), "n/a");
    let live_cfg = LiveCommitConfig {
        commit: CommitConfig {
            commit_chunk: 4,
            tail_pad_samples: 0,
            beam_size: 2,
            ..CommitConfig::default()
        },
        hangover_ms: 320,
        preroll_ms: 160,
        deadline_ms: 2_000,
        ..LiveCommitConfig::default()
    };
    let scfg = StreamConfig {
        segment_hard_cap_samples: u64::MAX,
        endpoint_hangover_ms: 320,
        ..StreamConfig::default()
    };
    let tick = Arc::new(AtomicU64::new(0));
    let t_ref = Arc::clone(&tick);
    let clk = StreamClock::custom(move || t_ref.fetch_add(1_000_000, Ordering::SeqCst) + 1);
    let inner = backend
        .create_live_commit_stream_with_clock(&scfg, &live_cfg, clk.clone())
        .expect("live stream");
    let mut coord = StreamCoordinator::with_clock(scfg, inner, clk);

    // Push 1 chunk (2560 samples = 160 ms) -> emits token 1 ("the"), arms speech, no hangover yet.
    let mut evs = Vec::new();
    for (k, pkt) in vec![0i16; 2_560].chunks(320).enumerate() {
        coord.push_pcm((k * 320) as u64, pkt).expect("push speech");
        evs.extend(drain(&mut coord));
    }
    assert!(
        evs.iter().any(|e| matches!(e.body, Body::Preview(_))),
        "expected provisional Preview after beam speech chunk"
    );
    assert!(
        !evs.iter().any(|e| matches!(e.body, Body::Final { .. })),
        "must not emit Final before hangover_ms of silence"
    );

    // Push 480 ms of pure blank frames -> exceeds 320 ms hangover -> must trigger Final("hangover").
    for (k, pkt) in vec![0i16; 7_680].chunks(320).enumerate() {
        coord
            .push_pcm(2_560 + (k * 320) as u64, pkt)
            .expect("push silence");
        evs.extend(drain(&mut coord));
    }
    coord.finish_input().expect("finish");
    evs.extend(drain(&mut coord));
    let has_hangover = evs.iter().any(|e| match &e.body {
        Body::Status { code, detail } if code == "commit_pass" => {
            detail.contains("\"trigger\":\"hangover\"")
        }
        _ => false,
    });
    assert!(
        has_hangover,
        "genuine trailing silence must trigger hangover commit"
    );
}

struct StreamStageTempDir(PathBuf);

impl StreamStageTempDir {
    fn new(tag: &str) -> Self {
        let p = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
            .join(format!("f10-stream-stage-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).expect("mkdir");
        Self(p)
    }
}

impl Drop for StreamStageTempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn write_staged_stream_model(
    dir: &std::path::Path,
    tokens_name: &str,
    tokens_bytes: &[u8],
    fbank_name: &str,
    fbank_bytes: &[u8],
) {
    use serde_json::{json, Map, Value};
    use ultra_core::receipt::sha256_hex;
    std::fs::create_dir_all(dir).expect("mkdir");
    let (d, ch, ffn, k, vocab, pred_emb, joiner, f2) = (4, 4, 8, 3, 4, 4, 4, 19);
    let ckpt_sha = "8".repeat(64);
    let ones = |n: usize| vec![1.0f32; n];
    let zeros = |n: usize| vec![0.0f32; n];
    let tensors: Vec<(&str, Vec<usize>, Vec<f32>)> = vec![
        ("cmvn.mean", vec![80], zeros(80)),
        ("cmvn.istd", vec![80], ones(80)),
        ("encoder.stem.c1.weight", vec![ch, 1, 3, 3], zeros(ch * 9)),
        ("encoder.stem.c1.bias", vec![ch], zeros(ch)),
        (
            "encoder.stem.c2.weight",
            vec![ch, ch, 3, 3],
            zeros(ch * ch * 9),
        ),
        ("encoder.stem.c2.bias", vec![ch], zeros(ch)),
        (
            "encoder.stem.out.weight",
            vec![d, ch * f2],
            zeros(d * ch * f2),
        ),
        ("encoder.stem.out.bias", vec![d], zeros(d)),
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
        ("joiner.enc_proj.weight", vec![joiner, d], zeros(joiner * d)),
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
        (
            "joiner.out.weight",
            vec![vocab, joiner],
            zeros(vocab * joiner),
        ),
        ("joiner.out.bias", vec![vocab], vec![-2.0, 2.0, 1.5, -4.0]),
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

    let bpe_bytes = b"stream-bpe-v1";
    std::fs::write(dir.join("weights.safetensors"), &st_bytes).expect("weights");
    std::fs::write(dir.join(tokens_name), tokens_bytes).expect("tokens");
    std::fs::write(dir.join(fbank_name), fbank_bytes).expect("fbank");
    std::fs::write(dir.join("bpe.model"), bpe_bytes).expect("bpe");
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
            "tokens": { "path": tokens_name, "sha256": sha256_hex(tokens_bytes) },
            "mel_fbank": { "path": fbank_name, "sha256": sha256_hex(fbank_bytes) },
            "bpe": { "path": "bpe.model", "sha256": sha256_hex(bpe_bytes) }
        },
        "features": {
            "sample_rate": 16000, "frame_length": 400, "frame_shift": 160,
            "n_fft": 512, "n_mels": 80, "log_floor": 1e-6,
            "cmvn": { "mean": vec![0.0f32; 80], "istd": vec![1.0f32; 80] }
        },
        "decoder": {
            "kind": "stateless", "context": 2, "vocab_size": 4,
            "joiner_activation": "tanh", "blank_id": 0, "max_symbols": 1, "ctx_init": [0, 0]
        }
    });
    std::fs::write(dir.join("arch.json"), arch.to_string()).expect("arch.json");
}

#[test]
fn f10_live_commit_stream_is_staging_dir_invariant_and_sensitive_to_tail_pad_and_session_bias() {
    use ultra_engine::commit::SessionLexicalConfig;

    let tmp = StreamStageTempDir::new("live-commit");
    let dir_a = tmp.0.join("host_a/bundle");
    let dir_b = tmp.0.join("host_b/other/bundle");
    let tokens = VOCAB4.as_bytes();
    let fbank = vec![0u8; 257 * 80 * 4];
    write_staged_stream_model(&dir_a, "lexicon_a.tok", tokens, "fbank_a.bin", &fbank);
    write_staged_stream_model(&dir_b, "lexicon_b.tok", tokens, "fbank_b.bin", &fbank);
    // Add a decoy tokens.txt in dir_a that differs from arch.json's lexicon_a.tok.
    std::fs::write(
        dir_a.join("tokens.txt"),
        b"<blk> 0\n\xe2\x96\x81wrong 1\n\xe2\x96\x81wrong2 2\n\xe2\x96\x81wrong3 3\n",
    )
    .expect("decoy");

    let lm_mem =
        NgramLm::from_arpa_str(ARPA_PREFER_CAT, Some(ARPA_LODR_FLAT), 4, false).expect("lm");
    let lm_a_path = tmp.0.join("host_a/lm_one.ulm");
    let lm_b_path = tmp.0.join("host_b/other/lm_two.ulm");
    lm_mem.save(&lm_a_path).expect("save lm a");
    lm_mem.save(&lm_b_path).expect("save lm b");

    let lm_cfg = LmConfig {
        lm_weight: 0.8,
        lodr_weight: 0.6,
        blank_penalty: 0.0,
    };
    let backend_a = NativeEngineBackend::load_with_config(
        &dir_a,
        Precision::F32,
        1,
        true,
        &NativeEngineLoadConfig {
            beam: 1,
            beam_max_symbols: 1,
            lm: Some((Arc::new(NgramLm::load(&lm_a_path).expect("lm a")), lm_cfg)),
            ..NativeEngineLoadConfig::default()
        },
    )
    .expect("backend a");
    let backend_b = NativeEngineBackend::load_with_config(
        &dir_b,
        Precision::F32,
        1,
        true,
        &NativeEngineLoadConfig {
            beam: 1,
            beam_max_symbols: 1,
            lm: Some((Arc::new(NgramLm::load(&lm_b_path).expect("lm b")), lm_cfg)),
            ..NativeEngineLoadConfig::default()
        },
    )
    .expect("backend b");

    assert_ne!(
        backend_a.effective_decode_config().lm_path,
        backend_b.effective_decode_config().lm_path
    );
    assert_eq!(
        backend_a.effective_decode_config().policy_sha256(),
        backend_b.effective_decode_config().policy_sha256()
    );

    let scfg = StreamConfig {
        segment_hard_cap_samples: u64::MAX,
        endpoint_hangover_ms: 300,
        ..StreamConfig::default()
    };
    let base_live = LiveCommitConfig {
        commit: CommitConfig {
            commit_chunk: 4,
            tail_pad_samples: 5_120,
            beam_size: 1,
            ..CommitConfig::default()
        },
        hangover_ms: 300,
        preroll_ms: 300,
        deadline_ms: 1_000,
        session_bias: SessionLexicalConfig::default(),
        ..LiveCommitConfig::default()
    };
    let stream_a = ultra_stream::LiveCommitAsrStream::new(
        &backend_a,
        &scfg,
        &base_live,
        StreamClock::custom(|| 1),
    )
    .expect("stream a");
    let stream_b = ultra_stream::LiveCommitAsrStream::new(
        &backend_b,
        &scfg,
        &base_live,
        StreamClock::custom(|| 1),
    )
    .expect("stream b");

    // Two different staging directories with custom-named tokens/fbank + different LM paths
    // yield identical provisional and commit policy_sha256().
    let base_prov_sha = stream_a.provisional_decode_config().policy_sha256();
    let base_commit_sha = stream_a.commit_decode_config().policy_sha256();
    assert_eq!(
        base_prov_sha,
        stream_b.provisional_decode_config().policy_sha256()
    );
    assert_eq!(
        base_commit_sha,
        stream_b.commit_decode_config().policy_sha256()
    );
    assert_eq!(stream_a.commit_decode_config().tail_pad_samples, 5_120);
    assert_eq!(stream_a.commit_decode_config().tail_pad_frames, 8);
    assert!(
        !stream_a
            .provisional_decode_config()
            .session_bias_policy
            .enabled
    );

    // Changing tail_pad_samples on LiveCommitAsrStream strictly changes commit policy_sha256().
    let mut no_pad_live = base_live.clone();
    no_pad_live.commit.tail_pad_samples = 0;
    let stream_no_pad = ultra_stream::LiveCommitAsrStream::new(
        &backend_a,
        &scfg,
        &no_pad_live,
        StreamClock::custom(|| 1),
    )
    .expect("stream no pad");
    assert_eq!(stream_no_pad.commit_decode_config().tail_pad_frames, 0);
    assert_ne!(
        stream_no_pad.commit_decode_config().policy_sha256(),
        base_commit_sha
    );

    // Enabling active session_bias and changing its parameters strictly changes BOTH
    // provisional_decode_config().policy_sha256() and commit_decode_config().policy_sha256().
    let mut bias_on = base_live.clone();
    bias_on.session_bias = SessionLexicalConfig {
        enabled: true,
        min_occurrences: 2,
        max_entries: 64,
        bias_bonus: 0.35,
        min_word_chars: 5,
        min_bpe_pieces: 2,
    };
    let stream_bias_on = ultra_stream::LiveCommitAsrStream::new(
        &backend_a,
        &scfg,
        &bias_on,
        StreamClock::custom(|| 1),
    )
    .expect("stream bias on");
    let bias_on_prov_sha = stream_bias_on.provisional_decode_config().policy_sha256();
    let bias_on_commit_sha = stream_bias_on.commit_decode_config().policy_sha256();
    assert!(
        stream_bias_on
            .provisional_decode_config()
            .session_bias_policy
            .enabled
    );
    assert!(
        (stream_bias_on
            .commit_decode_config()
            .session_bias_policy
            .bias_bonus
            - 0.35)
            .abs()
            < 1e-6
    );
    assert_ne!(bias_on_prov_sha, base_prov_sha);
    assert_ne!(bias_on_commit_sha, base_commit_sha);

    // Tuning session_bias bonus / max_entries also strictly changes policy_sha256().
    let mut bias_tuned = bias_on.clone();
    bias_tuned.session_bias.bias_bonus = 0.50;
    bias_tuned.session_bias.max_entries = 128;
    let stream_bias_tuned = ultra_stream::LiveCommitAsrStream::new(
        &backend_a,
        &scfg,
        &bias_tuned,
        StreamClock::custom(|| 1),
    )
    .expect("stream bias tuned");
    assert_ne!(
        stream_bias_tuned
            .provisional_decode_config()
            .policy_sha256(),
        bias_on_prov_sha
    );
    assert_ne!(
        stream_bias_tuned.commit_decode_config().policy_sha256(),
        bias_on_commit_sha
    );
}
