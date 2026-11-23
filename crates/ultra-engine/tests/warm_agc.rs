//! Unit, gain-sweep, sleep-persistence, calibration, and speed tests for H47 Warm AGC.

use std::sync::Arc;
use std::time::Instant;

use ultra_engine::agc::{db_to_linear, linear_to_db, DEFAULT_TARGET_RMS};
use ultra_engine::model::{Arch, Linear, Model};
use ultra_engine::{AgcStream, Engine, Isa, Precision, WarmAgc, WarmAgcConfig, Workspace};
use ultra_frontend::{GlobalCmvn, TrainerMelConfig};
use ultra_gp::TokenTable;

const SR: usize = 16_000;

fn rms(s: &[f32]) -> f32 {
    (s.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>() / (s.len().max(1) as f64)).sqrt() as f32
}

fn synth_speech(n: usize, target: f32) -> Vec<f32> {
    let tau = std::f32::consts::TAU;
    let mut out: Vec<f32> = (0..n)
        .map(|i| {
            let t = i as f32 / SR as f32;
            (0.75 + 0.25 * (tau * 4.0 * t).sin())
                * (0.6 * (tau * 190.0 * t).sin()
                    + 0.3 * (tau * 520.0 * t).sin()
                    + 0.15 * (tau * 1310.0 * t).sin())
        })
        .collect();
    let sc = target / rms(&out).max(1e-12);
    out.iter_mut().for_each(|x| *x *= sc);
    out
}

fn synth_noise(n: usize, dbfs: f32) -> Vec<f32> {
    let mut st = 0x9e37_79b9_7f4a_7c15u64;
    let mut out: Vec<f32> = (0..n)
        .map(|_| {
            st ^= st << 13;
            st ^= st >> 7;
            st ^= st << 17;
            2.0 * ((st >> 40) as f32 / (1u32 << 24) as f32) - 1.0
        })
        .collect();
    let sc = db_to_linear(dbfs) / rms(&out).max(1e-12);
    out.iter_mut().for_each(|x| *x *= sc);
    out
}

fn tiny_model() -> Arc<Model> {
    let cmvn = GlobalCmvn::new(&[-13.815_511; 80], &[1.0; 80]).expect("cmvn");
    let mel_cfg = Arc::new(TrainerMelConfig::default_htk(cmvn).expect("mel"));
    let tokens = Arc::new(
        TokenTable::from_str_checked("<blk> 0\n<unk> 1\n\u{2581}hello 2\n\u{2581}world 3\n", 4)
            .expect("toks"),
    );
    let w = |n: usize, s: f32| {
        (0..n)
            .map(|i| (i as f32 * 0.41).sin() * s)
            .collect::<Vec<_>>()
    };
    let arch = Arch {
        n_mels: 80,
        d: 4,
        n_blocks: 0,
        lower: 0,
        mid: 0,
        fin: 0,
        heads: 1,
        dh: 4,
        ffn: 16,
        kernel: 3,
        stem_ch: 2,
        f1: 39,
        f2: 19,
        chunk: 4,
        left: 16,
        post_block_norm: false,
        vocab: 4,
        pred_emb: 4,
        joiner: 4,
        blank: 0,
        max_symbols: 1,
        ctx_init: vec![0, 0],
        log_floor: 1e-6,
        checkpoint_sha256: String::new(),
        weights_sha256: String::new(),
        qat_stem_int8: false,
    };
    Arc::new(Model {
        arch,
        root: Default::default(),
        c1_w: vec![0.15; 18],
        c1_b: vec![0.0; 2],
        c2: Linear::from_fp32(&[0.15; 36], 2, 18, None),
        stem_out: Linear::from_fp32(&[0.15; 152], 4, 38, None),
        blocks: Vec::new(),
        final_norm: None,
        pair: None,
        restore: None,
        enc_proj: Linear::from_fp32(&w(16, 0.5), 4, 4, None),
        emb: w(16, 0.05),
        pred_proj: Linear::from_fp32(&w(32, 0.05), 4, 8, None),
        pred_cache: None,
        joiner_out: Linear::from_fp32(&w(16, 1.5), 4, 4, Some(&[0.4, -2.0, -0.1, -0.1])),
        blank_cert_f32: None,
        blank_cert_i8: None,
        rope_inv: vec![1.0, 0.1],
        mel_cfg,
        tokens,
        ctc_head: None,
        int8_prepared: None,
    })
}

#[test]
fn identity_when_disabled_is_bit_exact_and_stream_identical() {
    let cfg = WarmAgcConfig::default();
    assert!(!cfg.warm_agc && (linear_to_db(cfg.target_rms) + 24.6).abs() < 0.05);
    let mut agc = WarmAgc::disabled();
    let raw_f32 = synth_speech(SR, DEFAULT_TARGET_RMS * 0.1);
    let mut buf_f32 = raw_f32.clone();
    agc.seed_from_preroll(&raw_f32[..8_000]);
    agc.process_chunk(&mut buf_f32);
    assert!(raw_f32
        .iter()
        .zip(&buf_f32)
        .all(|(a, b)| a.to_bits() == b.to_bits()));

    let raw_i16: Vec<i16> = raw_f32.iter().map(|&x| (x * 32767.0) as i16).collect();
    let mut buf_i16 = raw_i16.clone();
    agc.seed_from_preroll_i16(&raw_i16[..8_000]);
    agc.process_i16(&mut buf_i16);
    assert_eq!(raw_i16, buf_i16);

    let eng = Engine::new(tiny_model(), Isa::Scalar, Precision::F32).unwrap();
    let (mut plain, mut ws1) = (eng.create_stream(), Workspace::default());
    let (mut wrapped, mut ws2) = (
        AgcStream::from_engine(&eng, cfg).unwrap(),
        Workspace::default(),
    );
    for pkt in raw_i16.chunks(320) {
        plain.accept_i16(pkt).unwrap();
        while eng.decode(&mut [&mut plain], &mut ws1).unwrap() > 0 {}
        wrapped.accept_i16(pkt).unwrap();
        wrapped.decode_ready(&eng, &mut ws2).unwrap();
    }
    plain.input_finished();
    wrapped.input_finished();
    while eng.decode(&mut [&mut plain], &mut ws1).unwrap() > 0 {}
    wrapped.decode_ready(&eng, &mut ws2).unwrap();
    assert_eq!(
        (wrapped.tokens(), eng.text(wrapped.stream())),
        (plain.tokens(), eng.text(&plain))
    );
}

#[test]
fn preroll_wake_seeding_and_leading_silence_guard() {
    for atten_db in [-20.0f32, -30.0f32] {
        let speech_rms = DEFAULT_TARGET_RMS * db_to_linear(atten_db);
        let mut mixed = vec![0.0f32; (SR * 35) / 100];
        mixed.extend_from_slice(&synth_speech((SR * 15) / 100, speech_rms));
        for pre in [synth_speech(SR / 2, speech_rms), mixed] {
            let (mut first, mut agc) = (
                synth_speech(SR / 10, speech_rms),
                WarmAgc::enabled_default(),
            );
            agc.seed_from_preroll(&pre);
            agc.process_chunk(&mut first);
            assert!(linear_to_db(rms(&first) / DEFAULT_TARGET_RMS).abs() <= 1.5);
        }
    }
    for mut pre in [
        vec![0.0f32; (SR * 40) / 100],
        synth_noise((SR * 40) / 100, -70.0),
    ] {
        pre.extend_from_slice(&synth_speech(SR / 10, DEFAULT_TARGET_RMS));
        let mut agc = WarmAgc::enabled_default();
        agc.seed_from_preroll(&pre);
        assert!(agc.gain_db().abs() <= 1.5);
    }
    let mut short = vec![0.0f32; SR / 2 - 320];
    short.extend_from_slice(&synth_speech(320, DEFAULT_TARGET_RMS * db_to_linear(-20.0)));
    let mut agc = WarmAgc::enabled_default();
    agc.seed_from_preroll(&short);
    assert!(agc.gain_db().abs() < 1e-4 && !agc.is_warm());
}

#[test]
fn silence_and_sleep_persistence_and_hard_reset() {
    assert_eq!(WarmAgcConfig::default().tau_s, 5.0);
    let mut agc = WarmAgc::enabled_default();
    synth_noise(SR * 10, -70.0)
        .chunks_mut(320)
        .for_each(|p| agc.process_chunk(p));
    assert!(agc.gain_db().abs() < 1e-4 && !agc.is_warm());

    let quiet_rms = DEFAULT_TARGET_RMS * db_to_linear(-20.0);
    synth_speech(SR * 30, quiet_rms)
        .chunks_mut(320)
        .for_each(|p| agc.process_chunk(p));
    let (learned_db, learned_rms) = (agc.gain_db(), agc.level_rms());
    assert!((learned_db - 20.0).abs() <= 1.5);
    agc.process_chunk(&mut synth_noise(SR * 10, -70.0));
    let settled_db = agc.gain_db();
    assert!(
        agc.level_rms().to_bits() == learned_rms.to_bits()
            && (settled_db - learned_db).abs() < 0.05
    );
    agc.reset_utterance();
    agc.on_sleep();
    assert!(
        agc.is_sleeping()
            && (agc.gain_db() - settled_db).abs() < 1e-5
            && (agc.level_rms() - learned_rms).abs() < 1e-6
    );

    agc.seed_from_preroll(&synth_noise(SR / 2, -70.0));
    assert!((agc.gain_db() - settled_db).abs() < 1e-5);
    let mut utt2 = synth_speech(SR / 10, quiet_rms);
    agc.process_chunk(&mut utt2);
    assert!(linear_to_db(rms(&utt2) / DEFAULT_TARGET_RMS).abs() <= 1.5);
    agc.hard_reset();
    assert!(!agc.is_warm() && agc.gain_db().abs() < 1e-5);
}

#[test]
fn multi_utterance_warm_carry_converges_on_0_neg20_neg30_db() {
    let eng = Engine::new(tiny_model(), Isa::Scalar, Precision::F32).unwrap();
    for (atten_db, clips) in [(0.0f32, 6usize), (-20.0, 12), (-30.0, 16)] {
        let mut st = AgcStream::from_engine(&eng, WarmAgcConfig::enabled()).unwrap();
        let clip = synth_speech(SR * 3, DEFAULT_TARGET_RMS * db_to_linear(atten_db));
        for i in 0..clips {
            if i > 0 {
                st.reset_utterance(&eng);
            }
            clip.chunks(320).for_each(|p| st.accept_f32(p).unwrap());
        }
        assert!((st.agc().gain_db() + atten_db).abs() <= 1.5);
    }
}

#[test]
fn clean_0db_guard_bounds_invariance_and_speed() {
    let mut agc = WarmAgc::enabled_default();
    let mut speech = synth_speech(SR * 3, DEFAULT_TARGET_RMS);
    speech.chunks_mut(320).for_each(|p| agc.process_chunk(p));
    assert!(linear_to_db(rms(&speech) / DEFAULT_TARGET_RMS).abs() <= 1.0);
    assert!(speech.iter().map(|x| x.abs()).fold(0.0f32, f32::max) < 0.90);

    let cfg = WarmAgcConfig {
        speech_floor_dbfs: -75.0,
        ..WarmAgcConfig::enabled()
    };
    let (mut lo, mut tiny) = (
        WarmAgc::new(cfg).unwrap(),
        synth_speech(SR * 2, db_to_linear(-68.0)),
    );
    lo.seed_from_preroll(&tiny[..SR / 2]);
    lo.process_chunk(&mut tiny);
    assert!((39.5..=40.001).contains(&lo.gain_db()));

    let (mut hi, mut loud) = (WarmAgc::enabled_default(), synth_speech(SR * 2, 0.707));
    hi.seed_from_preroll(&loud[..SR / 2]);
    hi.process_chunk(&mut loud);
    assert!(
        (-12.001..=-11.5).contains(&hi.gain_db())
            && loud
                .iter()
                .all(|&x| x.is_finite() && (-1.0..=1.0).contains(&x))
    );

    let raw = synth_speech(SR, DEFAULT_TARGET_RMS * db_to_linear(-20.0));
    let (mut a, mut b) = (raw.clone(), raw.clone());
    let (mut ga, mut gb) = (WarmAgc::enabled_default(), WarmAgc::enabled_default());
    a.chunks_mut(160).for_each(|p| ga.process_chunk(p));
    b.chunks_mut(592).for_each(|p| gb.process_chunk(p));
    assert!(a.iter().zip(&b).all(|(x, y)| (x - y).abs() <= 1e-6));

    let eng = Engine::new(tiny_model(), Isa::Scalar, Precision::F32).unwrap();
    let mut st = AgcStream::from_engine(&eng, WarmAgcConfig::enabled()).unwrap();
    let ring: Vec<i16> = raw[..SR / 2]
        .iter()
        .map(|&x| (x * 32767.0) as i16)
        .collect();
    let copy = ring.clone();
    st.wake_with_preroll_i16(&eng, &ring).unwrap();
    assert_eq!(ring, copy);

    let (mut spd, mut audio) = (
        WarmAgc::enabled_default(),
        synth_speech(SR * 10, DEFAULT_TARGET_RMS * db_to_linear(-20.0)),
    );
    spd.seed_from_preroll(&audio[..SR / 2]);
    let t0 = Instant::now();
    audio.chunks_mut(320).for_each(|p| spd.process_chunk(p));
    assert!(t0.elapsed().as_secs_f64() / 10.0 < if cfg!(debug_assertions) { 0.02 } else { 0.001 });
}
