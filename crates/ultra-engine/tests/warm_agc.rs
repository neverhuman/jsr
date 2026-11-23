//! Unit, gain-sweep, sleep-persistence, and speed tests for H47 Warm AGC.

use std::sync::Arc;
use std::time::Instant;

use ultra_engine::agc::{db_to_linear, linear_to_db, DEFAULT_TARGET_RMS};
use ultra_engine::model::{Arch, Linear, Model};
use ultra_engine::{AgcStream, Engine, Isa, Precision, WarmAgc, WarmAgcConfig, Workspace};
use ultra_frontend::{GlobalCmvn, TrainerMelConfig};
use ultra_gp::TokenTable;

const SR: usize = 16_000;

fn rms(samples: &[f32]) -> f32 {
    let msq = samples
        .iter()
        .map(|&x| f64::from(x) * f64::from(x))
        .sum::<f64>()
        / (samples.len().max(1) as f64);
    msq.sqrt() as f32
}

fn synth_speech(n: usize, target_rms: f32) -> Vec<f32> {
    let mut out: Vec<f32> = (0..n)
        .map(|i| {
            let t = (i as f32) / (SR as f32);
            let env = 0.75 + 0.25 * (std::f32::consts::TAU * 4.0 * t).sin();
            env * (0.60 * (std::f32::consts::TAU * 190.0 * t).sin()
                + 0.30 * (std::f32::consts::TAU * 520.0 * t).sin()
                + 0.15 * (std::f32::consts::TAU * 1310.0 * t).sin())
        })
        .collect();
    let scale = target_rms / rms(&out).max(1e-12);
    out.iter_mut().for_each(|x| *x *= scale);
    out
}

fn synth_noise(n: usize, dbfs: f32) -> Vec<f32> {
    let mut st = 0x9e37_79b9_7f4a_7c15u64;
    let mut out: Vec<f32> = (0..n)
        .map(|_| {
            st ^= st << 13;
            st ^= st >> 7;
            st ^= st << 17;
            2.0 * (((st >> 40) as f32) / ((1u32 << 24) as f32)) - 1.0
        })
        .collect();
    let scale = db_to_linear(dbfs) / rms(&out).max(1e-12);
    out.iter_mut().for_each(|x| *x *= scale);
    out
}

fn tiny_model() -> Arc<Model> {
    let mut fb = vec![0u8; 257 * 80 * 4];
    for m in 0..80 {
        for bin in 1..40 {
            let o = (bin * 80 + m) * 4;
            fb[o..o + 4].copy_from_slice(&0.25f32.to_le_bytes());
        }
    }
    let cmvn = GlobalCmvn::new(&[-13.815_511; 80], &[1.0; 80]).expect("cmvn");
    let mel_cfg = Arc::new(TrainerMelConfig::from_fbank_le_bytes(&fb, 1e-6, cmvn).expect("mel"));
    let toks = "<blk> 0\n<unk> 1\n\u{2581}hello 2\n\u{2581}world 3\n";
    let tokens = Arc::new(TokenTable::from_str_checked(toks, 4).expect("toks"));
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
        int8_prepared: None,
    })
}

#[test]
fn identity_when_disabled_is_bit_exact_and_stream_identical() {
    assert!(!WarmAgcConfig::default().warm_agc);
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
    let mut wrapped = AgcStream::from_engine(&eng, WarmAgcConfig::default()).unwrap();
    let mut ws2 = Workspace::default();
    for pkt in raw_i16.chunks(320) {
        plain.accept_i16(pkt).unwrap();
        while eng.decode(&mut [&mut plain], &mut ws1).unwrap() > 0 {}
        wrapped.accept_i16(pkt).unwrap();
        wrapped.decode_ready(&eng, &mut ws2).unwrap();
    }
    plain.input_finished();
    while eng.decode(&mut [&mut plain], &mut ws1).unwrap() > 0 {}
    wrapped.input_finished();
    wrapped.decode_ready(&eng, &mut ws2).unwrap();
    assert_eq!(wrapped.tokens(), plain.tokens());
    assert_eq!(eng.text(wrapped.stream()), eng.text(&plain));
}

#[test]
fn preroll_wake_seeding_recovers_first_100ms_within_1_5_db() {
    for atten_db in [-20.0f32, -30.0f32] {
        let speech_rms = DEFAULT_TARGET_RMS * db_to_linear(atten_db);
        let mut mixed = vec![0.0f32; (SR * 35) / 100];
        mixed.extend_from_slice(&synth_speech((SR * 15) / 100, speech_rms));
        for preroll in [synth_speech(SR / 2, speech_rms), mixed] {
            let mut first_word = synth_speech(SR / 10, speech_rms);
            let mut agc = WarmAgc::enabled_default();
            agc.seed_from_preroll(&preroll);
            agc.process_chunk(&mut first_word);
            let err = linear_to_db(rms(&first_word) / DEFAULT_TARGET_RMS).abs();
            assert!(
                err <= 1.5,
                "{atten_db} dB preroll: err {err:.2} dB > 1.5 dB"
            );
        }
    }
}

#[test]
fn silence_and_low_noise_never_pump_gain() {
    let mut agc = WarmAgc::enabled_default();
    let mut noise = synth_noise(SR * 10, -70.0);
    for pkt in noise.chunks_mut(320) {
        agc.process_chunk(pkt);
    }
    assert!(agc.gain_db().abs() < 1e-4 && !agc.is_warm());

    agc.seed_from_preroll(&synth_speech(
        SR / 2,
        DEFAULT_TARGET_RMS * db_to_linear(-20.0),
    ));
    let warm_db = agc.gain_db();
    agc.process_chunk(&mut synth_noise(SR * 10, -70.0));
    assert!((agc.gain_db() - warm_db).abs() < 1e-4);
}

#[test]
fn level_persists_across_utterances_and_sleep_until_hard_reset() {
    assert_eq!(WarmAgcConfig::default().tau_s, 5.0);
    let quiet_rms = DEFAULT_TARGET_RMS * db_to_linear(-20.0);
    let mut agc = WarmAgc::enabled_default();
    // 20 s (4 * tau_s) of -20 dB speech converges smoothly under tau = 5 s without seeding.
    for pkt in synth_speech(SR * 20, quiet_rms).chunks_mut(320) {
        agc.process_chunk(pkt);
    }
    let (learned_db, learned_rms) = (agc.gain_db(), agc.level_rms());
    assert!(
        (learned_db - 20.0).abs() <= 1.5,
        "learned_db={learned_db:.2}"
    );

    agc.reset_utterance();
    agc.on_sleep();
    assert!(agc.is_sleeping() && (agc.gain_db() - learned_db).abs() < 1e-5);
    assert!((agc.level_rms() - learned_rms).abs() < 1e-6);

    agc.seed_from_preroll(&synth_noise(SR / 2, -70.0));
    assert!((agc.gain_db() - learned_db).abs() < 1e-5);
    let mut utt2_first = synth_speech(SR / 10, quiet_rms);
    agc.process_chunk(&mut utt2_first);
    assert!(linear_to_db(rms(&utt2_first) / DEFAULT_TARGET_RMS).abs() <= 1.5);

    agc.hard_reset();
    assert!(!agc.is_warm() && agc.gain_db().abs() < 1e-5);
}

#[test]
fn clean_0db_guard_stays_within_1db_without_clipping() {
    let mut agc = WarmAgc::enabled_default();
    agc.seed_from_preroll(&synth_speech(SR / 2, DEFAULT_TARGET_RMS));
    let mut speech = synth_speech(SR * 2, DEFAULT_TARGET_RMS);
    for pkt in speech.chunks_mut(320) {
        agc.process_chunk(pkt);
    }
    let err = linear_to_db(rms(&speech) / DEFAULT_TARGET_RMS).abs();
    let peak = speech.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    assert!(err <= 1.0 && peak < 0.90, "err={err:.2} dB peak={peak:.3}");
}

#[test]
fn gain_bounds_and_soft_limiter_are_strictly_enforced() {
    let mut agc_lo = WarmAgc::enabled_default();
    let mut tiny = synth_speech(SR * 4, db_to_linear(-64.0));
    agc_lo.seed_from_preroll(&tiny[..SR / 2]);
    agc_lo.process_chunk(&mut tiny);
    assert!((39.5..=40.001).contains(&agc_lo.gain_db()));

    let mut agc_hi = WarmAgc::enabled_default();
    let mut loud = synth_speech(SR * 2, 0.707);
    agc_hi.seed_from_preroll(&loud[..SR / 2]);
    agc_hi.process_chunk(&mut loud);
    assert!((-12.001..=-11.5).contains(&agc_hi.gain_db()));
    assert!(loud
        .iter()
        .all(|&x| x.is_finite() && (-1.0..=1.0).contains(&x)));
}

#[test]
fn packet_invariance_and_recogniser_only_raw_ring_preservation() {
    let raw = synth_speech(SR, DEFAULT_TARGET_RMS * db_to_linear(-20.0));
    let (mut a, mut b) = (raw.clone(), raw.clone());
    let (mut agc_a, mut agc_b) = (WarmAgc::enabled_default(), WarmAgc::enabled_default());
    for pkt in a.chunks_mut(160) {
        agc_a.process_chunk(pkt);
    }
    for pkt in b.chunks_mut(592) {
        agc_b.process_chunk(pkt);
    }
    assert!(a.iter().zip(&b).all(|(x, y)| (x - y).abs() <= 1e-6));

    let eng = Engine::new(tiny_model(), Isa::Scalar, Precision::F32).unwrap();
    let mut st = AgcStream::from_engine(&eng, WarmAgcConfig::enabled()).unwrap();
    let raw_ring: Vec<i16> = raw[..SR / 2]
        .iter()
        .map(|&x| (x * 32767.0) as i16)
        .collect();
    let ring_copy = raw_ring.clone();
    st.wake_with_preroll_i16(&eng, &raw_ring).unwrap();
    assert_eq!(raw_ring, ring_copy);
    assert!((st.agc().gain_db() - 20.0).abs() <= 1.5);
}

#[test]
fn speed_micro_check_well_under_realtime_budget() {
    let mut agc = WarmAgc::enabled_default();
    let mut audio = synth_speech(SR * 10, DEFAULT_TARGET_RMS * db_to_linear(-20.0));
    agc.seed_from_preroll(&audio[..SR / 2]);
    let t0 = Instant::now();
    for pkt in audio.chunks_mut(320) {
        agc.process_chunk(pkt);
    }
    let frac = t0.elapsed().as_secs_f64() / 10.0;
    let budget = if cfg!(debug_assertions) { 0.02 } else { 0.001 };
    assert!(frac < budget, "WarmAgc used {:.4}% RT", frac * 100.0);
}
