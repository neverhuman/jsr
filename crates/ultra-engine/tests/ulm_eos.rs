//! Unit, round-trip, and backward-compatibility tests for `</s>` EOS packing in `NgramLm`
//! (`ultra.lm-ngram/v2`, `FLAG_HAS_EOS = 1 << 2`).

use ultra_engine::{EngineError, NgramLm};

const SAMPLE_ARPA_WITH_EOS: &str = "\
\\data\\
ngram 1=7
ngram 2=7
ngram 3=4

\\1-grams:
-4.0\t<unk>
-99.0\t<s>\t-0.3
-1.2\t</s>
-1.0\t2\t-0.2
-1.5\t3\t-0.4
-2.0\t4\t-0.1
-2.5\t5\t-0.5

\\2-grams:
-0.5\t<s> 2\t-0.15
-0.8\t2 3\t-0.25
-0.6\t3 4\t-0.35
-0.9\t2 4\t-0.05
-1.1\t3 </s>
-0.7\t4 </s>
-2.2\t<s> </s>

\\3-grams:
-0.2\t<s> 2 3
-0.3\t2 3 4
-0.4\t2 3 </s>
-0.25\t3 4 </s>

\\end\\
";

const SAMPLE_LODR: &str = "\
\\data\\
ngram 1=5
ngram 2=2

\\1-grams:
-3.5\t<unk>
-99.0\t<s>\t-0.2
-1.1\t2\t-0.3
-1.4\t3\t-0.1
-1.8\t4

\\2-grams:
-0.7\t<s> 2
-0.5\t2 3

\\end\\
";

#[test]
fn eos_backoff_matches_exact_arpa_kenlm_semantics_fp32_and_q8() {
    let ln10 = std::f32::consts::LN_10;
    let lm_f32 =
        NgramLm::from_arpa_str_with_eos(SAMPLE_ARPA_WITH_EOS, Some(SAMPLE_LODR), 8, false, true)
            .expect("fp32 eos");
    assert!(lm_f32.has_eos());
    assert_eq!(lm_f32.eos_id(), Some(9));

    let approx = |got: f32, expect_log10: f32, tol: f32| {
        let exp = expect_log10 * ln10;
        assert!(
            (got - exp).abs() < tol,
            "got {got} vs expected {exp} (diff {})",
            (got - exp).abs()
        );
    };

    // Direct 2-gram: <s> </s> = -2.2
    approx(lm_f32.score_eos(&[]), -2.2, 1e-5);
    // Backoff from [<s>, 2] -> bow(<s> 2) + bow(2) + P(</s>) = -0.15 - 0.2 - 1.2
    approx(lm_f32.score_eos(&[2]), -0.15 - 0.2 - 1.2, 1e-5);
    // Direct 3-gram: 2 3 </s> = -0.4
    approx(lm_f32.score_eos(&[2, 3]), -0.4, 1e-5);
    // Direct 3-gram: 3 4 </s> = -0.25
    approx(lm_f32.score_eos(&[2, 3, 4]), -0.25, 1e-5);
    // Backoff 3->2: [2, 4] has bow(2 4) = -0.05 + P(4 </s>) = -0.7
    approx(lm_f32.score_eos(&[2, 4]), -0.05 - 0.7, 1e-5);
    // Backoff to unigram: [3, 5] has bow(5) = -0.5 + P(</s>) = -1.2
    approx(lm_f32.score_eos(&[3, 5]), -0.5 - 1.2, 1e-5);

    // Regular token scoring and LODR remain exact.
    let c1 = lm_f32.ctx_state(&[2]);
    approx(lm_f32.lm_logp(&c1, 3), -0.2, 1e-5);
    approx(lm_f32.lodr_logp(&c1, 3), -0.5, 1e-5);

    // Q8 packed model matches within 8-bit quantile tolerance and round-trips bit-identically.
    let lm_q8 =
        NgramLm::from_arpa_str_with_eos(SAMPLE_ARPA_WITH_EOS, Some(SAMPLE_LODR), 8, true, true)
            .expect("q8 eos");
    assert!(lm_q8.has_eos());
    approx(lm_q8.score_eos(&[2, 3]), -0.4, 0.05);
    approx(lm_q8.score_eos(&[2, 3, 4]), -0.25, 0.05);

    let tmp = std::env::temp_dir().join(format!("ultra_lm_eos_v2_{}.ulm", std::process::id()));
    lm_q8.save(&tmp).expect("save v2 eos");
    let loaded = NgramLm::load(&tmp).expect("load v2 eos");
    assert!(loaded.has_eos());
    assert_eq!(loaded.eos_id(), Some(9));
    for hist in [&[][..], &[2][..], &[2, 3][..], &[2, 3, 4][..], &[3, 5][..]] {
        assert_eq!(
            lm_q8.score_eos(hist).to_bits(),
            loaded.score_eos(hist).to_bits()
        );
        let c0 = lm_q8.ctx_state(hist);
        let c1 = loaded.ctx_state(hist);
        for k in 1..8 {
            assert_eq!(
                lm_q8.lm_logp(&c0, k).to_bits(),
                loaded.lm_logp(&c1, k).to_bits()
            );
            assert_eq!(
                lm_q8.lodr_logp(&c0, k).to_bits(),
                loaded.lodr_logp(&c1, k).to_bits()
            );
        }
    }
    let mut bytes = std::fs::read(&tmp).expect("read");
    bytes[20] ^= 0xaa;
    std::fs::write(&tmp, &bytes).expect("corrupt");
    assert!(matches!(
        NgramLm::load(&tmp),
        Err(EngineError::Integrity(_))
    ));
    let _ = std::fs::remove_file(&tmp);
}

#[test]
fn legacy_packs_without_eos_flag_stay_bit_identical_and_score_eos_zero() {
    let lm_no_eos =
        NgramLm::from_arpa_str(SAMPLE_ARPA_WITH_EOS, Some(SAMPLE_LODR), 8, false).expect("no eos");
    assert!(!lm_no_eos.has_eos());
    assert_eq!(lm_no_eos.eos_id(), None);
    assert_eq!(lm_no_eos.score_eos(&[2, 3]), 0.0);

    let tmp_v1 = std::env::temp_dir().join(format!("ultra_lm_noeos_v1_{}.ulm", std::process::id()));
    let tmp_v2 = std::env::temp_dir().join(format!("ultra_lm_noeos_v2_{}.ulm", std::process::id()));
    lm_no_eos.save_v1(&tmp_v1).expect("save v1");
    lm_no_eos.save(&tmp_v2).expect("save v2");
    for p in [&tmp_v1, &tmp_v2] {
        let loaded = NgramLm::load(p).expect("load");
        assert!(!loaded.has_eos());
        assert_eq!(loaded.score_eos(&[2, 3]), 0.0);
        let c0 = lm_no_eos.ctx_state(&[2, 3]);
        let c1 = loaded.ctx_state(&[2, 3]);
        for k in 1..8 {
            assert_eq!(
                lm_no_eos.lm_logp(&c0, k).to_bits(),
                loaded.lm_logp(&c1, k).to_bits()
            );
            assert_eq!(
                lm_no_eos.lodr_logp(&c0, k).to_bits(),
                loaded.lodr_logp(&c1, k).to_bits()
            );
        }
        let _ = std::fs::remove_file(p);
    }
}

#[test]
fn final_beam_select_best_with_eos_prefers_complete_utterance() {
    let lm =
        NgramLm::from_arpa_str_with_eos(SAMPLE_ARPA_WITH_EOS, Some(SAMPLE_LODR), 8, false, true)
            .expect("eos lm");
    // Truncated hyp [2] has slightly higher raw beam score (-10.0 vs -10.4),
    // but complete hyp [2, 3, 4] has much higher P(</s> | 3, 4) = -0.25*ln10 vs -1.55*ln10.
    let hyps: [(&[u32], f32); 2] = [(&[2], -10.0), (&[2, 3, 4], -10.4)];
    assert_eq!(lm.select_best_with_eos(&hyps, 0.0), Some(0));
    assert_eq!(lm.select_best_with_eos(&hyps, 0.8), Some(1));
    assert_eq!(lm.select_best_with_eos(&[], 0.8), None);
}
