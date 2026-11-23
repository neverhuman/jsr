//! Cross-ISA determinism, golden-trace, packet-invariance, silence-rejection, speech-onset,
//! and uclamp tests for `ultra-gate`.

use ultra_gate::{
    get_thread_uclamp_max, ilog2_q8, parse_uclamp_max, set_thread_uclamp_max, GateConfig,
    Sentinel0, UclampError, HOP, LOG2_LUT, UCLAMP_MAX_LIMIT,
};

/// Platform-independent test signal (Appendix C of ULTRA_ALWAYS_ON_SLEEP_LADDER_SPEC):
/// 32-bit LCG noise (~-42 dBFS) plus loud and quiet 200 Hz square bursts.
fn golden_pcm(n: usize) -> Vec<i16> {
    let mut s: u32 = 12_345;
    (0..n)
        .map(|i| {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let mut v = i32::from((s >> 16) as u16 as i16) >> 7;
            let sq = if (i / 40) % 2 == 0 { 1 } else { -1 };
            if (32_000..40_000).contains(&i) {
                v += 2_000 * sq;
            }
            if (64_000..72_000).contains(&i) {
                v += 260 * sq;
            }
            v as i16
        })
        .collect()
}

fn fnv1a64(h: &mut u64, bytes: &[u8]) {
    for &b in bytes {
        *h ^= u64::from(b);
        *h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
}

#[test]
fn log2_lut_and_ilog2_q8_exact() {
    assert_eq!(ilog2_q8(0), 0);
    assert_eq!(ilog2_q8(1), 0);
    assert_eq!(ilog2_q8(2), 256);
    assert_eq!(ilog2_q8(3), 406);
    assert_eq!(ilog2_q8(1 << 40), 10_240);
    assert_eq!(LOG2_LUT[0], 0);
    assert_eq!(LOG2_LUT[128], 150);
    assert_eq!(LOG2_LUT[255], 255);
    assert!(LOG2_LUT.windows(2).all(|w| w[0] <= w[1]));
}

#[test]
fn golden_trace_bit_exact_cross_isa() {
    let pcm = golden_pcm(96_000);
    assert_eq!(&pcm[..8], &[10, 8, -234, -187, -47, 57, 253, -232]);

    let mut hp = 0xcbf2_9ce4_8422_2325u64;
    for s in &pcm {
        fnv1a64(&mut hp, &s.to_le_bytes());
    }
    assert_eq!(hp, 0x6f5d_0b63_084f_8ffd, "golden PCM checksum");

    let mut gate = Sentinel0::new(GateConfig::default());
    let mut trace_hash = 0xcbf2_9ce4_8422_2325u64;
    let mut hot = Vec::new();
    for (i, hop) in pcm.chunks_exact(HOP).enumerate() {
        let out = gate.push_hop(hop.try_into().expect("exact hop"));
        if i == 200 {
            assert_eq!(out.path, 3, "hop 200 fires both fast and slow paths");
            assert_eq!(out.level, [6623, 7586, 6811, 7605]);
            assert_eq!(out.floor, [6017, 5831, 6570, 6506]);
        }
        for v in out.level.iter().chain(out.floor.iter()) {
            fnv1a64(&mut trace_hash, &v.to_le_bytes());
        }
        fnv1a64(&mut trace_hash, &[out.path]);
        if out.hot {
            hot.push(i);
        }
    }
    let expected_hot: Vec<usize> = (200..=257).chain(401..=455).collect();
    assert_eq!(gate.hops(), 600);
    assert_eq!(hot, expected_hot, "hot hop indices");
    assert_eq!(
        trace_hash, 0xc7b5_b4e8_da91_b73f,
        "FNV-1a-64 trace checksum"
    );
}

#[test]
fn packet_size_invariance_80_160_320_640_and_odd() {
    let pcm = golden_pcm(48_000);
    let run = |packet: usize| {
        let mut gate = Sentinel0::new(GateConfig::default());
        let mut buf: Vec<i16> = Vec::new();
        let mut out = Vec::new();
        for pk in pcm.chunks(packet) {
            buf.extend_from_slice(pk);
            while buf.len() >= HOP {
                let hop: [i16; HOP] = buf[..HOP].try_into().expect("exact hop");
                out.push(gate.push_hop(&hop));
                buf.drain(..HOP);
            }
        }
        out
    };
    let baseline = run(160);
    assert_eq!(baseline.len(), 300);
    for packet in [73, 80, 320, 592, 640, 1024] {
        assert_eq!(run(packet), baseline, "mismatch for packet size {packet}");
    }
}

#[test]
fn silence_and_sub_floor_rejection() {
    let mut gate = Sentinel0::new(GateConfig::default());
    let zeros = [0i16; HOP];
    assert!(
        (0..100_000).all(|_| !gate.push_hop(&zeros).hot),
        "digital silence must never wake"
    );

    // Sub -72 dBFS low-amplitude noise (+/- 3 LSB) must stay below absolute floors.
    let mut sub_floor = Sentinel0::new(GateConfig::default());
    let mut s: u32 = 99_991;
    let mut hop = [0i16; HOP];
    for _ in 0..2_000 {
        for sample in &mut hop {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *sample = (((s >> 16) % 7) as i16) - 3;
        }
        assert!(
            !sub_floor.push_hop(&hop).hot,
            "sub -72 dBFS noise must not wake"
        );
    }
}

#[test]
fn speech_onset_wakes_and_thresholds_are_configurable() {
    let pcm = golden_pcm(96_000);
    let run_count = |cfg: GateConfig| -> usize {
        let mut gate = Sentinel0::new(cfg);
        pcm.chunks_exact(HOP)
            .filter(|h| gate.push_hop((*h).try_into().expect("exact hop")).hot)
            .count()
    };
    let default_hot = run_count(GateConfig::default());
    let stricter = run_count(GateConfig {
        theta_fast: 2048,
        theta_slow: 2048,
        ..GateConfig::default()
    });
    assert!(
        default_hot > 100,
        "speech-like bursts must wake default gate"
    );
    assert!(
        stricter < default_hot,
        "stricter thresholds must reduce hot hops ({stricter} vs {default_hot})"
    );
}

#[test]
fn uclamp_validation_and_safe_syscall_contract() {
    assert_eq!(parse_uclamp_max("0"), Ok(0));
    assert_eq!(parse_uclamp_max("512"), Ok(512));
    assert_eq!(parse_uclamp_max("1024"), Ok(UCLAMP_MAX_LIMIT));
    assert_eq!(
        parse_uclamp_max("1025"),
        Err(UclampError::OutOfRange {
            value: 1025,
            max: UCLAMP_MAX_LIMIT
        })
    );
    assert!(matches!(
        parse_uclamp_max("abc"),
        Err(UclampError::InvalidFormat(_))
    ));
    assert_eq!(
        set_thread_uclamp_max(UCLAMP_MAX_LIMIT + 1),
        Err(UclampError::OutOfRange {
            value: UCLAMP_MAX_LIMIT + 1,
            max: UCLAMP_MAX_LIMIT
        })
    );

    // Calling with a valid value must succeed or return a typed kernel error (never panic).
    let prev = get_thread_uclamp_max();
    let res = set_thread_uclamp_max(UCLAMP_MAX_LIMIT);
    match res {
        Ok(()) => {
            if let Ok(Some(cur)) = get_thread_uclamp_max() {
                assert!(cur <= UCLAMP_MAX_LIMIT);
            }
        }
        Err(
            UclampError::Unsupported { .. }
            | UclampError::PermissionDenied { .. }
            | UclampError::Syscall { .. },
        ) => {}
        Err(other) => panic!("unexpected uclamp error: {other}"),
    }
    let _ = prev;
}
