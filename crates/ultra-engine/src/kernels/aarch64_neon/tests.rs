//! sdot, smmla and widening int8 paths agree exactly (run on aarch64 hosts).

use std::arch::aarch64::*;

use super::super::NR;
use super::int8::{acc_sdot, acc_widen, has_dotprod, has_i8mm};
use super::smmla::acc_smmla;

#[test]
fn smmla_matches_sdot_for_every_row_count() {
    if !(has_dotprod() && has_i8mm()) {
        return;
    }
    let kp = 64usize;
    let w: Vec<f32> = (0..16 * kp)
        .map(|i| ((i * 37 % 101) as f32 - 50.0) * 0.01)
        .collect();
    let p = crate::quant::PackedI8::quantize(&w, 16, kp, None);
    assert!(!p.mm.is_empty());
    for m in 1..=4 {
        let a: Vec<i8> = (0..m * kp)
            .map(|i| (i as i32 * 53 % 255 - 127) as i8)
            .collect();
        // SAFETY: both features detected above.
        let (x, y) = unsafe {
            (
                acc_sdot(&a, kp, p.panel(0), m),
                acc_smmla(&a, kp, p.mm_panel(0), m),
            )
        };
        for r in 0..m {
            for c in 0..4 {
                let (mut u, mut v) = ([0i32; 4], [0i32; 4]);
                // SAFETY: each array holds 4 i32.
                unsafe {
                    vst1q_s32(u.as_mut_ptr(), x[r][c]);
                    vst1q_s32(v.as_mut_ptr(), y[r][c]);
                }
                assert_eq!(u, v, "m {m} row {r} block {c}");
            }
        }
    }
}

#[test]
fn sdot_and_widening_paths_agree() {
    if !has_dotprod() {
        return;
    }
    let (kp, m) = (64usize, 4usize);
    let a: Vec<i8> = (0..m * kp)
        .map(|i| (i as i32 * 37 % 255 - 127) as i8)
        .collect();
    let panel: Vec<i8> = (0..kp * NR)
        .map(|i| (i as i32 * 53 % 255 - 127) as i8)
        .collect();
    // SAFETY: dotprod detected above.
    let s = unsafe { acc_sdot(&a, kp, &panel, m) };
    let w = acc_widen(&a, kp, &panel, m);
    for r in 0..m {
        for c in 0..4 {
            let (mut x, mut y) = ([0i32; 4], [0i32; 4]);
            // SAFETY: each array holds 4 i32.
            unsafe {
                vst1q_s32(x.as_mut_ptr(), s[r][c]);
                vst1q_s32(y.as_mut_ptr(), w[r][c]);
            }
            assert_eq!(x, y);
        }
    }
}
