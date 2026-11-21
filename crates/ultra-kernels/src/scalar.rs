//! Scalar reference kernels: loops over the ultra-core reference dot products. Every SIMD
//! kernel must match these bit for bit.

use crate::tlut2::trits;
use crate::{Frames, Gemv, I8View, T1View, TLut2View};
use ultra_core::numeric::dot_i8_a7;
use ultra_core::ternary::dot_t1;

/// Int8 weights x A7 frames (`ultra_core::numeric::dot_i8_a7` per row and frame).
pub fn gemv_i8(m: I8View<'_>, x: &Frames<'_>, out: &mut [i32]) {
    let f_n = x.count();
    for (r, o) in out.chunks_exact_mut(f_n).enumerate().take(m.rows()) {
        for (f, v) in o.iter_mut().enumerate() {
            *v = dot_i8_a7(m.row(r), x.frame(f));
        }
    }
}

/// T1 weights x A7 frames (`ultra_core::ternary::dot_t1` per row and frame).
pub fn gemv_t1(m: T1View<'_>, x: &Frames<'_>, out: &mut [i32]) {
    let f_n = x.count();
    for (r, o) in out.chunks_exact_mut(f_n).enumerate().take(m.rows()) {
        for (f, v) in o.iter_mut().enumerate() {
            *v = dot_t1(m.row(r), x.frame(f));
        }
    }
}

/// T-LUT2 weights x A7 frames, by decoding each pair state to its two trits.
pub fn gemv_tlut2(m: TLut2View<'_>, x: &Frames<'_>, out: &mut [i32]) {
    let (f_n, din) = (x.count(), m.din());
    for (r, o) in out.chunks_exact_mut(f_n).enumerate().take(m.rows()) {
        for (f, v) in o.iter_mut().enumerate() {
            let a = x.frame(f);
            let mut acc = 0i32;
            for p in 0..din.div_ceil(2) {
                let (t0, t1) = trits(m.pair_state(r, p));
                acc += t0 * i32::from(a[2 * p]);
                if 2 * p + 1 < din {
                    acc += t1 * i32::from(a[2 * p + 1]);
                }
            }
            *v = acc;
        }
    }
}

/// Signed-activation twin of [`gemv_tlut2`]: `x` is `f_n` frames of `din` values in -127..=127.
pub fn gemv_tlut2_signed(m: TLut2View<'_>, x: &[i8], f_n: usize, out: &mut [i32]) {
    let din = m.din();
    for (r, o) in out.chunks_exact_mut(f_n).enumerate().take(m.rows()) {
        for (f, v) in o.iter_mut().enumerate() {
            let a = &x[f * din..(f + 1) * din];
            let mut acc = 0i32;
            for p in 0..din.div_ceil(2) {
                let (t0, t1) = trits(m.pair_state(r, p));
                acc += t0 * i32::from(a[2 * p]);
                if 2 * p + 1 < din {
                    acc += t1 * i32::from(a[2 * p + 1]);
                }
            }
            *v = acc;
        }
    }
}
