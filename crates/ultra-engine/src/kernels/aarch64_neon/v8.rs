//! The AVX2 `__m256` shape on NEON (two four-lane halves) and the four-lane `exp`.
//! Safety model: see the parent module docs.

use std::arch::aarch64::*;

use super::super::scalar::{EXP_HI, EXP_LO, EXP_P, LN2_HI, LN2_LO, LOG2E};

/// Eight f32 lanes as two NEON registers (lanes 0-3, 4-7): the AVX2 `__m256` shape.
#[derive(Clone, Copy)]
pub(super) struct V8(pub(super) float32x4_t, pub(super) float32x4_t);

impl V8 {
    #[inline(always)]
    pub(super) fn zero() -> Self {
        // SAFETY: vdupq_n_f32 is a register-only NEON op (baseline on aarch64).
        unsafe { V8(vdupq_n_f32(0.0), vdupq_n_f32(0.0)) }
    }

    #[inline(always)]
    pub(super) fn splat(v: f32) -> Self {
        // SAFETY: register-only NEON op.
        unsafe { V8(vdupq_n_f32(v), vdupq_n_f32(v)) }
    }

    /// Load 8 floats from `s[i..i + 8]`.
    #[inline(always)]
    pub(super) fn load(s: &[f32], i: usize) -> Self {
        assert!(i + 8 <= s.len());
        // SAFETY: i + 8 <= s.len() (asserted).
        unsafe {
            V8(
                vld1q_f32(s.as_ptr().add(i)),
                vld1q_f32(s.as_ptr().add(i + 4)),
            )
        }
    }

    #[inline(always)]
    pub(super) fn store(self, s: &mut [f32], i: usize) {
        assert!(i + 8 <= s.len());
        // SAFETY: i + 8 <= s.len() (asserted).
        unsafe {
            vst1q_f32(s.as_mut_ptr().add(i), self.0);
            vst1q_f32(s.as_mut_ptr().add(i + 4), self.1);
        }
    }

    #[inline(always)]
    pub(super) fn add(self, o: Self) -> Self {
        // SAFETY: register-only NEON ops.
        unsafe { V8(vaddq_f32(self.0, o.0), vaddq_f32(self.1, o.1)) }
    }

    #[inline(always)]
    pub(super) fn sub(self, o: Self) -> Self {
        // SAFETY: register-only NEON ops.
        unsafe { V8(vsubq_f32(self.0, o.0), vsubq_f32(self.1, o.1)) }
    }

    #[inline(always)]
    pub(super) fn mul(self, o: Self) -> Self {
        // SAFETY: register-only NEON ops.
        unsafe { V8(vmulq_f32(self.0, o.0), vmulq_f32(self.1, o.1)) }
    }

    #[inline(always)]
    pub(super) fn div(self, o: Self) -> Self {
        // SAFETY: register-only NEON ops.
        unsafe { V8(vdivq_f32(self.0, o.0), vdivq_f32(self.1, o.1)) }
    }

    /// `self * b + c` with one rounding (`_mm256_fmadd_ps(self, b, c)`).
    #[inline(always)]
    pub(super) fn fma(self, b: Self, c: Self) -> Self {
        // SAFETY: register-only NEON ops (vfmaq_f32(c, a, b) = c + a * b, fused).
        unsafe { V8(vfmaq_f32(c.0, self.0, b.0), vfmaq_f32(c.1, self.1, b.1)) }
    }

    /// The AVX2 `hsum` tree: (lane j + lane j+4), then (s0 + s2, s1 + s3), then their sum.
    #[inline(always)]
    pub(super) fn hsum(self) -> f32 {
        // SAFETY: register-only NEON ops.
        unsafe {
            let s = vaddq_f32(self.0, self.1);
            let a = vgetq_lane_f32::<0>(s) + vgetq_lane_f32::<2>(s);
            let b = vgetq_lane_f32::<1>(s) + vgetq_lane_f32::<3>(s);
            a + b
        }
    }
}

/// Four-lane twin of `scalar::exp_approx` (same roundings as the AVX2 `exp8`).
#[inline(always)]
pub(crate) fn exp4(x: float32x4_t) -> float32x4_t {
    // SAFETY: register-only NEON ops.
    unsafe {
        let x = vminq_f32(vmaxq_f32(x, vdupq_n_f32(EXP_LO)), vdupq_n_f32(EXP_HI));
        let fx = vrndnq_f32(vmulq_f32(x, vdupq_n_f32(LOG2E)));
        let r = vfmsq_f32(x, fx, vdupq_n_f32(LN2_HI));
        let r = vfmsq_f32(r, fx, vdupq_n_f32(LN2_LO));
        let z = vmulq_f32(r, r);
        let mut y = vdupq_n_f32(EXP_P[0]);
        for &c in &EXP_P[1..] {
            y = vfmaq_f32(vdupq_n_f32(c), y, r);
        }
        let y = vaddq_f32(vfmaq_f32(r, y, z), vdupq_n_f32(1.0));
        let e = vaddq_s32(vcvtq_s32_f32(fx), vdupq_n_s32(127));
        let e = vminq_s32(vmaxq_s32(e, vdupq_n_s32(0)), vdupq_n_s32(255));
        vmulq_f32(y, vreinterpretq_f32_s32(vshlq_n_s32::<23>(e)))
    }
}

#[inline(always)]
pub(super) fn exp8(x: V8) -> V8 {
    V8(exp4(x.0), exp4(x.1))
}
