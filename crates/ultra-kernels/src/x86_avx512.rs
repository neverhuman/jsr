//! AVX-512 F/BW/VBMI/VNNI kernels ported from the spec's Appendix A `wstream.c`.
//!
//! **UNTESTED.** The H0 host (Zen 2) has no AVX-512; this code compiles here but has never
//! run. The parity tests exercise it automatically on any host where
//! `Backend::Avx512Vnni.is_available()`; until such a run is recorded it must not be used
//! for evidence.
//!
//! # Safety model
//! As in `x86_avx2`: the `pub(crate)` entry points are reached only from `Gemv::gemv`
//! after `check_call` confirmed every required feature at run time, and all loads/stores
//! are unaligned and inside slices whose lengths are asserted first.
//!
//! # Arithmetic
//! `vpdpbusd` (u8 x s8, four products summed into i32, no saturation) is exact for any
//! u8 activation. T1 is unpacked in natural order by `vpmovzxwq` + `vpmultishiftqb`.

use crate::{Frames, Gemv, I8View, T1View};
use std::arch::x86_64::*;
use ultra_core::numeric::dot_i8_a7;
use ultra_core::ternary::dot_t1;

const FRAME_BLOCK: usize = 4;
/// Weights per 512-bit vector.
const LANES: usize = 64;

pub(crate) fn gemv_i8(m: I8View<'_>, x: &Frames<'_>, out: &mut [i32]) {
    // SAFETY: only called after check_call confirmed AVX-512 F/BW/VBMI/VNNI (module docs).
    unsafe { gemv_i8_vnni(m, x, out) }
}

pub(crate) fn gemv_t1(m: T1View<'_>, x: &Frames<'_>, out: &mut [i32]) {
    // SAFETY: as above.
    unsafe { gemv_t1_vnni(m, x, out) }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vbmi,avx512vnni")]
fn dot_block<const N: usize>(w: &[i8], a: &[&[u8]], n: usize) -> [i32; N] {
    assert!(n.is_multiple_of(LANES) && w.len() >= n && a.len() == N);
    assert!(a.iter().all(|s| s.len() >= n));
    let ptrs: [*const u8; N] = std::array::from_fn(|f| a[f].as_ptr());
    let mut acc = [_mm512_setzero_si512(); N];
    let mut j = 0;
    while j < n {
        // SAFETY: j + 64 <= n <= every slice length (asserted).
        let wv = unsafe { _mm512_loadu_si512(w.as_ptr().add(j).cast()) };
        for (acc_f, p) in acc.iter_mut().zip(ptrs) {
            // SAFETY: as above.
            let av = unsafe { _mm512_loadu_si512(p.add(j).cast()) };
            *acc_f = _mm512_dpbusd_epi32(*acc_f, av, wv);
        }
        j += LANES;
    }
    acc.map(|v| _mm512_reduce_add_epi32(v))
}

#[target_feature(enable = "avx512f,avx512bw,avx512vbmi,avx512vnni")]
fn frames_dot(w: &[i8], acts: &[&[u8]], n: usize, out: &mut [i32]) {
    let mut f = 0;
    while f < acts.len() {
        let take = (acts.len() - f).min(FRAME_BLOCK);
        let a = &acts[f..f + take];
        let o = &mut out[f..f + take];
        match take {
            4 => o.copy_from_slice(&dot_block::<4>(w, a, n)),
            3 => o.copy_from_slice(&dot_block::<3>(w, a, n)),
            2 => o.copy_from_slice(&dot_block::<2>(w, a, n)),
            _ => o.copy_from_slice(&dot_block::<1>(w, a, n)),
        }
        f += take;
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vbmi,avx512vnni")]
fn gemv_i8_vnni(m: I8View<'_>, x: &Frames<'_>, out: &mut [i32]) {
    let (f_n, din) = (x.count(), m.din());
    let n = din - din % LANES;
    let acts: Vec<&[u8]> = (0..f_n).map(|f| x.frame(f)).collect();
    for (r, o) in out.chunks_exact_mut(f_n).enumerate() {
        let w = m.row(r);
        o.fill(0);
        if n > 0 {
            frames_dot(w, &acts, n, o);
        }
        if n < din {
            for (v, a) in o.iter_mut().zip(&acts) {
                *v += dot_i8_a7(&w[n..], &a[n..]);
            }
        }
    }
}

/// Unpack `nb` groups of 16 packed bytes into 64 signed bytes each, natural order.
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi,avx512vnni")]
fn decode_t1(p: &[u8], out: &mut [i8], nb: usize) {
    assert!(p.len() >= nb * 16 && out.len() >= nb * LANES);
    let ctrl = _mm512_set1_epi64(0x0E0C_0A08_0604_0200);
    let (three, one) = (_mm512_set1_epi8(3), _mm512_set1_epi8(1));
    for b in 0..nb {
        // SAFETY: 16 * b + 16 <= p.len(); 64 * b + 64 <= out.len() (asserted).
        unsafe {
            let pk = _mm_loadu_si128(p.as_ptr().add(16 * b).cast());
            let q = _mm512_cvtepu16_epi64(pk);
            let codes = _mm512_and_si512(_mm512_multishift_epi64_epi8(ctrl, q), three);
            let dst = out.as_mut_ptr().add(LANES * b);
            _mm512_storeu_si512(dst.cast(), _mm512_sub_epi8(codes, one));
        }
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vbmi,avx512vnni")]
fn gemv_t1_vnni(m: T1View<'_>, x: &Frames<'_>, out: &mut [i32]) {
    let (f_n, din) = (x.count(), m.din());
    let nb = din / LANES;
    let n = nb * LANES;
    let acts: Vec<&[u8]> = (0..f_n).map(|f| x.frame(f)).collect();
    let mut scratch = vec![0i8; n];
    for (r, o) in out.chunks_exact_mut(f_n).enumerate() {
        let row = m.row(r);
        o.fill(0);
        if n > 0 {
            decode_t1(row, &mut scratch, nb);
            frames_dot(&scratch, &acts, n, o);
        }
        if n < din {
            for (v, a) in o.iter_mut().zip(&acts) {
                *v += dot_t1(&row[n / 4..], &a[n..]);
            }
        }
    }
}
