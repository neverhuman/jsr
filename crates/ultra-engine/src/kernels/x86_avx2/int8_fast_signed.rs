//! Single-call zero-spill AVX2 `M = 4` signed 2-group paired microkernel (`NTA` / `T0`).

use std::arch::asm;
use std::arch::x86_64::*;

use super::super::NR;
use super::int8_fast_prep::{epilogue_store, Align32};

/// Single-call zero-spill AVX2 `M = 4` signed 2-group paired kernel across ALL `pairs` of a panel.
/// Caller stores `ONES_I16` in `acc[0]` before calling; `ymm0..ymm7` stay live in registers for
/// the entire panel; rare bad pairs branch to cold label `3:` (2 plain signed groups).
///
/// # Safety
/// Caller must ensure AVX2+FMA is available, `acc` points to 32-byte aligned `[__m256i; 8]` with
/// `ONES_I16` in `acc[0]`, `wp` has at least `pairs * 128` readable bytes, rows `0..3` at stride
/// `ks` from `a0`/`aa0` (with `a3 = a0 + 3*ks`, `aa3 = aa0 + 3*ks`) each have at least `pairs * 8`
/// readable bytes, and `bad_ptr` is sorted and terminated by `u16::MAX`.
#[inline(never)]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn run_signed_pair_all_asm<const NTA: bool>(
    wp: *const i8,
    a0: *const i8,
    aa0: *const u8,
    a3: *const i8,
    aa3: *const u8,
    ks: usize,
    pairs: usize,
    bad_ptr: *const u16,
    acc: *mut __m256i,
) {
    if pairs == 0 {
        return;
    }
    macro_rules! sp_row {
        ($a0op:literal, $aa0op:literal, $a1op:literal, $aa1op:literal, $c0:literal, $c1:literal) => {
            concat!(
                "vpbroadcastd ymm11, dword ptr [",
                $a0op,
                "]\n",
                "vpbroadcastd ymm12, dword ptr [",
                $aa0op,
                "]\n",
                "vmovdqu ymm13, ymmword ptr [{w} + 0]\n",
                "vmovdqu ymm14, ymmword ptr [{w} + 32]\n",
                "vpsignb ymm13, ymm13, ymm11\n",
                "vpmaddubsw ymm9, ymm12, ymm13\n",
                "vpsignb ymm14, ymm14, ymm11\n",
                "vpmaddubsw ymm10, ymm12, ymm14\n",
                "vpbroadcastd ymm11, dword ptr [",
                $a1op,
                "]\n",
                "vpbroadcastd ymm12, dword ptr [",
                $aa1op,
                "]\n",
                "vmovdqu ymm13, ymmword ptr [{w} + 64]\n",
                "vmovdqu ymm14, ymmword ptr [{w} + 96]\n",
                "vpsignb ymm13, ymm13, ymm11\n",
                "vpmaddubsw ymm13, ymm12, ymm13\n",
                "vpaddw ymm9, ymm9, ymm13\n",
                "vpsignb ymm14, ymm14, ymm11\n",
                "vpmaddubsw ymm14, ymm12, ymm14\n",
                "vpaddw ymm10, ymm10, ymm14\n",
                "vpmaddwd ymm9, ymm9, ymm8\n",
                "vpaddd ",
                $c0,
                ", ",
                $c0,
                ", ymm9\n",
                "vpmaddwd ymm10, ymm10, ymm8\n",
                "vpaddd ",
                $c1,
                ", ",
                $c1,
                ", ymm10\n",
            )
        };
    }
    macro_rules! s_single_grp {
        ($woff0:literal, $woff1:literal, $aoff:literal) => {
            concat!(
                "vmovdqu ymm9,  ymmword ptr [{w} + ",
                $woff0,
                "]\n",
                "vmovdqu ymm10, ymmword ptr [{w} + ",
                $woff1,
                "]\n",
                // Row 0
                "vpbroadcastd ymm11, dword ptr [{a0} + ",
                $aoff,
                "]\n",
                "vpbroadcastd ymm12, dword ptr [{aa0} + ",
                $aoff,
                "]\n",
                "vpsignb ymm13, ymm9, ymm11\n",
                "vpmaddubsw ymm13, ymm12, ymm13\n",
                "vpmaddwd ymm13, ymm13, ymm8\n",
                "vpaddd ymm0, ymm0, ymm13\n",
                "vpsignb ymm14, ymm10, ymm11\n",
                "vpmaddubsw ymm14, ymm12, ymm14\n",
                "vpmaddwd ymm14, ymm14, ymm8\n",
                "vpaddd ymm1, ymm1, ymm14\n",
                // Row 1
                "vpbroadcastd ymm11, dword ptr [{a0} + {ks} + ",
                $aoff,
                "]\n",
                "vpbroadcastd ymm12, dword ptr [{aa0} + {ks} + ",
                $aoff,
                "]\n",
                "vpsignb ymm13, ymm9, ymm11\n",
                "vpmaddubsw ymm13, ymm12, ymm13\n",
                "vpmaddwd ymm13, ymm13, ymm8\n",
                "vpaddd ymm2, ymm2, ymm13\n",
                "vpsignb ymm14, ymm10, ymm11\n",
                "vpmaddubsw ymm14, ymm12, ymm14\n",
                "vpmaddwd ymm14, ymm14, ymm8\n",
                "vpaddd ymm3, ymm3, ymm14\n",
                // Row 2
                "vpbroadcastd ymm11, dword ptr [{a0} + 2*{ks} + ",
                $aoff,
                "]\n",
                "vpbroadcastd ymm12, dword ptr [{aa0} + 2*{ks} + ",
                $aoff,
                "]\n",
                "vpsignb ymm13, ymm9, ymm11\n",
                "vpmaddubsw ymm13, ymm12, ymm13\n",
                "vpmaddwd ymm13, ymm13, ymm8\n",
                "vpaddd ymm4, ymm4, ymm13\n",
                "vpsignb ymm14, ymm10, ymm11\n",
                "vpmaddubsw ymm14, ymm12, ymm14\n",
                "vpmaddwd ymm14, ymm14, ymm8\n",
                "vpaddd ymm5, ymm5, ymm14\n",
                // Row 3
                "vpbroadcastd ymm11, dword ptr [{a3} + ",
                $aoff,
                "]\n",
                "vpbroadcastd ymm12, dword ptr [{aa3} + ",
                $aoff,
                "]\n",
                "vpsignb ymm13, ymm9, ymm11\n",
                "vpmaddubsw ymm13, ymm12, ymm13\n",
                "vpmaddwd ymm13, ymm13, ymm8\n",
                "vpaddd ymm6, ymm6, ymm13\n",
                "vpsignb ymm14, ymm10, ymm11\n",
                "vpmaddubsw ymm14, ymm12, ymm14\n",
                "vpmaddwd ymm14, ymm14, ymm8\n",
                "vpaddd ymm7, ymm7, ymm14\n",
            )
        };
    }
    macro_rules! body {
        ($pf:literal) => {
            // SAFETY: bounds verified by caller; all clobbered YMM regs declared.
            unsafe {
                asm!(
                    "vmovdqa ymm8, ymmword ptr [{acc}]",
                    "vpxor ymm0, ymm0, ymm0",
                    "vpxor ymm1, ymm1, ymm1",
                    "vpxor ymm2, ymm2, ymm2",
                    "vpxor ymm3, ymm3, ymm3",
                    "vpxor ymm4, ymm4, ymm4",
                    "vpxor ymm5, ymm5, ymm5",
                    "vpxor ymm6, ymm6, ymm6",
                    "vpxor ymm7, ymm7, ymm7",
                    "movzx {nb:e}, word ptr [{bad}]",
                    ".p2align 4",
                    "2:",
                    concat!($pf, " [{w} + 2048]"),
                    concat!($pf, " [{w} + 2112]"),
                    "cmp {pi}, {nb}",
                    "je 3f",
                    sp_row!("{a0} + 0", "{aa0} + 0", "{a0} + 4", "{aa0} + 4", "ymm0", "ymm1"),
                    sp_row!("{a0} + {ks} + 0", "{aa0} + {ks} + 0", "{a0} + {ks} + 4", "{aa0} + {ks} + 4", "ymm2", "ymm3"),
                    sp_row!("{a0} + 2*{ks} + 0", "{aa0} + 2*{ks} + 0", "{a0} + 2*{ks} + 4", "{aa0} + 2*{ks} + 4", "ymm4", "ymm5"),
                    sp_row!("{a3} + 0", "{aa3} + 0", "{a3} + 4", "{aa3} + 4", "ymm6", "ymm7"),
                    "4:",
                    "add {w}, 128",
                    "add {a0}, 8",
                    "add {aa0}, 8",
                    "add {a3}, 8",
                    "add {aa3}, 8",
                    "inc {pi}",
                    "cmp {pi}, {pairs}",
                    "jb 2b",
                    "vmovdqa ymmword ptr [{acc} + 0],   ymm0",
                    "vmovdqa ymmword ptr [{acc} + 32],  ymm1",
                    "vmovdqa ymmword ptr [{acc} + 64],  ymm2",
                    "vmovdqa ymmword ptr [{acc} + 96],  ymm3",
                    "vmovdqa ymmword ptr [{acc} + 128], ymm4",
                    "vmovdqa ymmword ptr [{acc} + 160], ymm5",
                    "vmovdqa ymmword ptr [{acc} + 192], ymm6",
                    "vmovdqa ymmword ptr [{acc} + 224], ymm7",
                    "jmp 9f",
                    "3:",
                    "add {bad}, 2",
                    "movzx {nb:e}, word ptr [{bad}]",
                    s_single_grp!("0", "32", "0"),
                    s_single_grp!("64", "96", "4"),
                    "jmp 4b",
                    "9:",
                    w = inout(reg) wp => _,
                    a0 = inout(reg) a0 => _,
                    aa0 = inout(reg) aa0 => _,
                    a3 = inout(reg) a3 => _,
                    aa3 = inout(reg) aa3 => _,
                    ks = in(reg) ks,
                    pi = inout(reg) 0usize => _,
                    pairs = in(reg) pairs,
                    bad = inout(reg) bad_ptr => _,
                    nb = out(reg) _,
                    acc = in(reg) acc,
                    out("ymm0") _, out("ymm1") _, out("ymm2") _, out("ymm3") _,
                    out("ymm4") _, out("ymm5") _, out("ymm6") _, out("ymm7") _,
                    out("ymm8") _, out("ymm9") _, out("ymm10") _, out("ymm11") _,
                    out("ymm12") _, out("ymm13") _, out("ymm14") _,
                    options(nostack),
                );
            }
        };
    }
    if NTA {
        body!("prefetchnta");
    } else {
        body!("prefetcht0");
    }
}

/// Zero-spill `M = 4` signed 2-group paired kernel (single `asm!` call/panel).
#[inline(never)]
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2,fma")]
pub(super) fn tile_signed_pair_4<const NTA: bool>(
    a: &[i8],
    a_abs: &[u8],
    kp: usize,
    panel: &[i8],
    sw: &[f32],
    wsum: &[f32],
    bias: &[f32],
    sa: &[f32],
    ca: &[f32],
    out: &mut [f32],
    ldo: usize,
    ncols: usize,
    bad_pairs: &[u16],
) {
    let groups = kp / 4;
    let pairs = groups / 2;
    assert!(kp.is_multiple_of(4) && a.len() >= 4 * kp && a_abs.len() >= 4 * kp);
    assert!(
        panel.len() >= kp * NR && pairs < u16::MAX as usize && bad_pairs.last() == Some(&u16::MAX)
    );

    let ones = _mm256_set1_epi16(1);
    let mut acc = Align32([[_mm256_setzero_si256(); 2]; 4]);
    let (ap, aap, wp) = (a.as_ptr(), a_abs.as_ptr(), panel.as_ptr());
    if pairs > 0 {
        acc.0[0][0] = ones;
        // SAFETY: 3 * kp + pairs * 8 <= 4 * kp <= a.len() = a_abs.len(), pairs * 128 <= kp * NR, sentinel asserted.
        unsafe {
            run_signed_pair_all_asm::<NTA>(
                wp,
                ap,
                aap,
                ap.add(3 * kp),
                aap.add(3 * kp),
                kp,
                pairs,
                bad_pairs.as_ptr(),
                acc.0.as_mut_ptr().cast::<__m256i>(),
            );
        }
    }

    if pairs * 2 < groups {
        let g = pairs * 2;
        let kj = g * 4;
        // SAFETY: (g + 1) * 64 <= kp * NR and 4 * g + 4 <= kp.
        unsafe {
            let w0 = _mm256_loadu_si256(wp.add(g * 64).cast());
            let w1 = _mm256_loadu_si256(wp.add(g * 64 + 32).cast());
            for (m, acc_m) in acc.0.iter_mut().enumerate() {
                let av =
                    _mm256_set1_epi32(std::ptr::read_unaligned(ap.add(m * kp + kj).cast::<i32>()));
                let aa =
                    _mm256_set1_epi32(std::ptr::read_unaligned(aap.add(m * kp + kj).cast::<i32>()));
                let p0 = _mm256_maddubs_epi16(aa, _mm256_sign_epi8(w0, av));
                let p1 = _mm256_maddubs_epi16(aa, _mm256_sign_epi8(w1, av));
                acc_m[0] = _mm256_add_epi32(acc_m[0], _mm256_madd_epi16(p0, ones));
                acc_m[1] = _mm256_add_epi32(acc_m[1], _mm256_madd_epi16(p1, ones));
            }
        }
    }

    epilogue_store(&acc.0, sw, wsum, bias, sa, ca, out, ldo, ncols);
}
