//! Single-call zero-spill AVX2 `M = 4` unsigned-offset 2-group paired microkernel (`NTA`).

use std::arch::asm;
use std::arch::x86_64::*;

use super::super::NR;
use super::int8_fast_prep::{
    accum_u_group_exact, epilogue_u_store, Align32, Avx2Consts, AVX2_CONSTS,
};

/// Single-call zero-spill AVX2 `M = 4` unsigned-offset 2-group paired kernel across ALL `pairs`
/// of a panel. `ymm0..ymm7` stay live in registers for the entire panel; rare bad pairs (encoded
/// as `(pi << 1) | needs_split`, terminated by `u16::MAX`) branch to out-of-line cold labels `3:`
/// (2 single groups) / `5:` (2 branchless `u_lo + u_hi` exact groups) inside the same `asm!` block.
///
/// # Safety
/// Caller must ensure AVX2+FMA is available, `acc` points to 32-byte aligned `[__m256i; 8]`,
/// `wp` has at least `pairs * 128` readable bytes, rows `0..3` at stride `ks` from `u0` (with
/// `u3 = u0 + 3*ks`) each have at least `pairs * 8` readable bytes, and `bad_ptr` points to a
/// sorted `u16` slice terminated by `u16::MAX` with every entry `< (pairs << 1)`.
#[inline(never)]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn run_u128_pair_all_asm<const NTA: bool>(
    wp: *const i8,
    u0: *const u8,
    u3: *const u8,
    ks: usize,
    pairs: usize,
    bad_ptr: *const u16,
    acc: *mut __m256i,
) {
    if pairs == 0 {
        return;
    }
    let consts_ptr = (&AVX2_CONSTS as *const Avx2Consts).cast::<u8>();
    macro_rules! u_single_row {
        ($uop:literal, $a0:literal, $a1:literal) => {
            concat!(
                "vpbroadcastd ymm12, dword ptr [",
                $uop,
                "]\n",
                "vpmaddubsw ymm13, ymm12, ymm8\n",
                "vpmaddwd ymm13, ymm13, ymmword ptr [{consts}]\n",
                "vpaddd ",
                $a0,
                ", ",
                $a0,
                ", ymm13\n",
                "vpmaddubsw ymm12, ymm12, ymm9\n",
                "vpmaddwd ymm12, ymm12, ymmword ptr [{consts}]\n",
                "vpaddd ",
                $a1,
                ", ",
                $a1,
                ", ymm12\n",
            )
        };
    }
    macro_rules! u_split_row {
        ($uop:literal, $a0:literal, $a1:literal) => {
            concat!(
                "vpbroadcastd ymm12, dword ptr [",
                $uop,
                "]\n",
                "vpsrlw ymm13, ymm12, 1\n",
                "vpand ymm13, ymm13, ymmword ptr [{consts} + 32]\n",
                "vpsubb ymm12, ymm12, ymm13\n",
                "vpmaddubsw ymm14, ymm13, ymm8\n",
                "vpmaddwd ymm14, ymm14, ymmword ptr [{consts}]\n",
                "vpaddd ",
                $a0,
                ", ",
                $a0,
                ", ymm14\n",
                "vpmaddubsw ymm15, ymm12, ymm8\n",
                "vpmaddwd ymm15, ymm15, ymmword ptr [{consts}]\n",
                "vpaddd ",
                $a0,
                ", ",
                $a0,
                ", ymm15\n",
                "vpmaddubsw ymm14, ymm13, ymm9\n",
                "vpmaddwd ymm14, ymm14, ymmword ptr [{consts}]\n",
                "vpaddd ",
                $a1,
                ", ",
                $a1,
                ", ymm14\n",
                "vpmaddubsw ymm15, ymm12, ymm9\n",
                "vpmaddwd ymm15, ymm15, ymmword ptr [{consts}]\n",
                "vpaddd ",
                $a1,
                ", ",
                $a1,
                ", ymm15\n",
            )
        };
    }
    macro_rules! body {
        ($pf:literal) => {
            // SAFETY: bounds verified by caller; all clobbered YMM regs declared.
            unsafe {
                asm!(
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
                    "mov {tmp}, {nb}",
                    "shr {tmp}, 1",
                    "cmp {pi}, {tmp}",
                    "je 3f",
                    "vmovdqu ymm8,  ymmword ptr [{w} + 0]",
                    "vmovdqu ymm9,  ymmword ptr [{w} + 32]",
                    "vmovdqu ymm10, ymmword ptr [{w} + 64]",
                    "vmovdqu ymm11, ymmword ptr [{w} + 96]",
                    // Row 0
                    "vpbroadcastd ymm12, dword ptr [{u0} + 0]",
                    "vpmaddubsw ymm13, ymm12, ymm8",
                    "vpmaddubsw ymm12, ymm12, ymm9",
                    "vpbroadcastd ymm14, dword ptr [{u0} + 4]",
                    "vpmaddubsw ymm15, ymm14, ymm10",
                    "vpaddw ymm13, ymm13, ymm15",
                    "vpmaddwd ymm13, ymm13, ymmword ptr [{consts}]",
                    "vpaddd ymm0, ymm0, ymm13",
                    "vpmaddubsw ymm14, ymm14, ymm11",
                    "vpaddw ymm12, ymm12, ymm14",
                    "vpmaddwd ymm12, ymm12, ymmword ptr [{consts}]",
                    "vpaddd ymm1, ymm1, ymm12",
                    // Row 1
                    "vpbroadcastd ymm12, dword ptr [{u0} + {ks} + 0]",
                    "vpmaddubsw ymm13, ymm12, ymm8",
                    "vpmaddubsw ymm12, ymm12, ymm9",
                    "vpbroadcastd ymm14, dword ptr [{u0} + {ks} + 4]",
                    "vpmaddubsw ymm15, ymm14, ymm10",
                    "vpaddw ymm13, ymm13, ymm15",
                    "vpmaddwd ymm13, ymm13, ymmword ptr [{consts}]",
                    "vpaddd ymm2, ymm2, ymm13",
                    "vpmaddubsw ymm14, ymm14, ymm11",
                    "vpaddw ymm12, ymm12, ymm14",
                    "vpmaddwd ymm12, ymm12, ymmword ptr [{consts}]",
                    "vpaddd ymm3, ymm3, ymm12",
                    // Row 2
                    "vpbroadcastd ymm12, dword ptr [{u0} + 2*{ks} + 0]",
                    "vpmaddubsw ymm13, ymm12, ymm8",
                    "vpmaddubsw ymm12, ymm12, ymm9",
                    "vpbroadcastd ymm14, dword ptr [{u0} + 2*{ks} + 4]",
                    "vpmaddubsw ymm15, ymm14, ymm10",
                    "vpaddw ymm13, ymm13, ymm15",
                    "vpmaddwd ymm13, ymm13, ymmword ptr [{consts}]",
                    "vpaddd ymm4, ymm4, ymm13",
                    "vpmaddubsw ymm14, ymm14, ymm11",
                    "vpaddw ymm12, ymm12, ymm14",
                    "vpmaddwd ymm12, ymm12, ymmword ptr [{consts}]",
                    "vpaddd ymm5, ymm5, ymm12",
                    // Row 3
                    "vpbroadcastd ymm12, dword ptr [{u3} + 0]",
                    "vpmaddubsw ymm13, ymm12, ymm8",
                    "vpmaddubsw ymm12, ymm12, ymm9",
                    "vpbroadcastd ymm14, dword ptr [{u3} + 4]",
                    "vpmaddubsw ymm15, ymm14, ymm10",
                    "vpaddw ymm13, ymm13, ymm15",
                    "vpmaddwd ymm13, ymm13, ymmword ptr [{consts}]",
                    "vpaddd ymm6, ymm6, ymm13",
                    "vpmaddubsw ymm14, ymm14, ymm11",
                    "vpaddw ymm12, ymm12, ymm14",
                    "vpmaddwd ymm12, ymm12, ymmword ptr [{consts}]",
                    "vpaddd ymm7, ymm7, ymm12",
                    "4:",
                    "add {w}, 128",
                    "add {u0}, 8",
                    "add {u3}, 8",
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
                    // Cold bad pair: 2 plain single groups (needs_split == 0)
                    "3:",
                    "add {bad}, 2",
                    "test {nb}, 1",
                    "jnz 5f",
                    "movzx {nb:e}, word ptr [{bad}]",
                    "vmovdqu ymm8, ymmword ptr [{w} + 0]",
                    "vmovdqu ymm9, ymmword ptr [{w} + 32]",
                    u_single_row!("{u0} + 0", "ymm0", "ymm1"),
                    u_single_row!("{u0} + {ks} + 0", "ymm2", "ymm3"),
                    u_single_row!("{u0} + 2*{ks} + 0", "ymm4", "ymm5"),
                    u_single_row!("{u3} + 0", "ymm6", "ymm7"),
                    "vmovdqu ymm8, ymmword ptr [{w} + 64]",
                    "vmovdqu ymm9, ymmword ptr [{w} + 96]",
                    u_single_row!("{u0} + 4", "ymm0", "ymm1"),
                    u_single_row!("{u0} + {ks} + 4", "ymm2", "ymm3"),
                    u_single_row!("{u0} + 2*{ks} + 4", "ymm4", "ymm5"),
                    u_single_row!("{u3} + 4", "ymm6", "ymm7"),
                    "jmp 4b",
                    // Cold split pair: 2 exact u_lo + u_hi groups (needs_split == 1)
                    "5:",
                    "movzx {nb:e}, word ptr [{bad}]",
                    "vmovdqu ymm8, ymmword ptr [{w} + 0]",
                    "vmovdqu ymm9, ymmword ptr [{w} + 32]",
                    u_split_row!("{u0} + 0", "ymm0", "ymm1"),
                    u_split_row!("{u0} + {ks} + 0", "ymm2", "ymm3"),
                    u_split_row!("{u0} + 2*{ks} + 0", "ymm4", "ymm5"),
                    u_split_row!("{u3} + 0", "ymm6", "ymm7"),
                    "vmovdqu ymm8, ymmword ptr [{w} + 64]",
                    "vmovdqu ymm9, ymmword ptr [{w} + 96]",
                    u_split_row!("{u0} + 4", "ymm0", "ymm1"),
                    u_split_row!("{u0} + {ks} + 4", "ymm2", "ymm3"),
                    u_split_row!("{u0} + 2*{ks} + 4", "ymm4", "ymm5"),
                    u_split_row!("{u3} + 4", "ymm6", "ymm7"),
                    "jmp 4b",
                    "9:",
                    w = inout(reg) wp => _,
                    u0 = inout(reg) u0 => _,
                    u3 = inout(reg) u3 => _,
                    ks = in(reg) ks,
                    pi = inout(reg) 0usize => _,
                    pairs = in(reg) pairs,
                    bad = inout(reg) bad_ptr => _,
                    nb = out(reg) _,
                    tmp = out(reg) _,
                    acc = in(reg) acc,
                    consts = in(reg) consts_ptr,
                    out("ymm0") _, out("ymm1") _, out("ymm2") _, out("ymm3") _,
                    out("ymm4") _, out("ymm5") _, out("ymm6") _, out("ymm7") _,
                    out("ymm8") _, out("ymm9") _, out("ymm10") _, out("ymm11") _,
                    out("ymm12") _, out("ymm13") _, out("ymm14") _, out("ymm15") _,
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

/// Zero-spill `M = 4` streaming (`NTA`) unsigned-offset paired kernel (single `asm!` call/panel).
#[inline(never)]
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2,fma")]
pub(super) fn tile_u_pair_4<const NEG: bool>(
    u: &[u8],
    kp: usize,
    panel: &[i8],
    qsum: &[i32],
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
    assert!(kp.is_multiple_of(4) && u.len() >= 4 * kp && panel.len() >= kp * NR);
    assert!(pairs < 0x7FFF && bad_pairs.last() == Some(&u16::MAX));

    let mut acc = Align32([[_mm256_setzero_si256(); 2]; 4]);
    let (up, wp) = (u.as_ptr(), panel.as_ptr());
    if pairs > 0 {
        // SAFETY: 3 * kp + pairs * 8 <= 4 * kp <= u.len(), pairs * 128 <= kp * NR, sentinel asserted.
        unsafe {
            run_u128_pair_all_asm::<true>(
                wp,
                up,
                up.add(3 * kp),
                kp,
                pairs,
                bad_pairs.as_ptr(),
                acc.0.as_mut_ptr().cast::<__m256i>(),
            );
        }
    }

    if pairs * 2 < groups {
        let ones = _mm256_set1_epi16(1);
        let g = pairs * 2;
        // SAFETY: (g + 1) * 64 <= kp * NR and 4 * g + 4 <= kp.
        unsafe {
            let w0 = _mm256_loadu_si256(wp.add(g * 64).cast());
            let w1 = _mm256_loadu_si256(wp.add(g * 64 + 32).cast());
            let k0 = g * 4;
            let a = &mut acc.0;
            accum_u_group_exact(up.add(k0), w0, w1, ones, &mut a[0]);
            accum_u_group_exact(up.add(kp + k0), w0, w1, ones, &mut a[1]);
            accum_u_group_exact(up.add(2 * kp + k0), w0, w1, ones, &mut a[2]);
            accum_u_group_exact(up.add(3 * kp + k0), w0, w1, ones, &mut a[3]);
        }
    }

    epilogue_u_store::<NEG>(acc.0, qsum, sw, wsum, bias, sa, ca, out, ldo, ncols);
}
