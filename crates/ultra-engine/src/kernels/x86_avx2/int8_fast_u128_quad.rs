//! Single-call zero-spill AVX2 `M = 4` unsigned-offset 4-group quad microkernel (`T0`).

use std::arch::asm;
use std::arch::x86_64::*;

use super::super::NR;
use super::int8_fast_prep::{
    accum_u_group_exact, epilogue_u_store, Align32, Avx2Consts, AVX2_CONSTS,
};

/// Single-call zero-spill AVX2 `M = 4` unsigned-offset 4-group quad kernel across ALL `quads`
/// of a panel (`rows > 4`, `T0` prefetch). `ymm0..ymm7` stay live in registers for the entire
/// panel; rare bad quads branch to cold labels `3:` (4 plain groups) / `5:` (4 exact split groups).
///
/// # Safety
/// Caller must ensure AVX2+FMA is available, `acc` points to 32-byte aligned `[__m256i; 8]`,
/// `wp` has at least `quads * 256` readable bytes, rows `0..3` at stride `ks` from `u0` (with
/// `u3 = u0 + 3*ks`) each have at least `quads * 16` readable bytes, and `bad_ptr` points to a
/// sorted `u16` slice terminated by `u16::MAX` with every entry `< (quads << 1)`.
#[inline(never)]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn run_u128_quad_all_asm(
    wp: *const i8,
    u0: *const u8,
    u3: *const u8,
    ks: usize,
    quads: usize,
    bad_ptr: *const u16,
    acc: *mut __m256i,
) {
    if quads == 0 {
        return;
    }
    let consts_ptr = (&AVX2_CONSTS as *const Avx2Consts).cast::<u8>();
    macro_rules! uq_row {
        ($u0op:literal, $u1op:literal, $u2op:literal, $u3op:literal, $a0:literal, $a1:literal) => {
            concat!(
                "vpbroadcastd ymm11, dword ptr [",
                $u0op,
                "]\n",
                "vpmaddubsw ymm9,  ymm11, ymmword ptr [{w} + 0]\n",
                "vpmaddubsw ymm10, ymm11, ymmword ptr [{w} + 32]\n",
                "vpbroadcastd ymm11, dword ptr [",
                $u1op,
                "]\n",
                "vpmaddubsw ymm12, ymm11, ymmword ptr [{w} + 64]\n",
                "vpaddw ymm9, ymm9, ymm12\n",
                "vpmaddubsw ymm13, ymm11, ymmword ptr [{w} + 96]\n",
                "vpaddw ymm10, ymm10, ymm13\n",
                "vpbroadcastd ymm11, dword ptr [",
                $u2op,
                "]\n",
                "vpmaddubsw ymm12, ymm11, ymmword ptr [{w} + 128]\n",
                "vpaddw ymm9, ymm9, ymm12\n",
                "vpmaddubsw ymm13, ymm11, ymmword ptr [{w} + 160]\n",
                "vpaddw ymm10, ymm10, ymm13\n",
                "vpbroadcastd ymm11, dword ptr [",
                $u3op,
                "]\n",
                "vpmaddubsw ymm12, ymm11, ymmword ptr [{w} + 192]\n",
                "vpaddw ymm9, ymm9, ymm12\n",
                "vpmaddubsw ymm13, ymm11, ymmword ptr [{w} + 224]\n",
                "vpaddw ymm10, ymm10, ymm13\n",
                "vpmaddwd ymm9, ymm9, ymm8\n",
                "vpaddd ",
                $a0,
                ", ",
                $a0,
                ", ymm9\n",
                "vpmaddwd ymm10, ymm10, ymm8\n",
                "vpaddd ",
                $a1,
                ", ",
                $a1,
                ", ymm10\n",
            )
        };
    }
    macro_rules! uq_plain_grp {
        ($woff0:literal, $woff1:literal, $uoff:literal) => {
            concat!(
                "vmovdqu ymm9,  ymmword ptr [{w} + ",
                $woff0,
                "]\n",
                "vmovdqu ymm10, ymmword ptr [{w} + ",
                $woff1,
                "]\n",
                "vpbroadcastd ymm11, dword ptr [{u0} + ",
                $uoff,
                "]\n",
                "vpmaddubsw ymm12, ymm11, ymm9\n",
                "vpmaddwd ymm12, ymm12, ymm8\n",
                "vpaddd ymm0, ymm0, ymm12\n",
                "vpmaddubsw ymm11, ymm11, ymm10\n",
                "vpmaddwd ymm11, ymm11, ymm8\n",
                "vpaddd ymm1, ymm1, ymm11\n",
                "vpbroadcastd ymm11, dword ptr [{u0} + {ks} + ",
                $uoff,
                "]\n",
                "vpmaddubsw ymm12, ymm11, ymm9\n",
                "vpmaddwd ymm12, ymm12, ymm8\n",
                "vpaddd ymm2, ymm2, ymm12\n",
                "vpmaddubsw ymm11, ymm11, ymm10\n",
                "vpmaddwd ymm11, ymm11, ymm8\n",
                "vpaddd ymm3, ymm3, ymm11\n",
                "vpbroadcastd ymm11, dword ptr [{u0} + 2*{ks} + ",
                $uoff,
                "]\n",
                "vpmaddubsw ymm12, ymm11, ymm9\n",
                "vpmaddwd ymm12, ymm12, ymm8\n",
                "vpaddd ymm4, ymm4, ymm12\n",
                "vpmaddubsw ymm11, ymm11, ymm10\n",
                "vpmaddwd ymm11, ymm11, ymm8\n",
                "vpaddd ymm5, ymm5, ymm11\n",
                "vpbroadcastd ymm11, dword ptr [{u3} + ",
                $uoff,
                "]\n",
                "vpmaddubsw ymm12, ymm11, ymm9\n",
                "vpmaddwd ymm12, ymm12, ymm8\n",
                "vpaddd ymm6, ymm6, ymm12\n",
                "vpmaddubsw ymm11, ymm11, ymm10\n",
                "vpmaddwd ymm11, ymm11, ymm8\n",
                "vpaddd ymm7, ymm7, ymm11\n",
            )
        };
    }
    macro_rules! uq_split_row {
        ($base:literal, $uoff:literal, $a0:literal, $a1:literal) => {
            concat!(
                "vpbroadcastd ymm11, dword ptr [",
                $base,
                $uoff,
                "]\n",
                "vpsrlw ymm12, ymm11, 1\n",
                "vpand ymm12, ymm12, ymmword ptr [{consts} + 32]\n",
                "vpsubb ymm11, ymm11, ymm12\n",
                "vpmaddubsw ymm13, ymm12, ymm9\n",
                "vpmaddwd ymm13, ymm13, ymm8\n",
                "vpaddd ",
                $a0,
                ", ",
                $a0,
                ", ymm13\n",
                "vpmaddubsw ymm14, ymm11, ymm9\n",
                "vpmaddwd ymm14, ymm14, ymm8\n",
                "vpaddd ",
                $a0,
                ", ",
                $a0,
                ", ymm14\n",
                "vpmaddubsw ymm13, ymm12, ymm10\n",
                "vpmaddwd ymm13, ymm13, ymm8\n",
                "vpaddd ",
                $a1,
                ", ",
                $a1,
                ", ymm13\n",
                "vpmaddubsw ymm14, ymm11, ymm10\n",
                "vpmaddwd ymm14, ymm14, ymm8\n",
                "vpaddd ",
                $a1,
                ", ",
                $a1,
                ", ymm14\n",
            )
        };
    }
    macro_rules! uq_split_grp {
        ($woff0:literal, $woff1:literal, $uoff:literal) => {
            concat!(
                "vmovdqu ymm9,  ymmword ptr [{w} + ",
                $woff0,
                "]\n",
                "vmovdqu ymm10, ymmword ptr [{w} + ",
                $woff1,
                "]\n",
                uq_split_row!("{u0} + ", $uoff, "ymm0", "ymm1"),
                uq_split_row!("{u0} + {ks} + ", $uoff, "ymm2", "ymm3"),
                uq_split_row!("{u0} + 2*{ks} + ", $uoff, "ymm4", "ymm5"),
                uq_split_row!("{u3} + ", $uoff, "ymm6", "ymm7"),
            )
        };
    }
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
            "vmovdqa ymm8, ymmword ptr [{consts}]",
            "movzx {nb:e}, word ptr [{bad}]",
            ".p2align 4",
            "2:",
            "prefetcht0 [{w} + 2048]",
            "prefetcht0 [{w} + 2112]",
            "prefetcht0 [{w} + 2176]",
            "prefetcht0 [{w} + 2240]",
            "mov {tmp}, {nb}",
            "shr {tmp}, 1",
            "cmp {pi}, {tmp}",
            "je 3f",
            uq_row!("{u0} + 0", "{u0} + 4", "{u0} + 8", "{u0} + 12", "ymm0", "ymm1"),
            uq_row!("{u0} + {ks} + 0", "{u0} + {ks} + 4", "{u0} + {ks} + 8", "{u0} + {ks} + 12", "ymm2", "ymm3"),
            uq_row!("{u0} + 2*{ks} + 0", "{u0} + 2*{ks} + 4", "{u0} + 2*{ks} + 8", "{u0} + 2*{ks} + 12", "ymm4", "ymm5"),
            uq_row!("{u3} + 0", "{u3} + 4", "{u3} + 8", "{u3} + 12", "ymm6", "ymm7"),
            "4:",
            "add {w}, 256",
            "add {u0}, 16",
            "add {u3}, 16",
            "inc {pi}",
            "cmp {pi}, {quads}",
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
            "test {nb}, 1",
            "jnz 5f",
            "movzx {nb:e}, word ptr [{bad}]",
            uq_plain_grp!("0", "32", "0"),
            uq_plain_grp!("64", "96", "4"),
            uq_plain_grp!("128", "160", "8"),
            uq_plain_grp!("192", "224", "12"),
            "jmp 4b",
            "5:",
            "movzx {nb:e}, word ptr [{bad}]",
            uq_split_grp!("0", "32", "0"),
            uq_split_grp!("64", "96", "4"),
            uq_split_grp!("128", "160", "8"),
            uq_split_grp!("192", "224", "12"),
            "jmp 4b",
            "9:",
            w = inout(reg) wp => _,
            u0 = inout(reg) u0 => _,
            u3 = inout(reg) u3 => _,
            ks = in(reg) ks,
            pi = inout(reg) 0usize => _,
            quads = in(reg) quads,
            bad = inout(reg) bad_ptr => _,
            nb = out(reg) _,
            tmp = out(reg) _,
            acc = in(reg) acc,
            consts = in(reg) consts_ptr,
            out("ymm0") _, out("ymm1") _, out("ymm2") _, out("ymm3") _,
            out("ymm4") _, out("ymm5") _, out("ymm6") _, out("ymm7") _,
            out("ymm8") _, out("ymm9") _, out("ymm10") _, out("ymm11") _,
            out("ymm12") _, out("ymm13") _, out("ymm14") _,
            options(nostack),
        );
    }
}

/// Zero-spill `M = 4` compute-bound (`!NTA`, e.g. `commit c=16`) unsigned-offset quad kernel
/// (single `asm!` call/panel).
#[inline(never)]
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2,fma")]
pub(super) fn tile_u_quad_4<const NEG: bool>(
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
    bad_quads: &[u16],
) {
    let groups = kp / 4;
    let quads = groups / 4;
    assert!(kp.is_multiple_of(4) && u.len() >= 4 * kp && panel.len() >= kp * NR);
    assert!(quads < 0x7FFF && bad_quads.last() == Some(&u16::MAX));

    let mut acc = Align32([[_mm256_setzero_si256(); 2]; 4]);
    let (up, wp) = (u.as_ptr(), panel.as_ptr());
    if quads > 0 {
        // SAFETY: 3 * kp + quads * 16 <= 4 * kp <= u.len(), quads * 256 <= kp * NR, sentinel asserted.
        unsafe {
            run_u128_quad_all_asm(
                wp,
                up,
                up.add(3 * kp),
                kp,
                quads,
                bad_quads.as_ptr(),
                acc.0.as_mut_ptr().cast::<__m256i>(),
            );
        }
    }

    if quads * 4 < groups {
        let ones = _mm256_set1_epi16(1);
        for g in (quads * 4)..groups {
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
    }

    epilogue_u_store::<NEG>(acc.0, qsum, sw, wsum, bias, sa, ca, out, ldo, ncols);
}
