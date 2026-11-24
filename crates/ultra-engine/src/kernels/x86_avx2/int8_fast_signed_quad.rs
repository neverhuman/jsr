//! Single-call zero-spill AVX2 `M = 4` signed 4-group quad microkernel (`T0` / hot-L1 no-prefetch).

use std::arch::asm;
use std::arch::x86_64::*;

use super::super::NR;
use super::int8_fast_prep::{epilogue_store_preloaded, Align32, PanelEpilogue};

/// Single-call zero-spill AVX2 `M = 4` signed 4-group quad kernel across ALL `quads` of a panel
/// (`rows > 4`, amortised weight reuse). `ymm0..ymm7` stay live in registers for the entire panel;
/// rare bad quads branch to cold labels `3:` (2 safe pairs) / `5:` (4 plain single groups).
///
/// `PREFETCH == true` streams `prefetcht0` on `ti == 0`; `PREFETCH == false` skips redundant
/// prefetches on `ti > 0` while the panel is already hot in L1d.
///
/// # Safety
/// Caller must ensure AVX2+FMA is available, `acc` points to 32-byte aligned `[__m256i; 8]` with
/// `ONES_I16` in `acc[0]`, `wp` has at least `quads * 256` readable bytes, rows `0..3` at stride
/// `ks` from `a0`/`aa0` (with `a3 = a0 + 3*ks`, `aa3 = aa0 + 3*ks`) each have at least `quads * 16`
/// readable bytes, and `bad_ptr` is sorted and terminated by `u16::MAX` with entries `< (quads << 1)`.
#[inline(never)]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn run_signed_quad_all_asm<const PREFETCH: bool>(
    wp: *const i8,
    a0: *const i8,
    aa0: *const u8,
    a3: *const i8,
    aa3: *const u8,
    ks: usize,
    quads: usize,
    bad_ptr: *const u16,
    acc: *mut __m256i,
) {
    if quads == 0 {
        return;
    }
    let quads_end = quads * 2 + 1;
    macro_rules! sq_grp_add {
        ($a_op:literal, $aa_op:literal, $w0:literal, $w1:literal) => {
            concat!(
                "vpbroadcastd ymm11, dword ptr [",
                $a_op,
                "]\n",
                "vpbroadcastd ymm12, dword ptr [",
                $aa_op,
                "]\n",
                "vmovdqu ymm13, ymmword ptr [{w} + ",
                $w0,
                "]\n",
                "vmovdqu ymm14, ymmword ptr [{w} + ",
                $w1,
                "]\n",
                "vpsignb ymm13, ymm13, ymm11\n",
                "vpmaddubsw ymm13, ymm12, ymm13\n",
                "vpaddw ymm9, ymm9, ymm13\n",
                "vpsignb ymm14, ymm14, ymm11\n",
                "vpmaddubsw ymm14, ymm12, ymm14\n",
                "vpaddw ymm10, ymm10, ymm14\n",
            )
        };
    }
    macro_rules! sq_row {
        (
            $a0:literal, $aa0:literal,
            $a1:literal, $aa1:literal,
            $a2:literal, $aa2:literal,
            $a3:literal, $aa3:literal,
            $c0:literal, $c1:literal
        ) => {
            concat!(
                "vpbroadcastd ymm11, dword ptr [",
                $a0,
                "]\n",
                "vpbroadcastd ymm12, dword ptr [",
                $aa0,
                "]\n",
                "vmovdqu ymm13, ymmword ptr [{w} + 0]\n",
                "vmovdqu ymm14, ymmword ptr [{w} + 32]\n",
                "vpsignb ymm13, ymm13, ymm11\n",
                "vpmaddubsw ymm9, ymm12, ymm13\n",
                "vpsignb ymm14, ymm14, ymm11\n",
                "vpmaddubsw ymm10, ymm12, ymm14\n",
                sq_grp_add!($a1, $aa1, "64", "96"),
                sq_grp_add!($a2, $aa2, "128", "160"),
                sq_grp_add!($a3, $aa3, "192", "224"),
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
    macro_rules! sq_pair_row {
        ($a0:literal, $aa0:literal, $a1:literal, $aa1:literal, $w0:literal, $w1:literal, $w2:literal, $w3:literal, $c0:literal, $c1:literal) => {
            concat!(
                "vpbroadcastd ymm11, dword ptr [",
                $a0,
                "]\n",
                "vpbroadcastd ymm12, dword ptr [",
                $aa0,
                "]\n",
                "vmovdqu ymm13, ymmword ptr [{w} + ",
                $w0,
                "]\n",
                "vmovdqu ymm14, ymmword ptr [{w} + ",
                $w1,
                "]\n",
                "vpsignb ymm13, ymm13, ymm11\n",
                "vpmaddubsw ymm9, ymm12, ymm13\n",
                "vpsignb ymm14, ymm14, ymm11\n",
                "vpmaddubsw ymm10, ymm12, ymm14\n",
                sq_grp_add!($a1, $aa1, $w2, $w3),
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
    macro_rules! sq_single_grp {
        ($woff0:literal, $woff1:literal, $aoff:literal) => {
            concat!(
                "vmovdqu ymm9,  ymmword ptr [{w} + ",
                $woff0,
                "]\n",
                "vmovdqu ymm10, ymmword ptr [{w} + ",
                $woff1,
                "]\n",
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
        ($pf_block:literal) => {
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
                    $pf_block,
                    "cmp {pi}, {nb}",
                    "jae 3f",
                    sq_row!(
                        "{a0} + 0", "{aa0} + 0",
                        "{a0} + 4", "{aa0} + 4",
                        "{a0} + 8", "{aa0} + 8",
                        "{a0} + 12", "{aa0} + 12",
                        "ymm0", "ymm1"
                    ),
                    sq_row!(
                        "{a0} + {ks} + 0", "{aa0} + {ks} + 0",
                        "{a0} + {ks} + 4", "{aa0} + {ks} + 4",
                        "{a0} + {ks} + 8", "{aa0} + {ks} + 8",
                        "{a0} + {ks} + 12", "{aa0} + {ks} + 12",
                        "ymm2", "ymm3"
                    ),
                    sq_row!(
                        "{a0} + 2*{ks} + 0", "{aa0} + 2*{ks} + 0",
                        "{a0} + 2*{ks} + 4", "{aa0} + 2*{ks} + 4",
                        "{a0} + 2*{ks} + 8", "{aa0} + 2*{ks} + 8",
                        "{a0} + 2*{ks} + 12", "{aa0} + 2*{ks} + 12",
                        "ymm4", "ymm5"
                    ),
                    sq_row!(
                        "{a3} + 0", "{aa3} + 0",
                        "{a3} + 4", "{aa3} + 4",
                        "{a3} + 8", "{aa3} + 8",
                        "{a3} + 12", "{aa3} + 12",
                        "ymm6", "ymm7"
                    ),
                    "4:",
                    "add {w}, 256",
                    "add {a0}, 16",
                    "add {aa0}, 16",
                    "add {a3}, 16",
                    "add {aa3}, 16",
                    "add {pi}, 2",
                    "cmp {pi}, {qend}",
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
                    // Cold bad quad: bit0 == 0 -> 2 safe pairs; bit0 == 1 -> 4 plain single groups.
                    "3:",
                    "add {bad}, 2",
                    "test {nb}, 1",
                    "jnz 5f",
                    "movzx {nb:e}, word ptr [{bad}]",
                    sq_pair_row!("{a0} + 0", "{aa0} + 0", "{a0} + 4", "{aa0} + 4", "0", "32", "64", "96", "ymm0", "ymm1"),
                    sq_pair_row!("{a0} + {ks} + 0", "{aa0} + {ks} + 0", "{a0} + {ks} + 4", "{aa0} + {ks} + 4", "0", "32", "64", "96", "ymm2", "ymm3"),
                    sq_pair_row!("{a0} + 2*{ks} + 0", "{aa0} + 2*{ks} + 0", "{a0} + 2*{ks} + 4", "{aa0} + 2*{ks} + 4", "0", "32", "64", "96", "ymm4", "ymm5"),
                    sq_pair_row!("{a3} + 0", "{aa3} + 0", "{a3} + 4", "{aa3} + 4", "0", "32", "64", "96", "ymm6", "ymm7"),
                    sq_pair_row!("{a0} + 8", "{aa0} + 8", "{a0} + 12", "{aa0} + 12", "128", "160", "192", "224", "ymm0", "ymm1"),
                    sq_pair_row!("{a0} + {ks} + 8", "{aa0} + {ks} + 8", "{a0} + {ks} + 12", "{aa0} + {ks} + 12", "128", "160", "192", "224", "ymm2", "ymm3"),
                    sq_pair_row!("{a0} + 2*{ks} + 8", "{aa0} + 2*{ks} + 8", "{a0} + 2*{ks} + 12", "{aa0} + 2*{ks} + 12", "128", "160", "192", "224", "ymm4", "ymm5"),
                    sq_pair_row!("{a3} + 8", "{aa3} + 8", "{a3} + 12", "{aa3} + 12", "128", "160", "192", "224", "ymm6", "ymm7"),
                    "jmp 4b",
                    "5:",
                    "movzx {nb:e}, word ptr [{bad}]",
                    sq_single_grp!("0", "32", "0"),
                    sq_single_grp!("64", "96", "4"),
                    sq_single_grp!("128", "160", "8"),
                    sq_single_grp!("192", "224", "12"),
                    "jmp 4b",
                    "9:",
                    w = inout(reg) wp => _,
                    a0 = inout(reg) a0 => _,
                    aa0 = inout(reg) aa0 => _,
                    a3 = inout(reg) a3 => _,
                    aa3 = inout(reg) aa3 => _,
                    ks = in(reg) ks,
                    pi = inout(reg) 1usize => _,
                    qend = in(reg) quads_end,
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
    if PREFETCH {
        body!("prefetcht0 [{w} + 2048]\nprefetcht0 [{w} + 2112]\nprefetcht0 [{w} + 2176]\nprefetcht0 [{w} + 2240]\n");
    } else {
        body!("");
    }
}

/// Zero-spill `M = 4` compute-bound (`!NTA`, amortised weight reuse) signed 4-group quad kernel.
#[inline(never)]
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2,fma")]
pub(super) fn tile_signed_quad_4<const PREFETCH: bool>(
    a: &[i8],
    a_abs: &[u8],
    kp: usize,
    panel: &[i8],
    ep: &PanelEpilogue,
    sa: &[f32],
    ca: &[f32],
    out: &mut [f32],
    ldo: usize,
    ncols: usize,
    bad_quads: &[u16],
) {
    let groups = kp / 4;
    let quads = groups / 4;
    assert!(kp.is_multiple_of(4) && a.len() >= 4 * kp && a_abs.len() >= 4 * kp);
    assert!(panel.len() >= kp * NR && quads < 0x7FFF && bad_quads.last() == Some(&u16::MAX));

    let ones = _mm256_set1_epi16(1);
    let mut acc = Align32([[_mm256_setzero_si256(); 2]; 4]);
    let (ap, aap, wp) = (a.as_ptr(), a_abs.as_ptr(), panel.as_ptr());
    if quads > 0 {
        acc.0[0][0] = ones;
        // SAFETY: 3 * kp + quads * 16 <= 4 * kp <= a.len() = a_abs.len(), quads * 256 <= kp * NR, sentinel asserted.
        unsafe {
            run_signed_quad_all_asm::<PREFETCH>(
                wp,
                ap,
                aap,
                ap.add(3 * kp),
                aap.add(3 * kp),
                kp,
                quads,
                bad_quads.as_ptr(),
                acc.0.as_mut_ptr().cast::<__m256i>(),
            );
        }
    }

    if quads * 4 < groups {
        for g in (quads * 4)..groups {
            let kj = g * 4;
            // SAFETY: (g + 1) * 64 <= kp * NR and 4 * g + 4 <= kp.
            unsafe {
                let w0 = _mm256_loadu_si256(wp.add(g * 64).cast());
                let w1 = _mm256_loadu_si256(wp.add(g * 64 + 32).cast());
                for (m, acc_m) in acc.0.iter_mut().enumerate() {
                    let av = _mm256_set1_epi32(std::ptr::read_unaligned(
                        ap.add(m * kp + kj).cast::<i32>(),
                    ));
                    let aa = _mm256_set1_epi32(std::ptr::read_unaligned(
                        aap.add(m * kp + kj).cast::<i32>(),
                    ));
                    let p0 = _mm256_maddubs_epi16(aa, _mm256_sign_epi8(w0, av));
                    let p1 = _mm256_maddubs_epi16(aa, _mm256_sign_epi8(w1, av));
                    acc_m[0] = _mm256_add_epi32(acc_m[0], _mm256_madd_epi16(p0, ones));
                    acc_m[1] = _mm256_add_epi32(acc_m[1], _mm256_madd_epi16(p1, ones));
                }
            }
        }
    }

    epilogue_store_preloaded(&acc.0, ep, sa, ca, out, ldo, ncols);
}
