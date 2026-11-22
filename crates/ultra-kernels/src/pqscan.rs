//! Learned gain–shape 4-bit codebook-scan GEMV (`pqscan`, Wave H24) and block-16 FWHT.
//!
//! Implements the exact integer codebook-scan contract (`ULTRA-SU §6.2`, Appendix B):
//! - Input activations are per-row signed `i8` codes in `-127..=127` (`-128` refused).
//! - Each input vector of width `din` is split into `C = din / s` subspaces (`s in {2, 4, 8}`,
//!   `C` even) across `M in {1, 2}` additive stages.
//! - Each subspace has 16 signed `i8` codewords (`|cb| <= 127`). Per row, exact `i32` table
//!   values `v[m][c][j] = sum_i a[c*s + i] * cb[m][c][j][i]` are shifted right by the minimal
//!   `sh >= 0` such that `(max - min) >> sh <= 127`, producing 7-bit unsigned LUT entries
//!   `e[m][c][j] in 0..=127`.
//! - Two adjacent subspaces' LUT hits sum in `u8` (`<= 254`) and accumulate in `u16` for at most
//!   [`PQSCAN_FLUSH_PAIRS`] = 256 pairs (`256 * 254 = 65,024 < 65,535`) before widening to `i32`.
//! - Output is `acc[o] = (S[o] << sh) + sum_{m,c} mn[m][c] + ((M * C * ((1 << sh) - 1)) >> 1)`,
//!   bit-identical across scalar, x86 AVX2 (`vpshufb`), and ARM NEON (`vqtbl1q_u8`).

use crate::{Backend, KernelError};
use ultra_core::CoreError;

/// Output rows per interleaved SIMD block (32 bytes = one YMM or two NEON Q registers).
pub const PQSCAN_BLOCK: usize = 32;
/// Maximum subspace pairs summed in `u16` before widening to `i32` (`256 * 254 = 65,024 < 65,535`).
pub const PQSCAN_FLUSH_PAIRS: usize = 256;
/// Codewords per subspace (4-bit nibble index).
pub const PQSCAN_K: usize = 16;

/// Packed 4-bit codebook-scan weight matrix (`s in {2, 4, 8}`, `M in {1, 2}` additive stages).
#[derive(Debug, Clone)]
pub struct CodebookMatrix {
    rows: usize,
    din: usize,
    subspace_s: usize,
    stages_m: usize,
    /// `[stage][subspace][16][s]` signed i8 codewords (`|v| <= 127`).
    codebooks: Vec<i8>,
    /// `[blk][stage][pair][32]` nibble-packed codes (low = even subspace, high = odd subspace).
    codes: Vec<u8>,
}

/// Prebuilt per-frame 7-bit lookup tables for [`CodebookMatrix`] (reusable workspace).
#[derive(Debug, Clone, Default)]
pub struct CodebookTables {
    /// Flattened `[frame][stage][subspace][16]` unsigned 7-bit entries (`0..=127`).
    pub entries: Vec<[u8; PQSCAN_K]>,
    /// Per-frame right shift `sh >= 0`.
    pub shifts: Vec<u32>,
    /// Per-frame integer offset `sum_{m,c} mn[m][c] + ((M * C * ((1 << sh) - 1)) >> 1)`.
    pub offsets: Vec<i32>,
    /// Optional per-frame per-pair activity mask (false iff all `2 * s` activations in pair are 0).
    pub pair_nonzero: Vec<u8>,
    /// Reusable scratch buffer for exact i32 table values (`[stage * c][16]`).
    pub(crate) raw_scratch: Vec<[i32; PQSCAN_K]>,
    /// Reusable scratch buffer for per-subspace minimums (`[stage * c]`).
    pub(crate) mins_scratch: Vec<i32>,
    pub frames: usize,
    pub stages_m: usize,
    pub subspaces_c: usize,
}

impl CodebookTables {
    /// Resize internal buffers once; subsequent calls with the same shape allocate 0 bytes.
    pub fn resize(&mut self, frames: usize, stages_m: usize, subspaces_c: usize) {
        let mc = stages_m * subspaces_c;
        let pairs = subspaces_c / 2;
        self.entries.resize(frames * mc, [0u8; PQSCAN_K]);
        self.shifts.resize(frames, 0);
        self.offsets.resize(frames, 0);
        self.pair_nonzero.resize(frames * pairs, 1);
        self.raw_scratch.resize(mc, [0i32; PQSCAN_K]);
        self.mins_scratch.resize(mc, 0);
        self.frames = frames;
        self.stages_m = stages_m;
        self.subspaces_c = subspaces_c;
    }
}

impl CodebookMatrix {
    /// Construct and validate a packed codebook matrix from unpacked 4-bit codes (`0..16`)
    /// laid out `[stage][row][subspace]` and codebooks `[stage][subspace][16][s]`.
    pub fn from_parts(
        rows: usize,
        din: usize,
        subspace_s: usize,
        stages_m: usize,
        codebooks: Vec<i8>,
        row_codes: &[u8],
    ) -> Result<Self, KernelError> {
        if rows == 0 || din == 0 {
            return Err(KernelError::Shape("pqscan: empty matrix"));
        }
        if !matches!(subspace_s, 2 | 4 | 8) {
            return Err(KernelError::Shape("pqscan: subspace_s must be 2, 4, or 8"));
        }
        if !matches!(stages_m, 1 | 2) {
            return Err(KernelError::Shape("pqscan: stages_m must be 1 or 2"));
        }
        if !din.is_multiple_of(subspace_s * 2) {
            return Err(KernelError::Shape(
                "pqscan: din must be a multiple of 2 * s",
            ));
        }
        let c = din / subspace_s;
        if codebooks.len() != stages_m * c * PQSCAN_K * subspace_s {
            return Err(KernelError::Shape("pqscan: bad codebooks length"));
        }
        if codebooks.contains(&i8::MIN) {
            return Err(CoreError::WeightRange.into());
        }
        if row_codes.len() != stages_m * rows * c {
            return Err(KernelError::Shape("pqscan: bad row_codes length"));
        }
        if row_codes.iter().any(|&v| v >= PQSCAN_K as u8) {
            return Err(KernelError::Shape("pqscan: code >= 16"));
        }
        let blocks = rows.div_ceil(PQSCAN_BLOCK);
        let pairs = c / 2;
        let mut codes = vec![0u8; blocks * stages_m * pairs * PQSCAN_BLOCK];
        for blk in 0..blocks {
            for m in 0..stages_m {
                for p in 0..pairs {
                    let dst = ((blk * stages_m + m) * pairs + p) * PQSCAN_BLOCK;
                    for lane in 0..PQSCAN_BLOCK {
                        let r = blk * PQSCAN_BLOCK + lane;
                        if r < rows {
                            let base = (m * rows + r) * c + 2 * p;
                            codes[dst + lane] = row_codes[base] | (row_codes[base + 1] << 4);
                        }
                    }
                }
            }
        }
        Ok(Self {
            rows,
            din,
            subspace_s,
            stages_m,
            codebooks,
            codes,
        })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn din(&self) -> usize {
        self.din
    }
    pub fn subspace_s(&self) -> usize {
        self.subspace_s
    }
    pub fn stages_m(&self) -> usize {
        self.stages_m
    }
    pub fn subspaces_c(&self) -> usize {
        self.din / self.subspace_s
    }
    pub fn pairs(&self) -> usize {
        self.subspaces_c() / 2
    }
    pub fn codebooks(&self) -> &[i8] {
        &self.codebooks
    }
    pub fn codes(&self) -> &[u8] {
        &self.codes
    }

    /// Build exact 7-bit lookup tables into a reusable workspace (zero heap allocations on
    /// steady-state calls; dispatches to SIMD table builder when available).
    pub fn build_tables_into(
        &self,
        backend: Backend,
        x: &[i8],
        sparse: bool,
        out: &mut CodebookTables,
    ) -> Result<(), KernelError> {
        backend.require()?;
        validate_signed(x, self.din)?;
        #[cfg(target_arch = "x86_64")]
        if matches!(backend, Backend::Avx2 | Backend::Avx512Vnni) {
            crate::pqscan_avx2::build_tables_into(self, x, sparse, out);
            return Ok(());
        }
        self.build_tables_scalar_into(x, sparse, out);
        Ok(())
    }

    fn build_tables_scalar_into(&self, x: &[i8], sparse: bool, out: &mut CodebookTables) {
        let f_n = x.len() / self.din;
        let (m_n, c_n, s) = (self.stages_m, self.subspaces_c(), self.subspace_s);
        let pairs = c_n / 2;
        let mc = m_n * c_n;
        out.resize(f_n, m_n, c_n);

        for f in 0..f_n {
            let a = &x[f * self.din..(f + 1) * self.din];
            if sparse {
                for p in 0..pairs {
                    let seg = &a[2 * p * s..(2 * p + 2) * s];
                    out.pair_nonzero[f * pairs + p] = u8::from(seg.iter().any(|&v| v != 0));
                }
            }
            let mut max_rng = 0i32;
            for m in 0..m_n {
                for c in 0..c_n {
                    let idx = m * c_n + c;
                    if sparse && out.pair_nonzero[f * pairs + (c / 2)] == 0 {
                        out.raw_scratch[idx] = [0; PQSCAN_K];
                        out.mins_scratch[idx] = 0;
                        continue;
                    }
                    let ax = &a[c * s..(c + 1) * s];
                    let cb_base = idx * PQSCAN_K * s;
                    let (mut mn, mut mx) = (i32::MAX, i32::MIN);
                    for (j, slot) in out.raw_scratch[idx].iter_mut().enumerate() {
                        let cw = &self.codebooks[cb_base + j * s..cb_base + (j + 1) * s];
                        let mut v = 0i32;
                        for i in 0..s {
                            v += i32::from(ax[i]) * i32::from(cw[i]);
                        }
                        *slot = v;
                        if v < mn {
                            mn = v;
                        }
                        if v > mx {
                            mx = v;
                        }
                    }
                    out.mins_scratch[idx] = mn;
                    let r = mx - mn;
                    if r > max_rng {
                        max_rng = r;
                    }
                }
            }
            let mut sh = 0u32;
            while (max_rng >> sh) > 127 {
                sh += 1;
            }
            out.shifts[f] = sh;
            let mut sum_mn = 0i64;
            let mut active_subspaces = 0i64;
            for idx in 0..mc {
                let c = idx % c_n;
                if sparse && out.pair_nonzero[f * pairs + (c / 2)] == 0 {
                    out.entries[f * mc + idx] = [0u8; PQSCAN_K];
                    continue;
                }
                active_subspaces += 1;
                let mn = out.mins_scratch[idx];
                sum_mn += i64::from(mn);
                let dst = &mut out.entries[f * mc + idx];
                for (j, slot) in dst.iter_mut().enumerate() {
                    *slot = ((out.raw_scratch[idx][j] - mn) >> sh) as u8;
                }
            }
            let round_bias = (active_subspaces * ((1i64 << sh) - 1)) >> 1;
            out.offsets[f] = (sum_mn + round_bias) as i32;
        }
    }

    /// Build exact 7-bit lookup tables for `f_n` signed activation frames (`|a| <= 127`).
    pub fn build_tables(&self, x: &[i8], sparse: bool) -> Result<CodebookTables, KernelError> {
        let mut out = CodebookTables::default();
        self.build_tables_into(Backend::detect(), x, sparse, &mut out)?;
        Ok(out)
    }

    /// Execute codebook-scan GEMV using prebuilt [`CodebookTables`].
    pub fn scan_with_tables(
        &self,
        backend: Backend,
        tables: &CodebookTables,
        sparse: bool,
        out: &mut [i32],
    ) -> Result<(), KernelError> {
        backend.require()?;
        if tables.stages_m != self.stages_m || tables.subspaces_c != self.subspaces_c() {
            return Err(KernelError::Shape("pqscan: tables shape mismatch"));
        }
        if out.len() != self.rows * tables.frames {
            return Err(KernelError::Shape("pqscan: out.len() != rows * frames"));
        }
        match backend {
            #[cfg(target_arch = "x86_64")]
            Backend::Avx2 | Backend::Avx512Vnni => {
                crate::pqscan_avx2::scan(self, tables, sparse, out);
            }
            #[cfg(target_arch = "aarch64")]
            Backend::Neon => {
                crate::pqscan_neon::scan(self, tables, sparse, out);
            }
            #[cfg(target_arch = "aarch64")]
            Backend::Scalar => {
                // Scalar oracle explicitly requested; keep portable path for bit-exact testing.
                scan_scalar(self, tables, sparse, out);
            }
            _ => scan_scalar(self, tables, sparse, out),
        }
        Ok(())
    }

    /// Quantise a non-negative row (e.g. post-`ReLU^2` FFN activation) into zero-anchored
    /// signed `i8` codes in `0..=127` (`ULTRA-SU §6.8`): exact `0.0 -> 0i8` preserves
    /// `pair_nonzero` zero-skipping and yields 1 extra bit of positive precision over centred
    /// int8. Returns the dequantisation scale `max(0, max_i x_i) / 127.0`.
    pub fn quantize_row_zero_anchored_u7(x: &[f32], out: &mut [i8]) -> f32 {
        assert_eq!(x.len(), out.len());
        let mut mx = 0.0f32;
        for &v in x {
            if v.is_finite() && v > mx {
                mx = v;
            }
        }
        if mx == 0.0 {
            out.fill(0);
            return 0.0;
        }
        let scale = mx / 127.0;
        let inv = 127.0 / mx;
        for (o, &v) in out.iter_mut().zip(x) {
            if !v.is_finite() || v <= 0.0 {
                *o = 0;
            } else {
                *o = (v * inv).round_ties_even().clamp(0.0, 127.0) as i8;
            }
        }
        scale
    }

    /// ARM64 NEON explicit entry point (for cross-ISA verification & M3 Ultra benchmarks).
    #[cfg(target_arch = "aarch64")]
    pub fn scan_neon(
        &self,
        tables: &CodebookTables,
        sparse: bool,
        out: &mut [i32],
    ) -> Result<(), KernelError> {
        if out.len() != self.rows * tables.frames {
            return Err(KernelError::Shape("pqscan: out.len() != rows * frames"));
        }
        crate::pqscan_neon::scan(self, tables, sparse, out);
        Ok(())
    }

    /// Convenience end-to-end table build + scan (`out[r * frames + f]`).
    pub fn gemv(
        &self,
        backend: Backend,
        x: &[i8],
        sparse: bool,
        out: &mut [i32],
    ) -> Result<(), KernelError> {
        let mut tables = CodebookTables::default();
        self.build_tables_into(backend, x, sparse, &mut tables)?;
        self.scan_with_tables(backend, &tables, sparse, out)
    }
}

fn validate_signed(x: &[i8], din: usize) -> Result<(), KernelError> {
    if x.is_empty() || !x.len().is_multiple_of(din) {
        return Err(KernelError::Shape(
            "pqscan: x.len() must be a multiple of din",
        ));
    }
    if x.contains(&i8::MIN) {
        return Err(KernelError::Shape("pqscan: -128 outside -127..=127"));
    }
    Ok(())
}

/// Bit-exact portable scalar oracle matching [`ULTRA-SU §6.2`].
pub fn scan_scalar(m: &CodebookMatrix, t: &CodebookTables, sparse: bool, out: &mut [i32]) {
    let (rows, f_n, m_n, c_n) = (m.rows(), t.frames, m.stages_m(), m.subspaces_c());
    let pairs = c_n / 2;
    let mc = m_n * c_n;
    let blocks = rows.div_ceil(PQSCAN_BLOCK);
    let blk_stride = m_n * pairs * PQSCAN_BLOCK;

    for blk in 0..blocks {
        let blk_codes = &m.codes()[blk * blk_stride..(blk + 1) * blk_stride];
        let r0 = blk * PQSCAN_BLOCK;
        let r_end = (r0 + PQSCAN_BLOCK).min(rows);
        for f in 0..f_n {
            let f_ent = &t.entries[f * mc..(f + 1) * mc];
            let f_nz = &t.pair_nonzero[f * pairs..(f + 1) * pairs];
            let sh = t.shifts[f];
            let off = t.offsets[f];
            let mut sums = [0u32; PQSCAN_BLOCK];
            for st in 0..m_n {
                for p in 0..pairs {
                    if sparse && f_nz[p] == 0 {
                        continue;
                    }
                    let t_even = &f_ent[st * c_n + 2 * p];
                    let t_odd = &f_ent[st * c_n + 2 * p + 1];
                    let lane_codes = &blk_codes
                        [(st * pairs + p) * PQSCAN_BLOCK..(st * pairs + p + 1) * PQSCAN_BLOCK];
                    for lane in 0..(r_end - r0) {
                        let b = lane_codes[lane];
                        sums[lane] += u32::from(t_even[(b & 0x0F) as usize])
                            + u32::from(t_odd[(b >> 4) as usize]);
                    }
                }
            }
            for lane in 0..(r_end - r0) {
                out[(r0 + lane) * f_n + f] = ((sums[lane] as i64) << sh) as i32 + off;
            }
        }
    }
}

/// Block-16 orthonormal Walsh-Hadamard transform (`H_16 / 4`, zero multiplies except power-of-2
/// scaling `0.25`, self-inverse: `fwht16(fwht16(x)) == x`).
///
/// Requires `x.len()` to be a multiple of 16.
pub fn fwht16_inplace(x: &mut [f32]) -> Result<(), KernelError> {
    if !x.len().is_multiple_of(16) {
        return Err(KernelError::Shape(
            "fwht16: length must be a multiple of 16",
        ));
    }
    #[cfg(target_arch = "x86_64")]
    if Backend::Avx2.is_available() {
        crate::pqscan_avx2::fwht16(x);
        return Ok(());
    }
    #[cfg(target_arch = "aarch64")]
    {
        crate::pqscan_neon::fwht16(x);
        return Ok(());
    }
    #[cfg(not(target_arch = "aarch64"))]
    fwht16_scalar(x);
    #[cfg(not(target_arch = "aarch64"))]
    Ok(())
}

/// Portable scalar reference for block-16 orthonormal Walsh-Hadamard transform.
pub fn fwht16_scalar(x: &mut [f32]) {
    for blk in x.chunks_exact_mut(16) {
        let mut s = 1usize;
        while s < 16 {
            let mut i = 0;
            while i < 16 {
                for j in 0..s {
                    let a = blk[i + j];
                    let b = blk[i + j + s];
                    blk[i + j] = a + b;
                    blk[i + j + s] = a - b;
                }
                i += 2 * s;
            }
            s <<= 1;
        }
        for v in blk.iter_mut() {
            *v *= 0.25;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(seed: &mut u64) -> u64 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *seed
    }

    fn rand_matrix(rows: usize, din: usize, s: usize, m: usize, seed: &mut u64) -> CodebookMatrix {
        let c = din / s;
        let mut cb = vec![0i8; m * c * PQSCAN_K * s];
        for v in &mut cb {
            let r = ((lcg(seed) >> 33) % 255) as i32 - 127;
            *v = r as i8;
        }
        let mut codes = vec![0u8; m * rows * c];
        for v in &mut codes {
            *v = ((lcg(seed) >> 33) & 15) as u8;
        }
        CodebookMatrix::from_parts(rows, din, s, m, cb, &codes).unwrap()
    }

    #[test]
    fn pqscan_matches_scalar_across_shapes_tails_and_sparsity() {
        let mut seed = 0x2026_1122u64;
        for (rows, din, s, m_st) in [
            (1, 16, 2, 1),
            (31, 64, 4, 1),
            (33, 64, 4, 2),
            (64, 128, 8, 1),
            (384, 384, 4, 1),
        ] {
            let mat = rand_matrix(rows, din, s, m_st, &mut seed);
            for frames in [1usize, 2, 3, 4, 5, 9] {
                let mut x = vec![0i8; frames * din];
                for v in &mut x {
                    *v = (((lcg(&mut seed) >> 33) % 255) as i32 - 127) as i8;
                }
                let mut ref_out = vec![0i32; rows * frames];
                let mut simd_out = vec![0i32; rows * frames];
                mat.gemv(Backend::Scalar, &x, false, &mut ref_out).unwrap();
                mat.gemv(Backend::detect(), &x, false, &mut simd_out)
                    .unwrap();
                assert_eq!(
                    ref_out, simd_out,
                    "dense mismatch rows={rows} din={din} s={s} f={frames}"
                );

                // Zero out 75% of subspace pairs to test sparse scan parity.
                let pair_w = 2 * s;
                for f in 0..frames {
                    for p in 0..(din / pair_w) {
                        if p % 4 != 0 {
                            x[f * din + p * pair_w..f * din + (p + 1) * pair_w].fill(0);
                        }
                    }
                }
                mat.gemv(Backend::Scalar, &x, true, &mut ref_out).unwrap();
                mat.gemv(Backend::detect(), &x, true, &mut simd_out)
                    .unwrap();
                assert_eq!(
                    ref_out, simd_out,
                    "sparse mismatch rows={rows} din={din} s={s} f={frames}"
                );
            }
        }
    }

    #[test]
    fn pqscan_extreme_accumulator_flush_768_subspaces() {
        // ULTRA-SU Appendix A worst case: 768 subspaces (384 pairs > 256 flush limit), all codes=15,
        // maximal table entries -> sum per output = 768 * 127 = 97,536 (> u16::MAX).
        let (rows, din, s, m_st) = (64, 1536, 2, 1);
        let c = din / s; // 768
        let mut cb = vec![-127i8; m_st * c * PQSCAN_K * s];
        for ci in 0..c {
            let base = (ci * PQSCAN_K + 15) * s;
            cb[base..base + s].fill(127);
        }
        let codes = vec![15u8; rows * c];
        let mat = CodebookMatrix::from_parts(rows, din, s, m_st, cb, &codes).unwrap();
        let x = vec![127i8; 4 * din];
        let mut ref_out = vec![0i32; rows * 4];
        let mut simd_out = vec![0i32; rows * 4];
        mat.gemv(Backend::Scalar, &x, false, &mut ref_out).unwrap();
        mat.gemv(Backend::detect(), &x, false, &mut simd_out)
            .unwrap();
        assert_eq!(ref_out, simd_out);
        assert!(ref_out[0] > 0);
    }

    #[test]
    fn fwht16_orthonormal_self_inverse_and_simd_parity() {
        let mut a: Vec<f32> = (0..64).map(|i| (i as f32 * 0.37) - 11.0).collect();
        let orig = a.clone();
        let mut b = a.clone();
        fwht16_scalar(&mut a);
        fwht16_inplace(&mut b).unwrap();
        for (u, v) in a.iter().zip(&b) {
            assert!((u - v).abs() < 1e-5, "fwht16 mismatch: {u} vs {v}");
        }
        fwht16_inplace(&mut b).unwrap();
        for (u, v) in orig.iter().zip(&b) {
            assert!((u - v).abs() < 1e-4, "fwht16 inverse mismatch: {u} vs {v}");
        }
    }

    #[test]
    fn zero_anchored_u7_preserves_exact_zeros_for_relu2_pqscan() {
        let x = [0.0f32, 0.0, 0.0, 0.0, 2.5, 0.0, 10.0, -0.5];
        let mut q = [0i8; 8];
        let scale = CodebookMatrix::quantize_row_zero_anchored_u7(&x, &mut q);
        assert!((scale - 10.0 / 127.0).abs() < 1e-6);
        assert_eq!(q[0..4], [0, 0, 0, 0]);
        assert_eq!(q[5], 0);
        assert_eq!(q[6], 127);
        assert_eq!(q[7], 0);
        assert!((f32::from(q[4]) * scale - 2.5).abs() <= scale * 0.51);
    }
}
