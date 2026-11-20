//! T-LUT2: two trits per 4-bit table index (cadence §5.3 item 2), 2 bits per weight.
//!
//! **State.** A pair `(t0, t1)` is stored as `s = 3 (t0 + 1) + (t1 + 1)`, 0..=8; indices
//! 9..=15 are refused at load (the analogue of T1 code 3). Zero pair = 4.
//!
//! **Table.** For an activation pair `(a0, a1)` the kernel needs `t0 a0 + t1 a1` for the
//! nine states, a value in -254..=254. Cadence §5.3 proposes a five-entry unsigned table
//! `{0, a0, a1, |a0 - a1|, a0 + a1}` plus a weight sign bit; that is not exact, because the
//! sign of `a0 - a1` depends on the activations, not the weights. This kernel instead keeps
//! the full signed nine-entry table as two 16-byte planes (low and high byte of the i16),
//! looks both up with `vpshufb` and re-interleaves them into i16 lanes. Exact, no multiply.
//!
//! **Layout.** Rows are grouped in blocks of [`TLUT2_BLOCK`] = 32 (one `vpshufb` lane per
//! row). Per block, per column quad `q` (columns `4q..4q+4`), 32 bytes: byte `i` belongs to
//! row `32 b + i`; its low nibble is the state of columns `(4q, 4q+1)`, its high nibble the
//! state of `(4q+2, 4q+3)`. Missing rows and columns are padded with state 4.

use crate::{check_call, check_slice, scalar, Backend, Frames, Gemv, KernelError, T1View};
use ultra_core::numeric::prove_row_no_overflow;
use ultra_core::CoreError;

/// Rows per interleaved block (one byte lane of a 256-bit register each).
pub const TLUT2_BLOCK: usize = 32;
/// State of the zero pair.
const ZERO_STATE: u8 = 4;

/// Pair state from two trits.
fn state(t0: i8, t1: i8) -> u8 {
    (3 * (t0 + 1) + (t1 + 1)) as u8
}

/// Trits from a pair state (`s <= 8`).
pub(crate) fn trits(s: u8) -> (i32, i32) {
    (i32::from(s / 3) - 1, i32::from(s % 3) - 1)
}

/// Row-interleaved T-LUT2 weights.
#[derive(Debug, Clone)]
pub struct TLut2Matrix {
    rows: usize,
    din: usize,
    idx: Vec<u8>,
}

impl TLut2Matrix {
    /// Encode `rows * din` trits (each in {-1, 0, 1}).
    pub fn from_trits(rows: usize, din: usize, trits: &[i8]) -> Result<Self, KernelError> {
        if rows == 0 || din == 0 || trits.len() != rows * din {
            return Err(KernelError::Shape("tlut2 matrix: len != rows * din"));
        }
        if trits.iter().any(|t| !(-1..=1).contains(t)) {
            return Err(CoreError::WeightRange.into());
        }
        Ok(Self::build(rows, din, |r, j| trits[r * din + j]))
    }

    /// Re-encode a T1 matrix (already validated).
    pub fn from_t1(m: T1View<'_>) -> Self {
        Self::build(m.rows(), m.din(), |r, j| {
            ((m.row(r)[j / 4] >> (2 * (j % 4))) & 3) as i8 - 1
        })
    }

    /// Adopt a raw interleaved image; refuses indices 9..=15.
    pub fn from_raw(rows: usize, din: usize, idx: Vec<u8>) -> Result<Self, KernelError> {
        if rows == 0 || din == 0 || idx.len() != Self::image_len(rows, din) {
            return Err(KernelError::Shape("tlut2 matrix: bad image length"));
        }
        if idx.iter().any(|&b| (b & 15) > 8 || (b >> 4) > 8) {
            return Err(CoreError::BadTernaryCode.into());
        }
        prove_row_no_overflow(din, 1)?;
        Ok(Self { rows, din, idx })
    }

    fn image_len(rows: usize, din: usize) -> usize {
        rows.div_ceil(TLUT2_BLOCK) * din.div_ceil(4) * TLUT2_BLOCK
    }

    fn build(rows: usize, din: usize, trit: impl Fn(usize, usize) -> i8) -> Self {
        let quads = din.div_ceil(4);
        let mut idx = vec![0u8; Self::image_len(rows, din)];
        let pair = |r: usize, j: usize| -> u8 {
            if r >= rows || j >= din {
                return ZERO_STATE;
            }
            let t1 = if j + 1 < din { trit(r, j + 1) } else { 0 };
            state(trit(r, j), t1)
        };
        for (chunk, out) in idx.chunks_exact_mut(TLUT2_BLOCK).enumerate() {
            let (blk, q) = (chunk / quads, chunk % quads);
            for (lane, byte) in out.iter_mut().enumerate() {
                let r = blk * TLUT2_BLOCK + lane;
                *byte = pair(r, 4 * q) | (pair(r, 4 * q + 2) << 4);
            }
        }
        Self { rows, din, idx }
    }

    /// Borrow as a kernel view.
    pub fn view(&self) -> TLut2View<'_> {
        TLut2View {
            rows: self.rows,
            din: self.din,
            idx: &self.idx,
        }
    }
}

/// Borrowed row blocks of a [`TLut2Matrix`].
#[derive(Debug, Clone, Copy)]
pub struct TLut2View<'a> {
    rows: usize,
    din: usize,
    idx: &'a [u8],
}

impl<'a> TLut2View<'a> {
    /// Column quads per row (`ceil(din / 4)`).
    pub fn quads(&self) -> usize {
        self.din.div_ceil(4)
    }

    /// The interleaved image (`blocks * quads * 32` bytes).
    pub fn image(&self) -> &'a [u8] {
        self.idx
    }

    /// State of the pair at columns `(2p, 2p + 1)` of row `r`.
    pub fn pair_state(&self, r: usize, p: usize) -> u8 {
        let (blk, lane) = (r / TLUT2_BLOCK, r % TLUT2_BLOCK);
        let byte = self.idx[(blk * self.quads() + p / 2) * TLUT2_BLOCK + lane];
        if p.is_multiple_of(2) {
            byte & 15
        } else {
            byte >> 4
        }
    }
}

impl Gemv for TLut2View<'_> {
    fn rows(&self) -> usize {
        self.rows
    }
    fn din(&self) -> usize {
        self.din
    }
    fn row_align(&self) -> usize {
        TLUT2_BLOCK
    }
    fn slice_rows(&self, start: usize, end: usize) -> Result<Self, KernelError> {
        check_slice(start, end, self.rows, TLUT2_BLOCK)?;
        let per_block = self.quads() * TLUT2_BLOCK;
        let (b0, b1) = (start / TLUT2_BLOCK, end.div_ceil(TLUT2_BLOCK));
        Ok(Self {
            rows: end - start,
            din: self.din,
            idx: &self.idx[b0 * per_block..b1 * per_block],
        })
    }
    fn gemv(&self, backend: Backend, x: &Frames<'_>, out: &mut [i32]) -> Result<(), KernelError> {
        check_call(backend, self.rows, self.din, x, out)?;
        match backend {
            // AVX-512 has no T-LUT2 port; its feature set implies AVX2.
            #[cfg(target_arch = "x86_64")]
            Backend::Avx2 | Backend::Avx512Vnni => crate::x86_avx2::gemv_tlut2(*self, x, out),
            _ => scalar::gemv_tlut2(*self, x, out),
        }
        Ok(())
    }
}

/// Per-frame lookup tables: for frame `f` and pair `p`, entry `2 (f * pairs + p)` holds
/// the low bytes and entry `+1` the high bytes of `t0 a0 + t1 a1` for states 0..=8
/// (entries 9..=15 are zero). Pairs cover `2 * quads` columns; missing columns read 0.
pub(crate) fn build_tables(x: &Frames<'_>, quads: usize) -> Vec<[u8; 16]> {
    let pairs = 2 * quads;
    let mut t = vec![[0u8; 16]; 2 * pairs * x.count()];
    for f in 0..x.count() {
        let a = x.frame(f);
        let at = |j: usize| a.get(j).map_or(0, |&v| i32::from(v));
        for p in 0..pairs {
            let (a0, a1) = (at(2 * p), at(2 * p + 1));
            let k = 2 * (f * pairs + p);
            for s in 0..9u8 {
                let (t0, t1) = trits(s);
                let v = (t0 * a0 + t1 * a1) as i16;
                let [lo, hi] = v.to_le_bytes();
                t[k][usize::from(s)] = lo;
                t[k + 1][usize::from(s)] = hi;
            }
        }
    }
    t
}
