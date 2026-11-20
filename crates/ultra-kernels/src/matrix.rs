//! Int8 and T1 weight matrices. Construction enforces the load-time rules (N1, N2, code 3
//! refused), so every view handed to a kernel is already legal.

use crate::{check_call, check_slice, scalar, Backend, Frames, Gemv, KernelError};
use ultra_core::numeric::{check_i8_weights, prove_row_no_overflow};
use ultra_core::ternary::{pack_t1, validate_t1};

/// Row-major int8 weights in -127..=127.
#[derive(Debug, Clone)]
pub struct I8Matrix {
    rows: usize,
    din: usize,
    w: Vec<i8>,
}

impl I8Matrix {
    /// Take ownership of `rows * din` weights; refuses -128 (N1) and rows that could
    /// overflow int32 (N2).
    pub fn new(rows: usize, din: usize, w: Vec<i8>) -> Result<Self, KernelError> {
        if rows == 0 || din == 0 || w.len() != rows * din {
            return Err(KernelError::Shape("i8 matrix: len != rows * din"));
        }
        check_i8_weights(&w)?;
        let max_abs = w.iter().map(|&x| x.unsigned_abs()).max().unwrap_or(0);
        prove_row_no_overflow(din, u32::from(max_abs))?;
        Ok(Self { rows, din, w })
    }

    /// Borrow as a kernel view.
    pub fn view(&self) -> I8View<'_> {
        I8View {
            rows: self.rows,
            din: self.din,
            w: &self.w,
        }
    }
}

/// Borrowed rows of an [`I8Matrix`].
#[derive(Debug, Clone, Copy)]
pub struct I8View<'a> {
    rows: usize,
    din: usize,
    w: &'a [i8],
}

impl<'a> I8View<'a> {
    /// Row `r` (panics if out of range).
    pub fn row(&self, r: usize) -> &'a [i8] {
        &self.w[r * self.din..(r + 1) * self.din]
    }

    /// The raw row-major weights.
    pub fn weights(&self) -> &'a [i8] {
        self.w
    }
}

impl Gemv for I8View<'_> {
    fn rows(&self) -> usize {
        self.rows
    }
    fn din(&self) -> usize {
        self.din
    }
    fn row_align(&self) -> usize {
        1
    }
    fn slice_rows(&self, start: usize, end: usize) -> Result<Self, KernelError> {
        check_slice(start, end, self.rows, 1)?;
        Ok(Self {
            rows: end - start,
            din: self.din,
            w: &self.w[start * self.din..end * self.din],
        })
    }
    fn gemv(&self, backend: Backend, x: &Frames<'_>, out: &mut [i32]) -> Result<(), KernelError> {
        check_call(backend, self.rows, self.din, x, out)?;
        match backend {
            #[cfg(target_arch = "x86_64")]
            Backend::Avx2 => crate::x86_avx2::gemv_i8(*self, x, out),
            #[cfg(target_arch = "x86_64")]
            Backend::Avx512Vnni => crate::x86_avx512::gemv_i8(*self, x, out),
            _ => scalar::gemv_i8(*self, x, out),
        }
        Ok(())
    }
}

/// T1 packed ternary: each row is `ceil(din / 4)` bytes, four trits per byte, code
/// `w + 1` at bits `2k..2k+1`. Trits past `din` in a row's last byte are ignored.
#[derive(Debug, Clone)]
pub struct T1Matrix {
    rows: usize,
    din: usize,
    p: Vec<u8>,
}

impl T1Matrix {
    /// Pack `rows * din` trits (each in {-1, 0, 1}); rows are padded with zero trits.
    pub fn from_trits(rows: usize, din: usize, trits: &[i8]) -> Result<Self, KernelError> {
        if rows == 0 || din == 0 || trits.len() != rows * din {
            return Err(KernelError::Shape("t1 matrix: len != rows * din"));
        }
        let stride = din.div_ceil(4);
        let mut p = Vec::with_capacity(rows * stride);
        let mut padded = vec![0i8; stride * 4];
        for row in trits.chunks_exact(din) {
            padded[..din].copy_from_slice(row);
            p.extend(pack_t1(&padded)?);
        }
        Self::from_packed(rows, din, p)
    }

    /// Adopt an already packed image; refuses any code 3 (load-time check, §5.1).
    pub fn from_packed(rows: usize, din: usize, p: Vec<u8>) -> Result<Self, KernelError> {
        if rows == 0 || din == 0 || p.len() != rows * din.div_ceil(4) {
            return Err(KernelError::Shape("t1 matrix: len != rows * ceil(din / 4)"));
        }
        validate_t1(&p)?;
        prove_row_no_overflow(din, 1)?;
        Ok(Self { rows, din, p })
    }

    /// Borrow as a kernel view.
    pub fn view(&self) -> T1View<'_> {
        T1View {
            rows: self.rows,
            din: self.din,
            p: &self.p,
        }
    }
}

/// Borrowed rows of a [`T1Matrix`].
#[derive(Debug, Clone, Copy)]
pub struct T1View<'a> {
    rows: usize,
    din: usize,
    p: &'a [u8],
}

impl<'a> T1View<'a> {
    /// Bytes per packed row.
    pub fn stride(&self) -> usize {
        self.din.div_ceil(4)
    }

    /// Packed row `r` (panics if out of range).
    pub fn row(&self, r: usize) -> &'a [u8] {
        let s = self.stride();
        &self.p[r * s..(r + 1) * s]
    }
}

impl Gemv for T1View<'_> {
    fn rows(&self) -> usize {
        self.rows
    }
    fn din(&self) -> usize {
        self.din
    }
    fn row_align(&self) -> usize {
        1
    }
    fn slice_rows(&self, start: usize, end: usize) -> Result<Self, KernelError> {
        check_slice(start, end, self.rows, 1)?;
        let s = self.stride();
        Ok(Self {
            rows: end - start,
            din: self.din,
            p: &self.p[start * s..end * s],
        })
    }
    fn gemv(&self, backend: Backend, x: &Frames<'_>, out: &mut [i32]) -> Result<(), KernelError> {
        check_call(backend, self.rows, self.din, x, out)?;
        match backend {
            #[cfg(target_arch = "x86_64")]
            Backend::Avx2 => crate::x86_avx2::gemv_t1(*self, x, out),
            #[cfg(target_arch = "x86_64")]
            Backend::Avx512Vnni => crate::x86_avx512::gemv_t1(*self, x, out),
            _ => scalar::gemv_t1(*self, x, out),
        }
        Ok(())
    }
}
