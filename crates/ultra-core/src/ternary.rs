//! T1 ternary packing: four trits per byte, code = w + 1 in bits 2k..2k+1 (the layout of
//! the spec's `wstream.c`). Code 3 is refused at load (§5.1).

use crate::CoreError;

/// Pack ternary weights (each in {-1, 0, 1}) four per byte. Length must be a multiple of 4.
pub fn pack_t1(w: &[i8]) -> Result<Vec<u8>, CoreError> {
    if !w.len().is_multiple_of(4) {
        return Err(CoreError::Shape("pack_t1 length"));
    }
    w.chunks_exact(4)
        .map(|c| {
            c.iter().enumerate().try_fold(0u8, |acc, (k, &t)| {
                if !(-1..=1).contains(&t) {
                    return Err(CoreError::WeightRange);
                }
                Ok(acc | (((t + 1) as u8) << (2 * k)))
            })
        })
        .collect()
}

/// Unpack T1 bytes into trits, refusing code 3.
pub fn unpack_t1(p: &[u8]) -> Result<Vec<i8>, CoreError> {
    let mut out = Vec::with_capacity(p.len() * 4);
    for &b in p {
        for k in 0..4 {
            let code = (b >> (2 * k)) & 3;
            if code == 3 {
                return Err(CoreError::BadTernaryCode);
            }
            out.push(code as i8 - 1);
        }
    }
    Ok(out)
}

/// Validate a packed T1 image without unpacking (load-time check).
pub fn validate_t1(p: &[u8]) -> Result<(), CoreError> {
    // A byte contains code 3 iff some pair of bits is 0b11.
    if p.iter().any(|&b| (b & (b >> 1) & 0x55) != 0) {
        Err(CoreError::BadTernaryCode)
    } else {
        Ok(())
    }
}

/// Scalar reference: dot of a packed T1 row with A7 (or U8) activations.
pub fn dot_t1(p: &[u8], a: &[u8]) -> i32 {
    let mut acc = 0i32;
    for (i, &b) in p.iter().enumerate() {
        for k in 0..4 {
            let j = 4 * i + k;
            if j >= a.len() {
                break;
            }
            let t = i32::from((b >> (2 * k)) & 3) - 1;
            acc += t * i32::from(a[j]);
        }
    }
    acc
}

/// Row sum of a packed T1 row (for the A7z zero-point correction).
pub fn row_sum_t1(p: &[u8]) -> i32 {
    p.iter()
        .map(|&b| {
            (0..4)
                .map(|k| i32::from((b >> (2 * k)) & 3) - 1)
                .sum::<i32>()
        })
        .sum()
}
