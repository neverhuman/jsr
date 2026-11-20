//! Integer numeric contract (ULTRA spec §5.1–§5.3). Every function here is the scalar
//! reference that SIMD kernels and the Python integer simulator must match bit for bit.

use crate::CoreError;

/// Largest A7 / A7z activation value (rule N1).
pub const A7_MAX: u8 = 127;
/// Zero point of the A7z format.
pub const A7Z_ZERO: i32 = 64;
/// Smallest legal int8 weight (rule N1: weights never reach -128).
pub const I8_MIN: i8 = -127;

/// `EXP[k] = round(65535 * exp(-k / 16))`, k = 0..=255. Bundle data (§5.2 item 8);
/// regenerated and compared by `exp_table_regenerates` in the tests.
pub const EXP: [u16; 256] = [
    65535, 61564, 57834, 54330, 51039, 47946, 45042, 42313, 39749, 37341, 35078, 32953, 30957,
    29081, 27319, 25664, 24109, 22648, 21276, 19987, 18776, 17639, 16570, 15566, 14623, 13737,
    12905, 12123, 11388, 10698, 10050, 9441, 8869, 8332, 7827, 7353, 6907, 6489, 6096, 5726, 5379,
    5054, 4747, 4460, 4190, 3936, 3697, 3473, 3263, 3065, 2879, 2705, 2541, 2387, 2242, 2107, 1979,
    1859, 1746, 1641, 1541, 1448, 1360, 1278, 1200, 1128, 1059, 995, 935, 878, 825, 775, 728, 684,
    642, 604, 567, 533, 500, 470, 442, 415, 390, 366, 344, 323, 303, 285, 268, 252, 236, 222, 209,
    196, 184, 173, 162, 153, 143, 135, 127, 119, 112, 105, 99, 93, 87, 82, 77, 72, 68, 64, 60, 56,
    53, 50, 47, 44, 41, 39, 36, 34, 32, 30, 28, 27, 25, 23, 22, 21, 19, 18, 17, 16, 15, 14, 13, 13,
    12, 11, 10, 10, 9, 9, 8, 8, 7, 7, 6, 6, 6, 5, 5, 5, 4, 4, 4, 4, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2,
    2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
];

/// Requantise an int32 accumulator: `sat16((acc*M + b + 2^(s-1)) >> s)` in int64 with an
/// arithmetic shift. Halves round toward +inf for both signs. `s` must be in 1..=31.
pub fn requant(acc: i32, mult: i32, bias: i64, shift: u32) -> Result<i16, CoreError> {
    if !(1..=31).contains(&shift) {
        return Err(CoreError::BadShift(shift));
    }
    let v = (i64::from(acc) * i64::from(mult))
        .saturating_add(bias)
        .saturating_add(1i64 << (shift - 1));
    Ok(sat16(v >> shift))
}

/// Saturate an i64 to the i16 range.
pub fn sat16(v: i64) -> i16 {
    v.clamp(i64::from(i16::MIN), i64::from(i16::MAX)) as i16
}

/// Clipped ReLU into A7: `clamp(y >> e, 0, 127)`.
pub fn clipped_relu(y: i16, e: u32) -> u8 {
    (i32::from(y) >> e.min(15)).clamp(0, i32::from(A7_MAX)) as u8
}

/// Floor integer square root of a u64.
pub fn isqrt_u64(n: u64) -> u64 {
    if n < 2 {
        return n;
    }
    let mut x = (n as f64).sqrt() as u64;
    // Correct the float estimate to the exact floor.
    while x.checked_mul(x).is_none_or(|sq| sq > n) {
        x -= 1;
    }
    while (x + 1).checked_mul(x + 1).is_some_and(|sq| sq <= n) {
        x += 1;
    }
    x
}

/// Integer RMS norm (§5.2 item 4). Input is the R16 residual, output is A7z.
/// `gain` is per channel; `shift` is the stack's output shift.
pub fn rms_norm(r: &[i16], gain: &[i16], shift: u32, out: &mut [u8]) -> Result<(), CoreError> {
    let d = r.len();
    if d == 0 || gain.len() != d || out.len() != d || d > 4096 {
        return Err(CoreError::Shape("rms_norm"));
    }
    let n: i64 = r.iter().map(|&x| i64::from(x) * i64::from(x)).sum();
    let rms = isqrt_u64((n / d as i64) as u64) as i64;
    let inv = (1i64 << 30) / rms.max(1);
    let s = shift.min(62);
    for ((o, &x), &g) in out.iter_mut().zip(r).zip(gain) {
        let v = ((i64::from(x) * i64::from(g) * inv) >> s) + i64::from(A7Z_ZERO);
        *o = v.clamp(0, i64::from(A7_MAX)) as u8;
    }
    Ok(())
}

/// Exact integer softmax (§5.2 item 6). `scores` are int32 attention scores; `shift` maps
/// one step to 1/16 nat. Writes Q15 weights. Equal scores give equal weights.
pub fn softmax_q15(scores: &[i32], shift: u32, out: &mut [u16]) -> Result<(), CoreError> {
    if scores.is_empty() || scores.len() != out.len() || scores.len() > 256 {
        return Err(CoreError::Shape("softmax_q15"));
    }
    let s = shift.min(31);
    let q: Vec<i32> = scores.iter().map(|&x| x >> s).collect();
    let m = q.iter().copied().max().unwrap_or(0);
    let mut z: u32 = 0;
    let e: Vec<u32> = q
        .iter()
        .map(|&qi| {
            let dist = (i64::from(m) - i64::from(qi)).min(255) as usize;
            let v = u32::from(EXP[dist]);
            z += v;
            v
        })
        .collect();
    // z >= 65535 because the max element has dist 0.
    for (o, ei) in out.iter_mut().zip(e) {
        *o = ((u64::from(ei) * (1u64 << 15) + u64::from(z / 2)) / u64::from(z)) as u16;
    }
    Ok(())
}

/// Dot product of int8 weights with A7 activations, in int32 (zero point 0).
pub fn dot_i8_a7(w: &[i8], a: &[u8]) -> i32 {
    w.iter()
        .zip(a)
        .map(|(&wi, &ai)| i32::from(wi) * i32::from(ai))
        .sum()
}

/// A7z zero-point correction: `sum w_i (a_i - 64) = acc - 64 * sum w_i`.
pub fn a7z_correct(acc: i32, row_sum: i32) -> i32 {
    acc - A7Z_ZERO * row_sum
}

/// Rule N2: refuse a row unless `fan_in * 127 * max|w| < 2^31`.
pub fn prove_row_no_overflow(fan_in: usize, max_abs_w: u32) -> Result<(), CoreError> {
    let bound = (fan_in as u128) * 127 * u128::from(max_abs_w);
    if bound < (1u128 << 31) {
        Ok(())
    } else {
        Err(CoreError::Overflow { fan_in, max_abs_w })
    }
}

/// Check an int8 weight slice obeys rule N1.
pub fn check_i8_weights(w: &[i8]) -> Result<(), CoreError> {
    if w.contains(&i8::MIN) {
        Err(CoreError::WeightRange)
    } else {
        Ok(())
    }
}

/// Check activations obey rule N1.
pub fn check_a7(a: &[u8]) -> Result<(), CoreError> {
    if a.iter().any(|&x| x > A7_MAX) {
        Err(CoreError::ActivationRange)
    } else {
        Ok(())
    }
}
