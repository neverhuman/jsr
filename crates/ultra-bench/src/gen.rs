//! Deterministic data of `wstream.c`: the same xorshift stream, the same trit draw
//! (`code = (x >> 8k) % 3`, weight = code - 1) and the same T1 packing, so a sweep whose
//! largest size is 1536M reproduces the Appendix A checksums exactly.

use crate::{DIN, MAX_FRAMES};

/// The `wstream.c` xorshift64 generator.
#[derive(Debug, Clone)]
pub struct XorShift(u64);

impl Default for XorShift {
    fn default() -> Self {
        Self(0x9E37_79B9_7F4A_7C15)
    }
}

impl XorShift {
    /// Next 64-bit value.
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
}

/// Generated benchmark data for the largest requested size.
#[derive(Debug)]
pub struct Data {
    /// T1-packed weights (`max_weights / 4` bytes).
    pub packed: Vec<u8>,
    /// The same weights as int8 trits, when requested.
    pub trits: Option<Vec<i8>>,
    /// `MAX_FRAMES * DIN` A7 activations.
    pub acts: Vec<u8>,
}

/// Generate `max_weights` weights (a multiple of `DIN`), then the activations, in the
/// order `wstream.c` draws them. `with_trits` also materialises the int8 copy.
pub fn generate(max_weights: usize, with_trits: bool) -> Data {
    let mut rng = XorShift::default();
    let mut packed = vec![0u8; max_weights / 4];
    let mut trits = with_trits.then(|| vec![0i8; max_weights]);
    for (i, byte) in packed.iter_mut().enumerate() {
        let x = rng.next_u64();
        let mut b = 0u8;
        for k in 0..4 {
            let code = ((x >> (8 * k)) % 3) as u8;
            b |= code << (2 * k);
            if let Some(t) = trits.as_mut() {
                t[4 * i + k] = code as i8 - 1;
            }
        }
        *byte = b;
    }
    let acts = (0..MAX_FRAMES * DIN)
        .map(|_| (rng.next_u64() & 0x7F) as u8)
        .collect();
    Data {
        packed,
        trits,
        acts,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trits_and_packing_agree() {
        let d = generate(4 * DIN, true);
        let t = d.trits.unwrap();
        for (i, &b) in d.packed.iter().enumerate() {
            for k in 0..4 {
                assert_eq!(((b >> (2 * k)) & 3) as i8 - 1, t[4 * i + k]);
            }
        }
        assert!(d.acts.iter().all(|&a| a <= 127));
        assert_eq!(d.acts.len(), MAX_FRAMES * DIN);
    }

    #[test]
    fn same_stream_without_trits() {
        let a = generate(2 * DIN, true);
        let b = generate(2 * DIN, false);
        assert_eq!(a.packed, b.packed);
        assert_eq!(a.acts, b.acts);
    }
}
