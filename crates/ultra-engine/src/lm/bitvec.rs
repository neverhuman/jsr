//! Fixed-width bit-packed unsigned integer vector (`1..=32` bits/entry) for compact
//! `ultra.lm-ngram/v2` trie storage (100% safe Rust, O(1) random access).

/// Packed array of `len` unsigned integers, each stored in `bits` (`1..=32`) bits inside
/// little-endian `u64` words with one trailing zero sentinel word so cross-word reads are
/// branchless and never out of bounds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct BitVec {
    pub(super) bits: u8,
    pub(super) len: usize,
    pub(super) mask: u64,
    pub(super) words: Vec<u64>,
}

impl BitVec {
    pub(super) fn empty() -> Self {
        Self {
            bits: 0,
            len: 0,
            mask: 0,
            words: Vec::new(),
        }
    }

    pub(super) fn bits_needed(max_val: u32) -> u8 {
        let needed = 32 - max_val.leading_zeros();
        (needed as u8).max(1)
    }

    pub(super) fn from_u16_slice(vals: &[u16], bits: u8) -> Self {
        if vals.is_empty() {
            return Self::empty();
        }
        assert!((1..=16).contains(&bits), "BitVec u16 bits out of range");
        let b = usize::from(bits);
        let mask = if bits == 64 {
            u64::MAX
        } else {
            (1u64 << bits) - 1
        };
        let n_words = (vals.len() * b).div_ceil(64) + 1;
        let mut words = vec![0u64; n_words];
        for (i, &v) in vals.iter().enumerate() {
            let v64 = u64::from(v);
            debug_assert!(v64 <= mask);
            let bit_pos = i * b;
            let w = bit_pos >> 6;
            let sh = bit_pos & 63;
            words[w] |= v64 << sh;
            if sh + b > 64 {
                words[w + 1] |= v64 >> (64 - sh);
            }
        }
        Self {
            bits,
            len: vals.len(),
            mask,
            words,
        }
    }

    pub(super) fn from_u32_slice(vals: &[u32], bits: u8) -> Self {
        if vals.is_empty() {
            return Self::empty();
        }
        assert!((1..=32).contains(&bits), "BitVec u32 bits out of range");
        let b = usize::from(bits);
        let mask = (1u64 << bits) - 1;
        let n_words = (vals.len() * b).div_ceil(64) + 1;
        let mut words = vec![0u64; n_words];
        for (i, &v) in vals.iter().enumerate() {
            let v64 = u64::from(v);
            debug_assert!(v64 <= mask);
            let bit_pos = i * b;
            let w = bit_pos >> 6;
            let sh = bit_pos & 63;
            words[w] |= v64 << sh;
            if sh + b > 64 {
                words[w + 1] |= v64 >> (64 - sh);
            }
        }
        Self {
            bits,
            len: vals.len(),
            mask,
            words,
        }
    }

    pub(super) fn from_raw(bits: u8, len: usize, mut words: Vec<u64>) -> Option<Self> {
        if len == 0 {
            return Some(Self::empty());
        }
        if bits == 0 || bits > 32 {
            return None;
        }
        let min_words = len.checked_mul(usize::from(bits))?.div_ceil(64);
        if words.len() < min_words {
            return None;
        }
        if words.len() == min_words {
            words.push(0);
        }
        Some(Self {
            bits,
            len,
            mask: (1u64 << bits) - 1,
            words,
        })
    }

    #[inline]
    pub(super) fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub(super) fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    pub(super) fn memory_bytes(&self) -> usize {
        self.words.len() * std::mem::size_of::<u64>()
    }

    /// Number of payload words needed on disk (excluding the in-memory trailing sentinel).
    pub(super) fn disk_words(&self) -> usize {
        if self.len == 0 {
            0
        } else {
            (self.len * usize::from(self.bits)).div_ceil(64)
        }
    }

    #[inline]
    pub(super) fn get_u32(&self, idx: usize) -> u32 {
        debug_assert!(idx < self.len);
        let b = usize::from(self.bits);
        let bit_pos = idx * b;
        let w = bit_pos >> 6;
        let sh = bit_pos & 63;
        let lo = self.words[w] >> sh;
        let hi = if sh == 0 {
            0
        } else {
            self.words[w + 1] << (64 - sh)
        };
        ((lo | hi) & self.mask) as u32
    }

    #[inline]
    pub(super) fn get_u16(&self, idx: usize) -> u16 {
        self.get_u32(idx) as u16
    }

    #[inline]
    pub(super) fn find_sorted_u16(
        &self,
        mut lo: usize,
        mut hi: usize,
        target: u16,
    ) -> Option<usize> {
        if hi <= lo {
            return None;
        }
        if hi - lo <= 16 {
            for idx in lo..hi {
                let k = self.get_u16(idx);
                if k == target {
                    return Some(idx);
                }
                if k > target {
                    return None;
                }
            }
            return None;
        }
        while lo < hi {
            let mid = lo + ((hi - lo) >> 1);
            let k = self.get_u16(mid);
            match k.cmp(&target) {
                std::cmp::Ordering::Equal => return Some(mid),
                std::cmp::Ordering::Less => lo = mid + 1,
                std::cmp::Ordering::Greater => hi = mid,
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bitvec_9bit_toks_and_23bit_offs_exact() {
        let toks: Vec<u16> = (0..5000).map(|i| (i % 501) as u16).collect();
        let bits = BitVec::bits_needed(500);
        assert_eq!(bits, 9);
        let bv = BitVec::from_u16_slice(&toks, bits);
        assert_eq!(bv.len(), toks.len());
        assert!(bv.memory_bytes() < toks.len() * 2 * 60 / 100);
        for (i, &want) in toks.iter().enumerate() {
            assert_eq!(bv.get_u16(i), want);
        }
        let sorted: Vec<u16> = (0..250).map(|i| i * 2).collect();
        let sbv = BitVec::from_u16_slice(&sorted, bits);
        for (i, &v) in sorted.iter().enumerate() {
            assert_eq!(sbv.find_sorted_u16(0, sorted.len(), v), Some(i));
            assert_eq!(sbv.find_sorted_u16(0, sorted.len(), v + 1), None);
        }
        let offs: Vec<u32> = (0..4096).map(|i| i * 1973).collect();
        let obits = BitVec::bits_needed(*offs.last().unwrap());
        assert_eq!(obits, 23);
        let obv = BitVec::from_u32_slice(&offs, obits);
        for (i, &want) in offs.iter().enumerate() {
            assert_eq!(obv.get_u32(i), want);
        }
    }
}
