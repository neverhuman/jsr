//! ULTRA cost-law benchmarking (spec Appendix A, W01): the `wstream` data generator and
//! sweep plumbing, and the least-squares fit `ps per weight per pass = a + c F`.

pub mod fit;
pub mod gen;

/// Activation width of the Appendix A benchmark (bytes per frame).
pub const DIN: usize = 1024;
/// Largest frame count in the sweep; activations are generated for this many frames.
pub const MAX_FRAMES: usize = 16;

/// Parse a weight count such as `1M`, `64M` or `1536M` (M = 2^20).
pub fn parse_size(s: &str) -> Option<usize> {
    let n: usize = s.strip_suffix('M')?.parse().ok()?;
    let w = n.checked_mul(1 << 20)?;
    (n > 0 && w.is_multiple_of(DIN)).then_some(w)
}

/// Inverse of [`parse_size`].
pub fn size_name(w: usize) -> String {
    format!("{}M", w >> 20)
}

/// Checksums printed in Appendix A (`wstream.c`, max size 1536M) for (size, F). A sweep
/// whose largest size is 1536M regenerates the same data and must reproduce them.
pub const APPENDIX_A_CHECKSUMS: [(&str, usize, i64); 24] = [
    ("1M", 1, -2382),
    ("1M", 2, -17142),
    ("1M", 3, 36833),
    ("1M", 4, 21659),
    ("1M", 6, -8181),
    ("1M", 8, -76071),
    ("1M", 12, -169674),
    ("1M", 16, -242266),
    ("64M", 1, -591026),
    ("64M", 2, -1222025),
    ("64M", 3, -1656515),
    ("64M", 4, -2293234),
    ("64M", 6, -3868138),
    ("64M", 8, -4661303),
    ("64M", 12, -6520161),
    ("64M", 16, -9251333),
    ("1536M", 1, -1644903),
    ("1536M", 2, -2181117),
    ("1536M", 3, -6017931),
    ("1536M", 4, -5822672),
    ("1536M", 6, -12027474),
    ("1536M", 8, -19606366),
    ("1536M", 12, -31876366),
    ("1536M", 16, -43717451),
];

/// The Appendix A checksum for `(size, frames)`, if it printed one.
pub fn appendix_a_checksum(size: &str, frames: usize) -> Option<i64> {
    APPENDIX_A_CHECKSUMS
        .iter()
        .find(|(s, f, _)| *s == size && *f == frames)
        .map(|&(_, _, c)| c)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes_round_trip() {
        for s in ["1M", "64M", "1536M", "4096M"] {
            assert_eq!(size_name(parse_size(s).unwrap()), s);
        }
        assert_eq!(parse_size("1536M"), Some(1536 << 20));
        assert!(parse_size("0M").is_none());
        assert!(parse_size("12").is_none());
        assert!(parse_size("xM").is_none());
    }

    #[test]
    fn appendix_a_lookup() {
        assert_eq!(appendix_a_checksum("1M", 1), Some(-2382));
        assert_eq!(appendix_a_checksum("4096M", 1), None);
    }
}
