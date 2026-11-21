//! Pure zero-heap streaming SHA-256 helper over [`std::io::Read`] (`64 KiB` stack buffer).

use sha2::{Digest, Sha256};

/// Compute lower-case hex SHA-256 over any [`std::io::Read`] without heap buffer allocation.
pub fn sha256_reader<R: std::io::Read>(reader: &mut R) -> std::io::Result<String> {
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 65_536];
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    let d = hasher.finalize();
    Ok(d.iter().map(|b| format!("{b:02x}")).collect())
}
