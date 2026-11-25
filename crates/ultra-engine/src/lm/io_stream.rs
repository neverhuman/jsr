//! Streaming two-pass SHA-256 verifier and exact-capacity reader for `ultra.lm-ngram/v1|v2`.

use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

use sha2::{Digest, Sha256};

use super::super::bitvec::BitVec;
use crate::EngineError;

const IO_BUF_BYTES: usize = 65_536;

/// Verify trailing 32-byte SHA-256 digest in `64 KiB` chunks without buffering the file in RAM.
/// Returns `(payload_len, whole_file_sha256_hex)`.
pub(super) fn verify_sha256_streaming(path: &Path) -> Result<(usize, String), EngineError> {
    let file_len = std::fs::metadata(path)?.len();
    if file_len < 32 + 28 {
        return Err(EngineError::Format("LM file truncated".into()));
    }
    let payload_len = usize::try_from(file_len - 32)
        .map_err(|_| EngineError::Format("LM file exceeds addressable size".into()))?;
    let mut f = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; IO_BUF_BYTES];
    let mut rem = payload_len;
    while rem > 0 {
        let take = rem.min(buf.len());
        f.read_exact(&mut buf[..take])?;
        hasher.update(&buf[..take]);
        rem -= take;
    }
    let mut tail = [0u8; 32];
    f.read_exact(&mut tail)?;
    let mut whole_hasher = hasher.clone();
    if tail.as_slice() != hasher.finalize().as_slice() {
        return Err(EngineError::Integrity(
            "LM file SHA-256 verification failed".into(),
        ));
    }
    whole_hasher.update(tail);
    Ok((payload_len, format!("{:x}", whole_hasher.finalize())))
}

pub(super) fn open_payload_reader(
    path: &Path,
    payload_len: usize,
) -> Result<StreamReader<BufReader<File>>, EngineError> {
    Ok(StreamReader::new(
        BufReader::with_capacity(IO_BUF_BYTES, File::open(path)?),
        payload_len,
    ))
}

pub(super) fn read_bitvec_stream<R: Read>(
    cur: &mut StreamReader<R>,
    len: usize,
) -> Result<BitVec, EngineError> {
    let mut b = [0u8; 1];
    cur.read_exact_buf(&mut b)?;
    let bits = b[0];
    let nw = cur.read_u32()? as usize;
    let words = cur.read_u64_vec_with_sentinel_cap(nw)?;
    BitVec::from_raw(bits, len, words).ok_or_else(|| {
        EngineError::Format(format!(
            "corrupt bit-packed vector bits={bits} len={len} words={nw}"
        ))
    })
}

pub(super) struct StreamReader<R: Read> {
    inner: R,
    rem: usize,
}

impl<R: Read> StreamReader<R> {
    pub(super) fn new(inner: R, rem: usize) -> Self {
        Self { inner, rem }
    }

    pub(super) fn remaining(&self) -> usize {
        self.rem
    }

    fn take_rem(&mut self, n: usize) -> Result<(), EngineError> {
        if n > self.rem {
            return Err(EngineError::Format("unexpected EOF reading LM file".into()));
        }
        self.rem -= n;
        Ok(())
    }

    pub(super) fn read_exact_buf(&mut self, dst: &mut [u8]) -> Result<(), EngineError> {
        self.take_rem(dst.len())?;
        self.inner.read_exact(dst)?;
        Ok(())
    }

    pub(super) fn read_u32(&mut self) -> Result<u32, EngineError> {
        let mut b = [0u8; 4];
        self.read_exact_buf(&mut b)?;
        Ok(u32::from_le_bytes(b))
    }

    pub(super) fn read_f32(&mut self) -> Result<f32, EngineError> {
        let mut b = [0u8; 4];
        self.read_exact_buf(&mut b)?;
        Ok(f32::from_le_bytes(b))
    }

    pub(super) fn read_u8_vec(&mut self, n: usize) -> Result<Vec<u8>, EngineError> {
        self.take_rem(n)?;
        let mut out = vec![0u8; n];
        self.inner.read_exact(&mut out)?;
        Ok(out)
    }

    pub(super) fn read_u16_vec(&mut self, n: usize) -> Result<Vec<u16>, EngineError> {
        let bytes = n
            .checked_mul(2)
            .ok_or_else(|| EngineError::Format("LM u16 overflow".into()))?;
        self.take_rem(bytes)?;
        let mut out = Vec::with_capacity(n);
        let mut buf = [0u8; 32_768];
        let mut left = n;
        while left > 0 {
            let chunk = left.min(buf.len() / 2);
            self.inner.read_exact(&mut buf[..chunk * 2])?;
            for c in buf[..chunk * 2].chunks_exact(2) {
                out.push(u16::from_le_bytes([c[0], c[1]]));
            }
            left -= chunk;
        }
        Ok(out)
    }

    pub(super) fn read_u32_vec(&mut self, n: usize) -> Result<Vec<u32>, EngineError> {
        let bytes = n
            .checked_mul(4)
            .ok_or_else(|| EngineError::Format("LM u32 overflow".into()))?;
        self.take_rem(bytes)?;
        let mut out = Vec::with_capacity(n);
        let mut buf = [0u8; 32_768];
        let mut left = n;
        while left > 0 {
            let chunk = left.min(buf.len() / 4);
            self.inner.read_exact(&mut buf[..chunk * 4])?;
            for c in buf[..chunk * 4].chunks_exact(4) {
                out.push(u32::from_le_bytes([c[0], c[1], c[2], c[3]]));
            }
            left -= chunk;
        }
        Ok(out)
    }

    /// Pre-allocate `n + 1` words so `BitVec::from_raw`'s trailing zero sentinel push never
    /// triggers a `Vec` doubling reallocation.
    pub(super) fn read_u64_vec_with_sentinel_cap(
        &mut self,
        n: usize,
    ) -> Result<Vec<u64>, EngineError> {
        let bytes = n
            .checked_mul(8)
            .ok_or_else(|| EngineError::Format("LM u64 overflow".into()))?;
        self.take_rem(bytes)?;
        let mut out = Vec::with_capacity(n.saturating_add(1));
        let mut buf = [0u8; 32_768];
        let mut left = n;
        while left > 0 {
            let chunk = left.min(buf.len() / 8);
            self.inner.read_exact(&mut buf[..chunk * 8])?;
            for c in buf[..chunk * 8].chunks_exact(8) {
                out.push(u64::from_le_bytes([
                    c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7],
                ]));
            }
            left -= chunk;
        }
        Ok(out)
    }

    pub(super) fn read_f32_vec(&mut self, n: usize) -> Result<Vec<f32>, EngineError> {
        let bytes = n
            .checked_mul(4)
            .ok_or_else(|| EngineError::Format("LM f32 overflow".into()))?;
        self.take_rem(bytes)?;
        let mut out = Vec::with_capacity(n);
        let mut buf = [0u8; 32_768];
        let mut left = n;
        while left > 0 {
            let chunk = left.min(buf.len() / 4);
            self.inner.read_exact(&mut buf[..chunk * 4])?;
            for c in buf[..chunk * 4].chunks_exact(4) {
                out.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
            }
            left -= chunk;
        }
        Ok(out)
    }
}

pub(super) fn validate_csr_slice(offs: &[u32], child_len: u32) -> Result<(), EngineError> {
    if offs.is_empty() || *offs.last().unwrap_or(&0) != child_len {
        return Err(EngineError::Format("CSR offset sentinel mismatch".into()));
    }
    for w in offs.windows(2) {
        if w[0] > w[1] {
            return Err(EngineError::Format("non-monotone CSR offsets".into()));
        }
    }
    Ok(())
}

pub(super) fn validate_csr_bitvec(offs: &BitVec, child_len: u32) -> Result<(), EngineError> {
    if offs.is_empty() || offs.get_u32(offs.len() - 1) != child_len {
        return Err(EngineError::Format("CSR offset sentinel mismatch".into()));
    }
    let mut prev = offs.get_u32(0);
    for i in 1..offs.len() {
        let cur = offs.get_u32(i);
        if prev > cur {
            return Err(EngineError::Format("non-monotone CSR offsets".into()));
        }
        prev = cur;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lm::NgramLm;

    const MINI_LM: &str = "\\data\\\nngram 1=6\nngram 2=4\nngram 3=2\n\n\\1-grams:\n-4.0\t<unk>\n-99.0\t<s>\t-0.3\n-1.2\t</s>\n-1.0\t2\t-0.2\n-1.5\t3\t-0.4\n-2.0\t4\t-0.1\n\n\\2-grams:\n-0.5\t<s> 2\t-0.15\n-0.8\t2 3\t-0.25\n-0.6\t3 4\t-0.35\n-0.9\t4 </s>\n\n\\3-grams:\n-0.2\t<s> 2 3\n-0.3\t2 3 4\n\n\\end\\\n";
    const MINI_LODR: &str = "\\data\\\nngram 1=4\nngram 2=2\n\n\\1-grams:\n-3.5\t<unk>\n-99.0\t<s>\t-0.2\n-1.1\t2\t-0.3\n-1.4\t3\n\n\\2-grams:\n-0.7\t<s> 2\n-0.5\t2 3\n\n\\end\\\n";

    #[test]
    fn streaming_load_matches_v1_v2_and_avoids_bitvec_doubling() {
        // Verify exact sentinel-headroom capacity on large BitVec (no 2x Vec doubling).
        let big: Vec<u16> = (0..10_000).map(|i| (i % 501) as u16).collect();
        let bv = BitVec::from_u16_slice(&big, 9);
        let nw = bv.disk_words();
        let mut wire = Vec::new();
        wire.push(bv.bits);
        wire.extend_from_slice(&(nw as u32).to_le_bytes());
        for &w in &bv.words[..nw] {
            wire.extend_from_slice(&w.to_le_bytes());
        }
        let mut sr = StreamReader::new(wire.as_slice(), wire.len());
        let loaded_bv = read_bitvec_stream(&mut sr, big.len()).expect("read bitvec");
        assert_eq!(loaded_bv, bv);
        assert!(
            loaded_bv.words.capacity() <= nw + 2,
            "expected no doubling: cap={} nw={}",
            loaded_bv.words.capacity(),
            nw
        );

        let pid = std::process::id();
        for (q8, eos) in [(true, false), (true, true), (false, true)] {
            let lm = NgramLm::from_arpa_str_with_eos(MINI_LM, Some(MINI_LODR), 8, q8, eos)
                .expect("build");
            let p_v2 = std::env::temp_dir().join(format!("stream_lm_{pid}_{q8}_{eos}.v2.ulm"));
            let p_v1 = std::env::temp_dir().join(format!("stream_lm_{pid}_{q8}_{eos}.v1.ulm"));
            lm.save(&p_v2).expect("save v2");
            lm.save_v1(&p_v1).expect("save v1");
            for p in [&p_v2, &p_v1] {
                let got = NgramLm::load_streaming(p).expect("load_streaming");
                assert_eq!(got.memory_bytes(), lm.memory_bytes());
                for hist in [&[][..], &[2][..], &[2, 3][..], &[3, 4][..]] {
                    let c0 = lm.ctx_state(hist);
                    let c1 = got.ctx_state(hist);
                    assert_eq!(lm.eos_logp(&c0).to_bits(), got.eos_logp(&c1).to_bits());
                    for k in 1..8 {
                        assert_eq!(lm.lm_logp(&c0, k).to_bits(), got.lm_logp(&c1, k).to_bits());
                        assert_eq!(
                            lm.lodr_logp(&c0, k).to_bits(),
                            got.lodr_logp(&c1, k).to_bits()
                        );
                    }
                }
            }
            let mut raw = std::fs::read(&p_v2).expect("read");
            raw[16] ^= 0xff;
            std::fs::write(&p_v2, &raw).expect("corrupt");
            assert!(matches!(
                NgramLm::load_streaming(&p_v2),
                Err(EngineError::Integrity(_))
            ));
            let _ = std::fs::remove_file(&p_v2);
            let _ = std::fs::remove_file(&p_v1);
        }
    }
}
