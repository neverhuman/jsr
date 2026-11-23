//! Streaming 16 kHz mono PCM16 WAV reader for `ultra-engine-endless`.

use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

/// Open a 16 kHz mono PCM16 WAV file, positioning the reader at the start of the `data` payload
/// and returning `(reader, data_bytes_remaining)`.
pub(crate) fn open_wav_pcm16(path: &Path) -> Result<(BufReader<File>, u64), String> {
    let mut r = BufReader::with_capacity(256 * 1024, File::open(path).map_err(|e| e.to_string())?);
    let mut riff = [0u8; 12];
    r.read_exact(&mut riff).map_err(|e| e.to_string())?;
    if &riff[0..4] != b"RIFF" || &riff[8..12] != b"WAVE" {
        return Err(format!("{}: not a RIFF/WAVE file", path.display()));
    }
    let mut fmt_ok = false;
    loop {
        let mut hdr = [0u8; 8];
        r.read_exact(&mut hdr).map_err(|e| e.to_string())?;
        let id = &hdr[0..4];
        let sz = u32::from_le_bytes([hdr[4], hdr[5], hdr[6], hdr[7]]) as u64;
        if id == b"fmt " {
            let mut body = vec![0u8; sz as usize];
            r.read_exact(&mut body).map_err(|e| e.to_string())?;
            if body.len() < 16 {
                return Err(format!("{}: truncated fmt chunk", path.display()));
            }
            let audio_fmt = u16::from_le_bytes([body[0], body[1]]);
            let channels = u16::from_le_bytes([body[2], body[3]]);
            let sr = u32::from_le_bytes([body[4], body[5], body[6], body[7]]);
            let bits = u16::from_le_bytes([body[14], body[15]]);
            if audio_fmt != 1 || channels != 1 || sr != 16_000 || bits != 16 {
                return Err(format!(
                    "{}: expected 16kHz mono PCM16, got fmt={audio_fmt} ch={channels} sr={sr} bits={bits}",
                    path.display()
                ));
            }
            fmt_ok = true;
            if sz % 2 == 1 {
                let mut pad = [0u8; 1];
                let _ = r.read_exact(&mut pad);
            }
        } else if id == b"data" {
            if !fmt_ok {
                return Err(format!("{}: data chunk before fmt chunk", path.display()));
            }
            return Ok((r, sz));
        } else {
            let skip = sz + (sz % 2);
            std::io::copy(&mut r.by_ref().take(skip), &mut std::io::sink())
                .map_err(|e| e.to_string())?;
        }
    }
}
