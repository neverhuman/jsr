//! Safe zero-Python `.npz` shard reader with strict inference/scorer separation.
//!
//! Reads uncompressed NumPy `.npz` archives produced by `python/ultra/ultra/libri.py`
//! (`pcm.npy` `<i2`, `offsets.npy` `<i8`, `uids.npy` `<U*`, `speakers.npy` `<i8`,
//! `texts.npy` `<U*`), supporting ZIP32 and ZIP64 central directories.
//!
//! - [`load_npz_audio_only`] reads **only** `offsets.npy` and the requested slice of
//!   `pcm.npy`, returning opaque numeric utterance indices without ever opening or
//!   parsing `texts.npy` or `uids.npy` (reference-blind contract, spec v1.0-B §12.1, §35).
//! - [`load_npz_references`] reads `uids.npy`, `speakers.npy`, `texts.npy`, and
//!   `offsets.npy` (for duration) for the separate scoring phase.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::FrontendError;

/// Reference-blind audio utterance for inference workers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AudioUtterance {
    /// Zero-based index inside the split shard (opaque ID `u00000`, `u00001`, ...).
    pub index: usize,
    /// 16 kHz mono signed 16-bit PCM.
    pub pcm: Vec<i16>,
}

impl AudioUtterance {
    pub fn opaque_id(&self) -> String {
        format!("u{:05}", self.index)
    }

    pub fn seconds(&self) -> f64 {
        self.pcm.len() as f64 / 16_000.0
    }
}

/// Scorer-side reference metadata for an utterance.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RefUtterance {
    pub index: usize,
    pub uid: String,
    pub speaker: i64,
    pub text: String,
    pub duration_s: f64,
}

#[derive(Debug, Clone, Copy)]
struct ZipEntryLoc {
    local_header_offset: u64,
    uncompressed_size: u64,
}

/// Parse the PKZIP / ZIP64 central directory at the end of `file` to locate stored entries.
fn read_zip_index(file: &mut File) -> Result<BTreeMap<String, ZipEntryLoc>, FrontendError> {
    let file_len = file.seek(SeekFrom::End(0))?;
    let tail_len = file_len.min(65_557) as usize;
    file.seek(SeekFrom::End(-(tail_len as i64)))?;
    let mut tail = vec![0u8; tail_len];
    file.read_exact(&mut tail)?;

    // Scan backwards for End of Central Directory signature 0x06054b50.
    let eocd_pos = (0..=tail.len().saturating_sub(22))
        .rev()
        .find(|&i| &tail[i..i + 4] == b"PK\x05\x06")
        .ok_or_else(|| FrontendError::Npz("missing ZIP EOCD signature".into()))?;

    let mut cd_size = u32::from_le_bytes([
        tail[eocd_pos + 12],
        tail[eocd_pos + 13],
        tail[eocd_pos + 14],
        tail[eocd_pos + 15],
    ]) as u64;
    let mut cd_offset = u32::from_le_bytes([
        tail[eocd_pos + 16],
        tail[eocd_pos + 17],
        tail[eocd_pos + 18],
        tail[eocd_pos + 19],
    ]) as u64;

    // Check for ZIP64 EOCD Locator (20 bytes immediately before EOCD: PK\x06\x07).
    if eocd_pos >= 20 && &tail[eocd_pos - 20..eocd_pos - 16] == b"PK\x06\x07" {
        let z64_eocd_off = u64::from_le_bytes([
            tail[eocd_pos - 12],
            tail[eocd_pos - 11],
            tail[eocd_pos - 10],
            tail[eocd_pos - 9],
            tail[eocd_pos - 8],
            tail[eocd_pos - 7],
            tail[eocd_pos - 6],
            tail[eocd_pos - 5],
        ]);
        file.seek(SeekFrom::Start(z64_eocd_off))?;
        let mut z64 = [0u8; 56];
        file.read_exact(&mut z64)?;
        if &z64[0..4] != b"PK\x06\x06" {
            return Err(FrontendError::Npz("corrupt ZIP64 EOCD record".into()));
        }
        cd_size = u64::from_le_bytes([
            z64[40], z64[41], z64[42], z64[43], z64[44], z64[45], z64[46], z64[47],
        ]);
        cd_offset = u64::from_le_bytes([
            z64[48], z64[49], z64[50], z64[51], z64[52], z64[53], z64[54], z64[55],
        ]);
    }

    file.seek(SeekFrom::Start(cd_offset))?;
    let mut cd = vec![0u8; cd_size as usize];
    file.read_exact(&mut cd)?;

    let mut entries = BTreeMap::new();
    let mut p = 0usize;
    while p + 46 <= cd.len() {
        if &cd[p..p + 4] != b"PK\x01\x02" {
            break;
        }
        let method = u16::from_le_bytes([cd[p + 10], cd[p + 11]]);
        if method != 0 {
            return Err(FrontendError::Npz(format!(
                "compressed .npz method {method} not supported; expected stored (0)"
            )));
        }
        let comp_size = u32::from_le_bytes([cd[p + 20], cd[p + 21], cd[p + 22], cd[p + 23]]) as u64;
        let mut uncomp_size =
            u32::from_le_bytes([cd[p + 24], cd[p + 25], cd[p + 26], cd[p + 27]]) as u64;
        let name_len = u16::from_le_bytes([cd[p + 28], cd[p + 29]]) as usize;
        let extra_len = u16::from_le_bytes([cd[p + 30], cd[p + 31]]) as usize;
        let comment_len = u16::from_le_bytes([cd[p + 32], cd[p + 33]]) as usize;
        let mut local_off =
            u32::from_le_bytes([cd[p + 42], cd[p + 43], cd[p + 44], cd[p + 45]]) as u64;

        let name_end = p + 46 + name_len;
        let extra_end = name_end + extra_len;
        if extra_end > cd.len() {
            return Err(FrontendError::Npz(
                "truncated central directory entry".into(),
            ));
        }
        let name = String::from_utf8_lossy(&cd[p + 46..name_end]).into_owned();
        let extra = &cd[name_end..extra_end];

        // Parse ZIP64 extra field (header id 0x0001) when 32-bit fields hold 0xFFFF_FFFF.
        if uncomp_size == 0xFFFF_FFFF || comp_size == 0xFFFF_FFFF || local_off == 0xFFFF_FFFF {
            let mut ep = 0usize;
            while ep + 4 <= extra.len() {
                let tag = u16::from_le_bytes([extra[ep], extra[ep + 1]]);
                let elen = u16::from_le_bytes([extra[ep + 2], extra[ep + 3]]) as usize;
                let ebody = &extra[ep + 4..(ep + 4 + elen).min(extra.len())];
                if tag == 0x0001 {
                    let mut q = 0usize;
                    if uncomp_size == 0xFFFF_FFFF && q + 8 <= ebody.len() {
                        uncomp_size = u64::from_le_bytes([
                            ebody[q],
                            ebody[q + 1],
                            ebody[q + 2],
                            ebody[q + 3],
                            ebody[q + 4],
                            ebody[q + 5],
                            ebody[q + 6],
                            ebody[q + 7],
                        ]);
                        q += 8;
                    }
                    if comp_size == 0xFFFF_FFFF && q + 8 <= ebody.len() {
                        let _ = comp_size;
                        q += 8;
                    }
                    if local_off == 0xFFFF_FFFF && q + 8 <= ebody.len() {
                        local_off = u64::from_le_bytes([
                            ebody[q],
                            ebody[q + 1],
                            ebody[q + 2],
                            ebody[q + 3],
                            ebody[q + 4],
                            ebody[q + 5],
                            ebody[q + 6],
                            ebody[q + 7],
                        ]);
                    }
                    break;
                }
                ep += 4 + elen;
            }
        }
        entries.insert(
            name,
            ZipEntryLoc {
                local_header_offset: local_off,
                uncompressed_size: uncomp_size,
            },
        );
        p = extra_end + comment_len;
    }
    Ok(entries)
}

#[derive(Debug, Clone)]
struct NpyMeta {
    data_file_offset: u64,
    descr: String,
    len_1d: usize,
}

/// Seek to `loc` and parse the local ZIP header + NumPy `.npy` v1/v2/v3 header.
fn read_npy_meta(file: &mut File, loc: ZipEntryLoc) -> Result<NpyMeta, FrontendError> {
    file.seek(SeekFrom::Start(loc.local_header_offset))?;
    let mut lh = [0u8; 30];
    file.read_exact(&mut lh)?;
    if &lh[0..4] != b"PK\x03\x04" {
        return Err(FrontendError::Npz("bad local file header signature".into()));
    }
    let name_len = u16::from_le_bytes([lh[26], lh[27]]) as u64;
    let extra_len = u16::from_le_bytes([lh[28], lh[29]]) as u64;
    let npy_start = loc.local_header_offset + 30 + name_len + extra_len;

    file.seek(SeekFrom::Start(npy_start))?;
    let mut magic = [0u8; 10];
    file.read_exact(&mut magic)?;
    if &magic[0..6] != b"\x93NUMPY" {
        return Err(FrontendError::Npz("missing \\x93NUMPY magic".into()));
    }
    let major = magic[6];
    let (hdr_len, hdr_pos) = if major == 1 {
        let l = u16::from_le_bytes([magic[8], magic[9]]) as usize;
        (l, npy_start + 10)
    } else {
        file.seek(SeekFrom::Start(npy_start + 8))?;
        let mut l4 = [0u8; 4];
        file.read_exact(&mut l4)?;
        (u32::from_le_bytes(l4) as usize, npy_start + 12)
    };
    file.seek(SeekFrom::Start(hdr_pos))?;
    let mut hdr_bytes = vec![0u8; hdr_len];
    file.read_exact(&mut hdr_bytes)?;
    let hdr = String::from_utf8_lossy(&hdr_bytes);
    if hdr.contains("'fortran_order': True") {
        return Err(FrontendError::Npz(
            "fortran_order arrays unsupported".into(),
        ));
    }
    let descr = extract_quoted_after(&hdr, "'descr':")
        .ok_or_else(|| FrontendError::Npz(format!("missing descr in {hdr}")))?;
    let shape_inner = extract_between(&hdr, "'shape': (", ")")
        .ok_or_else(|| FrontendError::Npz(format!("missing shape in {hdr}")))?;
    let first_dim = shape_inner
        .split(',')
        .next()
        .unwrap_or("")
        .trim()
        .parse::<usize>()
        .map_err(|_| FrontendError::Npz(format!("invalid 1D shape in {hdr}")))?;
    let data_file_offset = hdr_pos + hdr_len as u64;
    let max_end = npy_start + loc.uncompressed_size;
    if data_file_offset > max_end {
        return Err(FrontendError::Npz("npy header exceeds entry size".into()));
    }
    Ok(NpyMeta {
        data_file_offset,
        descr,
        len_1d: first_dim,
    })
}

fn extract_quoted_after(s: &str, key: &str) -> Option<String> {
    let idx = s.find(key)? + key.len();
    let rest = s[idx..].trim_start();
    let quote = rest.chars().next()?;
    if quote != '\'' && quote != '"' {
        return None;
    }
    let body = &rest[quote.len_utf8()..];
    let end = body.find(quote)?;
    Some(body[..end].to_owned())
}

fn extract_between<'a>(s: &'a str, open: &str, close: &str) -> Option<&'a str> {
    let i = s.find(open)? + open.len();
    let j = s[i..].find(close)? + i;
    Some(&s[i..j])
}

fn read_i64_vec(file: &mut File, loc: ZipEntryLoc) -> Result<Vec<i64>, FrontendError> {
    let meta = read_npy_meta(file, loc)?;
    if meta.descr != "<i8" {
        return Err(FrontendError::Npz(format!(
            "expected <i8 array, got {}",
            meta.descr
        )));
    }
    file.seek(SeekFrom::Start(meta.data_file_offset))?;
    let mut raw = vec![0u8; meta.len_1d * 8];
    file.read_exact(&mut raw)?;
    Ok(raw
        .chunks_exact(8)
        .map(|b| i64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]))
        .collect())
}

fn read_unicode_vec(
    file: &mut File,
    loc: ZipEntryLoc,
    start: usize,
    end: usize,
) -> Result<Vec<String>, FrontendError> {
    let meta = read_npy_meta(file, loc)?;
    let width: usize = meta
        .descr
        .strip_prefix("<U")
        .and_then(|n| n.parse().ok())
        .ok_or_else(|| FrontendError::Npz(format!("expected <U* array, got {}", meta.descr)))?;
    let end = end.min(meta.len_1d);
    if start >= end {
        return Ok(Vec::new());
    }
    let stride_bytes = width * 4;
    file.seek(SeekFrom::Start(
        meta.data_file_offset + (start * stride_bytes) as u64,
    ))?;
    let count = end - start;
    let mut raw = vec![0u8; count * stride_bytes];
    file.read_exact(&mut raw)?;
    let mut out = Vec::with_capacity(count);
    for rec in raw.chunks_exact(stride_bytes) {
        let mut s = String::new();
        for ch_bytes in rec.chunks_exact(4) {
            let code = u32::from_le_bytes([ch_bytes[0], ch_bytes[1], ch_bytes[2], ch_bytes[3]]);
            if code != 0 {
                if let Some(c) = char::from_u32(code) {
                    s.push(c);
                }
            }
        }
        // Trim trailing NULs only (embedded NULs already skipped above).
        out.push(s);
    }
    Ok(out)
}

/// Reference-blind loader: reads ONLY `offsets.npy` and the `[offset .. offset + limit]`
/// slice of `pcm.npy`. Never reads `texts.npy`, `uids.npy`, or `speakers.npy`.
pub fn load_npz_audio_only(
    npz_path: &Path,
    offset: usize,
    limit: Option<usize>,
) -> Result<Vec<AudioUtterance>, FrontendError> {
    let mut file = File::open(npz_path)?;
    let index = read_zip_index(&mut file)?;
    let off_loc = *index
        .get("offsets.npy")
        .ok_or_else(|| FrontendError::Npz("missing offsets.npy".into()))?;
    let pcm_loc = *index
        .get("pcm.npy")
        .ok_or_else(|| FrontendError::Npz("missing pcm.npy".into()))?;

    let offsets = read_i64_vec(&mut file, off_loc)?;
    if offsets.is_empty() {
        return Ok(Vec::new());
    }
    let total_utts = offsets.len() - 1;
    if offset >= total_utts {
        return Ok(Vec::new());
    }
    let end_utt = match limit {
        Some(n) => offset.saturating_add(n).min(total_utts),
        None => total_utts,
    };

    let pcm_meta = read_npy_meta(&mut file, pcm_loc)?;
    if pcm_meta.descr != "<i2" {
        return Err(FrontendError::Npz(format!(
            "expected <i2 pcm.npy, got {}",
            pcm_meta.descr
        )));
    }
    let first_sample = offsets[offset].max(0) as usize;
    let last_sample = offsets[end_utt].max(0) as usize;
    if last_sample > pcm_meta.len_1d || first_sample > last_sample {
        return Err(FrontendError::Npz("offsets out of pcm.npy bounds".into()));
    }
    let sample_span = last_sample - first_sample;
    file.seek(SeekFrom::Start(
        pcm_meta.data_file_offset + (first_sample * 2) as u64,
    ))?;
    let mut raw_pcm = vec![0u8; sample_span * 2];
    file.read_exact(&mut raw_pcm)?;

    let mut utts = Vec::with_capacity(end_utt - offset);
    for i in offset..end_utt {
        let rel_start = (offsets[i] as usize - first_sample) * 2;
        let rel_end = (offsets[i + 1] as usize - first_sample) * 2;
        let pcm: Vec<i16> = raw_pcm[rel_start..rel_end]
            .chunks_exact(2)
            .map(|b| i16::from_le_bytes([b[0], b[1]]))
            .collect();
        utts.push(AudioUtterance { index: i, pcm });
    }
    Ok(utts)
}

/// Scorer-only loader: reads `offsets.npy`, `uids.npy`, `speakers.npy`, and `texts.npy`
/// without reading `pcm.npy`.
pub fn load_npz_references(
    npz_path: &Path,
    offset: usize,
    limit: Option<usize>,
) -> Result<Vec<RefUtterance>, FrontendError> {
    let mut file = File::open(npz_path)?;
    let index = read_zip_index(&mut file)?;
    let off_loc = *index
        .get("offsets.npy")
        .ok_or_else(|| FrontendError::Npz("missing offsets.npy".into()))?;
    let uid_loc = *index
        .get("uids.npy")
        .ok_or_else(|| FrontendError::Npz("missing uids.npy".into()))?;
    let spk_loc = *index
        .get("speakers.npy")
        .ok_or_else(|| FrontendError::Npz("missing speakers.npy".into()))?;
    let txt_loc = *index
        .get("texts.npy")
        .ok_or_else(|| FrontendError::Npz("missing texts.npy".into()))?;

    let offsets = read_i64_vec(&mut file, off_loc)?;
    if offsets.is_empty() {
        return Ok(Vec::new());
    }
    let total_utts = offsets.len() - 1;
    if offset >= total_utts {
        return Ok(Vec::new());
    }
    let end_utt = match limit {
        Some(n) => offset.saturating_add(n).min(total_utts),
        None => total_utts,
    };

    let speakers_all = read_i64_vec(&mut file, spk_loc)?;
    let uids = read_unicode_vec(&mut file, uid_loc, offset, end_utt)?;
    let texts = read_unicode_vec(&mut file, txt_loc, offset, end_utt)?;
    if uids.len() != end_utt - offset
        || texts.len() != end_utt - offset
        || speakers_all.len() < end_utt
    {
        return Err(FrontendError::Npz(
            "mismatched metadata array lengths".into(),
        ));
    }

    let mut out = Vec::with_capacity(end_utt - offset);
    for (k, i) in (offset..end_utt).enumerate() {
        let samples = (offsets[i + 1] - offsets[i]).max(0) as f64;
        out.push(RefUtterance {
            index: i,
            uid: uids[k].clone(),
            speaker: speakers_all[i],
            text: texts[k].clone(),
            duration_s: samples / 16_000.0,
        });
    }
    Ok(out)
}
