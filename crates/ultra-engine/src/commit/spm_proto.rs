//! Minimal pure-Rust protobuf wire decoder for SentencePiece `ModelProto`.

use crate::EngineError;

pub struct RawSpmModel {
    pub model_type_id: u32,
    pub add_dummy_prefix: bool,
    pub escape_whitespaces: bool,
    pub pieces: Vec<(String, f32, u32)>,
}

fn read_varint(buf: &[u8], pos: &mut usize) -> Result<u64, EngineError> {
    let (mut val, mut shift) = (0u64, 0u32);
    loop {
        let b = *buf
            .get(*pos)
            .ok_or_else(|| EngineError::Format("truncated protobuf varint".into()))?;
        *pos += 1;
        val |= u64::from(b & 0x7f) << shift;
        if (b & 0x80) == 0 {
            return Ok(val);
        }
        shift += 7;
        if shift >= 64 {
            return Err(EngineError::Format("protobuf varint overflow".into()));
        }
    }
}

fn read_bytes<'a>(buf: &'a [u8], pos: &mut usize) -> Result<&'a [u8], EngineError> {
    let len = read_varint(buf, pos)? as usize;
    let end = pos
        .checked_add(len)
        .ok_or_else(|| EngineError::Format("protobuf length overflow".into()))?;
    let slice = buf
        .get(*pos..end)
        .ok_or_else(|| EngineError::Format("truncated protobuf bytes".into()))?;
    *pos = end;
    Ok(slice)
}

fn skip_field(buf: &[u8], pos: &mut usize, wire: u64) -> Result<(), EngineError> {
    match wire {
        0 => {
            let _ = read_varint(buf, pos)?;
        }
        1 => {
            *pos = pos
                .checked_add(8)
                .filter(|&e| e <= buf.len())
                .ok_or_else(|| EngineError::Format("truncated 64-bit field".into()))?;
        }
        2 => {
            let _ = read_bytes(buf, pos)?;
        }
        5 => {
            *pos = pos
                .checked_add(4)
                .filter(|&e| e <= buf.len())
                .ok_or_else(|| EngineError::Format("truncated 32-bit field".into()))?;
        }
        _ => return Err(EngineError::Format(format!("unsupported wire {wire}"))),
    }
    Ok(())
}

fn parse_piece(buf: &[u8]) -> Result<(String, f32, u32), EngineError> {
    let (mut pos, mut piece, mut score, mut ptype) = (0usize, String::new(), 0.0f32, 1u32);
    while pos < buf.len() {
        let tag = read_varint(buf, &mut pos)?;
        match (tag >> 3, tag & 7) {
            (1, 2) => {
                let b = read_bytes(buf, &mut pos)?;
                piece = std::str::from_utf8(b)
                    .map_err(|e| EngineError::Format(format!("bad piece utf8: {e}")))?
                    .to_owned();
            }
            (2, 5) => {
                let b = buf
                    .get(pos..pos + 4)
                    .ok_or_else(|| EngineError::Format("truncated score".into()))?;
                score = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
                pos += 4;
            }
            (3, 0) => ptype = read_varint(buf, &mut pos)? as u32,
            (_, w) => skip_field(buf, &mut pos, w)?,
        }
    }
    Ok((piece, score, ptype))
}

pub fn decode_model_proto(bytes: &[u8]) -> Result<RawSpmModel, EngineError> {
    let (mut pos, mut pieces) = (0usize, Vec::new());
    let (mut model_type_id, mut add_dummy_prefix, mut escape_whitespaces) = (1u32, true, true);
    while pos < bytes.len() {
        let tag = read_varint(bytes, &mut pos)?;
        match (tag >> 3, tag & 7) {
            (1, 2) => pieces.push(parse_piece(read_bytes(bytes, &mut pos)?)?),
            (2, 2) => {
                let sub = read_bytes(bytes, &mut pos)?;
                let mut sp = 0usize;
                while sp < sub.len() {
                    let st = read_varint(sub, &mut sp)?;
                    if (st >> 3, st & 7) == (3, 0) {
                        model_type_id = read_varint(sub, &mut sp)? as u32;
                    } else {
                        skip_field(sub, &mut sp, st & 7)?;
                    }
                }
            }
            (3, 2) => {
                let sub = read_bytes(bytes, &mut pos)?;
                let mut sp = 0usize;
                while sp < sub.len() {
                    let st = read_varint(sub, &mut sp)?;
                    match (st >> 3, st & 7) {
                        (3, 0) => add_dummy_prefix = read_varint(sub, &mut sp)? != 0,
                        (5, 0) => escape_whitespaces = read_varint(sub, &mut sp)? != 0,
                        (_, w) => skip_field(sub, &mut sp, w)?,
                    }
                }
            }
            (_, w) => skip_field(bytes, &mut pos, w)?,
        }
    }
    Ok(RawSpmModel {
        model_type_id,
        add_dummy_prefix,
        escape_whitespaces,
        pieces,
    })
}
