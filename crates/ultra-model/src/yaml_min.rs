//! Minimal zero-dependency parser for indentation-based YAML candidate manifests into
//! `serde_json::Value` (supporting `---` multi-doc streams, `#` comments, `>-` / `|` block
//! scalars, nested maps, and `- item` lists as used in `docs/ultra/candidates/*.yaml`).

use serde_json::{Map, Number, Value};

use crate::ModelError;

#[derive(Debug, Clone)]
struct Line<'a> {
    indent: usize,
    text: &'a str,
}

/// Strip an inline `# comment` outside single or double quotes.
fn strip_comment(raw: &str) -> &str {
    let mut in_single = false;
    let mut in_double = false;
    let bytes = raw.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'\\' if in_double => {
                i += 2;
                continue;
            }
            b'\'' if !in_double => in_single = !in_single,
            b'"' if !in_single => in_double = !in_double,
            b'#' if !in_single && !in_double && (i == 0 || bytes[i - 1].is_ascii_whitespace()) => {
                return raw[..i].trim_end();
            }
            _ => {}
        }
        i += 1;
    }
    raw.trim_end()
}

fn parse_scalar(s: &str) -> Value {
    let t = s.trim();
    if t.is_empty() || t == "null" || t == "~" {
        return Value::Null;
    }
    if t == "true" {
        return Value::Bool(true);
    }
    if t == "false" {
        return Value::Bool(false);
    }
    if t == "[]" {
        return Value::Array(Vec::new());
    }
    if t == "{}" {
        return Value::Object(Map::new());
    }
    if (t.starts_with('"') && t.ends_with('"') && t.len() >= 2)
        || (t.starts_with('\'') && t.ends_with('\'') && t.len() >= 2)
    {
        let inner = &t[1..t.len() - 1];
        if t.starts_with('"') {
            return Value::String(inner.replace("\\\"", "\"").replace("\\\\", "\\"));
        }
        return Value::String(inner.replace("''", "'"));
    }
    if let Ok(i) = t.parse::<i64>() {
        return Value::Number(Number::from(i));
    }
    if let Ok(u) = t.parse::<u64>() {
        return Value::Number(Number::from(u));
    }
    if let Ok(f) = t.parse::<f64>() {
        if let Some(n) = Number::from_f64(f) {
            return Value::Number(n);
        }
    }
    Value::String(t.to_owned())
}

/// Split `key: rest` where `:` is outside quotes and followed by end-of-line or whitespace.
fn split_key_val(text: &str) -> Option<(&str, &str)> {
    let mut in_single = false;
    let mut in_double = false;
    let bytes = text.as_bytes();
    for i in 0..bytes.len() {
        match bytes[i] {
            b'\'' if !in_double => in_single = !in_single,
            b'"' if !in_single => in_double = !in_double,
            b':' if !in_single
                && !in_double
                && (i + 1 == bytes.len() || bytes[i + 1].is_ascii_whitespace()) =>
            {
                return Some((text[..i].trim(), text[i + 1..].trim()));
            }
            _ => {}
        }
    }
    None
}

fn parse_node(lines: &[Line<'_>], idx: &mut usize, min_indent: usize) -> Result<Value, ModelError> {
    if *idx >= lines.len() || lines[*idx].indent < min_indent {
        return Ok(Value::Null);
    }
    let base_indent = lines[*idx].indent;
    if lines[*idx].text.starts_with("- ") || lines[*idx].text == "-" {
        let mut arr = Vec::new();
        while *idx < lines.len() && lines[*idx].indent == base_indent {
            let t = lines[*idx].text;
            let Some(after_dash) = t.strip_prefix('-') else {
                break;
            };
            let rest = after_dash.trim_start();
            let item_indent = base_indent + 2;
            *idx += 1;
            if rest.is_empty() {
                arr.push(parse_node(lines, idx, base_indent + 1)?);
            } else if let Some((k, v)) = split_key_val(rest) {
                let mut map = Map::new();
                let first_val = parse_rhs(lines, idx, item_indent, v)?;
                map.insert(k.to_owned(), first_val);
                if *idx < lines.len() && lines[*idx].indent >= item_indent {
                    if let Value::Object(extra) = parse_node(lines, idx, item_indent)? {
                        for (ek, ev) in extra {
                            map.insert(ek, ev);
                        }
                    }
                }
                arr.push(Value::Object(map));
            } else {
                arr.push(parse_scalar(rest));
            }
        }
        Ok(Value::Array(arr))
    } else {
        let mut map = Map::new();
        while *idx < lines.len() && lines[*idx].indent == base_indent {
            let t = lines[*idx].text;
            if t.starts_with('-') {
                break;
            }
            let Some((k, v)) = split_key_val(t) else {
                return Err(ModelError::Parse(format!("expected `key: value` at `{t}`")));
            };
            *idx += 1;
            let val = parse_rhs(lines, idx, base_indent + 1, v)?;
            map.insert(k.to_owned(), val);
        }
        Ok(Value::Object(map))
    }
}

fn parse_rhs(
    lines: &[Line<'_>],
    idx: &mut usize,
    child_min_indent: usize,
    rhs: &str,
) -> Result<Value, ModelError> {
    if matches!(rhs, ">" | ">-" | "|" | "|-") {
        let fold = rhs.starts_with('>');
        let mut parts = Vec::new();
        let block_indent = lines
            .get(*idx)
            .map(|l| l.indent)
            .unwrap_or(child_min_indent);
        if block_indent >= child_min_indent {
            while *idx < lines.len() && lines[*idx].indent >= block_indent {
                parts.push(lines[*idx].text);
                *idx += 1;
            }
        }
        let joined = if fold {
            parts.join(" ")
        } else {
            parts.join("\n")
        };
        Ok(Value::String(joined))
    } else if rhs.is_empty() {
        if *idx < lines.len() && lines[*idx].indent >= child_min_indent {
            parse_node(lines, idx, child_min_indent)
        } else {
            Ok(Value::Null)
        }
    } else {
        Ok(parse_scalar(rhs))
    }
}

/// Parse a YAML or JSON text into one or more JSON documents (`---` separates YAML docs).
pub fn parse_yaml_or_json_docs(input: &str) -> Result<Vec<Value>, ModelError> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Ok(Vec::new());
    }
    if trimmed.starts_with('{') || trimmed.starts_with('[') {
        let v: Value =
            serde_json::from_str(trimmed).map_err(|e| ModelError::Parse(e.to_string()))?;
        return Ok(vec![v]);
    }
    let mut docs = Vec::new();
    let mut cur: Vec<Line<'_>> = Vec::new();
    for raw in input.lines() {
        let stripped = strip_comment(raw);
        let ws = stripped.len() - stripped.trim_start_matches(' ').len();
        let text = stripped.trim();
        if text.is_empty() {
            continue;
        }
        if ws == 0 && text == "---" {
            if !cur.is_empty() {
                let mut idx = 0;
                docs.push(parse_node(&cur, &mut idx, 0)?);
                cur.clear();
            }
            continue;
        }
        if ws == 0 && text == "..." {
            continue;
        }
        cur.push(Line { indent: ws, text });
    }
    if !cur.is_empty() {
        let mut idx = 0;
        docs.push(parse_node(&cur, &mut idx, 0)?);
    }
    Ok(docs)
}
