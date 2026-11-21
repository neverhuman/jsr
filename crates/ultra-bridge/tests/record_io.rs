//! Complete records stay visible immediately, including on short-write sinks.

use std::io::{self, Write};

use ultra_bridge::records::json_line;

#[derive(Default)]
struct Sink {
    bytes: Vec<u8>,
    writes: usize,
    max_write: Option<usize>,
    fail_after: Option<usize>,
}

impl Write for Sink {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.writes += 1;
        if self
            .fail_after
            .is_some_and(|limit| self.bytes.len() >= limit)
        {
            return Err(io::Error::other("injected record sink failure"));
        }
        let n = bytes.len().min(self.max_write.unwrap_or(usize::MAX));
        self.bytes.extend_from_slice(&bytes[..n]);
        Ok(n)
    }

    fn flush(&mut self) -> io::Result<()> {
        panic!("records must be delivered without requiring a later flush")
    }
}

#[test]
fn one_record_write_preserves_unicode_escapes_and_line_boundaries() {
    let row = serde_json::json!({"text":"café\n東京\t\"quote\"", "samples":320, "final":false});
    let mut sink = Sink::default();
    json_line(&mut sink, &row).expect("write first complete record");
    assert_eq!(sink.writes, 1, "avoid per-fragment file syscalls");
    assert_eq!(sink.bytes.iter().filter(|&&b| b == b'\n').count(), 1);
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&sink.bytes).unwrap(),
        row
    );
    json_line(&mut sink, &serde_json::json!({"final":true})).unwrap();
    assert_eq!(sink.writes, 2);
    assert_eq!(sink.bytes.iter().filter(|&&b| b == b'\n').count(), 2);
}

#[test]
fn short_writes_finish_the_record_and_sink_failure_is_an_error() {
    let row = serde_json::json!({"text":"partial writes must not lose output"});
    let mut sink = Sink {
        max_write: Some(3),
        ..Sink::default()
    };
    json_line(&mut sink, &row).expect("write_all retries partial writes");
    assert!(sink.writes > 1);
    assert_eq!(sink.bytes.last(), Some(&b'\n'));
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&sink.bytes).unwrap(),
        row
    );
    let mut failed = Sink {
        max_write: Some(3),
        fail_after: Some(6),
        ..Sink::default()
    };
    assert!(json_line(&mut failed, &row).is_err());
    assert_ne!(
        failed.bytes.last(),
        Some(&b'\n'),
        "failed line is never complete"
    );
}
