//! `ultra-engine-endless`: run a single continuous `EngineStream` over one or more sequential
//! 16 kHz mono PCM16 WAV files (e.g., the 27.6 h always-on endless stream `part-1..4.wav`),
//! emitting compact committed-word traces (`{"audio_s": t, "words": [...]}` matching
//! `score_alwayson.py`) and 1 Hz `/proc/self/status` `VmRSS`/`VmHWM` samples.
//!
//! ```text
//! ultra-engine-endless --weights <dir> --precision int8 \
//!     --wavs part-1.wav,part-2.wav,part-3.wav,part-4.wav \
//!     --trace-out trace.jsonl [--rss-out rss.jsonl] \
//!     [--endpoint-reset-ms 0] [--chunk 4] [--until-s 3600]
//! ```

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

use serde_json::json;
use ultra_engine::{run_info, Engine, Isa, Model, Precision, Workspace};

struct Args {
    weights: PathBuf,
    wavs: Vec<PathBuf>,
    trace_out: PathBuf,
    rss_out: Option<PathBuf>,
    precision: Precision,
    chunk: Option<usize>,
    beam: usize,
    endpoint_reset_ms: u32,
    endpoint_replay_ms: u32,
    packet_ms: usize,
    until_s: Option<f64>,
    kv_int8: bool,
    isa: Isa,
}

fn parse() -> Result<Args, String> {
    let mut it = std::env::args().skip(1);
    let mut a = Args {
        weights: PathBuf::new(),
        wavs: Vec::new(),
        trace_out: PathBuf::new(),
        rss_out: None,
        precision: Precision::Int8,
        chunk: None,
        beam: 1,
        endpoint_reset_ms: 0,
        endpoint_replay_ms: 0,
        packet_ms: 20,
        until_s: None,
        kv_int8: false,
        isa: Isa::detect(),
    };
    while let Some(k) = it.next() {
        let mut v = || it.next().ok_or_else(|| format!("{k} needs a value"));
        match k.as_str() {
            "--weights" => a.weights = v()?.into(),
            "--wavs" => {
                a.wavs = v()?
                    .split(',')
                    .filter(|s| !s.is_empty())
                    .map(PathBuf::from)
                    .collect();
            }
            "--trace-out" => a.trace_out = v()?.into(),
            "--rss-out" => a.rss_out = Some(v()?.into()),
            "--precision" => a.precision = Precision::parse(&v()?).map_err(|e| e.to_string())?,
            "--chunk" => a.chunk = Some(v()?.parse().map_err(|e| format!("--chunk: {e}"))?),
            "--beam" => a.beam = v()?.parse().map_err(|e| format!("--beam: {e}"))?,
            "--endpoint-reset-ms" => {
                a.endpoint_reset_ms = v()?
                    .parse()
                    .map_err(|e| format!("--endpoint-reset-ms: {e}"))?
            }
            "--endpoint-replay-ms" => {
                a.endpoint_replay_ms = v()?
                    .parse()
                    .map_err(|e| format!("--endpoint-replay-ms: {e}"))?
            }
            "--packet-ms" => a.packet_ms = v()?.parse().map_err(|e| format!("--packet-ms: {e}"))?,
            "--until-s" => a.until_s = Some(v()?.parse().map_err(|e| format!("--until-s: {e}"))?),
            "--kv-int8" => a.kv_int8 = true,
            "--isa" => {
                a.isa = match v()?.as_str() {
                    "scalar" => Isa::Scalar,
                    "avx2fma" => Isa::Avx2Fma,
                    "neon" => Isa::Neon,
                    o => return Err(format!("unknown isa {o}")),
                }
            }
            o => return Err(format!("unknown argument {o}")),
        }
    }
    if a.weights.as_os_str().is_empty() || a.wavs.is_empty() || a.trace_out.as_os_str().is_empty() {
        return Err("--weights, --wavs, and --trace-out are required".into());
    }
    Ok(a)
}

/// Open a 16 kHz mono PCM16 WAV file, positioning the reader at the start of the `data` payload
/// and returning `(reader, data_bytes_remaining)`.
fn open_wav_pcm16(path: &Path) -> Result<(BufReader<File>, u64), String> {
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

/// Read current `(VmRSS_kib, VmHWM_kib)` from `/proc/self/status` on Linux.
fn read_proc_rss_kib() -> (u64, u64) {
    let mut rss = 0u64;
    let mut hwm = 0u64;
    if let Ok(txt) = std::fs::read_to_string("/proc/self/status") {
        for line in txt.lines() {
            if let Some(rest) = line.strip_prefix("VmRSS:") {
                if let Some(tok) = rest.split_whitespace().next() {
                    rss = tok.parse().unwrap_or(0);
                }
            } else if let Some(rest) = line.strip_prefix("VmHWM:") {
                if let Some(tok) = rest.split_whitespace().next() {
                    hwm = tok.parse().unwrap_or(0);
                }
            }
        }
    }
    (rss, hwm)
}

/// O(1) exact committed-word emitter for greedy endless streaming.
/// Matches `score_alwayson.py` snapshot stability timestamps to 0.000 ms while using O(1) memory.
struct WordCommitter {
    word_tok_start: usize,
    last_tok_len: usize,
    pending: Option<(String, f64)>,
    total_words: u64,
}

impl WordCommitter {
    fn new() -> Self {
        Self {
            word_tok_start: 0,
            last_tok_len: 0,
            pending: None,
            total_words: 0,
        }
    }

    fn update<W: Write>(
        &mut self,
        engine: &Engine,
        tokens: &[u32],
        audio_s: f64,
        out: &mut W,
    ) -> Result<(), String> {
        if tokens.len() == self.last_tok_len {
            return Ok(());
        }
        self.last_tok_len = tokens.len();
        let ids: Vec<i64> = tokens[self.word_tok_start..]
            .iter()
            .map(|&t| i64::from(t))
            .collect();
        let tail = engine.model.tokens.decode_ids(&ids);
        let ws: Vec<&str> = tail.split_whitespace().collect();
        if ws.is_empty() {
            return Ok(());
        }
        if ws.len() == 1 {
            let w0 = ws[0];
            match &self.pending {
                Some((pw, _)) if pw == w0 => {}
                _ => {
                    self.pending = Some((w0.to_owned(), audio_s));
                }
            }
            return Ok(());
        }
        // Multiple words in tail: every word before the last is now finalized.
        let mut emit_prev: Option<(String, f64)> = None;
        let mut emit_now: Vec<String> = Vec::new();
        match self.pending.take() {
            Some((pw, pt)) if pw == ws[0] => {
                emit_prev = Some((pw, pt));
            }
            _ => {
                emit_now.push(ws[0].to_owned());
            }
        }
        for &w in &ws[1..ws.len() - 1] {
            emit_now.push(w.to_owned());
        }
        if let Some((pw, pt)) = emit_prev {
            writeln!(out, "{}", json!({"audio_s": pt, "words": [pw]}))
                .map_err(|e| e.to_string())?;
            self.total_words += 1;
        }
        if !emit_now.is_empty() {
            self.total_words += emit_now.len() as u64;
            writeln!(out, "{}", json!({"audio_s": audio_s, "words": emit_now}))
                .map_err(|e| e.to_string())?;
        }
        let last_word = ws[ws.len() - 1];
        self.pending = Some((last_word.to_owned(), audio_s));

        // Advance `word_tok_start` to the start of `last_word` (small backward scan).
        let mut new_start = tokens.len().saturating_sub(1);
        for k in (self.word_tok_start..tokens.len()).rev() {
            let sub: Vec<i64> = tokens[k..].iter().map(|&t| i64::from(t)).collect();
            let dec = engine.model.tokens.decode_ids(&sub);
            if dec.trim() == last_word {
                new_start = k;
            } else if dec.split_whitespace().count() > 1 {
                break;
            }
        }
        self.word_tok_start = new_start;
        Ok(())
    }

    fn flush<W: Write>(&mut self, out: &mut W) -> Result<(), String> {
        if let Some((pw, pt)) = self.pending.take() {
            writeln!(out, "{}", json!({"audio_s": pt, "words": [pw]}))
                .map_err(|e| e.to_string())?;
            self.total_words += 1;
        }
        Ok(())
    }
}

fn run(a: &Args) -> Result<serde_json::Value, String> {
    if let Some(parent) = a.trace_out.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
    }
    let mut model = Model::load(&a.weights, true).map_err(|e| e.to_string())?;
    if a.precision == Precision::Int8 && model.int8_prepared.is_none() {
        model.quantize_int8();
    }
    let mut engine = Engine::new(Arc::new(model), a.isa, a.precision)
        .map_err(|e| e.to_string())?
        .with_beam(a.beam)
        .with_kv_int8(a.kv_int8)
        .with_endpoint_reset(a.endpoint_reset_ms, a.endpoint_replay_ms);
    if let Some(c) = a.chunk {
        engine = engine.with_chunk(c).map_err(|e| e.to_string())?;
    }

    let mut trace_w = BufWriter::new(File::create(&a.trace_out).map_err(|e| e.to_string())?);
    let mut rss_w = match &a.rss_out {
        Some(p) => {
            if let Some(parent) = p.parent() {
                if !parent.as_os_str().is_empty() {
                    std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
                }
            }
            Some(BufWriter::new(File::create(p).map_err(|e| e.to_string())?))
        }
        None => None,
    };

    let mut stream = engine.create_stream();
    let mut ws = Workspace::default();
    let mut committer = WordCommitter::new();
    let packet_samples = 16 * a.packet_ms;
    let mut raw_buf = vec![0u8; packet_samples * 2];
    let mut pcm_buf = vec![0i16; packet_samples];

    let max_samples: Option<u64> = a.until_s.map(|s| (s * 16_000.0).round() as u64);
    let mut total_samples: u64 = 0;

    let (c0, w0) = (run_info::thread_cpu_s(), Instant::now());
    let mut last_rss_wall = -1.0f64;
    let (rss0, _) = read_proc_rss_kib();
    let mut rss_peak_kib = rss0;

    'wav_loop: for wav_path in &a.wavs {
        let (mut reader, mut rem_bytes) = open_wav_pcm16(wav_path)?;
        while rem_bytes >= 2 {
            if let Some(cap) = max_samples {
                if total_samples >= cap {
                    break 'wav_loop;
                }
            }
            let want_samples = match max_samples {
                Some(cap) => (cap - total_samples).min(packet_samples as u64) as usize,
                None => packet_samples,
            };
            let want_bytes = (want_samples * 2).min(rem_bytes as usize) & !1;
            if want_bytes == 0 {
                break;
            }
            let mut got = 0usize;
            while got < want_bytes {
                let n = reader
                    .read(&mut raw_buf[got..want_bytes])
                    .map_err(|e| e.to_string())?;
                if n == 0 {
                    break;
                }
                got += n;
            }
            let got_samples = got / 2;
            if got_samples == 0 {
                break;
            }
            rem_bytes -= (got_samples * 2) as u64;
            for (dst, chunk) in pcm_buf[..got_samples]
                .iter_mut()
                .zip(raw_buf[..got_samples * 2].chunks_exact(2))
            {
                *dst = i16::from_le_bytes([chunk[0], chunk[1]]);
            }
            stream
                .accept_i16(&pcm_buf[..got_samples])
                .map_err(|e| e.to_string())?;
            total_samples += got_samples as u64;
            let audio_s = total_samples as f64 / 16_000.0;

            while engine
                .decode(&mut [&mut stream], &mut ws)
                .map_err(|e| e.to_string())?
                > 0
            {
                committer.update(&engine, stream.tokens(), audio_s, &mut trace_w)?;
            }

            let wall_now = w0.elapsed().as_secs_f64();
            if wall_now - last_rss_wall >= 1.0 {
                last_rss_wall = wall_now;
                let (rss_kib, hwm_kib) = read_proc_rss_kib();
                rss_peak_kib = rss_peak_kib.max(rss_kib).max(hwm_kib);
                if let Some(rw) = rss_w.as_mut() {
                    writeln!(
                        rw,
                        "{}",
                        json!({
                            "wall_s": (wall_now * 100.0).round() / 100.0,
                            "audio_s": (audio_s * 100.0).round() / 100.0,
                            "rss_mb": (rss_kib as f64 / 1024.0 * 100.0).round() / 100.0,
                            "hwm_mb": (hwm_kib as f64 / 1024.0 * 100.0).round() / 100.0,
                            "resets": stream.endpoint_resets(),
                            "words": committer.total_words
                        })
                    )
                    .map_err(|e| e.to_string())?;
                }
            }
        }
    }

    stream.input_finished();
    let final_audio_s = total_samples as f64 / 16_000.0;
    while engine
        .decode(&mut [&mut stream], &mut ws)
        .map_err(|e| e.to_string())?
        > 0
    {
        committer.update(&engine, stream.tokens(), final_audio_s, &mut trace_w)?;
    }
    committer.flush(&mut trace_w)?;
    trace_w.flush().map_err(|e| e.to_string())?;
    if let Some(rw) = rss_w.as_mut() {
        rw.flush().map_err(|e| e.to_string())?;
    }

    let cpu_s = run_info::thread_cpu_s() - c0;
    let wall_s = w0.elapsed().as_secs_f64();
    let (final_rss, final_hwm) = read_proc_rss_kib();
    rss_peak_kib = rss_peak_kib.max(final_rss).max(final_hwm);
    let audio_h = final_audio_s / 3600.0;
    let rss_start_mb = rss0 as f64 / 1024.0;
    let rss_end_mb = final_rss as f64 / 1024.0;
    let rss_peak_mb = rss_peak_kib as f64 / 1024.0;

    Ok(json!({
        "audio_s": final_audio_s,
        "audio_h": audio_h,
        "cpu_s": cpu_s,
        "wall_s": wall_s,
        "cpu_s_per_audio_s": if final_audio_s > 0.0 { cpu_s / final_audio_s } else { 0.0 },
        "rtfx": if cpu_s > 0.0 { final_audio_s / cpu_s } else { 0.0 },
        "endpoint_reset_ms": a.endpoint_reset_ms,
        "endpoint_replay_ms": a.endpoint_replay_ms,
        "endpoint_resets": stream.endpoint_resets(),
        "words": committer.total_words,
        "tokens": stream.tokens().len(),
        "rss_start_mb": rss_start_mb,
        "rss_end_mb": rss_end_mb,
        "rss_peak_mb": rss_peak_mb,
        "rss_delta_mb": rss_end_mb - rss_start_mb,
        "trace_out": a.trace_out,
        "rss_out": a.rss_out,
    }))
}

fn main() -> ExitCode {
    match parse().and_then(|a| run(&a)) {
        Ok(s) => {
            println!("{}", serde_json::to_string_pretty(&s).unwrap_or_default());
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("ultra-engine-endless: {e}");
            ExitCode::FAILURE
        }
    }
}
