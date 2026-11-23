//! End-to-end tests for `ultra-engine-endless` LM + beam CLI flags and `.ulm` v2 loading.

use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use ultra_engine::NgramLm;

const SYNTH_LM_ARPA: &str = "\
\\data\\
ngram 1=7
ngram 2=5
ngram 3=2

\\1-grams:
-4.0\t<unk>
-99.0\t<s>\t-0.2
-1.0\t2\t-0.1
-1.2\t3\t-0.2
-1.5\t4\t-0.1
-1.8\t5\t-0.1
-2.0\t6

\\2-grams:
-0.2\t<s> 2\t-0.1
-0.2\t2 3\t-0.1
-0.3\t3 4\t-0.1
-0.4\t4 5
-0.5\t5 6

\\3-grams:
-0.1\t<s> 2 3
-0.1\t2 3 4

\\end\\
";

const SYNTH_LODR_ARPA: &str = "\
\\data\\
ngram 1=5
ngram 2=2

\\1-grams:
-3.5\t<unk>
-99.0\t<s>\t-0.2
-1.1\t2\t-0.2
-1.4\t3\t-0.1
-1.8\t4

\\2-grams:
-0.7\t<s> 2
-0.5\t2 3

\\end\\
";

struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        let p = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
            .join(format!("endless-lm-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).expect("mkdir");
        Self(p)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    format!("{:x}", h.finalize())
}

fn write_wav_pcm16(path: &Path, samples: &[i16]) {
    let data_len = (samples.len() * 2) as u32;
    let riff_len = 36 + data_len;
    let mut out = Vec::with_capacity((44 + data_len) as usize);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&riff_len.to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM
    out.extend_from_slice(&1u16.to_le_bytes()); // mono
    out.extend_from_slice(&16_000u32.to_le_bytes());
    out.extend_from_slice(&32_000u32.to_le_bytes());
    out.extend_from_slice(&2u16.to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());
    for &s in samples {
        out.extend_from_slice(&s.to_le_bytes());
    }
    std::fs::write(path, out).expect("write wav");
}

fn build_safetensors(ckpt_sha: &str, tensors: &[(&str, Vec<usize>, Vec<f32>)]) -> Vec<u8> {
    let mut hdr = Map::new();
    hdr.insert(
        "__metadata__".into(),
        json!({"checkpoint_sha256": ckpt_sha}),
    );
    let mut payload = Vec::new();
    for (name, shape, data) in tensors {
        let n: usize = shape.iter().product();
        assert_eq!(n, data.len(), "shape mismatch for {name}");
        let start = payload.len();
        for &v in data {
            payload.extend_from_slice(&v.to_le_bytes());
        }
        let end = payload.len();
        hdr.insert(
            (*name).into(),
            json!({"dtype": "F32", "shape": shape, "data_offsets": [start, end]}),
        );
    }
    let hdr_bytes = serde_json::to_vec(&Value::Object(hdr)).expect("json");
    let mut out = Vec::with_capacity(8 + hdr_bytes.len() + payload.len());
    out.extend_from_slice(&(hdr_bytes.len() as u64).to_le_bytes());
    out.extend_from_slice(&hdr_bytes);
    out.extend_from_slice(&payload);
    out
}

/// Write a tiny hash-verified 1-block dense R1 weight directory (`vocab = 8`, `d = 4`).
fn write_tiny_weights(dir: &Path) {
    std::fs::create_dir_all(dir).expect("mkdir weights");
    let (d, ch, ffn, k, vocab, pred_emb, joiner, f2) = (4, 4, 8, 3, 8, 4, 4, 19);
    let ckpt_sha = "1".repeat(64);
    let det = |n: usize, seed: usize| -> Vec<f32> {
        (0..n)
            .map(|i| (((i + seed * 17) as f32) * 0.31).sin() * 0.15)
            .collect()
    };
    let ones = |n: usize| vec![1.0f32; n];
    let zeros = |n: usize| vec![0.0f32; n];

    // Bias joiner slightly toward non-blank tokens so beam+LM emits real trace words.
    let joiner_b = vec![-0.4, -2.0, 0.35, 0.30, 0.25, -1.0, -1.0, -1.0];
    let tensors: Vec<(&str, Vec<usize>, Vec<f32>)> = vec![
        ("cmvn.mean", vec![80], zeros(80)),
        ("cmvn.istd", vec![80], ones(80)),
        ("encoder.stem.c1.weight", vec![ch, 1, 3, 3], det(ch * 9, 1)),
        ("encoder.stem.c1.bias", vec![ch], zeros(ch)),
        (
            "encoder.stem.c2.weight",
            vec![ch, ch, 3, 3],
            det(ch * ch * 9, 2),
        ),
        ("encoder.stem.c2.bias", vec![ch], zeros(ch)),
        (
            "encoder.stem.out.weight",
            vec![d, ch * f2],
            det(d * ch * f2, 3),
        ),
        ("encoder.stem.out.bias", vec![d], zeros(d)),
        ("encoder.blocks.0.ff1.norm.weight", vec![d], ones(d)),
        ("encoder.blocks.0.ff1.norm.bias", vec![d], zeros(d)),
        (
            "encoder.blocks.0.ff1.w1.weight",
            vec![ffn, d],
            det(ffn * d, 4),
        ),
        ("encoder.blocks.0.ff1.w1.bias", vec![ffn], zeros(ffn)),
        (
            "encoder.blocks.0.ff1.w2.weight",
            vec![d, ffn],
            det(d * ffn, 5),
        ),
        ("encoder.blocks.0.ff1.w2.bias", vec![d], zeros(d)),
        ("encoder.blocks.0.att.norm.weight", vec![d], ones(d)),
        ("encoder.blocks.0.att.norm.bias", vec![d], zeros(d)),
        (
            "encoder.blocks.0.att.qkv.weight",
            vec![3 * d, d],
            det(3 * d * d, 6),
        ),
        ("encoder.blocks.0.att.qkv.bias", vec![3 * d], zeros(3 * d)),
        ("encoder.blocks.0.att.out.weight", vec![d, d], det(d * d, 7)),
        ("encoder.blocks.0.att.out.bias", vec![d], zeros(d)),
        ("encoder.blocks.0.conv.norm.weight", vec![d], ones(d)),
        ("encoder.blocks.0.conv.norm.bias", vec![d], zeros(d)),
        (
            "encoder.blocks.0.conv.pw1.weight",
            vec![2 * d, d],
            det(2 * d * d, 8),
        ),
        ("encoder.blocks.0.conv.pw1.bias", vec![2 * d], zeros(2 * d)),
        (
            "encoder.blocks.0.conv.dw.weight",
            vec![d, 1, k],
            det(d * k, 9),
        ),
        ("encoder.blocks.0.conv.dw.bias", vec![d], zeros(d)),
        ("encoder.blocks.0.conv.dnorm.weight", vec![d], ones(d)),
        ("encoder.blocks.0.conv.dnorm.bias", vec![d], zeros(d)),
        (
            "encoder.blocks.0.conv.pw2.weight",
            vec![d, d],
            det(d * d, 10),
        ),
        ("encoder.blocks.0.conv.pw2.bias", vec![d], zeros(d)),
        ("encoder.blocks.0.ff2.norm.weight", vec![d], ones(d)),
        ("encoder.blocks.0.ff2.norm.bias", vec![d], zeros(d)),
        (
            "encoder.blocks.0.ff2.w1.weight",
            vec![ffn, d],
            det(ffn * d, 11),
        ),
        ("encoder.blocks.0.ff2.w1.bias", vec![ffn], zeros(ffn)),
        (
            "encoder.blocks.0.ff2.w2.weight",
            vec![d, ffn],
            det(d * ffn, 12),
        ),
        ("encoder.blocks.0.ff2.w2.bias", vec![d], zeros(d)),
        ("encoder.final_norm.weight", vec![d], ones(d)),
        ("encoder.final_norm.bias", vec![d], zeros(d)),
        ("engine.rope_inv_freq", vec![d / 2], vec![1.0, 0.01]),
        (
            "joiner.enc_proj.weight",
            vec![joiner, d],
            det(joiner * d, 13),
        ),
        ("joiner.enc_proj.bias", vec![joiner], zeros(joiner)),
        (
            "predictor.emb.weight",
            vec![vocab, pred_emb],
            det(vocab * pred_emb, 14),
        ),
        (
            "predictor.proj.weight",
            vec![joiner, 2 * pred_emb],
            det(joiner * 2 * pred_emb, 15),
        ),
        ("predictor.proj.bias", vec![joiner], zeros(joiner)),
        (
            "joiner.out.weight",
            vec![vocab, joiner],
            det(vocab * joiner, 16),
        ),
        ("joiner.out.bias", vec![vocab], joiner_b),
    ];

    let st_bytes = build_safetensors(&ckpt_sha, &tensors);
    let tokens_bytes =
        b"<blk> 0\n<unk> 1\n\xe2\x96\x81alpha 2\n\xe2\x96\x81beta 3\n\xe2\x96\x81gamma 4\n\xe2\x96\x81delta 5\n\xe2\x96\x81eps 6\n\xe2\x96\x81zeta 7\n";
    let mel_bytes = vec![0u8; 257 * 80 * 4];
    let bpe_bytes = b"dummy-bpe";

    std::fs::write(dir.join("weights.safetensors"), &st_bytes).expect("write weights");
    std::fs::write(dir.join("tokens.txt"), tokens_bytes).expect("write tokens");
    std::fs::write(dir.join("mel_fbank.f32"), &mel_bytes).expect("write mel");
    std::fs::write(dir.join("bpe.model"), bpe_bytes).expect("write bpe");

    let arch = json!({
        "schema": "ultra.r1-weights/v1",
        "model": {
            "family": "dense",
            "config": {
                "vocab_rnnt": vocab,
                "pred_emb": pred_emb,
                "joiner": joiner,
                "joiner_act": "tanh"
            },
            "encoder": {
                "n_mels": 80,
                "d_model": d,
                "n_blocks": 1,
                "n_heads": 1,
                "ffn_mult": 2,
                "conv_kernel": k,
                "stem_channels": ch,
                "chunk": 4,
                "left_context": 4,
                "post_block_norm": false,
                "family": "dense",
                "lower_blocks": 1,
                "middle_blocks": 0,
                "final_blocks": 0
            }
        },
        "source": { "checkpoint_sha256": ckpt_sha },
        "files": {
            "weights": { "path": "weights.safetensors", "sha256": sha256_hex(&st_bytes) },
            "tokens": { "path": "tokens.txt", "sha256": sha256_hex(tokens_bytes) },
            "mel_fbank": { "path": "mel_fbank.f32", "sha256": sha256_hex(&mel_bytes) },
            "bpe": { "path": "bpe.model", "sha256": sha256_hex(bpe_bytes) }
        },
        "features": {
            "sample_rate": 16000,
            "frame_length": 400,
            "frame_shift": 160,
            "n_fft": 512,
            "n_mels": 80,
            "log_floor": 1e-6,
            "cmvn": { "mean": zeros(80), "istd": ones(80) }
        },
        "decoder": {
            "kind": "stateless",
            "context": 2,
            "vocab_size": vocab,
            "joiner_activation": "tanh",
            "blank_id": 0,
            "max_symbols": 1,
            "ctx_init": [0, 0]
        }
    });
    std::fs::write(dir.join("arch.json"), arch.to_string()).expect("write arch");
}

#[test]
fn endless_binary_loads_ulm_v2_and_writes_trace() {
    let tmp = TempDir::new("v2-trace");
    let weights = tmp.0.join("weights");
    write_tiny_weights(&weights);

    let wav = tmp.0.join("clip.wav");
    let pcm: Vec<i16> = (0..16_000)
        .map(|i| (((i as f32) * 0.07).sin() * 4000.0) as i16)
        .collect();
    write_wav_pcm16(&wav, &pcm);

    let lm_path = tmp.0.join("synth_v2.ulm");
    let lm = NgramLm::from_arpa_str(SYNTH_LM_ARPA, Some(SYNTH_LODR_ARPA), 8, true).expect("arpa");
    lm.save(&lm_path).expect("save v2 ulm");

    let bin = env!("CARGO_BIN_EXE_ultra-engine-endless");
    let w_str = weights.to_str().expect("weights path");
    let wav_str = wav.to_str().expect("wav path");
    let lm_str = lm_path.to_str().expect("lm path");

    // 1. Greedy + --lm must fail loudly before loading weights.
    let bad_trace = tmp.0.join("bad.jsonl");
    let bad = Command::new(bin)
        .args(["--weights", w_str])
        .args(["--wavs", wav_str])
        .args(["--trace-out", bad_trace.to_str().expect("bad trace")])
        .args(["--beam", "1", "--lm", lm_str])
        .output()
        .expect("run endless");
    assert!(!bad.status.success(), "--lm with --beam 1 must fail");
    let bad_err = String::from_utf8_lossy(&bad.stderr);
    assert!(bad_err.contains("--beam > 1"), "stderr: {bad_err}");

    // 2. Beam + synthetic v2 .ulm succeeds and writes valid JSONL trace + LM summary fields.
    let lm_trace = tmp.0.join("lm_trace.jsonl");
    let ok = Command::new(bin)
        .args(["--weights", w_str])
        .args(["--precision", "fp32", "--isa", "scalar"])
        .args(["--wavs", wav_str])
        .args(["--trace-out", lm_trace.to_str().expect("lm trace")])
        .args(["--beam", "4", "--beam-max-symbols", "2"])
        .args(["--lm", lm_str])
        .args([
            "--lm-weight",
            "0.8",
            "--lodr-weight",
            "0.6",
            "--blank-penalty",
            "0.1",
        ])
        .output()
        .expect("run endless with lm");
    assert!(
        ok.status.success(),
        "endless + lm failed: {}",
        String::from_utf8_lossy(&ok.stderr)
    );
    let summary: Value = serde_json::from_slice(&ok.stdout).expect("summary json");
    assert_eq!(summary["beam"], 4);
    assert_eq!(summary["beam_max_symbols"], 2);
    assert_eq!(summary["lm"], lm_str);
    assert!((summary["lm_weight"].as_f64().expect("lm_weight") - 0.8).abs() < 1e-5);
    assert!((summary["lodr_weight"].as_f64().expect("lodr_weight") - 0.6).abs() < 1e-5);
    assert!((summary["blank_penalty"].as_f64().expect("blank_penalty") - 0.1).abs() < 1e-5);
    assert!(summary["tokens"].as_u64().expect("tokens") > 0);
    assert!(summary["words"].as_u64().expect("words") > 0);

    let trace_txt = std::fs::read_to_string(&lm_trace).expect("read trace");
    assert!(!trace_txt.trim().is_empty(), "trace must be non-empty");
    for line in trace_txt.lines() {
        let row: Value = serde_json::from_str(line).expect("valid trace jsonl");
        assert!(row["audio_s"].is_number());
        assert!(row["words"].is_array());
    }

    // 3. Omitting --lm stays intact and records `"lm": null`.
    let nolm_trace = tmp.0.join("nolm_trace.jsonl");
    let nolm = Command::new(bin)
        .args(["--weights", w_str])
        .args(["--precision", "fp32", "--isa", "scalar"])
        .args(["--wavs", wav_str])
        .args(["--trace-out", nolm_trace.to_str().expect("nolm trace")])
        .args(["--beam", "4", "--beam-max-symbols", "2"])
        .output()
        .expect("run endless without lm");
    assert!(
        nolm.status.success(),
        "endless without lm failed: {}",
        String::from_utf8_lossy(&nolm.stderr)
    );
    let nolm_summary: Value = serde_json::from_slice(&nolm.stdout).expect("nolm summary");
    assert!(nolm_summary["lm"].is_null());

    // 4. Vocab-mismatched .ulm pack fails cleanly.
    let bad_lm_path = tmp.0.join("wrong_vocab.ulm");
    let bad_lm = NgramLm::from_arpa_str(SYNTH_LM_ARPA, None, 16, true).expect("arpa 16");
    bad_lm.save(&bad_lm_path).expect("save");
    let mismatch = Command::new(bin)
        .args(["--weights", w_str])
        .args(["--precision", "fp32", "--isa", "scalar"])
        .args(["--wavs", wav_str])
        .args([
            "--trace-out",
            tmp.0.join("mm.jsonl").to_str().expect("mm trace"),
        ])
        .args(["--beam", "4", "--lm", bad_lm_path.to_str().expect("bad lm")])
        .output()
        .expect("run mismatch");
    assert!(!mismatch.status.success(), "vocab mismatch must fail");
    let mm_err = String::from_utf8_lossy(&mismatch.stderr);
    assert!(mm_err.contains("vocab"), "stderr: {mm_err}");
}
