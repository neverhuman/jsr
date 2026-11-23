//! CLI parsing, validation, and LM attachment for `ultra-engine-endless`.

use std::path::PathBuf;
use std::sync::Arc;

use ultra_engine::{Engine, Isa, LmConfig, NgramLm, Precision};

pub(crate) struct Args {
    pub(crate) weights: PathBuf,
    pub(crate) wavs: Vec<PathBuf>,
    pub(crate) trace_out: PathBuf,
    pub(crate) rss_out: Option<PathBuf>,
    pub(crate) precision: Precision,
    pub(crate) chunk: Option<usize>,
    pub(crate) beam: usize,
    pub(crate) beam_max_symbols: usize,
    pub(crate) lm_path: Option<PathBuf>,
    pub(crate) lm_weight: f32,
    pub(crate) lodr_weight: f32,
    pub(crate) blank_penalty: f32,
    pub(crate) endpoint_reset_ms: u32,
    pub(crate) endpoint_replay_ms: u32,
    pub(crate) packet_ms: usize,
    pub(crate) until_s: Option<f64>,
    pub(crate) kv_int8: bool,
    pub(crate) isa: Isa,
}

pub(crate) fn parse() -> Result<Args, String> {
    parse_from(std::env::args().skip(1))
}

pub(crate) fn parse_from<I, S>(iter: I) -> Result<Args, String>
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    let mut it = iter.into_iter().map(Into::into);
    let mut a = Args {
        weights: PathBuf::new(),
        wavs: Vec::new(),
        trace_out: PathBuf::new(),
        rss_out: None,
        precision: Precision::Int8,
        chunk: None,
        beam: 1,
        beam_max_symbols: 1,
        lm_path: None,
        lm_weight: 0.0,
        lodr_weight: 0.0,
        blank_penalty: 0.0,
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
            "--beam-max-symbols" => {
                a.beam_max_symbols = v()?
                    .parse()
                    .map_err(|e| format!("--beam-max-symbols: {e}"))?
            }
            "--lm" => a.lm_path = Some(v()?.into()),
            "--lm-weight" => a.lm_weight = v()?.parse().map_err(|e| format!("--lm-weight: {e}"))?,
            "--lodr-weight" => {
                a.lodr_weight = v()?.parse().map_err(|e| format!("--lodr-weight: {e}"))?
            }
            "--blank-penalty" => {
                a.blank_penalty = v()?.parse().map_err(|e| format!("--blank-penalty: {e}"))?
            }
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
                    "avx512vnni" => Isa::Avx512Vnni,
                    "neon" => Isa::Neon,
                    o => return Err(format!("unknown isa {o}")),
                }
            }
            o => return Err(format!("unknown argument {o}")),
        }
    }
    validate(&a)?;
    Ok(a)
}

fn validate(a: &Args) -> Result<(), String> {
    if a.weights.as_os_str().is_empty() || a.wavs.is_empty() || a.trace_out.as_os_str().is_empty() {
        return Err("--weights, --wavs, and --trace-out are required".into());
    }
    if a.packet_ms == 0 {
        return Err("--packet-ms must be positive".into());
    }
    if a.beam == 0 {
        return Err("--beam must be >= 1".into());
    }
    if a.beam_max_symbols == 0 {
        return Err("--beam-max-symbols must be >= 1".into());
    }
    lm_config(a).validate().map_err(|e| e.to_string())?;
    if a.lm_path.is_some() && a.beam <= 1 {
        return Err(
            "--lm requires --beam > 1 (LM shallow fusion is inactive in greedy mode)".into(),
        );
    }
    if a.lm_path.is_none() && (a.lm_weight != 0.0 || a.lodr_weight != 0.0 || a.blank_penalty != 0.0)
    {
        return Err("--lm-weight, --lodr-weight, and --blank-penalty require --lm".into());
    }
    Ok(())
}

pub(crate) fn lm_config(a: &Args) -> LmConfig {
    LmConfig {
        lm_weight: a.lm_weight,
        lodr_weight: a.lodr_weight,
        blank_penalty: a.blank_penalty,
    }
}

pub(crate) fn attach_lm(engine: Engine, a: &Args) -> Result<Engine, String> {
    match &a.lm_path {
        Some(lm_path) => {
            let lm = Arc::new(NgramLm::load(lm_path).map_err(|e| e.to_string())?);
            engine.with_lm(lm, lm_config(a)).map_err(|e| e.to_string())
        }
        None => Ok(engine),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_args() -> Vec<&'static str> {
        vec![
            "--weights",
            "/tmp/w",
            "--wavs",
            "/tmp/a.wav",
            "--trace-out",
            "/tmp/t.jsonl",
        ]
    }

    #[test]
    fn defaults_match_engine_eval() {
        let a = parse_from(base_args()).expect("valid minimal args");
        assert_eq!(a.beam, 1);
        assert_eq!(a.beam_max_symbols, 1);
        assert!(a.lm_path.is_none());
        assert_eq!(a.lm_weight, 0.0);
        assert_eq!(a.lodr_weight, 0.0);
        assert_eq!(a.blank_penalty, 0.0);
    }

    #[test]
    fn parses_lm_and_beam_flags() {
        let mut raw = base_args();
        raw.extend([
            "--beam",
            "4",
            "--beam-max-symbols",
            "2",
            "--lm",
            "/tmp/m5.ulm",
            "--lm-weight",
            "0.8",
            "--lodr-weight",
            "0.6",
            "--blank-penalty",
            "0.15",
        ]);
        let a = parse_from(raw).expect("valid lm args");
        assert_eq!(a.beam, 4);
        assert_eq!(a.beam_max_symbols, 2);
        assert_eq!(
            a.lm_path.as_deref(),
            Some(std::path::Path::new("/tmp/m5.ulm"))
        );
        assert!((a.lm_weight - 0.8).abs() < 1e-6);
        assert!((a.lodr_weight - 0.6).abs() < 1e-6);
        assert!((a.blank_penalty - 0.15).abs() < 1e-6);
    }

    #[test]
    fn rejects_lm_with_greedy_beam() {
        let mut default_beam = base_args();
        default_beam.extend(["--lm", "/tmp/m5.ulm", "--lm-weight", "0.8"]);
        let err = parse_from(default_beam).err().expect("must fail");
        assert!(err.contains("--beam > 1"), "unexpected error: {err}");

        let mut explicit_one = base_args();
        explicit_one.extend(["--beam", "1", "--lm", "/tmp/m5.ulm"]);
        let err = parse_from(explicit_one).err().expect("must fail");
        assert!(err.contains("--beam > 1"), "unexpected error: {err}");
    }

    #[test]
    fn rejects_invalid_lm_weights_and_beam_bounds() {
        for bad in [
            vec!["--beam", "0"],
            vec!["--beam-max-symbols", "0"],
            vec!["--beam", "4", "--lm", "/tmp/m.ulm", "--lm-weight", "-0.1"],
            vec!["--beam", "4", "--lm", "/tmp/m.ulm", "--lodr-weight", "nan"],
            vec!["--beam", "4", "--lm", "/tmp/m.ulm", "--blank-penalty", "-1"],
            vec!["--beam", "4", "--lm-weight", "0.8"],
        ] {
            let mut raw = base_args();
            raw.extend(bad.iter().copied());
            assert!(
                parse_from(raw).is_err(),
                "expected parse error for {:?}",
                bad
            );
        }
    }
}
