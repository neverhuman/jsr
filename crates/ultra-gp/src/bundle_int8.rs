//! Strict `manifest.int8` contract validator for `ultra.r1-export/{v1,v1.1}` bundles.
//!
//! Distinguishes graph-level arbitrary-batch invariance (`static_qdq` only) from consumer-enforced
//! single-row invariance (`dynamic_per_row` with `max_batch == 1`), and enforces explicit `v1.1`
//! `static_qdq` mixed FP32-joiner declarations without ever allowing silent fallback on incomplete
//! `v1` or `v1.1` bundles.

use serde::Deserialize;

use crate::bundle::{R1Precision, R1_EXPORT_SCHEMA_V1_1};
use crate::GpError;

#[derive(Debug, Deserialize)]
pub(crate) struct RawInt8Meta {
    pub(crate) mode: Option<String>,
    #[serde(default)]
    pub(crate) batch_invariant: bool,
    pub(crate) max_batch: Option<usize>,
    pub(crate) joiner: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ValidatedInt8Contract {
    pub(crate) int8_mode: Option<String>,
    pub(crate) int8_graph_batch_invariant: bool,
    pub(crate) int8_consumer_batch_invariant: bool,
    pub(crate) int8_max_batch_enc: Option<usize>,
    pub(crate) int8_max_batch_joi: Option<usize>,
    pub(crate) enc_file_key: &'static str,
    pub(crate) joi_file_key: &'static str,
    pub(crate) joiner_precision: R1Precision,
}

pub(crate) fn validate_int8_contract(
    schema: &str,
    raw: Option<&RawInt8Meta>,
    requested: R1Precision,
) -> Result<ValidatedInt8Contract, GpError> {
    let mut int8_mode = None;
    let mut graph_inv = false;
    let mut consumer_inv = false;
    let mut max_batch = None;
    let mut declares_fp32_joiner = false;

    if let Some(m) = raw {
        int8_mode = m.mode.clone();
        match m.mode.as_deref() {
            Some("dynamic") => {
                if m.batch_invariant || m.max_batch.is_some() {
                    return Err(GpError::Manifest(
                        "int8.mode 'dynamic' cannot set batch_invariant=true or max_batch".into(),
                    ));
                }
                if !matches!(m.joiner.as_deref(), None | Some("int8")) {
                    return Err(GpError::Manifest(format!(
                        "int8.mode 'dynamic' requires int8 joiner, got {:?}",
                        m.joiner
                    )));
                }
            }
            Some("dynamic_per_row") => {
                if schema != R1_EXPORT_SCHEMA_V1_1 {
                    return Err(GpError::Manifest(format!(
                        "int8.mode 'dynamic_per_row' requires schema '{R1_EXPORT_SCHEMA_V1_1}', got '{schema}'"
                    )));
                }
                if !m.batch_invariant || m.max_batch != Some(1) {
                    return Err(GpError::Manifest(format!(
                        "int8.mode 'dynamic_per_row' requires batch_invariant=true and max_batch=1, got invariant={} max_batch={:?}",
                        m.batch_invariant, m.max_batch
                    )));
                }
                if !matches!(m.joiner.as_deref(), None | Some("int8")) {
                    return Err(GpError::Manifest(format!(
                        "int8.mode 'dynamic_per_row' requires int8 joiner, got {:?}",
                        m.joiner
                    )));
                }
                // Graph itself still uses DynamicQuantizeLinear; consumer enforces B=1 & N=1.
                graph_inv = false;
                consumer_inv = true;
                max_batch = Some(1);
            }
            Some("static_qdq") => {
                if schema != R1_EXPORT_SCHEMA_V1_1 {
                    return Err(GpError::Manifest(format!(
                        "int8.mode 'static_qdq' requires schema '{R1_EXPORT_SCHEMA_V1_1}', got '{schema}'"
                    )));
                }
                if !m.batch_invariant || m.max_batch.is_some() {
                    return Err(GpError::Manifest(format!(
                        "int8.mode 'static_qdq' requires batch_invariant=true and no max_batch, got invariant={} max_batch={:?}",
                        m.batch_invariant, m.max_batch
                    )));
                }
                match m.joiner.as_deref() {
                    None | Some("int8") => declares_fp32_joiner = false,
                    Some(j) if j == "fp32" || j.starts_with("fp32 ") => declares_fp32_joiner = true,
                    Some(other) => {
                        return Err(GpError::Manifest(format!(
                            "unsupported int8.joiner declaration '{other}' in static_qdq"
                        )))
                    }
                }
                graph_inv = true;
                consumer_inv = true;
            }
            Some(other) => {
                return Err(GpError::Manifest(format!(
                    "unsupported int8.mode '{other}' (expected 'dynamic', 'dynamic_per_row', or 'static_qdq')"
                )))
            }
            None => {
                if m.batch_invariant || m.max_batch.is_some() || m.joiner.is_some() {
                    return Err(GpError::Manifest(
                        "manifest.int8 missing 'mode' while declaring invariance/max_batch/joiner"
                            .into(),
                    ));
                }
            }
        }
    }

    let (enc_file_key, joi_file_key, joiner_precision, int8_max_batch_enc, int8_max_batch_joi) =
        match requested {
            R1Precision::Fp32 => ("encoder.fp32", "joiner.fp32", R1Precision::Fp32, None, None),
            R1Precision::Int8 => {
                if declares_fp32_joiner {
                    (
                        "encoder.int8",
                        "joiner.fp32",
                        R1Precision::Fp32,
                        max_batch,
                        None,
                    )
                } else {
                    (
                        "encoder.int8",
                        "joiner.int8",
                        R1Precision::Int8,
                        max_batch,
                        max_batch,
                    )
                }
            }
        };

    Ok(ValidatedInt8Contract {
        int8_mode,
        int8_graph_batch_invariant: graph_inv,
        int8_consumer_batch_invariant: consumer_inv,
        int8_max_batch_enc,
        int8_max_batch_joi,
        enc_file_key,
        joi_file_key,
        joiner_precision,
    })
}
