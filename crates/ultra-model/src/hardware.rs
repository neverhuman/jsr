//! Hardware inventory contract (`ultra.hardware/v1`, matching `docs/ultra/HARDWARE.json`).

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::ModelError;

pub const HARDWARE_SCHEMA: &str = "ultra.hardware/v1";

/// One execution host entry in `HARDWARE.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MachineRecord {
    pub id: String,
    pub host: String,
    pub cpu: String,
    pub cores: u32,
    pub threads_per_core: u32,
    #[serde(default)]
    pub l1d: String,
    #[serde(default)]
    pub l2: String,
    #[serde(default)]
    pub l3: String,
    pub isa: Vec<String>,
    pub ram_gib: f64,
    pub os: String,
    pub governor: String,
    pub status: String,
}

impl MachineRecord {
    /// True if all `required_isa` features (case-insensitive) exist in `self.isa`.
    pub fn supports_all_isa(&self, required_isa: &[String]) -> Result<(), ModelError> {
        for req in required_isa {
            let ok = self
                .isa
                .iter()
                .any(|have| have.eq_ignore_ascii_case(req.trim()));
            if !ok {
                return Err(ModelError::UnsupportedIsa {
                    isa: req.clone(),
                    machine: self.id.clone(),
                });
            }
        }
        Ok(())
    }
}

/// Top-level `HARDWARE.json` document.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HardwareInventory {
    pub schema: String,
    pub machines: Vec<MachineRecord>,
}

impl HardwareInventory {
    /// Parse and validate a `HARDWARE.json` string.
    pub fn from_json_str(text: &str) -> Result<Self, ModelError> {
        let inv: Self = serde_json::from_str(text).map_err(|e| ModelError::Parse(e.to_string()))?;
        inv.validate()?;
        Ok(inv)
    }

    /// Load and validate `HARDWARE.json` from a file path.
    pub fn load(path: &Path) -> Result<Self, ModelError> {
        let text = std::fs::read_to_string(path)?;
        Self::from_json_str(&text)
    }

    /// Validate schema and basic host invariants.
    pub fn validate(&self) -> Result<(), ModelError> {
        if self.schema != HARDWARE_SCHEMA {
            return Err(ModelError::WrongSchema {
                expected: HARDWARE_SCHEMA,
                found: self.schema.clone(),
            });
        }
        if self.machines.is_empty() {
            return Err(ModelError::InvalidBundle(
                "hardware inventory has no machines".into(),
            ));
        }
        for m in &self.machines {
            if m.id.trim().is_empty() || m.cores == 0 || m.ram_gib <= 0.0 {
                return Err(ModelError::InvalidBundle(format!(
                    "invalid machine entry `{}`",
                    m.id
                )));
            }
        }
        Ok(())
    }

    /// Look up a machine by `id` (e.g. `"H0"`), or return the first machine when `id` is `None`.
    pub fn machine(&self, id: Option<&str>) -> Result<&MachineRecord, ModelError> {
        match id {
            Some(target) => self
                .machines
                .iter()
                .find(|m| m.id == target)
                .ok_or_else(|| ModelError::MissingArtifact(format!("machine `{target}`"))),
            None => self.machines.first().ok_or_else(|| {
                ModelError::InvalidBundle("hardware inventory has no machines".into())
            }),
        }
    }
}
