use serde::{Deserialize, Serialize};

use crate::bridge::SoundingColumn;
use crate::error::SoundingBridgeError;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ParcelFlavor {
    SurfaceBased,
    MixedLayer,
    MostUnstable,
}

impl ParcelFlavor {
    pub const fn short_label(self) -> &'static str {
        match self {
            Self::SurfaceBased => "SB",
            Self::MixedLayer => "ML",
            Self::MostUnstable => "MU",
        }
    }

    pub const fn long_label(self) -> &'static str {
        match self {
            Self::SurfaceBased => "surface-based",
            Self::MixedLayer => "mixed-layer",
            Self::MostUnstable => "most-unstable",
        }
    }

    const fn sort_key(self) -> usize {
        match self {
            Self::SurfaceBased => 0,
            Self::MixedLayer => 1,
            Self::MostUnstable => 2,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PendingEcapeRequest {
    pub parcel: ParcelFlavor,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExternalEcapeValue {
    pub parcel: ParcelFlavor,
    pub ecape_j_kg: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct NativeParcelContext {
    pub parcel: ParcelFlavor,
    pub cape_j_kg: f64,
    pub cin_j_kg: f64,
    pub lcl_m_agl: f64,
    pub lfc_m_agl: f64,
    pub el_m_agl: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExternalEcapeAnnotationRow {
    pub parcel: ParcelFlavor,
    pub ecape_j_kg: f64,
    pub native_cape_j_kg: f64,
    pub native_cin_j_kg: f64,
    pub ecape_fraction_of_cape: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExternalEcapeAnnotationContext {
    pub source_label: String,
    pub storm_motion_label: Option<String>,
    pub rows: Vec<ExternalEcapeAnnotationRow>,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ExternalEcapeSummary {
    #[serde(default)]
    pub source: Option<String>,
    #[serde(default)]
    pub storm_motion: Option<String>,
    #[serde(default)]
    pub values: Vec<ExternalEcapeValue>,
    #[serde(default)]
    pub notes: Vec<String>,
}

impl ExternalEcapeSummary {
    pub fn validate(&self) -> Result<(), SoundingBridgeError> {
        if self.values.is_empty() {
            return Err(SoundingBridgeError::InvalidEcapeSummary(
                "expected at least one parcel ECAPE value".into(),
            ));
        }

        for value in &self.values {
            if !value.ecape_j_kg.is_finite() {
                return Err(SoundingBridgeError::InvalidEcapeSummary(format!(
                    "{} ECAPE must be finite",
                    value.parcel.short_label()
                )));
            }
        }

        for (index, value) in self.values.iter().enumerate() {
            if self.values[index + 1..]
                .iter()
                .any(|other| other.parcel == value.parcel)
            {
                return Err(SoundingBridgeError::InvalidEcapeSummary(format!(
                    "duplicate {} ECAPE value",
                    value.parcel.short_label()
                )));
            }
        }

        Ok(())
    }

    pub fn value_for(&self, parcel: ParcelFlavor) -> Option<&ExternalEcapeValue> {
        self.values.iter().find(|value| value.parcel == parcel)
    }

    pub fn ordered_values(&self) -> Vec<&ExternalEcapeValue> {
        let mut values: Vec<_> = self.values.iter().collect();
        values.sort_by_key(|value| value.parcel.sort_key());
        values
    }

    pub fn annotation_context(
        &self,
        native_context: &[NativeParcelContext],
    ) -> Result<ExternalEcapeAnnotationContext, SoundingBridgeError> {
        self.validate()?;

        let mut rows = Vec::with_capacity(self.values.len());
        for value in self.ordered_values() {
            let Some(native) = native_context
                .iter()
                .find(|native| native.parcel == value.parcel)
            else {
                return Err(SoundingBridgeError::InvalidEcapeSummary(format!(
                    "missing native {} parcel context",
                    value.parcel.short_label()
                )));
            };

            let ecape_fraction_of_cape = if native.cape_j_kg.is_finite() && native.cape_j_kg > 0.0 {
                Some(value.ecape_j_kg / native.cape_j_kg)
            } else {
                None
            };

            rows.push(ExternalEcapeAnnotationRow {
                parcel: value.parcel,
                ecape_j_kg: value.ecape_j_kg,
                native_cape_j_kg: native.cape_j_kg,
                native_cin_j_kg: native.cin_j_kg,
                ecape_fraction_of_cape,
            });
        }

        let mut notes = vec![
            "ECAPE values are external entraining diagnostics. CAPE/CIN context below is the native sharprs parcel solution using the same SB/ML/MU parcel families.".to_string(),
        ];
        notes.extend(
            self.notes
                .iter()
                .map(|note| note.trim().to_string())
                .filter(|note| !note.is_empty()),
        );

        Ok(ExternalEcapeAnnotationContext {
            source_label: self.source_label().to_string(),
            storm_motion_label: self
                .storm_motion
                .as_deref()
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(str::to_string),
            rows,
            notes,
        })
    }

    pub fn source_label(&self) -> &str {
        self.source
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .unwrap_or("caller supplied")
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EcapeIntegrationStatus {
    NativeVerifiedTableAndExternalAnnotationBridge,
}

pub fn supported_parcels() -> [ParcelFlavor; 3] {
    [
        ParcelFlavor::SurfaceBased,
        ParcelFlavor::MixedLayer,
        ParcelFlavor::MostUnstable,
    ]
}

pub fn ecape_status() -> EcapeIntegrationStatus {
    EcapeIntegrationStatus::NativeVerifiedTableAndExternalAnnotationBridge
}

pub fn require_future_ecape_bridge(
    _column: &SoundingColumn,
    request: PendingEcapeRequest,
) -> Result<(), SoundingBridgeError> {
    Err(SoundingBridgeError::EcapeUnavailable(format!(
        "{} parcel ECAPE is computed by rustwx-sounding for native table rendering through the verified ecape-rs path; caller-supplied external annotation blocks are still supported when an outside ECAPE source needs to be shown separately",
        request.parcel.long_label()
    )))
}
