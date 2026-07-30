//! On-disk format constants and metadata structures for rw-store files.

use serde::{Deserialize, Serialize};

/// Magic bytes at the start of every rw-store file.
pub const MAGIC: &[u8; 8] = b"RWSTORE1";
/// Current on-disk format version.
pub const VERSION: u32 = 1;
/// Format versions this build can read.
pub const SUPPORTED_VERSIONS: &[u32] = &[1];
/// Fixed-size file header length in bytes.
pub const HEADER_LEN: usize = 64;
/// Fixed-size chunk index record length in bytes.
pub const INDEX_RECORD_LEN: usize = 64;

/// 2D tile height in grid points.
pub const TILE_Y: usize = 256;
/// 2D tile width in grid points.
pub const TILE_X: usize = 256;
/// 3D column-chunk height in grid points.
pub const COL_Y: usize = 16;
/// 3D column-chunk width in grid points.
pub const COL_X: usize = 16;

/// Chunk contains no finite values.
pub const FLAG_EMPTY: u8 = 1;
/// All finite values in the chunk are equal.
pub const FLAG_CONSTANT: u8 = 2;
/// Chunk contains at least one missing (NaN) value.
pub const FLAG_HAS_MISSING: u8 = 4;

/// Chunk kind: 2D surface tile.
pub const KIND_TILE2D: u8 = 0;
/// Chunk kind: 3D pressure-level column chunk.
pub const KIND_COLUMN3D: u8 = 1;

/// Schema identifier embedded in per-hour metadata.
pub const SCHEMA_HOUR: &str = "rw-store.hour.v1";
/// Exact-time per-hour schema. The `forecast_hour` field is an ordinal
/// storage slot; `lead_seconds` and `valid_unix` carry the physical time.
/// Keeping a distinct schema makes pre-v2 readers reject these files instead
/// of treating an exact-time ordinal as a whole forecast hour.
pub const SCHEMA_HOUR_V2: &str = "rw-store.hour.v2";
/// Codec name for 2D tiles: zstd-compressed raw f32.
pub const CODEC_2D: &str = "zstd1_f32";
/// Codec name for 3D column chunks: zstd-compressed affine-quantized i16.
pub const CODEC_3D: &str = "zstd1_affine_i16";

/// Per-hour store file metadata.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct RwsHourMeta {
    pub schema: String,
    pub model: String,
    pub run: String,
    /// Whole forecast hour in v1; ordinal storage slot in exact-time v2.
    pub forecast_hour: u16,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lead_seconds: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub valid_unix: Option<i64>,
    pub nx: usize,
    pub ny: usize,
    pub grid_hash: String,
    pub variables: Vec<RwsVariableMeta>,
    pub chunking: RwsChunking,
    pub writer: RwsWriterInfo,
}

/// Exact physical time attached to one ordinal storage slot in a v2 run.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RwsExactTime {
    /// Nonnegative elapsed seconds from the run origin.
    pub lead_seconds: u64,
    /// UTC Unix timestamp of this valid time.
    pub valid_unix: i64,
}

impl RwsExactTime {
    pub const fn new(lead_seconds: u64, valid_unix: i64) -> Self {
        Self {
            lead_seconds,
            valid_unix,
        }
    }

    /// Run origin implied by this pair. Values that cannot be represented as
    /// an i64 Unix timestamp are rejected rather than wrapped.
    pub fn origin_unix(self) -> Option<i64> {
        let origin = i128::from(self.valid_unix) - i128::from(self.lead_seconds);
        i64::try_from(origin).ok()
    }
}

impl RwsHourMeta {
    pub fn exact_time(&self) -> Option<RwsExactTime> {
        Some(RwsExactTime {
            lead_seconds: self.lead_seconds?,
            valid_unix: self.valid_unix?,
        })
    }

    /// Validate the schema/timing contract without inspecting binary layout.
    pub fn validate_time_schema(&self) -> Result<Option<RwsExactTime>, String> {
        let timing = match (self.lead_seconds, self.valid_unix) {
            (Some(lead_seconds), Some(valid_unix)) => Some(RwsExactTime {
                lead_seconds,
                valid_unix,
            }),
            (None, None) => None,
            _ => {
                return Err(
                    "hour metadata must contain both lead_seconds and valid_unix or neither"
                        .to_string(),
                );
            }
        };
        match self.schema.as_str() {
            SCHEMA_HOUR if timing.is_none() => Ok(None),
            SCHEMA_HOUR => Err("v1 hour metadata must not contain exact-time metadata".to_string()),
            SCHEMA_HOUR_V2 => {
                let timing = timing.ok_or_else(|| {
                    "v2 hour metadata requires lead_seconds and valid_unix".to_string()
                })?;
                timing.origin_unix().ok_or_else(|| {
                    "v2 hour metadata cannot represent valid_unix - lead_seconds".to_string()
                })?;
                Ok(Some(timing))
            }
            other => Err(format!(
                "unexpected schema '{other}' (expected '{SCHEMA_HOUR}' or '{SCHEMA_HOUR_V2}')"
            )),
        }
    }
}

/// Metadata for a single variable stored in an hour file.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct RwsVariableMeta {
    pub id: u16,
    pub name: String,
    pub units: String,
    /// "surface2d" | "pressure3d"
    pub kind: String,
    pub codec: String,
    pub levels_hpa: Vec<u16>,
    pub selector: serde_json::Value,
}

/// Chunk geometry used by the writer.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct RwsChunking {
    pub tile_y: usize,
    pub tile_x: usize,
    pub col_y: usize,
    pub col_x: usize,
}

/// Provenance of the writer that produced a store file.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct RwsWriterInfo {
    pub name: String,
    pub version: String,
    pub build: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta(schema: &str, timing: Option<(Option<u64>, Option<i64>)>) -> RwsHourMeta {
        let (lead_seconds, valid_unix) = timing.unwrap_or((None, None));
        RwsHourMeta {
            schema: schema.to_string(),
            model: "wrf".to_string(),
            run: "research".to_string(),
            forecast_hour: 0,
            lead_seconds,
            valid_unix,
            nx: 1,
            ny: 1,
            grid_hash: "hash".to_string(),
            variables: Vec::new(),
            chunking: RwsChunking {
                tile_y: TILE_Y,
                tile_x: TILE_X,
                col_y: COL_Y,
                col_x: COL_X,
            },
            writer: RwsWriterInfo {
                name: "test".to_string(),
                version: "0".to_string(),
                build: "test".to_string(),
            },
        }
    }

    #[test]
    fn hour_schema_requires_exactly_the_matching_time_contract() {
        assert_eq!(
            meta(SCHEMA_HOUR, None).validate_time_schema().unwrap(),
            None
        );

        let v1_with_time = meta(SCHEMA_HOUR, Some((Some(0), Some(1_700_000_000))));
        assert!(
            v1_with_time
                .validate_time_schema()
                .unwrap_err()
                .contains("v1")
        );

        let v2_missing = meta(SCHEMA_HOUR_V2, None);
        assert!(
            v2_missing
                .validate_time_schema()
                .unwrap_err()
                .contains("requires")
        );
        let v2_partial = meta(SCHEMA_HOUR_V2, Some((Some(900), None)));
        assert!(
            v2_partial
                .validate_time_schema()
                .unwrap_err()
                .contains("both")
        );

        let exact = RwsExactTime {
            lead_seconds: 900,
            valid_unix: 1_700_000_900,
        };
        assert_eq!(
            meta(
                SCHEMA_HOUR_V2,
                Some((Some(exact.lead_seconds), Some(exact.valid_unix))),
            )
            .validate_time_schema()
            .unwrap(),
            Some(exact)
        );
        assert_eq!(RwsExactTime::new(900, 1_700_000_900), exact);
    }
}
