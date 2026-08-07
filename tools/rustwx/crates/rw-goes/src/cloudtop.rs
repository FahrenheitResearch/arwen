//! The `gpuwm-obs.goes-cloudtop.v1` pack: cloud-top height and pressure
//! on their own fixed grid, the deliberate sibling of the CWP pack rather
//! than extra planes inside it.
//!
//! ## Why this is a second pack and not two more planes
//!
//! Measured on real GOES-19 CONUS granules for scan `s20262161801170`:
//! COD, CPS and ACTP are the 2 km fixed grid, 2500 x 1500.  ACHA and CTP
//! are the 10 km fixed grid, 500 x 300.  Same satellite, same sector, same
//! scan start, same geostationary projection — a different grid.
//!
//! The `GPWMGOES` format's whole promise is "every plane in this pack is
//! bit-identically on the grid this pack states".  Upsampling cloud-top
//! height to 2 km to make one pack would break exactly the guarantee the
//! format exists to make, and would bury an interpolation choice inside an
//! ingest tool where no science reviewed it.  So `rw_goes cwp` refuses a
//! mixed-grid pack (naming both shapes), and the vertical-placement fields
//! get this pack instead: their own grid metadata, their own navigation,
//! their own DQF rows.
//!
//! The join happens at the consumer, where the interpolation is explicit,
//! chosen by the science, and recorded in that stage's own receipt.  Both
//! packs carry the same `(satellite, sector, scan_start)` triple, which is
//! the pairing key; `sibling` additionally pins the CWP pack's payload
//! digest when the caller passed `--pairs-with`, so a consumer can prove
//! it paired the right two files rather than trusting a file name.
//!
//! Two packs per scan, deliberately.
//!
//! Recorded ruling, 2026-08-06 (coordinator; surfaced to Drew in the
//! morning summary): SEPARATE PACK, the bridge never regrids to make one.

use std::collections::BTreeMap;
use std::error::Error;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::pack::{
    ArrayEntry, CWP_SCHEMA, ContainerMeta, ProjectionEntry, SourceEntry, decode_container,
    write_container,
};

/// The schema this pack declares, and the contract the Python reader
/// checks before it touches a payload byte.
///
/// **v2 (2026-08-06)** adds the per-pixel `<product>_dqf` planes, in step
/// with `gpuwm-obs.goes-cwp.v2` — the two families move together so that
/// "v2" means the same thing to a reader of either.  See `pack::CWP_SCHEMA`
/// for what the counts alone could not answer.
pub const CLOUDTOP_SCHEMA: &str = "gpuwm-obs.goes-cloudtop.v2";

/// The v1 cloud-top schema: no longer written, still read, for the same
/// reason [`crate::pack::CWP_SCHEMA_V1`] is.
pub const CLOUDTOP_SCHEMA_V1: &str = "gpuwm-obs.goes-cloudtop.v1";

/// Every cloud-top schema this build will read, newest first.
pub const CLOUDTOP_READABLE_SCHEMAS: &[&str] = &[CLOUDTOP_SCHEMA, CLOUDTOP_SCHEMA_V1];

/// The CWP pack this one is the vertical-placement half of, pinned by its
/// payload digest.  Present only when the caller named it with
/// `--pairs-with`; its absence means the pairing rests on the
/// `(satellite, sector, scan_start)` triple alone, which is stated either
/// way.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SiblingEntry {
    pub schema: String,
    pub filename: String,
    pub content_sha256: String,
    /// The sibling's grid, which is NOT this pack's grid — that is the
    /// whole reason there are two packs.
    pub nx: usize,
    pub ny: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub window: Option<[usize; 4]>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloudTopMeta {
    pub schema: String,
    pub status: String,
    pub satellite: String,
    pub sector: String,
    pub scan_start: String,
    pub scan_end: String,
    pub sources: Vec<SourceEntry>,
    /// `[x_start, x_count, y_start, y_count]` into THIS pack's fixed grid.
    /// A window index here is not comparable to a window index in the CWP
    /// sibling: the grids differ, which is why both packs state their own.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub window: Option<[usize; 4]>,
    pub projection: ProjectionEntry,
    pub nx: usize,
    pub ny: usize,
    pub x_scan_rad: Vec<f64>,
    pub y_scan_rad: Vec<f64>,
    /// Plane name -> key into `arrays`.  Declared order:
    /// cloud_top_height_m and/or cloud_top_pressure_hpa, then lat, lon.
    pub planes: BTreeMap<String, String>,
    pub plane_order: Vec<String>,
    pub arrays: BTreeMap<String, ArrayEntry>,
    pub payload_bytes: usize,
    pub content_sha256: String,
    /// The schema of the pack this one pairs with.  The pairing key is
    /// `(satellite, sector, scan_start)`, all stated above.
    pub pairs_with_schema: String,
    /// The interpolation this pack does NOT do, stated so no consumer
    /// mistakes silence for "the grids matched".
    pub regrid: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sibling: Option<SiblingEntry>,
}

impl ContainerMeta for CloudTopMeta {
    const WRITTEN_SCHEMA: &'static str = CLOUDTOP_SCHEMA;
    const READABLE_SCHEMAS: &'static [&'static str] = CLOUDTOP_READABLE_SCHEMAS;

    fn schema(&self) -> &str {
        &self.schema
    }

    fn content_sha256(&self) -> &str {
        &self.content_sha256
    }

    fn arrays(&self) -> &BTreeMap<String, ArrayEntry> {
        &self.arrays
    }
}

/// What `regrid` always says: this pack resamples nothing.
///
/// Deliberately names no schema version.  It used to spell the sibling's
/// schema out, which silently went stale the moment the CWP pack bumped
/// to v2 and shipped a wrong name in real metadata.  `pairs_with_schema`
/// carries that string, from the constant, so there is one place for it.
pub const NO_REGRID: &str =
    "none: planes are on the granules' own fixed grid, bit-identical; any join to the \
     pairs_with_schema sibling's grid is the consumer's explicit choice";

/// The schema string a cloud-top pack names as its sibling.
pub fn pairs_with_schema() -> String {
    CWP_SCHEMA.to_string()
}

pub fn decode_cloudtop_pack(bytes: &[u8]) -> Result<(CloudTopMeta, Vec<u8>), Box<dyn Error>> {
    decode_container(bytes)
}

pub fn write_cloudtop_pack(
    path: &Path,
    meta: &CloudTopMeta,
    payload: &[u8],
) -> Result<usize, Box<dyn Error>> {
    write_container(path, meta, payload)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pack::{
        PayloadBuilder, decode_pack, encode_container, hex_sha256, pack_schema,
    };

    fn minimal_meta(payload: &[u8]) -> CloudTopMeta {
        CloudTopMeta {
            schema: CLOUDTOP_SCHEMA.to_string(),
            status: "READY".to_string(),
            satellite: "G19".to_string(),
            sector: "C".to_string(),
            scan_start: "2026-08-04T18:01:17Z".to_string(),
            scan_end: "2026-08-04T18:03:54.500Z".to_string(),
            sources: Vec::new(),
            window: None,
            projection: ProjectionEntry {
                perspective_point_height_m: 35786023.0,
                semi_major_axis_m: 6378137.0,
                semi_minor_axis_m: 6356752.31414,
                longitude_of_projection_origin_deg: -75.0,
                sweep_angle_axis: "x".to_string(),
            },
            nx: 2,
            ny: 1,
            x_scan_rad: vec![0.0, 5.0e-4],
            y_scan_rad: vec![0.0],
            planes: BTreeMap::new(),
            plane_order: Vec::new(),
            arrays: BTreeMap::new(),
            payload_bytes: payload.len(),
            content_sha256: hex_sha256(payload),
            pairs_with_schema: pairs_with_schema(),
            regrid: NO_REGRID.to_string(),
            sibling: None,
        }
    }

    #[test]
    fn a_cloud_top_pack_round_trips_through_the_shared_container() {
        let mut builder = PayloadBuilder::new();
        let key = builder.push_f32(&[9500.0, f32::NAN], vec![1, 2]);
        let (payload, arrays) = builder.finish();
        let mut meta = minimal_meta(&payload);
        meta.arrays = arrays;
        meta.planes
            .insert("cloud_top_height_m".to_string(), key.clone());
        meta.plane_order = vec!["cloud_top_height_m".to_string()];

        let bytes = encode_container(&meta, &payload).unwrap();
        let (back, back_payload) = decode_cloudtop_pack(&bytes).unwrap();
        assert_eq!(back.schema, CLOUDTOP_SCHEMA);
        assert_eq!(back.pairs_with_schema, CWP_SCHEMA);
        assert_eq!(back_payload, payload);
        assert_eq!(back.arrays[&key].shape, vec![1, 2]);
        assert!(back.regrid.starts_with("none:"));
    }

    #[test]
    fn the_two_families_never_decode_as_each_other() {
        // One container, two schemas: a reader that asked for the wrong
        // family is refused by name rather than handed planes on a grid it
        // did not expect.
        let mut builder = PayloadBuilder::new();
        builder.push_f32(&[1.0, 2.0], vec![2]);
        let (payload, arrays) = builder.finish();
        let mut meta = minimal_meta(&payload);
        meta.arrays = arrays;
        let bytes = encode_container(&meta, &payload).unwrap();

        assert_eq!(pack_schema(&bytes).unwrap(), CLOUDTOP_SCHEMA);
        let err = decode_pack(&bytes).unwrap_err().to_string();
        assert!(err.contains(CWP_SCHEMA), "{err}");
        assert!(err.contains(CLOUDTOP_SCHEMA), "{err}");
    }

    #[test]
    fn a_corrupted_cloud_top_payload_is_refused_by_digest() {
        let mut builder = PayloadBuilder::new();
        builder.push_f32(&[300.0, 850.0], vec![2]);
        let (payload, arrays) = builder.finish();
        let mut meta = minimal_meta(&payload);
        meta.arrays = arrays;
        let mut bytes = encode_container(&meta, &payload).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 0xff;
        let err = decode_cloudtop_pack(&bytes).unwrap_err().to_string();
        assert!(err.contains("digest mismatch"), "{err}");
    }

    #[test]
    fn the_sibling_entry_pins_the_other_grid_not_this_one() {
        let mut builder = PayloadBuilder::new();
        builder.push_f32(&[1.0, 2.0], vec![2]);
        let (payload, arrays) = builder.finish();
        let mut meta = minimal_meta(&payload);
        meta.arrays = arrays;
        meta.sibling = Some(SiblingEntry {
            schema: CWP_SCHEMA.to_string(),
            filename: "g19_conus_20260804_1801.goespack".to_string(),
            content_sha256: "0".repeat(64),
            nx: 2500,
            ny: 1500,
            window: None,
        });
        let bytes = encode_container(&meta, &payload).unwrap();
        let (back, _) = decode_cloudtop_pack(&bytes).unwrap();
        let sibling = back.sibling.expect("pinned");
        assert_eq!((sibling.nx, sibling.ny), (2500, 1500));
        assert_ne!(
            (sibling.nx, sibling.ny),
            (back.nx, back.ny),
            "the sibling's grid is a different grid; that is why there are two packs"
        );
    }
}
