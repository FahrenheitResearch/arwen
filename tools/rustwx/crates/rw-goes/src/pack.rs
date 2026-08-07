//! The `gpuwm-obs.goes-cwp.v1` pack: a 64-byte little-endian header
//! (magic `GPWMGOES`), a JSON metadata block, and one contiguous payload
//! of `<f4` planes — the `GPWMRDR1` layout from rw-nexrad carried over
//! unchanged, per docs/obs-goes-cwp-bridge-design.md, so the Python
//! consumer needs `json` + `numpy.frombuffer` and nothing else.

use std::collections::BTreeMap;
use std::error::Error;
use std::path::Path;

use rw_store::atomic::atomic_write_bytes;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const PACK_MAGIC: &[u8; 8] = b"GPWMGOES";
pub const PACK_VERSION: u32 = 1;
pub const PACK_HEADER_BYTES: usize = 64;
/// The schema the pack metadata declares, and the contract the Python
/// reader (`gpuwm/obs/goes_cwp.py`) checks before it touches a payload
/// byte.
///
/// **v2 (2026-08-06)** adds a per-pixel `<product>_dqf` plane for every
/// source granule, plus `SourceEntry::dqf_plane` naming it.  v1 carried
/// only the DQF *counts* and the condemn mask, which is a summary: it
/// says how many pixels each cause condemned but not which, so the
/// thin (256) / thick (512) DCOMP bits — deliberately NOT in the condemn
/// mask, and the basis of the observation operator's error inflation —
/// were unrecoverable from a v1 pack.  The addition is otherwise purely
/// additive: no v1 plane was renamed, moved, retyped or removed, and the
/// counts and condemn mask are unchanged.  The bump exists so a consumer
/// that needs the per-pixel plane can require it and fail closed, rather
/// than discover it missing partway through building an error model.
pub const CWP_SCHEMA: &str = "gpuwm-obs.goes-cwp.v2";

/// The v1 CWP schema.  No longer written, still **read**: every receipt
/// that records a v1 pack's digest is only worth something while that
/// pack can still be re-verified, and a receipt that cannot be
/// re-verified is a dead receipt.  Dropping a reader is a decision to
/// invalidate history, and this format does not make that decision
/// casually — v1 packs stay checkable for as long as any receipt names
/// one.
pub const CWP_SCHEMA_V1: &str = "gpuwm-obs.goes-cwp.v1";

/// Every CWP schema this build will read, newest first.
pub const CWP_READABLE_SCHEMAS: &[&str] = &[CWP_SCHEMA, CWP_SCHEMA_V1];

pub fn boxed_error(message: impl Into<String>) -> Box<dyn Error> {
    Box::<dyn Error>::from(message.into())
}

pub fn hex_sha256(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let digest = hasher.finalize();
    let mut out = String::with_capacity(64);
    for byte in digest {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

/// One named `<f4` array inside the payload.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ArrayEntry {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub offset: usize,
    pub bytes: usize,
}

/// One source granule's identity and what its DQF gate did — the pack
/// must say how it was gated (design-note metadata contract).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceEntry {
    /// Product family token: ACHA / ACTP / COD / CPS / CTP.
    pub product: String,
    pub filename: String,
    /// Bytes of the granule exactly as it sat on disk; the identity the
    /// fetch receipt's sha256 was taken over.
    pub bytes: usize,
    pub sha256: String,
    /// `enumerated` or `bitfield`.
    pub dqf_rule: String,
    /// The condemn mask actually applied (bitfield rule only).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub condemn_mask: Option<u16>,
    pub dqf: DqfRow,
    /// The payload plane holding this product's per-pixel DQF, as
    /// published and UNGATED.  The counts above summarise the gate; this
    /// names the plane that lets a consumer recover the individual bits
    /// the summary threw away — the thin (256) / thick (512) DCOMP bits
    /// among them, which the condemn mask deliberately does not gate and
    /// which the CWP observation operator inflates obs error on.
    ///
    /// Empty only in a v1 pack, which carried no such plane.  It is
    /// `serde(default)` for exactly that reason and no other: v1 packs
    /// must keep deserializing so their receipts stay re-verifiable.
    #[serde(default)]
    pub dqf_plane: String,
}

/// `rw_sat::cloud::DqfReport`, restated as a serializable row.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub struct DqfRow {
    pub total: usize,
    pub primary_missing: usize,
    pub dqf_missing: usize,
    pub dqf_bad: usize,
    pub masked: usize,
    pub finite: usize,
}

/// `rw_sat::cwp::CwpCounts`, restated as a serializable row.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub struct CwpRow {
    pub clear_zero: usize,
    pub liquid: usize,
    pub supercooled: usize,
    pub mixed: usize,
    pub ice: usize,
    pub unknown: usize,
    pub phase_missing: usize,
    pub input_missing: usize,
    pub finite: usize,
}

/// The geostationary navigation of record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectionEntry {
    pub perspective_point_height_m: f64,
    pub semi_major_axis_m: f64,
    pub semi_minor_axis_m: f64,
    pub longitude_of_projection_origin_deg: f64,
    pub sweep_angle_axis: String,
}

/// The CWP coefficient table, PROVISIONAL flags carried verbatim so no
/// consumer can mistake the ice branch for settled physics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoefficientTable {
    pub formula: String,
    pub liquid_density_g_cm3: f32,
    pub ice_density_g_cm3: f32,
    pub ice_coefficient_provisional: bool,
    pub mixed_phase_takes_ice_branch_provisional: bool,
    pub clear_sky_emits_zero: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackMeta {
    pub schema: String,
    pub status: String,
    pub satellite: String,
    pub sector: String,
    pub scan_start: String,
    pub scan_end: String,
    pub sources: Vec<SourceEntry>,
    /// `[x_start, x_count, y_start, y_count]` into the full fixed grid
    /// when the pack holds a window; absent for a full-sector pack.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub window: Option<[usize; 4]>,
    pub projection: ProjectionEntry,
    pub nx: usize,
    pub ny: usize,
    /// Fixed-grid scan angles (radians), exactly as decoded — with the
    /// projection these ARE the navigation; lat/lon planes are derived.
    pub x_scan_rad: Vec<f64>,
    pub y_scan_rad: Vec<f64>,
    /// Plane name -> key into `arrays`.  Declared order: cwp, phase, cod,
    /// cps, optional cloud_top_height_m / cloud_top_pressure_hpa, lat, lon.
    pub planes: BTreeMap<String, String>,
    pub plane_order: Vec<String>,
    pub arrays: BTreeMap<String, ArrayEntry>,
    pub payload_bytes: usize,
    pub content_sha256: String,
    pub cwp_counts: CwpRow,
    pub coefficients: CoefficientTable,
}

/// Accumulates `<f4` planes into one contiguous payload (the rw-nexrad
/// `PayloadBuilder`, retold).
#[derive(Default)]
pub struct PayloadBuilder {
    payload: Vec<u8>,
    arrays: BTreeMap<String, ArrayEntry>,
    next: usize,
}

impl PayloadBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push_f32(&mut self, values: &[f32], shape: Vec<usize>) -> String {
        let expected: usize = shape.iter().product();
        debug_assert_eq!(expected, values.len(), "declared shape must match the data");
        let key = format!("a{:05}", self.next);
        self.next += 1;
        let offset = self.payload.len();
        self.payload.reserve(values.len() * 4);
        for value in values {
            self.payload.extend_from_slice(&value.to_le_bytes());
        }
        self.arrays.insert(
            key.clone(),
            ArrayEntry {
                dtype: "<f4".to_string(),
                shape,
                offset,
                bytes: values.len() * 4,
            },
        );
        key
    }

    pub fn finish(self) -> (Vec<u8>, BTreeMap<String, ArrayEntry>) {
        (self.payload, self.arrays)
    }
}

/// What every `GPWMGOES` metadata block must be able to answer, whatever
/// family of pack it belongs to.  Two families share this container: the
/// 2 km CWP pack and the 10 km cloud-top pack that is deliberately its
/// sibling rather than extra planes inside it (see `cloudtop`).
pub trait ContainerMeta {
    /// The schema this build WRITES.  Exactly one, always the newest.
    const WRITTEN_SCHEMA: &'static str;
    /// Every schema this build will READ.  Deliberately wider than what
    /// it writes: a pack whose digest some receipt records must stay
    /// verifiable after the writer has moved on, or the receipt dies
    /// with the reader.
    const READABLE_SCHEMAS: &'static [&'static str];
    fn schema(&self) -> &str;
    fn content_sha256(&self) -> &str;
    fn arrays(&self) -> &BTreeMap<String, ArrayEntry>;
}

impl ContainerMeta for PackMeta {
    const WRITTEN_SCHEMA: &'static str = CWP_SCHEMA;
    const READABLE_SCHEMAS: &'static [&'static str] = CWP_READABLE_SCHEMAS;

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

/// Serialize any pack family to bytes: 64-byte header, JSON metadata,
/// payload.  The framing is identical across families on purpose — one
/// header, one reader in Python.
pub fn encode_container<M: Serialize>(
    meta: &M,
    payload: &[u8],
) -> Result<Vec<u8>, Box<dyn Error>> {
    let meta_json = serde_json::to_vec(meta)?;
    let mut out = Vec::with_capacity(PACK_HEADER_BYTES + meta_json.len() + payload.len());
    out.extend_from_slice(PACK_MAGIC);
    out.extend_from_slice(&PACK_VERSION.to_le_bytes());
    out.extend_from_slice(&(meta_json.len() as u32).to_le_bytes());
    out.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    out.resize(PACK_HEADER_BYTES, 0);
    out.extend_from_slice(&meta_json);
    out.extend_from_slice(payload);
    Ok(out)
}

/// Prove the header and split a pack into its metadata JSON and payload.
/// Everything here is family-agnostic: magic, version, declared lengths.
fn split_container(bytes: &[u8]) -> Result<(&[u8], Vec<u8>), Box<dyn Error>> {
    if bytes.len() < PACK_HEADER_BYTES {
        return Err(boxed_error(format!(
            "not a GOES pack: {} bytes, the header alone is {PACK_HEADER_BYTES}",
            bytes.len()
        )));
    }
    if &bytes[..8] != PACK_MAGIC {
        return Err(boxed_error(format!(
            "not a GOES pack: magic {:?}",
            String::from_utf8_lossy(&bytes[..8])
        )));
    }
    let version = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
    if version != PACK_VERSION {
        return Err(boxed_error(format!(
            "GOES pack version {version}, this build reads {PACK_VERSION}"
        )));
    }
    let meta_len = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]) as usize;
    let payload_len = u64::from_le_bytes([
        bytes[16], bytes[17], bytes[18], bytes[19], bytes[20], bytes[21], bytes[22], bytes[23],
    ]) as usize;
    let meta_end = PACK_HEADER_BYTES + meta_len;
    let payload_end = meta_end + payload_len;
    if payload_end != bytes.len() {
        return Err(boxed_error(format!(
            "truncated GOES pack: header declares {payload_end} bytes, file has {}",
            bytes.len()
        )));
    }
    Ok((
        &bytes[PACK_HEADER_BYTES..meta_end],
        bytes[meta_end..payload_end].to_vec(),
    ))
}

/// The schema a metadata block declares, read without committing to a
/// family.  Every family's reader asks this first: a pack of the wrong
/// family must be refused by name, not by whichever field of the wrong
/// struct happens to be missing.
fn declared_schema(meta_json: &[u8]) -> Result<String, Box<dyn Error>> {
    #[derive(Deserialize)]
    struct SchemaOnly {
        schema: String,
    }
    let declared: SchemaOnly = serde_json::from_slice(meta_json)
        .map_err(|err| boxed_error(format!("GOES pack metadata declares no schema: {err}")))?;
    Ok(declared.schema)
}

/// The schema a pack declares — the one question `verify` must answer
/// before it can pick a reader.
pub fn pack_schema(bytes: &[u8]) -> Result<String, Box<dyn Error>> {
    let (meta_json, _) = split_container(bytes)?;
    declared_schema(meta_json)
}

/// Read back a pack of a known family, checking every self-describing
/// field the writer promised, before a single payload byte is interpreted.
pub fn decode_container<M>(bytes: &[u8]) -> Result<(M, Vec<u8>), Box<dyn Error>>
where
    M: DeserializeOwned + ContainerMeta,
{
    let (meta_json, payload) = split_container(bytes)?;
    // Schema before shape: the families share one container, so a
    // cross-family read must say which family it got, not report a
    // missing field of the struct it was hoping for.
    let declared = declared_schema(meta_json)?;
    if !M::READABLE_SCHEMAS.contains(&declared.as_str()) {
        return Err(boxed_error(format!(
            "GOES pack declares schema {declared:?}; this build reads {}",
            M::READABLE_SCHEMAS.join(" and ")
        )));
    }
    let meta: M = serde_json::from_slice(meta_json)?;
    if meta.schema() != declared {
        return Err(boxed_error(format!(
            "GOES pack metadata disagrees with itself: schema {:?} on the second read, \
             {declared:?} on the first",
            meta.schema()
        )));
    }
    let digest = hex_sha256(&payload);
    if digest != meta.content_sha256() {
        return Err(boxed_error(format!(
            "GOES pack payload digest mismatch: metadata says {}, bytes hash to {digest}",
            meta.content_sha256()
        )));
    }
    for (key, entry) in meta.arrays() {
        let end = entry.offset.saturating_add(entry.bytes);
        if end > payload.len() {
            return Err(boxed_error(format!(
                "pack array {key} spans bytes {}..{end} of a {}-byte payload",
                entry.offset,
                payload.len()
            )));
        }
        let elements: usize = entry.shape.iter().product();
        if elements * 4 != entry.bytes {
            return Err(boxed_error(format!(
                "pack array {key} declares shape {:?} ({elements} elements) but {} bytes",
                entry.shape, entry.bytes
            )));
        }
    }
    Ok((meta, payload))
}

pub fn write_container<M: Serialize>(
    path: &Path,
    meta: &M,
    payload: &[u8],
) -> Result<usize, Box<dyn Error>> {
    let bytes = encode_container(meta, payload)?;
    atomic_write_bytes(path, &bytes)?;
    Ok(bytes.len())
}

/// Serialize a CWP pack to bytes.
pub fn encode_pack(meta: &PackMeta, payload: &[u8]) -> Result<Vec<u8>, Box<dyn Error>> {
    encode_container(meta, payload)
}

/// Read back a CWP pack, refusing anything that is not one.
pub fn decode_pack(bytes: &[u8]) -> Result<(PackMeta, Vec<u8>), Box<dyn Error>> {
    decode_container(bytes)
}

pub fn write_pack(path: &Path, meta: &PackMeta, payload: &[u8]) -> Result<usize, Box<dyn Error>> {
    let bytes = encode_pack(meta, payload)?;
    atomic_write_bytes(path, &bytes)?;
    Ok(bytes.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn minimal_meta(payload: &[u8]) -> PackMeta {
        PackMeta {
            schema: CWP_SCHEMA.to_string(),
            status: "READY".to_string(),
            satellite: "G19".to_string(),
            sector: "C".to_string(),
            scan_start: "2026-08-01T10:01:18Z".to_string(),
            scan_end: "2026-08-01T10:03:55Z".to_string(),
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
            x_scan_rad: vec![0.0, 1.0e-4],
            y_scan_rad: vec![0.0],
            planes: BTreeMap::new(),
            plane_order: Vec::new(),
            arrays: BTreeMap::new(),
            payload_bytes: payload.len(),
            content_sha256: hex_sha256(payload),
            cwp_counts: CwpRow::default(),
            coefficients: CoefficientTable {
                formula: "CWP[g m^-2] = (2/3) * tau * r_e[um] * rho[g cm^-3]".to_string(),
                liquid_density_g_cm3: 1.0,
                ice_density_g_cm3: 0.917,
                ice_coefficient_provisional: true,
                mixed_phase_takes_ice_branch_provisional: true,
                clear_sky_emits_zero: true,
            },
        }
    }

    #[test]
    fn a_pack_round_trips_and_the_reader_reproves_the_digest() {
        let mut builder = PayloadBuilder::new();
        let key = builder.push_f32(&[1.5, f32::NAN], vec![1, 2]);
        let (payload, arrays) = builder.finish();
        let mut meta = minimal_meta(&payload);
        meta.arrays = arrays;
        meta.planes.insert("cwp".to_string(), key.clone());
        meta.plane_order = vec!["cwp".to_string()];

        let bytes = encode_pack(&meta, &payload).unwrap();
        assert_eq!(&bytes[..8], PACK_MAGIC);
        let (back, back_payload) = decode_pack(&bytes).unwrap();
        assert_eq!(back.schema, CWP_SCHEMA);
        assert_eq!(back_payload, payload);
        assert_eq!(back.arrays[&key].shape, vec![1, 2]);
    }

    #[test]
    fn a_dqf_plane_round_trips_with_every_bit_the_operator_reads() {
        // The v2 addition, end to end: a DQF plane goes in as f32, comes
        // back out of the payload bytes, and the thin (256) / thick (512)
        // bits the observation operator inflates on are still there.
        let published: [f32; 5] = [
            2.0,           // clean day clear sky
            134.0,         // wholesale-flagged cloudy retrieval
            2.0 + 256.0,   // thin
            2.0 + 512.0,   // thick
            f32::NAN,      // DQF itself was fill: not a value, an absence
        ];
        let mut builder = PayloadBuilder::new();
        let value_key = builder.push_f32(&[1.0, 2.0, 3.0, 4.0, 5.0], vec![1, 5]);
        let dqf_key = builder.push_f32(&published, vec![1, 5]);
        let (payload, arrays) = builder.finish();

        let mut meta = minimal_meta(&payload);
        meta.arrays = arrays;
        meta.planes.insert("cod".to_string(), value_key);
        meta.planes.insert("cod_dqf".to_string(), dqf_key.clone());
        meta.plane_order = vec!["cod".to_string(), "cod_dqf".to_string()];
        meta.sources = vec![SourceEntry {
            product: "COD".to_string(),
            filename: "OR_ABI-L2-CODC-M6_G19_s2026216.nc".to_string(),
            bytes: 3666798,
            sha256: "0".repeat(64),
            dqf_rule: "bitfield".to_string(),
            condemn_mask: Some(88),
            dqf: DqfRow::default(),
            dqf_plane: "cod_dqf".to_string(),
        }];

        let bytes = encode_pack(&meta, &payload).unwrap();
        let (back, back_payload) = decode_pack(&bytes).unwrap();

        // The source row names the plane, so a reader never guesses.
        assert_eq!(back.sources[0].dqf_plane, "cod_dqf");
        assert_eq!(back.sources[0].condemn_mask, Some(88));
        let key = &back.planes[&back.sources[0].dqf_plane];
        let entry = &back.arrays[key];
        assert_eq!(entry.dtype, "<f4");
        assert_eq!(entry.shape, vec![1, 5]);

        // Read the plane back out of the payload the way numpy would.
        let mut recovered = Vec::new();
        for index in 0..entry.bytes / 4 {
            let at = entry.offset + index * 4;
            recovered.push(f32::from_le_bytes([
                back_payload[at],
                back_payload[at + 1],
                back_payload[at + 2],
                back_payload[at + 3],
            ]));
        }
        assert_eq!(recovered.len(), published.len());
        assert!(recovered[4].is_nan(), "a fill DQF stays absent, never 0");
        for index in 0..4 {
            assert_eq!(recovered[index], published[index]);
        }
        // Which is the point of the whole exercise: the bits are maskable.
        let word = |value: f32| value as u16;
        assert_eq!(word(recovered[2]) & 256, 256, "thin bit recoverable");
        assert_eq!(word(recovered[3]) & 512, 512, "thick bit recoverable");
        assert_eq!(word(recovered[0]) & (256 | 512), 0, "and absent when unset");
    }

    /// A genuine v1 pack: v2 metadata with the field v1 never had
    /// removed and the schema set back, so the bytes are what the v1
    /// writer would actually have produced.
    fn as_v1_pack(meta: &PackMeta, payload: &[u8]) -> Vec<u8> {
        let mut value = serde_json::to_value(meta).unwrap();
        value["schema"] = serde_json::Value::String(CWP_SCHEMA_V1.to_string());
        for source in value["sources"].as_array_mut().unwrap() {
            source.as_object_mut().unwrap().remove("dqf_plane");
        }
        let meta_json = serde_json::to_vec(&value).unwrap();
        let mut out = Vec::new();
        out.extend_from_slice(PACK_MAGIC);
        out.extend_from_slice(&PACK_VERSION.to_le_bytes());
        out.extend_from_slice(&(meta_json.len() as u32).to_le_bytes());
        out.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        out.resize(PACK_HEADER_BYTES, 0);
        out.extend_from_slice(&meta_json);
        out.extend_from_slice(payload);
        out
    }

    #[test]
    fn a_v1_pack_still_reads_so_its_receipts_stay_alive() {
        // Receipts record digests. A digest nobody can re-verify is a
        // dead receipt, so dropping a reader invalidates history --
        // this build writes v2 and reads both.
        let mut builder = PayloadBuilder::new();
        let key = builder.push_f32(&[1.0, 2.0], vec![1, 2]);
        let (payload, arrays) = builder.finish();
        let mut meta = minimal_meta(&payload);
        meta.arrays = arrays;
        meta.planes.insert("cwp".to_string(), key);
        meta.plane_order = vec!["cwp".to_string()];
        meta.sources = vec![SourceEntry {
            product: "COD".to_string(),
            filename: "OR_ABI-L2-CODC-M6_G19_s2026216.nc".to_string(),
            bytes: 3666798,
            sha256: "0".repeat(64),
            dqf_rule: "bitfield".to_string(),
            condemn_mask: Some(88),
            dqf: DqfRow::default(),
            dqf_plane: "cod_dqf".to_string(),
        }];

        let v1_bytes = as_v1_pack(&meta, &payload);
        // The field really is absent, or this test proves nothing.
        let (meta_json, _) = split_container(&v1_bytes).unwrap();
        let text = String::from_utf8(meta_json.to_vec()).unwrap();
        assert!(!text.contains("dqf_plane"), "the fixture is not a v1 pack");
        assert_eq!(declared_schema(meta_json).unwrap(), CWP_SCHEMA_V1);

        let (back, back_payload) = decode_pack(&v1_bytes).unwrap();
        assert_eq!(back.schema, CWP_SCHEMA_V1);
        assert_eq!(back_payload, payload);
        // Everything v1 DID carry survives the read untouched.
        assert_eq!(back.sources[0].condemn_mask, Some(88));
        assert_eq!(back.sources[0].product, "COD");
        // And what it did not carry reads as absent, not as a plane name
        // that would send a consumer looking for a plane that isn't there.
        assert!(back.sources[0].dqf_plane.is_empty());

        // v2 is still what gets written.
        assert_eq!(PackMeta::WRITTEN_SCHEMA, CWP_SCHEMA);
        assert!(CWP_READABLE_SCHEMAS.contains(&CWP_SCHEMA_V1));
        assert!(CWP_READABLE_SCHEMAS.contains(&CWP_SCHEMA));
    }

    #[test]
    fn a_corrupted_payload_is_refused_by_digest() {
        let mut builder = PayloadBuilder::new();
        builder.push_f32(&[1.0, 2.0], vec![2]);
        let (payload, arrays) = builder.finish();
        let mut meta = minimal_meta(&payload);
        meta.arrays = arrays;
        let mut bytes = encode_pack(&meta, &payload).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 0xff;
        let err = decode_pack(&bytes).unwrap_err().to_string();
        assert!(err.contains("digest mismatch"), "{err}");
    }

    #[test]
    fn a_foreign_magic_is_refused_before_any_json_is_parsed() {
        let mut bytes = vec![0u8; PACK_HEADER_BYTES];
        bytes[..8].copy_from_slice(b"GPWMRDR1");
        let err = decode_pack(&bytes).unwrap_err().to_string();
        assert!(err.contains("magic"), "{err}");
    }
}
