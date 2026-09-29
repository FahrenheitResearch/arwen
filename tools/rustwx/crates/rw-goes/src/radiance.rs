//! ABI Level 1b radiances as brightness temperature, and their colocation
//! with a simulated brightness-temperature plane on the same fixed grid.
//!
//! Three subcommands live here, all on the `GPWMGOES` container the CWP
//! and cloud-top packs already use:
//!
//! * **bt** decodes one `ABI-L1b-Rad{C,F,M1,M2}` granule (`Rad`, its `DQF`,
//!   and the four Planck constants the file itself publishes), inverts the
//!   radiance to brightness temperature with the L1b product definition
//!   `T = (fk2 / ln(fk1 / L + 1) - bc1) / bc2`, and writes a
//!   `gpuwm-obs.goes-bt.v1` pack.  A pixel whose DQF is not 0 is NaN plus
//!   a count, never a value.  The `ABI-L2-ACM` clear-sky mask of the same
//!   scan may ride along as a `bcm` plane, and the `ABI-L2-CMIP` product of
//!   the same band as a cross-check plane: the CMIP brightness temperature
//!   is NOAA's own inversion of the same radiance, so the two must agree
//!   to the CMIP quantization, and the pack records how far they did.
//! * **colocate** places one or more simulated planes on the pack's grid
//!   BY LATTICE INDEX, never by interpolation: a simulated plane arrives
//!   with a sidecar naming its exact ABI 2 km global-lattice crop
//!   (the SimSat `abi_fixed_grid_crop`), the pack's own scan angles are
//!   proven to sit on that lattice, and pixel `(I, J)` of one is pixel
//!   `(I, J)` of the other.  It then accumulates the paired statistics per
//!   scene class (clear and cloudy on each side, from the ACM mask and the
//!   simulation's own condensate mask) and per satellite-zenith band, and
//!   the block means that stand in for superobservations.
//! * **quicklook** paints a pack plane or a set of simulated tiles through
//!   the rw-sat per-band palette, so a simulated and an observed panel of
//!   the same band wear the same colours.
//!
//! What this module deliberately does not do: it never resamples either
//! plane, never fills a gap, and never decides an observation error.  A
//! simulated plane whose sidecar is not on the lattice is refused with the
//! sidecar's own numbers.

use std::collections::BTreeMap;
use std::error::Error;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use rw_sat::abi::{GoesAbiField, GoesAbiScene, read_goes_abi_field, read_goes_abi_field_window};
use rw_sat::cloud::{CloudProduct, DqfRule, gate_by_dqf};
use rw_sat::goes::parse_goes_abi_filename;
use rw_sat::netcdf::{open_goes_netcdf_lossy, read_scaled_f32};

use crate::pack::{
    ArrayEntry, ContainerMeta, DqfRow, PayloadBuilder, ProjectionEntry, SourceEntry, boxed_error,
    decode_container, hex_sha256, write_container,
};

/// The schema this pack declares.
pub const BT_SCHEMA: &str = "gpuwm-obs.goes-bt.v1";
/// Every BT schema this build reads.
pub const BT_READABLE_SCHEMAS: &[&str] = &[BT_SCHEMA];

/// The receipt schemas of the three subcommands.
pub const BT_BUILD_SCHEMA: &str = "gpuwm-obs.goes-bt-build.v1";
pub const BT_VERIFY_SCHEMA: &str = "gpuwm-obs.goes-bt-verify.v1";
pub const COLOCATE_SCHEMA: &str = "gpuwm-da.abi-colocation.v1";
pub const QUICKLOOK_SCHEMA: &str = "gpuwm-da.abi-quicklook.v1";

/// The sidecar a simulated plane must carry (`<plane>.json`).
pub const SIM_PLANE_SCHEMA: &str = "gpuwm-da.simsat-plane.v1";

/// The ABI 2 km fixed-grid sample pitch (radians): 56 urad.
pub const ABI_2KM_PITCH_RAD: f64 = 56.0e-6;
/// Samples on one axis of the 2 km full disk.
pub const ABI_2KM_FULL_DISK_AXIS: i64 = 5424;
/// Tolerance for a decoded scan angle to sit on the lattice: the file
/// stores the axis as packed int16 with a float32 scale, so a decoded
/// centre lands within a few 1e-8 rad of the exact value; a twentieth of
/// a pixel is far outside that and far inside one sample.
pub const LATTICE_TOLERANCE_RAD: f64 = 0.05 * ABI_2KM_PITCH_RAD;

/// The brightness-temperature relation, verbatim in the pack.
pub const BT_FORMULA: &str =
    "T[K] = (planck_fk2 / ln(planck_fk1 / Rad + 1) - planck_bc1) / planck_bc2; Rad in mW m^-2 sr^-1 (cm^-1)^-1 as published, Rad <= 0 or DQF != 0 is NaN";
/// The DQF policy of the L1b decode, verbatim in the pack.
pub const L1B_DQF_POLICY: &str =
    "L1b DQF enumerated: 0 good keeps the pixel; 1 conditionally usable, 2 out of range, 3 no value, 4 focal-plane temperature exceeded and any fill DQF are NaN";

// ---------------------------------------------------------------------------
// the pack
// ---------------------------------------------------------------------------

/// The four Planck constants the L1b file publishes for its band.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct PlanckRow {
    pub fk1: f64,
    pub fk2: f64,
    pub bc1: f64,
    pub bc2: f64,
}

impl PlanckRow {
    /// Brightness temperature (K) of one radiance, NaN where the relation
    /// has no answer (non-positive or non-finite radiance).
    #[inline]
    pub fn brightness_temperature(&self, radiance: f64) -> f64 {
        if !(radiance > 0.0) || !radiance.is_finite() {
            return f64::NAN;
        }
        let ratio = self.fk1 / radiance + 1.0;
        if !(ratio > 1.0) {
            return f64::NAN;
        }
        (self.fk2 / ratio.ln() - self.bc1) / self.bc2
    }

    /// The radiance a brightness temperature (K) maps back to: the exact
    /// inverse of [`Self::brightness_temperature`], for the round-trip test.
    #[inline]
    pub fn radiance(&self, temperature_k: f64) -> f64 {
        let effective = self.bc1 + self.bc2 * temperature_k;
        self.fk1 / ((self.fk2 / effective).exp() - 1.0)
    }
}

/// What the L1b gate did.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq)]
pub struct BtCounts {
    pub total: usize,
    /// Radiance already fill or out of the published valid range.
    pub rad_missing: usize,
    /// Radiance present but not positive (no brightness temperature).
    pub rad_nonpositive: usize,
    /// DQF itself fill or unreadable.
    pub dqf_missing: usize,
    /// DQF not 0 (includes `dqf_missing`).
    pub dqf_bad: usize,
    /// Brightness temperatures that survive every gate.
    pub finite: usize,
}

/// The clear-sky mask that rode along, and how it was gated.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClearSkyRow {
    pub filename: String,
    pub sha256: String,
    pub dqf: DqfRow,
    /// BCM == 0 after the gate.
    pub clear: usize,
    /// BCM == 1 after the gate.
    pub cloudy: usize,
    /// NaN after the gate.
    pub missing: usize,
}

/// The cross-check against the L2 CMIP brightness temperature of the same
/// band and scan, over pixels finite in both.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossCheckRow {
    pub filename: String,
    pub sha256: String,
    pub n: usize,
    /// Mean of (this pack's BT minus CMIP), K.
    pub bias_k: f64,
    pub rmse_k: f64,
    pub max_abs_k: f64,
}

/// The observation's bookkeeping (design amendment F): when it was
/// measured, when the producer published it, when we first held it, and
/// which version it is.  Every field is the granule's own global attribute
/// except `received_utc` (the fetch record's wall) and the row-time model.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProvenanceRow {
    /// `dataset_name`: the granule's file name as the producer wrote it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dataset_name: Option<String>,
    /// `id`: the producer's UUID for this granule (its version identity).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// `date_created`: the publication instant (the `c` token of the file name).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub date_created: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub time_coverage_start: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub time_coverage_end: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub production_site: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub production_environment: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub production_data_source: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub platform_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub processing_level: Option<String>,
    /// When this system first held the granule (`--received-utc`, from the
    /// fetch manifest); `None` when the caller did not say, which the
    /// receipt then labels latency unverified.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub received_utc: Option<String>,
    /// How a pixel's measurement time is placed inside the scan window.
    pub row_time_model: String,
}

/// The row-time model, verbatim in every pack.
pub const ROW_TIME_MODEL: &str =
    "measurement time of a pixel = scan_start + (scan_end - scan_start) * (rows from the north + 1/2) / rows; the ABI \
     full disk scans north to south in about 22 swaths, so a row's time is inside its swath's about 26 s, which is the \
     stated uncertainty; the scan start is the nominal instant an analysis is compared at";

fn provenance_of(path: &Path, received_utc: Option<&str>) -> ProvenanceRow {
    let file = open_goes_netcdf_lossy(path).ok();
    let text = |name: &str| -> Option<String> {
        file.as_ref()
            .and_then(|f| f.attribute(name))
            .and_then(|a| a.as_string().map(|v| v.trim().to_string()))
            .filter(|v| !v.is_empty())
    };
    ProvenanceRow {
        dataset_name: text("dataset_name"),
        id: text("id"),
        date_created: text("date_created"),
        time_coverage_start: text("time_coverage_start"),
        time_coverage_end: text("time_coverage_end"),
        production_site: text("production_site"),
        production_environment: text("production_environment"),
        production_data_source: text("production_data_source"),
        platform_id: text("platform_ID"),
        processing_level: text("processing_level"),
        received_utc: received_utc.map(str::to_string),
        row_time_model: ROW_TIME_MODEL.to_string(),
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BtMeta {
    pub schema: String,
    pub status: String,
    pub satellite: String,
    pub sector: String,
    /// ABI band number of the radiance (1..=16).
    pub band: u8,
    pub scan_start: String,
    pub scan_end: String,
    /// Measurement, publication, receipt and version bookkeeping of the
    /// radiance granule (absent in packs written before it existed).
    #[serde(default)]
    pub provenance: ProvenanceRow,
    pub sources: Vec<SourceEntry>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub window: Option<[usize; 4]>,
    pub projection: ProjectionEntry,
    pub nx: usize,
    pub ny: usize,
    pub x_scan_rad: Vec<f64>,
    pub y_scan_rad: Vec<f64>,
    /// Plane name -> array key.  Declared order: bt, rad, lat, lon, then
    /// bcm when a clear-sky mask rode along, cmip_bt when a cross-check
    /// did, then the `_dqf` planes of every source.
    pub planes: BTreeMap<String, String>,
    pub plane_order: Vec<String>,
    pub arrays: BTreeMap<String, ArrayEntry>,
    pub payload_bytes: usize,
    pub content_sha256: String,
    pub planck: PlanckRow,
    pub brightness_temperature_formula: String,
    pub dqf_policy: String,
    pub counts: BtCounts,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub clear_sky_mask: Option<ClearSkyRow>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cross_check: Option<CrossCheckRow>,
}

impl ContainerMeta for BtMeta {
    const WRITTEN_SCHEMA: &'static str = BT_SCHEMA;
    const READABLE_SCHEMAS: &'static [&'static str] = BT_READABLE_SCHEMAS;

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

pub fn decode_bt_pack(bytes: &[u8]) -> Result<(BtMeta, Vec<u8>), Box<dyn Error>> {
    decode_container(bytes)
}

/// The options the three subcommands read; parsed by `main`.
#[derive(Debug, Default, Clone)]
pub struct RadianceOptions {
    pub rad: Option<PathBuf>,
    pub acm: Option<PathBuf>,
    pub cmip: Option<PathBuf>,
    pub out: Option<PathBuf>,
    pub window: Option<[usize; 4]>,
    pub pack: Option<PathBuf>,
    pub sim: Vec<PathBuf>,
    pub block: Option<usize>,
    pub stats: Option<PathBuf>,
    pub zenith_max_deg: Option<f64>,
    pub band: Option<u8>,
    pub plane_name: Option<String>,
    pub downsample: Option<usize>,
    pub bbox_index: Option<[i64; 4]>,
    /// `--received-utc`: when this system first held the granule.
    pub received_utc: Option<String>,
}

// ---------------------------------------------------------------------------
// bt
// ---------------------------------------------------------------------------

struct Granule {
    path: PathBuf,
    filename: String,
    bytes: usize,
    sha256: String,
}

fn identify(path: &Path) -> Result<Granule, Box<dyn Error>> {
    let filename = path
        .file_name()
        .map(|name| name.to_string_lossy().to_string())
        .unwrap_or_else(|| path.to_string_lossy().to_string());
    let raw = std::fs::read(path)
        .map_err(|err| boxed_error(format!("cannot read the granule {}: {err}", path.display())))?;
    Ok(Granule {
        path: path.to_path_buf(),
        filename,
        bytes: raw.len(),
        sha256: hex_sha256(&raw),
    })
}

fn read_field(
    path: &Path,
    variable: &str,
    window: Option<[usize; 4]>,
) -> Result<GoesAbiField, Box<dyn Error>> {
    match window {
        Some([x_start, x_count, y_start, y_count]) => {
            read_goes_abi_field_window(path, variable, x_start, x_count, y_start, y_count)
        }
        None => read_goes_abi_field(path, variable),
    }
}

fn read_scalar(path: &Path, name: &str) -> Result<f64, Box<dyn Error>> {
    let file = open_goes_netcdf_lossy(path)?;
    let variable = read_scaled_f32(&file, name)?;
    match variable.values.as_slice() {
        [value] if value.is_finite() => Ok(f64::from(*value)),
        [value] => Err(boxed_error(format!(
            "{}: {name} decodes to {value}, not a usable constant",
            path.display()
        ))),
        other => Err(boxed_error(format!(
            "{}: {name} holds {} values, a Planck constant is one scalar",
            path.display(),
            other.len()
        ))),
    }
}

fn same_scene(reference: &GoesAbiScene, other: &GoesAbiScene, what: &str) -> Result<(), Box<dyn Error>> {
    if other.start_time_utc != reference.start_time_utc {
        return Err(boxed_error(format!(
            "{what} is not the same scan: it starts {}, the radiance starts {}",
            other.start_time_utc, reference.start_time_utc
        )));
    }
    if other.satellite != reference.satellite {
        return Err(boxed_error(format!(
            "{what} is satellite {}, the radiance is {}",
            other.satellite.as_str(),
            reference.satellite.as_str()
        )));
    }
    if other.fixed_grid != reference.fixed_grid {
        return Err(boxed_error(format!(
            "{what} is on a {}x{} fixed grid whose scan angles differ from the radiance's {}x{}; \
             planes are only combined on one identical grid",
            other.fixed_grid.nx, other.fixed_grid.ny, reference.fixed_grid.nx, reference.fixed_grid.ny
        )));
    }
    if other.projection != reference.projection {
        return Err(boxed_error(format!(
            "{what} declares a different geostationary projection than the radiance"
        )));
    }
    Ok(())
}

fn push_plane(
    builder: &mut PayloadBuilder,
    planes: &mut BTreeMap<String, String>,
    order: &mut Vec<String>,
    name: &str,
    values: &[f32],
    shape: &[usize],
) -> Result<(), Box<dyn Error>> {
    let expected: usize = shape.iter().product();
    if values.len() != expected {
        return Err(boxed_error(format!(
            "plane {name} holds {} values but the declared shape {shape:?} needs {expected}",
            values.len()
        )));
    }
    let key = builder.push_f32(values, shape.to_vec());
    planes.insert(name.to_string(), key);
    order.push(name.to_string());
    Ok(())
}

fn enumerated_gate(values: &mut [f32], dqf: &[f32]) -> Result<DqfRow, Box<dyn Error>> {
    let report = gate_by_dqf(values, dqf, DqfRule::Enumerated)?;
    Ok(DqfRow {
        total: report.total,
        primary_missing: report.primary_missing,
        dqf_missing: report.dqf_missing,
        dqf_bad: report.dqf_bad,
        masked: report.masked,
        finite: report.finite,
    })
}

/// Invert one radiance plane, gated by its DQF, counting what happened.
pub fn brightness_temperature_plane(
    rad: &[f32],
    dqf: &[f32],
    planck: &PlanckRow,
) -> Result<(Vec<f32>, BtCounts), Box<dyn Error>> {
    if rad.len() != dqf.len() {
        return Err(boxed_error(format!(
            "the radiance plane holds {} values, its DQF {}; a quality flag that does not line \
             up pixel for pixel judges nothing",
            rad.len(),
            dqf.len()
        )));
    }
    let mut counts = BtCounts {
        total: rad.len(),
        ..BtCounts::default()
    };
    let mut out = vec![f32::NAN; rad.len()];
    for (idx, (&radiance, &flag)) in rad.iter().zip(dqf).enumerate() {
        let dqf_ok = flag == 0.0;
        if !flag.is_finite() {
            counts.dqf_missing += 1;
        }
        if !dqf_ok {
            counts.dqf_bad += 1;
        }
        if !radiance.is_finite() {
            counts.rad_missing += 1;
            continue;
        }
        if radiance <= 0.0 {
            counts.rad_nonpositive += 1;
            continue;
        }
        if !dqf_ok {
            continue;
        }
        let bt = planck.brightness_temperature(f64::from(radiance));
        if bt.is_finite() {
            out[idx] = bt as f32;
            counts.finite += 1;
        }
    }
    Ok((out, counts))
}

pub fn cmd_bt(options: &RadianceOptions) -> Result<String, Box<dyn Error>> {
    let rad_path = options
        .rad
        .as_deref()
        .ok_or_else(|| boxed_error("--rad is required (the ABI-L1b-Rad granule)"))?;
    let out = options
        .out
        .as_deref()
        .ok_or_else(|| boxed_error("--out is required (the pack destination)"))?;
    if out.is_dir() {
        return Err(boxed_error(format!(
            "--out {} is a directory; give the pack file path",
            out.display()
        )));
    }
    let window = options.window;

    let rad_granule = identify(rad_path)?;
    let parsed = parse_goes_abi_filename(&rad_granule.filename)?;
    if !parsed.product.to_ascii_uppercase().starts_with("ABI-L1B-RAD") {
        return Err(boxed_error(format!(
            "{} is a {} granule, not an ABI-L1b-Rad radiance file",
            rad_granule.filename, parsed.product
        )));
    }
    let band = parsed.channel.ok_or_else(|| {
        boxed_error(format!(
            "{} names no band (C01..C16) in its product token",
            rad_granule.filename
        ))
    })?;
    if let Some(requested) = options.band
        && requested != band
    {
        return Err(boxed_error(format!(
            "--band {requested} but {} is band {band}",
            rad_granule.filename
        )));
    }

    let rad = read_field(rad_path, "Rad", window)?;
    let rad_dqf = read_field(rad_path, "DQF", window)?;
    let planck = PlanckRow {
        fk1: read_scalar(rad_path, "planck_fk1")?,
        fk2: read_scalar(rad_path, "planck_fk2")?,
        bc1: read_scalar(rad_path, "planck_bc1")?,
        bc2: read_scalar(rad_path, "planck_bc2")?,
    };
    if !(planck.fk1 > 0.0 && planck.fk2 > 0.0 && planck.bc2 > 0.0) {
        return Err(boxed_error(format!(
            "{}: Planck constants {planck:?} are not a usable band response",
            rad_granule.filename
        )));
    }
    let (bt, counts) = brightness_temperature_plane(&rad.values, &rad_dqf.values, &planck)?;

    let scene = &rad.scene;
    let nx = scene.fixed_grid.nx;
    let ny = scene.fixed_grid.ny;
    let shape = vec![ny, nx];
    let (lat, lon) = scene.lat_lon_mesh();

    let mut sources: Vec<SourceEntry> = vec![SourceEntry {
        product: format!("RAD{band:02}"),
        filename: rad_granule.filename.clone(),
        bytes: rad_granule.bytes,
        sha256: rad_granule.sha256.clone(),
        dqf_rule: "enumerated".to_string(),
        condemn_mask: None,
        dqf: DqfRow {
            total: counts.total,
            primary_missing: counts.rad_missing,
            dqf_missing: counts.dqf_missing,
            dqf_bad: counts.dqf_bad,
            masked: counts.total - counts.finite - counts.rad_missing - counts.rad_nonpositive,
            finite: counts.finite,
        },
        dqf_plane: "rad_dqf".to_string(),
    }];

    let mut builder = PayloadBuilder::new();
    let mut planes: BTreeMap<String, String> = BTreeMap::new();
    let mut plane_order: Vec<String> = Vec::new();
    push_plane(&mut builder, &mut planes, &mut plane_order, "bt", &bt, &shape)?;
    push_plane(&mut builder, &mut planes, &mut plane_order, "rad", &rad.values, &shape)?;
    push_plane(&mut builder, &mut planes, &mut plane_order, "lat", &lat, &shape)?;
    push_plane(&mut builder, &mut planes, &mut plane_order, "lon", &lon, &shape)?;

    let mut dqf_planes: Vec<(String, Vec<f32>)> = vec![("rad_dqf".to_string(), rad_dqf.values)];

    let clear_sky_mask = match options.acm.as_deref() {
        None => None,
        Some(acm_path) => {
            let granule = identify(acm_path)?;
            let acm_parsed = parse_goes_abi_filename(&granule.filename)?;
            if !acm_parsed.product.to_ascii_uppercase().starts_with("ABI-L2-ACM") {
                return Err(boxed_error(format!(
                    "{} is a {} granule, not an ABI-L2-ACM clear-sky mask",
                    granule.filename, acm_parsed.product
                )));
            }
            let mut bcm = read_field(acm_path, CloudProduct::ClearSkyMask.primary_variable(), window)?;
            let dqf = read_field(acm_path, "DQF", window)?;
            same_scene(scene, &bcm.scene, &format!("the clear-sky mask {}", granule.filename))?;
            let row = enumerated_gate(&mut bcm.values, &dqf.values)?;
            let clear = bcm.values.iter().filter(|v| **v == 0.0).count();
            let cloudy = bcm.values.iter().filter(|v| **v == 1.0).count();
            let missing = bcm.values.iter().filter(|v| !v.is_finite()).count();
            push_plane(&mut builder, &mut planes, &mut plane_order, "bcm", &bcm.values, &shape)?;
            dqf_planes.push(("acm_dqf".to_string(), dqf.values));
            sources.push(SourceEntry {
                product: "ACM".to_string(),
                filename: granule.filename.clone(),
                bytes: granule.bytes,
                sha256: granule.sha256.clone(),
                dqf_rule: "enumerated".to_string(),
                condemn_mask: None,
                dqf: row,
                dqf_plane: "acm_dqf".to_string(),
            });
            Some(ClearSkyRow {
                filename: granule.filename,
                sha256: granule.sha256,
                dqf: row,
                clear,
                cloudy,
                missing,
            })
        }
    };

    let cross_check = match options.cmip.as_deref() {
        None => None,
        Some(cmip_path) => {
            let granule = identify(cmip_path)?;
            let cmip_parsed = parse_goes_abi_filename(&granule.filename)?;
            if !cmip_parsed.product.to_ascii_uppercase().starts_with("ABI-L2-CMIP") {
                return Err(boxed_error(format!(
                    "{} is a {} granule, not an ABI-L2-CMIP product",
                    granule.filename, cmip_parsed.product
                )));
            }
            if cmip_parsed.channel != Some(band) {
                return Err(boxed_error(format!(
                    "{} is band {:?}, the radiance is band {band}; a cross-check across bands \
                     compares two different quantities",
                    granule.filename, cmip_parsed.channel
                )));
            }
            let mut cmi = read_field(cmip_path, "CMI", window)?;
            let dqf = read_field(cmip_path, "DQF", window)?;
            same_scene(scene, &cmi.scene, &format!("the CMIP product {}", granule.filename))?;
            let row = enumerated_gate(&mut cmi.values, &dqf.values)?;
            let mut n = 0usize;
            let mut sum = 0.0f64;
            let mut sum_sq = 0.0f64;
            let mut max_abs = 0.0f64;
            for (&ours, &theirs) in bt.iter().zip(&cmi.values) {
                if ours.is_finite() && theirs.is_finite() {
                    let d = f64::from(ours) - f64::from(theirs);
                    n += 1;
                    sum += d;
                    sum_sq += d * d;
                    max_abs = max_abs.max(d.abs());
                }
            }
            push_plane(&mut builder, &mut planes, &mut plane_order, "cmip_bt", &cmi.values, &shape)?;
            dqf_planes.push((format!("cmip{band:02}_dqf"), dqf.values));
            sources.push(SourceEntry {
                product: format!("CMIP{band:02}"),
                filename: granule.filename.clone(),
                bytes: granule.bytes,
                sha256: granule.sha256.clone(),
                dqf_rule: "enumerated".to_string(),
                condemn_mask: None,
                dqf: row,
                dqf_plane: format!("cmip{band:02}_dqf"),
            });
            Some(CrossCheckRow {
                filename: granule.filename,
                sha256: granule.sha256,
                n,
                bias_k: if n > 0 { sum / n as f64 } else { f64::NAN },
                rmse_k: if n > 0 { (sum_sq / n as f64).sqrt() } else { f64::NAN },
                max_abs_k: if n > 0 { max_abs } else { f64::NAN },
            })
        }
    };

    for (name, values) in &dqf_planes {
        push_plane(&mut builder, &mut planes, &mut plane_order, name, values, &shape)?;
    }
    let (payload, arrays) = builder.finish();

    let meta = BtMeta {
        schema: BT_SCHEMA.to_string(),
        status: "READY".to_string(),
        satellite: scene.satellite.as_str().to_string(),
        sector: sector_token(scene),
        band,
        scan_start: crate::iso8601(scene.start_time_utc),
        scan_end: crate::iso8601(scene.end_time_utc),
        provenance: provenance_of(rad_path, options.received_utc.as_deref()),
        sources,
        window,
        projection: ProjectionEntry {
            perspective_point_height_m: scene.projection.perspective_point_height_m,
            semi_major_axis_m: scene.projection.semi_major_axis_m,
            semi_minor_axis_m: scene.projection.semi_minor_axis_m,
            longitude_of_projection_origin_deg: scene.projection.longitude_of_projection_origin_deg,
            sweep_angle_axis: scene.projection.sweep_angle_axis.as_str().to_string(),
        },
        nx,
        ny,
        x_scan_rad: scene.fixed_grid.x_scan_rad.clone(),
        y_scan_rad: scene.fixed_grid.y_scan_rad.clone(),
        planes,
        plane_order,
        arrays,
        payload_bytes: payload.len(),
        content_sha256: hex_sha256(&payload),
        planck,
        brightness_temperature_formula: BT_FORMULA.to_string(),
        dqf_policy: L1B_DQF_POLICY.to_string(),
        counts,
        clear_sky_mask,
        cross_check,
    };
    let pack_bytes = write_container(out, &meta, &payload)?;

    #[derive(Serialize)]
    struct BuildRecord<'a> {
        schema: &'static str,
        status: &'static str,
        path: String,
        pack_schema: &'a str,
        bytes: usize,
        payload_bytes: usize,
        content_sha256: &'a str,
        satellite: &'a str,
        sector: &'a str,
        band: u8,
        scan_start: &'a str,
        scan_end: &'a str,
        provenance: &'a ProvenanceRow,
        nx: usize,
        ny: usize,
        window: Option<[usize; 4]>,
        planes: &'a [String],
        planck: &'a PlanckRow,
        counts: &'a BtCounts,
        clear_sky_mask: &'a Option<ClearSkyRow>,
        cross_check: &'a Option<CrossCheckRow>,
        sources: &'a [SourceEntry],
    }
    let record = BuildRecord {
        schema: BT_BUILD_SCHEMA,
        status: "READY",
        path: out.to_string_lossy().to_string(),
        pack_schema: &meta.schema,
        bytes: pack_bytes,
        payload_bytes: meta.payload_bytes,
        content_sha256: &meta.content_sha256,
        satellite: &meta.satellite,
        sector: &meta.sector,
        band: meta.band,
        scan_start: &meta.scan_start,
        scan_end: &meta.scan_end,
        provenance: &meta.provenance,
        nx: meta.nx,
        ny: meta.ny,
        window: meta.window,
        planes: &meta.plane_order,
        planck: &meta.planck,
        counts: &meta.counts,
        clear_sky_mask: &meta.clear_sky_mask,
        cross_check: &meta.cross_check,
        sources: &meta.sources,
    };
    Ok(format!("{}\n", serde_json::to_string_pretty(&record)?))
}

fn sector_token(scene: &GoesAbiScene) -> String {
    use rw_sat::abi::AbiSector;
    match &scene.sector {
        AbiSector::Conus => "C".to_string(),
        AbiSector::FullDisk => "F".to_string(),
        AbiSector::Mesoscale1 => "M1".to_string(),
        AbiSector::Mesoscale2 => "M2".to_string(),
        AbiSector::Mesoscale => "M".to_string(),
        AbiSector::Unknown(token) => token.clone(),
    }
}

pub fn verify_bt_pack(path: &Path, bytes: &[u8]) -> Result<String, Box<dyn Error>> {
    let (meta, payload) = decode_bt_pack(bytes)?;
    let expected = meta.nx.saturating_mul(meta.ny);
    for name in &meta.plane_order {
        let key = meta.planes.get(name).ok_or_else(|| {
            boxed_error(format!("pack declares plane {name:?} in plane_order but no array key for it"))
        })?;
        let entry = meta.arrays.get(key).ok_or_else(|| {
            boxed_error(format!("pack plane {name:?} names array {key:?}, which the pack does not hold"))
        })?;
        let elements: usize = entry.shape.iter().product();
        if elements != expected {
            return Err(boxed_error(format!(
                "pack plane {name:?} holds {elements} values but the declared grid is {}x{}",
                meta.nx, meta.ny
            )));
        }
    }
    if meta.x_scan_rad.len() != meta.nx || meta.y_scan_rad.len() != meta.ny {
        return Err(boxed_error(format!(
            "pack states a {}x{} grid but carries {}/{} scan-angle values",
            meta.nx,
            meta.ny,
            meta.x_scan_rad.len(),
            meta.y_scan_rad.len()
        )));
    }
    for required in ["bt", "rad", "lat", "lon"] {
        if !meta.planes.contains_key(required) {
            return Err(boxed_error(format!("a {BT_SCHEMA} pack carries a {required:?} plane; this one does not")));
        }
    }
    for source in &meta.sources {
        if !meta.planes.contains_key(&source.dqf_plane) {
            return Err(boxed_error(format!(
                "pack source {} names DQF plane {:?}, which the pack does not hold",
                source.product, source.dqf_plane
            )));
        }
    }
    let lattice = lattice_indices(&meta.x_scan_rad, &meta.y_scan_rad).ok();

    #[derive(Serialize)]
    struct VerifyRecord<'a> {
        schema: &'static str,
        status: &'static str,
        path: String,
        pack_schema: &'a str,
        bytes: usize,
        payload_bytes: usize,
        content_sha256: &'a str,
        satellite: &'a str,
        sector: &'a str,
        band: u8,
        scan_start: &'a str,
        scan_end: &'a str,
        nx: usize,
        ny: usize,
        window: Option<[usize; 4]>,
        planes: &'a [String],
        on_abi_2km_lattice: bool,
        lattice_x_index_min: Option<i64>,
        lattice_y_index_max: Option<i64>,
        planck: &'a PlanckRow,
        counts: &'a BtCounts,
        clear_sky_mask: &'a Option<ClearSkyRow>,
        cross_check: &'a Option<CrossCheckRow>,
        sources: &'a [SourceEntry],
    }
    let record = VerifyRecord {
        schema: BT_VERIFY_SCHEMA,
        status: "PASS",
        path: path.to_string_lossy().to_string(),
        pack_schema: &meta.schema,
        bytes: bytes.len(),
        payload_bytes: payload.len(),
        content_sha256: &meta.content_sha256,
        satellite: &meta.satellite,
        sector: &meta.sector,
        band: meta.band,
        scan_start: &meta.scan_start,
        scan_end: &meta.scan_end,
        nx: meta.nx,
        ny: meta.ny,
        window: meta.window,
        planes: &meta.plane_order,
        on_abi_2km_lattice: lattice.is_some(),
        lattice_x_index_min: lattice.as_ref().map(|l| l.x_index[0]),
        lattice_y_index_max: lattice.as_ref().map(|l| l.y_index[0]),
        planck: &meta.planck,
        counts: &meta.counts,
        clear_sky_mask: &meta.clear_sky_mask,
        cross_check: &meta.cross_check,
        sources: &meta.sources,
    };
    Ok(format!("{}\n", serde_json::to_string_pretty(&record)?))
}

// ---------------------------------------------------------------------------
// lattice arithmetic
// ---------------------------------------------------------------------------

/// The signed global 2 km lattice index of every column and row of a
/// pack: `angle = (index + 1/2) * 56 urad`, the sub-satellite point being
/// the shared corner of four pixels.  Refused when any axis value is not
/// on the lattice, with the offending value named.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LatticeIndices {
    /// Per pack column, west to east as stored.
    pub x_index: Vec<i64>,
    /// Per pack row, as stored (north first for every ABI product).
    pub y_index: Vec<i64>,
}

pub fn lattice_index(angle_rad: f64) -> Option<i64> {
    let raw = angle_rad / ABI_2KM_PITCH_RAD - 0.5;
    let rounded = raw.round();
    let residual = (raw - rounded).abs() * ABI_2KM_PITCH_RAD;
    (residual <= LATTICE_TOLERANCE_RAD).then_some(rounded as i64)
}

pub fn lattice_indices(x_scan_rad: &[f64], y_scan_rad: &[f64]) -> Result<LatticeIndices, Box<dyn Error>> {
    let axis = |name: &str, values: &[f64]| -> Result<Vec<i64>, Box<dyn Error>> {
        let mut out: Vec<i64> = Vec::with_capacity(values.len());
        for (position, &angle) in values.iter().enumerate() {
            let index = lattice_index(angle).ok_or_else(|| {
                boxed_error(format!(
                    "{name}[{position}] = {angle:.9e} rad is not on the ABI 2 km lattice \
                     ((index + 1/2) * 56 urad within {:.1e} rad); this pack cannot be \
                     colocated by index",
                    LATTICE_TOLERANCE_RAD
                ))
            })?;
            if let Some(&previous) = out.last()
                && (index - previous).abs() != 1
            {
                return Err(boxed_error(format!(
                    "{name}[{position}] steps from lattice index {previous} to {index}; a fixed \
                     grid axis advances one sample at a time"
                )));
            }
            out.push(index);
        }
        if out.is_empty() {
            return Err(boxed_error(format!("{name} is empty")));
        }
        Ok(out)
    };
    Ok(LatticeIndices {
        x_index: axis("x_scan_rad", x_scan_rad)?,
        y_index: axis("y_scan_rad", y_scan_rad)?,
    })
}

impl LatticeIndices {
    /// Pack column of a global x index, if the pack covers it.
    pub fn column(&self, x_index: i64) -> Option<usize> {
        let first = self.x_index[0];
        let step = if self.x_index.len() > 1 { self.x_index[1] - first } else { 1 };
        let offset = (x_index - first) * step;
        (offset >= 0 && (offset as usize) < self.x_index.len()).then_some(offset as usize)
    }

    /// Pack row of a global y index, if the pack covers it.
    pub fn row(&self, y_index: i64) -> Option<usize> {
        let first = self.y_index[0];
        let step = if self.y_index.len() > 1 { self.y_index[1] - first } else { -1 };
        let offset = (y_index - first) * step;
        (offset >= 0 && (offset as usize) < self.y_index.len()).then_some(offset as usize)
    }
}

// ---------------------------------------------------------------------------
// simulated planes
// ---------------------------------------------------------------------------

/// The crop block of a simulated plane's sidecar: the SimSat
/// `abi_fixed_grid_crop` dictionary, verbatim.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct CropBlock {
    pub x_index_min: i64,
    pub x_index_max: i64,
    pub y_index_min: i64,
    pub y_index_max: i64,
    pub nx: usize,
    pub ny: usize,
    #[serde(default)]
    pub sample_angle_urad: Option<f64>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SimSidecar {
    pub schema: String,
    /// `[ny, nx]`, row 0 north.
    pub shape: [usize; 2],
    pub abi_fixed_grid_crop: CropBlock,
    /// ABI band the plane simulates.
    pub band: u8,
    /// Optional condensate mask on the same raster: u8, 0 clear, 1
    /// condensate, 255 no data (the SimSat `cloud-mask-out` plane).
    #[serde(default)]
    pub mask_plane: Option<String>,
    #[serde(default)]
    pub label: Option<String>,
}

/// One simulated tile loaded and placed on the global lattice.
pub struct SimTile {
    pub path: PathBuf,
    pub sidecar: SimSidecar,
    pub values: Vec<f32>,
    pub mask: Option<Vec<u8>>,
}

impl SimTile {
    pub fn nx(&self) -> usize {
        self.sidecar.shape[1]
    }
    pub fn ny(&self) -> usize {
        self.sidecar.shape[0]
    }
    /// Global x index of column `c`.
    pub fn x_index(&self, c: usize) -> i64 {
        self.sidecar.abi_fixed_grid_crop.x_index_min + c as i64
    }
    /// Global y index of row `r` (row 0 north = `y_index_max`).
    pub fn y_index(&self, r: usize) -> i64 {
        self.sidecar.abi_fixed_grid_crop.y_index_max - r as i64
    }
}

fn read_f32le_plane(path: &Path, expected: usize) -> Result<Vec<f32>, Box<dyn Error>> {
    let raw = std::fs::read(path)
        .map_err(|err| boxed_error(format!("cannot read the plane {}: {err}", path.display())))?;
    if raw.len() != expected * 4 {
        return Err(boxed_error(format!(
            "{} holds {} bytes, the sidecar's shape needs {} float32 values ({} bytes)",
            path.display(),
            raw.len(),
            expected,
            expected * 4
        )));
    }
    Ok(raw
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

pub fn load_sim_tile(path: &Path) -> Result<SimTile, Box<dyn Error>> {
    let sidecar_path = PathBuf::from(format!("{}.json", path.to_string_lossy()));
    let text = std::fs::read_to_string(&sidecar_path).map_err(|err| {
        boxed_error(format!(
            "a simulated plane needs its sidecar {}: {err}",
            sidecar_path.display()
        ))
    })?;
    let sidecar: SimSidecar = serde_json::from_str(&text)
        .map_err(|err| boxed_error(format!("{}: {err}", sidecar_path.display())))?;
    if sidecar.schema != SIM_PLANE_SCHEMA {
        return Err(boxed_error(format!(
            "{} declares schema {:?}; this build reads {SIM_PLANE_SCHEMA}",
            sidecar_path.display(),
            sidecar.schema
        )));
    }
    let crop = &sidecar.abi_fixed_grid_crop;
    if let Some(pitch) = crop.sample_angle_urad
        && (pitch - 56.0).abs() > 1.0e-9
    {
        return Err(boxed_error(format!(
            "{}: the crop is a {pitch} urad lattice, not the 56 urad ABI 2 km lattice",
            sidecar_path.display()
        )));
    }
    let [ny, nx] = sidecar.shape;
    if crop.nx != nx || crop.ny != ny {
        return Err(boxed_error(format!(
            "{}: shape {:?} disagrees with the crop's {}x{}",
            sidecar_path.display(),
            sidecar.shape,
            crop.nx,
            crop.ny
        )));
    }
    if crop.x_index_max - crop.x_index_min + 1 != nx as i64
        || crop.y_index_max - crop.y_index_min + 1 != ny as i64
    {
        return Err(boxed_error(format!(
            "{}: crop indices x {}..{} y {}..{} do not span {nx}x{ny}",
            sidecar_path.display(),
            crop.x_index_min,
            crop.x_index_max,
            crop.y_index_min,
            crop.y_index_max
        )));
    }
    let half = ABI_2KM_FULL_DISK_AXIS / 2;
    if crop.x_index_min < -half || crop.x_index_max >= half || crop.y_index_min < -half || crop.y_index_max >= half {
        return Err(boxed_error(format!(
            "{}: crop indices leave the {}-sample full disk",
            sidecar_path.display(),
            ABI_2KM_FULL_DISK_AXIS
        )));
    }
    let values = read_f32le_plane(path, nx * ny)?;
    let mask = match &sidecar.mask_plane {
        None => None,
        Some(name) => {
            let mask_path = if Path::new(name).is_absolute() {
                PathBuf::from(name)
            } else {
                path.parent().map(|p| p.join(name)).unwrap_or_else(|| PathBuf::from(name))
            };
            let raw = std::fs::read(&mask_path).map_err(|err| {
                boxed_error(format!("cannot read the mask plane {}: {err}", mask_path.display()))
            })?;
            if raw.len() != nx * ny {
                return Err(boxed_error(format!(
                    "{} holds {} bytes, the plane is {nx}x{ny}",
                    mask_path.display(),
                    raw.len()
                )));
            }
            Some(raw)
        }
    };
    Ok(SimTile {
        path: path.to_path_buf(),
        sidecar,
        values,
        mask,
    })
}

// ---------------------------------------------------------------------------
// colocate
// ---------------------------------------------------------------------------

/// Running moments of paired (observed, simulated) values.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq)]
pub struct PairMoments {
    pub n: usize,
    pub sum_obs: f64,
    pub sum_sim: f64,
    pub sum_obs_sq: f64,
    pub sum_sim_sq: f64,
    pub sum_obs_sim: f64,
    pub min_diff: f64,
    pub max_diff: f64,
}

impl PairMoments {
    pub fn new() -> Self {
        Self {
            min_diff: f64::INFINITY,
            max_diff: f64::NEG_INFINITY,
            ..Self::default()
        }
    }

    #[inline]
    pub fn push(&mut self, obs: f64, sim: f64) {
        self.n += 1;
        self.sum_obs += obs;
        self.sum_sim += sim;
        self.sum_obs_sq += obs * obs;
        self.sum_sim_sq += sim * sim;
        self.sum_obs_sim += obs * sim;
        let d = sim - obs;
        self.min_diff = self.min_diff.min(d);
        self.max_diff = self.max_diff.max(d);
    }

    pub fn mean_obs(&self) -> f64 {
        if self.n == 0 { f64::NAN } else { self.sum_obs / self.n as f64 }
    }

    pub fn mean_sim(&self) -> f64 {
        if self.n == 0 { f64::NAN } else { self.sum_sim / self.n as f64 }
    }

    /// Mean of (simulated minus observed).
    pub fn bias(&self) -> f64 {
        self.mean_sim() - self.mean_obs()
    }

    pub fn rmse(&self) -> f64 {
        if self.n == 0 {
            return f64::NAN;
        }
        let n = self.n as f64;
        ((self.sum_sim_sq - 2.0 * self.sum_obs_sim + self.sum_obs_sq) / n).max(0.0).sqrt()
    }

    pub fn std_obs(&self) -> f64 {
        if self.n < 2 {
            return f64::NAN;
        }
        let n = self.n as f64;
        (self.sum_obs_sq / n - (self.sum_obs / n).powi(2)).max(0.0).sqrt()
    }

    pub fn std_sim(&self) -> f64 {
        if self.n < 2 {
            return f64::NAN;
        }
        let n = self.n as f64;
        (self.sum_sim_sq / n - (self.sum_sim / n).powi(2)).max(0.0).sqrt()
    }

    pub fn correlation(&self) -> f64 {
        let so = self.std_obs();
        let ss = self.std_sim();
        if !(so > 0.0 && ss > 0.0) {
            return f64::NAN;
        }
        let n = self.n as f64;
        (self.sum_obs_sim / n - self.mean_obs() * self.mean_sim()) / (so * ss)
    }

    /// Least-squares `obs = intercept + slope * sim`, and the residual rms
    /// after that correction, over these pairs.
    pub fn linear_correction(&self) -> Option<(f64, f64, f64)> {
        let ss = self.std_sim();
        if !(ss > 0.0) || self.n < 3 {
            return None;
        }
        let n = self.n as f64;
        let cov = self.sum_obs_sim / n - self.mean_obs() * self.mean_sim();
        let slope = cov / (ss * ss);
        let intercept = self.mean_obs() - slope * self.mean_sim();
        let r = self.correlation();
        let residual = (self.std_obs().powi(2) * (1.0 - r * r)).max(0.0).sqrt();
        Some((intercept, slope, residual))
    }
}

/// A class of paired pixels with its moments and a difference histogram.
#[derive(Debug, Clone, Serialize)]
pub struct ClassStats {
    pub n: usize,
    pub mean_obs_k: f64,
    pub mean_sim_k: f64,
    pub bias_k: f64,
    pub rmse_k: f64,
    pub std_obs_k: f64,
    pub std_sim_k: f64,
    pub correlation: f64,
    /// `obs = intercept + slope * sim` fitted on this class alone.
    pub linear_fit_intercept_k: Option<f64>,
    pub linear_fit_slope: Option<f64>,
    /// Residual rms after that fit: the after-correction rmse.
    pub rmse_after_linear_k: Option<f64>,
    pub min_diff_k: f64,
    pub max_diff_k: f64,
    pub moments: PairMoments,
    /// Counts of (sim minus obs) in 1 K bins from -60 to +60 K; the two
    /// end bins collect everything beyond.
    pub diff_histogram_1k: Vec<usize>,
}

pub const HISTOGRAM_HALF_WIDTH_K: i64 = 60;

#[derive(Debug, Clone)]
struct ClassAccumulator {
    moments: PairMoments,
    histogram: Vec<usize>,
}

impl ClassAccumulator {
    fn new() -> Self {
        Self {
            moments: PairMoments::new(),
            histogram: vec![0; (2 * HISTOGRAM_HALF_WIDTH_K) as usize],
        }
    }

    #[inline]
    fn push(&mut self, obs: f64, sim: f64) {
        self.moments.push(obs, sim);
        let d = sim - obs;
        let bin = (d.floor() as i64 + HISTOGRAM_HALF_WIDTH_K).clamp(0, 2 * HISTOGRAM_HALF_WIDTH_K - 1);
        self.histogram[bin as usize] += 1;
    }

    fn finish(self) -> ClassStats {
        let m = self.moments;
        let fit = m.linear_correction();
        ClassStats {
            n: m.n,
            mean_obs_k: m.mean_obs(),
            mean_sim_k: m.mean_sim(),
            bias_k: m.bias(),
            rmse_k: m.rmse(),
            std_obs_k: m.std_obs(),
            std_sim_k: m.std_sim(),
            correlation: m.correlation(),
            linear_fit_intercept_k: fit.map(|f| f.0),
            linear_fit_slope: fit.map(|f| f.1),
            rmse_after_linear_k: fit.map(|f| f.2),
            min_diff_k: if m.n == 0 { f64::NAN } else { m.min_diff },
            max_diff_k: if m.n == 0 { f64::NAN } else { m.max_diff },
            moments: m,
            diff_histogram_1k: self.histogram,
        }
    }
}

/// The scene classes a pair can fall in.  `all` is every pair; the obs
/// side comes from the ACM clear-sky mask (absent mask: unknown), the
/// simulated side from the tile's condensate mask (absent: unknown).
pub const CLASSES: &[&str] = &[
    "all",
    "obs_clear",
    "obs_cloudy",
    "sim_clear",
    "sim_cloudy",
    "both_clear",
    "both_cloudy",
    "obs_clear_sim_cloudy",
    "obs_cloudy_sim_clear",
];

/// Zenith bands the classes are also split into (degrees, inclusive
/// upper bound).  `all` has no bound.
pub const ZENITH_BANDS: &[(&str, f64)] = &[
    ("all", f64::INFINITY),
    ("le40", 40.0),
    ("le60", 60.0),
    ("le70", 70.0),
];

/// Satellite zenith angle (degrees) at a surface point, spherical earth
/// of the projection's semi-major axis: a diagnostic for the zenith bands,
/// not a navigation quantity.
pub fn satellite_zenith_deg(lat_deg: f64, lon_deg: f64, sub_lon_deg: f64, radius_m: f64, height_m: f64) -> f64 {
    let lat = lat_deg.to_radians();
    let dlon = (lon_deg - sub_lon_deg).to_radians();
    let cos_beta = (lat.cos() * dlon.cos()).clamp(-1.0, 1.0);
    let sin_beta = (1.0 - cos_beta * cos_beta).max(0.0).sqrt();
    let rs = radius_m + height_m;
    (rs * sin_beta).atan2(rs * cos_beta - radius_m).to_degrees()
}

#[derive(Debug, Clone, Default)]
struct BlockAccumulator {
    n_obs: usize,
    n_sim: usize,
    pairs: PairMoments,
    sum_lat: f64,
    sum_lon: f64,
    sum_zenith: f64,
    obs_clear: usize,
    obs_cloudy: usize,
    sim_clear: usize,
    sim_cloudy: usize,
    both_clear: PairMoments,
}

pub fn cmd_colocate(options: &RadianceOptions) -> Result<String, Box<dyn Error>> {
    let pack_path = options
        .pack
        .as_deref()
        .ok_or_else(|| boxed_error("--pack is required (the gpuwm-obs.goes-bt.v1 pack)"))?;
    if options.sim.is_empty() {
        return Err(boxed_error("at least one --sim PLANE (with its PLANE.json sidecar) is required"));
    }
    let out = options
        .out
        .as_deref()
        .ok_or_else(|| boxed_error("--out is required (the block table CSV)"))?;
    let stats_path = options
        .stats
        .as_deref()
        .ok_or_else(|| boxed_error("--stats is required (the pixel statistics JSON)"))?;
    let block = options.block.unwrap_or(24);
    if block == 0 {
        return Err(boxed_error("--block must be at least 1 pixel"));
    }
    let zenith_max = options.zenith_max_deg.unwrap_or(f64::INFINITY);

    let bytes = std::fs::read(pack_path)
        .map_err(|err| boxed_error(format!("cannot read {}: {err}", pack_path.display())))?;
    let (meta, payload) = decode_bt_pack(&bytes)?;
    let plane = |name: &str| -> Result<Option<Vec<f32>>, Box<dyn Error>> {
        let Some(key) = meta.planes.get(name) else {
            return Ok(None);
        };
        let entry = &meta.arrays[key];
        let slice = &payload[entry.offset..entry.offset + entry.bytes];
        Ok(Some(
            slice
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
        ))
    };
    let bt = plane("bt")?.ok_or_else(|| boxed_error("the pack carries no bt plane"))?;
    let lat = plane("lat")?.ok_or_else(|| boxed_error("the pack carries no lat plane"))?;
    let lon = plane("lon")?.ok_or_else(|| boxed_error("the pack carries no lon plane"))?;
    let bcm = plane("bcm")?;
    let lattice = lattice_indices(&meta.x_scan_rad, &meta.y_scan_rad)?;
    let nx = meta.nx;

    let tiles: Vec<SimTile> = options
        .sim
        .iter()
        .map(|path| load_sim_tile(path))
        .collect::<Result<_, _>>()?;
    for tile in &tiles {
        if tile.sidecar.band != meta.band {
            return Err(boxed_error(format!(
                "{} simulates band {}, the pack is band {}",
                tile.path.display(),
                tile.sidecar.band,
                meta.band
            )));
        }
    }

    // Per class, per zenith band.
    let mut classes: BTreeMap<String, ClassAccumulator> = BTreeMap::new();
    for class in CLASSES {
        for (band, _) in ZENITH_BANDS {
            classes.insert(format!("{class}/{band}"), ClassAccumulator::new());
        }
    }
    let mut blocks: BTreeMap<(i64, i64), BlockAccumulator> = BTreeMap::new();
    // Pixels seen once already (tile overlap): the first tile wins and the
    // overlap is counted, never double-weighted.
    let mut seen: Vec<u8> = vec![0; bt.len()];
    let mut overlap = 0usize;
    let mut sim_finite = 0usize;
    let mut sim_off_pack = 0usize;
    let mut pairs = 0usize;
    let mut zenith_rejected = 0usize;
    let half = ABI_2KM_FULL_DISK_AXIS / 2;
    let sub_lon = meta.projection.longitude_of_projection_origin_deg;
    let radius = meta.projection.semi_major_axis_m;
    let height = meta.projection.perspective_point_height_m;

    for tile in &tiles {
        let tnx = tile.nx();
        for r in 0..tile.ny() {
            let Some(row) = lattice.row(tile.y_index(r)) else {
                sim_off_pack += tnx;
                continue;
            };
            for c in 0..tnx {
                let sim = tile.values[r * tnx + c];
                if !sim.is_finite() {
                    continue;
                }
                sim_finite += 1;
                let Some(col) = lattice.column(tile.x_index(c)) else {
                    sim_off_pack += 1;
                    continue;
                };
                let idx = row * nx + col;
                if seen[idx] != 0 {
                    overlap += 1;
                    continue;
                }
                seen[idx] = 1;
                let gi = tile.x_index(c);
                let gj = tile.y_index(r);
                let key = ((gi + half).div_euclid(block as i64), (gj + half).div_euclid(block as i64));
                let entry = blocks.entry(key).or_default();
                entry.n_sim += 1;
                let sim_mask = tile.mask.as_ref().map(|m| m[r * tnx + c]);
                match sim_mask {
                    Some(0) => entry.sim_clear += 1,
                    Some(1) => entry.sim_cloudy += 1,
                    _ => {}
                }
                let obs = bt[idx];
                let obs_mask = bcm.as_ref().map(|b| b[idx]);
                if obs.is_finite() {
                    entry.n_obs += 1;
                    match obs_mask {
                        Some(v) if v == 0.0 => entry.obs_clear += 1,
                        Some(v) if v == 1.0 => entry.obs_cloudy += 1,
                        _ => {}
                    }
                } else {
                    continue;
                }
                let plat = f64::from(lat[idx]);
                let plon = f64::from(lon[idx]);
                let zenith = satellite_zenith_deg(plat, plon, sub_lon, radius, height);
                if zenith > zenith_max {
                    zenith_rejected += 1;
                    continue;
                }
                pairs += 1;
                let obs = f64::from(obs);
                let sim = f64::from(sim);
                entry.pairs.push(obs, sim);
                entry.sum_lat += plat;
                entry.sum_lon += plon;
                entry.sum_zenith += zenith;
                let obs_clear = matches!(obs_mask, Some(v) if v == 0.0);
                let obs_cloudy = matches!(obs_mask, Some(v) if v == 1.0);
                let sim_clear = matches!(sim_mask, Some(0));
                let sim_cloudy = matches!(sim_mask, Some(1));
                if obs_clear && sim_clear {
                    entry.both_clear.push(obs, sim);
                }
                let flags = [
                    ("all", true),
                    ("obs_clear", obs_clear),
                    ("obs_cloudy", obs_cloudy),
                    ("sim_clear", sim_clear),
                    ("sim_cloudy", sim_cloudy),
                    ("both_clear", obs_clear && sim_clear),
                    ("both_cloudy", obs_cloudy && sim_cloudy),
                    ("obs_clear_sim_cloudy", obs_clear && sim_cloudy),
                    ("obs_cloudy_sim_clear", obs_cloudy && sim_clear),
                ];
                for (class, on) in flags {
                    if !on {
                        continue;
                    }
                    for (band, bound) in ZENITH_BANDS {
                        if zenith <= *bound {
                            classes
                                .get_mut(&format!("{class}/{band}"))
                                .expect("class table is pre-filled")
                                .push(obs, sim);
                        }
                    }
                }
            }
        }
    }

    // The block table.
    let mut csv = String::new();
    csv.push_str(
        "block_x,block_y,lat_mean_deg,lon_mean_deg,zenith_mean_deg,n_sim,n_obs,n_pair,obs_mean_k,sim_mean_k,bias_k,rmse_k,obs_clear,obs_cloudy,sim_clear,sim_cloudy,n_both_clear,obs_mean_both_clear_k,sim_mean_both_clear_k\n",
    );
    let mut block_rows = 0usize;
    for ((bx, by), acc) in &blocks {
        if acc.pairs.n == 0 {
            continue;
        }
        block_rows += 1;
        let n = acc.pairs.n as f64;
        csv.push_str(&format!(
            "{bx},{by},{:.4},{:.4},{:.2},{},{},{},{:.3},{:.3},{:.3},{:.3},{},{},{},{},{},{},{}\n",
            acc.sum_lat / n,
            acc.sum_lon / n,
            acc.sum_zenith / n,
            acc.n_sim,
            acc.n_obs,
            acc.pairs.n,
            acc.pairs.mean_obs(),
            acc.pairs.mean_sim(),
            acc.pairs.bias(),
            acc.pairs.rmse(),
            acc.obs_clear,
            acc.obs_cloudy,
            acc.sim_clear,
            acc.sim_cloudy,
            acc.both_clear.n,
            fmt_opt(acc.both_clear.mean_obs()),
            fmt_opt(acc.both_clear.mean_sim()),
        ));
    }
    rw_store::atomic::atomic_write_bytes(out, csv.as_bytes())?;

    #[derive(Serialize)]
    struct TileRecord {
        path: String,
        label: Option<String>,
        shape: [usize; 2],
        crop: CropBlock,
        has_mask: bool,
        finite: usize,
    }
    #[derive(Serialize)]
    struct ColocationRecord<'a> {
        schema: &'static str,
        status: &'static str,
        pack: String,
        pack_schema: &'a str,
        pack_content_sha256: &'a str,
        satellite: &'a str,
        band: u8,
        scan_start: &'a str,
        scan_end: &'a str,
        has_clear_sky_mask: bool,
        block_pixels: usize,
        block_table: String,
        block_rows: usize,
        zenith_max_deg: f64,
        lattice: &'static str,
        pack_lattice_x_index_min: i64,
        pack_lattice_y_index_max: i64,
        tiles: Vec<TileRecord>,
        counts: BTreeMap<&'static str, usize>,
        classes: BTreeMap<String, ClassStats>,
    }
    let mut counts = BTreeMap::new();
    counts.insert("sim_finite", sim_finite);
    counts.insert("sim_off_pack", sim_off_pack);
    counts.insert("tile_overlap_skipped", overlap);
    counts.insert("pairs", pairs);
    counts.insert("zenith_rejected", zenith_rejected);
    let record = ColocationRecord {
        schema: COLOCATE_SCHEMA,
        status: if pairs > 0 { "READY" } else { "EMPTY" },
        pack: pack_path.to_string_lossy().to_string(),
        pack_schema: &meta.schema,
        pack_content_sha256: &meta.content_sha256,
        satellite: &meta.satellite,
        band: meta.band,
        scan_start: &meta.scan_start,
        scan_end: &meta.scan_end,
        has_clear_sky_mask: bcm.is_some(),
        block_pixels: block,
        block_table: out.to_string_lossy().to_string(),
        block_rows,
        zenith_max_deg: zenith_max,
        lattice: "ABI 2 km fixed grid, (index + 1/2) * 56 urad, sub-satellite point the corner of four pixels; pixel (I, J) of the simulation is pixel (I, J) of the pack, no interpolation",
        pack_lattice_x_index_min: lattice.x_index[0],
        pack_lattice_y_index_max: lattice.y_index[0],
        tiles: tiles
            .iter()
            .map(|tile| TileRecord {
                path: tile.path.to_string_lossy().to_string(),
                label: tile.sidecar.label.clone(),
                shape: tile.sidecar.shape,
                crop: tile.sidecar.abi_fixed_grid_crop.clone(),
                has_mask: tile.mask.is_some(),
                finite: tile.values.iter().filter(|v| v.is_finite()).count(),
            })
            .collect(),
        counts,
        classes: classes.into_iter().map(|(k, v)| (k, v.finish())).collect(),
    };
    let text = format!("{}\n", serde_json::to_string_pretty(&record)?);
    rw_store::atomic::atomic_write_bytes(stats_path, text.as_bytes())?;
    Ok(text)
}

// ---------------------------------------------------------------------------
// superobs: the observation-only block table (the radiance stream's rows)
// ---------------------------------------------------------------------------

pub const SUPEROBS_SCHEMA: &str = "gpuwm-da.abi-superobs.v1";

/// One block of the observation-only table: the clear pixels' mean and
/// spread with the counts every gate reads.
#[derive(Debug, Clone, Default)]
struct SuperobsBlock {
    n_pixel: usize,
    n_obs: usize,
    n_clear: usize,
    n_cloudy: usize,
    sum_lat: f64,
    sum_lon: f64,
    sum_zenith: f64,
    clear_sum: f64,
    clear_sum_sq: f64,
    all_sum: f64,
}

/// `rw_goes superobs`: block means of the CLEAR pixels of a bt pack, with
/// no simulated plane at all.  The ensemble filter's radiance stream reads
/// this table as its observation vector (one block, one row): the column
/// names it shares with the colocation table (`n_pair`, `n_both_clear`,
/// `obs_mean_both_clear_k`) mean the same population read by the
/// observation side alone (the clear-sky mask's clear pixels among the
/// finite ones), so the stream's quality control (`minimum_pixels`,
/// `minimum_clear_fraction`, the zenith gate) applies unchanged.  Beside
/// them the table carries `obs_std_clear_k`, the within-block standard
/// deviation of the clear pixels (the representativeness read of a
/// block), and `obs_mean_all_k`, the mean over every finite pixel.  A pack
/// without a clear-sky mask is refused: a block mean over unknown scenes
/// is not a clear-sky observation.
pub fn cmd_superobs(options: &RadianceOptions) -> Result<String, Box<dyn Error>> {
    let pack_path = options
        .pack
        .as_deref()
        .ok_or_else(|| boxed_error("--pack is required (the gpuwm-obs.goes-bt.v1 pack)"))?;
    let out = options
        .out
        .as_deref()
        .ok_or_else(|| boxed_error("--out is required (the block table CSV)"))?;
    let block = options.block.unwrap_or(24);
    if block == 0 {
        return Err(boxed_error("--block must be at least 1 pixel"));
    }
    let zenith_max = options.zenith_max_deg.unwrap_or(f64::INFINITY);

    let bytes = std::fs::read(pack_path)
        .map_err(|err| boxed_error(format!("cannot read {}: {err}", pack_path.display())))?;
    let (meta, payload) = decode_bt_pack(&bytes)?;
    let plane = |name: &str| -> Result<Option<Vec<f32>>, Box<dyn Error>> {
        let Some(key) = meta.planes.get(name) else {
            return Ok(None);
        };
        let entry = &meta.arrays[key];
        let slice = &payload[entry.offset..entry.offset + entry.bytes];
        Ok(Some(
            slice
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
        ))
    };
    let bt = plane("bt")?.ok_or_else(|| boxed_error("the pack carries no bt plane"))?;
    let lat = plane("lat")?.ok_or_else(|| boxed_error("the pack carries no lat plane"))?;
    let lon = plane("lon")?.ok_or_else(|| boxed_error("the pack carries no lon plane"))?;
    let bcm = plane("bcm")?.ok_or_else(|| {
        boxed_error(
            "the pack carries no clear-sky mask (bcm plane): build it with `rw_goes bt --acm`, \
             because a block mean over pixels of unknown scene is not a clear-sky observation",
        )
    })?;
    let lattice = lattice_indices(&meta.x_scan_rad, &meta.y_scan_rad)?;
    let nx = meta.nx;
    let ny = meta.ny;
    let half = ABI_2KM_FULL_DISK_AXIS / 2;
    let sub_lon = meta.projection.longitude_of_projection_origin_deg;
    let radius = meta.projection.semi_major_axis_m;
    let height = meta.projection.perspective_point_height_m;

    let mut blocks: BTreeMap<(i64, i64), SuperobsBlock> = BTreeMap::new();
    let mut finite = 0usize;
    let mut clear = 0usize;
    let mut cloudy = 0usize;
    let mut unknown = 0usize;
    let mut zenith_rejected = 0usize;
    let mut off_earth = 0usize;
    for r in 0..ny {
        let gj = lattice.y_index[r];
        for c in 0..nx {
            let idx = r * nx + c;
            let gi = lattice.x_index[c];
            let key = ((gi + half).div_euclid(block as i64), (gj + half).div_euclid(block as i64));
            let plat = f64::from(lat[idx]);
            let plon = f64::from(lon[idx]);
            if !(plat.is_finite() && plon.is_finite()) {
                off_earth += 1;
                continue;
            }
            let entry = blocks.entry(key).or_default();
            entry.n_pixel += 1;
            let obs = bt[idx];
            if !obs.is_finite() {
                continue;
            }
            let zenith = satellite_zenith_deg(plat, plon, sub_lon, radius, height);
            if zenith > zenith_max {
                zenith_rejected += 1;
                continue;
            }
            finite += 1;
            entry.n_obs += 1;
            entry.sum_lat += plat;
            entry.sum_lon += plon;
            entry.sum_zenith += zenith;
            entry.all_sum += f64::from(obs);
            let mask = bcm[idx];
            if mask == 0.0 {
                clear += 1;
                entry.n_clear += 1;
                entry.clear_sum += f64::from(obs);
                entry.clear_sum_sq += f64::from(obs) * f64::from(obs);
            } else if mask == 1.0 {
                cloudy += 1;
                entry.n_cloudy += 1;
            } else {
                unknown += 1;
            }
        }
    }

    let mut csv = String::new();
    csv.push_str(
        "block_x,block_y,lat_mean_deg,lon_mean_deg,zenith_mean_deg,n_pixel,n_obs,n_pair,obs_mean_all_k,obs_clear,obs_cloudy,n_both_clear,obs_mean_both_clear_k,obs_std_clear_k\n",
    );
    let mut block_rows = 0usize;
    let mut clear_blocks = 0usize;
    for ((bx, by), acc) in &blocks {
        if acc.n_obs == 0 {
            continue;
        }
        block_rows += 1;
        let n = acc.n_obs as f64;
        let (clear_mean, clear_std) = if acc.n_clear > 0 {
            clear_blocks += 1;
            let m = acc.n_clear as f64;
            let mean = acc.clear_sum / m;
            let var = (acc.clear_sum_sq / m - mean * mean).max(0.0);
            (mean, var.sqrt())
        } else {
            (f64::NAN, f64::NAN)
        };
        csv.push_str(&format!(
            "{bx},{by},{:.4},{:.4},{:.2},{},{},{},{:.3},{},{},{},{},{}\n",
            acc.sum_lat / n,
            acc.sum_lon / n,
            acc.sum_zenith / n,
            acc.n_pixel,
            acc.n_obs,
            acc.n_obs,
            acc.all_sum / n,
            acc.n_clear,
            acc.n_cloudy,
            acc.n_clear,
            fmt_opt(clear_mean),
            fmt_opt(clear_std),
        ));
    }
    rw_store::atomic::atomic_write_bytes(out, csv.as_bytes())?;

    #[derive(Serialize)]
    struct SuperobsRecord<'a> {
        schema: &'static str,
        status: &'static str,
        pack: String,
        pack_schema: &'a str,
        pack_content_sha256: &'a str,
        satellite: &'a str,
        band: u8,
        scan_start: &'a str,
        scan_end: &'a str,
        provenance: &'a ProvenanceRow,
        clear_sky_mask: &'a Option<ClearSkyRow>,
        block_pixels: usize,
        block_table: String,
        block_rows: usize,
        blocks_with_clear_pixels: usize,
        zenith_max_deg: f64,
        population: &'static str,
        counts: BTreeMap<&'static str, usize>,
    }
    let mut counts = BTreeMap::new();
    counts.insert("pixels", nx * ny);
    counts.insert("off_earth", off_earth);
    counts.insert("finite", finite);
    counts.insert("clear", clear);
    counts.insert("cloudy", cloudy);
    counts.insert("mask_unknown", unknown);
    counts.insert("zenith_rejected", zenith_rejected);
    let record = SuperobsRecord {
        schema: SUPEROBS_SCHEMA,
        status: if clear_blocks > 0 { "READY" } else { "EMPTY" },
        pack: pack_path.to_string_lossy().to_string(),
        pack_schema: &meta.schema,
        pack_content_sha256: &meta.content_sha256,
        satellite: &meta.satellite,
        band: meta.band,
        scan_start: &meta.scan_start,
        scan_end: &meta.scan_end,
        provenance: &meta.provenance,
        clear_sky_mask: &meta.clear_sky_mask,
        block_pixels: block,
        block_table: out.to_string_lossy().to_string(),
        block_rows,
        blocks_with_clear_pixels: clear_blocks,
        zenith_max_deg: zenith_max,
        population: "the clear pixels (ACM BCM 0, DQF 0) among the finite pixels of each block, \
                     read by the observation side alone; n_pair and n_both_clear are the finite \
                     and clear counts so the colocation table's quality control applies unchanged",
        counts,
    };
    let text = format!("{}\n", serde_json::to_string_pretty(&record)?);
    if let Some(stats_path) = options.stats.as_deref() {
        rw_store::atomic::atomic_write_bytes(stats_path, text.as_bytes())?;
    }
    Ok(text)
}

fn fmt_opt(value: f64) -> String {
    if value.is_finite() { format!("{value:.3}") } else { String::new() }
}

// ---------------------------------------------------------------------------
// quicklook
// ---------------------------------------------------------------------------

pub fn cmd_quicklook(options: &RadianceOptions) -> Result<String, Box<dyn Error>> {
    let out = options
        .out
        .as_deref()
        .ok_or_else(|| boxed_error("--out is required (the PNG destination)"))?;
    let downsample = options.downsample.unwrap_or(1).max(1);

    // Source: a pack plane, or one or more simulated tiles composed on the
    // lattice.  Both end as (values, x_index of column 0, y_index of row 0,
    // nx, ny) on the global lattice.
    let (values, x0, y0, nx, ny, band, label): (Vec<f32>, i64, i64, usize, usize, u8, String) =
        if let Some(pack_path) = options.pack.as_deref() {
            let bytes = std::fs::read(pack_path)
                .map_err(|err| boxed_error(format!("cannot read {}: {err}", pack_path.display())))?;
            let (meta, payload) = decode_bt_pack(&bytes)?;
            let name = options.plane_name.clone().unwrap_or_else(|| "bt".to_string());
            let key = meta.planes.get(&name).ok_or_else(|| {
                boxed_error(format!(
                    "the pack carries no plane {name:?} (it has {})",
                    meta.plane_order.join(", ")
                ))
            })?;
            let entry = &meta.arrays[key];
            let slice = &payload[entry.offset..entry.offset + entry.bytes];
            let values: Vec<f32> = slice
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let lattice = lattice_indices(&meta.x_scan_rad, &meta.y_scan_rad)?;
            (
                values,
                lattice.x_index[0],
                lattice.y_index[0],
                meta.nx,
                meta.ny,
                options.band.unwrap_or(meta.band),
                format!("{} {} band {} {}", meta.satellite, name, meta.band, meta.scan_start),
            )
        } else if !options.sim.is_empty() {
            let tiles: Vec<SimTile> = options
                .sim
                .iter()
                .map(|path| load_sim_tile(path))
                .collect::<Result<_, _>>()?;
            let band = options.band.unwrap_or(tiles[0].sidecar.band);
            let x_min = tiles.iter().map(|t| t.sidecar.abi_fixed_grid_crop.x_index_min).min().unwrap();
            let x_max = tiles.iter().map(|t| t.sidecar.abi_fixed_grid_crop.x_index_max).max().unwrap();
            let y_min = tiles.iter().map(|t| t.sidecar.abi_fixed_grid_crop.y_index_min).min().unwrap();
            let y_max = tiles.iter().map(|t| t.sidecar.abi_fixed_grid_crop.y_index_max).max().unwrap();
            let nx = (x_max - x_min + 1) as usize;
            let ny = (y_max - y_min + 1) as usize;
            let mut values = vec![f32::NAN; nx * ny];
            for tile in &tiles {
                let tnx = tile.nx();
                for r in 0..tile.ny() {
                    let row = (y_max - tile.y_index(r)) as usize;
                    for c in 0..tnx {
                        let v = tile.values[r * tnx + c];
                        if !v.is_finite() {
                            continue;
                        }
                        let col = (tile.x_index(c) - x_min) as usize;
                        let slot = &mut values[row * nx + col];
                        if !slot.is_finite() {
                            *slot = v;
                        }
                    }
                }
            }
            (values, x_min, y_max, nx, ny, band, format!("simulated band {band}, {} tile(s)", tiles.len()))
        } else {
            return Err(boxed_error("quicklook needs --pack (a plane of a pack) or one or more --sim tiles"));
        };

    // Optional crop to a lattice index box [x0, x1, y0, y1] inclusive.
    let (values, x0, y0, nx, ny) = match options.bbox_index {
        None => (values, x0, y0, nx, ny),
        Some([bx0, bx1, by0, by1]) => {
            if bx1 < bx0 || by1 < by0 {
                return Err(boxed_error("--bbox-index expects X0,X1,Y0,Y1 with X0 <= X1 and Y0 <= Y1"));
            }
            let cnx = (bx1 - bx0 + 1) as usize;
            let cny = (by1 - by0 + 1) as usize;
            let mut cropped = vec![f32::NAN; cnx * cny];
            for r in 0..cny {
                let gj = by1 - r as i64;
                let src_row = y0 - gj;
                if src_row < 0 || src_row >= ny as i64 {
                    continue;
                }
                for c in 0..cnx {
                    let gi = bx0 + c as i64;
                    let src_col = gi - x0;
                    if src_col < 0 || src_col >= nx as i64 {
                        continue;
                    }
                    cropped[r * cnx + c] = values[src_row as usize * nx + src_col as usize];
                }
            }
            (cropped, bx0, by1, cnx, cny)
        }
    };

    // Block-mean downsample of finite pixels.
    let (values, nx, ny) = if downsample > 1 {
        let dnx = nx.div_ceil(downsample);
        let dny = ny.div_ceil(downsample);
        let mut sums = vec![0.0f64; dnx * dny];
        let mut counts = vec![0u32; dnx * dny];
        for r in 0..ny {
            for c in 0..nx {
                let v = values[r * nx + c];
                if v.is_finite() {
                    let slot = (r / downsample) * dnx + c / downsample;
                    sums[slot] += f64::from(v);
                    counts[slot] += 1;
                }
            }
        }
        let down: Vec<f32> = sums
            .iter()
            .zip(&counts)
            .map(|(s, &n)| if n == 0 { f32::NAN } else { (s / f64::from(n)) as f32 })
            .collect();
        (down, dnx, dny)
    } else {
        (values, nx, ny)
    };

    let image = rw_sat::export::render_band_image(&values, nx, ny, band);
    if let Some(parent) = out.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }
    image.save(out)?;
    let finite = values.iter().filter(|v| v.is_finite()).count();

    #[derive(Serialize)]
    struct QuicklookRecord {
        schema: &'static str,
        status: &'static str,
        path: String,
        label: String,
        band: u8,
        palette: &'static str,
        width: usize,
        height: usize,
        downsample: usize,
        lattice_x_index_min: i64,
        lattice_y_index_max: i64,
        finite_pixels: usize,
    }
    let record = QuicklookRecord {
        schema: QUICKLOOK_SCHEMA,
        status: "READY",
        path: out.to_string_lossy().to_string(),
        label,
        band,
        palette: "rw-sat per-band anchors (rw_sat::palette::band_anchors), NaN transparent",
        width: nx,
        height: ny,
        downsample,
        lattice_x_index_min: x0,
        lattice_y_index_max: y0,
        finite_pixels: finite,
    };
    Ok(format!("{}\n", serde_json::to_string_pretty(&record)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// GOES-R ABI band 13 constants as published in the L1b files (the
    /// order of magnitude of the real ones; the round trip holds for any
    /// positive set).
    fn band13() -> PlanckRow {
        PlanckRow {
            fk1: 10803.3,
            fk2: 1392.74,
            bc1: 0.07550,
            bc2: 0.99975,
        }
    }

    #[test]
    fn planck_round_trip_recovers_the_temperature() {
        let p = band13();
        for t in [180.0, 220.0, 260.0, 300.0, 330.0] {
            let l = p.radiance(t);
            assert!(l > 0.0);
            let back = p.brightness_temperature(l);
            assert!((back - t).abs() < 1.0e-9, "{t} -> {l} -> {back}");
        }
        assert!(p.brightness_temperature(0.0).is_nan());
        assert!(p.brightness_temperature(-1.0).is_nan());
        assert!(p.brightness_temperature(f64::NAN).is_nan());
    }

    #[test]
    fn a_warmer_scene_has_the_larger_radiance_and_the_larger_bt() {
        let p = band13();
        let cold = p.radiance(200.0);
        let warm = p.radiance(300.0);
        assert!(warm > cold);
        assert!(p.brightness_temperature(warm) > p.brightness_temperature(cold));
    }

    #[test]
    fn the_dqf_gate_is_fail_closed_and_counted() {
        let p = band13();
        let rad: Vec<f32> = vec![
            p.radiance(280.0) as f32,
            p.radiance(280.0) as f32,
            f32::NAN,
            0.0,
            p.radiance(250.0) as f32,
        ];
        let dqf = vec![0.0, 1.0, 0.0, 0.0, f32::NAN];
        let (bt, counts) = brightness_temperature_plane(&rad, &dqf, &p).unwrap();
        assert!((f64::from(bt[0]) - 280.0).abs() < 1.0e-3);
        assert!(bt[1].is_nan(), "DQF 1 is conditionally usable, gated");
        assert!(bt[2].is_nan());
        assert!(bt[3].is_nan());
        assert!(bt[4].is_nan(), "a fill DQF gates");
        assert_eq!(counts.total, 5);
        assert_eq!(counts.finite, 1);
        assert_eq!(counts.rad_missing, 1);
        assert_eq!(counts.rad_nonpositive, 1);
        assert_eq!(counts.dqf_missing, 1);
        assert_eq!(counts.dqf_bad, 2);
        assert!(brightness_temperature_plane(&rad, &dqf[..3], &p).is_err());
    }

    #[test]
    fn lattice_indices_follow_the_half_pitch_convention() {
        // The L1b axis: x[i] = i * 56e-6 - 0.151844 for a 5424 full disk,
        // which is (i - 2712 + 1/2) * 56e-6 exactly.
        let x: Vec<f64> = (0..8).map(|i| f64::from(i as f32 * 5.6e-5 - 0.151844)).collect();
        let y: Vec<f64> = (0..5).map(|j| f64::from(0.151844 - j as f32 * 5.6e-5)).collect();
        let l = lattice_indices(&x, &y).unwrap();
        assert_eq!(l.x_index, vec![-2712, -2711, -2710, -2709, -2708, -2707, -2706, -2705]);
        assert_eq!(l.y_index, vec![2711, 2710, 2709, 2708, 2707]);
        assert_eq!(l.column(-2710), Some(2));
        assert_eq!(l.column(-2713), None);
        assert_eq!(l.column(-2704), None);
        assert_eq!(l.row(2711), Some(0));
        assert_eq!(l.row(2707), Some(4));
        assert_eq!(l.row(2706), None);
        // Off-lattice axes are refused by name.
        let bad = vec![0.0, 5.6e-5, 1.4e-4];
        let err = lattice_indices(&bad, &y).unwrap_err().to_string();
        assert!(err.contains("x_scan_rad[0]"), "{err}");
        let skipping = vec![2.8e-5, 1.4e-4];
        let err = lattice_indices(&skipping, &y).unwrap_err().to_string();
        assert!(err.contains("advances one sample"), "{err}");
    }

    #[test]
    fn pair_moments_read_planted_bias_and_a_linear_correction_both_ways() {
        // Planted: sim = obs + 2 exactly -> bias +2, rmse 2, the linear fit
        // removes it whole.
        let mut m = PairMoments::new();
        for i in 0..50 {
            let obs = 250.0 + f64::from(i);
            m.push(obs, obs + 2.0);
        }
        assert!((m.bias() - 2.0).abs() < 1e-12);
        assert!((m.rmse() - 2.0).abs() < 1e-12);
        let (a, b, resid) = m.linear_correction().unwrap();
        assert!((b - 1.0).abs() < 1e-9);
        assert!((a + 2.0).abs() < 1e-9);
        assert!(resid.abs() < 1e-9);
        // The other direction: sim = obs - 3 -> bias -3.
        let mut m = PairMoments::new();
        for i in 0..50 {
            let obs = 250.0 + f64::from(i);
            m.push(obs, obs - 3.0);
        }
        assert!((m.bias() + 3.0).abs() < 1e-12);
        // Identical planes read exactly zero and no fit is invented from a
        // constant simulation.
        let mut m = PairMoments::new();
        for i in 0..10 {
            m.push(260.0 + f64::from(i), 260.0 + f64::from(i));
        }
        assert_eq!(m.bias(), 0.0);
        assert_eq!(m.rmse(), 0.0);
        let mut flat = PairMoments::new();
        for i in 0..10 {
            flat.push(260.0 + f64::from(i), 270.0);
        }
        assert!(flat.linear_correction().is_none());
    }

    #[test]
    fn the_histogram_bins_the_difference_at_one_kelvin() {
        let mut acc = ClassAccumulator::new();
        acc.push(250.0, 250.4); // +0.4 -> bin 60
        acc.push(250.0, 248.5); // -1.5 -> bin 58
        acc.push(250.0, 400.0); // beyond -> last bin
        let stats = acc.finish();
        assert_eq!(stats.diff_histogram_1k[60], 1);
        assert_eq!(stats.diff_histogram_1k[58], 1);
        assert_eq!(stats.diff_histogram_1k[119], 1);
        assert_eq!(stats.n, 3);
    }

    #[test]
    fn zenith_is_zero_at_the_sub_satellite_point_and_grows_toward_the_limb() {
        let z0 = satellite_zenith_deg(0.0, -75.0, -75.0, 6_378_137.0, 35_786_023.0);
        assert!(z0.abs() < 1e-9);
        let z1 = satellite_zenith_deg(40.0, -100.0, -75.0, 6_378_137.0, 35_786_023.0);
        assert!(z1 > 40.0 && z1 < 60.0, "{z1}");
        let limb = satellite_zenith_deg(0.0, -75.0 + 81.3, -75.0, 6_378_137.0, 35_786_023.0);
        assert!(limb > 89.0, "{limb}");
    }

    #[test]
    fn sim_tile_index_convention_is_row_zero_north() {
        let dir = std::env::temp_dir().join(format!("rw-goes-sim-tile-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let plane = dir.join("tile.f32");
        let values: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut bytes = Vec::new();
        for v in &values {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        std::fs::write(&plane, &bytes).unwrap();
        std::fs::write(dir.join("tile.f32.mask"), [0u8, 1, 255, 0, 0, 1]).unwrap();
        let sidecar = serde_json::json!({
            "schema": SIM_PLANE_SCHEMA,
            "shape": [2, 3],
            "band": 13,
            "mask_plane": "tile.f32.mask",
            "abi_fixed_grid_crop": {
                "sample_angle_urad": 56.0,
                "x_index_min": 10, "x_index_max": 12,
                "y_index_min": 100, "y_index_max": 101,
                "nx": 3, "ny": 2
            }
        });
        std::fs::write(dir.join("tile.f32.json"), sidecar.to_string()).unwrap();
        let tile = load_sim_tile(&plane).unwrap();
        assert_eq!(tile.x_index(0), 10);
        assert_eq!(tile.x_index(2), 12);
        assert_eq!(tile.y_index(0), 101, "row 0 is north = y_index_max");
        assert_eq!(tile.y_index(1), 100);
        assert_eq!(tile.mask.as_deref(), Some(&[0u8, 1, 255, 0, 0, 1][..]));
        // A sidecar whose shape and crop disagree is refused.
        let bad = serde_json::json!({
            "schema": SIM_PLANE_SCHEMA, "shape": [2, 3], "band": 13,
            "abi_fixed_grid_crop": {"x_index_min": 10, "x_index_max": 13, "y_index_min": 100, "y_index_max": 101, "nx": 4, "ny": 2}
        });
        std::fs::write(dir.join("tile.f32.json"), bad.to_string()).unwrap();
        assert!(load_sim_tile(&plane).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
