//! ODIM_H5 polar volume (`PVOL`) writer over [`recast_radar_hdf5::write`].
//!
//! [`write_odim_h5_volume`] writes any [`Volume`] whose sweeps are PPIs on a
//! uniform range grid as an ODIM_H5 polar volume (EUMETNET OPERA
//! WD_2008_03; `ODIM_H5/V2_3`, whose km `rstart` every reader agrees on; a
//! volume read from ODIM_H5 keeps its own version, v2.4 with `rstart` in
//! metres): `/what` (object, version, date, time, source),
//! `/where` (lat, lon, height), `/how` (with `software` and `sw_version`
//! naming this writer, by which the reader takes its km `rstart` as km at
//! any distance), and one `/datasetN` per sweep with
//! `what` (product `SCAN`, start and end date and time), `where` (elangle,
//! nbins, rstart, rscale in m, nrays, a1gate), `how` (per-ray
//! `startazA`/`stopazA`, `elangles`, `startazT`/`stopazT`, and `NI`,
//! `antspeed`, `pulsewidth`, `radconstH`/`radconstV`) and one `dataM` per
//! field: the plane (`CLASS` `IMAGE`, chunked, deflated) with `what`
//! (quantity, gain, offset, nodata, undetect), a `legend` for flag fields and
//! `qualityK` groups for quality fields.
//!
//! A field has the same `dataM` number in every dataset (numbered in order
//! of first appearance), so a dataset without it skips its number, as FMI's
//! volumes do: Py-ART's `read_odim_h5` takes the plane names of `dataset1`
//! and reads the same name in every dataset, so numbering each dataset's
//! planes from 1 would hand it one quantity's values under another's name.
//! [`OdimWriteOptions::every_quantity`] instead gives every dataset a plane
//! for every quantity of the volume, `nodata` where the sweep has none, for
//! readers that need the same planes in every dataset (Py-ART reads every
//! field then, LROSE Radx reads the file at all).
//!
//! What the volume holds is kept:
//!
//! - Planes keep their storage type and raw codes (`u8`, `i8`, `u16`, `i16`,
//!   `i32`, `f32`, `f64`) with `gain`/`offset` from the field's packing
//!   (NEXRAD `(raw - offset) / scale` becomes `gain = 1/scale`,
//!   `offset = -offset/scale`).
//! - ODIM has one `nodata` and one `undetect` code per plane. An integer
//!   plane whose fill code is also its undetect code (NEXRAD raw 0) gets a
//!   distinct `nodata`: the range-folded code when there is one (NEXRAD
//!   raw 1), else a code no gate uses. Range-folded gates, gates outside
//!   `valid_range`, absent rows and the padding of shorter fields are
//!   written as `nodata`. A plane without an undetect code gets an unused
//!   one when its type has one left. Float planes keep their values;
//!   `nodata`/`undetect` are written when the field has them.
//! - Rays are written in azimuth order (ODIM rows start at north), except
//!   for a volume read from ODIM_H5, whose rows are already in ODIM order.
//!   `a1gate` is the row of the first ray in time. `startazA`/`stopazA`
//!   are each ray's centre azimuth minus and plus the largest power of two
//!   not above half the ray spacing, so every reader's mean of the two is
//!   the centre exactly (a non-finite azimuth is written as it is);
//!   `elangles` (or `startelA`/`stopelA`) and `startazT`/`stopazT` hold
//!   the ray elevations and times (NaN for a ray without a time, which
//!   leaves the others theirs). A volume read
//!   from ODIM_H5 gets none of these derived: it writes back the arrays its
//!   ray coordinates were read from (kept in `Sweep::other`).
//! - Fields whose gates are coarser than the sweep's range, or start
//!   later, are repeated and padded onto the sweep's range (as the FM301
//!   view does).
//! - A volume read from ODIM_H5 is written back with every attribute the
//!   reader kept verbatim (`Volume::attrs.other`, `Sweep::other`,
//!   `FieldAttrs::other`) in the group it names: a `what`, `where` or `how`
//!   table attribute of ODIM_H5 in that group, `<group>.<name>` in that
//!   group (a `how` subgroup for any other group), `data.<name>` on the
//!   plane dataset, anything else in `how` (the reader names an attribute
//!   `<group>.<name>` whenever its bare name would place it elsewhere, see
//!   `tables`). Reading the output back gives the same volume (ray times to
//!   within a microsecond: they pass through seconds since 1970). The
//!   container can differ: values the reader lifts into typed slots are
//!   written from the slot (dataset `how` constants shared by every dataset
//!   at the root, `rpm` as `antspeed`, `beamwidth` as `beamwH`/`beamwV`, in
//!   the model's float32), an enumerated plane as its integer codes with a
//!   `legend`, and empty groups are not written (but a root `how`, which
//!   LROSE Radx needs to recognise the file).
//!
//! Refused with [`OdimWriteError::Unrepresentable`]: a volume without
//! sweeps, an RHI sweep (`rhi`, `manual_rhi`, `elevation_surveillance`), a
//! sweep without rays, gates or fields, and a sweep with explicit
//! (non-uniform) gate centres. Refused with [`OdimWriteError::TooLarge`],
//! before anything is allocated: a sweep of more than 16,384 gates per ray
//! or planes beyond the 1 GiB decode budget, which this crate's reader
//! would refuse.

use std::collections::{BTreeMap, HashSet};

use chrono::{DateTime, Datelike, Timelike, Utc};
use recast_radar_core::bounded_read::{MAX_DECODED_VOLUME_BYTES, MAX_GATES_PER_RADIAL};
use recast_radar_core::model::{
    ArrayBuf, AttrValue, Field, FieldData, FieldName, FloatCoding, GateMapping, IntCoding,
    LinearTransform, PackedInt, Polarization, Quantity, RangeCoord, Scalar, SourceFormat, Sweep,
    SweepMode, Volume,
};
use recast_radar_hdf5::write::{
    CharSet, Data, Layout, NewDataset, ObjectId, StringPadding, Value, WriteError, Writer,
};
use thiserror::Error;

use crate::tables::Level;

/// Options of [`write_odim_h5_volume`].
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct OdimWriteOptions {
    /// zlib level of the data planes (0-9); `None` stores them
    /// uncompressed. Default 6.
    pub deflate: Option<u32>,
    /// Name planes of other formats' fields by their ODIM quantity (`DBZ`
    /// becomes `DBZH`, `VEL` `VRADH`); `false` keeps every name. Default
    /// `true`.
    pub odim_quantities: bool,
    /// Give every dataset a plane for every quantity of the volume, all
    /// `nodata` where the sweep has no such field (module documentation).
    /// Default `false`: a dataset holds its sweep's fields only. Read back,
    /// such a plane is a field of the sweep whose every gate is missing.
    pub every_quantity: bool,
}

impl Default for OdimWriteOptions {
    fn default() -> Self {
        Self {
            deflate: Some(6),
            odim_quantities: true,
            every_quantity: false,
        }
    }
}

impl OdimWriteOptions {
    /// The same options with this plane compression.
    pub fn with_deflate(mut self, deflate: Option<u32>) -> Self {
        self.deflate = deflate;
        self
    }

    /// The same options, naming planes of other formats by their ODIM
    /// quantity or not.
    pub fn with_odim_quantities(mut self, odim_quantities: bool) -> Self {
        self.odim_quantities = odim_quantities;
        self
    }

    /// The same options, with or without a plane for every quantity in
    /// every dataset.
    pub fn with_every_quantity(mut self, every_quantity: bool) -> Self {
        self.every_quantity = every_quantity;
        self
    }
}

/// Errors from [`write_odim_h5_volume`].
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum OdimWriteError {
    /// Something ODIM_H5 cannot represent (see the module documentation).
    #[error("ODIM_H5 cannot represent {what}")]
    Unrepresentable {
        /// What, with the sweep it is in.
        what: String,
    },
    /// More than a reader of this crate accepts: a sweep over
    /// [`MAX_GATES_PER_RADIAL`] gates per ray, or planes beyond the decode
    /// budget ([`MAX_DECODED_VOLUME_BYTES`]). Refused before anything is
    /// allocated, so every file written reads back.
    #[error("too large for ODIM_H5 readers: {what}")]
    TooLarge {
        /// What, with the sweep it is in.
        what: String,
    },
    /// The HDF5 writer refused a structure.
    #[error(transparent)]
    Hdf5(#[from] WriteError),
}

/// Refuse what this crate's ODIM reader would not read back (see
/// [`OdimWriteError::TooLarge`]); planes are counted as the reader charges
/// them, rays times bins times the stored width, with the `nodata` planes of
/// [`OdimWriteOptions::every_quantity`].
fn check_size(volume: &Volume, options: &OdimWriteOptions) -> Result<(), OdimWriteError> {
    let width = |field: &Field| match &field.data {
        FieldData::U8 { .. } | FieldData::I8 { .. } => 1usize,
        FieldData::U16 { .. } | FieldData::I16 { .. } => 2,
        FieldData::I32 { .. } | FieldData::F32 { .. } => 4,
        FieldData::F64 { .. } => 8,
    };
    // The widest plane of each quantity, for `every_quantity`'s planes.
    let mut widest: BTreeMap<&str, usize> = BTreeMap::new();
    for field in volume.sweeps.iter().flat_map(|sweep| &sweep.fields) {
        let entry = widest.entry(field.name.as_str()).or_insert(0);
        *entry = (*entry).max(width(field));
    }
    let mut bytes = 0usize;
    for (index, sweep) in volume.sweeps.iter().enumerate() {
        let ngates = sweep.range.ngates();
        if ngates > MAX_GATES_PER_RADIAL {
            return Err(OdimWriteError::TooLarge {
                what: format!(
                    "sweep {index}: {ngates} gates per ray (at most {MAX_GATES_PER_RADIAL})"
                ),
            });
        }
        let cells = sweep.nrays().saturating_mul(ngates);
        let mut planes: usize = sweep.fields.iter().map(width).sum();
        if options.every_quantity {
            planes += widest
                .iter()
                .filter(|(name, _)| !sweep.fields.iter().any(|f| f.name.as_str() == **name))
                .map(|(_, width)| *width)
                .sum::<usize>();
        }
        bytes = bytes.saturating_add(cells.saturating_mul(planes));
    }
    if bytes > MAX_DECODED_VOLUME_BYTES {
        return Err(OdimWriteError::TooLarge {
            what: format!("{bytes} bytes of planes (at most {MAX_DECODED_VOLUME_BYTES})"),
        });
    }
    Ok(())
}

/// Refuse a field whose codes decode through a NEXRAD Level III level table
/// ([`LinearTransform::Levels`]): ODIM `gain` and `offset` state only a
/// linear coding. (The CfRadial 2 / FM301 writer writes such a field
/// decoded.)
fn check_linear(volume: &Volume) -> Result<(), OdimWriteError> {
    for (index, sweep) in volume.sweeps.iter().enumerate() {
        for field in &sweep.fields {
            if field
                .data
                .transform()
                .is_some_and(|transform| !transform.is_linear())
            {
                return Err(unrepresentable(format!(
                    "sweep {index} field {}: a level-table coding (NEXRAD Level III), which gain and offset cannot state",
                    field.name.as_str()
                )));
            }
        }
    }
    Ok(())
}

fn unrepresentable(what: impl Into<String>) -> OdimWriteError {
    OdimWriteError::Unrepresentable { what: what.into() }
}

/// The ODIM_H5 version written for volumes of other formats.
///
/// Version 2.3: its `where/rstart` is in km, which xradar, wradlib and
/// Py-ART all read as km (v2.4 states metres, which Py-ART reads as km).
const CONVENTIONS: &str = "ODIM_H5/V2_3";
/// Root `how/software` of the files this writer derives: the reader takes
/// their `where/rstart` as km at any distance (a pre-v2.4 `rstart` over
/// 20 is otherwise taken as metres).
pub(crate) const SOFTWARE: &str = "recast-radar-tools";
const VERSION: &str = "H5rad 2.3";
/// Largest error, in seconds, of a ray time read back as the mean of the
/// `startazT` and `stopazT` written for it; a ray whose ends would miss it
/// by more gets its own time as both.
const MAX_TIME_ERROR_S: f64 = 1e-6;

/// Where a kept attribute goes.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Target {
    What,
    Where,
    How,
    /// A subgroup of `how`.
    HowSub(String),
    /// The plane dataset.
    Data,
}

/// Where the name alone places a kept attribute: `Some((target, name,
/// explicit))` for a `<group>.<name>` spelling (`explicit`) or an ODIM_H5
/// table attribute, `None` for any other name.
fn place(name: &str, level: Level) -> Option<(Target, String, bool)> {
    if let Some((head, rest)) = name.split_once('.')
        && !rest.is_empty()
    {
        let target = match (head, level) {
            ("what", _) => Target::What,
            ("where", Level::Root | Level::Dataset) => Target::Where,
            ("how", _) => Target::How,
            ("data", Level::Plane) => Target::Data,
            _ => Target::HowSub(head.to_owned()),
        };
        return Some((target, rest.to_owned(), true));
    }
    match level.table_group(name)? {
        "what" => Some((Target::What, name.to_owned(), false)),
        _ => Some((Target::Where, name.to_owned(), false)),
    }
}

/// Order of the groups the ODIM reader collects kept attributes from:
/// `what`, then `where`, then `how` (its subgroups last).
fn rank(target: &Target) -> u8 {
    match target {
        Target::What => 0,
        Target::Where => 1,
        Target::How => 2,
        Target::HowSub(_) | Target::Data => 3,
    }
}

fn from_rank(rank: u8) -> Target {
    match rank {
        0 => Target::What,
        1 => Target::Where,
        _ => Target::How,
    }
}

/// The group and name of every kept attribute. The reader lists a level's
/// attributes group by group (`what`, `where`, `how`, then subgroups), so
/// the list order is evidence too: a table attribute listed after a later
/// group's attribute stays in that later group, and a name the tables do not
/// know goes to the group of the next table attribute (`how` when none
/// follows), never before the group of the one before it.
fn place_all(attrs: &[(Box<str>, AttrValue)], level: Level) -> Vec<(Target, String)> {
    let placed: Vec<Option<(Target, String, bool)>> =
        attrs.iter().map(|(name, _)| place(name, level)).collect();
    let how = rank(&Target::How);
    // The smallest table rank at or after each position.
    let mut next_rank = vec![how; placed.len() + 1];
    for index in (0..placed.len()).rev() {
        next_rank[index] = match &placed[index] {
            Some((target, _, false)) => rank(target).min(next_rank[index + 1]),
            _ => next_rank[index + 1],
        };
    }
    let mut current = 0u8;
    let mut out = Vec::with_capacity(attrs.len());
    for (index, ((name, _), placement)) in attrs.iter().zip(placed).enumerate() {
        let (target, name) = match placement {
            Some((target, name, true)) => (target, name),
            Some((target, name, false)) => {
                if rank(&target) >= current {
                    (target, name)
                } else {
                    (from_rank(current), name)
                }
            }
            None => (
                from_rank(next_rank[index + 1].max(current).min(how)),
                name.to_string(),
            ),
        };
        current = current.max(rank(&target).min(how));
        out.push((target, name));
    }
    out
}

/// Attributes to write, by target group, in order; a kept attribute
/// replaces a derived one of the same name.
#[derive(Default)]
struct Attrs(BTreeMap<Target, Vec<(String, Value)>>);

impl Attrs {
    fn derived(&mut self, target: Target, name: &str, value: Value) {
        let list = self.0.entry(target).or_default();
        if !list.iter().any(|(existing, _)| existing == name) {
            list.push((name.to_owned(), value));
        }
    }

    fn kept(&mut self, attrs: &[(Box<str>, AttrValue)], level: Level) {
        for ((target, name), (_, value)) in place_all(attrs, level).into_iter().zip(attrs) {
            if name.is_empty() || name.contains('\0') {
                continue;
            }
            let list = self.0.entry(target).or_default();
            list.retain(|(existing, _)| *existing != name);
            list.push((name, attr_value(value)));
        }
    }

    /// Write every target but `Data` below `group` (creating `what`,
    /// `where`, `how` and `how` subgroups as needed; `how` even when empty
    /// with `always_how`).
    fn write(
        mut self,
        writer: &mut Writer,
        group: ObjectId,
        always_how: bool,
    ) -> Result<Option<Vec<(String, Value)>>, WriteError> {
        if always_how
            && !self.0.iter().any(|(target, list)| {
                matches!(target, Target::How | Target::HowSub(_)) && !list.is_empty()
            })
        {
            self.0.remove(&Target::How);
            writer.add_group(group, "how")?;
        }
        let mut data = None;
        let mut how: Option<ObjectId> = None;
        for (target, list) in self.0 {
            if list.is_empty() {
                continue;
            }
            let object = match &target {
                Target::What => writer.add_group(group, "what")?,
                Target::Where => writer.add_group(group, "where")?,
                Target::How => {
                    let id = writer.add_group(group, "how")?;
                    how = Some(id);
                    id
                }
                Target::HowSub(sub) => {
                    let parent = match how {
                        Some(id) => id,
                        None => {
                            let id = writer.add_group(group, "how")?;
                            how = Some(id);
                            id
                        }
                    };
                    writer.add_group(parent, sub)?
                }
                Target::Data => {
                    data = Some(list);
                    continue;
                }
            };
            for (name, value) in list {
                writer.add_attribute(object, &name, value)?;
            }
        }
        Ok(data)
    }
}

fn text(value: &str) -> Value {
    Value::text_nul(value)
}

fn f64_value(value: f64) -> Value {
    Value::scalar(Data::F64(vec![value]))
}

fn i64_value(value: i64) -> Value {
    Value::scalar(Data::I64(vec![value]))
}

fn f64_array(values: Vec<f64>) -> Value {
    Value::vector(Data::F64(values))
}

/// Fixed-length NUL-terminated strings, as long as the longest plus one.
fn text_array(values: &[Box<str>]) -> Value {
    let size = values.iter().map(|value| value.len()).max().unwrap_or(0) + 1;
    let mut bytes = Vec::with_capacity(values.len() * size);
    for value in values {
        bytes.extend_from_slice(value.as_bytes());
        bytes.resize(bytes.len() + size - value.len(), 0);
    }
    Value::vector(Data::FixedStrings {
        bytes,
        size,
        padding: StringPadding::NullTerminate,
        charset: if values.iter().all(|value| value.is_ascii()) {
            CharSet::Ascii
        } else {
            CharSet::Utf8
        },
    })
}

/// A model attribute as an ODIM attribute: text as a NUL-terminated string,
/// numbers in their type, a bool as h5py's boolean enumeration.
fn attr_value(value: &AttrValue) -> Value {
    match value {
        AttrValue::Text(text) => Value::text_nul(text),
        AttrValue::Bool(value) => Value::scalar(Data::Bools(vec![*value])),
        AttrValue::Scalar(scalar) => Value::scalar(match *scalar {
            Scalar::I8(v) => Data::I8(vec![v]),
            Scalar::U8(v) => Data::U8(vec![v]),
            Scalar::I16(v) => Data::I16(vec![v]),
            Scalar::U16(v) => Data::U16(vec![v]),
            Scalar::I32(v) => Data::I32(vec![v]),
            Scalar::U32(v) => Data::U32(vec![v]),
            Scalar::I64(v) => Data::I64(vec![v]),
            Scalar::U64(v) => Data::U64(vec![v]),
            Scalar::F32(v) => Data::F32(vec![v]),
            Scalar::F64(v) => Data::F64(vec![v]),
        }),
        AttrValue::Array(array) => match array {
            ArrayBuf::I8(v) => Value::vector(Data::I8(v.clone())),
            ArrayBuf::U8(v) => Value::vector(Data::U8(v.clone())),
            ArrayBuf::I16(v) => Value::vector(Data::I16(v.clone())),
            ArrayBuf::U16(v) => Value::vector(Data::U16(v.clone())),
            ArrayBuf::I32(v) => Value::vector(Data::I32(v.clone())),
            ArrayBuf::U32(v) => Value::vector(Data::U32(v.clone())),
            ArrayBuf::I64(v) => Value::vector(Data::I64(v.clone())),
            ArrayBuf::F32(v) => Value::vector(Data::F32(v.clone())),
            ArrayBuf::F64(v) => Value::vector(Data::F64(v.clone())),
            ArrayBuf::Text(v) => text_array(v),
        },
    }
}

fn date_time(time: DateTime<Utc>) -> (String, String) {
    (
        format!("{:04}{:02}{:02}", time.year(), time.month(), time.day()),
        format!("{:02}{:02}{:02}", time.hour(), time.minute(), time.second()),
    )
}

/// The ODIM quantity of a field of another format.
fn odim_quantity(field: &Field) -> Option<&'static str> {
    let vertical = matches!(
        field.polarization,
        Polarization::V | Polarization::CopolarV | Polarization::CrosspolarV
    );
    Some(match field.quantity {
        Quantity::Reflectivity => {
            if vertical {
                "DBZV"
            } else {
                "DBZH"
            }
        }
        Quantity::TotalPower => {
            if vertical {
                "TV"
            } else {
                "TH"
            }
        }
        Quantity::RadialVelocity => {
            if vertical {
                "VRADV"
            } else {
                "VRADH"
            }
        }
        Quantity::DealiasedRadialVelocity => "VRADDH",
        Quantity::SpectrumWidth => {
            if vertical {
                "WRADV"
            } else {
                "WRADH"
            }
        }
        Quantity::DifferentialReflectivity => "ZDR",
        Quantity::LinearDepolarizationRatio => "LDR",
        Quantity::DifferentialPhase => "PHIDP",
        Quantity::SpecificDifferentialPhase => "KDP",
        Quantity::CorrelationCoefficient => "RHOHV",
        Quantity::SignalToNoiseRatio => {
            if vertical {
                "SNRV"
            } else {
                "SNRH"
            }
        }
        Quantity::SignalQualityIndex => {
            if vertical {
                "SQIV"
            } else {
                "SQIH"
            }
        }
        Quantity::ClutterCorrection => {
            if vertical {
                "CCORV"
            } else {
                "CCORH"
            }
        }
        Quantity::PrecipitationRate => "RATE",
        Quantity::EchoClassification => "CLASS",
        _ => return None,
    })
}

/// Largest power of two not above `value` (positive, finite).
fn power_of_two_below(value: f64) -> f64 {
    if !(value.is_finite() && value > 0.0) {
        return 0.25;
    }
    2f64.powi(value.log2().floor() as i32)
}

/// Write `volume` as an ODIM_H5 polar volume. See the module documentation.
pub fn write_odim_h5_volume(
    volume: &Volume,
    options: &OdimWriteOptions,
) -> Result<Vec<u8>, OdimWriteError> {
    if volume.sweeps.is_empty() {
        return Err(unrepresentable("a volume without sweeps"));
    }
    check_size(volume, options)?;
    check_linear(volume)?;
    let odim = volume.provenance.source_format == SourceFormat::OdimH5;
    let mut writer = Writer::new();
    let root = writer.root();
    // An ODIM volume keeps its own `Conventions` (or its lack of one).
    let conventions = if odim {
        volume.provenance.source_conventions.as_deref()
    } else {
        Some(CONVENTIONS)
    };
    if let Some(conventions) = conventions {
        writer.add_attribute(root, "Conventions", text(conventions))?;
    }

    let mut attrs = Attrs::default();
    let (date, time) = date_time(volume.time_reference);
    if !odim {
        attrs.derived(Target::What, "object", text("PVOL"));
    }
    let version = odim
        .then_some(volume.provenance.source_version.as_deref())
        .flatten()
        .unwrap_or(VERSION);
    attrs.derived(Target::What, "version", text(version));
    attrs.derived(Target::What, "date", text(&date));
    attrs.derived(Target::What, "time", text(&time));
    attrs.derived(Target::What, "source", text(&source_string(volume, odim)));
    // ODIM_H5 requires the location: a volume without one (NEXRAD Message
    // 1) gets NaN.
    let location = &volume.location;
    for (name, value) in [
        ("lon", location.longitude_deg),
        ("lat", location.latitude_deg),
        ("height", location.altitude_m),
    ] {
        if value.is_some() || !odim {
            attrs.derived(Target::Where, name, f64_value(value.unwrap_or(f64::NAN)));
        }
    }
    let parameters = &volume.radar_parameters;
    for (name, value) in [
        ("beamwH", parameters.beam_width_h_deg),
        ("beamwV", parameters.beam_width_v_deg),
        ("antgainH", parameters.antenna_gain_h_db),
        ("antgainV", parameters.antenna_gain_v_db),
        (
            "RXbandwidth",
            parameters.receiver_bandwidth_hz.map(|hz| hz / 1e6),
        ),
    ] {
        if let Some(value) = value {
            attrs.derived(Target::How, name, f64_value(f64::from(value)));
        }
    }
    if !odim {
        if let Some(frequency) = parameters.frequency_hz.first().filter(|f| **f > 0.0) {
            attrs.derived(
                Target::How,
                "wavelength",
                f64_value(299_792_458.0 / frequency * 100.0),
            );
        }
        attrs.derived(Target::How, "software", text(SOFTWARE));
        attrs.derived(Target::How, "sw_version", text(env!("CARGO_PKG_VERSION")));
        let global = &volume.attrs;
        for (name, value) in [
            ("title", &global.title),
            ("institution", &global.institution),
            ("references", &global.references),
            ("history", &global.history),
            ("comment", &global.comment),
        ] {
            if let Some(value) = value {
                attrs.derived(Target::How, name, text(value));
            }
        }
    }
    attrs.kept(&volume.attrs.other, Level::Root);
    // A root `how` group even when empty: LROSE Radx tells ODIM_H5 by it.
    attrs.write(&mut writer, root, true)?;

    // One `dataM` number per data plane field, in order of first
    // appearance.
    let mut planes: Vec<&Field> = Vec::new();
    for sweep in &volume.sweeps {
        let classes = PlaneClasses::of(sweep);
        for field in &sweep.fields {
            if !classes.placed.contains(field.name.as_str())
                && !planes.iter().any(|plane| plane.name == field.name)
            {
                planes.push(field);
            }
        }
    }
    for (index, sweep) in volume.sweeps.iter().enumerate() {
        let metres = conventions.is_some_and(|c| c.trim() == "ODIM_H5/V2_4");
        write_sweep(
            &mut writer,
            volume,
            sweep,
            index,
            odim,
            metres,
            options,
            &planes,
        )?;
    }
    Ok(writer.finish()?)
}

/// `/what/source`: the source's own for an ODIM volume (empty when its file
/// had none: the reader then names the volume "ODIM", which is no node
/// identifier), else built from the identity (`NOD:` the instrument name,
/// `WMO:`, `WIGOS:`, `PLC:` the site name).
fn source_string(volume: &Volume, odim: bool) -> String {
    if odim {
        return volume.attrs.source.clone().unwrap_or_default();
    }
    let mut pairs = Vec::new();
    if !volume.attrs.instrument_name.is_empty() {
        pairs.push(format!("NOD:{}", volume.attrs.instrument_name));
    }
    if let Some(wmo) = &volume.attrs.wmo.id {
        pairs.push(format!("WMO:{wmo}"));
    }
    if let Some(wigos) = &volume.attrs.wmo.wsi {
        pairs.push(format!("WIGOS:{wigos}"));
    }
    if let Some(site) = &volume.attrs.site_name {
        pairs.push(format!("PLC:{site}"));
    }
    pairs.join(",")
}

/// The value every ray has, when there is one.
fn constant<T: Copy + PartialEq>(values: &[T]) -> Option<T> {
    let first = *values.first()?;
    values.iter().all(|value| *value == first).then_some(first)
}

#[allow(clippy::too_many_arguments)]
fn write_sweep(
    writer: &mut Writer,
    volume: &Volume,
    sweep: &Sweep,
    index: usize,
    odim: bool,
    rstart_in_metres: bool,
    options: &OdimWriteOptions,
    planes: &[&Field],
) -> Result<(), OdimWriteError> {
    let label = format!("sweep {index}");
    if matches!(
        sweep.sweep_mode,
        SweepMode::Rhi | SweepMode::ManualRhi | SweepMode::ElevationSurveillance
    ) {
        return Err(unrepresentable(format!(
            "{label}: {} sweeps (a PVOL holds PPIs)",
            sweep.sweep_mode.as_str()
        )));
    }
    let nrays = sweep.nrays();
    let (first_center_m, spacing_m, ngates) = match sweep.range {
        RangeCoord::Uniform {
            first_center_m,
            spacing_m,
            ngates,
        } => (first_center_m, spacing_m, ngates as usize),
        RangeCoord::Explicit { .. } => {
            return Err(unrepresentable(format!(
                "{label}: explicit (non-uniform) gate centres"
            )));
        }
    };
    if nrays == 0 || ngates == 0 || spacing_m.is_nan() || spacing_m <= 0.0 {
        return Err(unrepresentable(format!(
            "{label}: a sweep without rays or gates"
        )));
    }
    if sweep.fields.is_empty() {
        return Err(unrepresentable(format!("{label}: a sweep without fields")));
    }

    // Row order: azimuth order, except for ODIM sources (already in ODIM
    // row order).
    let mut order: Vec<usize> = (0..nrays).collect();
    if !odim {
        let azimuth = &sweep.rays.azimuth_deg;
        order.sort_by(|a, b| azimuth[*a].total_cmp(&azimuth[*b]));
    }
    let group = writer.add_group(writer.root(), &format!("dataset{}", index + 1))?;
    let mut attrs = Attrs::default();

    // Ray times, seconds since 1970, in row order. A ray without a time
    // (NaN) leaves the others theirs: the dataset's start and end, `a1gate`
    // and the time step come from the rays that have one.
    let reference = volume.time_reference.timestamp() as f64;
    let times: Vec<f64> = order
        .iter()
        .map(|row| reference + sweep.rays.time_s[*row])
        .collect();
    let known = || times.iter().copied().filter(|time| time.is_finite());
    if !odim {
        attrs.derived(Target::What, "product", text("SCAN"));
        let lo = known().fold(f64::INFINITY, f64::min);
        let hi = known().fold(f64::NEG_INFINITY, f64::max);
        let (start, end) = if lo <= hi {
            (lo.floor(), hi.ceil())
        } else {
            (reference, reference)
        };
        for (prefix, seconds) in [("start", start), ("end", end)] {
            let instant =
                DateTime::<Utc>::from_timestamp(seconds as i64, 0).unwrap_or(volume.time_reference);
            let (date, time) = date_time(instant);
            attrs.derived(Target::What, &format!("{prefix}date"), text(&date));
            attrs.derived(Target::What, &format!("{prefix}time"), text(&time));
        }
    }
    attrs.derived(
        Target::Where,
        "elangle",
        f64_value(f64::from(sweep.fixed_angle_deg)),
    );
    attrs.derived(Target::Where, "nbins", i64_value(ngates as i64));
    let first_edge_m = first_center_m - spacing_m / 2.0;
    attrs.derived(
        Target::Where,
        "rstart",
        f64_value(if rstart_in_metres {
            first_edge_m
        } else {
            first_edge_m / 1000.0
        }),
    );
    attrs.derived(Target::Where, "rscale", f64_value(spacing_m));
    attrs.derived(Target::Where, "nrays", i64_value(nrays as i64));
    if !odim {
        let first = (0..nrays)
            .filter(|row| times[*row].is_finite())
            .min_by(|a, b| times[*a].total_cmp(&times[*b]))
            .unwrap_or(0);
        attrs.derived(Target::Where, "a1gate", i64_value(first as i64));
    }

    // Per-ray angles and times. A volume read from ODIM_H5 keeps its own
    // `how` arrays (in `Sweep::other`, written back below) and gets none
    // derived: its ray coordinates read back from the same attributes.
    if !odim {
        let spacing = sweep
            .rays_angle_resolution_deg
            .map(f64::from)
            .filter(|res| *res > 0.0)
            .unwrap_or(360.0 / nrays as f64);
        let half = power_of_two_below(spacing / 2.0);
        let azimuths: Vec<f64> = order
            .iter()
            .map(|row| f64::from(sweep.rays.azimuth_deg[*row]))
            .collect();
        // A non-finite azimuth is written as it is (wrapped, an infinity
        // would become NaN), so it reads back unchanged.
        let edge = |az: f64, offset: f64| {
            if az.is_finite() {
                (az + offset).rem_euclid(360.0)
            } else {
                az
            }
        };
        attrs.derived(
            Target::How,
            "startazA",
            f64_array(azimuths.iter().map(|az| edge(*az, -half)).collect()),
        );
        attrs.derived(
            Target::How,
            "stopazA",
            f64_array(azimuths.iter().map(|az| edge(*az, half)).collect()),
        );
        let elevations: Vec<f64> = order
            .iter()
            .map(|row| f64::from(sweep.rays.elevation_deg[*row]))
            .collect();
        let kept_elangles = place_all(&sweep.other, Level::Dataset)
            .iter()
            .any(|(target, name)| *target == Target::How && name == "elangles");
        if kept_elangles {
            // The source's own `elangles` stays; the pair takes precedence.
            attrs.derived(Target::How, "startelA", f64_array(elevations.clone()));
            attrs.derived(Target::How, "stopelA", f64_array(elevations));
        } else {
            attrs.derived(Target::How, "elangles", f64_array(elevations));
        }
        // Every ray's time, NaN where the volume has none: a sweep with one
        // unknown ray time keeps the others to the sub-second (without the
        // arrays, readers spread the rays over whole-second start and end
        // times).
        let mut steps: Vec<f64> = {
            let mut sorted: Vec<f64> = known().collect();
            sorted.sort_by(f64::total_cmp);
            sorted
                .windows(2)
                .map(|pair| pair[1] - pair[0])
                .filter(|step| step.is_finite() && *step > 0.0)
                .collect()
        };
        steps.sort_by(f64::total_cmp);
        let half_step = steps.get(steps.len() / 2).map_or(0.0, |step| step / 2.0);
        // A time within half a step of the float range keeps itself as both
        // ends (the step would overflow it), and so does a time the ends
        // would not give back to the microsecond: readers take the ray time
        // as the ends' mean, and a step far larger than the time (a corrupt
        // ray time elsewhere in the sweep) rounds the ends.
        let time_edge = |time: f64, offset: f64| {
            let (start, stop) = (time - offset.abs(), time + offset.abs());
            let kept = start.is_finite()
                && stop.is_finite()
                && (start / 2.0 + stop / 2.0 - time).abs() <= MAX_TIME_ERROR_S;
            match (kept, offset < 0.0) {
                (false, _) => time,
                (true, true) => start,
                (true, false) => stop,
            }
        };
        attrs.derived(
            Target::How,
            "startazT",
            f64_array(
                times
                    .iter()
                    .map(|time| time_edge(*time, -half_step))
                    .collect(),
            ),
        );
        attrs.derived(
            Target::How,
            "stopazT",
            f64_array(
                times
                    .iter()
                    .map(|time| time_edge(*time, half_step))
                    .collect(),
            ),
        );
    }
    let vars = &sweep.ray_vars;
    let nyquist = vars.nyquist_velocity_mps.as_deref().and_then(|values| {
        let finite: Vec<f32> = values.iter().copied().filter(|v| v.is_finite()).collect();
        constant(&finite).or_else(|| {
            // A sweep's Nyquist velocity varies only by rounding: the
            // median is its value.
            let mut sorted = finite;
            sorted.sort_by(f32::total_cmp);
            sorted.get(sorted.len() / 2).copied()
        })
    });
    if let Some(nyquist) = nyquist.filter(|value| *value > 0.0) {
        attrs.derived(Target::How, "NI", f64_value(f64::from(nyquist)));
    }
    if let Some(rate) = sweep.target_scan_rate_deg_per_s.filter(|rate| *rate > 0.0) {
        // `antspeed` (v2.4), or `rpm` when the volume keeps an `antspeed`
        // of its own (the reader took the rate from `rpm` then).
        let kept_antspeed = place_all(&sweep.other, Level::Dataset)
            .iter()
            .any(|(target, name)| *target == Target::How && name == "antspeed");
        if kept_antspeed {
            attrs.derived(Target::How, "rpm", f64_value(f64::from(rate) / 6.0));
        } else {
            attrs.derived(Target::How, "antspeed", f64_value(f64::from(rate)));
        }
    }
    let pulse_width = vars
        .pulse_width_s
        .as_deref()
        .and_then(constant)
        .or(volume.radar_parameters.pulse_width_s)
        .filter(|width| width.is_finite() && *width > 0.0);
    if let Some(width) = pulse_width {
        attrs.derived(Target::How, "pulsewidth", f64_value(f64::from(width) * 1e6));
    }
    let calibration = vars
        .calib_index
        .as_deref()
        .and_then(constant)
        .and_then(|index| {
            volume
                .radar_calibration
                .iter()
                .find(|entry| entry.calib_index == Some(index))
        });
    if let Some(entry) = calibration {
        for (name, value) in [
            ("radconstH", entry.radar_constant_h),
            ("radconstV", entry.radar_constant_v),
        ] {
            if let Some(value) = value {
                attrs.derived(Target::How, name, f64_value(f64::from(value)));
            }
        }
        let parameters = &volume.radar_parameters;
        for (name, value, site) in [
            (
                "antgainH",
                entry.antenna_gain_h_db,
                parameters.antenna_gain_h_db,
            ),
            (
                "antgainV",
                entry.antenna_gain_v_db,
                parameters.antenna_gain_v_db,
            ),
        ] {
            if let Some(value) = value
                && Some(value) != site
            {
                attrs.derived(Target::How, name, f64_value(f64::from(value)));
            }
        }
    }
    attrs.kept(&sweep.other, Level::Dataset);
    attrs.write(writer, group, false)?;

    write_planes(
        writer, sweep, group, &order, ngates, odim, options, &label, planes,
    )
}

/// A plane's ODIM coding: `gain`, `offset`, `nodata`, `undetect` (codes as
/// numbers).
struct PlaneCoding {
    gain: f64,
    offset: f64,
    nodata: Option<f64>,
    undetect: Option<f64>,
}

/// A sweep's fields sorted into data planes, plane quality groups and
/// dataset quality groups.
struct PlaneClasses<'a> {
    /// Fields written as quality groups.
    placed: HashSet<&'a str>,
    /// `<plane>_qualityK`: the plane, `K`, the field.
    plane_quality: Vec<(&'a str, u32, &'a Field)>,
    /// `qualityK` of the dataset.
    dataset_quality: Vec<(u32, &'a Field)>,
}

impl<'a> PlaneClasses<'a> {
    /// Plane quality fields `<plane>_qualityK` and dataset quality fields
    /// `qualityK`, as the ODIM reader names them.
    fn of(sweep: &'a Sweep) -> Self {
        let is_quality = |field: &Field| field.attrs.is_quality_field == Some(true);
        let quality_number = |name: &str, prefix: &str| -> Option<u32> {
            name.strip_prefix(prefix)?
                .strip_prefix("quality")?
                .parse::<u32>()
                .ok()
        };
        let all_data: Vec<&str> = sweep
            .fields
            .iter()
            .filter(|f| !is_quality(f))
            .map(|f| f.name.as_str())
            .collect();
        let mut classes = Self {
            placed: HashSet::new(),
            plane_quality: Vec::new(),
            dataset_quality: Vec::new(),
        };
        for field in sweep.fields.iter().filter(|f| is_quality(f)) {
            let qualified: Vec<&str> = field
                .attrs
                .qualified_variables
                .iter()
                .map(FieldName::as_str)
                .collect();
            if let [plane] = qualified.as_slice()
                && let Some(plane) = all_data.iter().find(|name| *name == plane)
                && let Some(k) = quality_number(field.name.as_str(), &format!("{plane}_"))
            {
                classes.plane_quality.push((plane, k, field));
                classes.placed.insert(field.name.as_str());
            } else if qualified == all_data
                && let Some(k) = quality_number(field.name.as_str(), "")
            {
                classes.dataset_quality.push((k, field));
                classes.placed.insert(field.name.as_str());
            }
        }
        classes
    }
}

/// A plane for `template`'s quantity in a sweep without it: `nrays` absent
/// rows of `ngates`, written as `nodata`.
fn absent_like(template: &Field, nrays: usize, ngates: usize) -> Field {
    let len = nrays * ngates;
    let data = match &template.data {
        FieldData::U8 { coding, .. } => FieldData::U8 {
            values: vec![0; len],
            coding: *coding,
        },
        FieldData::I8 { coding, .. } => FieldData::I8 {
            values: vec![0; len],
            coding: *coding,
        },
        FieldData::U16 { coding, .. } => FieldData::U16 {
            values: vec![0; len],
            coding: *coding,
        },
        FieldData::I16 { coding, .. } => FieldData::I16 {
            values: vec![0; len],
            coding: *coding,
        },
        FieldData::I32 { coding, .. } => FieldData::I32 {
            values: vec![0; len],
            coding: *coding,
        },
        FieldData::F32 { coding, .. } => FieldData::F32 {
            values: vec![f32::NAN; len],
            coding: *coding,
        },
        FieldData::F64 { coding, .. } => FieldData::F64 {
            values: vec![f64::NAN; len],
            coding: *coding,
        },
    };
    let mut field = Field::new(
        template.name.clone(),
        GateMapping::IDENTITY,
        ngates as u32,
        data,
    );
    field.quantity = template.quantity;
    field.polarization = template.polarization;
    field.attrs = template.attrs.clone();
    field.absent_rows = (0..nrays as u32).collect();
    field
}

#[allow(clippy::too_many_arguments)]
fn write_planes(
    writer: &mut Writer,
    sweep: &Sweep,
    group: ObjectId,
    order: &[usize],
    ngates: usize,
    odim: bool,
    options: &OdimWriteOptions,
    label: &str,
    planes: &[&Field],
) -> Result<(), OdimWriteError> {
    let PlaneClasses {
        placed,
        plane_quality,
        mut dataset_quality,
    } = PlaneClasses::of(sweep);
    let mut used_quantities: HashSet<String> = HashSet::new();
    let absent: Vec<Field> = if options.every_quantity {
        planes
            .iter()
            .filter(|plane| !sweep.fields.iter().any(|field| field.name == plane.name))
            .map(|plane| absent_like(plane, sweep.nrays(), ngates))
            .collect()
    } else {
        Vec::new()
    };
    let mut data_planes: Vec<(usize, &Field)> = sweep
        .fields
        .iter()
        .chain(&absent)
        .filter(|field| !placed.contains(field.name.as_str()))
        .filter_map(|field| {
            let number = planes.iter().position(|plane| plane.name == field.name)?;
            Some((number + 1, field))
        })
        .collect();
    data_planes.sort_by_key(|(number, _)| *number);
    for (plane_index, field) in data_planes {
        let quantity = plane_quantity(field, odim, options, &used_quantities);
        used_quantities.insert(quantity.clone());
        let plane = writer.add_group(group, &format!("data{plane_index}"))?;
        write_plane(
            writer,
            field,
            plane,
            order,
            ngates,
            odim,
            options,
            Some(&quantity),
            label,
        )?;
        let mut qualities: Vec<&(&str, u32, &Field)> = plane_quality
            .iter()
            .filter(|(name, _, _)| *name == field.name.as_str())
            .collect();
        qualities.sort_by_key(|(_, k, _)| *k);
        for (_, k, quality) in qualities {
            let quality_group = writer.add_group(plane, &format!("quality{k}"))?;
            write_plane(
                writer,
                quality,
                quality_group,
                order,
                ngates,
                odim,
                options,
                None,
                label,
            )?;
        }
    }
    dataset_quality.sort_by_key(|(k, _)| *k);
    for (k, quality) in dataset_quality {
        let quality_group = writer.add_group(group, &format!("quality{k}"))?;
        write_plane(
            writer,
            quality,
            quality_group,
            order,
            ngates,
            odim,
            options,
            None,
            label,
        )?;
    }
    Ok(())
}

/// The `what/quantity` of a data plane.
fn plane_quantity(
    field: &Field,
    odim: bool,
    options: &OdimWriteOptions,
    used: &HashSet<String>,
) -> String {
    let name = field.name.as_str();
    if odim {
        // A second plane of a quantity was named `<quantity>_dataM`.
        if let Some((quantity, plane)) = name.rsplit_once("_data")
            && plane.parse::<u32>().is_ok()
            && used.contains(quantity)
        {
            return quantity.to_owned();
        }
        return name.to_owned();
    }
    if options.odim_quantities
        && let Some(quantity) = odim_quantity(field)
        && !used.contains(quantity)
    {
        return quantity.to_owned();
    }
    name.to_owned()
}

#[allow(clippy::too_many_arguments)]
fn write_plane(
    writer: &mut Writer,
    field: &Field,
    group: ObjectId,
    order: &[usize],
    ngates: usize,
    odim: bool,
    options: &OdimWriteOptions,
    quantity: Option<&str>,
    label: &str,
) -> Result<(), OdimWriteError> {
    let nrays = order.len();
    let (data, coding) = plane_data(field, order, ngates, odim).ok_or_else(|| {
        unrepresentable(format!(
            "{label}: field {} has no free nodata code",
            field.name
        ))
    })?;
    let rows = chunk_rows(nrays, ngates, element_bytes(&data));
    let layout = match options.deflate {
        Some(level) => Layout::chunked(vec![rows as u64, ngates as u64], Some(level.min(9))),
        None => Layout::Contiguous,
    };
    let mut attrs = Attrs::default();
    if let Some(quantity) = quantity {
        attrs.derived(Target::What, "quantity", text(quantity));
    }
    attrs.derived(Target::What, "gain", f64_value(coding.gain));
    attrs.derived(Target::What, "offset", f64_value(coding.offset));
    if let Some(nodata) = coding.nodata {
        attrs.derived(Target::What, "nodata", f64_value(nodata));
    }
    if let Some(undetect) = coding.undetect {
        attrs.derived(Target::What, "undetect", f64_value(undetect));
    }
    if !odim {
        attrs.derived(Target::Data, "CLASS", text("IMAGE"));
        attrs.derived(Target::Data, "IMAGE_VERSION", text("1.2"));
    }
    attrs.kept(&field.attrs.other, Level::Plane);
    let dataset_attrs = attrs.write(writer, group, false)?;
    let dataset = writer.add_dataset(
        group,
        "data",
        NewDataset::new(
            Value::array(data, vec![nrays as u64, ngates as u64]),
            layout,
        ),
    )?;
    for (name, value) in dataset_attrs.unwrap_or_default() {
        writer.add_attribute(dataset, &name, value)?;
    }
    let flags = &field.attrs;
    if !flags.flag_values.is_empty() && flags.flag_values.len() == flags.flag_meanings.len() {
        let keys: Vec<Box<str>> = flags.flag_meanings.clone();
        let values: Vec<Box<str>> = flags
            .flag_values
            .iter()
            .map(|code| code.to_string().into_boxed_str())
            .collect();
        let column = |texts: &[Box<str>]| text_array(texts).data;
        writer.add_dataset(
            group,
            "legend",
            NewDataset::new(
                Value::vector(Data::Compound(vec![
                    ("key".to_owned(), column(&keys)),
                    ("value".to_owned(), column(&values)),
                ])),
                Layout::Contiguous,
            ),
        )?;
    }
    Ok(())
}

fn element_bytes(data: &Data) -> usize {
    match data {
        Data::I8(_) | Data::U8(_) => 1,
        Data::I16(_) | Data::U16(_) => 2,
        Data::I32(_) | Data::F32(_) => 4,
        _ => 8,
    }
}

/// Rows per chunk: the whole plane up to 4 MiB, else about 1 MiB chunks.
fn chunk_rows(nrays: usize, ngates: usize, element: usize) -> usize {
    let row = ngates.max(1) * element;
    if nrays * row <= 4 << 20 {
        nrays.max(1)
    } else {
        ((1 << 20) / row).clamp(1, nrays.max(1))
    }
}

/// The plane values in row order on the sweep's range, and their coding.
/// `None` when an integer plane has no free code for `nodata`.
fn plane_data(
    field: &Field,
    order: &[usize],
    ngates: usize,
    odim: bool,
) -> Option<(Data, PlaneCoding)> {
    macro_rules! int_plane {
        ($values:expr, $coding:expr, $variant:ident, $t:ty) => {{
            let (values, coding) = int_plane::<$t>($values, $coding, field, order, ngates, odim)?;
            (Data::$variant(values), coding)
        }};
    }
    Some(match &field.data {
        FieldData::U8 { values, coding } => int_plane!(values, coding, U8, u8),
        FieldData::U16 { values, coding } => int_plane!(values, coding, U16, u16),
        FieldData::I8 { values, coding } => int_plane!(values, coding, I8, i8),
        FieldData::I16 { values, coding } => int_plane!(values, coding, I16, i16),
        FieldData::I32 { values, coding } => int_plane!(values, coding, I32, i32),
        FieldData::F32 { values, coding } => {
            let fill = coding.fill_code();
            let out = map_rows(values, fill, field, order, ngates, |v| v);
            (Data::F32(out), float_coding(coding, f64::from))
        }
        FieldData::F64 { values, coding } => {
            let fill = coding.fill_code();
            let out = map_rows(values, fill, field, order, ngates, |v| v);
            (Data::F64(out), float_coding(coding, |v| v))
        }
    })
}

/// ODIM `gain` and `offset` of a linear transform ([`check_linear`] refuses
/// the others before any plane is written).
fn gain_offset(transform: LinearTransform) -> (f64, f64) {
    (
        transform.scale_factor().unwrap_or(1.0),
        transform.add_offset().unwrap_or(0.0),
    )
}

fn float_coding<T: Copy>(coding: &FloatCoding<T>, widen: impl Fn(T) -> f64) -> PlaneCoding {
    let (gain, offset) = coding.transform.map_or((1.0, 0.0), gain_offset);
    PlaneCoding {
        gain,
        offset,
        nodata: coding.fill_value.map(&widen),
        undetect: coding.undetect.map(&widen),
    }
}

/// Integer codes a plane can take.
trait Code: PackedInt + Ord + std::hash::Hash {
    /// Every code of the type, from the top (`MAX` first) down, for the
    /// free-code search (bounded to 65,536 candidates).
    fn candidates() -> Box<dyn Iterator<Item = Self>>;
    fn widen(self) -> f64;
}

macro_rules! code {
    ($t:ty) => {
        impl Code for $t {
            fn candidates() -> Box<dyn Iterator<Item = Self>> {
                Box::new(
                    [<$t>::MAX, <$t>::MIN]
                        .into_iter()
                        .chain((<$t>::MIN..<$t>::MAX).rev().take(65_536)),
                )
            }
            fn widen(self) -> f64 {
                f64::from(self)
            }
        }
    };
}

code!(u8);
code!(i8);
code!(u16);
code!(i16);
code!(i32);

fn int_plane<T: Code>(
    values: &[T],
    coding: &IntCoding<T>,
    field: &Field,
    order: &[usize],
    ngates: usize,
    odim: bool,
) -> Option<(Vec<T>, PlaneCoding)> {
    let (gain, offset) = gain_offset(coding.transform);
    if odim {
        // ODIM sources: the codes as read.
        let fill = coding.fill_code();
        let out = map_rows(values, fill, field, order, ngates, |v| v);
        return Some((
            out,
            PlaneCoding {
                gain,
                offset,
                nodata: coding.fill_value.map(Code::widen),
                undetect: coding.undetect.map(Code::widen),
            },
        ));
    }
    let used: HashSet<T> = values.iter().copied().collect();
    let free = |taken: &[Option<T>]| {
        T::candidates().find(|code| !used.contains(code) && !taken.contains(&Some(*code)))
    };
    let nodata = match coding.fill_value {
        Some(fill) if coding.undetect != Some(fill) => fill,
        _ => match coding.range_folded {
            Some(folded) if coding.undetect != Some(folded) => folded,
            _ => free(&[coding.undetect])?,
        },
    };
    // A plane without an undetect code gets an unused one, when a code is
    // left (a plane using every other code of its type gets none).
    let undetect = match coding.undetect {
        Some(undetect) => Some(undetect),
        None => free(&[Some(nodata)]),
    };
    let remap = |raw: T| -> T {
        if coding.undetect == Some(raw) {
            raw
        } else if coding.fill_value == Some(raw)
            || coding.range_folded == Some(raw)
            || matches!(coding.valid_range, Some([lo, hi]) if raw < lo || raw > hi)
        {
            nodata
        } else {
            raw
        }
    };
    let out = map_rows(values, nodata, field, order, ngates, remap);
    Some((
        out,
        PlaneCoding {
            gain,
            offset,
            nodata: Some(nodata.widen()),
            undetect: undetect.map(Code::widen),
        },
    ))
}

/// `values` (the field's `[nrays × native gates]`) in row `order` on the
/// sweep's `ngates` range gates: native gates repeated by the gate stride
/// from the gate start, every other gate and every absent row `fill`.
fn map_rows<T: Copy>(
    values: &[T],
    fill: T,
    field: &Field,
    order: &[usize],
    ngates: usize,
    remap: impl Fn(T) -> T,
) -> Vec<T> {
    let native = field.ngates as usize;
    let stride = field.gates.stride.max(1) as usize;
    let start = field.gates.start as usize;
    let mut out = vec![fill; order.len() * ngates];
    for (out_row, row) in order.iter().enumerate() {
        if field.is_absent(*row) {
            continue;
        }
        let Some(source) = values.get(row * native..(row + 1) * native) else {
            continue;
        };
        let dest = &mut out[out_row * ngates..(out_row + 1) * ngates];
        for (gate, value) in source.iter().enumerate() {
            let first = start + gate * stride;
            if first >= ngates {
                break;
            }
            let value = remap(*value);
            for slot in &mut dest[first..(first + stride).min(ngates)] {
                *slot = value;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kept_attributes_go_back_to_their_group() {
        let names = |names: &[&str]| -> Vec<(Box<str>, AttrValue)> {
            names
                .iter()
                .map(|name| ((*name).into(), AttrValue::Text("x".into())))
                .collect()
        };
        // Table attributes, unknown names up to the next table attribute
        // (`how` when none follows), subgroups by their prefix.
        assert_eq!(
            place_all(
                &names(&["object", "wavelength", "towerheight", "radar_system.name"]),
                Level::Root
            ),
            vec![
                (Target::What, "object".into()),
                (Target::How, "wavelength".into()),
                (Target::How, "towerheight".into()),
                (Target::HowSub("radar_system".into()), "name".into()),
            ]
        );
        assert_eq!(
            place_all(
                &names(&["product", "endtime", "range", "a1gate", "NEZH"]),
                Level::Dataset
            ),
            vec![
                (Target::What, "product".into()),
                (Target::What, "endtime".into()),
                (Target::Where, "range".into()),
                (Target::Where, "a1gate".into()),
                (Target::How, "NEZH".into()),
            ]
        );
        assert_eq!(
            place_all(
                &names(&["quantity", "task", "data.CLASS", "how.gain"]),
                Level::Plane
            ),
            vec![
                (Target::What, "quantity".into()),
                (Target::How, "task".into()),
                (Target::Data, "CLASS".into()),
                (Target::How, "gain".into()),
            ]
        );
    }

    #[test]
    fn power_of_two_half_widths() {
        assert_eq!(power_of_two_below(0.25), 0.25);
        assert_eq!(power_of_two_below(0.45), 0.25);
        assert_eq!(power_of_two_below(0.5), 0.5);
        assert_eq!(power_of_two_below(f64::NAN), 0.25);
    }
}
