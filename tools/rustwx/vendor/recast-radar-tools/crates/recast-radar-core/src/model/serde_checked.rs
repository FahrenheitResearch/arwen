//! Checked deserialization of [`Field`], [`Sweep`] and [`Volume`].
//!
//! The model's invariants (`docs/design/fm301-model.md` section 3) hold for a
//! volume a decoder built, because every decoder ends with [`Volume::seal`].
//! Deserialized data has not been through a decoder, so each of these types
//! deserializes into a private mirror with the same fields and serde shape
//! and converts through the checks below. Serialization is the derived one.
//!
//! - A [`Field`] holds `nrays × ngates` values and its absent rows ascend
//!   below `nrays`. Its methods then allocate no more than the document holds
//!   (before, `u32::MAX × u32::MAX` made [`Field::to_physical`] panic on the
//!   capacity).
//! - A [`Sweep`]'s fields have one row per ray, and the sweep passes
//!   [`Sweep::seal`]. With a row for every ray, `seal` appends no absent rows,
//!   so a document cannot make it fill `rays × ngates` values it does not
//!   hold.
//! - A [`Sweep`]'s range coordinate, and the range gates each field covers
//!   (`gates.start + gates.stride × ngates`, which `seal` grows a uniform
//!   range to), stay within [`MAX_GATES_PER_RADIAL`], the ceiling every
//!   decoder applies to a radial. A uniform range is three numbers, so
//!   without this a document of a few bytes could claim `u32::MAX` gates, and
//!   the FM301 view would allocate a `range` array of that length and map
//!   every field onto it (`nrays × u32::MAX` values on
//!   [`Values::materialize`](crate::fm301::Values::materialize)). With it,
//!   a field materializes to at most `nrays × 16,384` values.
//! - Every [`ExtraVariable`], of a sweep or of the volume, names one
//!   dimension per `shape` entry, and the product of `shape` is the number of
//!   values it holds. The FM301 view declares the variable's dimensions from
//!   `shape` and reorders a per-ray variable's rows by it, so without this a
//!   few bytes could declare a `u32::MAX` dimension over four values and make
//!   the view reserve `nrays × u32::MAX` values.
//! - A [`Volume`] passes [`Volume::seal`]: `sweeps[i].sweep_number == i`.
//!
//! Each mirror lists every field of its type, and the conversions destructure
//! the mirror and build the type with every field named, so a field added to
//! the type and not to its mirror fails to compile.

use chrono::{DateTime, Utc};
use serde::Deserialize;

use crate::bounded_read::MAX_GATES_PER_RADIAL;

use super::field::{Field, FieldAttrs, FieldData, GateMapping};
use super::names::{FieldName, Polarization, Quantity};
use super::sweep::{
    FollowMode, Monitoring, PlatformTrack, PolarizationMode, PrtMode, RangeCoord, RayVariables,
    Rays, Sweep, SweepError, SweepMode,
};
use super::values::{AttrValue, ExtraVariable, VariableAttrs};
use super::volume::{
    GeoreferencingCorrection, GlobalAttrs, InstrumentType, Location, PlatformType, PrimaryAxis,
    Provenance, RadarCalibration, RadarParameters, ScanStrategy, SimulationProvenance,
    TimeCoverage, Volume,
};

/// The serde shape of [`Field`], before its checks.
#[derive(Deserialize)]
#[serde(rename = "Field")]
pub(crate) struct FieldRepr {
    name: FieldName,
    quantity: Quantity,
    polarization: Polarization,
    attrs: FieldAttrs,
    nrays: u32,
    ngates: u32,
    gates: GateMapping,
    data: FieldData,
    absent_rows: Vec<u32>,
}

impl TryFrom<FieldRepr> for Field {
    type Error = SweepError;

    fn try_from(repr: FieldRepr) -> Result<Self, SweepError> {
        let FieldRepr {
            name,
            quantity,
            polarization,
            attrs,
            nrays,
            ngates,
            gates,
            data,
            absent_rows,
        } = repr;
        let field = Field {
            name,
            quantity,
            polarization,
            attrs,
            nrays,
            ngates,
            gates,
            data,
            absent_rows,
        };
        check_field_shape(&field)?;
        Ok(field)
    }
}

/// `data` holds `nrays × ngates` values; `absent_rows` ascend strictly and
/// name rows below `nrays`.
fn check_field_shape(field: &Field) -> Result<(), SweepError> {
    let expected = (field.nrays as usize).checked_mul(field.ngates as usize);
    if expected != Some(field.data.len()) {
        return Err(SweepError::FieldLength {
            field: field.name.as_str().to_owned(),
            len: field.data.len(),
            expected: (field.nrays as usize).saturating_mul(field.ngates as usize),
        });
    }
    if !field.absent_rows.windows(2).all(|pair| pair[0] < pair[1])
        || field
            .absent_rows
            .last()
            .is_some_and(|last| *last >= field.nrays)
    {
        return Err(SweepError::AbsentRows {
            field: field.name.as_str().to_owned(),
        });
    }
    Ok(())
}

/// The serde shape of [`Sweep`], before its checks.
#[derive(Deserialize)]
#[serde(rename = "Sweep")]
pub(crate) struct SweepRepr {
    sweep_number: u32,
    sweep_mode: SweepMode,
    follow_mode: Option<FollowMode>,
    prt_mode: Option<PrtMode>,
    polarization_mode: Option<PolarizationMode>,
    polarization_sequence: Option<Vec<Box<str>>>,
    fixed_angle_deg: f32,
    target_scan_rate_deg_per_s: Option<f32>,
    rays_are_indexed: Option<bool>,
    rays_angle_resolution_deg: Option<f32>,
    qc_procedures: Option<String>,
    rays: Rays,
    range: RangeCoord,
    ray_vars: RayVariables,
    monitoring: Option<Box<Monitoring>>,
    platform_track: Option<Box<PlatformTrack>>,
    extra_vars: Vec<ExtraVariable>,
    other: Vec<(Box<str>, AttrValue)>,
    fields: Vec<Field>,
    elevation_number: Option<u16>,
    complete: bool,
}

impl TryFrom<SweepRepr> for Sweep {
    type Error = SweepError;

    fn try_from(repr: SweepRepr) -> Result<Self, SweepError> {
        let SweepRepr {
            sweep_number,
            sweep_mode,
            follow_mode,
            prt_mode,
            polarization_mode,
            polarization_sequence,
            fixed_angle_deg,
            target_scan_rate_deg_per_s,
            rays_are_indexed,
            rays_angle_resolution_deg,
            qc_procedures,
            rays,
            range,
            ray_vars,
            monitoring,
            platform_track,
            extra_vars,
            other,
            fields,
            elevation_number,
            complete,
        } = repr;
        let mut sweep = Sweep {
            sweep_number,
            sweep_mode,
            follow_mode,
            prt_mode,
            polarization_mode,
            polarization_sequence,
            fixed_angle_deg,
            target_scan_rate_deg_per_s,
            rays_are_indexed,
            rays_angle_resolution_deg,
            qc_procedures,
            rays,
            range,
            ray_vars,
            monitoring,
            platform_track,
            extra_vars,
            other,
            fields,
            elevation_number,
            complete,
        };
        // Every field was checked on its own; a row for every ray keeps
        // `seal` from appending absent rows.
        let nrays = sweep.rays.azimuth_deg.len();
        for field in &sweep.fields {
            if field.nrays as usize != nrays {
                return Err(SweepError::RayLength {
                    what: format!("field {} rows", field.name.as_str()),
                    len: field.nrays as usize,
                    nrays,
                });
            }
        }
        check_range_gates(&sweep)?;
        check_extra_vars(&sweep.extra_vars)?;
        sweep.seal()?;
        Ok(sweep)
    }
}

/// The range coordinate and every field's extent on it (which `seal` grows
/// a uniform range to) are within [`MAX_GATES_PER_RADIAL`].
fn check_range_gates(sweep: &Sweep) -> Result<(), SweepError> {
    let limit = MAX_GATES_PER_RADIAL;
    let over = |what: String, gates: u64| {
        if gates > limit as u64 {
            Err(SweepError::RangeGates { what, gates, limit })
        } else {
            Ok(())
        }
    };
    over("range".to_owned(), sweep.range.ngates() as u64)?;
    for field in &sweep.fields {
        over(
            format!("field {}", field.name.as_str()),
            field.gates.end(field.ngates).unwrap_or(u64::MAX),
        )?;
    }
    Ok(())
}

/// Each variable has one dimension name per `shape` entry, and `shape`
/// multiplies out to the number of values it holds (`[]` to one value).
fn check_extra_vars(extras: &[ExtraVariable]) -> Result<(), SweepError> {
    for extra in extras {
        let product = extra
            .shape
            .iter()
            .try_fold(1usize, |product, len| product.checked_mul(*len as usize));
        if extra.dims.len() != extra.shape.len() || product != Some(extra.values.len()) {
            return Err(SweepError::ExtraShape {
                name: extra.name.to_string(),
                dims: extra.dims.iter().map(ToString::to_string).collect(),
                shape: extra.shape.clone(),
                len: extra.values.len(),
            });
        }
    }
    Ok(())
}

/// The serde shape of [`Volume`], before its checks.
#[derive(Deserialize)]
#[serde(rename = "Volume")]
pub(crate) struct VolumeRepr {
    attrs: GlobalAttrs,
    volume_number: Option<i32>,
    time_reference: DateTime<Utc>,
    time_coverage: Option<TimeCoverage>,
    location: Location,
    platform_type: PlatformType,
    instrument_type: InstrumentType,
    primary_axis: Option<PrimaryAxis>,
    status_str: Option<String>,
    scan: ScanStrategy,
    radar_parameters: RadarParameters,
    radar_calibration: Vec<RadarCalibration>,
    georeferencing_correction: Option<Box<GeoreferencingCorrection>>,
    extra_vars: Vec<ExtraVariable>,
    #[serde(default)]
    variable_attrs: Vec<VariableAttrs>,
    provenance: Provenance,
    simulation: Option<Box<SimulationProvenance>>,
    sweeps: Vec<Sweep>,
}

impl TryFrom<VolumeRepr> for Volume {
    type Error = SweepError;

    fn try_from(repr: VolumeRepr) -> Result<Self, SweepError> {
        let VolumeRepr {
            attrs,
            volume_number,
            time_reference,
            time_coverage,
            location,
            platform_type,
            instrument_type,
            primary_axis,
            status_str,
            scan,
            radar_parameters,
            radar_calibration,
            georeferencing_correction,
            extra_vars,
            variable_attrs,
            provenance,
            simulation,
            sweeps,
        } = repr;
        let mut volume = Volume {
            attrs,
            volume_number,
            time_reference,
            time_coverage,
            location,
            platform_type,
            instrument_type,
            primary_axis,
            status_str,
            scan,
            radar_parameters,
            radar_calibration,
            georeferencing_correction,
            extra_vars,
            variable_attrs,
            provenance,
            simulation,
            sweeps,
        };
        check_extra_vars(&volume.extra_vars)?;
        // Each sweep was sealed as it was read; this checks sweep numbers.
        volume.seal()?;
        Ok(volume)
    }
}
