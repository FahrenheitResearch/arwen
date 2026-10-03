//! CfRadial 1.4 writer: any [`Volume`] as classic netCDF (CDF-2).
//!
//! Layout (M. Dixon and W.-C. Lee, "CfRadial Data File Format", v1.4):
//! dimensions `time` (every ray of every sweep, sweeps in the order of
//! their first ray time, as xradar's reader requires, rays in storage
//! order, or in time order in `n_points` storage ([`time_order`]); a sweep
//! moved from its place keeps its index as `sweep_number`), `range`,
//! `n_points` (`n_gates_vary` storage), `sweep`, `string_length`, and
//! `frequency`, `r_calib` and the dimensions of kept variables when needed;
//! global attributes; `volume_number`, `platform_type`, `instrument_type`,
//! `primary_axis`, `status_str`, `time_coverage_start`/`_end`, the site
//! location, the radar parameters, the sweep table (`sweep_number`,
//! `sweep_mode`, `fixed_angle`, `sweep_start_ray_index`, ...), the
//! calibration table (`r_calib_*`), `range`, `time`, `azimuth`,
//! `elevation`, the per-ray instrument variables, `ray_n_gates`,
//! `ray_start_index`, `ray_start_range` and `ray_gate_spacing` when the
//! layout needs them, the fields and the variables the volume keeps
//! verbatim.
//!
//! - **Gate geometry** ([`RangeLayout`]). The default puts every sweep on
//!   one `range(range)` coordinate when their gates fall on one grid: the
//!   finest gate spacing of the volume, from the earliest gate edge. A
//!   sweep whose gates are a whole multiple of it (NEXRAD's 1 km surveillance
//!   gates beside 250 m Doppler gates) has each value repeated; a sweep that
//!   starts later is padded with the fill code. Every CfRadial reader reads
//!   that. Sweeps whose gates do not fall on one grid (500 m and 250 m
//!   gates, every first gate centred at 0 m, as LROSE Radx reads FMI's
//!   volumes) keep
//!   their own geometry in a `range(sweep, range)` coordinate (CfRadial 1.4
//!   section 4.4) with `meters_to_center_of_first_gate` and
//!   `meters_between_gates` per sweep, and `ray_start_range` /
//!   `ray_gate_spacing` per ray. `RangeLayout::PerSweep` does that for any
//!   volume whose sweeps differ, `RangeLayout::PerRay` states the geometry
//!   per ray only (LROSE Radx's reading), `RangeLayout::Common` refuses
//!   instead. Explicit gate centres must be the same in every sweep.
//! - **Gates per ray.** When the sweeps' rows differ in length the fields
//!   are stored over `n_points` (`n_gates_vary = "true"`, section 2.3.1):
//!   each ray holds its sweep's gates, located by `ray_start_index` and
//!   `ray_n_gates`, rather than the longest sweep's padded with fill. The
//!   rays of each sweep are then stored in time order: xradar lays `n_points`
//!   rows out by ray time and misplaces rays stored out of time order.
//!   Otherwise the fields are `(time, range)`.
//! - **One variable per field.** Fields of one sweep share its rows: a
//!   field on a coarser part of the sweep's range (NEXRAD Message 1
//!   reflectivity) has each value repeated. A field keeps its storage type
//!   and raw codes when every sweep has the same coding; the classic format has no
//!   unsigned types, so `u8` and `u16` are widened to `short` and `int`
//!   (codes unchanged), or with [`Cfradial1Options::unsigned_attribute`]
//!   stored as `byte` and `short` with `_Unsigned = "true"`, which LROSE
//!   Radx does not apply; `scale_factor`/`add_offset` (in the width the source
//!   wrote them), `_FillValue`, `_Undetect`, `valid_range`, and
//!   `flag_values`/`flag_meanings` with the range-folded code, in the packed
//!   type. A field whose coding differs between sweeps (ODIM gains per
//!   sweep) is written as `float` physical values with `_FillValue =
//!   -9999`: its undetect and range-folded gates become missing. A sweep
//!   without the field, an absent ray and padding hold the fill code; a
//!   field without one gets an unused code as `_FillValue`.
//! - **Metadata.** Global attributes, the attributes of every variable
//!   (`Volume::variable_attrs`), and the variables a CfRadial volume keeps
//!   verbatim are written back as read, so a CfRadial 1 file read and
//!   written again reads back unchanged. Other formats' volume attributes
//!   become global attributes (text arrays joined with ", "), their sweep
//!   attributes variables: text `(sweep, string_length)`, numbers `double`
//!   `(sweep)`, arrays with a value per ray `(time)` (ODIM's per-ray `how`
//!   arrays), other arrays `(sweep, <name>_len)`; names netCDF cannot store
//!   are made valid (`/` and control characters become `_`).
//! - Attribute and variable types outside the classic set widen: `ubyte`
//!   to `short`, `ushort` to `int`, `uint`, `int64` and `uint64` to `int`
//!   when they fit, else `double`.

use std::collections::HashSet;

use chrono::{DateTime, Datelike, Timelike, Utc};
use recast_radar_core::bounded_read::{MAX_DECODED_VOLUME_BYTES, MAX_GATES_PER_RADIAL};
use recast_radar_core::model::{
    ArrayBuf, AttrValue, ExtraVariable, Field, FieldData, FloatWidth, Gate, GateMapping, IntCoding,
    LinearTransform, PackedInt, RangeCoord, Scalar, SourceFormat, Sweep, Volume,
};

use super::CfWriteError;
use super::netcdf3::{Nc3Values, Nc3Variable, Nc3Writer, sanitize_name};

/// Options of [`write_cfradial1`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct Cfradial1Options {
    /// Store `u8` and `u16` fields as `byte` and `short` with
    /// `_Unsigned = "true"` (the netCDF Users Guide convention; netCDF-Java,
    /// netCDF4-python, xarray, Py-ART and xradar apply it). Default `false`:
    /// they are widened to `short` and `int`, raw codes unchanged, which
    /// every CfRadial reader reads (LROSE Radx ignores `_Unsigned`).
    pub unsigned_attribute: bool,
    /// How sweeps with different gate geometries share the file (module
    /// documentation, "Gate geometry").
    pub range_layout: RangeLayout,
}

impl Cfradial1Options {
    /// The same options, storing unsigned fields with `_Unsigned` or
    /// widened.
    pub fn with_unsigned_attribute(mut self, unsigned_attribute: bool) -> Self {
        self.unsigned_attribute = unsigned_attribute;
        self
    }

    /// The same options with another [`RangeLayout`].
    pub fn with_range_layout(mut self, range_layout: RangeLayout) -> Self {
        self.range_layout = range_layout;
        self
    }
}

/// Where the gates of sweeps with different gate geometries go in a
/// CfRadial 1 file. CfRadial 1.4 (section 4.4) has one `range(range)`
/// coordinate for a volume whose geometry is constant and `range(sweep,
/// range)` for one whose geometry varies from sweep to sweep. Readers
/// differ: xradar reads both; Py-ART reads only `range(range)`; LROSE Radx
/// refuses `range(sweep, range)` and instead takes each ray's geometry from
/// `ray_start_range` and `ray_gate_spacing`, which Py-ART and xradar ignore.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum RangeLayout {
    /// [`RangeLayout::Common`] whenever every sweep's gates fall on the grid
    /// of the finest gate spacing (every reader reads that), otherwise
    /// [`RangeLayout::PerSweep`].
    #[default]
    Auto,
    /// Always one `range(range)`: a sweep whose gates are a whole multiple
    /// of the finest spacing has each value repeated, a sweep that starts
    /// later is padded; sweeps whose gate edges do not fall on one grid are
    /// [`CfWriteError::Unrepresentable`].
    Common,
    /// Each sweep keeps its own gate geometry: `range(sweep, range)` when the
    /// sweeps' first gate or spacing differ (CfRadial 1.4), with
    /// `ray_start_range` and `ray_gate_spacing` per ray. xradar and this
    /// crate read it; Py-ART and LROSE Radx refuse the two-dimensional
    /// `range`.
    PerSweep,
    /// Each sweep keeps its own gate geometry, stated per ray in
    /// `ray_start_range` and `ray_gate_spacing` beside a `range(range)` of
    /// the longest sweep: how LROSE Radx reads a volume whose geometry
    /// varies (this crate reads it too). Py-ART and xradar take every ray's
    /// gates from `range(range)`, so they misplace the gates of the other
    /// sweeps. A sweep whose first centre or spacing a float cannot state
    /// as a start above -9999 m and a positive spacing (readers take those
    /// as fill) is [`CfWriteError::Unrepresentable`].
    PerRay,
}

const CONVENTIONS: &str = "CF/Radial instrument_parameters radar_parameters radar_calibration";
const VERSION: &str = "1.4";
/// Fill of `float` values the writer produces.
const FLOAT_FILL: f32 = -9999.0;
/// Fill of `int` sweep and ray variables.
const INT_FILL: i32 = -9999;

fn text(value: &str) -> Nc3Values {
    Nc3Values::text(value)
}

fn time_string(time: DateTime<Utc>) -> String {
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        time.year(),
        time.month(),
        time.day(),
        time.hour(),
        time.minute(),
        time.second()
    )
}

fn time_string_ms(time: DateTime<Utc>) -> String {
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}.{:03}Z",
        time.year(),
        time.month(),
        time.day(),
        time.hour(),
        time.minute(),
        time.second(),
        time.timestamp_subsec_millis()
    )
}

fn int_or_double(values: impl Iterator<Item = f64> + Clone) -> Nc3Values {
    if values
        .clone()
        .all(|v| v.fract() == 0.0 && (f64::from(i32::MIN)..=f64::from(i32::MAX)).contains(&v))
    {
        Nc3Values::Int(values.map(|v| v as i32).collect())
    } else {
        Nc3Values::Double(values.collect())
    }
}

/// A model attribute as a classic attribute (module docs for widening).
pub(super) fn attr_values(value: &AttrValue) -> Nc3Values {
    match value {
        AttrValue::Text(value) => text(value),
        AttrValue::Bool(value) => text(if *value { "true" } else { "false" }),
        AttrValue::Scalar(scalar) => match *scalar {
            Scalar::I8(v) => Nc3Values::Byte(vec![v]),
            Scalar::U8(v) => Nc3Values::Short(vec![i16::from(v)]),
            Scalar::I16(v) => Nc3Values::Short(vec![v]),
            Scalar::U16(v) => Nc3Values::Int(vec![i32::from(v)]),
            Scalar::I32(v) => Nc3Values::Int(vec![v]),
            Scalar::U32(v) => int_or_double(std::iter::once(f64::from(v))),
            Scalar::I64(v) => int_or_double(std::iter::once(v as f64)),
            Scalar::U64(v) => int_or_double(std::iter::once(v as f64)),
            Scalar::F32(v) => Nc3Values::Float(vec![v]),
            Scalar::F64(v) => Nc3Values::Double(vec![v]),
        },
        AttrValue::Array(array) => array_values(array),
    }
}

/// A model attribute as text: text as it is, a bool as "true"/"false",
/// numbers in Rust's shortest round-trip form, arrays joined with ", ".
fn attr_text(value: &AttrValue) -> String {
    match value {
        AttrValue::Text(text) => text.to_string(),
        AttrValue::Bool(value) => value.to_string(),
        AttrValue::Scalar(scalar) => scalar.as_f64().to_string(),
        AttrValue::Array(ArrayBuf::Text(texts)) => texts.join(", "),
        AttrValue::Array(array) => (0..array.len())
            .map(|index| array.get_f64(index).unwrap_or(f64::NAN).to_string())
            .collect::<Vec<_>>()
            .join(", "),
    }
}

fn array_values(array: &ArrayBuf) -> Nc3Values {
    match array {
        ArrayBuf::I8(v) => Nc3Values::Byte(v.clone()),
        ArrayBuf::U8(v) => Nc3Values::Short(v.iter().map(|x| i16::from(*x)).collect()),
        ArrayBuf::I16(v) => Nc3Values::Short(v.clone()),
        ArrayBuf::U16(v) => Nc3Values::Int(v.iter().map(|x| i32::from(*x)).collect()),
        ArrayBuf::I32(v) => Nc3Values::Int(v.clone()),
        ArrayBuf::U32(v) => int_or_double(v.iter().map(|x| f64::from(*x))),
        ArrayBuf::I64(v) => int_or_double(v.iter().map(|x| *x as f64)),
        ArrayBuf::F32(v) => Nc3Values::Float(v.clone()),
        ArrayBuf::F64(v) => Nc3Values::Double(v.clone()),
        ArrayBuf::Text(v) => text(&v.join(", ")),
    }
}

/// A sweep's per-ray `float` values, when it has them.
type RayF32 = dyn Fn(&Sweep) -> Option<&[f32]>;
/// A sweep's per-ray `int` values, when it has them.
type RayI32 = dyn Fn(&Sweep) -> Option<&[i32]>;

/// Where each sweep's gates go in the file (module documentation, "Gate
/// geometry").
struct Grid {
    /// Per sweep: its rows.
    sweeps: Vec<SweepGrid>,
    /// `range(sweep, range)` rather than `range(range)`.
    per_sweep: bool,
    /// Rows whose geometry is not `range(range)`'s, stated per ray
    /// ([`RangeLayout::PerRay`]).
    per_ray: bool,
    /// Fields over `n_points` (`n_gates_vary`): each sweep's rows have its
    /// own length rather than the `range` dimension's.
    ragged: bool,
    /// Length of the `range` dimension.
    ngates: usize,
}

/// One sweep's rows: their gate centres and where the sweep's own range
/// gates fall on them.
#[derive(Clone, Debug)]
struct SweepGrid {
    /// First gate centre and spacing of the rows (metres), or `None` for
    /// explicit centres.
    uniform: Option<(f64, f64)>,
    /// Explicit gate centres (a volume whose sweeps all share them).
    explicit: Vec<f64>,
    /// Row gate of the sweep's first gate.
    offset: usize,
    /// Row gates per sweep gate.
    refine: usize,
    /// Gates of the rows that hold the sweep (its last gate's end).
    ngates: usize,
}

impl SweepGrid {
    /// Centre of row gate `gate`.
    fn center(&self, gate: usize) -> f64 {
        match self.uniform {
            Some((first, spacing)) => first + gate as f64 * spacing,
            None => self.explicit.get(gate).copied().unwrap_or(f64::NAN),
        }
    }
}

/// A sweep's own geometry: first centre, spacing and gates (`None` for a
/// sweep without gates).
fn sweep_geometry(range: &RangeCoord) -> Option<(f64, f64, usize)> {
    match range {
        RangeCoord::Uniform {
            first_center_m,
            spacing_m,
            ngates,
        } if *spacing_m > 0.0 && spacing_m.is_finite() && *ngates > 0 => {
            Some((*first_center_m, *spacing_m, *ngates as usize))
        }
        _ => None,
    }
}

impl Grid {
    fn new(volume: &Volume, layout: RangeLayout) -> Result<Self, CfWriteError> {
        let ranges: Vec<&RangeCoord> = volume.sweeps.iter().map(|sweep| &sweep.range).collect();
        if ranges
            .iter()
            .any(|range| matches!(range, RangeCoord::Explicit { .. }))
        {
            return Self::explicit(&ranges);
        }
        let grid = match layout {
            RangeLayout::PerSweep => Self::own(&ranges, false)?,
            RangeLayout::PerRay => Self::own(&ranges, true)?,
            RangeLayout::Common => Self::common(&ranges)?,
            RangeLayout::Auto => match Self::common(&ranges) {
                Ok(grid) => grid,
                Err(CfWriteError::Unrepresentable(_) | CfWriteError::TooLarge(_)) => {
                    Self::own(&ranges, false)?
                }
                Err(err) => return Err(err),
            },
        };
        // Every reader of this crate (and the others') refuses more gates
        // than this, so a file with more could not be read back.
        if grid.ngates > MAX_GATES_PER_RADIAL {
            return Err(CfWriteError::TooLarge(format!(
                "rows of {} gates (at most {MAX_GATES_PER_RADIAL})",
                grid.ngates
            )));
        }
        Ok(grid)
    }

    /// Explicit gate centres: one `range(range)` when every sweep has the
    /// same ones.
    fn explicit(ranges: &[&RangeCoord]) -> Result<Self, CfWriteError> {
        if !ranges
            .iter()
            .all(|range| matches!(range, RangeCoord::Explicit { .. }))
        {
            return Err(CfWriteError::Unrepresentable(
                "explicit gate centres beside uniform sweeps".into(),
            ));
        }
        let first = ranges[0].centers_f32();
        if ranges.iter().any(|range| range.centers_f32() != first) {
            return Err(CfWriteError::Unrepresentable(
                "explicit gate centres that differ between sweeps".into(),
            ));
        }
        let mut explicit: Vec<f64> = first.iter().map(|c| f64::from(*c)).collect();
        let ngates = explicit.len();
        // One gate: a second centre (1 m on) for the padding gate
        // [`Grid::finish`] adds.
        if let [only] = explicit.as_slice() {
            explicit.push(only + 1.0);
        }
        let sweep = SweepGrid {
            uniform: None,
            explicit,
            offset: 0,
            refine: 1,
            ngates,
        };
        Ok(Self::finish(vec![sweep; ranges.len()], false, false))
    }

    /// One `range(range)` at the finest gate spacing, from the earliest gate
    /// edge; each sweep's gates repeated and offset onto it.
    fn common(ranges: &[&RangeCoord]) -> Result<Self, CfWriteError> {
        let geometries: Vec<Option<(f64, f64, usize)>> =
            ranges.iter().map(|range| sweep_geometry(range)).collect();
        let spacing = geometries
            .iter()
            .flatten()
            .map(|(_, spacing, _)| *spacing)
            .fold(f64::INFINITY, f64::min);
        if !spacing.is_finite() {
            return Err(CfWriteError::Unrepresentable(
                "a volume without gates".into(),
            ));
        }
        let tolerance = 1e-6 * spacing;
        let edge0 = geometries
            .iter()
            .flatten()
            .map(|(first, s, _)| first - s / 2.0)
            .fold(f64::INFINITY, f64::min);
        let first = edge0 + spacing / 2.0;
        let mut sweeps = Vec::with_capacity(geometries.len());
        for (index, geometry) in geometries.iter().enumerate() {
            let Some((center, sweep_spacing, sweep_gates)) = *geometry else {
                sweeps.push(SweepGrid {
                    uniform: Some((first, spacing)),
                    explicit: Vec::new(),
                    offset: 0,
                    refine: 1,
                    ngates: 0,
                });
                continue;
            };
            let refine = (sweep_spacing / spacing).round();
            let edge = center - sweep_spacing / 2.0;
            let offset = ((edge - edge0) / spacing).round();
            if refine < 1.0
                || (refine * spacing - sweep_spacing).abs() > tolerance.max(1e-6)
                || (offset * spacing - (edge - edge0)).abs() > tolerance.max(1e-3)
            {
                return Err(CfWriteError::Unrepresentable(format!(
                    "sweep {index} on one range(range) coordinate: its gates of {sweep_spacing} m \
                     with the first centred at {center} m do not fall on the {spacing} m grid of \
                     the volume's finest sweep, centred from {first} m (RangeLayout::PerSweep \
                     keeps each sweep's own geometry)"
                )));
            }
            let (offset, refine) = (offset as usize, refine as usize);
            let ngates = offset.saturating_add(sweep_gates.saturating_mul(refine));
            if ngates > MAX_GATES_PER_RADIAL {
                return Err(CfWriteError::TooLarge(format!(
                    "sweep {index} on a common range of {spacing} m gates from {first} m: {ngates} \
                     gates (at most {MAX_GATES_PER_RADIAL})"
                )));
            }
            sweeps.push(SweepGrid {
                uniform: Some((first, spacing)),
                explicit: Vec::new(),
                offset,
                refine,
                ngates,
            });
        }
        Ok(Self::finish(sweeps, false, false))
    }

    /// Each sweep with its own geometry; when their first centres or
    /// spacings differ, `range(sweep, range)`, or with `per_ray` the
    /// geometry per ray beside `range(range)`.
    fn own(ranges: &[&RangeCoord], per_ray: bool) -> Result<Self, CfWriteError> {
        let geometries: Vec<Option<(f64, f64, usize)>> =
            ranges.iter().map(|range| sweep_geometry(range)).collect();
        let Some(&(first, spacing, _)) = geometries.iter().flatten().next() else {
            return Err(CfWriteError::Unrepresentable(
                "a volume without gates".into(),
            ));
        };
        let same = |a: f64, b: f64| (a - b).abs() <= 1e-6 * spacing.max(1.0);
        let differ = geometries
            .iter()
            .flatten()
            .any(|(f, s, _)| !same(*f, first) || !same(*s, spacing));
        if differ && per_ray {
            // `ray_start_range` and `ray_gate_spacing` are floats, and
            // readers take a spacing that is not positive, or a start at or
            // below -9999 m, as fill (and read the rays on `range(range)`).
            for (index, geometry) in geometries.iter().enumerate() {
                let Some((f, s, _)) = geometry else { continue };
                let (start, step) = (*f as f32, *s as f32);
                if !(step.is_finite() && step > 0.0 && start.is_finite() && start > -9999.0) {
                    return Err(CfWriteError::Unrepresentable(format!(
                        "sweep {index}: gates of {s} m centred from {f} m, which float \
                         ray_start_range and ray_gate_spacing cannot state \
                         (RangeLayout::PerSweep can)"
                    )));
                }
            }
        }
        let sweeps = geometries
            .iter()
            .map(|geometry| {
                let (f, s, n) = geometry.unwrap_or((first, spacing, 0));
                SweepGrid {
                    uniform: Some((f, s)),
                    explicit: Vec::new(),
                    offset: 0,
                    refine: 1,
                    ngates: n,
                }
            })
            .collect();
        Ok(Self::finish(sweeps, differ && !per_ray, differ && per_ray))
    }

    fn finish(mut sweeps: Vec<SweepGrid>, per_sweep: bool, per_ray: bool) -> Self {
        // A sweep without gates keeps one gate of fill, so that every ray
        // has a row a reader can lay out.
        for sweep in &mut sweeps {
            sweep.ngates = sweep.ngates.max(1);
        }
        // Readers need two gate centres to state the spacing (this crate's
        // refuses fewer): a volume of one gate gets a second of fill, and
        // `n_points` storage keeps each ray to its own gate.
        let ngates = sweeps
            .iter()
            .map(|sweep| sweep.ngates)
            .max()
            .unwrap_or(0)
            .max(2);
        let ragged = sweeps.iter().any(|sweep| sweep.ngates != ngates);
        Self {
            sweeps,
            per_sweep,
            per_ray,
            ragged,
            ngates,
        }
    }

    /// Where field `field` of sweep `sweep` lands on the sweep's rows.
    fn mapping(&self, sweep: usize, field: &Field) -> GateMapping {
        let SweepGrid { offset, refine, .. } = self.sweeps[sweep];
        GateMapping {
            start: (offset + field.gates.start as usize * refine) as u32,
            stride: (field.gates.stride.max(1) as usize * refine) as u32,
        }
    }

    /// Length of the `range` dimension.
    fn ngates(&self) -> usize {
        self.ngates
    }

    /// Gates of each row of sweep `sweep` in the file.
    fn row_len(&self, sweep: usize) -> usize {
        if self.ragged {
            self.sweeps[sweep].ngates
        } else {
            self.ngates
        }
    }

    /// Values of every field variable: each sweep's rays times its rows.
    fn cells(&self, volume: &Volume) -> usize {
        volume
            .sweeps
            .iter()
            .enumerate()
            .map(|(index, sweep)| sweep.nrays().saturating_mul(self.row_len(index)))
            .fold(0usize, usize::saturating_add)
    }
}

/// Integer codes a classic variable can hold (the storage types of the
/// model's integer fields).
pub(super) trait Code: PackedInt + Eq + std::hash::Hash {
    fn candidates() -> Box<dyn Iterator<Item = Self>>;
    /// The values in their classic type: an unsigned type widened to the
    /// next signed one (raw codes unchanged), or with `unsigned_attribute`
    /// the signed type of its width, to be read with `_Unsigned = "true"`.
    fn classic(values: Vec<Self>, unsigned_attribute: bool) -> Nc3Values;
}

macro_rules! code {
    ($t:ty, $same:ident, $same_cast:ty, $wide:ident, $wide_cast:ty) => {
        impl Code for $t {
            fn candidates() -> Box<dyn Iterator<Item = Self>> {
                Box::new(
                    [<$t>::MIN, <$t>::MAX]
                        .into_iter()
                        .chain((<$t>::MIN..<$t>::MAX).take(65_536)),
                )
            }
            fn classic(values: Vec<Self>, unsigned_attribute: bool) -> Nc3Values {
                if unsigned_attribute {
                    Nc3Values::$same(values.into_iter().map(|v| v as $same_cast).collect())
                } else {
                    Nc3Values::$wide(values.into_iter().map(|v| v as $wide_cast).collect())
                }
            }
        }
    };
}

code!(u8, Byte, i8, Short, i16);
code!(i8, Byte, i8, Byte, i8);
code!(u16, Short, i16, Int, i32);
code!(i16, Short, i16, Short, i16);
code!(i32, Int, i32, Int, i32);

/// Where a variable's attributes come from and which the writer owns.
struct AttrSource<'a> {
    volume: &'a Volume,
    /// Write the writer's default attributes where the source has none
    /// (not for a CfRadial 1 volume: its variables without attributes had
    /// none).
    defaults: bool,
}

impl AttrSource<'_> {
    /// The source's own attributes of variable `name` (root first, then the
    /// first sweep group of a CfRadial 2 volume).
    fn source(&self, name: &str) -> Option<&[(Box<str>, AttrValue)]> {
        let entries = &self.volume.variable_attrs;
        entries
            .iter()
            .find(|entry| entry.group.is_empty() && &*entry.name == name)
            .or_else(|| {
                entries
                    .iter()
                    .find(|entry| entry.group.starts_with("sweep_") && &*entry.name == name)
            })
            .or_else(|| entries.iter().find(|entry| &*entry.name == name))
            .map(|entry| entry.attrs.as_slice())
    }

    /// The attributes of `name`: the source's own, verbatim, with `owned`
    /// values replacing the source's (and appended when missing); the
    /// writer's `defaults` when the source has none (and defaults apply).
    fn attrs(
        &self,
        name: &str,
        defaults: Vec<(&str, Nc3Values)>,
        owned: Vec<(&str, Nc3Values)>,
    ) -> Vec<(String, Nc3Values)> {
        match self.source(name) {
            Some(source) => {
                let mut out: Vec<(String, Nc3Values)> = source
                    .iter()
                    .map(|(key, value)| (sanitize_name(key), attr_values(value)))
                    .collect();
                dedupe(&mut out);
                for (key, value) in owned {
                    match out.iter_mut().find(|(have, _)| have == key) {
                        Some(slot) => slot.1 = value,
                        None => out.push((key.to_owned(), value)),
                    }
                }
                out
            }
            None => {
                let mut out: Vec<(String, Nc3Values)> = if self.defaults {
                    defaults
                        .into_iter()
                        .map(|(key, value)| (key.to_owned(), value))
                        .collect()
                } else {
                    Vec::new()
                };
                for (key, value) in owned {
                    match out.iter_mut().find(|(have, _)| have == key) {
                        Some(slot) => slot.1 = value,
                        None => out.push((key.to_owned(), value)),
                    }
                }
                out
            }
        }
    }

    /// The position of variable `name` in the source (for writing variables
    /// in the source's order).
    fn position(&self, name: &str) -> usize {
        self.volume
            .variable_attrs
            .iter()
            .position(|entry| entry.group.is_empty() && &*entry.name == name)
            .unwrap_or(usize::MAX)
    }
}

/// Keep the first of attributes whose sanitized names collide.
fn dedupe(attrs: &mut Vec<(String, Nc3Values)>) {
    let mut seen = HashSet::new();
    attrs.retain(|(name, _)| seen.insert(name.clone()));
}

/// Variables collected before writing, so slotted ones go out in the
/// source's order.
struct Pending {
    slotted: Vec<(usize, Nc3Variable)>,
    fields: Vec<Nc3Variable>,
    extras: Vec<Nc3Variable>,
}

/// The order a CfRadial 1 file written with `options` stores sweeps and
/// rays in. Sweeps go by their first ray time (stable; unchanged when a
/// sweep has no finite time), as xradar's reader requires: it sorts every
/// ray of the file by time before cutting the sweeps out by their ray
/// indices. Rays keep their order within a sweep, except in `n_points`
/// storage (module documentation, "Gates per ray"), where each sweep's rays
/// go in time order when all of them have a finite time: xradar lays those
/// rows out by ray time and misplaces the rays of a sweep stored out of time
/// order (sweeps stored in azimuth order). Returns the sweep order and each
/// sweep's ray order, or `None` when the volume already is in that order.
pub fn time_order(
    volume: &Volume,
    options: &Cfradial1Options,
) -> Option<(Vec<usize>, Vec<Vec<usize>>)> {
    let first_time = |sweep: &Sweep| {
        sweep
            .rays
            .time_s
            .iter()
            .copied()
            .filter(|t| t.is_finite())
            .fold(f64::INFINITY, f64::min)
    };
    let mut sweeps: Vec<usize> = (0..volume.sweeps.len()).collect();
    if volume
        .sweeps
        .iter()
        .all(|sweep| first_time(sweep).is_finite())
    {
        sweeps.sort_by(|a, b| {
            first_time(&volume.sweeps[*a]).total_cmp(&first_time(&volume.sweeps[*b]))
        });
    }
    let ragged = Grid::new(volume, options.range_layout).is_ok_and(|grid| grid.ragged);
    let rays: Vec<Vec<usize>> = sweeps
        .iter()
        .map(|index| {
            let sweep = &volume.sweeps[*index];
            let mut order: Vec<usize> = (0..sweep.nrays()).collect();
            let times = &sweep.rays.time_s;
            if ragged && times.len() == order.len() && times.iter().all(|t| t.is_finite()) {
                order.sort_by(|a, b| times[*a].total_cmp(&times[*b]));
            }
            order
        })
        .collect();
    let identity = sweeps.iter().enumerate().all(|(i, s)| i == *s)
        && rays
            .iter()
            .all(|order| order.iter().enumerate().all(|(i, r)| i == *r));
    (!identity).then_some((sweeps, rays))
}

/// `volume` in [`time_order`]: a moved sweep keeps its original index as
/// `sweep_number` (in `Sweep::other`, which the writer's `sweep_number`
/// variable takes), scan legs follow their sweeps.
fn in_time_order(
    volume: &Volume,
    options: &Cfradial1Options,
) -> Result<Option<Volume>, CfWriteError> {
    let Some((sweeps, rays)) = time_order(volume, options) else {
        return Ok(None);
    };
    let mut ordered = volume.clone();
    let mut moved: Vec<Sweep> = Vec::with_capacity(sweeps.len());
    for (position, (index, order)) in sweeps.iter().zip(&rays).enumerate() {
        let mut sweep = volume.sweeps[*index].clone();
        let order: Vec<u32> = order
            .iter()
            .map(|row| u32::try_from(*row))
            .collect::<Result<_, _>>()
            .map_err(|_| CfWriteError::Invalid(format!("sweep {index}: too many rays")))?;
        sweep
            .permute_rays(&order)
            .map_err(|err| CfWriteError::Invalid(format!("sweep {index}: {err}")))?;
        if position != *index {
            let original = sweep
                .other
                .iter()
                .find(|(name, _)| &**name == "sweep_number")
                .and_then(|(_, value)| value.as_f64())
                .unwrap_or(*index as f64);
            sweep.other.retain(|(name, _)| &**name != "sweep_number");
            sweep.other.push((
                "sweep_number".into(),
                AttrValue::Scalar(Scalar::F64(original)),
            ));
        }
        sweep.sweep_number = position as u32;
        moved.push(sweep);
    }
    ordered.sweeps = moved;
    if let Some(definition) = &mut ordered.scan.definition
        && definition.legs.len() == sweeps.len()
    {
        definition.legs = sweeps
            .iter()
            .map(|index| definition.legs[*index].clone())
            .collect();
    }
    Ok(Some(ordered))
}

/// Refuse a volume whose fields on the common range take more memory than
/// a reader's decode budget ([`MAX_DECODED_VOLUME_BYTES`]): the writer holds
/// that much, and the file could not be read back.
fn check_size(volume: &Volume, grid: &Grid) -> Result<(), CfWriteError> {
    let nrays: usize = volume.sweeps.iter().map(Sweep::nrays).sum();
    let cells = grid.cells(volume);
    let mut seen: Vec<&str> = Vec::new();
    let mut bytes = 0usize;
    let fields = || volume.sweeps.iter().flat_map(|sweep| &sweep.fields);
    for field in fields() {
        if seen.contains(&field.name.as_str()) {
            continue;
        }
        seen.push(field.name.as_str());
        let coding = field.data.coding();
        let same = fields()
            .filter(|other| other.name == field.name)
            .all(|other| other.data.coding() == coding);
        // A field whose coding differs between sweeps is written as float.
        let element = match &field.data {
            _ if !same => 4,
            FieldData::U8 { .. } | FieldData::I8 { .. } => 1,
            FieldData::U16 { .. } | FieldData::I16 { .. } => 2,
            FieldData::I32 { .. } | FieldData::F32 { .. } => 4,
            FieldData::F64 { .. } => 8,
        };
        bytes = bytes.saturating_add(cells.saturating_mul(element));
    }
    if bytes > MAX_DECODED_VOLUME_BYTES {
        return Err(CfWriteError::TooLarge(format!(
            "{bytes} bytes of fields in {cells} gates of {nrays} rays (at most \
             {MAX_DECODED_VOLUME_BYTES})"
        )));
    }
    Ok(())
}

/// Write `volume` as CfRadial 1.4 classic netCDF. See the module
/// documentation.
pub fn write_cfradial1(
    volume: &Volume,
    options: &Cfradial1Options,
) -> Result<Vec<u8>, CfWriteError> {
    let ordered = in_time_order(volume, options)?;
    let volume = ordered.as_ref().unwrap_or(volume);
    if volume.sweeps.is_empty() {
        return Err(CfWriteError::Unrepresentable(
            "a volume without sweeps".into(),
        ));
    }
    if let Some((index, _)) = volume
        .sweeps
        .iter()
        .enumerate()
        .find(|(_, sweep)| sweep.nrays() == 0)
    {
        return Err(CfWriteError::Unrepresentable(format!(
            "sweep {index}: a sweep without rays"
        )));
    }
    // A NEXRAD Level III level table is not a linear coding, which
    // `scale_factor` / `add_offset` alone would misstate. (The CfRadial 2 /
    // FM301 writer writes such a field decoded.)
    for (index, sweep) in volume.sweeps.iter().enumerate() {
        if let Some(field) = sweep
            .fields
            .iter()
            .find(|field| field.data.transform().is_some_and(|t| !t.is_linear()))
        {
            return Err(CfWriteError::Unrepresentable(format!(
                "sweep {index} field {}: a level-table coding (NEXRAD Level III), which scale_factor and add_offset cannot state",
                field.name.as_str()
            )));
        }
    }
    let grid = Grid::new(volume, options.range_layout)?;
    if grid.ngates() == 0 {
        return Err(CfWriteError::Unrepresentable(
            "a volume without gates".into(),
        ));
    }
    check_size(volume, &grid)?;
    let cfradial1 = volume.provenance.source_format == SourceFormat::CfRadial1;
    let source = AttrSource {
        volume,
        defaults: !cfradial1,
    };
    let mut nc = Nc3Writer::new();
    let nrays: usize = volume.sweeps.iter().map(Sweep::nrays).sum();
    let time = nc.add_dim("time", nrays as u64)?;
    let range = nc.add_dim("range", grid.ngates() as u64)?;
    let n_points = if grid.ragged {
        let points = grid.cells(volume) as u64;
        // `ray_start_index` is an `int`.
        if points > i32::MAX as u64 {
            return Err(CfWriteError::TooLarge(format!(
                "{points} gates in n_points (at most {})",
                i32::MAX
            )));
        }
        Some(nc.add_dim("n_points", points)?)
    } else {
        None
    };
    let sweep_dim = nc.add_dim("sweep", volume.sweeps.len() as u64)?;
    let mut writer = Writer {
        nc: &mut nc,
        volume,
        source: &source,
        time,
        range,
        n_points,
        sweep: sweep_dim,
        grid: &grid,
        pending: Pending {
            slotted: Vec::new(),
            fields: Vec::new(),
            extras: Vec::new(),
        },
        cfradial1,
        unsigned_attribute: options.unsigned_attribute,
    };
    writer.global_attrs()?;
    writer.root_variables()?;
    writer.sweep_variables()?;
    writer.calibration()?;
    writer.coordinates()?;
    writer.ray_variables()?;
    writer.gate_geometry();
    writer.fields()?;
    writer.extras()?;
    if cfradial1 {
        writer.restore_empty_variables();
    }
    let Pending {
        mut slotted,
        fields,
        extras,
    } = writer.pending;
    slotted.sort_by_key(|(position, _)| *position);
    for var in slotted
        .into_iter()
        .map(|(_, var)| var)
        .chain(fields)
        .chain(extras)
    {
        if !nc.has_var(&var.name) {
            nc.add_var(var)?;
        }
    }
    nc.finish()
}

struct Writer<'a> {
    nc: &'a mut Nc3Writer,
    volume: &'a Volume,
    source: &'a AttrSource<'a>,
    time: usize,
    range: usize,
    /// The `n_points` dimension of `n_gates_vary` storage.
    n_points: Option<usize>,
    sweep: usize,
    grid: &'a Grid,
    pending: Pending,
    cfradial1: bool,
    /// [`Cfradial1Options::unsigned_attribute`].
    unsigned_attribute: bool,
}

impl Writer<'_> {
    fn global(&mut self, name: &str, value: Nc3Values) -> Result<(), CfWriteError> {
        let name = sanitize_name(name);
        if self.nc.has_attr(&name) {
            return Ok(());
        }
        self.nc.add_attr(&name, value)
    }

    fn global_attrs(&mut self) -> Result<(), CfWriteError> {
        let volume = self.volume;
        let global = &volume.attrs;
        let (conventions, version) = if self.cfradial1 {
            (
                volume.provenance.source_conventions.clone(),
                volume.provenance.source_version.clone(),
            )
        } else {
            (Some(CONVENTIONS.to_owned()), Some(VERSION.to_owned()))
        };
        if let Some(conventions) = conventions {
            self.global("Conventions", text(&conventions))?;
        }
        if let Some(version) = version {
            self.global("version", text(&version))?;
        }
        for (name, value) in [
            ("title", &global.title),
            ("institution", &global.institution),
            ("references", &global.references),
            ("source", &global.source),
            ("history", &global.history),
            ("comment", &global.comment),
        ] {
            if let Some(value) = value {
                self.global(name, text(value))?;
            }
        }
        // Required by CfRadial 1.4 (section 4.1) and by LROSE Radx, which
        // refuses a file without it: written empty for a volume with no name
        // (an ARCHIVE2 header without an ICAO), as the FM301 view does.
        self.global("instrument_name", text(&global.instrument_name))?;
        if let Some(site) = &global.site_name {
            self.global("site_name", text(site))?;
        }
        let scan = &volume.scan;
        if let Some(name) = &scan.name {
            self.global("scan_name", text(name))?;
        }
        let scan_id_text = scan
            .definition
            .as_ref()
            .and_then(|definition| definition.scan_id_text.clone());
        match (scan_id_text, scan.id) {
            (Some(id), _) => self.global("scan_id", text(&id))?,
            (None, Some(id)) => {
                self.global("scan_id", int_or_double(std::iter::once(id as f64)))?
            }
            (None, None) => {}
        }
        self.global(
            "platform_is_mobile",
            text(if global.platform_is_mobile {
                "true"
            } else {
                "false"
            }),
        )?;
        if let Some(increase) = global.ray_times_increase {
            self.global(
                "ray_times_increase",
                text(if increase { "true" } else { "false" }),
            )?;
        }
        if global.simulated && volume.simulation.is_none() {
            self.global("simulated", text("true"))?;
        }
        if let Some(pattern) = scan.vcp_pattern {
            self.global("vcp_pattern", Nc3Values::Int(vec![i32::from(pattern)]))?;
        }
        if let Some(definition) = &scan.definition {
            for (name, value) in [
                ("vcp_source_document", &definition.source_document),
                ("vcp_source_revision", &definition.source_revision),
                ("vcp_source_rda_build", &definition.source_rda_build),
                ("vcp_source_figure", &definition.source_figure),
                ("vcp_pulse_length", &definition.pulse_length),
                ("vcp_adaptations", &definition.adaptations),
            ] {
                if let Some(value) = value {
                    self.global(name, text(value))?;
                }
            }
        }
        let provenance = &volume.provenance;
        if let Some(note) = &provenance.polarization_note {
            self.global("polarization", text(note))?;
        }
        if let Some(note) = &provenance.calibration_note {
            self.global("calibration", text(note))?;
        }
        if let Some(simulation) = &volume.simulation {
            for (name, value) in [
                ("forward_operator", &simulation.forward_operator),
                (
                    "forward_operator_config",
                    &simulation.forward_operator_config,
                ),
                ("source_model", &simulation.source_model),
                ("microphysics_scheme", &simulation.microphysics_scheme),
                ("scattering_model", &simulation.scattering_model),
            ] {
                if let Some(value) = value {
                    self.global(name, text(value))?;
                }
            }
        }
        // Whether fields are over `n_points` is this file's layout, not the
        // source's (a kept `n_gates_vary` stays where it was).
        let n_gates_vary = text(if self.grid.ragged { "true" } else { "false" });
        for (name, value) in &global.other {
            match &**name {
                // `_NCProperties` describes a netCDF-4 container, not this
                // file.
                "_NCProperties" => {}
                "n_gates_vary" => self.global(name, n_gates_vary.clone())?,
                _ => self.global(name, attr_values(value))?,
            }
        }
        if self.grid.ragged {
            self.global("n_gates_vary", n_gates_vary)?;
        }
        Ok(())
    }

    /// A string dimension `string_length_<n>` (created on first use).
    fn string_dim(&mut self, len: usize) -> Result<usize, CfWriteError> {
        let name = format!("string_length_{len}");
        match self.nc.dim(&name) {
            Some(id) => Ok(id),
            None => self.nc.add_dim(&name, len as u64),
        }
    }

    /// A slotted variable with the source's attributes or `defaults`.
    fn slotted(
        &mut self,
        name: &str,
        dims: Vec<usize>,
        values: Nc3Values,
        defaults: Vec<(&str, Nc3Values)>,
        owned: Vec<(&str, Nc3Values)>,
    ) {
        let attrs = self.source.attrs(name, defaults, owned);
        let position = self.source.position(name);
        self.pending.slotted.push((
            position,
            Nc3Variable {
                name: name.to_owned(),
                dims,
                attrs,
                values,
            },
        ));
    }

    /// A scalar `char` variable.
    fn text_variable(
        &mut self,
        name: &str,
        value: &str,
        defaults: Vec<(&str, Nc3Values)>,
    ) -> Result<(), CfWriteError> {
        let len = value.len().max(1);
        let dim = self.string_dim(len.max(8).next_power_of_two().min(len.max(32)))?;
        let width = self.nc.dim_len(dim).unwrap_or(len as u64) as usize;
        let mut bytes = value.as_bytes().to_vec();
        bytes.resize(width.max(bytes.len()), 0);
        let dim = if bytes.len() > width {
            self.string_dim(bytes.len())?
        } else {
            dim
        };
        self.slotted(
            name,
            vec![dim],
            Nc3Values::Char(bytes),
            defaults,
            Vec::new(),
        );
        Ok(())
    }

    fn root_variables(&mut self) -> Result<(), CfWriteError> {
        let volume = self.volume;
        if let Some(number) = volume.volume_number {
            self.slotted(
                "volume_number",
                Vec::new(),
                Nc3Values::Int(vec![number]),
                vec![("long_name", text("data_volume_index_number"))],
                Vec::new(),
            );
        }
        self.text_variable(
            "platform_type",
            volume.platform_type.as_str(),
            vec![("long_name", text("platform_type"))],
        )?;
        self.text_variable(
            "instrument_type",
            volume.instrument_type.as_str(),
            vec![("long_name", text("type_of_instrument"))],
        )?;
        if let Some(axis) = volume.primary_axis {
            self.text_variable(
                "primary_axis",
                axis.as_str(),
                vec![("long_name", text("primary_axis_of_rotation"))],
            )?;
        }
        if let Some(status) = &volume.status_str {
            self.text_variable("status_str", status, Vec::new())?;
        }
        let coverage = volume.time_coverage.or_else(|| volume.ray_time_extent());
        let (start, end) = coverage.map_or((volume.time_reference, volume.time_reference), |c| {
            (c.start, c.end)
        });
        self.text_variable(
            "time_coverage_start",
            &time_string(start),
            vec![
                ("standard_name", text("data_volume_start_time_utc")),
                (
                    "comment",
                    text("ray times are relative to start time in secs"),
                ),
            ],
        )?;
        self.text_variable(
            "time_coverage_end",
            &time_string(end),
            vec![("standard_name", text("data_volume_end_time_utc"))],
        )?;
        // A moving platform's track is written per ray (ray_variables).
        let moving = volume
            .sweeps
            .iter()
            .all(|sweep| sweep.platform_track.is_some());
        if !moving {
            let location = &volume.location;
            for (name, value, units, standard_name) in [
                (
                    "latitude",
                    location.latitude_deg,
                    "degrees_north",
                    "latitude",
                ),
                (
                    "longitude",
                    location.longitude_deg,
                    "degrees_east",
                    "longitude",
                ),
                ("altitude", location.altitude_m, "meters", "altitude"),
            ] {
                // CfRadial requires the location: a volume without one
                // (NEXRAD Message 1) gets NaN with a NaN `_FillValue`.
                let mut defaults = vec![
                    ("long_name", text(standard_name)),
                    ("standard_name", text(standard_name)),
                    ("units", text(units)),
                ];
                if value.is_none() {
                    defaults.push(("_FillValue", Nc3Values::Double(vec![f64::NAN])));
                }
                // A CfRadial 1 volume writes it when its file had it.
                let in_source = self
                    .volume
                    .variable_attrs
                    .iter()
                    .any(|entry| entry.group.is_empty() && &*entry.name == name);
                if value.is_some() || !self.cfradial1 || in_source {
                    self.slotted(
                        name,
                        Vec::new(),
                        Nc3Values::Double(vec![value.unwrap_or(f64::NAN)]),
                        defaults,
                        Vec::new(),
                    );
                }
            }
            if let Some(value) = location.altitude_agl_m {
                self.slotted(
                    "altitude_agl",
                    Vec::new(),
                    Nc3Values::Double(vec![value]),
                    vec![("units", text("meters"))],
                    Vec::new(),
                );
            }
        }
        let parameters = &volume.radar_parameters;
        if !parameters.frequency_hz.is_empty() {
            let dim = self
                .nc
                .add_dim("frequency", parameters.frequency_hz.len() as u64)?;
            self.slotted(
                "frequency",
                vec![dim],
                Nc3Values::Double(parameters.frequency_hz.clone()),
                vec![
                    ("standard_name", text("radiation_frequency")),
                    ("units", text("s-1")),
                ],
                Vec::new(),
            );
        }
        for (name, value, units) in [
            ("radar_antenna_gain_h", parameters.antenna_gain_h_db, "dB"),
            ("radar_antenna_gain_v", parameters.antenna_gain_v_db, "dB"),
            ("radar_beam_width_h", parameters.beam_width_h_deg, "degrees"),
            ("radar_beam_width_v", parameters.beam_width_v_deg, "degrees"),
            (
                "radar_rx_bandwidth",
                parameters.receiver_bandwidth_hz,
                "s-1",
            ),
        ] {
            if let Some(value) = value {
                self.slotted(
                    name,
                    Vec::new(),
                    Nc3Values::Float(vec![value]),
                    vec![("units", text(units))],
                    Vec::new(),
                );
            }
        }
        // Volume constants a sweep has no per-ray values for become (time)
        // variables in ray_variables; the georeferencing corrections become
        // scalar variables.
        if let Some(correction) = &volume.georeferencing_correction {
            for (name, value) in correction.entries() {
                if let Some(value) = value {
                    self.slotted(
                        name,
                        Vec::new(),
                        Nc3Values::Float(vec![value]),
                        Vec::new(),
                        Vec::new(),
                    );
                }
            }
        }
        Ok(())
    }

    fn sweep_variables(&mut self) -> Result<(), CfWriteError> {
        let volume = self.volume;
        let sweeps = &volume.sweeps;
        let sweep = self.sweep;
        let numbers: Vec<i32> = sweeps
            .iter()
            .enumerate()
            .map(|(index, sweep)| {
                sweep
                    .other
                    .iter()
                    .find(|(name, _)| &**name == "sweep_number")
                    .and_then(|(_, value)| value.as_f64())
                    .filter(|v| v.fract() == 0.0 && v.abs() <= f64::from(i32::MAX))
                    .map_or(index as i32, |v| v as i32)
            })
            .collect();
        self.slotted(
            "sweep_number",
            vec![sweep],
            Nc3Values::Int(numbers),
            vec![("long_name", text("sweep_index_number_0_based"))],
            Vec::new(),
        );
        let mut start = 0i32;
        let mut starts = Vec::with_capacity(sweeps.len());
        let mut ends = Vec::with_capacity(sweeps.len());
        for sweep in sweeps {
            starts.push(start);
            start += sweep.nrays() as i32;
            ends.push(start - 1);
        }
        self.slotted(
            "fixed_angle",
            vec![sweep],
            Nc3Values::Float(sweeps.iter().map(|s| s.fixed_angle_deg).collect()),
            vec![
                ("long_name", text("ray_target_fixed_angle")),
                ("units", text("degrees")),
            ],
            Vec::new(),
        );
        self.slotted(
            "sweep_start_ray_index",
            vec![sweep],
            Nc3Values::Int(starts),
            vec![("long_name", text("index_of_first_ray_in_sweep"))],
            Vec::new(),
        );
        self.slotted(
            "sweep_end_ray_index",
            vec![sweep],
            Nc3Values::Int(ends),
            vec![("long_name", text("index_of_last_ray_in_sweep"))],
            Vec::new(),
        );
        let modes: Vec<String> = sweeps
            .iter()
            .map(|s| s.sweep_mode.as_str().to_owned())
            .collect();
        self.sweep_strings("sweep_mode", &modes, "scan_mode_for_sweep")?;
        let optional = |f: &dyn Fn(&Sweep) -> Option<String>| -> Option<Vec<String>> {
            let values: Vec<Option<String>> = sweeps.iter().map(f).collect();
            values
                .iter()
                .any(Option::is_some)
                .then(|| values.into_iter().map(Option::unwrap_or_default).collect())
        };
        if let Some(values) =
            optional(&|s| s.polarization_mode.as_ref().map(|m| m.as_str().to_owned()))
        {
            self.sweep_strings("polarization_mode", &values, "polarization_mode_for_sweep")?;
        }
        if let Some(values) = optional(&|s| s.prt_mode.as_ref().map(|m| m.as_str().to_owned())) {
            self.sweep_strings("prt_mode", &values, "transmit_pulse_mode")?;
        }
        if let Some(values) = optional(&|s| s.follow_mode.as_ref().map(|m| m.as_str().to_owned())) {
            self.sweep_strings("follow_mode", &values, "follow_mode_for_scan_strategy")?;
        }
        if let Some(values) = optional(&|s| {
            s.rays_are_indexed
                .map(|indexed| if indexed { "true" } else { "false" }.to_owned())
        }) {
            self.sweep_strings("rays_are_indexed", &values, "flag_for_indexed_rays")?;
        }
        for (name, values, long_name) in [
            (
                "target_scan_rate",
                sweeps
                    .iter()
                    .map(|s| s.target_scan_rate_deg_per_s)
                    .collect::<Vec<_>>(),
                "target_scan_rate_for_sweep",
            ),
            (
                "ray_angle_res",
                sweeps.iter().map(|s| s.rays_angle_resolution_deg).collect(),
                "angular_resolution_between_rays",
            ),
        ] {
            if values.iter().any(Option::is_some) {
                let units = if name == "target_scan_rate" {
                    "degrees per second"
                } else {
                    "degrees"
                };
                self.slotted(
                    name,
                    vec![sweep],
                    Nc3Values::Float(values.iter().map(|v| v.unwrap_or(FLOAT_FILL)).collect()),
                    vec![
                        ("long_name", text(long_name)),
                        ("units", text(units)),
                        ("_FillValue", Nc3Values::Float(vec![FLOAT_FILL])),
                    ],
                    Vec::new(),
                );
            }
        }
        self.scan_legs()?;
        if !self.cfradial1 {
            self.sweep_attributes()?;
        }
        Ok(())
    }

    /// A `(sweep, string_length)` char variable.
    fn sweep_strings(
        &mut self,
        name: &str,
        values: &[String],
        long_name: &str,
    ) -> Result<(), CfWriteError> {
        let width = values.iter().map(String::len).max().unwrap_or(0).max(32);
        let dim = self.string_dim(width)?;
        let mut bytes = Vec::with_capacity(values.len() * width);
        for value in values {
            bytes.extend_from_slice(value.as_bytes());
            bytes.resize(bytes.len() + width - value.len(), 0);
        }
        self.slotted(
            name,
            vec![self.sweep, dim],
            Nc3Values::Char(bytes),
            vec![("long_name", text(long_name))],
            Vec::new(),
        );
        Ok(())
    }

    /// The BowEcho scan-leg variables (`vcp_*(sweep)`).
    fn scan_legs(&mut self) -> Result<(), CfWriteError> {
        let Some(definition) = &self.volume.scan.definition else {
            return Ok(());
        };
        let legs = &definition.legs;
        if legs.len() != self.volume.sweeps.len() {
            return Ok(());
        }
        let sweep = self.sweep;
        let int = |f: &dyn Fn(&recast_radar_core::model::ScanLeg) -> Option<i32>| {
            legs.iter()
                .map(|leg| f(leg).unwrap_or(INT_FILL))
                .collect::<Vec<_>>()
        };
        let float = |f: &dyn Fn(&recast_radar_core::model::ScanLeg) -> Option<f32>| {
            legs.iter()
                .map(|leg| f(leg).unwrap_or(FLOAT_FILL))
                .collect::<Vec<_>>()
        };
        let waveform = |leg: &recast_radar_core::model::ScanLeg| {
            Some(match leg.waveform.as_deref()? {
                "CS" => 1,
                "CD/W" => 2,
                "B" => 3,
                "CD/WO" => 4,
                "SZCS" => 5,
                "SZCD" => 6,
                _ => return None,
            })
        };
        let coverage = |leg: &recast_radar_core::model::ScanLeg| {
            Some(match leg.moment_coverage.as_deref()? {
                "surveillance" => 1,
                "doppler" => 2,
                "all" => 3,
                _ => return None,
            })
        };
        let columns: Vec<(&str, Nc3Values)> = vec![
            (
                "vcp_source_row_index",
                Nc3Values::Int(int(&|l| l.source_row_index.map(i32::from))),
            ),
            (
                "vcp_azimuth_rate",
                Nc3Values::Float(float(&|l| l.azimuth_rate_deg_per_second)),
            ),
            (
                "vcp_source_period",
                Nc3Values::Float(float(&|l| l.source_period_seconds)),
            ),
            ("vcp_waveform_code", Nc3Values::Int(int(&waveform))),
            ("vcp_moment_coverage_code", Nc3Values::Int(int(&coverage))),
            (
                "vcp_surveillance_prf_code",
                Nc3Values::Int(int(&|l| l.surveillance_prf_code.map(i32::from))),
            ),
            (
                "vcp_surveillance_pulse_count",
                Nc3Values::Int(int(&|l| l.surveillance_pulse_count.map(i32::from))),
            ),
            (
                "vcp_doppler_prf_code",
                Nc3Values::Int(int(&|l| l.doppler_prf_code.map(i32::from))),
            ),
            (
                "vcp_doppler_pulse_count",
                Nc3Values::Int(int(&|l| l.doppler_pulse_count.map(i32::from))),
            ),
        ];
        for (name, values) in columns {
            let empty = match &values {
                Nc3Values::Int(v) => v.iter().all(|x| *x == INT_FILL),
                Nc3Values::Float(v) => v.iter().all(|x| *x == FLOAT_FILL),
                _ => false,
            };
            if !empty {
                self.slotted(name, vec![sweep], values, Vec::new(), Vec::new());
            }
        }
        Ok(())
    }

    /// Sweep attributes of other formats (`Sweep::other`) as variables:
    /// text (and text arrays, joined with ", ") as `(sweep, string_length)`
    /// `char`; numbers as `double` with a NaN `_FillValue` for sweeps
    /// without the attribute: scalars `(sweep)`, arrays with a value per
    /// ray of their sweep `(time)` (ODIM's per-ray `how` arrays), other
    /// arrays `(sweep, <name>_len)` padded with NaN. An attribute that is
    /// text in one sweep and a number in another is written as text.
    fn sweep_attributes(&mut self) -> Result<(), CfWriteError> {
        let sweeps = &self.volume.sweeps;
        let mut names: Vec<&str> = Vec::new();
        for sweep in sweeps {
            for (name, _) in &sweep.other {
                if !names.contains(&&**name) {
                    names.push(name);
                }
            }
        }
        for name in names {
            let values: Vec<Option<&AttrValue>> = sweeps
                .iter()
                .map(|sweep| {
                    sweep
                        .other
                        .iter()
                        .find(|(key, _)| &**key == name)
                        .map(|(_, value)| value)
                })
                .collect();
            let var_name = sanitize_name(name);
            // xradar renames `sweep_number` and `fixed_angle` to these.
            if matches!(var_name.as_str(), "sweep_group_name" | "sweep_fixed_angle")
                || self.nc.has_var(&var_name)
                || self.pending.slotted.iter().any(|(_, v)| v.name == var_name)
                || self.pending.extras.iter().any(|v| v.name == var_name)
            {
                continue;
            }
            let numeric = |value: &AttrValue| {
                matches!(value, AttrValue::Scalar(_))
                    || matches!(value, AttrValue::Array(array) if !matches!(array, ArrayBuf::Text(_)))
            };
            let var = if values.iter().flatten().all(|value| numeric(value)) {
                self.numeric_sweep_attribute(var_name, &values)?
            } else {
                let texts: Vec<String> = values
                    .iter()
                    .map(|value| value.map(attr_text).unwrap_or_default())
                    .collect();
                let width = texts.iter().map(String::len).max().unwrap_or(1).max(1);
                let dim = self.string_dim(width)?;
                let mut bytes = Vec::with_capacity(texts.len() * width);
                for value in &texts {
                    bytes.extend_from_slice(value.as_bytes());
                    bytes.resize(bytes.len() + width - value.len(), 0);
                }
                Nc3Variable {
                    name: var_name,
                    dims: vec![self.sweep, dim],
                    attrs: Vec::new(),
                    values: Nc3Values::Char(bytes),
                }
            };
            self.pending.extras.push(var);
        }
        Ok(())
    }

    /// A numeric sweep attribute as a `double` variable (see
    /// [`Self::sweep_attributes`]).
    fn numeric_sweep_attribute(
        &mut self,
        name: String,
        values: &[Option<&AttrValue>],
    ) -> Result<Nc3Variable, CfWriteError> {
        let sweeps = &self.volume.sweeps;
        let numbers = |value: &AttrValue| -> Vec<f64> {
            match value {
                AttrValue::Scalar(scalar) => vec![scalar.as_f64()],
                AttrValue::Array(array) => (0..array.len())
                    .map(|index| array.get_f64(index).unwrap_or(f64::NAN))
                    .collect(),
                AttrValue::Text(_) | AttrValue::Bool(_) => Vec::new(),
            }
        };
        let fill = vec![("_FillValue".to_owned(), Nc3Values::Double(vec![f64::NAN]))];
        let per_ray = values.iter().zip(sweeps).all(|(value, sweep)| match value {
            None => true,
            Some(AttrValue::Array(array)) => array.len() == sweep.nrays(),
            Some(_) => false,
        });
        let scalar = values
            .iter()
            .flatten()
            .all(|value| matches!(value, AttrValue::Scalar(_)));
        if scalar {
            return Ok(Nc3Variable {
                name,
                dims: vec![self.sweep],
                attrs: fill,
                values: Nc3Values::Double(
                    values
                        .iter()
                        .map(|value| value.and_then(AttrValue::as_f64).unwrap_or(f64::NAN))
                        .collect(),
                ),
            });
        }
        if per_ray {
            let mut out = Vec::new();
            for (value, sweep) in values.iter().zip(sweeps) {
                match value {
                    Some(value) => out.extend(numbers(value)),
                    None => out.extend(std::iter::repeat_n(f64::NAN, sweep.nrays())),
                }
            }
            return Ok(Nc3Variable {
                name,
                dims: vec![self.time],
                attrs: fill,
                values: Nc3Values::Double(out),
            });
        }
        let rows: Vec<Vec<f64>> = values
            .iter()
            .map(|value| value.map(numbers).unwrap_or_default())
            .collect();
        let width = rows.iter().map(Vec::len).max().unwrap_or(1).max(1);
        let dim_name = sanitize_name(&format!("{name}_len"));
        let dim = match self.nc.dim(&dim_name) {
            Some(id) if self.nc.dim_len(id) == Some(width as u64) => id,
            Some(_) => {
                return Err(CfWriteError::Invalid(format!(
                    "dimension {dim_name} exists with another length"
                )));
            }
            None => self.nc.add_dim(&dim_name, width as u64)?,
        };
        let mut out = Vec::with_capacity(rows.len() * width);
        for row in rows {
            let len = row.len();
            out.extend(row);
            out.extend(std::iter::repeat_n(f64::NAN, width - len));
        }
        Ok(Nc3Variable {
            name,
            dims: vec![self.sweep, dim],
            attrs: fill,
            values: Nc3Values::Double(out),
        })
    }

    fn calibration(&mut self) -> Result<(), CfWriteError> {
        let calibration = &self.volume.radar_calibration;
        if calibration.is_empty() {
            return Ok(());
        }
        let dim = self.nc.add_dim("r_calib", calibration.len() as u64)?;
        if calibration.iter().any(|entry| entry.time_s.is_some()) {
            let texts: Vec<String> = calibration
                .iter()
                .map(|entry| {
                    entry
                        .time_s
                        .and_then(|time| self.volume.instant(time))
                        .map(time_string_ms)
                        .unwrap_or_default()
                })
                .collect();
            let width = texts.iter().map(String::len).max().unwrap_or(1).max(32);
            let string_dim = self.string_dim(width)?;
            let mut bytes = Vec::new();
            for value in &texts {
                bytes.extend_from_slice(value.as_bytes());
                bytes.resize(bytes.len() + width - value.len(), 0);
            }
            self.slotted(
                "r_calib_time",
                vec![dim, string_dim],
                Nc3Values::Char(bytes),
                vec![("long_name", text("calibration_time_utc"))],
                Vec::new(),
            );
        }
        let entries: Vec<_> = calibration
            .iter()
            .map(|entry| entry.float_entries())
            .collect();
        for column in 0..entries[0].len() {
            let name = entries[0][column].0;
            if entries.iter().all(|row| row[column].1.is_none()) {
                continue;
            }
            let var_name = match name {
                "base_1km_hc" => "r_calib_base_dbz_1km_hc".to_owned(),
                "base_1km_vc" => "r_calib_base_dbz_1km_vc".to_owned(),
                "base_1km_hx" => "r_calib_base_dbz_1km_hx".to_owned(),
                "base_1km_vx" => "r_calib_base_dbz_1km_vx".to_owned(),
                other => format!("r_calib_{other}"),
            };
            self.slotted(
                &var_name,
                vec![dim],
                Nc3Values::Float(
                    entries
                        .iter()
                        .map(|row| row[column].1.unwrap_or(f32::NAN))
                        .collect(),
                ),
                Vec::new(),
                Vec::new(),
            );
        }
        let mut extra_names: Vec<&str> = Vec::new();
        for entry in calibration {
            for (name, _) in &entry.extra {
                if !extra_names.contains(&&**name) {
                    extra_names.push(name);
                }
            }
        }
        for name in extra_names {
            let values: Vec<f32> = calibration
                .iter()
                .map(|entry| {
                    entry
                        .extra
                        .iter()
                        .find(|(key, _)| &**key == name)
                        .and_then(|(_, value)| value.as_f64())
                        .map_or(f32::NAN, |v| v as f32)
                })
                .collect();
            let var_name = sanitize_name(&format!("r_calib_{name}"));
            self.slotted(
                &var_name,
                vec![dim],
                Nc3Values::Float(values),
                Vec::new(),
                Vec::new(),
            );
        }
        Ok(())
    }

    fn coordinates(&mut self) -> Result<(), CfWriteError> {
        let volume = self.volume;
        let grid = self.grid;
        let mut range_attrs: Vec<(&str, Nc3Values)> = vec![
            ("long_name", text("range_to_center_of_measurement_volume")),
            ("standard_name", text("projection_range_coordinate")),
            ("units", text("meters")),
            ("axis", text("radial_range_coordinate")),
        ];
        let mut owned: Vec<(&str, Nc3Values)> = Vec::new();
        // The rows of the longest sweep give `range(range)`.
        let longest = grid
            .sweeps
            .iter()
            .max_by_key(|sweep| sweep.ngates)
            .cloned()
            .ok_or_else(|| CfWriteError::Unrepresentable("a volume without sweeps".into()))?;
        if grid.per_sweep {
            // CfRadial 1.4 section 4.4: range(sweep, range), the geometry
            // attributes as float(sweep).
            let firsts: Vec<f32> = grid
                .sweeps
                .iter()
                .map(|sweep| sweep.center(0) as f32)
                .collect();
            let spacings: Vec<f32> = grid
                .sweeps
                .iter()
                .map(|sweep| {
                    sweep
                        .uniform
                        .map_or(f32::NAN, |(_, spacing)| spacing as f32)
                })
                .collect();
            owned.push(("spacing_is_constant", text("true")));
            owned.push(("meters_to_center_of_first_gate", Nc3Values::Float(firsts)));
            owned.push(("meters_between_gates", Nc3Values::Float(spacings)));
            let mut centres = Vec::with_capacity(grid.sweeps.len() * grid.ngates());
            for sweep in &grid.sweeps {
                centres.extend((0..grid.ngates()).map(|gate| sweep.center(gate) as f32));
            }
            self.slotted(
                "range",
                vec![self.sweep, self.range],
                Nc3Values::Float(centres),
                range_attrs,
                owned,
            );
            return self.time_and_angles(volume);
        }
        match longest.uniform {
            Some((first, spacing)) => {
                range_attrs.push(("spacing_is_constant", text("true")));
                range_attrs.push((
                    "meters_to_center_of_first_gate",
                    Nc3Values::Float(vec![first as f32]),
                ));
                range_attrs.push((
                    "meters_between_gates",
                    Nc3Values::Float(vec![spacing as f32]),
                ));
                // A source whose range this is not (another sweep's, a
                // refined grid) gets this range's geometry.
                if !self.cfradial1 {
                    owned.push((
                        "meters_to_center_of_first_gate",
                        Nc3Values::Float(vec![first as f32]),
                    ));
                    owned.push((
                        "meters_between_gates",
                        Nc3Values::Float(vec![spacing as f32]),
                    ));
                }
            }
            None => range_attrs.push(("spacing_is_constant", text("false"))),
        }
        self.slotted(
            "range",
            vec![self.range],
            Nc3Values::Float(
                (0..grid.ngates())
                    .map(|gate| longest.center(gate) as f32)
                    .collect(),
            ),
            range_attrs,
            owned,
        );
        self.time_and_angles(volume)
    }

    /// `time`, `azimuth` and `elevation`.
    fn time_and_angles(&mut self, volume: &Volume) -> Result<(), CfWriteError> {
        let units = format!("seconds since {}", time_string(volume.time_reference));
        // The source's own `time:units` stays when it names this reference.
        let keep_units = self
            .source
            .source("time")
            .and_then(|attrs| attrs.iter().find(|(name, _)| &**name == "units"))
            .and_then(|(_, value)| value.as_text())
            .and_then(|units| units.split_once("since"))
            .and_then(|(_, rest)| crate::cfradial::parse_iso_instant(rest))
            .is_some_and(|instant| instant == volume.time_reference);
        let times: Vec<f64> = volume
            .sweeps
            .iter()
            .flat_map(|sweep| sweep.rays.time_s.iter().copied())
            .collect();
        self.slotted(
            "time",
            vec![self.time],
            Nc3Values::Double(times),
            vec![
                ("standard_name", text("time")),
                ("long_name", text("time_in_seconds_since_volume_start")),
                ("units", text(&units)),
                ("calendar", text("gregorian")),
            ],
            if keep_units {
                Vec::new()
            } else {
                vec![("units", text(&units))]
            },
        );
        let azimuth: Vec<f32> = volume
            .sweeps
            .iter()
            .flat_map(|sweep| sweep.rays.azimuth_deg.iter().copied())
            .collect();
        self.slotted(
            "azimuth",
            vec![self.time],
            Nc3Values::Float(azimuth),
            vec![
                ("standard_name", text("ray_azimuth_angle")),
                ("long_name", text("azimuth_angle_from_true_north")),
                ("units", text("degrees")),
                ("axis", text("radial_azimuth_coordinate")),
            ],
            Vec::new(),
        );
        let elevation: Vec<f32> = volume
            .sweeps
            .iter()
            .flat_map(|sweep| sweep.rays.elevation_deg.iter().copied())
            .collect();
        self.slotted(
            "elevation",
            vec![self.time],
            Nc3Values::Float(elevation),
            vec![
                ("standard_name", text("ray_elevation_angle")),
                ("long_name", text("elevation_angle_from_horizontal_plane")),
                ("units", text("degrees")),
                ("axis", text("radial_elevation_coordinate")),
                ("positive", text("up")),
            ],
            Vec::new(),
        );
        Ok(())
    }

    /// A float `(time)` variable from each sweep's values, `fill` where a
    /// sweep has none; `None` when no sweep has any.
    fn ray_floats(&self, get: impl Fn(&Sweep) -> Option<&[f32]>, fill: f32) -> Option<Vec<f32>> {
        let sweeps = &self.volume.sweeps;
        if sweeps.iter().all(|sweep| get(sweep).is_none()) {
            return None;
        }
        let mut out = Vec::new();
        for sweep in sweeps {
            match get(sweep) {
                Some(values) => out.extend_from_slice(values),
                None => out.extend(std::iter::repeat_n(fill, sweep.nrays())),
            }
        }
        Some(out)
    }

    fn ray_variables(&mut self) -> Result<(), CfWriteError> {
        let volume = self.volume;
        let parameters = &volume.radar_parameters;
        let time = self.time;
        let floats: [(&str, &RayF32, Option<f32>, &str); 9] = [
            (
                "nyquist_velocity",
                &|s| s.ray_vars.nyquist_velocity_mps.as_deref(),
                None,
                "meters per second",
            ),
            (
                "unambiguous_range",
                &|s| s.ray_vars.unambiguous_range_m.as_deref(),
                parameters.unambiguous_range_m,
                "meters",
            ),
            (
                "prt",
                &|s| s.ray_vars.prt_s.as_deref(),
                parameters.prt_s,
                "seconds",
            ),
            ("prt_ratio", &|s| s.ray_vars.prt_ratio.as_deref(), None, ""),
            (
                "pulse_width",
                &|s| s.ray_vars.pulse_width_s.as_deref(),
                parameters.pulse_width_s,
                "seconds",
            ),
            (
                "scan_rate",
                &|s| s.ray_vars.scan_rate_deg_per_s.as_deref(),
                None,
                "degrees per second",
            ),
            (
                "independent_samples",
                &|s| s.ray_vars.independent_samples.as_deref(),
                None,
                "",
            ),
            (
                "measured_transmit_power_h",
                &|s| {
                    s.monitoring
                        .as_ref()
                        .and_then(|m| m.radar_measured_transmit_power_h_dbm.as_deref())
                },
                None,
                "dBm",
            ),
            (
                "measured_transmit_power_v",
                &|s| {
                    s.monitoring
                        .as_ref()
                        .and_then(|m| m.radar_measured_transmit_power_v_dbm.as_deref())
                },
                None,
                "dBm",
            ),
        ];
        for (name, get, constant, units) in floats {
            let values = self.ray_floats(get, constant.unwrap_or(f32::NAN));
            let values = match (values, constant) {
                (Some(values), _) => values,
                // A volume constant no sweep has per-ray values for.
                (None, Some(constant)) if !self.cfradial1 => {
                    vec![constant; volume.sweeps.iter().map(Sweep::nrays).sum()]
                }
                _ => continue,
            };
            let mut defaults = Vec::new();
            if !units.is_empty() {
                defaults.push(("units", text(units)));
            }
            self.slotted(
                name,
                vec![time],
                Nc3Values::Float(values),
                defaults,
                Vec::new(),
            );
        }
        let ints: [(&str, &RayI32); 2] = [
            ("n_samples", &|s| s.ray_vars.n_samples.as_deref()),
            ("r_calib_index", &|s| s.ray_vars.calib_index.as_deref()),
        ];
        for (name, get) in ints {
            let sweeps = &volume.sweeps;
            if sweeps.iter().all(|sweep| get(sweep).is_none()) {
                continue;
            }
            let mut values = Vec::new();
            for sweep in sweeps {
                match get(sweep) {
                    Some(v) => values.extend_from_slice(v),
                    None => values.extend(std::iter::repeat_n(INT_FILL, sweep.nrays())),
                }
            }
            self.slotted(
                name,
                vec![time],
                Nc3Values::Int(values),
                vec![("_FillValue", Nc3Values::Int(vec![INT_FILL]))],
                Vec::new(),
            );
        }
        if volume
            .sweeps
            .iter()
            .any(|sweep| sweep.ray_vars.antenna_transition.is_some())
        {
            let mut values = Vec::new();
            for sweep in &volume.sweeps {
                match &sweep.ray_vars.antenna_transition {
                    Some(v) => values.extend(v.iter().map(|x| *x as i8)),
                    None => values.extend(std::iter::repeat_n(0, sweep.nrays())),
                }
            }
            self.slotted(
                "antenna_transition",
                vec![time],
                Nc3Values::Byte(values),
                vec![(
                    "comment",
                    text("1 if antenna is in transition, 0 otherwise"),
                )],
                Vec::new(),
            );
        }
        // Per-ray variables without a CfRadial 1 name (Table 301-8a and
        // Table 301-11), under their FM301 names.
        let extra_floats: [(&str, &RayF32); 1] = [("rx_range_resolution", &|s| {
            s.ray_vars.rx_range_resolution_m.as_deref()
        })];
        for (name, get) in extra_floats {
            if let Some(values) = self.ray_floats(get, f32::NAN) {
                self.pending.extras.push(Nc3Variable {
                    name: name.to_owned(),
                    dims: vec![time],
                    attrs: vec![("units".to_owned(), text("meters"))],
                    values: Nc3Values::Float(values),
                });
            }
        }
        let monitoring_names: Vec<&'static str> = {
            let mut names = Vec::new();
            for sweep in &volume.sweeps {
                if let Some(monitoring) = &sweep.monitoring {
                    for (name, _) in monitoring.variables() {
                        if !name.starts_with("radar_measured_transmit_power")
                            && !names.contains(&name)
                        {
                            names.push(name);
                        }
                    }
                }
            }
            names
        };
        for name in monitoring_names {
            let values = self.ray_floats(
                |sweep| {
                    sweep.monitoring.as_ref().and_then(|monitoring| {
                        monitoring
                            .variables()
                            .into_iter()
                            .find(|(key, _)| *key == name)
                            .map(|(_, values)| values)
                    })
                },
                f32::NAN,
            );
            if let Some(values) = values {
                self.pending.extras.push(Nc3Variable {
                    name: name.to_owned(),
                    dims: vec![time],
                    attrs: Vec::new(),
                    values: Nc3Values::Float(values),
                });
            }
        }
        self.platform_track()?;
        Ok(())
    }

    /// `ray_n_gates` and `ray_start_index` of `n_gates_vary` storage, and
    /// `ray_start_range` and `ray_gate_spacing` whenever the rays' geometry
    /// is not the one `range(range)` states (or a kept copy of them
    /// disagrees with the geometry written).
    fn gate_geometry(&mut self) {
        let volume = self.volume;
        let grid = self.grid;
        let time = self.time;
        let int_attrs = |long_name: &str| {
            vec![
                ("long_name", text(long_name)),
                ("units", text("")),
                ("_FillValue", Nc3Values::Int(vec![INT_FILL])),
            ]
        };
        if grid.ragged {
            let mut counts = Vec::new();
            let mut starts = Vec::new();
            let mut start = 0usize;
            for (index, sweep) in volume.sweeps.iter().enumerate() {
                let len = grid.row_len(index);
                for _ in 0..sweep.nrays() {
                    counts.push(len as i32);
                    starts.push(start as i32);
                    start += len;
                }
            }
            self.slotted(
                "ray_n_gates",
                vec![time],
                Nc3Values::Int(counts),
                Vec::new(),
                int_attrs("number_of_gates"),
            );
            self.slotted(
                "ray_start_index",
                vec![time],
                Nc3Values::Int(starts),
                Vec::new(),
                int_attrs("array_index_to_start_of_ray"),
            );
        }
        let mut starts = Vec::new();
        let mut spacings = Vec::new();
        for (index, sweep) in volume.sweeps.iter().enumerate() {
            let row = &grid.sweeps[index];
            let spacing = row
                .uniform
                .map_or(row.center(1) - row.center(0), |(_, spacing)| spacing);
            starts.extend(std::iter::repeat_n(row.center(0) as f32, sweep.nrays()));
            spacings.extend(std::iter::repeat_n(spacing as f32, sweep.nrays()));
        }
        let kept = |name: &str| {
            volume
                .sweeps
                .iter()
                .any(|sweep| sweep.extra_vars.iter().any(|extra| &*extra.name == name))
        };
        let disagree = |name: &str, values: &[f32]| {
            let mut ray = 0usize;
            volume.sweeps.iter().any(|sweep| {
                let expected = &values[ray..ray + sweep.nrays()];
                ray += sweep.nrays();
                sweep
                    .extra_vars
                    .iter()
                    .find(|extra| &*extra.name == name)
                    .is_some_and(|extra| {
                        (0..sweep.nrays()).any(|row| {
                            extra.values.get_f64(row).is_some_and(|value| {
                                value.is_finite()
                                    && value > -9999.0
                                    && (value - f64::from(expected[row])).abs() > 1e-3
                            })
                        })
                    })
            })
        };
        let needed = grid.ragged
            || grid.per_sweep
            || grid.per_ray
            || (kept("ray_start_range") && disagree("ray_start_range", &starts))
            || (kept("ray_gate_spacing") && disagree("ray_gate_spacing", &spacings));
        if needed {
            for (name, long_name, values) in [
                ("ray_start_range", "start_range_for_ray", starts),
                ("ray_gate_spacing", "gate_spacing_for_ray", spacings),
            ] {
                self.slotted(
                    name,
                    vec![time],
                    Nc3Values::Float(values),
                    Vec::new(),
                    vec![
                        ("long_name", text(long_name)),
                        ("units", text("meters")),
                        ("_FillValue", Nc3Values::Float(vec![FLOAT_FILL])),
                    ],
                );
            }
        }
    }

    fn platform_track(&mut self) -> Result<(), CfWriteError> {
        let sweeps = &self.volume.sweeps;
        if !sweeps.iter().all(|sweep| sweep.platform_track.is_some()) {
            return Ok(());
        }
        let tracks: Vec<&recast_radar_core::model::PlatformTrack> = sweeps
            .iter()
            .filter_map(|sweep| sweep.platform_track.as_deref())
            .collect();
        let time = self.time;
        for (name, get, units) in [
            (
                "latitude",
                &(|t: &recast_radar_core::model::PlatformTrack| Some(t.latitude_deg.clone()))
                    as &dyn Fn(&recast_radar_core::model::PlatformTrack) -> Option<Vec<f64>>,
                "degrees_north",
            ),
            (
                "longitude",
                &|t| Some(t.longitude_deg.clone()),
                "degrees_east",
            ),
            ("altitude", &|t| Some(t.altitude_m.clone()), "meters"),
            ("altitude_agl", &|t| t.altitude_agl_m.clone(), "meters"),
        ] {
            let values: Option<Vec<Vec<f64>>> = tracks.iter().map(|t| get(t)).collect();
            if let Some(values) = values {
                self.slotted(
                    name,
                    vec![time],
                    Nc3Values::Double(values.concat()),
                    vec![("units", text(units))],
                    Vec::new(),
                );
            }
        }
        type Angles = fn(&recast_radar_core::model::PlatformTrack) -> Option<&Vec<f32>>;
        let angles: [(&str, Angles); 6] = [
            ("heading", |t| t.heading_deg.as_ref()),
            ("roll", |t| t.roll_deg.as_ref()),
            ("pitch", |t| t.pitch_deg.as_ref()),
            ("drift", |t| t.drift_deg.as_ref()),
            ("rotation", |t| t.rotation_deg.as_ref()),
            ("tilt", |t| t.tilt_deg.as_ref()),
        ];
        for (name, get) in angles {
            let values: Option<Vec<f32>> = tracks
                .iter()
                .map(|t| get(t).cloned())
                .collect::<Option<Vec<Vec<f32>>>>()
                .map(|v| v.concat());
            if let Some(values) = values {
                self.slotted(
                    name,
                    vec![time],
                    Nc3Values::Float(values),
                    vec![("units", text("degrees"))],
                    Vec::new(),
                );
            }
        }
        Ok(())
    }

    fn fields(&mut self) -> Result<(), CfWriteError> {
        let volume = self.volume;
        let mut names: Vec<&str> = Vec::new();
        for sweep in &volume.sweeps {
            for field in &sweep.fields {
                if !names.contains(&field.name.as_str()) {
                    names.push(field.name.as_str());
                }
            }
        }
        for name in names {
            let instances: Vec<Option<&Field>> = volume
                .sweeps
                .iter()
                .map(|sweep| sweep.fields.iter().find(|f| f.name.as_str() == name))
                .collect();
            let first = instances
                .iter()
                .flatten()
                .next()
                .copied()
                .ok_or_else(|| CfWriteError::Invalid(format!("field {name} vanished")))?;
            let coding = first.data.coding();
            let same = instances
                .iter()
                .flatten()
                .all(|field| field.data.coding() == coding);
            let var = if same {
                self.packed_field(&instances, first)?
            } else {
                self.float_field(&instances, first)
            };
            self.pending.fields.push(var);
        }
        Ok(())
    }

    /// Whether any gate of the output lacks a source value (a sweep without
    /// the field, an absent row, padding).
    fn needs_fill(&self, instances: &[Option<&Field>]) -> bool {
        let grid = self.grid;
        instances
            .iter()
            .enumerate()
            .any(|(sweep, field)| match field {
                None => true,
                Some(field) => {
                    let mapping = grid.mapping(sweep, field);
                    !field.absent_rows.is_empty()
                        || mapping.start > 0
                        || mapping.end(field.ngates).unwrap_or(0) < grid.row_len(sweep) as u64
                }
            })
    }

    fn packed_field(
        &mut self,
        instances: &[Option<&Field>],
        first: &Field,
    ) -> Result<Nc3Variable, CfWriteError> {
        let needs_fill = self.needs_fill(instances);
        macro_rules! int_field {
            ($variant:ident, $t:ty, $coding:expr) => {{
                let coding: IntCoding<$t> = *$coding;
                let fill = match coding.fill_value {
                    Some(fill) => Some(fill),
                    None if needs_fill => {
                        let used: HashSet<$t> = instances
                            .iter()
                            .flatten()
                            .flat_map(|field| match &field.data {
                                FieldData::$variant { values, .. } => values.clone(),
                                _ => Vec::new(),
                            })
                            .collect();
                        let taken = [coding.undetect, coding.range_folded];
                        let free = <$t as Code>::candidates()
                            .find(|code| !used.contains(code) && !taken.contains(&Some(*code)));
                        Some(free.ok_or_else(|| {
                            CfWriteError::Unrepresentable(format!(
                                "field {}: no free code for _FillValue",
                                first.name
                            ))
                        })?)
                    }
                    None => None,
                };
                let pad = fill.unwrap_or(<$t>::from_u8(0));
                let values = self.materialize(instances, pad, |field| match &field.data {
                    FieldData::$variant { values, .. } => Some(values.as_slice()),
                    _ => None,
                });
                let coding = IntCoding {
                    fill_value: fill,
                    ..coding
                };
                let unsigned_attribute = self.unsigned_attribute;
                (
                    <$t as Code>::classic(values, unsigned_attribute),
                    int_attrs(&coding, unsigned_attribute),
                )
            }};
        }
        let (values, packing) = match &first.data {
            FieldData::U8 { coding, .. } => int_field!(U8, u8, coding),
            FieldData::U16 { coding, .. } => int_field!(U16, u16, coding),
            FieldData::I8 { coding, .. } => int_field!(I8, i8, coding),
            FieldData::I16 { coding, .. } => int_field!(I16, i16, coding),
            FieldData::I32 { coding, .. } => int_field!(I32, i32, coding),
            FieldData::F32 { coding, .. } => {
                let pad = coding.fill_code();
                let values = self.materialize(instances, pad, |field| match &field.data {
                    FieldData::F32 { values, .. } => Some(values.as_slice()),
                    _ => None,
                });
                let mut attrs = Vec::new();
                if let Some(transform) = coding.transform {
                    attrs.extend(scale_attrs(transform));
                }
                if let Some(fill) = coding.fill_value {
                    attrs.push(("_FillValue".to_owned(), Nc3Values::Float(vec![fill])));
                }
                if let Some(undetect) = coding.undetect {
                    attrs.push(("_Undetect".to_owned(), Nc3Values::Float(vec![undetect])));
                }
                (Nc3Values::Float(values), attrs)
            }
            FieldData::F64 { coding, .. } => {
                let pad = coding.fill_code();
                let values = self.materialize(instances, pad, |field| match &field.data {
                    FieldData::F64 { values, .. } => Some(values.as_slice()),
                    _ => None,
                });
                let mut attrs = Vec::new();
                if let Some(transform) = coding.transform {
                    attrs.extend(scale_attrs(transform));
                }
                if let Some(fill) = coding.fill_value {
                    attrs.push(("_FillValue".to_owned(), Nc3Values::Double(vec![fill])));
                }
                if let Some(undetect) = coding.undetect {
                    attrs.push(("_Undetect".to_owned(), Nc3Values::Double(vec![undetect])));
                }
                (Nc3Values::Double(values), attrs)
            }
        };
        let flags = match &first.data {
            FieldData::U8 { coding, .. } if self.unsigned_attribute => {
                flag_attrs(first, coding.range_folded, Nc3Values::Byte, |v| v as i8)
            }
            FieldData::U8 { coding, .. } => {
                flag_attrs(first, coding.range_folded, Nc3Values::Short, |v| v as i16)
            }
            FieldData::U16 { coding, .. } if self.unsigned_attribute => {
                flag_attrs(first, coding.range_folded, Nc3Values::Short, |v| v as i16)
            }
            FieldData::U16 { coding, .. } => {
                flag_attrs(first, coding.range_folded, Nc3Values::Int, |v| v as i32)
            }
            FieldData::I8 { coding, .. } => {
                flag_attrs(first, coding.range_folded, Nc3Values::Byte, |v| v as i8)
            }
            FieldData::I16 { coding, .. } => {
                flag_attrs(first, coding.range_folded, Nc3Values::Short, |v| v as i16)
            }
            FieldData::I32 { coding, .. } => {
                flag_attrs(first, coding.range_folded, Nc3Values::Int, |v| v as i32)
            }
            FieldData::F32 { .. } => {
                flag_attrs::<f32, f32>(first, None, Nc3Values::Float, |v| v as f32)
            }
            FieldData::F64 { .. } => {
                flag_attrs::<f64, f64>(first, None, Nc3Values::Double, |v| v as f64)
            }
        };
        Ok(self.field_variable(first, values, packing, flags))
    }

    /// A field whose coding differs between sweeps, as float physical
    /// values (module docs).
    fn float_field(&mut self, instances: &[Option<&Field>], first: &Field) -> Nc3Variable {
        let grid = self.grid;
        let mut values = Vec::new();
        for (sweep_index, (sweep, field)) in self.volume.sweeps.iter().zip(instances).enumerate() {
            let ngates = grid.row_len(sweep_index);
            for row in 0..sweep.nrays() {
                let start = values.len();
                values.resize(start + ngates, FLOAT_FILL);
                let Some(field) = field else {
                    continue;
                };
                let mapping = grid.mapping(sweep_index, field);
                let dest = &mut values[start..start + ngates];
                for gate in 0..field.ngates as usize {
                    let value = match field.gate(row, gate) {
                        Some(Gate::Value(value)) if value.is_finite() => value,
                        _ => FLOAT_FILL,
                    };
                    let from = mapping.start as usize + gate * mapping.stride as usize;
                    if from >= ngates {
                        break;
                    }
                    let to = (from + mapping.stride as usize).min(ngates);
                    dest[from..to].fill(value);
                }
            }
        }
        let packing = vec![("_FillValue".to_owned(), Nc3Values::Float(vec![FLOAT_FILL]))];
        let mut flag_first = first.clone();
        flag_first.attrs.flag_values.clear();
        flag_first.attrs.flag_masks.clear();
        flag_first.attrs.flag_meanings.clear();
        self.field_variable(&flag_first, Nc3Values::Float(values), packing, Vec::new())
    }

    /// `(time, range)` values of every sweep, each field's rows mapped onto
    /// the grid, `fill` elsewhere.
    fn materialize<T: Copy>(
        &self,
        instances: &[Option<&Field>],
        fill: T,
        values_of: impl Fn(&Field) -> Option<&[T]>,
    ) -> Vec<T> {
        let grid = self.grid;
        let mut out = vec![fill; grid.cells(self.volume)];
        let mut start = 0usize;
        for (sweep_index, (sweep, field)) in self.volume.sweeps.iter().zip(instances).enumerate() {
            let ngates = grid.row_len(sweep_index);
            let row0 = start;
            start += sweep.nrays() * ngates;
            if let Some(field) = field
                && let Some(values) = values_of(field)
            {
                let mapping = grid.mapping(sweep_index, field);
                let native = field.ngates as usize;
                let stride = mapping.stride.max(1) as usize;
                for row in 0..sweep.nrays() {
                    if field.is_absent(row) {
                        continue;
                    }
                    let Some(source) = values.get(row * native..(row + 1) * native) else {
                        continue;
                    };
                    let dest = &mut out[row0 + row * ngates..row0 + (row + 1) * ngates];
                    for (gate, value) in source.iter().enumerate() {
                        let from = mapping.start as usize + gate * stride;
                        if from >= ngates {
                            break;
                        }
                        dest[from..(from + stride).min(ngates)].fill(*value);
                    }
                }
            }
        }
        out
    }

    fn field_variable(
        &self,
        field: &Field,
        values: Nc3Values,
        packing: Vec<(String, Nc3Values)>,
        flags: Vec<(String, Nc3Values)>,
    ) -> Nc3Variable {
        let model = &field.attrs;
        let info = field.name.info();
        let mut attrs: Vec<(String, Nc3Values)> = Vec::new();
        let foreign = !matches!(
            self.volume.provenance.source_format,
            SourceFormat::CfRadial1 | SourceFormat::CfRadial2
        );
        for (name, model_value, table_value) in [
            (
                "standard_name",
                &model.standard_name,
                info.and_then(|info| info.standard_name),
            ),
            (
                "long_name",
                &model.long_name,
                info.map(|info| info.long_name),
            ),
            ("units", &model.units, info.map(|info| info.units)),
        ] {
            let value = model_value
                .as_deref()
                .or(if foreign { table_value } else { None });
            if let Some(value) = value {
                attrs.push((name.to_owned(), text(value)));
            }
        }
        attrs.extend(packing);
        attrs.extend(flags);
        if let Some(ratio) = model.sampling_ratio {
            attrs.push(("sampling_ratio".to_owned(), Nc3Values::Float(vec![ratio])));
        }
        for (name, value) in [
            ("is_discrete", model.is_discrete),
            ("field_folds", model.field_folds),
            ("is_quality_field", model.is_quality_field),
        ] {
            if let Some(value) = value {
                attrs.push((name.to_owned(), text(if value { "true" } else { "false" })));
            }
        }
        for (name, value) in [
            ("fold_limit_lower", model.fold_limit_lower),
            ("fold_limit_upper", model.fold_limit_upper),
        ] {
            if let Some(value) = value {
                attrs.push((name.to_owned(), Nc3Values::Float(vec![value])));
            }
        }
        for (name, list) in [
            ("qualified_variables", &model.qualified_variables),
            ("ancillary_variables", &model.ancillary_variables),
        ] {
            if !list.is_empty() {
                let joined: Vec<String> = list.iter().map(|n| sanitize_name(n.as_str())).collect();
                attrs.push((name.to_owned(), text(&joined.join(" "))));
            }
        }
        if let Some(xml) = &model.thresholding_xml {
            attrs.push(("thresholding_xml".to_owned(), text(xml)));
        }
        if foreign && !model.other.iter().any(|(name, _)| &**name == "coordinates") {
            attrs.push(("coordinates".to_owned(), text("elevation azimuth range")));
        }
        for (name, value) in &model.other {
            attrs.push((sanitize_name(name), attr_values(value)));
        }
        dedupe(&mut attrs);
        Nc3Variable {
            name: sanitize_name(field.name.as_str()),
            dims: match self.n_points {
                Some(points) => vec![points],
                None => vec![self.time, self.range],
            },
            attrs,
            values,
        }
    }

    /// A CfRadial 1 volume's variables that hold only fill values: no typed
    /// slot took a value, but their attributes are kept
    /// (`Volume::variable_attrs`). They are written back, full of their
    /// `_FillValue` (else NaN, or -9999 for the sweep table), so the file
    /// reads back unchanged.
    fn restore_empty_variables(&mut self) {
        const ROOT_SCALARS: &[&str] = &[
            "radar_antenna_gain_h",
            "radar_antenna_gain_v",
            "radar_beam_width_h",
            "radar_beam_width_v",
            "radar_rx_bandwidth",
            "radar_receiver_bandwidth",
            "altitude_agl",
            "volume_number",
        ];
        const SWEEP_VARS: &[&str] = &["target_scan_rate", "ray_angle_res"];
        let volume = self.volume;
        let r_calib = self.nc.dim("r_calib");
        let written = |name: &str, pending: &Pending| {
            pending.slotted.iter().any(|(_, var)| var.name == name)
                || pending.fields.iter().any(|var| var.name == name)
                || pending.extras.iter().any(|var| var.name == name)
        };
        for (position, entry) in volume.variable_attrs.iter().enumerate() {
            if !entry.group.is_empty() || written(&entry.name, &self.pending) {
                continue;
            }
            let name: &str = &entry.name;
            let (dims, count) = if ROOT_SCALARS.contains(&name) {
                (Vec::new(), 1)
            } else if SWEEP_VARS.contains(&name) {
                (vec![self.sweep], volume.sweeps.len())
            } else if let Some(dim) = r_calib.filter(|_| name.starts_with("r_calib_")) {
                (vec![dim], volume.radar_calibration.len())
            } else {
                continue;
            };
            let fill = entry
                .attrs
                .iter()
                .find(|(key, _)| &**key == "_FillValue")
                .map(|(_, value)| attr_values(value));
            let values = match fill {
                Some(Nc3Values::Float(v)) if v.len() == 1 => Nc3Values::Float(vec![v[0]; count]),
                Some(Nc3Values::Double(v)) if v.len() == 1 => Nc3Values::Double(vec![v[0]; count]),
                Some(Nc3Values::Int(v)) if v.len() == 1 => Nc3Values::Int(vec![v[0]; count]),
                Some(Nc3Values::Short(v)) if v.len() == 1 => Nc3Values::Short(vec![v[0]; count]),
                Some(Nc3Values::Byte(v)) if v.len() == 1 => Nc3Values::Byte(vec![v[0]; count]),
                _ if SWEEP_VARS.contains(&name) => Nc3Values::Float(vec![FLOAT_FILL; count]),
                _ => Nc3Values::Float(vec![f32::NAN; count]),
            };
            let mut attrs: Vec<(String, Nc3Values)> = entry
                .attrs
                .iter()
                .map(|(key, value)| (sanitize_name(key), attr_values(value)))
                .collect();
            dedupe(&mut attrs);
            self.pending.slotted.push((
                position,
                Nc3Variable {
                    name: sanitize_name(name),
                    dims,
                    attrs,
                    values,
                },
            ));
        }
    }

    /// Variables the volume keeps verbatim: the root's, and each sweep's
    /// per-ray and per-sweep ones recombined over `time` and `sweep`.
    fn extras(&mut self) -> Result<(), CfWriteError> {
        let volume = self.volume;
        for extra in &volume.extra_vars {
            if let Some(var) = self.extra_variable(extra, None)? {
                self.pending.extras.push(var);
            }
        }
        let mut names: Vec<&str> = Vec::new();
        for sweep in &volume.sweeps {
            for extra in &sweep.extra_vars {
                if !names.contains(&&*extra.name) {
                    names.push(&extra.name);
                }
            }
        }
        for name in names {
            // A source's `n_gates_vary` locators describe its own layout.
            if matches!(name, "ray_n_gates" | "ray_start_index") {
                continue;
            }
            let instances: Vec<Option<&ExtraVariable>> = volume
                .sweeps
                .iter()
                .map(|sweep| sweep.extra_vars.iter().find(|e| &*e.name == name))
                .collect();
            let Some(first) = instances.iter().flatten().next().copied() else {
                continue;
            };
            let per_ray = first.is_per_ray();
            if instances.iter().flatten().any(|extra| {
                extra.is_per_ray() != per_ray || extra.values.dtype() != first.values.dtype()
            }) {
                continue;
            }
            // One array over `time` (per-ray) or `sweep` (per-sweep).
            let row_shape: Vec<u32> = if per_ray {
                first.shape.iter().skip(1).copied().collect()
            } else {
                first.shape.clone()
            };
            let row_len: usize = row_shape
                .iter()
                .map(|n| *n as usize)
                .product::<usize>()
                .max(1);
            let mut parts: Vec<ArrayBuf> = Vec::new();
            for (sweep, extra) in volume.sweeps.iter().zip(&instances) {
                let rows = if per_ray { sweep.nrays() } else { 1 };
                match extra {
                    Some(extra)
                        if extra.values.len() == rows * row_len
                            && (if per_ray {
                                &extra.shape[1..]
                            } else {
                                &extra.shape[..]
                            }) == row_shape.as_slice() =>
                    {
                        parts.push(extra.values.clone());
                    }
                    _ => parts.push(filled(&first.values, rows * row_len)),
                }
            }
            let values = concat(parts);
            let (lead, lead_len) = if per_ray {
                (
                    "time",
                    volume.sweeps.iter().map(Sweep::nrays).sum::<usize>(),
                )
            } else {
                ("sweep", volume.sweeps.len())
            };
            let mut dims: Vec<Box<str>> = vec![lead.into()];
            let mut shape = vec![lead_len as u32];
            let inner_dims = if per_ray {
                &first.dims[1..]
            } else {
                &first.dims[..]
            };
            dims.extend(inner_dims.iter().cloned());
            shape.extend(row_shape);
            let combined = ExtraVariable {
                name: first.name.clone(),
                dims,
                shape,
                values,
                attrs: first.attrs.clone(),
            };
            if let Some(var) = self.extra_variable(&combined, Some(lead))? {
                self.pending.extras.push(var);
            }
        }
        Ok(())
    }

    /// One kept variable. `lead` names its first dimension when it is the
    /// writer's `time` or `sweep`.
    fn extra_variable(
        &mut self,
        extra: &ExtraVariable,
        lead: Option<&str>,
    ) -> Result<Option<Nc3Variable>, CfWriteError> {
        let name = sanitize_name(&extra.name);
        if self.pending.slotted.iter().any(|(_, var)| var.name == name)
            || self.pending.fields.iter().any(|var| var.name == name)
            || self.pending.extras.iter().any(|var| var.name == name)
        {
            return Ok(None);
        }
        let mut dims = Vec::with_capacity(extra.dims.len() + 1);
        for (axis, (dim, len)) in extra.dims.iter().zip(&extra.shape).enumerate() {
            let id = match (axis, lead) {
                (0, Some("time")) => self.time,
                (0, Some("sweep")) => self.sweep,
                _ => {
                    let dim_name = sanitize_name(dim);
                    match self.nc.dim(&dim_name) {
                        Some(id) if self.nc.dim_len(id) == Some(u64::from(*len)) => id,
                        Some(_) => {
                            let renamed = format!("{dim_name}_{len}");
                            match self.nc.dim(&renamed) {
                                Some(id) => id,
                                None => self.nc.add_dim(&renamed, u64::from(*len))?,
                            }
                        }
                        None => {
                            if *len == 0 {
                                return Ok(None);
                            }
                            self.nc.add_dim(&dim_name, u64::from(*len))?
                        }
                    }
                }
            };
            dims.push(id);
        }
        let values = match &extra.values {
            ArrayBuf::Text(texts) => {
                let width = texts.iter().map(|t| t.len()).max().unwrap_or(1).max(1);
                dims.push(self.string_dim(width)?);
                let mut bytes = Vec::with_capacity(texts.len() * width);
                for value in texts {
                    bytes.extend_from_slice(value.as_bytes());
                    bytes.resize(bytes.len() + width - value.len(), 0);
                }
                Nc3Values::Char(bytes)
            }
            other => array_values(other),
        };
        let mut attrs: Vec<(String, Nc3Values)> = extra
            .attrs
            .iter()
            .map(|(name, value)| (sanitize_name(name), attr_values(value)))
            .collect();
        dedupe(&mut attrs);
        Ok(Some(Nc3Variable {
            name,
            dims,
            attrs,
            values,
        }))
    }
}

/// `len` fill values of `like`'s type (NaN, -9999 or empty text).
fn filled(like: &ArrayBuf, len: usize) -> ArrayBuf {
    match like {
        ArrayBuf::I8(_) => ArrayBuf::I8(vec![i8::MIN; len]),
        ArrayBuf::U8(_) => ArrayBuf::U8(vec![0; len]),
        ArrayBuf::I16(_) => ArrayBuf::I16(vec![-9999; len]),
        ArrayBuf::U16(_) => ArrayBuf::U16(vec![0; len]),
        ArrayBuf::I32(_) => ArrayBuf::I32(vec![INT_FILL; len]),
        ArrayBuf::U32(_) => ArrayBuf::U32(vec![0; len]),
        ArrayBuf::I64(_) => ArrayBuf::I64(vec![-9999; len]),
        ArrayBuf::F32(_) => ArrayBuf::F32(vec![f32::NAN; len]),
        ArrayBuf::F64(_) => ArrayBuf::F64(vec![f64::NAN; len]),
        ArrayBuf::Text(_) => ArrayBuf::Text(vec!["".into(); len]),
    }
}

/// Arrays of one type joined.
fn concat(parts: Vec<ArrayBuf>) -> ArrayBuf {
    let mut iter = parts.into_iter();
    let Some(mut out) = iter.next() else {
        return ArrayBuf::F64(Vec::new());
    };
    for part in iter {
        match (&mut out, part) {
            (ArrayBuf::I8(a), ArrayBuf::I8(b)) => a.extend(b),
            (ArrayBuf::U8(a), ArrayBuf::U8(b)) => a.extend(b),
            (ArrayBuf::I16(a), ArrayBuf::I16(b)) => a.extend(b),
            (ArrayBuf::U16(a), ArrayBuf::U16(b)) => a.extend(b),
            (ArrayBuf::I32(a), ArrayBuf::I32(b)) => a.extend(b),
            (ArrayBuf::U32(a), ArrayBuf::U32(b)) => a.extend(b),
            (ArrayBuf::I64(a), ArrayBuf::I64(b)) => a.extend(b),
            (ArrayBuf::F32(a), ArrayBuf::F32(b)) => a.extend(b),
            (ArrayBuf::F64(a), ArrayBuf::F64(b)) => a.extend(b),
            (ArrayBuf::Text(a), ArrayBuf::Text(b)) => a.extend(b),
            _ => {}
        }
    }
    out
}

/// `scale_factor` and `add_offset` in the width the source wrote them.
fn scale_attrs(transform: LinearTransform) -> Vec<(String, Nc3Values)> {
    // `write_cfradial1` refuses the transforms without a scale and offset.
    let (scale, offset) = (
        transform.scale_factor().unwrap_or(1.0),
        transform.add_offset().unwrap_or(0.0),
    );
    let (scale, offset) = match transform.attr_width() {
        FloatWidth::F32 => (
            Nc3Values::Float(vec![scale as f32]),
            Nc3Values::Float(vec![offset as f32]),
        ),
        FloatWidth::F64 => (
            Nc3Values::Double(vec![scale]),
            Nc3Values::Double(vec![offset]),
        ),
    };
    vec![
        ("scale_factor".to_owned(), scale),
        ("add_offset".to_owned(), offset),
    ]
}

/// Packing attributes of an integer coding, in its classic type
/// (`_Unsigned` for `u8`/`u16` with `unsigned_attribute`).
fn int_attrs<T: Code>(coding: &IntCoding<T>, unsigned_attribute: bool) -> Vec<(String, Nc3Values)> {
    let mut attrs = Vec::new();
    let unsigned = unsigned_attribute && matches!(T::DTYPE, "uint8" | "uint16");
    let transform = coding.transform;
    let identity = matches!(
        transform,
        LinearTransform::CfScaleOffset { scale_factor, add_offset, attr_width: FloatWidth::F64 }
            if scale_factor == 1.0 && add_offset == 0.0
    );
    if !identity {
        attrs.extend(scale_attrs(transform));
    }
    if unsigned {
        attrs.push(("_Unsigned".to_owned(), text("true")));
    }
    if let Some(fill) = coding.fill_value {
        attrs.push((
            "_FillValue".to_owned(),
            T::classic(vec![fill], unsigned_attribute),
        ));
    }
    if let Some(undetect) = coding.undetect {
        attrs.push((
            "_Undetect".to_owned(),
            T::classic(vec![undetect], unsigned_attribute),
        ));
    }
    if let Some([lo, hi]) = coding.valid_range {
        attrs.push((
            "valid_range".to_owned(),
            T::classic(vec![lo, hi], unsigned_attribute),
        ));
    }
    attrs
}

/// `flag_values` (range-folded code first), `flag_masks` and
/// `flag_meanings` in the field's classic type.
fn flag_attrs<T: Copy + Into<f64>, C>(
    field: &Field,
    range_folded: Option<T>,
    classic: fn(Vec<C>) -> Nc3Values,
    from_i64: fn(i64) -> C,
) -> Vec<(String, Nc3Values)> {
    let model = &field.attrs;
    let mut values: Vec<C> = Vec::new();
    let mut meanings: Vec<&str> = Vec::new();
    if let Some(folded) = range_folded {
        values.push(from_i64(folded.into() as i64));
        meanings.push("range_folded");
    }
    values.extend(model.flag_values.iter().map(|v| from_i64(*v)));
    meanings.extend(model.flag_meanings.iter().map(|m| &**m));
    let mut attrs = Vec::new();
    if !values.is_empty() {
        attrs.push(("flag_values".to_owned(), classic(values)));
    }
    if !model.flag_masks.is_empty() {
        attrs.push((
            "flag_masks".to_owned(),
            classic(model.flag_masks.iter().map(|v| from_i64(*v)).collect()),
        ));
    }
    if !meanings.is_empty() {
        attrs.push(("flag_meanings".to_owned(), text(&meanings.join(" "))));
    }
    attrs
}
