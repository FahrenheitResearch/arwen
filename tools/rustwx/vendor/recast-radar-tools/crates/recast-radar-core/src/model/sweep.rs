//! `sweep_<n>` groups: ray coordinates, the range coordinate, per-ray
//! instrument variables and fields (`docs/design/fm301-model.md` sections 3,
//! 6, 9 and 10).

use std::collections::HashSet;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::field::{Field, FieldData, FieldError, GateMapping};
use super::names::{FieldName, Polarization, Quantity};
use super::values::{AttrValue, ExtraVariable, RayAlignment};
use super::volume::SourceFormat;

/// One FM301 sweep group: one physical elevation cut, numbered in acquisition
/// order.
///
/// With the `serde` feature, deserializing a sweep checks that every field has
/// a row for every ray, that the range and every field's extent on it stay
/// within [`MAX_GATES_PER_RADIAL`](crate::bounded_read::MAX_GATES_PER_RADIAL)
/// gates, and what [`Sweep::seal`] checks, and fails when a check does.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(try_from = "super::serde_checked::SweepRepr")
)]
pub struct Sweep {
    /// `sweep_number`: 0-based acquisition index; the group is
    /// `sweep_<sweep_number>`.
    pub sweep_number: u32,
    /// `sweep_mode` (Table 301-15; section 10).
    pub sweep_mode: SweepMode,
    /// `follow_mode`. `None` means the source does not say; exported as
    /// "not_set".
    pub follow_mode: Option<FollowMode>,
    /// `prt_mode`. `None` means the source does not say; exported as "not_set".
    pub prt_mode: Option<PrtMode>,
    /// `polarization_mode`.
    pub polarization_mode: Option<PolarizationMode>,
    /// `polarization_sequence(prt)` (Table 301-8a): "H" or "V" per PRT.
    pub polarization_sequence: Option<Vec<Box<str>>>,
    /// `fixed_angle` (xradar: `sweep_fixed_angle`). Target elevation; target
    /// azimuth for RHI.
    pub fixed_angle_deg: f32,
    /// `target_scan_rate`.
    pub target_scan_rate_deg_per_s: Option<f32>,
    /// `rays_are_indexed`, `rays_angle_resolution`.
    pub rays_are_indexed: Option<bool>,
    /// `rays_angle_resolution`: nominal angle between indexed rays, degrees.
    pub rays_angle_resolution_deg: Option<f32>,
    /// `qc_procedures`.
    pub qc_procedures: Option<String>,
    /// `time(time)`, `azimuth(time)`, `elevation(time)`, in source storage order.
    pub rays: Rays,
    /// `range(range)`: gate centres shared by every field (section 6).
    pub range: RangeCoord,
    /// Optional `(time)` instrument variables (section 9).
    pub ray_vars: RayVariables,
    /// `monitoring` subgroup (Table 301-11).
    pub monitoring: Option<Box<Monitoring>>,
    /// Per-ray platform position and attitude for moving platforms. Not FM301.
    pub platform_track: Option<Box<PlatformTrack>>,
    /// Sweep variables with no slot above, verbatim and in file order.
    pub extra_vars: Vec<ExtraVariable>,
    /// Sweep group attributes with no slot above, verbatim. A per-ray array
    /// ([`AttrValue::ray_alignment`], such as an ODIM `how/TXpower`) moves
    /// with its ray in [`Sweep::permute_rays`] and follows the FM301 view's
    /// ray order; any other value stays as stored.
    pub other: Vec<(Box<str>, AttrValue)>,
    /// Dataset variables, in source order.
    pub fields: Vec<Field>,
    /// Source cut number (NEXRAD ICD elevation number, 1-based). Not an FM301
    /// variable.
    pub elevation_number: Option<u16>,
    /// `false` when rays stop before the end-of-elevation marker (truncated
    /// archive, real-time chunk).
    pub complete: bool,
}

/// Ray coordinates, struct-of-arrays.
#[derive(Clone, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Rays {
    /// `time`: seconds since `Volume::time_reference`, at ray centre.
    pub time_s: Vec<f64>,
    /// `azimuth`: degrees clockwise from true north.
    pub azimuth_deg: Vec<f32>,
    /// `elevation`: degrees above horizontal.
    pub elevation_deg: Vec<f32>,
}

impl Rays {
    /// Number of rays (the azimuth vector's length).
    pub fn len(&self) -> usize {
        self.azimuth_deg.len()
    }

    /// Whether there are no rays.
    pub fn is_empty(&self) -> bool {
        self.azimuth_deg.is_empty()
    }
}

/// The sweep's `range` coordinate: gate centres in metres.
///
/// Exhaustive, like [`crate::model::FieldData`]: the file writers each map
/// every gate geometry to their format or refuse it, so a new geometry is a
/// breaking change instead of a case a wildcard arm would mishandle.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum RangeCoord {
    /// `spacing_is_constant = "true"`: centre of gate j is
    /// `first_center_m + j * spacing_m`. `spacing_m == 0` with no gates means
    /// "not yet set" ([`Sweep::new`]).
    Uniform {
        /// Centre of gate 0, in metres.
        first_center_m: f64,
        /// Distance between gate centres, in metres.
        spacing_m: f64,
        /// Number of gates.
        ngates: u32,
    },
    /// `spacing_is_constant = "false"`: explicit gate centres.
    Explicit {
        /// Gate centres, metres.
        centers_m: Vec<f32>,
    },
}

impl RangeCoord {
    /// The unset coordinate a new sweep starts with.
    pub const UNSET: RangeCoord = RangeCoord::Uniform {
        first_center_m: 0.0,
        spacing_m: 0.0,
        ngates: 0,
    };

    /// Number of gates.
    pub fn ngates(&self) -> usize {
        match self {
            Self::Uniform { ngates, .. } => *ngates as usize,
            Self::Explicit { centers_m } => centers_m.len(),
        }
    }

    /// Centre of gate `gate` in metres.
    pub fn center_m(&self, gate: usize) -> Option<f64> {
        match self {
            Self::Uniform {
                first_center_m,
                spacing_m,
                ngates,
            } => (gate < *ngates as usize).then(|| first_center_m + gate as f64 * spacing_m),
            Self::Explicit { centers_m } => centers_m.get(gate).map(|c| f64::from(*c)),
        }
    }

    /// Every gate centre in metres, as float32 (FM301 `range` type).
    pub fn centers_f32(&self) -> Vec<f32> {
        match self {
            Self::Uniform {
                first_center_m,
                spacing_m,
                ngates,
            } => (0..*ngates)
                .map(|gate| (first_center_m + f64::from(gate) * spacing_m) as f32)
                .collect(),
            Self::Explicit { centers_m } => centers_m.clone(),
        }
    }

    /// Constant gate spacing, `None` for explicit centres.
    pub fn spacing_m(&self) -> Option<f64> {
        match self {
            Self::Uniform { spacing_m, .. } => Some(*spacing_m),
            Self::Explicit { .. } => None,
        }
    }
}

macro_rules! string_enum {
    (
        $(#[$meta:meta])*
        $name:ident { $($variant:ident => $text:literal,)* }
    ) => {
        $(#[$meta])*
        #[derive(Clone, Debug, PartialEq, Eq, Hash)]
        #[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
        #[non_exhaustive]
        pub enum $name {
            $(
                #[doc = concat!("The spelling `", $text, "`.")]
                $variant,
            )*
            /// A source spelling outside the table, verbatim. Never a table
            /// spelling.
            Other(Box<str>),
        }

        impl $name {
            /// The FM301 / CfRadial spelling.
            pub fn as_str(&self) -> &str {
                match self {
                    $(Self::$variant => $text,)*
                    Self::Other(text) => text,
                }
            }

            /// Table spelling to its variant (exact, after trimming); anything
            /// else is `Other` verbatim.
            pub fn parse(text: &str) -> Self {
                match text.trim() {
                    $($text => Self::$variant,)*
                    _ => Self::Other(text.into()),
                }
            }
        }
    };
}

string_enum! {
    /// `sweep_mode` (FM301 Table 301-15; CfRadial 1.x values outside it are
    /// `Other`).
    SweepMode {
        Sector => "sector",
        Coplane => "coplane",
        Rhi => "rhi",
        VerticalPointing => "vertical_pointing",
        Idle => "idle",
        AzimuthSurveillance => "azimuth_surveillance",
        ElevationSurveillance => "elevation_surveillance",
        Sunscan => "sunscan",
        Pointing => "pointing",
        ManualPpi => "manual_ppi",
        ManualRhi => "manual_rhi",
        DopplerBeamSwinging => "doppler_beam_swinging",
        ComplexTrajectory => "complex_trajectory",
        ElectronicSteering => "electronic_steering",
    }
}

string_enum! {
    /// `follow_mode` (Table 301-15).
    FollowMode {
        None => "none",
        Sun => "sun",
        Vehicle => "vehicle",
        Aircraft => "aircraft",
        Target => "target",
        Manual => "manual",
    }
}

string_enum! {
    /// `prt_mode` (Table 301-15).
    PrtMode {
        Fixed => "fixed",
        Staggered => "staggered",
        Dual => "dual",
        Hybrid => "hybrid",
    }
}

string_enum! {
    /// `polarization_mode` (Table 301-15).
    PolarizationMode {
        Horizontal => "horizontal",
        Vertical => "vertical",
        HvAlt => "hv_alt",
        HvSim => "hv_sim",
        Circular => "circular",
    }
}

/// Table 301-11. Each variable is `(time)`; `None` means not provided.
#[derive(Clone, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Monitoring {
    /// `radar_measured_transmit_power_h`: measured transmit power, horizontal channel, dBm.
    pub radar_measured_transmit_power_h_dbm: Option<Vec<f32>>,
    /// `radar_measured_transmit_power_v`: measured transmit power, vertical channel, dBm.
    pub radar_measured_transmit_power_v_dbm: Option<Vec<f32>>,
    /// `radar_measured_sky_noise`: measured sky noise, dBm.
    pub radar_measured_sky_noise_dbm: Option<Vec<f32>>,
    /// `radar_measured_cold_noise`: measured cold noise, dBm.
    pub radar_measured_cold_noise_dbm: Option<Vec<f32>>,
    /// `radar_measured_hot_noise`: measured hot noise, dBm.
    pub radar_measured_hot_noise_dbm: Option<Vec<f32>>,
    /// `phase_difference_transmit_hv`: transmit phase difference between H and V, degrees.
    pub phase_difference_transmit_hv_deg: Option<Vec<f32>>,
    /// `antenna_pointing_accuracy_elev`: antenna pointing accuracy in elevation, degrees.
    pub antenna_pointing_accuracy_elev_deg: Option<Vec<f32>>,
    /// `antenna_pointing_accuracy_az`: antenna pointing accuracy in azimuth, degrees.
    pub antenna_pointing_accuracy_az_deg: Option<Vec<f32>>,
    /// `calibration_offset_h`: calibration offset, horizontal channel, dB.
    pub calibration_offset_h_db: Option<Vec<f32>>,
    /// `calibration_offset_v`: calibration offset, vertical channel, dB.
    pub calibration_offset_v_db: Option<Vec<f32>>,
    /// `zdr_offset`: differential reflectivity offset, dB.
    pub zdr_offset_db: Option<Vec<f32>>,
}

impl Monitoring {
    /// The slot of the Table 301-11 variable `name` (FM301 name, as
    /// [`Monitoring::variables`] lists them); `None` for another name.
    pub fn variable_mut(&mut self, name: &str) -> Option<&mut Option<Vec<f32>>> {
        Some(match name {
            "radar_measured_transmit_power_h" => &mut self.radar_measured_transmit_power_h_dbm,
            "radar_measured_transmit_power_v" => &mut self.radar_measured_transmit_power_v_dbm,
            "radar_measured_sky_noise" => &mut self.radar_measured_sky_noise_dbm,
            "radar_measured_cold_noise" => &mut self.radar_measured_cold_noise_dbm,
            "radar_measured_hot_noise" => &mut self.radar_measured_hot_noise_dbm,
            "phase_difference_transmit_hv" => &mut self.phase_difference_transmit_hv_deg,
            "antenna_pointing_accuracy_elev" => &mut self.antenna_pointing_accuracy_elev_deg,
            "antenna_pointing_accuracy_az" => &mut self.antenna_pointing_accuracy_az_deg,
            "calibration_offset_h" => &mut self.calibration_offset_h_db,
            "calibration_offset_v" => &mut self.calibration_offset_v_db,
            "zdr_offset" => &mut self.zdr_offset_db,
            _ => return None,
        })
    }

    /// `(FM301 name, values)` for every present variable, in table order.
    pub fn variables(&self) -> Vec<(&'static str, &[f32])> {
        let all: [(&'static str, &Option<Vec<f32>>); 11] = [
            (
                "radar_measured_transmit_power_h",
                &self.radar_measured_transmit_power_h_dbm,
            ),
            (
                "radar_measured_transmit_power_v",
                &self.radar_measured_transmit_power_v_dbm,
            ),
            (
                "radar_measured_sky_noise",
                &self.radar_measured_sky_noise_dbm,
            ),
            (
                "radar_measured_cold_noise",
                &self.radar_measured_cold_noise_dbm,
            ),
            (
                "radar_measured_hot_noise",
                &self.radar_measured_hot_noise_dbm,
            ),
            (
                "phase_difference_transmit_hv",
                &self.phase_difference_transmit_hv_deg,
            ),
            (
                "antenna_pointing_accuracy_elev",
                &self.antenna_pointing_accuracy_elev_deg,
            ),
            (
                "antenna_pointing_accuracy_az",
                &self.antenna_pointing_accuracy_az_deg,
            ),
            ("calibration_offset_h", &self.calibration_offset_h_db),
            ("calibration_offset_v", &self.calibration_offset_v_db),
            ("zdr_offset", &self.zdr_offset_db),
        ];
        all.into_iter()
            .filter_map(|(name, values)| values.as_deref().map(|values| (name, values)))
            .collect()
    }
}

/// Moving-platform position and attitude per ray (CfRadial 1 georeference
/// variables). Not FM301.
#[derive(Clone, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PlatformTrack {
    /// `latitude`: platform latitude per ray, WGS84 degrees north.
    pub latitude_deg: Vec<f64>,
    /// `longitude`: platform longitude per ray, WGS84 degrees east.
    pub longitude_deg: Vec<f64>,
    /// `altitude`: platform altitude per ray, metres above mean sea level.
    pub altitude_m: Vec<f64>,
    /// `altitude_agl`: platform altitude per ray, metres above ground.
    pub altitude_agl_m: Option<Vec<f64>>,
    /// `heading`: platform heading per ray, degrees clockwise from true north.
    pub heading_deg: Option<Vec<f32>>,
    /// `roll`: platform roll per ray, degrees.
    pub roll_deg: Option<Vec<f32>>,
    /// `pitch`: platform pitch per ray, degrees.
    pub pitch_deg: Option<Vec<f32>>,
    /// `drift`: platform drift angle per ray, degrees.
    pub drift_deg: Option<Vec<f32>>,
    /// `rotation`: antenna rotation angle per ray, degrees.
    pub rotation_deg: Option<Vec<f32>>,
    /// `tilt`: antenna tilt angle per ray, degrees.
    pub tilt_deg: Option<Vec<f32>>,
}

/// Optional `(time)` instrument variables (Table 301-8a). A present vector has
/// one entry per ray.
#[derive(Clone, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct RayVariables {
    /// `nyquist_velocity(time)`, m/s, NaN = missing.
    pub nyquist_velocity_mps: Option<Vec<f32>>,
    /// `unambiguous_range(time)`, m.
    pub unambiguous_range_m: Option<Vec<f32>>,
    /// `prt(time)`, s.
    pub prt_s: Option<Vec<f32>>,
    /// `prt_ratio(time)`.
    pub prt_ratio: Option<Vec<f32>>,
    /// `prt_sequence(time, prt)`.
    pub prt_sequence_s: Option<PrtSequence>,
    /// `n_samples(time)`, -9999 = missing.
    pub n_samples: Option<Vec<i32>>,
    /// `pulse_width(time)`, s.
    pub pulse_width_s: Option<Vec<f32>>,
    /// `scan_rate(time)`, deg/s.
    pub scan_rate_deg_per_s: Option<Vec<f32>>,
    /// `antenna_transition(time)`, 0/1.
    pub antenna_transition: Option<Vec<u8>>,
    /// `calib_index(time)`, FM301 int (301-8a).
    pub calib_index: Option<Vec<i32>>,
    /// `rx_range_resolution(time)`, m.
    pub rx_range_resolution_m: Option<Vec<f32>>,
    /// Not FM301: effective independent samples (DORADE / research radars).
    pub independent_samples: Option<Vec<f32>>,
}

/// `prt_sequence(time, prt)`, row-major.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PrtSequence {
    /// Pulses per sequence (the `prt` dimension).
    pub nprt: u32,
    /// The pulse repetition times in seconds, row-major `[nrays × nprt]`.
    pub values_s: Vec<f32>,
}

impl RayVariables {
    /// `(name, length)` of every present per-ray vector, for invariant checks.
    fn lengths(&self) -> Vec<(&'static str, usize)> {
        let mut out = Vec::new();
        let mut push = |name: &'static str, len: Option<usize>| {
            if let Some(len) = len {
                out.push((name, len));
            }
        };
        push(
            "nyquist_velocity",
            self.nyquist_velocity_mps.as_ref().map(Vec::len),
        );
        push(
            "unambiguous_range",
            self.unambiguous_range_m.as_ref().map(Vec::len),
        );
        push("prt", self.prt_s.as_ref().map(Vec::len));
        push("prt_ratio", self.prt_ratio.as_ref().map(Vec::len));
        push(
            "prt_sequence",
            self.prt_sequence_s
                .as_ref()
                .map(|seq| seq.values_s.len() / seq.nprt.max(1) as usize),
        );
        push("n_samples", self.n_samples.as_ref().map(Vec::len));
        push("pulse_width", self.pulse_width_s.as_ref().map(Vec::len));
        push("scan_rate", self.scan_rate_deg_per_s.as_ref().map(Vec::len));
        push(
            "antenna_transition",
            self.antenna_transition.as_ref().map(Vec::len),
        );
        push("calib_index", self.calib_index.as_ref().map(Vec::len));
        push(
            "rx_range_resolution",
            self.rx_range_resolution_m.as_ref().map(Vec::len),
        );
        push(
            "independent_samples",
            self.independent_samples.as_ref().map(Vec::len),
        );
        out
    }

    /// `true` when no variable is present.
    pub fn is_empty(&self) -> bool {
        self.lengths().is_empty()
    }
}

/// A field's gate geometry cannot be placed on the sweep's range coordinate.
#[derive(Clone, Debug, PartialEq, Error)]
#[non_exhaustive]
pub enum GeometryError {
    /// The field's gate geometry is not finite, or its spacing is not positive.
    #[error("invalid gate geometry: first centre {first_center_m} m, spacing {spacing_m} m")]
    Invalid {
        /// Centre of the first gate, metres.
        first_center_m: f64,
        /// Gate spacing, metres.
        spacing_m: f64,
    },
    /// The field's gates fall between the range coordinate's gates.
    #[error(
        "gates (first centre {first_center_m} m, spacing {spacing_m} m) do not align with the \
         sweep range (first centre {range_first_center_m} m, spacing {range_spacing_m} m)"
    )]
    Unaligned {
        /// Centre of the field's first gate, metres.
        first_center_m: f64,
        /// The field's gate spacing, metres.
        spacing_m: f64,
        /// Centre of the range coordinate's first gate, metres.
        range_first_center_m: f64,
        /// The range coordinate's gate spacing, metres.
        range_spacing_m: f64,
    },
    /// The field's gates are not a subset of the explicit range centres.
    #[error("gates do not match the explicit range centres")]
    ExplicitMismatch,
    /// The range needs more than `u32::MAX` gates.
    #[error("range gate count exceeds u32")]
    TooManyGates,
}

/// A sweep that breaks the model's invariants.
#[derive(Clone, Debug, PartialEq, Error)]
#[non_exhaustive]
pub enum SweepError {
    /// A per-ray variable does not have one entry per ray.
    #[error("sweep {what} has {len} entries for {nrays} rays")]
    RayLength {
        /// Name of the variable.
        what: String,
        /// Its length.
        len: usize,
        /// Rays of the sweep.
        nrays: usize,
    },
    /// A field failed its own checks.
    #[error("field {field}: {source}")]
    Field {
        /// Name of the field.
        field: String,
        /// The field's error.
        #[source]
        source: FieldError,
    },
    /// A field's buffer does not hold `nrays × ngates` values.
    #[error("field {field} has {len} values, expected {expected}")]
    FieldLength {
        /// Name of the field.
        field: String,
        /// Values in the buffer.
        len: usize,
        /// Values the shape requires.
        expected: usize,
    },
    /// A field's absent rows are not ascending, or name a ray past the last.
    #[error("field {field} absent rows are not ascending and within the rays")]
    AbsentRows {
        /// Name of the field.
        field: String,
    },
    /// A field extends past the end of an explicit range coordinate.
    #[error("field {field} extends to range gate {end}, past the explicit range's {range_gates}")]
    FieldExtent {
        /// Name of the field.
        field: String,
        /// One past the last range gate the field covers.
        end: u64,
        /// Gates of the range coordinate.
        range_gates: usize,
    },
    /// A field's gate stride is 0, or not 1 on an explicit range coordinate.
    #[error("field {field} has stride {stride} on a range that does not allow it")]
    Stride {
        /// Name of the field.
        field: String,
        /// The field's gate stride.
        stride: u32,
    },
    /// Two fields of a sweep have the same name.
    #[error("duplicate field name {name}")]
    DuplicateName {
        /// The name.
        name: String,
    },
    /// A sweep's `sweep_number` is not its position in the volume.
    #[error("sweep at index {index} has sweep_number {sweep_number}")]
    SweepNumber {
        /// Position of the sweep in `Volume::sweeps`.
        index: usize,
        /// Its `sweep_number`.
        sweep_number: u32,
    },
    /// [`Sweep::permute_rays`] found a verbatim array attribute of the sweep
    /// or of one of its fields with one entry per ray under a name not known
    /// to be per ray ([`RayAlignment::Unknown`]): moving it could scramble
    /// it and leaving it could misalign it, so the rays stay where they are.
    #[error(
        "attribute {name} has one entry per ray but is not known to be per ray; the rays were not reordered"
    )]
    UnknownRayAttribute {
        /// The attribute's name, `field/name` for a field's.
        name: String,
    },
    /// [`Sweep::permute_rays`] got an order that is not a permutation of the
    /// sweep's rays.
    #[error("a ray order of {len} entries is not a permutation of the sweep's {nrays} rays")]
    RayOrder {
        /// Entries in the order.
        len: usize,
        /// Rays in the sweep.
        nrays: usize,
    },
    /// The field's gate geometry does not fit the range coordinate.
    #[error(transparent)]
    Geometry(#[from] GeometryError),
    /// A deserialized sweep's range coordinate, or the range gates one of
    /// its fields covers, exceeds
    /// [`MAX_GATES_PER_RADIAL`](crate::bounded_read::MAX_GATES_PER_RADIAL),
    /// the ceiling every decoder applies to a radial.
    #[error("{what} spans {gates} range gates (limit {limit})")]
    RangeGates {
        /// `range`, or the field (`field DBZH`).
        what: String,
        /// Range gates it spans.
        gates: u64,
        /// The ceiling.
        limit: usize,
    },
    /// A deserialized [`ExtraVariable`]'s `shape` does not describe its
    /// values: it has a different number of entries than `dims`, or its
    /// product is not the number of values.
    #[error(
        "extra variable {name}: dims {dims:?} and shape {shape:?} do not describe a value \
         count of {len}"
    )]
    ExtraShape {
        /// Name of the variable.
        name: String,
        /// Its dimension names.
        dims: Vec<String>,
        /// Its `shape`.
        shape: Vec<u32>,
        /// Values it holds.
        len: usize,
    },
}

/// `length` is a whole number of `step`s (0 included), within `tolerance`.
fn is_multiple(length: f64, step: f64, tolerance: f64) -> bool {
    let steps = (length / step).round();
    (steps * step - length).abs() <= tolerance
}

impl Sweep {
    /// An empty sweep with an unset range and every optional item absent.
    pub fn new(sweep_number: u32, sweep_mode: SweepMode, fixed_angle_deg: f32) -> Self {
        Self {
            sweep_number,
            sweep_mode,
            follow_mode: None,
            prt_mode: None,
            polarization_mode: None,
            polarization_sequence: None,
            fixed_angle_deg,
            target_scan_rate_deg_per_s: None,
            rays_are_indexed: None,
            rays_angle_resolution_deg: None,
            qc_procedures: None,
            rays: Rays::default(),
            range: RangeCoord::UNSET,
            ray_vars: RayVariables::default(),
            monitoring: None,
            platform_track: None,
            extra_vars: Vec::new(),
            other: Vec::new(),
            fields: Vec::new(),
            elevation_number: None,
            complete: true,
        }
    }

    /// Number of rays.
    pub fn nrays(&self) -> usize {
        self.rays.len()
    }

    /// The sweep's elevation as one angle, the way the products take it:
    /// beam height and ground range of the column products, the lowest-tilt
    /// choice of the trackers and the bench, and the dealiasers' tilt
    /// geometry.
    ///
    /// For NEXRAD Level II (`source` is [`SourceFormat::NexradLevel2`]) this
    /// is the elevation of the first ray in storage order. The legacy model
    /// stored that value as the cut elevation, and the BowEcho app and `main`
    /// still use it. It differs from [`Sweep::fixed_angle_deg`], the
    /// Message 5 cut angle that xradar and Py-ART report, while the antenna
    /// settles onto the cut: KTLX 2024-03-15 sweeps 0, 1, 4 and 9 read 0.582,
    /// 0.483, 0.409 and 0.478 deg on 0.4834 deg cuts. For every other source,
    /// and for a Level II sweep without rays, it is `fixed_angle_deg`
    /// (design note `docs/design/fm301-model.md` section 5.2).
    ///
    /// `source` is the whole volume's `provenance.source_format`, so every
    /// sweep of a volume is read under one format. That holds after a merge
    /// too: [`merge_volumes`](crate::merge_volumes) rejects parts whose
    /// source formats differ, so no volume mixes sweeps whose tilt elevation
    /// follows different rules.
    pub fn tilt_elevation_deg(&self, source: SourceFormat) -> f32 {
        match source {
            SourceFormat::NexradLevel2 => self
                .rays
                .elevation_deg
                .first()
                .copied()
                .unwrap_or(self.fixed_angle_deg),
            _ => self.fixed_angle_deg,
        }
    }

    /// Reserve storage for `rays` more rays in the ray coordinates.
    pub fn reserve_rays(&mut self, rays: usize) {
        self.rays.time_s.reserve(rays);
        self.rays.azimuth_deg.reserve(rays);
        self.rays.elevation_deg.reserve(rays);
    }

    /// Append a ray and return its index, which the decoder passes to
    /// `Field::push_row_*`.
    pub fn push_ray(&mut self, time_s: f64, azimuth_deg: f32, elevation_deg: f32) -> usize {
        let index = self.rays.azimuth_deg.len();
        self.rays.time_s.push(time_s);
        self.rays.azimuth_deg.push(azimuth_deg);
        self.rays.elevation_deg.push(elevation_deg);
        index
    }

    /// Register a field's native geometry (centre of its first gate, spacing,
    /// gate count) and return its mapping onto the sweep range, growing or
    /// refining the range as needed (section 6.5).
    ///
    /// - A spacing that is an integer multiple of the range spacing maps with
    ///   that stride; a finer spacing that divides it refines the range and
    ///   rewrites the mappings of fields already in [`Sweep::fields`] (no gate
    ///   data moves). Add a field to `fields` before attaching the next
    ///   geometry.
    /// - When one spacing divides the other and the first gates are a whole
    ///   number of the finer spacing apart centre to centre, but not edge to
    ///   edge, the range is refined to half the finer spacing (gates of 1 km
    ///   and 500 m both centred at 0 m map onto a 250 m range with strides 4
    ///   and 2).
    /// - A start edge before the range extends the range backwards.
    /// - Anything else is [`GeometryError::Unaligned`]. Tolerance: 1e-6 of the
    ///   range spacing.
    pub fn attach_geometry(
        &mut self,
        first_center_m: f64,
        spacing_m: f64,
        ngates: u32,
    ) -> Result<GateMapping, GeometryError> {
        if !first_center_m.is_finite() || !spacing_m.is_finite() || spacing_m <= 0.0 {
            return Err(GeometryError::Invalid {
                first_center_m,
                spacing_m,
            });
        }
        let fields = &mut self.fields;
        match &mut self.range {
            RangeCoord::Explicit { centers_m } => {
                if ngates as usize > centers_m.len() {
                    return Err(GeometryError::ExplicitMismatch);
                }
                for (gate, center) in centers_m.iter().take(ngates as usize).enumerate() {
                    let expected = first_center_m + gate as f64 * spacing_m;
                    if (f64::from(*center) - expected).abs() > 1e-3_f64.max(1e-6 * spacing_m) {
                        return Err(GeometryError::ExplicitMismatch);
                    }
                }
                Ok(GateMapping::IDENTITY)
            }
            RangeCoord::Uniform {
                first_center_m: c0,
                spacing_m: s0,
                ngates: n0,
            } => {
                if *s0 <= 0.0 || !s0.is_finite() {
                    *c0 = first_center_m;
                    *s0 = spacing_m;
                    *n0 = ngates;
                    return Ok(GateMapping::IDENTITY);
                }
                let unaligned = GeometryError::Unaligned {
                    first_center_m,
                    spacing_m,
                    range_first_center_m: *c0,
                    range_spacing_m: *s0,
                };
                // Work on copies so a failed call leaves the sweep unchanged.
                let (mut range_c0, mut range_s0, mut range_n0) = (*c0, *s0, u64::from(*n0));
                let mut refine = 1u32;
                // The range spacing after this call: the finer of the two
                // spacings when it divides the other and the offset between
                // the start edges. When the spacings nest but the first
                // gates share a centre instead of an edge (converted feeds
                // put gates of 1 km and 500 m both centred at 0 m, whose
                // edges are 250 m apart), half the finer spacing.
                let tolerance = 1e-6 * range_s0;
                let edge_offset = (first_center_m - spacing_m / 2.0) - (range_c0 - range_s0 / 2.0);
                let finer = range_s0.min(spacing_m);
                let mut target = finer;
                if !is_multiple(edge_offset, finer, tolerance) {
                    let nested = is_multiple(range_s0.max(spacing_m), finer, tolerance);
                    let centred = is_multiple(first_center_m - range_c0, finer, tolerance);
                    if !(nested && centred) {
                        return Err(unaligned);
                    }
                    target = finer / 2.0;
                }
                if target < range_s0 - tolerance {
                    // The target spacing must divide the range spacing.
                    let k = (range_s0 / target).round();
                    if k < 2.0
                        || (k * target - range_s0).abs() > 1e-6 * range_s0
                        || k > f64::from(u32::MAX)
                    {
                        return Err(unaligned);
                    }
                    refine = k as u32;
                    let edge = range_c0 - range_s0 / 2.0;
                    range_s0 = target;
                    range_c0 = edge + target / 2.0;
                    range_n0 *= u64::from(refine);
                }
                let tolerance = 1e-6 * range_s0;
                let stride = (spacing_m / range_s0).round();
                if stride < 1.0
                    || (stride * range_s0 - spacing_m).abs() > tolerance
                    || stride > f64::from(u32::MAX)
                {
                    return Err(unaligned);
                }
                let edge = first_center_m - spacing_m / 2.0;
                let edge0 = range_c0 - range_s0 / 2.0;
                let m = ((edge - edge0) / range_s0).round();
                if (edge - edge0 - m * range_s0).abs() > tolerance || m.abs() > f64::from(u32::MAX)
                {
                    return Err(unaligned);
                }
                let shift = if m < 0.0 { (-m) as u64 } else { 0 };
                let start = if m < 0.0 { 0 } else { m as u64 };
                let mapping = GateMapping {
                    start: u32::try_from(start).map_err(|_| GeometryError::TooManyGates)?,
                    stride: stride as u32,
                };
                let end = mapping.end(ngates).ok_or(GeometryError::TooManyGates)?;
                let range_gates = u32::try_from((range_n0 + shift).max(end))
                    .map_err(|_| GeometryError::TooManyGates)?;
                let remap = |gates: GateMapping| -> Option<GateMapping> {
                    Some(GateMapping {
                        start: gates
                            .start
                            .checked_mul(refine)?
                            .checked_add(u32::try_from(shift).ok()?)?,
                        stride: gates.stride.max(1).checked_mul(refine)?,
                    })
                };
                if fields.iter().any(|field| remap(field.gates).is_none()) {
                    return Err(GeometryError::TooManyGates);
                }
                for field in fields.iter_mut() {
                    if let Some(gates) = remap(field.gates) {
                        field.gates = gates;
                    }
                }
                *c0 = range_c0 - shift as f64 * range_s0;
                *s0 = range_s0;
                *n0 = range_gates;
                Ok(mapping)
            }
        }
    }

    /// Append a field, rejecting a name already present (compared by
    /// `as_str`). Returns its index.
    pub fn add_field(&mut self, field: Field) -> Result<usize, SweepError> {
        if self.field(&field.name).is_some() {
            return Err(SweepError::DuplicateName {
                name: field.name.as_str().to_owned(),
            });
        }
        self.fields.push(field);
        Ok(self.fields.len() - 1)
    }

    /// The field named `name`.
    pub fn field(&self, name: &FieldName) -> Option<&Field> {
        self.fields
            .iter()
            .find(|field| field.name.as_str() == name.as_str())
    }

    /// The field named `name`, mutably.
    pub fn field_mut(&mut self, name: &FieldName) -> Option<&mut Field> {
        self.fields
            .iter_mut()
            .find(|field| field.name.as_str() == name.as_str())
    }

    /// Position of the field named `name` in [`Sweep::fields`].
    pub fn field_index(&self, name: &FieldName) -> Option<usize> {
        self.fields
            .iter()
            .position(|field| field.name.as_str() == name.as_str())
    }

    /// Preferred field for a quantity: H before unspecified before V, then
    /// source order.
    pub fn find(&self, quantity: Quantity) -> Option<&Field> {
        fn rank(polarization: Polarization) -> u8 {
            match polarization {
                Polarization::H | Polarization::CopolarH => 0,
                Polarization::Unspecified | Polarization::Hv => 1,
                Polarization::V | Polarization::CopolarV => 2,
                Polarization::CrosspolarH | Polarization::CrosspolarV => 3,
            }
        }
        self.fields
            .iter()
            .filter(|field| field.quantity == quantity)
            .min_by_key(|field| rank(field.polarization))
    }

    /// Append absent rows for rays at the end that a field never received, grow
    /// a uniform range to cover every field, then check the invariants
    /// (section 3). Decoders call it once per sweep.
    pub fn seal(&mut self) -> Result<(), SweepError> {
        let nrays = self.rays.azimuth_deg.len();
        let ray_len = |what: &str, len: usize| -> Result<(), SweepError> {
            if len == nrays {
                Ok(())
            } else {
                Err(SweepError::RayLength {
                    what: what.to_owned(),
                    len,
                    nrays,
                })
            }
        };
        ray_len("time", self.rays.time_s.len())?;
        ray_len("elevation", self.rays.elevation_deg.len())?;
        for (name, len) in self.ray_vars.lengths() {
            ray_len(name, len)?;
        }
        if let Some(seq) = &self.ray_vars.prt_sequence_s
            && nrays.checked_mul(seq.nprt as usize) != Some(seq.values_s.len())
        {
            return Err(SweepError::RayLength {
                what: "prt_sequence values".to_owned(),
                len: seq.values_s.len(),
                nrays,
            });
        }
        if let Some(monitoring) = &self.monitoring {
            for (name, values) in monitoring.variables() {
                ray_len(name, values.len())?;
            }
        }
        if let Some(track) = &self.platform_track {
            ray_len("latitude", track.latitude_deg.len())?;
            ray_len("longitude", track.longitude_deg.len())?;
            ray_len("altitude", track.altitude_m.len())?;
            for (name, len) in [
                ("altitude_agl", track.altitude_agl_m.as_ref().map(Vec::len)),
                ("heading", track.heading_deg.as_ref().map(Vec::len)),
                ("roll", track.roll_deg.as_ref().map(Vec::len)),
                ("pitch", track.pitch_deg.as_ref().map(Vec::len)),
                ("drift", track.drift_deg.as_ref().map(Vec::len)),
                ("rotation", track.rotation_deg.as_ref().map(Vec::len)),
                ("tilt", track.tilt_deg.as_ref().map(Vec::len)),
            ] {
                if let Some(len) = len {
                    ray_len(name, len)?;
                }
            }
        }
        for extra in &self.extra_vars {
            if extra.is_per_ray() {
                ray_len(&extra.name, extra.shape.first().map_or(0, |n| *n as usize))?;
            }
        }

        let mut names = HashSet::with_capacity(self.fields.len());
        let range_gates = self.range.ngates();
        let explicit = matches!(self.range, RangeCoord::Explicit { .. });
        let mut needed_gates = range_gates as u64;
        for field in &mut self.fields {
            let label = field.name.as_str().to_owned();
            if !names.insert(label.clone()) {
                return Err(SweepError::DuplicateName { name: label });
            }
            field
                .push_absent_rows_to(nrays)
                .map_err(|source| SweepError::Field {
                    field: label.clone(),
                    source,
                })?;
            let expected = nrays.checked_mul(field.ngates as usize);
            if expected != Some(field.data.len()) {
                return Err(SweepError::FieldLength {
                    field: label,
                    len: field.data.len(),
                    expected: nrays.saturating_mul(field.ngates as usize),
                });
            }
            if !field.absent_rows.windows(2).all(|pair| pair[0] < pair[1])
                || field
                    .absent_rows
                    .last()
                    .is_some_and(|last| *last as usize >= nrays)
            {
                return Err(SweepError::AbsentRows { field: label });
            }
            if field.gates.stride == 0 || (explicit && field.gates.stride != 1) {
                return Err(SweepError::Stride {
                    field: label,
                    stride: field.gates.stride,
                });
            }
            let end = field.gates.end(field.ngates).unwrap_or(u64::MAX);
            if explicit && end > range_gates as u64 {
                return Err(SweepError::FieldExtent {
                    field: label,
                    end,
                    range_gates,
                });
            }
            needed_gates = needed_gates.max(end);
        }
        if let RangeCoord::Uniform { ngates, .. } = &mut self.range {
            *ngates = u32::try_from(needed_gates).map_err(|_| GeometryError::TooManyGates)?;
        }
        Ok(())
    }

    /// Reorder the rays in storage: ray `i` afterwards is ray `order[i]`
    /// before. Every per-ray item moves with its ray (coordinates, the
    /// `(time)` instrument variables, `monitoring`, the platform track,
    /// per-ray `extra_vars`, every field row and its absent-row mark, and
    /// every per-ray array among the verbatim attributes of the sweep and
    /// its fields, [`AttrValue::ray_alignment`]); field rows move in place,
    /// one row at a time, so no second copy of a field is allocated.
    ///
    /// Storage order is the source's order everywhere else in the model
    /// (design note 12.1); this is for callers that hand field buffers to
    /// another owner in a given ray order, such as a binding that moves them
    /// into NumPy under xradar's azimuth order
    /// ([`crate::fm301::order_rays_for_view`]). Reads that depend on storage
    /// order, like the first ray's elevation in
    /// [`Sweep::tilt_elevation_deg`], see the new order.
    ///
    /// # Errors
    ///
    /// [`SweepError::RayOrder`] when `order` is not a permutation of
    /// `0..nrays`, [`SweepError::RayLength`] when a per-ray item does not
    /// have one entry per ray, and [`SweepError::UnknownRayAttribute`] when a
    /// verbatim array attribute has one entry per ray but is not known to be
    /// per ray ([`RayAlignment::Unknown`]); the sweep is unchanged then.
    pub fn permute_rays(&mut self, order: &[u32]) -> Result<(), SweepError> {
        // Check everything before moving anything.
        self.check_permutation(order)?;
        let nrays = self.nrays();

        // Exhaustive destructuring: a new per-ray item must be added here
        // (and to `seal`) or this does not compile.
        let Sweep {
            sweep_number: _,
            sweep_mode: _,
            follow_mode: _,
            prt_mode: _,
            polarization_mode: _,
            polarization_sequence: _,
            fixed_angle_deg: _,
            target_scan_rate_deg_per_s: _,
            rays_are_indexed: _,
            rays_angle_resolution_deg: _,
            qc_procedures: _,
            rays,
            range: _,
            ray_vars,
            monitoring,
            platform_track,
            extra_vars,
            other,
            fields,
            elevation_number: _,
            complete: _,
        } = self;
        let Rays {
            time_s,
            azimuth_deg,
            elevation_deg,
        } = rays;
        permute_rows(time_s, 1, order);
        permute_rows(azimuth_deg, 1, order);
        permute_rows(elevation_deg, 1, order);

        let RayVariables {
            nyquist_velocity_mps,
            unambiguous_range_m,
            prt_s,
            prt_ratio,
            prt_sequence_s,
            n_samples,
            pulse_width_s,
            scan_rate_deg_per_s,
            antenna_transition,
            calib_index,
            rx_range_resolution_m,
            independent_samples,
        } = ray_vars;
        for values in [
            nyquist_velocity_mps,
            unambiguous_range_m,
            prt_s,
            prt_ratio,
            pulse_width_s,
            scan_rate_deg_per_s,
            rx_range_resolution_m,
            independent_samples,
        ]
        .into_iter()
        .flatten()
        {
            permute_rows(values, 1, order);
        }
        for values in [n_samples, calib_index].into_iter().flatten() {
            permute_rows(values, 1, order);
        }
        if let Some(values) = antenna_transition {
            permute_rows(values, 1, order);
        }
        if let Some(sequence) = prt_sequence_s {
            permute_rows(&mut sequence.values_s, sequence.nprt as usize, order);
        }

        if let Some(monitoring) = monitoring {
            let Monitoring {
                radar_measured_transmit_power_h_dbm,
                radar_measured_transmit_power_v_dbm,
                radar_measured_sky_noise_dbm,
                radar_measured_cold_noise_dbm,
                radar_measured_hot_noise_dbm,
                phase_difference_transmit_hv_deg,
                antenna_pointing_accuracy_elev_deg,
                antenna_pointing_accuracy_az_deg,
                calibration_offset_h_db,
                calibration_offset_v_db,
                zdr_offset_db,
            } = &mut **monitoring;
            for values in [
                radar_measured_transmit_power_h_dbm,
                radar_measured_transmit_power_v_dbm,
                radar_measured_sky_noise_dbm,
                radar_measured_cold_noise_dbm,
                radar_measured_hot_noise_dbm,
                phase_difference_transmit_hv_deg,
                antenna_pointing_accuracy_elev_deg,
                antenna_pointing_accuracy_az_deg,
                calibration_offset_h_db,
                calibration_offset_v_db,
                zdr_offset_db,
            ]
            .into_iter()
            .flatten()
            {
                permute_rows(values, 1, order);
            }
        }

        if let Some(track) = platform_track {
            let PlatformTrack {
                latitude_deg,
                longitude_deg,
                altitude_m,
                altitude_agl_m,
                heading_deg,
                roll_deg,
                pitch_deg,
                drift_deg,
                rotation_deg,
                tilt_deg,
            } = &mut **track;
            for values in [latitude_deg, longitude_deg, altitude_m] {
                permute_rows(values, 1, order);
            }
            if let Some(values) = altitude_agl_m {
                permute_rows(values, 1, order);
            }
            for values in [
                heading_deg,
                roll_deg,
                pitch_deg,
                drift_deg,
                rotation_deg,
                tilt_deg,
            ]
            .into_iter()
            .flatten()
            {
                permute_rows(values, 1, order);
            }
        }

        for extra in extra_vars.iter_mut().filter(|extra| extra.is_per_ray()) {
            let row_len = extra.shape.iter().skip(1).map(|n| *n as usize).product();
            if let Some(values) = extra.values.take_rows(row_len, order) {
                extra.values = values;
            }
        }

        // Verbatim attributes: a per-ray array (an ODIM `how/TXpower` or
        // `how/startT` the decoder has no slot for) moves too.
        let attrs = other.iter_mut().chain(
            fields
                .iter_mut()
                .flat_map(|field| field.attrs.other.iter_mut()),
        );
        for (name, value) in attrs {
            if value.ray_alignment(name, nrays) == RayAlignment::PerRay {
                *value = value.in_ray_order(nrays, order);
            }
        }

        let mut was_absent = vec![false; nrays];
        for field in fields.iter_mut() {
            let row_len = field.ngates as usize;
            match &mut field.data {
                FieldData::U8 { values, .. } => permute_rows(values, row_len, order),
                FieldData::U16 { values, .. } => permute_rows(values, row_len, order),
                FieldData::I8 { values, .. } => permute_rows(values, row_len, order),
                FieldData::I16 { values, .. } => permute_rows(values, row_len, order),
                FieldData::I32 { values, .. } => permute_rows(values, row_len, order),
                FieldData::F32 { values, .. } => permute_rows(values, row_len, order),
                FieldData::F64 { values, .. } => permute_rows(values, row_len, order),
            }
            if !field.absent_rows.is_empty() {
                was_absent.fill(false);
                for &row in &field.absent_rows {
                    if let Some(slot) = was_absent.get_mut(row as usize) {
                        *slot = true;
                    }
                }
                field.absent_rows = order
                    .iter()
                    .enumerate()
                    .filter(|(_, old)| was_absent[**old as usize])
                    .map(|(new, _)| new as u32)
                    .collect();
            }
        }
        Ok(())
    }

    /// The first verbatim array attribute of this sweep or of its fields
    /// that has one entry per ray but is not known to be per ray
    /// ([`RayAlignment::Unknown`]), as `name` or `field/name`. Such a sweep
    /// cannot be reordered ([`Sweep::permute_rays`]).
    pub fn unknown_ray_attribute(&self) -> Option<String> {
        let nrays = self.nrays();
        let unknown = |(name, value): &&(Box<str>, AttrValue)| {
            value.ray_alignment(name, nrays) == RayAlignment::Unknown
        };
        if let Some((name, _)) = self.other.iter().find(unknown) {
            return Some(name.to_string());
        }
        self.fields.iter().find_map(|field| {
            field
                .attrs
                .other
                .iter()
                .find(unknown)
                .map(|(name, _)| format!("{}/{name}", field.name.as_str()))
        })
    }

    /// Every check of [`Sweep::permute_rays`], without moving anything:
    /// `order` is a permutation of `0..nrays`, every per-ray item has one
    /// entry per ray, and no verbatim array attribute of unknown alignment
    /// has one entry per ray. [`crate::fm301::order_rays_for_view`] checks
    /// every sweep with it before reordering any, so a failure leaves the
    /// whole volume unchanged.
    pub(crate) fn check_permutation(&self, order: &[u32]) -> Result<(), SweepError> {
        let nrays = self.nrays();
        let invalid = || SweepError::RayOrder {
            len: order.len(),
            nrays,
        };
        if order.len() != nrays {
            return Err(invalid());
        }
        let mut seen = vec![false; nrays];
        for &ray in order {
            match seen.get_mut(ray as usize) {
                Some(slot) if !*slot => *slot = true,
                _ => return Err(invalid()),
            }
        }
        self.check_ray_lengths(nrays)?;
        if let Some(name) = self.unknown_ray_attribute() {
            return Err(SweepError::UnknownRayAttribute { name });
        }
        Ok(())
    }

    /// The per-ray length checks of [`Sweep::seal`] plus every field's row
    /// count, so [`Sweep::permute_rays`] fails before moving anything.
    fn check_ray_lengths(&self, nrays: usize) -> Result<(), SweepError> {
        let ray_len = |what: &str, len: usize| -> Result<(), SweepError> {
            if len == nrays {
                Ok(())
            } else {
                Err(SweepError::RayLength {
                    what: what.to_owned(),
                    len,
                    nrays,
                })
            }
        };
        ray_len("time", self.rays.time_s.len())?;
        ray_len("elevation", self.rays.elevation_deg.len())?;
        for (name, len) in self.ray_vars.lengths() {
            ray_len(name, len)?;
        }
        if let Some(seq) = &self.ray_vars.prt_sequence_s {
            ray_len(
                "prt_sequence values",
                seq.values_s.len() / (seq.nprt as usize).max(1),
            )?;
        }
        if let Some(monitoring) = &self.monitoring {
            for (name, values) in monitoring.variables() {
                ray_len(name, values.len())?;
            }
        }
        if let Some(track) = &self.platform_track {
            ray_len("latitude", track.latitude_deg.len())?;
            ray_len("longitude", track.longitude_deg.len())?;
            ray_len("altitude", track.altitude_m.len())?;
            for (name, len) in [
                ("altitude_agl", track.altitude_agl_m.as_ref().map(Vec::len)),
                ("heading", track.heading_deg.as_ref().map(Vec::len)),
                ("roll", track.roll_deg.as_ref().map(Vec::len)),
                ("pitch", track.pitch_deg.as_ref().map(Vec::len)),
                ("drift", track.drift_deg.as_ref().map(Vec::len)),
                ("rotation", track.rotation_deg.as_ref().map(Vec::len)),
                ("tilt", track.tilt_deg.as_ref().map(Vec::len)),
            ] {
                if let Some(len) = len {
                    ray_len(name, len)?;
                }
            }
        }
        for extra in self.extra_vars.iter().filter(|extra| extra.is_per_ray()) {
            ray_len(&extra.name, extra.shape.first().map_or(0, |n| *n as usize))?;
            let row_len: usize = extra.shape.iter().skip(1).map(|n| *n as usize).product();
            if row_len > 0 {
                ray_len(&extra.name, extra.values.len() / row_len)?;
            }
        }
        for field in &self.fields {
            ray_len(field.name.as_str(), field.nrays as usize)?;
            let expected = nrays * field.ngates as usize;
            if field.data.len() != expected {
                return Err(SweepError::FieldLength {
                    field: field.name.as_str().to_owned(),
                    len: field.data.len(),
                    expected,
                });
            }
        }
        Ok(())
    }
}

/// Rows of `values` (`row_len` elements each) reordered in place so that row
/// `i` becomes the old row `order[i]`. `order` is a permutation of the row
/// indices and `values` holds exactly `order.len()` rows (checked by the
/// caller). A rotation, which is what a sweep that starts mid-circle needs,
/// moves as one block; any other order follows its cycles through one
/// temporary row.
fn permute_rows<T: Copy>(values: &mut [T], row_len: usize, order: &[u32]) {
    let rows = order.len();
    if row_len == 0 || values.len() != rows.saturating_mul(row_len) {
        return;
    }
    if let Some(&first) = order.first() {
        let shift = first as usize;
        if order
            .iter()
            .enumerate()
            .all(|(row, &source)| source as usize == (row + shift) % rows)
        {
            values.rotate_left(shift * row_len);
            return;
        }
    }
    let mut done = vec![false; rows];
    let mut temp: Vec<T> = Vec::with_capacity(row_len);
    for start in 0..rows {
        if done[start] {
            continue;
        }
        if order[start] as usize == start {
            done[start] = true;
            continue;
        }
        temp.clear();
        temp.extend_from_slice(&values[start * row_len..(start + 1) * row_len]);
        let mut target = start;
        loop {
            done[target] = true;
            let source = order[target] as usize;
            if source == start {
                values[target * row_len..(target + 1) * row_len].copy_from_slice(&temp);
                break;
            }
            values.copy_within(source * row_len..(source + 1) * row_len, target * row_len);
            target = source;
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    /// In-place row permutation against taking the rows in order, for
    /// rotations, cycles, the identity and a reversal.
    #[test]
    fn permute_rows_matches_taking_rows_in_order() {
        let orders: [Vec<u32>; 5] = [
            vec![0, 1, 2, 3, 4, 5],
            vec![2, 3, 4, 5, 0, 1],
            vec![5, 4, 3, 2, 1, 0],
            vec![1, 0, 3, 2, 5, 4],
            vec![3, 0, 4, 1, 5, 2],
        ];
        for order in &orders {
            for row_len in [1usize, 3] {
                let before: Vec<u16> = (0..(6 * row_len) as u16).collect();
                let mut after = before.clone();
                permute_rows(&mut after, row_len, order);
                let want: Vec<u16> = order
                    .iter()
                    .flat_map(|&row| {
                        before[row as usize * row_len..(row as usize + 1) * row_len].to_vec()
                    })
                    .collect();
                assert_eq!(after, want, "{order:?} x {row_len}");
            }
        }
    }

    #[test]
    fn mode_strings_parse_and_keep_unknown_spellings() {
        assert_eq!(SweepMode::parse("rhi"), SweepMode::Rhi);
        assert_eq!(
            SweepMode::parse("calibration"),
            SweepMode::Other("calibration".into())
        );
        assert_eq!(FollowMode::parse("not_set").as_str(), "not_set");
        assert_eq!(PolarizationMode::HvSim.as_str(), "hv_sim");
    }
}
