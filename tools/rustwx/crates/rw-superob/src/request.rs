//! What crosses the seam: a JSON request naming arrays by index in a
//! table of raw buffers.
//!
//! Floats in the JSON are written by the Python side as `repr(float)`
//! STRINGS and parsed here with Rust's correctly rounded `str::parse`.
//! A JSON number would go through serde_json's default float parser,
//! which is approximate for 17-digit mantissas -- one last bit off in a
//! threshold is a different set of gates -- and switching on its
//! `float_roundtrip` feature would change float parsing for every other
//! crate built in this workspace at the same time.

use std::ffi::c_void;

use serde::{Deserialize, Deserializer};

use crate::{Result, SuperobFailure};

/// One buffer in the table, exactly as `numpy.ndarray.ctypes` describes it.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SoArray {
    pub ptr: *mut c_void,
    /// Element count (not bytes).
    pub len: u64,
    /// One of the `DTYPE_*` codes.
    pub dtype: u32,
    /// 1 when the caller hands this buffer over for writing.
    pub writable: u32,
}

pub const DTYPE_F32: u32 = 1;
pub const DTYPE_F64: u32 = 2;
pub const DTYPE_U8: u32 = 3;
pub const DTYPE_I8: u32 = 4;
pub const DTYPE_I32: u32 = 5;
pub const DTYPE_I64: u32 = 6;
pub const DTYPE_I16: u32 = 7;

fn dtype_name(code: u32) -> &'static str {
    match code {
        DTYPE_F32 => "float32",
        DTYPE_F64 => "float64",
        DTYPE_U8 => "uint8",
        DTYPE_I8 => "int8",
        DTYPE_I32 => "int32",
        DTYPE_I64 => "int64",
        DTYPE_I16 => "int16",
        _ => "unknown",
    }
}

/// The buffer table for one call.  Borrowing is by index; the caller
/// guarantees the buffers outlive the call and that no index names the
/// same memory as another written one (checked for the written set).
pub struct Arrays<'a> {
    items: &'a [SoArray],
}

macro_rules! reader {
    ($name:ident, $ty:ty, $code:expr) => {
        pub fn $name(&self, index: usize, what: &str) -> Result<&'a [$ty]> {
            let item = self.item(index, what, $code)?;
            if item.len == 0 {
                return Ok(&[]);
            }
            Ok(unsafe { std::slice::from_raw_parts(item.ptr as *const $ty, item.len as usize) })
        }
    };
}

macro_rules! writer {
    ($name:ident, $ty:ty, $code:expr) => {
        /// # Safety contract
        /// The caller of the seam hands this buffer over for writing and no
        /// other index in the call aliases it (`check_disjoint`).
        #[allow(clippy::mut_from_ref)]
        pub fn $name(&self, index: usize, what: &str) -> Result<&'a mut [$ty]> {
            let item = self.item(index, what, $code)?;
            if item.writable != 1 {
                return Err(SuperobFailure::request(format!(
                    "{what} (array {index}) was not handed over for writing"
                )));
            }
            if item.len == 0 {
                return Ok(&mut []);
            }
            Ok(unsafe { std::slice::from_raw_parts_mut(item.ptr as *mut $ty, item.len as usize) })
        }
    };
}

impl<'a> Arrays<'a> {
    pub fn new(items: &'a [SoArray]) -> Self {
        Self { items }
    }

    fn item(&self, index: usize, what: &str, dtype: u32) -> Result<&'a SoArray> {
        let item = self.items.get(index).ok_or_else(|| {
            SuperobFailure::request(format!(
                "{what} names array {index}, the table holds {}",
                self.items.len()
            ))
        })?;
        if item.dtype != dtype {
            return Err(SuperobFailure::request(format!(
                "{what} (array {index}) is {}, this use needs {}",
                dtype_name(item.dtype),
                dtype_name(dtype)
            )));
        }
        if item.len > 0 {
            if item.ptr.is_null() {
                return Err(SuperobFailure::request(format!("{what} (array {index}) is null")));
            }
            let align = match dtype {
                DTYPE_F32 | DTYPE_I32 => 4,
                DTYPE_I16 => 2,
                DTYPE_F64 | DTYPE_I64 => 8,
                _ => 1,
            };
            if (item.ptr as usize) % align != 0 {
                return Err(SuperobFailure::request(format!(
                    "{what} (array {index}) is not {align}-byte aligned"
                )));
            }
        }
        Ok(item)
    }

    reader!(f32s, f32, DTYPE_F32);
    reader!(f64s, f64, DTYPE_F64);
    reader!(u8s, u8, DTYPE_U8);
    reader!(i64s, i64, DTYPE_I64);
    writer!(f64s_mut, f64, DTYPE_F64);
    writer!(i8s_mut, i8, DTYPE_I8);
    writer!(i32s_mut, i32, DTYPE_I32);
    writer!(i64s_mut, i64, DTYPE_I64);
    writer!(i16s_mut, i16, DTYPE_I16);

    /// Refuse a request whose written buffers overlap each other or any
    /// buffer it reads: two `&mut` views of one allocation are undefined
    /// behaviour, not merely wrong numbers.
    pub fn check_disjoint(&self, written: &[usize]) -> Result<()> {
        let span = |item: &SoArray| {
            let width = match item.dtype {
                DTYPE_F32 | DTYPE_I32 => 4u64,
                DTYPE_I16 => 2,
                DTYPE_F64 | DTYPE_I64 => 8,
                _ => 1,
            };
            let start = item.ptr as u64;
            (start, start + item.len * width)
        };
        for (position, &a) in written.iter().enumerate() {
            let Some(item_a) = self.items.get(a) else { continue };
            if item_a.len == 0 {
                continue;
            }
            let (a0, a1) = span(item_a);
            for (other, item_b) in self.items.iter().enumerate() {
                if other == a || item_b.len == 0 {
                    continue;
                }
                let (b0, b1) = span(item_b);
                if a0 < b1 && b0 < a1 {
                    return Err(SuperobFailure::request(format!(
                        "written array {a} overlaps array {other}"
                    )));
                }
            }
            if written[position + 1..].contains(&a) {
                return Err(SuperobFailure::request(format!("array {a} is written twice")));
            }
        }
        Ok(())
    }
}

/// A float carried as its `repr` string.
pub fn de_f64<'de, D: Deserializer<'de>>(deserializer: D) -> std::result::Result<f64, D::Error> {
    let text = String::deserialize(deserializer)?;
    parse_f64(&text).map_err(serde::de::Error::custom)
}

/// An optional float carried as its `repr` string, or null.
pub fn de_opt_f64<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> std::result::Result<Option<f64>, D::Error> {
    let text = Option::<String>::deserialize(deserializer)?;
    text.map(|t| parse_f64(&t).map_err(serde::de::Error::custom)).transpose()
}

fn parse_f64(text: &str) -> std::result::Result<f64, String> {
    text.trim()
        .parse::<f64>()
        .map_err(|error| format!("{text:?} is not a float: {error}"))
}

/// `SuperobParams`, the float fields (the parameter objects travel apart).
#[derive(Debug, Clone, Deserialize)]
pub struct Params {
    #[serde(deserialize_with = "de_f64")]
    pub nyquist_reject_fraction: f64,
    #[serde(deserialize_with = "de_f64")]
    pub nyquist_min_ms: f64,
    #[serde(deserialize_with = "de_f64")]
    pub nyquist_max_ms: f64,
    #[serde(deserialize_with = "de_f64")]
    pub nyquist_spread_fraction: f64,
    #[serde(deserialize_with = "de_f64")]
    pub shear_fold_fraction: f64,
    #[serde(deserialize_with = "de_f64")]
    pub min_reflectivity_dbz: f64,
    #[serde(deserialize_with = "de_f64")]
    pub max_range_km: f64,
    #[serde(deserialize_with = "de_f64")]
    pub max_elevation_deg: f64,
    #[serde(deserialize_with = "de_f64")]
    pub z_error_base_dbz: f64,
    #[serde(deserialize_with = "de_f64")]
    pub vr_error_base_ms: f64,
    #[serde(deserialize_with = "de_f64")]
    pub z_error_floor_dbz: f64,
    #[serde(deserialize_with = "de_f64")]
    pub vr_error_floor_ms: f64,
    #[serde(deserialize_with = "de_f64")]
    pub refraction_factor: f64,
    #[serde(deserialize_with = "de_f64")]
    pub earth_radius_m: f64,
    #[serde(deserialize_with = "de_f64")]
    pub clear_air_min_gates: f64,
    #[serde(deserialize_with = "de_f64")]
    pub clear_air_error_dbz: f64,
}

impl Params {
    /// The ranges `SuperobParams.validate` enforces, re-checked at the
    /// seam because this is where they are used.
    pub fn validate(&self) -> Result<()> {
        let bad = |name: &str, value: f64, rule: &str| {
            Err(SuperobFailure::request(format!("{name} is {value}; {rule}")))
        };
        for (name, value) in [
            ("nyquist_reject_fraction", self.nyquist_reject_fraction),
            ("nyquist_spread_fraction", self.nyquist_spread_fraction),
            ("shear_fold_fraction", self.shear_fold_fraction),
        ] {
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                return bad(name, value, "a Nyquist fraction lies in [0, 1]");
            }
        }
        for (name, value) in [
            ("nyquist_min_ms", self.nyquist_min_ms),
            ("nyquist_max_ms", self.nyquist_max_ms),
            ("max_range_km", self.max_range_km),
            ("z_error_base_dbz", self.z_error_base_dbz),
            ("vr_error_base_ms", self.vr_error_base_ms),
            ("z_error_floor_dbz", self.z_error_floor_dbz),
            ("vr_error_floor_ms", self.vr_error_floor_ms),
            ("refraction_factor", self.refraction_factor),
            ("earth_radius_m", self.earth_radius_m),
            ("clear_air_min_gates", self.clear_air_min_gates),
            ("clear_air_error_dbz", self.clear_air_error_dbz),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return bad(name, value, "it must be finite and strictly positive");
            }
        }
        if !self.min_reflectivity_dbz.is_finite() {
            return bad("min_reflectivity_dbz", self.min_reflectivity_dbz, "it must be finite");
        }
        if !(self.nyquist_min_ms < self.nyquist_max_ms) {
            return bad("nyquist_min_ms", self.nyquist_min_ms, "it must be below nyquist_max_ms");
        }
        if self.clear_air_min_gates < 1.0 {
            return bad("clear_air_min_gates", self.clear_air_min_gates, "it counts gates, >= 1");
        }
        if !self.max_elevation_deg.is_finite()
            || !(self.max_elevation_deg > 0.0 && self.max_elevation_deg <= 90.0)
        {
            return bad("max_elevation_deg", self.max_elevation_deg, "it lies in (0, 90]");
        }
        Ok(())
    }
}

/// The two `DealiasParams` fields the gridding pass reads.
#[derive(Debug, Clone, Deserialize)]
pub struct DealiasUse {
    pub keep_beyond_reject_fraction: bool,
    #[serde(deserialize_with = "de_f64")]
    pub max_speed_ms: f64,
}

/// `CcQcParams`.
#[derive(Debug, Clone, Deserialize)]
pub struct CcParams {
    #[serde(deserialize_with = "de_f64")]
    pub rho_min: f64,
    #[serde(deserialize_with = "de_f64")]
    pub ref_shield_dbz: f64,
    #[serde(deserialize_with = "de_opt_f64")]
    pub rho_min_velocity: Option<f64>,
    #[serde(deserialize_with = "de_opt_f64")]
    pub rho_floor: Option<f64>,
    pub pair_companion_sweeps: bool,
    #[serde(deserialize_with = "de_f64")]
    pub companion_elevation_tolerance_deg: f64,
    #[serde(deserialize_with = "de_f64")]
    pub companion_azimuth_tolerance_deg: f64,
    pub tds_fringe_exempt: bool,
    #[serde(deserialize_with = "de_f64")]
    pub tds_rho_floor: f64,
    #[serde(deserialize_with = "de_f64")]
    pub tds_ref_min_dbz: f64,
    #[serde(deserialize_with = "de_f64")]
    pub tds_couplet_delta_v_ms: f64,
    #[serde(deserialize_with = "de_f64")]
    pub tds_couplet_lobe_ms: f64,
    #[serde(deserialize_with = "de_f64")]
    pub tds_couplet_max_azimuth_gap_deg: f64,
    pub tds_couplet_min_seeds: i64,
    pub tds_couplet_cluster_radius: i64,
    pub tds_rotation_radius_radials: i64,
    pub tds_rotation_radius_gates: i64,
}

impl CcParams {
    pub fn velocity_threshold(&self) -> f64 {
        self.rho_min_velocity.unwrap_or(self.rho_min)
    }

    pub fn validate(&self) -> Result<()> {
        if self.tds_couplet_min_seeds < 1
            || self.tds_couplet_cluster_radius < 0
            || self.tds_rotation_radius_radials < 0
            || self.tds_rotation_radius_gates < 0
        {
            return Err(SuperobFailure::request(
                "cc_qc counts (seeds >= 1, radii >= 0) are out of range",
            ));
        }
        if !(self.rho_min > 0.0 && self.rho_min <= 1.0) || !self.ref_shield_dbz.is_finite() {
            return Err(SuperobFailure::request("cc_qc rho_min / ref_shield_dbz out of range"));
        }
        Ok(())
    }
}

/// The projection a grid places latitude/longitude with.
#[derive(Debug, Clone, Deserialize)]
pub struct GridIn {
    pub spec: SpecIn,
    #[serde(default)]
    pub translations: Vec<(i64, i64)>,
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub z_w: usize,
}

/// `ProjectedGrid._rust_spec()` with its floats as strings.
#[derive(Debug, Clone, Deserialize)]
pub struct SpecIn {
    pub kind: String,
    #[serde(deserialize_with = "de_f64")]
    pub ref_lat: f64,
    #[serde(deserialize_with = "de_f64")]
    pub ref_lon: f64,
    #[serde(deserialize_with = "de_f64")]
    pub truelat1: f64,
    #[serde(deserialize_with = "de_f64")]
    pub truelat2: f64,
    #[serde(deserialize_with = "de_f64")]
    pub stand_lon: f64,
    #[serde(deserialize_with = "de_f64")]
    pub dx: f64,
    #[serde(deserialize_with = "de_f64")]
    pub dy: f64,
    pub e_we: i64,
    pub e_sn: i64,
    #[serde(deserialize_with = "de_f64")]
    pub known_x: f64,
    #[serde(deserialize_with = "de_f64")]
    pub known_y: f64,
    #[serde(deserialize_with = "de_f64")]
    pub moad_cen_lat: f64,
    #[serde(deserialize_with = "de_f64")]
    pub moad_cen_lon: f64,
}

impl SpecIn {
    pub fn to_spec(&self) -> Result<static_fields::projection::GridSpec> {
        // Through the crate's own serde so the kind spelling is the one it
        // owns.  `json!` stores each f64 as a number value and
        // `from_value` hands it back without printing or parsing text.
        let value = serde_json::json!({
            "kind": self.kind,
            "ref_lat": self.ref_lat, "ref_lon": self.ref_lon,
            "truelat1": self.truelat1, "truelat2": self.truelat2,
            "stand_lon": self.stand_lon, "dx": self.dx, "dy": self.dy,
            "e_we": self.e_we, "e_sn": self.e_sn,
            "known_x": self.known_x, "known_y": self.known_y,
            "moad_cen_lat": self.moad_cen_lat, "moad_cen_lon": self.moad_cen_lon,
        });
        let spec: static_fields::projection::GridSpec = serde_json::from_value(value)
            .map_err(|error| SuperobFailure::request(format!("grid spec: {error}")))?;
        // `from_value` moves the f64 values through serde's data model
        // without printing them, so they arrive unchanged; check anyway,
        // since the whole placement rests on these bits.
        let same = spec.ref_lat.to_bits() == self.ref_lat.to_bits()
            && spec.ref_lon.to_bits() == self.ref_lon.to_bits()
            && spec.truelat1.to_bits() == self.truelat1.to_bits()
            && spec.truelat2.to_bits() == self.truelat2.to_bits()
            && spec.stand_lon.to_bits() == self.stand_lon.to_bits()
            && spec.dx.to_bits() == self.dx.to_bits()
            && spec.known_x.to_bits() == self.known_x.to_bits()
            && spec.known_y.to_bits() == self.known_y.to_bits();
        if !same {
            return Err(SuperobFailure::request("grid spec floats changed crossing serde"));
        }
        Ok(spec)
    }
}
