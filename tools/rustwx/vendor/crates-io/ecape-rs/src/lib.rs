use serde::{Deserialize, Deserializer, Serialize, de};
use std::{error::Error, fmt, str::FromStr};

const RD: f64 = 287.04;
const RV: f64 = 461.5;
const PHI: f64 = RD / RV;
const CPD: f64 = 1005.0;
const CPV: f64 = 1870.0;
const CPL: f64 = 4190.0;
const CPI: f64 = 2106.0;
const G: f64 = 9.81;
const P0: f64 = 100000.0;
const KAPPA: f64 = RD / CPD;
const LV_TRIP: f64 = 2_501_000.0;
const LI_TRIP: f64 = 333_000.0;
const T_TRIP: f64 = 273.15;
const VAPOR_PRES_REF: f64 = 611.2;
const MOLAR_GAS_CONSTANT: f64 = 8.314;
const AVG_MOLAR_MASS: f64 = 0.029;
const DEFAULT_STEP_M: f64 = 20.0;
const KELVIN_OFFSET: f64 = 273.15;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CapeType {
    SurfaceBased,
    MostUnstable,
    MixedLayer,
    UserDefined,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StormMotionType {
    RightMoving,
    LeftMoving,
    MeanWind,
    UserDefined,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseEcapeOptionError {
    option: &'static str,
    value: String,
}

impl ParseEcapeOptionError {
    fn new(option: &'static str, value: &str) -> Self {
        Self {
            option,
            value: value.to_string(),
        }
    }

    pub fn option(&self) -> &'static str {
        self.option
    }

    pub fn value(&self) -> &str {
        &self.value
    }
}

impl fmt::Display for ParseEcapeOptionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid {} value: {}", self.option, self.value)
    }
}

impl Error for ParseEcapeOptionError {}

fn normalized_option_token(value: &str) -> String {
    value.trim().to_ascii_lowercase().replace(['-', ' '], "_")
}

impl CapeType {
    pub fn parse_normalized(value: &str) -> Result<Self, ParseEcapeOptionError> {
        match normalized_option_token(value).as_str() {
            "sb" | "surface" | "surface_based" | "surfacebased" => Ok(Self::SurfaceBased),
            "ml" | "mixed_layer" | "mixedlayer" => Ok(Self::MixedLayer),
            "mu" | "most_unstable" | "mostunstable" => Ok(Self::MostUnstable),
            "user" | "user_defined" => Ok(Self::UserDefined),
            _ => Err(ParseEcapeOptionError::new("cape_type", value)),
        }
    }

    pub fn parse_or_default(value: Option<&str>) -> Self {
        value
            .and_then(|raw| Self::parse_normalized(raw).ok())
            .unwrap_or_default()
    }
}

impl FromStr for CapeType {
    type Err = ParseEcapeOptionError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::parse_normalized(value)
    }
}

impl<'de> Deserialize<'de> for CapeType {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = String::deserialize(deserializer)?;
        Self::parse_normalized(&raw).map_err(de::Error::custom)
    }
}

impl StormMotionType {
    pub fn parse_normalized(value: &str) -> Result<Self, ParseEcapeOptionError> {
        match normalized_option_token(value).as_str() {
            "right_moving" | "right" | "bunkers_right" | "bunkers_rm" | "rm" => {
                Ok(Self::RightMoving)
            }
            "left_moving" | "left" | "bunkers_left" | "bunkers_lm" | "lm" => {
                Ok(Self::LeftMoving)
            }
            "mean_wind" | "meanwind" | "mean" | "mw" => Ok(Self::MeanWind),
            "user" | "user_defined" | "custom" => Ok(Self::UserDefined),
            _ => Err(ParseEcapeOptionError::new("storm_motion_type", value)),
        }
    }

    pub fn parse_or_default(value: Option<&str>) -> Self {
        value
            .and_then(|raw| Self::parse_normalized(raw).ok())
            .unwrap_or_default()
    }
}

impl FromStr for StormMotionType {
    type Err = ParseEcapeOptionError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::parse_normalized(value)
    }
}

impl<'de> Deserialize<'de> for StormMotionType {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = String::deserialize(deserializer)?;
        Self::parse_normalized(&raw).map_err(de::Error::custom)
    }
}

impl Default for CapeType {
    fn default() -> Self {
        Self::SurfaceBased
    }
}

impl Default for StormMotionType {
    fn default() -> Self {
        Self::RightMoving
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParcelOptions {
    #[serde(default)]
    pub cape_type: CapeType,
    #[serde(default)]
    pub storm_motion_type: StormMotionType,
    #[serde(default)]
    pub origin_pressure_pa: Option<f64>,
    #[serde(default)]
    pub origin_height_m: Option<f64>,
    #[serde(default)]
    pub mixed_layer_depth_pa: Option<f64>,
    #[serde(default)]
    pub inflow_layer_bottom_m: Option<f64>,
    #[serde(default)]
    pub inflow_layer_top_m: Option<f64>,
    #[serde(default)]
    pub storm_motion_u_ms: Option<f64>,
    #[serde(default)]
    pub storm_motion_v_ms: Option<f64>,
    #[serde(default)]
    pub entrainment_rate: Option<f64>,
    #[serde(default)]
    pub pseudoadiabatic: Option<bool>,
}

impl Default for ParcelOptions {
    fn default() -> Self {
        Self {
            cape_type: CapeType::SurfaceBased,
            storm_motion_type: StormMotionType::RightMoving,
            origin_pressure_pa: None,
            origin_height_m: None,
            mixed_layer_depth_pa: Some(10000.0),
            inflow_layer_bottom_m: Some(0.0),
            inflow_layer_top_m: Some(1000.0),
            storm_motion_u_ms: None,
            storm_motion_v_ms: None,
            entrainment_rate: None,
            pseudoadiabatic: Some(true),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParcelProfile {
    pub pressure_pa: Vec<f64>,
    pub height_m: Vec<f64>,
    pub temperature_k: Vec<f64>,
    pub qv_kgkg: Vec<f64>,
    pub qt_kgkg: Vec<f64>,
    pub buoyancy_ms2: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapeCinLfcEl {
    pub cape_jkg: f64,
    pub cin_jkg: f64,
    pub lfc_m: Option<f64>,
    pub el_m: Option<f64>,
    pub origin_index: usize,
    pub pressure_pa: Vec<f64>,
    pub height_m: Vec<f64>,
    pub parcel_temperature_k: Vec<f64>,
    pub buoyancy_ms2: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EcapeNcape {
    pub ecape_jkg: f64,
    pub ncape_jkg: f64,
    pub cape_jkg: f64,
    pub lfc_m: Option<f64>,
    pub el_m: Option<f64>,
    pub storm_motion_u_ms: f64,
    pub storm_motion_v_ms: f64,
    pub storm_relative_wind_ms: f64,
    pub psi: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EcapeParcelResult {
    pub ecape_jkg: f64,
    pub ncape_jkg: f64,
    pub cape_jkg: f64,
    pub cin_jkg: f64,
    pub lfc_m: Option<f64>,
    pub el_m: Option<f64>,
    pub storm_motion_u_ms: f64,
    pub storm_motion_v_ms: f64,
    pub parcel_profile: ParcelProfile,
}

#[derive(Debug, Clone)]
struct ParcelOriginState {
    index: usize,
    theta_override_k: Option<f64>,
    qv_override_kgkg: Option<f64>,
    height_override_m: Option<f64>,
}

#[derive(Debug)]
pub enum EcapeError {
    DimensionMismatch,
    EmptyProfile,
    NonMonotonicPressure,
    NonMonotonicHeight,
    NonFiniteInput,
    OriginNotFound,
}

impl std::fmt::Display for EcapeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DimensionMismatch => write!(f, "profile arrays must have the same length"),
            Self::EmptyProfile => write!(f, "profile is empty"),
            Self::NonMonotonicPressure => write!(f, "pressure must decrease monotonically"),
            Self::NonMonotonicHeight => write!(f, "height must increase monotonically"),
            Self::NonFiniteInput => write!(f, "profile contains non-finite values"),
            Self::OriginNotFound => write!(f, "could not determine parcel origin"),
        }
    }
}

impl std::error::Error for EcapeError {}

fn validate_profile(
    height_m: &[f64],
    pressure_pa: &[f64],
    temperature_k: &[f64],
    qv_kgkg: &[f64],
    u_wind_ms: &[f64],
    v_wind_ms: &[f64],
) -> Result<(), EcapeError> {
    let n = height_m.len();
    if n == 0 {
        return Err(EcapeError::EmptyProfile);
    }
    if pressure_pa.len() != n
        || temperature_k.len() != n
        || qv_kgkg.len() != n
        || u_wind_ms.len() != n
        || v_wind_ms.len() != n
    {
        return Err(EcapeError::DimensionMismatch);
    }
    for i in 0..n {
        if !height_m[i].is_finite()
            || !pressure_pa[i].is_finite()
            || !temperature_k[i].is_finite()
            || !qv_kgkg[i].is_finite()
            || !u_wind_ms[i].is_finite()
            || !v_wind_ms[i].is_finite()
        {
            return Err(EcapeError::NonFiniteInput);
        }
        if i > 0 {
            if height_m[i] <= height_m[i - 1] {
                return Err(EcapeError::NonMonotonicHeight);
            }
            if pressure_pa[i] >= pressure_pa[i - 1] {
                return Err(EcapeError::NonMonotonicPressure);
            }
        }
    }
    Ok(())
}

fn clamp01(x: f64) -> f64 {
    x.clamp(0.0, 1.0)
}

fn linear_interp(x0: f64, x1: f64, y0: f64, y1: f64, x: f64) -> f64 {
    if (x1 - x0).abs() < 1e-12 {
        return y0;
    }
    y0 + (y1 - y0) * (x - x0) / (x1 - x0)
}

fn reverse_linear_interp(xs: &[f64], ys: &[f64], target: f64) -> Option<f64> {
    for i in 1..xs.len() {
        let y0 = ys[i - 1];
        let y1 = ys[i];
        if (target >= y0 && target <= y1) || (target >= y1 && target <= y0) {
            return Some(linear_interp(y0, y1, xs[i - 1], xs[i], target));
        }
    }
    None
}

fn find_bracketing_index_desc(values: &[f64], target: f64) -> usize {
    if target >= values[0] {
        return 0;
    }
    for i in 1..values.len() {
        if target >= values[i] {
            return i - 1;
        }
    }
    values.len() - 2
}

fn find_bracketing_index_asc(values: &[f64], target: f64) -> usize {
    if target <= values[0] {
        return 0;
    }
    for i in 1..values.len() {
        if target <= values[i] {
            return i - 1;
        }
    }
    values.len() - 2
}

fn interp_pressure_to_height(heights: &[f64], pressures: &[f64], z: f64) -> f64 {
    let i = find_bracketing_index_asc(heights, z);
    linear_interp(
        heights[i],
        heights[i + 1],
        pressures[i],
        pressures[i + 1],
        z,
    )
}

fn interp_height_to_pressure(pressures: &[f64], heights: &[f64], p: f64) -> f64 {
    let i = find_bracketing_index_desc(pressures, p);
    linear_interp(
        pressures[i],
        pressures[i + 1],
        heights[i],
        heights[i + 1],
        p,
    )
}

fn interp_profile_at_height(heights: &[f64], values: &[f64], z: f64) -> f64 {
    if z <= heights[0] {
        return values[0];
    }
    if z >= heights[heights.len() - 1] {
        return values[values.len() - 1];
    }
    let i = find_bracketing_index_asc(heights, z);
    linear_interp(heights[i], heights[i + 1], values[i], values[i + 1], z)
}

fn saturation_vapor_pressure_water(temp_k: f64) -> f64 {
    let latent_heat = LV_TRIP - (CPL - CPV) * (temp_k - T_TRIP);
    let heat_power = (CPL - CPV) / RV;
    let exp_term = ((LV_TRIP / T_TRIP - latent_heat / temp_k) / RV).exp();
    VAPOR_PRES_REF * (T_TRIP / temp_k).powf(heat_power) * exp_term
}

fn omega(temp_k: f64, warmest_mixed_phase_temp: f64, coldest_mixed_phase_temp: f64) -> f64 {
    if temp_k >= warmest_mixed_phase_temp {
        0.0
    } else if temp_k <= coldest_mixed_phase_temp {
        1.0
    } else {
        (temp_k - warmest_mixed_phase_temp) / (coldest_mixed_phase_temp - warmest_mixed_phase_temp)
    }
}

fn omega_deriv(temp_k: f64, warmest_mixed_phase_temp: f64, coldest_mixed_phase_temp: f64) -> f64 {
    if temp_k >= warmest_mixed_phase_temp || temp_k <= coldest_mixed_phase_temp {
        0.0
    } else {
        1.0 / (coldest_mixed_phase_temp - warmest_mixed_phase_temp)
    }
}

fn r_sat(temp_k: f64, pressure_pa: f64, ice_flag: i32) -> f64 {
    let warm = 273.15;
    let cold = 253.15;
    let omeg = omega(temp_k, warm, cold);
    if ice_flag == 0 {
        let term1 = (CPV - CPL) / RV;
        let term2 = (LV_TRIP - T_TRIP * (CPV - CPL)) / RV;
        let esl = ((temp_k - T_TRIP) * term2 / (temp_k * T_TRIP)).exp()
            * VAPOR_PRES_REF
            * (temp_k / T_TRIP).powf(term1);
        PHI * esl / (pressure_pa - esl).max(1e-9)
    } else if ice_flag == 1 {
        let qsat_l = r_sat(temp_k, pressure_pa, 0);
        let qsat_i = r_sat(temp_k, pressure_pa, 2);
        (1.0 - omeg) * qsat_l + omeg * qsat_i
    } else {
        let term1 = (CPV - CPI) / RV;
        let term2 = (LV_TRIP - T_TRIP * (CPV - CPI)) / RV;
        let esl = ((temp_k - T_TRIP) * term2 / (temp_k * T_TRIP)).exp()
            * VAPOR_PRES_REF
            * (temp_k / T_TRIP).powf(term1);
        PHI * esl / (pressure_pa - esl).max(1e-9)
    }
}

fn vapor_pressure_from_specific_humidity(pressure_pa: f64, qv_kgkg: f64) -> f64 {
    pressure_pa * qv_kgkg / (PHI + (1.0 - PHI) * qv_kgkg)
}

fn specific_humidity_from_vapor_pressure(pressure_pa: f64, vapor_pressure_pa: f64) -> f64 {
    PHI * vapor_pressure_pa / (pressure_pa - (1.0 - PHI) * vapor_pressure_pa)
}

fn dewpoint_from_vapor_pressure(vapor_pressure_pa: f64) -> f64 {
    let ln_ratio = (vapor_pressure_pa / 611.2).ln();
    let td_c = 243.5 * ln_ratio / (17.67 - ln_ratio);
    td_c + 273.15
}

fn dewpoint_from_specific_humidity(pressure_pa: f64, qv_kgkg: f64) -> f64 {
    dewpoint_from_vapor_pressure(vapor_pressure_from_specific_humidity(pressure_pa, qv_kgkg))
}

fn specific_humidity_from_dewpoint(pressure_pa: f64, dewpoint_k: f64) -> f64 {
    let vapor_pressure_pa =
        611.2 * ((17.67 * (dewpoint_k - KELVIN_OFFSET)) / (dewpoint_k - 29.65)).exp();
    specific_humidity_from_vapor_pressure(pressure_pa, vapor_pressure_pa)
}

fn potential_temperature(temp_k: f64, pressure_pa: f64) -> f64 {
    temp_k * (P0 / pressure_pa).powf(KAPPA)
}

fn temperature_from_potential_temperature(theta_k: f64, pressure_pa: f64) -> f64 {
    theta_k * (pressure_pa / P0).powf(KAPPA)
}

fn density_temperature(temp_k: f64, qv_kgkg: f64, qt_kgkg: f64) -> f64 {
    temp_k * (1.0 - qt_kgkg + qv_kgkg / PHI)
}

fn equivalent_potential_temperature(temp_k: f64, dewpoint_k: f64, pressure_pa: f64) -> f64 {
    let q = specific_humidity_from_dewpoint(pressure_pa, dewpoint_k);
    let e = 611.2
        * ((17.67 * (dewpoint_k - KELVIN_OFFSET)) / ((dewpoint_k - KELVIN_OFFSET) + 243.5)).exp();
    let w = PHI * e / (pressure_pa - e).max(1e-9);
    let tlcl = 1.0
        / (1.0 / (dewpoint_k - 56.0).max(1e-6) + ((temp_k / dewpoint_k).max(1e-9)).ln() / 800.0)
        + 56.0;
    let theta_l = temp_k * (P0 / pressure_pa).powf(0.2854 * (1.0 - 0.28 * q));
    theta_l * (((3376.0 / tlcl) - 2.54) * w * (1.0 + 0.81 * w)).exp()
}

fn closest_index(values: &[f64], target: f64) -> usize {
    values
        .iter()
        .enumerate()
        .min_by(|a, b| {
            (a.1 - target)
                .abs()
                .partial_cmp(&(b.1 - target).abs())
                .unwrap()
        })
        .map(|(idx, _)| idx)
        .unwrap_or(0)
}

fn interp_log_pressure(target_hpa: f64, pressure_hpa: &[f64], values: &[f64]) -> f64 {
    let n = pressure_hpa.len();
    if n == 0 {
        return 0.0;
    }
    if target_hpa >= pressure_hpa[0] {
        return values[0];
    }
    if target_hpa <= pressure_hpa[n - 1] {
        return values[n - 1];
    }
    for i in 1..n {
        if pressure_hpa[i] <= target_hpa {
            let log_p0 = pressure_hpa[i - 1].ln();
            let log_p1 = pressure_hpa[i].ln();
            let log_pt = target_hpa.ln();
            let frac = (log_pt - log_p0) / (log_p1 - log_p0);
            return values[i - 1] + frac * (values[i] - values[i - 1]);
        }
    }
    values[n - 1]
}

fn interp_linear_pressure(target_hpa: f64, pressure_hpa: &[f64], values: &[f64]) -> f64 {
    let n = pressure_hpa.len();
    if n == 0 {
        return 0.0;
    }
    if target_hpa >= pressure_hpa[0] {
        return values[0];
    }
    if target_hpa <= pressure_hpa[n - 1] {
        return values[n - 1];
    }
    for i in 1..n {
        if pressure_hpa[i] <= target_hpa {
            let p0 = pressure_hpa[i - 1];
            let p1 = pressure_hpa[i];
            let frac = (target_hpa - p0) / (p1 - p0);
            return values[i - 1] + frac * (values[i] - values[i - 1]);
        }
    }
    values[n - 1]
}

fn log_pressure_intersections_direction(
    pressure_hpa: &[f64],
    y1: &[f64],
    y2: &[f64],
    direction: &str,
) -> (Vec<f64>, Vec<f64>) {
    let mut x_out = Vec::new();
    let mut y_out = Vec::new();
    let tol = 1e-9;
    for i in 0..pressure_hpa.len().saturating_sub(1) {
        let d0 = y1[i] - y2[i];
        let d1 = y1[i + 1] - y2[i + 1];
        if !(pressure_hpa[i].is_finite()
            && pressure_hpa[i + 1].is_finite()
            && d0.is_finite()
            && d1.is_finite())
        {
            continue;
        }

        let mut crossing = match direction {
            "increasing" => (d0 <= 0.0 && d1 > 0.0) || (d0 < 0.0 && d1 >= 0.0),
            "decreasing" => (d0 >= 0.0 && d1 < 0.0) || (d0 > 0.0 && d1 <= 0.0),
            _ => false,
        };
        if !crossing && d0.abs() <= tol {
            if direction == "increasing" && d1 > 0.0 {
                crossing = true;
            } else if direction == "decreasing" && d1 < 0.0 {
                crossing = true;
            }
        }
        if !crossing {
            continue;
        }

        let frac = if (d1 - d0).abs() <= tol {
            0.0
        } else {
            (-d0 / (d1 - d0)).clamp(0.0, 1.0)
        };
        let log_px =
            pressure_hpa[i].ln() + frac * (pressure_hpa[i + 1].ln() - pressure_hpa[i].ln());
        let x_val = log_px.exp();
        let y_val = y1[i] + frac * (y1[i + 1] - y1[i]);
        if x_out
            .last()
            .is_some_and(|last| (x_val - last).abs() <= 1e-6)
        {
            continue;
        }
        x_out.push(x_val);
        y_out.push(y_val);
    }
    (x_out, y_out)
}

fn find_log_pressure_intersections_native(
    pressure_hpa: &[f64],
    profile_a: &[f64],
    profile_b: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let mut x_out = Vec::new();
    let mut y_out = Vec::new();
    let tol = 1e-12;
    for idx in 0..pressure_hpa.len().saturating_sub(1) {
        let p0 = pressure_hpa[idx];
        let p1 = pressure_hpa[idx + 1];
        let d0 = profile_a[idx] - profile_b[idx];
        let d1 = profile_a[idx + 1] - profile_b[idx + 1];
        if !(p0.is_finite() && p1.is_finite() && d0.is_finite() && d1.is_finite()) {
            continue;
        }
        if d0.abs() <= tol {
            x_out.push(p0);
            y_out.push(profile_b[idx]);
            continue;
        }
        if d0 * d1 > 0.0 {
            continue;
        }

        let log_p0 = p0.ln();
        let log_p1 = p1.ln();
        let log_px = log_p0 - d0 * (log_p1 - log_p0) / (d1 - d0);
        let frac = (log_px - log_p0) / (log_p1 - log_p0);
        x_out.push(log_px.exp());
        y_out.push(profile_b[idx] + frac * (profile_b[idx + 1] - profile_b[idx]));
    }
    (x_out, y_out)
}

fn select_profile_intersection(
    pressures_hpa: &[f64],
    temperatures_c: &[f64],
    which: &str,
) -> (Option<f64>, Option<f64>) {
    if which == "all" {
        return (
            pressures_hpa.first().copied(),
            temperatures_c.first().copied(),
        );
    }
    if pressures_hpa.is_empty() {
        return (None, None);
    }
    let idx = match which {
        "bottom" => 0,
        "top" => pressures_hpa.len() - 1,
        _ => pressures_hpa.len() - 1,
    };
    (Some(pressures_hpa[idx]), Some(temperatures_c[idx]))
}

fn multiple_el_lfc_options_native(
    intersect_pressures_hpa: &[f64],
    intersect_temperatures_c: &[f64],
    valid_mask: &[bool],
    which: &str,
) -> (Option<f64>, Option<f64>) {
    let mut p_list = Vec::new();
    let mut t_list = Vec::new();
    for i in 0..intersect_pressures_hpa.len() {
        if valid_mask.get(i).copied().unwrap_or(false) {
            p_list.push(intersect_pressures_hpa[i]);
            t_list.push(intersect_temperatures_c[i]);
        }
    }
    match which {
        "top" | "bottom" | "all" => select_profile_intersection(&p_list, &t_list, which),
        _ => select_profile_intersection(&p_list, &t_list, "top"),
    }
}

fn lfc_native_profile(
    pressure_hpa: &[f64],
    temperature_c: &[f64],
    dewpoint_c: &[f64],
    parcel_temperature_c: &[f64],
    dewpoint_start_c: f64,
    which: &str,
) -> (Option<f64>, Option<f64>) {
    let (x, y) = if (parcel_temperature_c[0] - temperature_c[0]).abs() <= 1e-9 {
        log_pressure_intersections_direction(
            &pressure_hpa[1..],
            &parcel_temperature_c[1..],
            &temperature_c[1..],
            "increasing",
        )
    } else {
        log_pressure_intersections_direction(
            pressure_hpa,
            parcel_temperature_c,
            temperature_c,
            "increasing",
        )
    };

    let (lcl_p, lcl_t) =
        wx_math::thermo::drylift(pressure_hpa[0], parcel_temperature_c[0], dewpoint_start_c);
    if x.is_empty() {
        let mask: Vec<usize> = pressure_hpa
            .iter()
            .enumerate()
            .filter_map(|(i, p)| if *p < lcl_p { Some(i) } else { None })
            .collect();
        if mask
            .iter()
            .all(|&i| parcel_temperature_c[i] <= temperature_c[i] + 1e-9)
        {
            return (None, None);
        }
        return (Some(lcl_p), Some(lcl_t));
    }
    let valid: Vec<bool> = x.iter().map(|p| *p < lcl_p).collect();
    if !valid.iter().any(|v| *v) {
        let (el_x, _) = log_pressure_intersections_direction(
            &pressure_hpa[1..],
            &parcel_temperature_c[1..],
            &temperature_c[1..],
            "decreasing",
        );
        if !el_x.is_empty() && el_x.iter().copied().fold(f64::INFINITY, f64::min) > lcl_p {
            return (None, None);
        }
        return (Some(lcl_p), Some(lcl_t));
    }
    multiple_el_lfc_options_native(&x, &y, &valid, which)
}

fn el_native_profile(
    pressure_hpa: &[f64],
    temperature_c: &[f64],
    dewpoint_c: &[f64],
    parcel_temperature_c: &[f64],
    which: &str,
) -> (Option<f64>, Option<f64>) {
    if parcel_temperature_c.last().copied().unwrap_or(f64::NAN)
        > temperature_c.last().copied().unwrap_or(f64::INFINITY)
    {
        return (None, None);
    }
    let (x, y) = log_pressure_intersections_direction(
        &pressure_hpa[1..],
        &parcel_temperature_c[1..],
        &temperature_c[1..],
        "decreasing",
    );
    let (lcl_p, _) = wx_math::thermo::drylift(pressure_hpa[0], temperature_c[0], dewpoint_c[0]);
    let valid: Vec<bool> = x.iter().map(|p| *p < lcl_p).collect();
    if valid.iter().any(|v| *v) {
        multiple_el_lfc_options_native(&x, &y, &valid, which)
    } else {
        (None, None)
    }
}

fn append_zero_crossings_native(
    pressure_hpa: &[f64],
    profile_delta: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let zero_profile = vec![0.0; profile_delta.len()];
    let (crossings_p, crossings_y) = find_log_pressure_intersections_native(
        &pressure_hpa[1..],
        &profile_delta[1..],
        &zero_profile[1..],
    );
    let mut pressure_vals = pressure_hpa.to_vec();
    pressure_vals.extend(crossings_p);
    let mut profile_vals = profile_delta.to_vec();
    profile_vals.extend(crossings_y);
    let mut paired: Vec<(f64, f64)> = pressure_vals
        .into_iter()
        .zip(profile_vals.into_iter())
        .collect();
    paired.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    paired.dedup_by(|a, b| (a.0 - b.0).abs() <= 1e-6);
    paired.into_iter().unzip()
}

fn trapz_log_pressure(x: &[f64], y: &[f64]) -> f64 {
    if x.len() < 2 || y.len() < 2 {
        return 0.0;
    }
    let mut total = 0.0;
    for i in 1..x.len() {
        total += 0.5 * (y[i - 1] + y[i]) * (x[i].ln() - x[i - 1].ln());
    }
    total
}

fn cape_cin_profile_native(
    pressure_hpa: &[f64],
    temperature_c: &[f64],
    dewpoint_c: &[f64],
    parcel_temperature_c: &[f64],
    which_lfc: &str,
    which_el: &str,
) -> (f64, f64) {
    let (lcl_p, _) = wx_math::thermo::drylift(pressure_hpa[0], temperature_c[0], dewpoint_c[0]);
    let parcel_start_mixing_ratio =
        wx_math::thermo::saturation_mixing_ratio(pressure_hpa[0], dewpoint_c[0]) / 1000.0;
    let below_lcl: Vec<bool> = pressure_hpa.iter().map(|p| *p > lcl_p).collect();
    let parcel_mixing_ratio: Vec<f64> = pressure_hpa
        .iter()
        .enumerate()
        .map(|(i, p)| {
            if below_lcl[i] {
                parcel_start_mixing_ratio
            } else {
                wx_math::thermo::saturation_mixing_ratio(*p, parcel_temperature_c[i]) / 1000.0
            }
        })
        .collect();
    let env_mixing_ratio: Vec<f64> = pressure_hpa
        .iter()
        .zip(dewpoint_c.iter())
        .map(|(p, td)| wx_math::thermo::saturation_mixing_ratio(*p, *td) / 1000.0)
        .collect();
    let env_virtual: Vec<f64> = temperature_c
        .iter()
        .zip(env_mixing_ratio.iter())
        .map(|(t, w)| {
            let t_k = *t + KELVIN_OFFSET;
            let tv_k = t_k * (1.0 + *w / wx_math::thermo::EPS) / (1.0 + *w);
            tv_k - KELVIN_OFFSET
        })
        .collect();
    let parcel_virtual: Vec<f64> = parcel_temperature_c
        .iter()
        .zip(parcel_mixing_ratio.iter())
        .map(|(t, w)| {
            let t_k = *t + KELVIN_OFFSET;
            let tv_k = t_k * (1.0 + *w / wx_math::thermo::EPS) / (1.0 + *w);
            tv_k - KELVIN_OFFSET
        })
        .collect();

    let (lfc_pressure_hpa, _) = lfc_native_profile(
        pressure_hpa,
        &env_virtual,
        dewpoint_c,
        &parcel_virtual,
        dewpoint_c[0],
        which_lfc,
    );
    let Some(lfc_pressure_hpa) = lfc_pressure_hpa else {
        return (0.0, 0.0);
    };
    let (el_pressure_hpa, _) = el_native_profile(
        pressure_hpa,
        &env_virtual,
        dewpoint_c,
        &parcel_virtual,
        which_el,
    );
    let el_pressure_hpa =
        el_pressure_hpa.unwrap_or(*pressure_hpa.last().unwrap_or(&lfc_pressure_hpa));

    let delta: Vec<f64> = parcel_virtual
        .iter()
        .zip(env_virtual.iter())
        .map(|(p, e)| p - e)
        .collect();
    let (x_vals, y_vals) = append_zero_crossings_native(pressure_hpa, &delta);

    let mut cape_x = Vec::new();
    let mut cape_y = Vec::new();
    let mut cin_x = Vec::new();
    let mut cin_y = Vec::new();
    for i in 0..x_vals.len() {
        if x_vals[i] <= lfc_pressure_hpa + 1e-9 && x_vals[i] >= el_pressure_hpa - 1e-9 {
            cape_x.push(x_vals[i]);
            cape_y.push(y_vals[i]);
        }
        if x_vals[i] >= lfc_pressure_hpa - 1e-9 {
            cin_x.push(x_vals[i]);
            cin_y.push(y_vals[i]);
        }
    }

    let cape = wx_math::thermo::RD * trapz_log_pressure(&cape_x, &cape_y);
    let mut cin = wx_math::thermo::RD * trapz_log_pressure(&cin_x, &cin_y);
    if cin > 0.0 {
        cin = 0.0;
    }
    (cape, cin)
}

fn snap_height_to_last_level_below_pressure(
    pressure_hpa: &[f64],
    height_m: &[f64],
    crossing_hpa: Option<f64>,
) -> Option<f64> {
    let target = crossing_hpa?;
    let mut last_idx = None;
    for (idx, pressure) in pressure_hpa.iter().enumerate() {
        if *pressure > target {
            last_idx = Some(idx);
        } else {
            break;
        }
    }
    Some(height_m[last_idx.unwrap_or(0)])
}

fn profile_to_met_inputs(
    pressure_pa: &[f64],
    temperature_k: &[f64],
    qv_kgkg: &[f64],
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let pressure_hpa: Vec<f64> = pressure_pa.iter().map(|p| p / 100.0).collect();
    let temperature_c: Vec<f64> = temperature_k.iter().map(|t| t - KELVIN_OFFSET).collect();
    let dewpoint_c: Vec<f64> = pressure_hpa
        .iter()
        .zip(qv_kgkg.iter())
        .map(|(p, q)| metrust::calc::thermo::dewpoint_from_specific_humidity(*p, *q))
        .collect();
    (pressure_hpa, temperature_c, dewpoint_c)
}

fn metpy_style_mixed_parcel_start(
    pressure_hpa: &[f64],
    temperature_c: &[f64],
    dewpoint_c: &[f64],
    depth_hpa: f64,
) -> (f64, f64, f64) {
    let theta_k: Vec<f64> = pressure_hpa
        .iter()
        .zip(temperature_c.iter())
        .map(|(p, t)| wx_math::thermo::potential_temperature(*p, *t))
        .collect();
    let mixing_ratio_kgkg: Vec<f64> = pressure_hpa
        .iter()
        .zip(dewpoint_c.iter())
        .map(|(p, td)| wx_math::thermo::saturation_mixing_ratio(*p, *td) / 1000.0)
        .collect();

    let parcel_pressure_hpa = pressure_hpa[0];
    let mean_theta_k = wx_math::thermo::mixed_layer(pressure_hpa, &theta_k, depth_hpa);
    let mean_mixing_ratio_kgkg =
        wx_math::thermo::mixed_layer(pressure_hpa, &mixing_ratio_kgkg, depth_hpa);
    let parcel_temperature_c =
        mean_theta_k * wx_math::thermo::exner_function(parcel_pressure_hpa) - KELVIN_OFFSET;
    let parcel_dewpoint_c =
        wx_math::thermo::temp_at_mixrat(mean_mixing_ratio_kgkg * 1000.0, parcel_pressure_hpa);

    (parcel_pressure_hpa, parcel_temperature_c, parcel_dewpoint_c)
}

fn parcel_profile_with_lcl_native(
    pressure_hpa: &[f64],
    temperature_c: &[f64],
    dewpoint_c: &[f64],
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    if pressure_hpa.is_empty() {
        return (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    }

    let (p_lcl, _) = wx_math::thermo::drylift(pressure_hpa[0], temperature_c[0], dewpoint_c[0]);
    let insert_idx = pressure_hpa
        .iter()
        .position(|p| *p <= p_lcl)
        .unwrap_or(pressure_hpa.len());

    let mut pressure_out = pressure_hpa.to_vec();
    pressure_out.insert(insert_idx, p_lcl);

    let mut temperature_out = temperature_c.to_vec();
    temperature_out.insert(
        insert_idx,
        interp_linear_pressure(p_lcl, pressure_hpa, temperature_c),
    );

    let mut dewpoint_out = dewpoint_c.to_vec();
    dewpoint_out.insert(
        insert_idx,
        interp_linear_pressure(p_lcl, pressure_hpa, dewpoint_c),
    );

    let parcel_out =
        wx_math::thermo::parcel_profile(&pressure_out, temperature_c[0], dewpoint_c[0]);

    (pressure_out, temperature_out, dewpoint_out, parcel_out)
}

fn package_style_mixed_layer_cape_cin_lfc_el(
    height_m: &[f64],
    pressure_hpa: &[f64],
    temperature_c: &[f64],
    dewpoint_c: &[f64],
    options: &ParcelOptions,
) -> CapeCinLfcEl {
    let depth_hpa = options.mixed_layer_depth_pa.unwrap_or(10000.0) / 100.0;
    let (parcel_pressure_hpa, parcel_temperature_c, parcel_dewpoint_c) =
        metpy_style_mixed_parcel_start(pressure_hpa, temperature_c, dewpoint_c, depth_hpa);

    let top_pressure_hpa = pressure_hpa[0] - depth_hpa;
    let mut pressure_prof = vec![parcel_pressure_hpa];
    let mut temperature_prof = vec![parcel_temperature_c];
    let mut dewpoint_prof = vec![parcel_dewpoint_c];

    for i in 0..pressure_hpa.len() {
        if pressure_hpa[i] < top_pressure_hpa {
            pressure_prof.push(pressure_hpa[i]);
            temperature_prof.push(temperature_c[i]);
            dewpoint_prof.push(dewpoint_c[i]);
        }
    }

    let (pressure_prof, temperature_prof, dewpoint_prof, parcel_prof) =
        parcel_profile_with_lcl_native(&pressure_prof, &temperature_prof, &dewpoint_prof);
    let (cape_jkg, cin_jkg) = cape_cin_profile_native(
        &pressure_prof,
        &temperature_prof,
        &dewpoint_prof,
        &parcel_prof,
        "bottom",
        "top",
    );
    let (lfc_p, _) = lfc_native_profile(
        &pressure_prof,
        &temperature_prof,
        &dewpoint_prof,
        &parcel_prof,
        dewpoint_prof[0],
        "top",
    );
    let (el_p, _) = el_native_profile(
        &pressure_prof,
        &temperature_prof,
        &dewpoint_prof,
        &parcel_prof,
        "top",
    );

    CapeCinLfcEl {
        cape_jkg,
        cin_jkg,
        lfc_m: snap_height_to_last_level_below_pressure(pressure_hpa, height_m, lfc_p),
        el_m: snap_height_to_last_level_below_pressure(pressure_hpa, height_m, el_p),
        origin_index: 0,
        pressure_pa: Vec::new(),
        height_m: Vec::new(),
        parcel_temperature_k: Vec::new(),
        buoyancy_ms2: Vec::new(),
    }
}

fn reference_parcel_start(
    _heights: &[f64],
    pressure_pa: &[f64],
    temperature_k: &[f64],
    qv_kgkg: &[f64],
    options: &ParcelOptions,
) -> (usize, f64, f64, f64) {
    let (pressure_hpa, temperature_c, dewpoint_c) =
        profile_to_met_inputs(pressure_pa, temperature_k, qv_kgkg);
    match options.cape_type {
        CapeType::MixedLayer => {
            let depth_hpa = options.mixed_layer_depth_pa.unwrap_or(10000.0) / 100.0;
            let (p_start, t_start, td_start) = metpy_style_mixed_parcel_start(
                &pressure_hpa,
                &temperature_c,
                &dewpoint_c,
                depth_hpa,
            );
            (0, p_start, t_start, td_start)
        }
        CapeType::MostUnstable => {
            let (p_start, t_start, td_start) = metrust::calc::thermo::get_most_unstable_parcel(
                &pressure_hpa,
                &temperature_c,
                &dewpoint_c,
                300.0,
            );
            (
                closest_index(&pressure_hpa, p_start),
                p_start,
                t_start,
                td_start,
            )
        }
        _ => (0, pressure_hpa[0], temperature_c[0], dewpoint_c[0]),
    }
}

fn reference_crossing_pressures(
    pressure_hpa: &[f64],
    temperature_c: &[f64],
    dewpoint_c: &[f64],
    p_start_hpa: f64,
    t_start_c: f64,
    td_start_c: f64,
) -> (Option<f64>, Option<f64>) {
    let (p_lcl, t_lcl) = wx_math::thermo::drylift(p_start_hpa, t_start_c, td_start_c);
    let theta_dry_k = (t_start_c + wx_math::thermo::ZEROCNK)
        * ((1000.0_f64 / p_start_hpa).powf(wx_math::thermo::ROCP));
    let theta_l_k =
        (t_lcl + wx_math::thermo::ZEROCNK) * ((1000.0_f64 / p_lcl).powf(wx_math::thermo::ROCP));
    let theta_l_c = theta_l_k - wx_math::thermo::ZEROCNK;
    let thetam = theta_l_c - wx_math::thermo::wobf(theta_l_c) + wx_math::thermo::wobf(t_lcl);
    let mixing_ratio_kgkg = metrust::calc::thermo::mixing_ratio(p_start_hpa, td_start_c) / 1000.0;

    let mut parcel_tv = vec![0.0; pressure_hpa.len()];
    for i in 0..pressure_hpa.len() {
        if pressure_hpa[i] >= p_lcl {
            let t_parcel_k =
                theta_dry_k * ((pressure_hpa[i] / 1000.0_f64).powf(wx_math::thermo::ROCP));
            let t_parcel_c = t_parcel_k - wx_math::thermo::ZEROCNK;
            parcel_tv[i] = (t_parcel_c + wx_math::thermo::ZEROCNK)
                * (1.0 + mixing_ratio_kgkg / wx_math::thermo::EPS)
                / (1.0 + mixing_ratio_kgkg)
                - wx_math::thermo::ZEROCNK;
        } else {
            let t_parcel_c = wx_math::thermo::satlift(pressure_hpa[i], thetam);
            parcel_tv[i] =
                metrust::calc::thermo::virtual_temp(t_parcel_c, pressure_hpa[i], t_parcel_c);
        }
    }

    let mut lfc_p = None;
    for i in 1..pressure_hpa.len() {
        if pressure_hpa[i] > p_lcl {
            continue;
        }
        let buoy_prev = parcel_tv[i - 1]
            - metrust::calc::thermo::virtual_temp(
                temperature_c[i - 1],
                pressure_hpa[i - 1],
                dewpoint_c[i - 1],
            );
        let buoy = parcel_tv[i]
            - metrust::calc::thermo::virtual_temp(temperature_c[i], pressure_hpa[i], dewpoint_c[i]);
        if buoy_prev <= 0.0 && buoy > 0.0 {
            let frac = (0.0 - buoy_prev) / (buoy - buoy_prev);
            lfc_p = Some(pressure_hpa[i - 1] + frac * (pressure_hpa[i] - pressure_hpa[i - 1]));
            break;
        }
        if buoy > 0.0 && pressure_hpa[i] <= p_lcl && pressure_hpa[i - 1] > p_lcl {
            lfc_p = Some(pressure_hpa[i]);
            break;
        }
    }

    let mut found_positive = false;
    let mut el_p = None;
    for i in 1..pressure_hpa.len() {
        if pressure_hpa[i] > p_lcl {
            continue;
        }
        let buoy_prev = parcel_tv[i - 1]
            - metrust::calc::thermo::virtual_temp(
                temperature_c[i - 1],
                pressure_hpa[i - 1],
                dewpoint_c[i - 1],
            );
        let buoy = parcel_tv[i]
            - metrust::calc::thermo::virtual_temp(temperature_c[i], pressure_hpa[i], dewpoint_c[i]);
        if buoy > 0.0 {
            found_positive = true;
        }
        if found_positive && buoy_prev > 0.0 && buoy <= 0.0 {
            let frac = (0.0 - buoy_prev) / (buoy - buoy_prev);
            el_p = Some(pressure_hpa[i - 1] + frac * (pressure_hpa[i] - pressure_hpa[i - 1]));
        }
    }

    (lfc_p, el_p)
}

fn package_style_reference_cape_cin_lfc_el(
    height_m: &[f64],
    pressure_pa: &[f64],
    temperature_k: &[f64],
    qv_kgkg: &[f64],
    options: &ParcelOptions,
) -> Result<CapeCinLfcEl, EcapeError> {
    let (pressure_hpa, temperature_c, dewpoint_c) =
        profile_to_met_inputs(pressure_pa, temperature_k, qv_kgkg);

    if matches!(options.cape_type, CapeType::MixedLayer) {
        return Ok(package_style_mixed_layer_cape_cin_lfc_el(
            height_m,
            &pressure_hpa,
            &temperature_c,
            &dewpoint_c,
            options,
        ));
    }

    let (origin_index, p_start_hpa, t_start_c, td_start_c) =
        reference_parcel_start(height_m, pressure_pa, temperature_k, qv_kgkg, options);
    let (cape_pressure_hpa, cape_temperature_c, cape_dewpoint_c) = match options.cape_type {
        CapeType::SurfaceBased => (
            pressure_hpa.clone(),
            temperature_c.clone(),
            dewpoint_c.clone(),
        ),
        CapeType::MixedLayer => {
            let depth_hpa = options.mixed_layer_depth_pa.unwrap_or(10000.0) / 100.0;
            let top_pressure_hpa = pressure_hpa[0] - depth_hpa;
            let mut pressure_prof = vec![p_start_hpa];
            let mut temperature_prof = vec![t_start_c];
            let mut dewpoint_prof = vec![td_start_c];
            for i in 0..pressure_hpa.len() {
                if pressure_hpa[i] < top_pressure_hpa {
                    pressure_prof.push(pressure_hpa[i]);
                    temperature_prof.push(temperature_c[i]);
                    dewpoint_prof.push(dewpoint_c[i]);
                }
            }
            (pressure_prof, temperature_prof, dewpoint_prof)
        }
        CapeType::MostUnstable => (
            pressure_hpa[origin_index..].to_vec(),
            temperature_c[origin_index..].to_vec(),
            dewpoint_c[origin_index..].to_vec(),
        ),
        CapeType::UserDefined => return Err(EcapeError::OriginNotFound),
    };
    let (cape_pressure_hpa, cape_temperature_c, cape_dewpoint_c, cape_parcel_temperature_c) =
        parcel_profile_with_lcl_native(&cape_pressure_hpa, &cape_temperature_c, &cape_dewpoint_c);
    let (cape_jkg, cin_jkg) = cape_cin_profile_native(
        &cape_pressure_hpa,
        &cape_temperature_c,
        &cape_dewpoint_c,
        &cape_parcel_temperature_c,
        "bottom",
        "top",
    );

    let (lfc_p, el_p) = match options.cape_type {
        CapeType::SurfaceBased => {
            let (lfc_p, _) = lfc_native_profile(
                &cape_pressure_hpa,
                &cape_temperature_c,
                &cape_dewpoint_c,
                &cape_parcel_temperature_c,
                cape_dewpoint_c[0],
                "top",
            );
            let (el_p, _) = el_native_profile(
                &cape_pressure_hpa,
                &cape_temperature_c,
                &cape_dewpoint_c,
                &cape_parcel_temperature_c,
                "top",
            );
            (lfc_p, el_p)
        }
        CapeType::MixedLayer | CapeType::MostUnstable => {
            let full_parcel_temperature_c =
                wx_math::thermo::parcel_profile(&pressure_hpa, t_start_c, td_start_c);
            let (lfc_p, _) = lfc_native_profile(
                &pressure_hpa,
                &temperature_c,
                &dewpoint_c,
                &full_parcel_temperature_c,
                td_start_c,
                "top",
            );
            let (el_p, _) = el_native_profile(
                &pressure_hpa,
                &temperature_c,
                &dewpoint_c,
                &full_parcel_temperature_c,
                "top",
            );
            (lfc_p, el_p)
        }
        CapeType::UserDefined => return Err(EcapeError::OriginNotFound),
    };

    Ok(CapeCinLfcEl {
        cape_jkg,
        cin_jkg,
        lfc_m: snap_height_to_last_level_below_pressure(&pressure_hpa, height_m, lfc_p),
        el_m: snap_height_to_last_level_below_pressure(&pressure_hpa, height_m, el_p),
        origin_index,
        pressure_pa: Vec::new(),
        height_m: Vec::new(),
        parcel_temperature_k: Vec::new(),
        buoyancy_ms2: Vec::new(),
    })
}

fn unsaturated_adiabatic_lapse_rate(
    temperature_parcel: f64,
    qv_parcel: f64,
    temperature_env: f64,
    qv_env: f64,
    entrainment_rate: f64,
) -> f64 {
    let temperature_entrainment = -entrainment_rate * (temperature_parcel - temperature_env);
    let density_temperature_parcel = density_temperature(temperature_parcel, qv_parcel, qv_parcel);
    let density_temperature_env = density_temperature(temperature_env, qv_env, qv_env);
    let buoyancy =
        G * (density_temperature_parcel - density_temperature_env) / density_temperature_env;
    let c_pmv = (1.0 - qv_parcel) * CPD + qv_parcel * CPV;
    (-G / CPD) * ((1.0 + buoyancy / G) / (c_pmv / CPD)) + temperature_entrainment
}

fn saturated_adiabatic_lapse_rate(
    temperature_parcel: f64,
    qt_parcel: f64,
    pressure_parcel: f64,
    temperature_env: f64,
    qv_env: f64,
    entrainment_rate: f64,
    prate: f64,
    qt_entrainment: Option<f64>,
) -> f64 {
    let omega = omega(temperature_parcel, 273.15, 253.15);
    let d_omega = omega_deriv(temperature_parcel, 273.15, 253.15);
    let q_vsl = (1.0 - qt_parcel) * r_sat(temperature_parcel, pressure_parcel, 0);
    let q_vsi = (1.0 - qt_parcel) * r_sat(temperature_parcel, pressure_parcel, 2);
    let qv_parcel = (1.0 - omega) * q_vsl + omega * q_vsi;
    let temperature_entrainment = -entrainment_rate * (temperature_parcel - temperature_env);
    let qv_entrainment = -entrainment_rate * (qv_parcel - qv_env);
    let qt_entrainment = qt_entrainment
        .unwrap_or(-entrainment_rate * (qt_parcel - qv_env) - prate * (qt_parcel - qv_parcel));
    let q_condensate = qt_parcel - qv_parcel;
    let ql_parcel = q_condensate * (1.0 - omega);
    let qi_parcel = q_condensate * omega;
    let c_pm = (1.0 - qt_parcel) * CPD + qv_parcel * CPV + ql_parcel * CPL + qi_parcel * CPI;
    let density_temperature_parcel = density_temperature(temperature_parcel, qv_parcel, qt_parcel);
    let density_temperature_env = density_temperature(temperature_env, qv_env, qv_env);
    let buoyancy =
        G * (density_temperature_parcel - density_temperature_env) / density_temperature_env;
    let l_v = LV_TRIP + (temperature_parcel - T_TRIP) * (CPV - CPL);
    let l_i = LI_TRIP + (temperature_parcel - T_TRIP) * (CPL - CPI);
    let l_s = l_v + omega * l_i;
    let q_vsl_cap = q_vsl / (PHI - PHI * qt_parcel + qv_parcel);
    let q_vsi_cap = q_vsi / (PHI - PHI * qt_parcel + qv_parcel);
    let q_m = (1.0 - omega) * q_vsl / (1.0 - q_vsl_cap) + omega * q_vsi / (1.0 - q_vsi_cap);
    let l_m = (1.0 - omega) * l_v * q_vsl / (1.0 - q_vsl_cap)
        + omega * (l_v + l_i) * q_vsi / (1.0 - q_vsi_cap);
    let r_m0 = (1.0 - qv_env) * RD + qv_env * RV;
    let term_1 = buoyancy;
    let term_2 = G;
    let term_3 = ((l_s * q_m) / (r_m0 * temperature_env)) * G;
    let term_4 = (c_pm - l_i * (qt_parcel - qv_parcel) * d_omega) * temperature_entrainment;
    let term_5 = l_s * (qv_entrainment + qv_parcel / (1.0 - qt_parcel) * qt_entrainment);
    let term_6 = c_pm;
    let term_7 = (l_i * (qt_parcel - qv_parcel) - l_s * (q_vsi - q_vsl)) * d_omega;
    let term_8 = (l_s * l_m) / (RV * temperature_parcel * temperature_parcel);
    -(term_1 + term_2 + term_3 - term_4 - term_5) / (term_6 - term_7 + term_8)
}

fn pressure_at_height(ref_pressure: f64, height_above_ref_pressure: f64, temperature: f64) -> f64 {
    let scale_height = (MOLAR_GAS_CONSTANT * temperature) / (AVG_MOLAR_MASS * G);
    ref_pressure * (-height_above_ref_pressure / scale_height).exp()
}

fn moist_static_energy(z_m: f64, temp_k: f64, qv_kgkg: f64) -> f64 {
    CPD * temp_k + G * z_m + LV_TRIP * qv_kgkg
}

fn layer_mean(values: &[f64], heights: &[f64], bottom: f64, top: f64) -> f64 {
    let mut accum = 0.0;
    let mut weight = 0.0;
    for i in 1..heights.len() {
        let z0 = heights[i - 1].max(bottom);
        let z1 = heights[i].min(top);
        if z1 <= z0 {
            continue;
        }
        let v0 = linear_interp(heights[i - 1], heights[i], values[i - 1], values[i], z0);
        let v1 = linear_interp(heights[i - 1], heights[i], values[i - 1], values[i], z1);
        let dz = z1 - z0;
        accum += 0.5 * (v0 + v1) * dz;
        weight += dz;
    }
    if weight == 0.0 {
        values[0]
    } else {
        accum / weight
    }
}

fn wind_components_from_direction_speed_scalar(direction_deg: f64, speed: f64) -> (f64, f64) {
    let rad = direction_deg.to_radians();
    (-speed * rad.sin(), -speed * rad.cos())
}

pub fn wind_components_from_direction_speed(direction_deg: f64, speed: f64) -> (f64, f64) {
    wind_components_from_direction_speed_scalar(direction_deg, speed)
}

fn resolve_parcel_origin(
    heights: &[f64],
    pressures: &[f64],
    temperatures: &[f64],
    qv: &[f64],
    options: &ParcelOptions,
) -> Result<ParcelOriginState, EcapeError> {
    if let Some(origin_p) = options.origin_pressure_pa {
        let idx = find_bracketing_index_desc(pressures, origin_p);
        return Ok(ParcelOriginState {
            index: idx,
            theta_override_k: None,
            qv_override_kgkg: None,
            height_override_m: None,
        });
    }
    if let Some(origin_z) = options.origin_height_m {
        let idx = find_bracketing_index_asc(heights, origin_z);
        return Ok(ParcelOriginState {
            index: idx,
            theta_override_k: None,
            qv_override_kgkg: None,
            height_override_m: None,
        });
    }
    match options.cape_type {
        CapeType::SurfaceBased => Ok(ParcelOriginState {
            index: 0,
            theta_override_k: None,
            qv_override_kgkg: None,
            height_override_m: None,
        }),
        CapeType::MixedLayer => {
            let (_, p_start_hpa, t_start_c, td_start_c) =
                reference_parcel_start(heights, pressures, temperatures, qv, options);
            Ok(ParcelOriginState {
                index: 0,
                theta_override_k: Some(potential_temperature(
                    t_start_c + KELVIN_OFFSET,
                    p_start_hpa * 100.0,
                )),
                qv_override_kgkg: Some(specific_humidity_from_dewpoint(
                    p_start_hpa * 100.0,
                    td_start_c + KELVIN_OFFSET,
                )),
                height_override_m: Some(heights[0]),
            })
        }
        CapeType::MostUnstable => {
            let (best_idx, p_start_hpa, t_start_c, td_start_c) =
                reference_parcel_start(heights, pressures, temperatures, qv, options);
            Ok(ParcelOriginState {
                index: best_idx,
                theta_override_k: None,
                qv_override_kgkg: None,
                height_override_m: if (pressures[best_idx] - p_start_hpa * 100.0).abs() < 1.0
                    && (temperatures[best_idx] - (t_start_c + KELVIN_OFFSET)).abs() < 1e-6
                    && (dewpoint_from_specific_humidity(pressures[best_idx], qv[best_idx])
                        - (td_start_c + KELVIN_OFFSET))
                        .abs()
                        < 1e-6
                {
                    None
                } else {
                    Some(heights[best_idx])
                },
            })
        }
        CapeType::UserDefined => Err(EcapeError::OriginNotFound),
    }
}

fn lcl_temperature(temp_k: f64, dewpoint_k: f64) -> f64 {
    1.0 / (1.0 / (dewpoint_k - 56.0) + (temp_k / dewpoint_k).ln() / 800.0) + 56.0
}

fn lcl_pressure(temp_k: f64, dewpoint_k: f64, pressure_pa: f64) -> f64 {
    let tl = lcl_temperature(temp_k, dewpoint_k);
    pressure_pa * (tl / temp_k).powf(1.0 / KAPPA)
}

fn lifting_condensation_level(temp_k: f64, dewpoint_k: f64, pressure_pa: f64) -> (f64, f64) {
    let plcl = lcl_pressure(temp_k, dewpoint_k, pressure_pa);
    let zlcl =
        (RD * 0.5 * (temp_k + lcl_temperature(temp_k, dewpoint_k)) / G) * (pressure_pa / plcl).ln();
    (plcl, zlcl.max(0.0))
}

fn bunkers_storm_motion(
    pressures: &[f64],
    heights: &[f64],
    u: &[f64],
    v: &[f64],
) -> ((f64, f64), (f64, f64), (f64, f64)) {
    let z0 = heights[0];
    let height_agl: Vec<f64> = heights.iter().map(|z| z - z0).collect();
    let pressure_hpa: Vec<f64> = pressures.iter().map(|p| p / 100.0).collect();
    metrust::calc::bunkers_storm_motion(&pressure_hpa, u, v, &height_agl)
}

fn resolve_storm_motion(
    pressures: &[f64],
    heights: &[f64],
    u: &[f64],
    v: &[f64],
    options: &ParcelOptions,
) -> (f64, f64) {
    if let (Some(u_sm), Some(v_sm)) = (options.storm_motion_u_ms, options.storm_motion_v_ms) {
        return (u_sm, v_sm);
    }
    let z0 = heights[0];
    let heights_agl: Vec<f64> = heights.iter().map(|z| z - z0).collect();
    let (rm, lm, mean) = bunkers_storm_motion(pressures, &heights_agl, u, v);
    match options.storm_motion_type {
        StormMotionType::RightMoving => rm,
        StormMotionType::LeftMoving => lm,
        StormMotionType::MeanWind => mean,
        StormMotionType::UserDefined => options
            .storm_motion_u_ms
            .zip(options.storm_motion_v_ms)
            .unwrap_or(rm),
    }
}

fn calc_sr_wind(
    heights: &[f64],
    u: &[f64],
    v: &[f64],
    storm_u: f64,
    storm_v: f64,
    bottom: f64,
    top: f64,
) -> f64 {
    let z0 = heights[0];
    let mut values = Vec::new();
    for i in 0..heights.len() {
        let agl = heights[i] - z0;
        if agl >= bottom && agl <= top {
            values.push(((u[i] - storm_u).powi(2) + (v[i] - storm_v).powi(2)).sqrt());
        }
    }
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

fn parcel_profile_from(
    heights: &[f64],
    pressures: &[f64],
    temperatures: &[f64],
    qv_env: &[f64],
    origin_idx: usize,
    entrainment_rate: f64,
    pseudoadiabatic: bool,
    origin_theta_override: Option<f64>,
    origin_qv_override: Option<f64>,
    origin_height_override: Option<f64>,
) -> ParcelProfile {
    let origin_z = origin_height_override.unwrap_or(heights[origin_idx]);
    let mut parcel_pressure = interp_pressure_to_height(heights, pressures, origin_z);
    let mut parcel_height = origin_z;
    let mut parcel_temperature = origin_theta_override
        .map(|theta| temperature_from_potential_temperature(theta, parcel_pressure))
        .unwrap_or(temperatures[origin_idx]);
    let origin_qv = origin_qv_override.unwrap_or(qv_env[origin_idx]);
    let mut parcel_qv = origin_qv;
    let mut parcel_qt = parcel_qv;
    let prate = if pseudoadiabatic {
        1.0 / DEFAULT_STEP_M
    } else {
        0.0
    };
    let mut dqt_dz = 0.0;

    let mut out_p = vec![parcel_pressure];
    let mut out_z = vec![parcel_height];
    let mut out_t = vec![parcel_temperature];
    let mut out_qv = vec![parcel_qv];
    let mut out_qt = vec![parcel_qt];

    while parcel_pressure >= pressures[pressures.len() - 1] {
        let env_temperature = interp_profile_at_height(heights, temperatures, parcel_height);
        let parcel_saturation_qv =
            (1.0 - parcel_qt) * r_sat(parcel_temperature, parcel_pressure, 1);
        if parcel_saturation_qv > parcel_qv {
            parcel_pressure = pressure_at_height(parcel_pressure, DEFAULT_STEP_M, env_temperature);
            parcel_height += DEFAULT_STEP_M;
            let env_temperature = interp_profile_at_height(heights, temperatures, parcel_height);
            let env_qv = interp_profile_at_height(heights, qv_env, parcel_height);
            let d_t_dz = unsaturated_adiabatic_lapse_rate(
                parcel_temperature,
                parcel_qv,
                env_temperature,
                env_qv,
                entrainment_rate,
            );
            let dqv_dz = -entrainment_rate * (parcel_qv - env_qv);
            parcel_temperature += d_t_dz * DEFAULT_STEP_M;
            parcel_qv += dqv_dz * DEFAULT_STEP_M;
            parcel_qt = parcel_qv;
        } else {
            parcel_pressure = pressure_at_height(parcel_pressure, DEFAULT_STEP_M, env_temperature);
            parcel_height += DEFAULT_STEP_M;
            let env_temperature = interp_profile_at_height(heights, temperatures, parcel_height);
            let env_qv = interp_profile_at_height(heights, qv_env, parcel_height);
            let d_t_dz = if pseudoadiabatic {
                saturated_adiabatic_lapse_rate(
                    parcel_temperature,
                    parcel_qt,
                    parcel_pressure,
                    env_temperature,
                    env_qv,
                    entrainment_rate,
                    prate,
                    Some(dqt_dz),
                )
            } else {
                saturated_adiabatic_lapse_rate(
                    parcel_temperature,
                    parcel_qt,
                    parcel_pressure,
                    env_temperature,
                    env_qv,
                    entrainment_rate,
                    prate,
                    None,
                )
            };
            let new_parcel_qv = (1.0 - parcel_qt) * r_sat(parcel_temperature, parcel_pressure, 1);
            if pseudoadiabatic {
                dqt_dz = (new_parcel_qv - parcel_qv) / DEFAULT_STEP_M;
            } else {
                dqt_dz = -entrainment_rate * (parcel_qt - env_qv) - prate * (parcel_qt - parcel_qv);
            }
            parcel_temperature += d_t_dz * DEFAULT_STEP_M;
            parcel_qv = new_parcel_qv;
            if pseudoadiabatic {
                parcel_qt = parcel_qv;
            } else {
                dqt_dz = -entrainment_rate * (parcel_qt - env_qv) - prate * (parcel_qt - parcel_qv);
                parcel_qt += dqt_dz * DEFAULT_STEP_M;
            }
            if parcel_qt < parcel_qv {
                parcel_qv = parcel_qt;
            }
        }

        out_p.push(parcel_pressure);
        out_z.push(parcel_height);
        out_t.push(parcel_temperature);
        out_qv.push(parcel_qv);
        out_qt.push(parcel_qt);

        if out_p.len() > 20000 {
            break;
        }
    }

    let buoyancy: Vec<f64> = out_z
        .iter()
        .enumerate()
        .map(|(i, z)| {
            let env_t = interp_profile_at_height(heights, temperatures, *z);
            let env_q = interp_profile_at_height(heights, qv_env, *z);
            let parcel_t_rho = density_temperature(out_t[i], out_qv[i], out_qt[i]);
            let env_t_rho = density_temperature(env_t, env_q, env_q);
            G * (parcel_t_rho - env_t_rho) / env_t_rho
        })
        .collect();
    ParcelProfile {
        pressure_pa: out_p,
        height_m: out_z,
        temperature_k: out_t,
        qv_kgkg: out_qv,
        qt_kgkg: out_qt,
        buoyancy_ms2: buoyancy,
    }
}

fn continuous_cape_cin_lfc_el(
    height_m: &[f64],
    pressure_pa: &[f64],
    temperature_k: &[f64],
    qv_kgkg: &[f64],
    options: &ParcelOptions,
) -> Result<CapeCinLfcEl, EcapeError> {
    let zero_wind = vec![0.0; height_m.len()];
    validate_profile(
        height_m,
        pressure_pa,
        temperature_k,
        qv_kgkg,
        &zero_wind,
        &zero_wind,
    )?;
    let origin = resolve_parcel_origin(height_m, pressure_pa, temperature_k, qv_kgkg, options)?;
    let origin_idx = origin.index;

    let profile = parcel_profile_from(
        height_m,
        pressure_pa,
        temperature_k,
        qv_kgkg,
        origin_idx,
        0.0,
        true,
        origin.theta_override_k,
        origin.qv_override_kgkg,
        origin.height_override_m,
    );

    let env_mse: Vec<f64> = height_m
        .iter()
        .zip(temperature_k.iter())
        .zip(qv_kgkg.iter())
        .map(|((z, t), q)| moist_static_energy(*z, *t, *q))
        .collect();
    let height_min_mse_idx = env_mse
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);
    let height_min_mse = height_m[height_min_mse_idx];
    let mut cape = 0.0;
    let mut cin = 0.0;
    let mut lfc = None;
    let mut el = None;
    for i in (1..profile.height_m.len()).rev() {
        let z0 = profile.height_m[i];
        let dz = profile.height_m[i] - profile.height_m[i - 1];
        let env_t = interp_profile_at_height(height_m, temperature_k, z0);
        let env_q = interp_profile_at_height(height_m, qv_kgkg, z0);
        let env_t_rho = density_temperature(env_t, env_q, env_q);
        let parcel_t_rho = density_temperature(
            profile.temperature_k[i],
            profile.qv_kgkg[i],
            profile.qt_kgkg[i],
        );
        let buoyancy = G * (parcel_t_rho - env_t_rho) / env_t_rho;
        if buoyancy > 0.0 && el.is_none() {
            el = Some(z0);
        }
        if buoyancy > 0.0 && lfc.is_none() {
            cape += buoyancy * dz;
        }
        if z0 < height_min_mse && buoyancy < 0.0 {
            cin += buoyancy * dz;
            if lfc.is_none() {
                lfc = Some(z0);
            }
        }
    }
    if lfc.is_none() {
        lfc = Some(height_m[0]);
    }

    Ok(CapeCinLfcEl {
        cape_jkg: cape,
        cin_jkg: cin,
        lfc_m: lfc,
        el_m: el,
        origin_index: origin_idx,
        pressure_pa: profile.pressure_pa,
        height_m: profile.height_m,
        parcel_temperature_k: profile.temperature_k,
        buoyancy_ms2: profile.buoyancy_ms2,
    })
}

pub fn custom_cape_cin_lfc_el(
    height_m: &[f64],
    pressure_pa: &[f64],
    temperature_k: &[f64],
    qv_kgkg: &[f64],
    options: &ParcelOptions,
) -> Result<CapeCinLfcEl, EcapeError> {
    if options.origin_pressure_pa.is_none()
        && options.origin_height_m.is_none()
        && !matches!(options.cape_type, CapeType::UserDefined)
    {
        return package_style_reference_cape_cin_lfc_el(
            height_m,
            pressure_pa,
            temperature_k,
            qv_kgkg,
            options,
        );
    }

    continuous_cape_cin_lfc_el(height_m, pressure_pa, temperature_k, qv_kgkg, options)
}

pub fn summarize_parcel_profile(
    parcel_height_m: &[f64],
    parcel_temperature_k: &[f64],
    parcel_qv_kgkg: &[f64],
    parcel_qt_kgkg: &[f64],
    env_height_m: &[f64],
    env_temperature_k: &[f64],
    env_qv_kgkg: &[f64],
) -> CapeCinLfcEl {
    let env_mse: Vec<f64> = env_height_m
        .iter()
        .zip(env_temperature_k.iter())
        .zip(env_qv_kgkg.iter())
        .map(|((z, t), q)| moist_static_energy(*z, *t, *q))
        .collect();
    let height_min_mse_idx = env_mse
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);
    let height_min_mse = env_height_m[height_min_mse_idx];

    let mut cape = 0.0;
    let mut cin = 0.0;
    let mut lfc = None;
    let mut el = None;
    let mut buoyancy_ms2 = Vec::with_capacity(parcel_height_m.len());

    for i in 0..parcel_height_m.len() {
        let z = parcel_height_m[i];
        let env_t = interp_profile_at_height(env_height_m, env_temperature_k, z);
        let env_q = interp_profile_at_height(env_height_m, env_qv_kgkg, z);
        let env_t_rho = density_temperature(env_t, env_q, env_q);
        let parcel_t_rho = density_temperature(
            parcel_temperature_k[i],
            parcel_qv_kgkg[i],
            parcel_qt_kgkg[i],
        );
        buoyancy_ms2.push(G * (parcel_t_rho - env_t_rho) / env_t_rho);
    }

    for i in (1..parcel_height_m.len()).rev() {
        let z0 = parcel_height_m[i];
        let dz = parcel_height_m[i] - parcel_height_m[i - 1];
        let buoyancy = buoyancy_ms2[i];
        if buoyancy > 0.0 && el.is_none() {
            el = Some(z0);
        }
        if buoyancy > 0.0 && lfc.is_none() {
            cape += buoyancy * dz;
        }
        if z0 < height_min_mse && buoyancy < 0.0 {
            cin += buoyancy * dz;
            if lfc.is_none() {
                lfc = Some(z0);
            }
        }
    }

    if lfc.is_none() {
        lfc = Some(env_height_m[0]);
    }

    CapeCinLfcEl {
        cape_jkg: cape,
        cin_jkg: cin,
        lfc_m: lfc,
        el_m: el,
        origin_index: 0,
        pressure_pa: Vec::new(),
        height_m: parcel_height_m.to_vec(),
        parcel_temperature_k: parcel_temperature_k.to_vec(),
        buoyancy_ms2,
    }
}

fn calc_psi(el_z: f64) -> f64 {
    let sigma = 1.1;
    let alpha = 0.8;
    let l_mix = 120.0;
    let pr = 1.0 / 3.0;
    let ksq = 0.18;
    (ksq * alpha * alpha * std::f64::consts::PI * std::f64::consts::PI * l_mix)
        / (4.0 * pr * sigma * sigma * el_z.max(1.0))
}

fn compute_ncape_reference(
    height_m: &[f64],
    pressure_pa: &[f64],
    temperature_k: &[f64],
    qv_kgkg: &[f64],
    lfc_m: f64,
    el_m: f64,
) -> f64 {
    if el_m <= lfc_m {
        return 0.0;
    }
    let mse0: Vec<f64> = temperature_k
        .iter()
        .zip(qv_kgkg.iter())
        .zip(height_m.iter())
        .map(|((t, q), z)| moist_static_energy(*z, *t, *q))
        .collect();
    let qsat: Vec<f64> = temperature_k
        .iter()
        .zip(pressure_pa.iter())
        .map(|(t, p)| {
            let rsat = r_sat(*t, *p, 0);
            rsat / (1.0 + rsat)
        })
        .collect();
    let mse0_star: Vec<f64> = temperature_k
        .iter()
        .zip(qsat.iter())
        .zip(height_m.iter())
        .map(|((t, q), z)| moist_static_energy(*z, *t, *q))
        .collect();
    let mut mse0bar = vec![0.0; mse0.len()];
    mse0bar[0] = mse0[0];
    for iz in 1..mse0bar.len() {
        let mut sum = 0.0;
        for j in 0..iz {
            sum += (mse0[j] + mse0[j + 1]) * (height_m[j + 1] - height_m[j]);
        }
        mse0bar[iz] = 0.5 * sum / (height_m[iz] - height_m[0]);
    }
    let int_arg: Vec<f64> = mse0bar
        .iter()
        .zip(mse0_star.iter())
        .zip(temperature_k.iter())
        .map(|((bar, star), t)| -(G / (CPD * *t)) * (bar - star))
        .collect();
    let ind_lfc = height_m
        .iter()
        .enumerate()
        .min_by(|a, b| {
            (a.1 - lfc_m)
                .abs()
                .partial_cmp(&(b.1 - lfc_m).abs())
                .unwrap()
        })
        .map(|(i, _)| i)
        .unwrap_or(0);
    let ind_el = height_m
        .iter()
        .enumerate()
        .min_by(|a, b| (a.1 - el_m).abs().partial_cmp(&(b.1 - el_m).abs()).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(ind_lfc);
    if ind_el <= ind_lfc + 1 {
        return 0.0;
    }
    let mut ncape = 0.0;
    for i in ind_lfc..(ind_el - 1) {
        ncape += (0.5 * int_arg[i] + 0.5 * int_arg[i + 1]) * (height_m[i + 1] - height_m[i]);
    }
    ncape.max(0.0)
}

fn calc_ecape_a(sr_wind: f64, psi: f64, ncape: f64, cape: f64) -> f64 {
    let sr2 = (sr_wind * sr_wind).max(1e-9);
    let denom = 4.0 * psi / sr2;
    let term_a = sr2 / 2.0;
    let term_b = (-1.0 - psi - (2.0 * psi / sr2) * ncape) / denom;
    let term_c = ((1.0 + psi + (2.0 * psi / sr2) * ncape).powi(2)
        + 8.0 * (psi / sr2) * (cape - psi * ncape))
        .sqrt()
        / denom;
    let ecape_a = term_a + term_b + term_c;
    if ecape_a >= 0.0 { ecape_a } else { 0.0 }
}

fn entrainment_rate(cape: f64, ecape: f64, ncape: f64, vsr: f64, storm_column_height: f64) -> f64 {
    let e_a_tilde = ecape / cape.max(1e-9);
    let n_tilde = ncape / cape.max(1e-9);
    let vsr_tilde = vsr / (2.0 * cape.max(1e-9)).sqrt();
    let e_tilde = e_a_tilde - vsr_tilde * vsr_tilde;
    (2.0 * (1.0 - e_tilde) / (e_tilde + n_tilde)) / storm_column_height.max(1e-9)
}

pub fn calc_ecape_ncape_from_reference(
    height_m: &[f64],
    pressure_pa: &[f64],
    temperature_k: &[f64],
    qv_kgkg: &[f64],
    u_wind_ms: &[f64],
    v_wind_ms: &[f64],
    options: &ParcelOptions,
    cape: f64,
    lfc_m: Option<f64>,
    el_m: Option<f64>,
) -> EcapeNcape {
    let (storm_u, storm_v) =
        resolve_storm_motion(pressure_pa, height_m, u_wind_ms, v_wind_ms, options);
    let bottom = options.inflow_layer_bottom_m.unwrap_or(0.0);
    let top = options.inflow_layer_top_m.unwrap_or(1000.0);
    let vsr = calc_sr_wind(
        height_m, u_wind_ms, v_wind_ms, storm_u, storm_v, bottom, top,
    );
    let ncape = match (lfc_m, el_m) {
        (Some(lfc), Some(el)) if el > lfc => {
            compute_ncape_reference(height_m, pressure_pa, temperature_k, qv_kgkg, lfc, el)
        }
        _ => 0.0,
    };
    let psi = el_m.map(calc_psi).unwrap_or(0.0);
    let ecape = if el_m.is_some() && psi > 0.0 && vsr > 0.0 {
        calc_ecape_a(vsr, psi, ncape, cape)
    } else {
        0.0
    };
    EcapeNcape {
        ecape_jkg: ecape,
        ncape_jkg: ncape,
        cape_jkg: cape,
        lfc_m,
        el_m,
        storm_motion_u_ms: storm_u,
        storm_motion_v_ms: storm_v,
        storm_relative_wind_ms: vsr,
        psi,
    }
}

pub fn calc_ecape_ncape(
    height_m: &[f64],
    pressure_pa: &[f64],
    temperature_k: &[f64],
    qv_kgkg: &[f64],
    u_wind_ms: &[f64],
    v_wind_ms: &[f64],
    options: &ParcelOptions,
) -> Result<EcapeNcape, EcapeError> {
    validate_profile(
        height_m,
        pressure_pa,
        temperature_k,
        qv_kgkg,
        u_wind_ms,
        v_wind_ms,
    )?;
    let cape_info = custom_cape_cin_lfc_el(height_m, pressure_pa, temperature_k, qv_kgkg, options)?;
    Ok(calc_ecape_ncape_from_reference(
        height_m,
        pressure_pa,
        temperature_k,
        qv_kgkg,
        u_wind_ms,
        v_wind_ms,
        options,
        cape_info.cape_jkg,
        cape_info.lfc_m,
        cape_info.el_m,
    ))
}

pub fn calc_ecape_parcel(
    height_m: &[f64],
    pressure_pa: &[f64],
    temperature_k: &[f64],
    dewpoint_k: &[f64],
    u_wind_ms: &[f64],
    v_wind_ms: &[f64],
    options: &ParcelOptions,
) -> Result<EcapeParcelResult, EcapeError> {
    let qv: Vec<f64> = pressure_pa
        .iter()
        .zip(dewpoint_k.iter())
        .map(|(p, td)| specific_humidity_from_dewpoint(*p, *td))
        .collect();
    validate_profile(
        height_m,
        pressure_pa,
        temperature_k,
        &qv,
        u_wind_ms,
        v_wind_ms,
    )?;
    let origin = resolve_parcel_origin(height_m, pressure_pa, temperature_k, &qv, options)?;
    let origin_idx = origin.index;
    let origin_z = origin.height_override_m.unwrap_or(height_m[origin_idx]);
    let pseudoadiabatic = options.pseudoadiabatic.unwrap_or(true);
    let base_profile = parcel_profile_from(
        height_m,
        pressure_pa,
        temperature_k,
        &qv,
        origin_idx,
        0.0,
        pseudoadiabatic,
        origin.theta_override_k,
        origin.qv_override_kgkg,
        origin.height_override_m,
    );
    let base = summarize_parcel_profile(
        &base_profile.height_m,
        &base_profile.temperature_k,
        &base_profile.qv_kgkg,
        &base_profile.qt_kgkg,
        height_m,
        temperature_k,
        &qv,
    );
    let ecape_info = calc_ecape_ncape_from_reference(
        height_m,
        pressure_pa,
        temperature_k,
        &qv,
        u_wind_ms,
        v_wind_ms,
        options,
        base.cape_jkg,
        base.lfc_m,
        base.el_m,
    );
    let entrainment = options.entrainment_rate.unwrap_or_else(|| {
        if let (Some(el), vsr) = (ecape_info.el_m, ecape_info.storm_relative_wind_ms) {
            if el > origin_z && base.cape_jkg > 0.0 {
                entrainment_rate(
                    base.cape_jkg,
                    ecape_info.ecape_jkg,
                    ecape_info.ncape_jkg,
                    vsr,
                    el - origin_z,
                )
            } else {
                0.0
            }
        } else {
            0.0
        }
    });

    let parcel = parcel_profile_from(
        height_m,
        pressure_pa,
        temperature_k,
        &qv,
        origin_idx,
        entrainment.max(0.0),
        pseudoadiabatic,
        origin.theta_override_k,
        origin.qv_override_kgkg,
        origin.height_override_m,
    );

    let parcel_summary = summarize_parcel_profile(
        &parcel.height_m,
        &parcel.temperature_k,
        &parcel.qv_kgkg,
        &parcel.qt_kgkg,
        height_m,
        temperature_k,
        &qv,
    );
    let parcel_ecape = calc_ecape_ncape_from_reference(
        height_m,
        pressure_pa,
        temperature_k,
        &qv,
        u_wind_ms,
        v_wind_ms,
        options,
        parcel_summary.cape_jkg,
        parcel_summary.lfc_m,
        parcel_summary.el_m,
    );

    Ok(EcapeParcelResult {
        ecape_jkg: parcel_ecape.ecape_jkg,
        ncape_jkg: parcel_ecape.ncape_jkg,
        cape_jkg: parcel_summary.cape_jkg,
        cin_jkg: parcel_summary.cin_jkg,
        lfc_m: parcel_summary.lfc_m,
        el_m: parcel_summary.el_m,
        storm_motion_u_ms: parcel_ecape.storm_motion_u_ms,
        storm_motion_v_ms: parcel_ecape.storm_motion_v_ms,
        parcel_profile: parcel,
    })
}
