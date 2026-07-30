//! Meteorological calculations -- aggregated from wx-math submodules.
//!
//! Mirrors MetPy's `metpy.calc` namespace:
//! - `thermo`      -- thermodynamic functions (CAPE/CIN, LCL, theta-e, ...)
//! - `wind`        -- wind speed/direction, components, storm motion
//! - `kinematics`  -- derivatives, divergence, vorticity, frontogenesis
//! - `severe`      -- composite parameters (STP, SCP, critical angle)
//! - `atmo`        -- atmospheric profiles, stability indices, comfort
//! - `smooth`      -- grid smoothing and spatial derivatives

pub mod atmo;
pub mod kinematics;
pub mod severe;
pub mod smooth;
pub mod thermo;
pub mod utils;
pub mod wind;

// ── Convenience re-exports from submodules ──────────────────────────
// Pull the most commonly used items to `metrust::calc::*`.

// Thermo essentials
pub use thermo::{
    cape_cin, dewpoint_from_relative_humidity, downdraft_cape, el,
    equivalent_potential_temperature, k_index, lcl, lfc, mixing_ratio, potential_temperature,
    relative_humidity_from_dewpoint, saturation_mixing_ratio, saturation_mixing_ratio_with_phase,
    saturation_vapor_pressure, saturation_vapor_pressure_with_phase, showalter_index,
    specific_humidity_from_mixing_ratio, thickness_hydrostatic_from_relative_humidity,
    total_totals, virtual_temperature, wet_bulb_temperature, Phase,
};

// Wind essentials
pub use wind::{
    bulk_shear, bunkers_storm_motion, corfidi_storm_motion, friction_velocity,
    gradient_richardson_number, mean_wind, storm_relative_helicity, tke, wind_components,
    wind_direction, wind_speed,
};

// Kinematics essentials
pub use kinematics::{
    absolute_vorticity, advection, advection_3d, ageostrophic_wind, divergence, frontogenesis,
    geostrophic_wind, normal_component, potential_vorticity_baroclinic, tangential_component,
    unit_vectors_from_cross_section, vector_derivative, vorticity,
};

// Severe essentials
pub use severe::{
    // Point-based re-exports from wx_math::composite
    boyden_index,
    bulk_richardson_number,
    convective_inhibition_depth,
    critical_angle,
    dendritic_growth_zone,
    fosberg_fire_weather_index,
    freezing_rain_composite,
    // From wx_math::thermo
    galvez_davison_index,
    haines_index,
    hot_dry_windy,
    significant_tornado_parameter,
    supercell_composite_parameter,
    warm_nose_check,
};

// Atmo essentials
pub use atmo::{
    altimeter_to_sea_level_pressure, altimeter_to_station_pressure, apparent_temperature,
    heat_index, height_to_pressure_std, pressure_to_height_std, sigma_to_pressure,
    station_to_altimeter_pressure, windchill,
};

// Smooth essentials
pub use smooth::{
    first_derivative, gradient_x, gradient_y, laplacian, lat_lon_grid_deltas, second_derivative,
    smooth_window,
};

// Utils essentials
pub use utils::{
    angle_to_direction, angle_to_direction_ext, azimuth_range_to_lat_lon, find_bounding_indices,
    find_peaks, nearest_intersection_idx, parse_angle, peak_persistence, resample_nn_1d,
};
