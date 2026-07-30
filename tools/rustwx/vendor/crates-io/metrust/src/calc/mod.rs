//! Meteorological calculations -- aggregated from wx-math submodules.
//!
//! Mirrors MetPy's `metpy.calc` namespace:
//! - `thermo`      -- thermodynamic functions (CAPE/CIN, LCL, theta-e, ...)
//! - `wind`        -- wind speed/direction, components, storm motion
//! - `kinematics`  -- derivatives, divergence, vorticity, frontogenesis
//! - `severe`      -- composite parameters (STP, SCP, critical angle)
//! - `atmo`        -- atmospheric profiles, stability indices, comfort
//! - `smooth`      -- grid smoothing and spatial derivatives

pub mod thermo;
pub mod wind;
pub mod kinematics;
pub mod severe;
pub mod atmo;
pub mod smooth;
pub mod utils;

// ── Convenience re-exports from submodules ──────────────────────────
// Pull the most commonly used items to `metrust::calc::*`.

// Thermo essentials
pub use thermo::{
    potential_temperature, equivalent_potential_temperature,
    saturation_vapor_pressure, saturation_vapor_pressure_with_phase,
    saturation_mixing_ratio, saturation_mixing_ratio_with_phase,
    Phase,
    wet_bulb_temperature, lfc, el, lcl,
    dewpoint_from_relative_humidity, relative_humidity_from_dewpoint,
    virtual_temperature, cape_cin, mixing_ratio,
    showalter_index, k_index, total_totals,
    downdraft_cape,
    specific_humidity_from_mixing_ratio,
    thickness_hydrostatic_from_relative_humidity,
};

// Wind essentials
pub use wind::{
    wind_speed, wind_direction, wind_components,
    bulk_shear, mean_wind, storm_relative_helicity,
    bunkers_storm_motion, corfidi_storm_motion,
    friction_velocity, tke, gradient_richardson_number,
};

// Kinematics essentials
pub use kinematics::{
    divergence, vorticity, absolute_vorticity,
    advection, advection_3d, frontogenesis,
    geostrophic_wind, ageostrophic_wind,
    potential_vorticity_baroclinic,
    normal_component, tangential_component,
    unit_vectors_from_cross_section,
    vector_derivative,
};

// Severe essentials
pub use severe::{
    significant_tornado_parameter,
    supercell_composite_parameter,
    critical_angle,
    // Point-based re-exports from wx_math::composite
    boyden_index, bulk_richardson_number, convective_inhibition_depth,
    dendritic_growth_zone, fosberg_fire_weather_index, freezing_rain_composite,
    haines_index, hot_dry_windy, warm_nose_check,
    // From wx_math::thermo
    galvez_davison_index,
};

// Atmo essentials
pub use atmo::{
    pressure_to_height_std, height_to_pressure_std,
    altimeter_to_station_pressure, station_to_altimeter_pressure,
    altimeter_to_sea_level_pressure,
    sigma_to_pressure, heat_index, windchill, apparent_temperature,
};

// Smooth essentials
pub use smooth::{
    gradient_x, gradient_y, laplacian,
    first_derivative, second_derivative,
    lat_lon_grid_deltas,
    smooth_window,
};

// Utils essentials
pub use utils::{
    angle_to_direction, angle_to_direction_ext, parse_angle,
    find_bounding_indices, nearest_intersection_idx,
    resample_nn_1d,
    find_peaks, peak_persistence,
    azimuth_range_to_lat_lon,
};
