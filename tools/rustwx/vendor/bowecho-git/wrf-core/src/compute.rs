use crate::error::{WrfError, WrfResult};
use crate::file::WrfFile;
use crate::units::{convert_array, parse_units};
use crate::variables::{get_var_def, VarDim};

/// Custom storm motion for SRH-family diagnostics.
#[derive(Debug, Clone, PartialEq)]
pub enum StormMotion {
    /// One `(u, v)` motion applied to every grid cell.
    Uniform { u: f64, v: f64 },
    /// Per-grid-cell storm motion fields, flattened `[ny, nx]`.
    Grid { u: Vec<f64>, v: Vec<f64> },
}

impl StormMotion {
    /// Return the storm motion at flattened grid index `ij`.
    pub fn at(&self, ij: usize) -> (f64, f64) {
        match self {
            Self::Uniform { u, v } => (*u, *v),
            Self::Grid { u, v } => (u[ij], v[ij]),
        }
    }
}

/// Algorithm used to derive default Bunkers storm motion when no explicit storm vector is given.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StormMotionMethod {
    /// Use pressure-weighted layer means when pressure data are available.
    PressureWeighted,
    /// Use non-pressure-weighted Bunkers layer means.
    NonPressureWeighted,
}

/// Options controlling variable computation.
#[derive(Debug, Clone, Default)]
pub struct ComputeOpts {
    /// Requested output units (overrides default).
    pub units: Option<String>,
    /// Parcel type for CAPE: "sb", "ml", "mu".
    pub parcel_type: Option<String>,
    /// Custom storm motion in m/s for SRH-family diagnostics.
    pub storm_motion: Option<StormMotion>,
    /// Default Bunkers storm-motion algorithm when `storm_motion` is not supplied.
    pub storm_motion_method: Option<StormMotionMethod>,
    /// Storm-motion family for ECAPE calculations: "right_moving", "left_moving", or "mean_wind".
    pub storm_motion_type: Option<String>,
    /// Integration top (meters AGL) for CAPE, lapse rates, updraft helicity, etc.
    pub top_m: Option<f64>,
    /// Bottom of layer (meters AGL) for shear, mean wind, lapse rates, UH, etc.
    pub bottom_m: Option<f64>,
    /// Depth for SRH (meters AGL). Default varies by variable.
    pub depth_m: Option<f64>,
    /// Custom parcel starting pressure (hPa).
    pub parcel_pressure: Option<f64>,
    /// Custom parcel starting temperature (deg C).
    pub parcel_temperature: Option<f64>,
    /// Custom parcel starting dewpoint (deg C).
    pub parcel_dewpoint: Option<f64>,
    /// Bottom of layer in pressure (hPa) -- alternative to bottom_m for lapse rates.
    pub bottom_p: Option<f64>,
    /// Top of layer in pressure (hPa) -- alternative to top_m for lapse rates.
    pub top_p: Option<f64>,
    /// Layer type: "fixed" (default) or "effective" (for STP, SRH).
    pub layer_type: Option<String>,
    /// Use virtual temperature instead of absolute temperature (for lapse rates).
    pub use_virtual: Option<bool>,
    /// Interpolate 2m fields over water bodies smaller than this area (km2).
    /// 0 or None = disabled. Typical value: 1000.
    /// Removes lake artifacts in T2/Q2 that corrupt CAPE, STP, etc.
    pub lake_interp: Option<f64>,
    /// Explicit entrainment rate for ECAPE parcel calculations (1/m). If omitted, ecape-rs derives it.
    pub entrainment_rate: Option<f64>,
    /// Use pseudoadiabatic ascent for ECAPE parcel calculations. Default follows ecape-rs.
    pub pseudoadiabatic: Option<bool>,
    /// If true, ECAPE grid calculations error with failure counts instead of silently zero-filling failed columns.
    pub ecape_strict: Option<bool>,
    /// Use variable intercept parameters (Thompson microphysics) for DBZ.
    /// Default (None/false) = constant intercepts matching wrf-python defaults.
    pub use_varint: Option<bool>,
    /// Use liquid-skin bright-band correction for DBZ.
    /// Default (None/false) = no correction, matching wrf-python defaults.
    pub use_liqskin: Option<bool>,
}

#[cfg(test)]
mod tests {
    use super::StormMotion;

    #[test]
    fn grid_storm_motion_returns_column_components() {
        let motion = StormMotion::Grid {
            u: vec![10.0, 11.0, 12.0, 13.0],
            v: vec![20.0, 21.0, 22.0, 23.0],
        };

        assert_eq!(motion.at(0), (10.0, 20.0));
        assert_eq!(motion.at(3), (13.0, 23.0));
    }

    #[test]
    fn uniform_storm_motion_is_reused_for_every_column() {
        let motion = StormMotion::Uniform { u: 12.0, v: 8.0 };

        assert_eq!(motion.at(0), (12.0, 8.0));
        assert_eq!(motion.at(99), (12.0, 8.0));
    }
}

/// The result of computing a WRF variable.
#[derive(Debug, Clone)]
pub struct VarOutput {
    /// Flat data array.
    pub data: Vec<f64>,
    /// Shape: `[ny, nx]` for 2-D, `[nz, ny, nx]` for 3-D.
    pub shape: Vec<usize>,
    /// Unit string of the returned data.
    pub units: String,
    /// Human-readable description.
    pub description: String,
}

/// Top-level variable retrieval: look up name, compute, apply unit conversion.
pub fn getvar(
    file: &WrfFile,
    name: &str,
    timeidx: Option<usize>,
    opts: &ComputeOpts,
) -> WrfResult<VarOutput> {
    let t = timeidx.unwrap_or(0);
    if t >= file.nt {
        return Err(WrfError::InvalidParam(format!(
            "timeidx {t} out of range (file has {} times)",
            file.nt
        )));
    }

    file.prepare_cache_for_time(t);

    // Look up in computed variable registry first
    let vardef = match get_var_def(name) {
        Some(v) => v,
        None => {
            // Fallback: try reading as a raw WRF variable directly from the file.
            // This handles RAINNC, RAINC, PBLH, HFX, LU_INDEX, SWDOWN, TSK, SST, etc.
            return getvar_raw(file, name, t, opts);
        }
    };

    let mut data = (vardef.compute)(file, t, opts)?;

    let shape = match vardef.dim {
        VarDim::TwoD => vec![file.ny, file.nx],
        VarDim::ThreeD => vec![file.nz, file.ny, file.nx],
    };

    // Validate output size
    let expected = shape.iter().product::<usize>();
    // Some variables return multi-field (e.g., cape2d returns 4*nxy, uvmet returns 2*nxyz).
    // Only validate when sizes match the declared dim.
    let actual_units;

    if let Some(ref req_units) = opts.units {
        let from = parse_units(vardef.default_units)?;
        let to = parse_units(req_units)?;
        convert_array(&mut data, from, to)?;
        actual_units = req_units.clone();
    } else {
        actual_units = vardef.default_units.to_string();
    }

    // If data length doesn't match the standard shape, adjust shape to reflect
    // the actual output (multi-field variables).
    let final_shape = if data.len() == expected {
        shape
    } else if data.len() % expected == 0 {
        let nfields = data.len() / expected;
        match vardef.dim {
            VarDim::TwoD => vec![nfields, file.ny, file.nx],
            VarDim::ThreeD => vec![nfields, file.nz, file.ny, file.nx],
        }
    } else {
        shape
    };

    Ok(VarOutput {
        data,
        shape: final_shape,
        units: actual_units,
        description: vardef.description.to_string(),
    })
}

/// Compute and stack all timesteps for a variable inside Rust so Python does
/// not need to bounce across the FFI boundary once per timestep.
pub fn getvar_all_times(file: &WrfFile, name: &str, opts: &ComputeOpts) -> WrfResult<VarOutput> {
    if file.nt == 0 {
        return Err(WrfError::InvalidParam(
            "cannot stack ALL_TIMES for a file with zero timesteps".into(),
        ));
    }

    let first = getvar(file, name, Some(0), opts)?;
    let frame_len = first.data.len();
    let mut data = Vec::with_capacity(frame_len * file.nt);
    data.extend_from_slice(&first.data);

    for t in 1..file.nt {
        let frame = getvar(file, name, Some(t), opts)?;
        if frame.shape != first.shape {
            return Err(WrfError::DimMismatch(format!(
                "shape mismatch for ALL_TIMES on '{name}': {:?} at t=0 vs {:?} at t={t}",
                first.shape, frame.shape
            )));
        }
        data.extend_from_slice(&frame.data);
    }

    let mut shape = Vec::with_capacity(first.shape.len() + 1);
    shape.push(file.nt);
    shape.extend(first.shape.iter().copied());

    Ok(VarOutput {
        data,
        shape,
        units: first.units,
        description: first.description,
    })
}

/// Fallback: read a raw WRF variable directly from the file.
/// Used when the variable name is not in the computed registry.
fn getvar_raw(file: &WrfFile, name: &str, t: usize, opts: &ComputeOpts) -> WrfResult<VarOutput> {
    let upper_name = name.to_uppercase();
    // Try reading the variable by its exact name (case-sensitive, uppercase WRF convention)
    let data = file
        .read_var(name, t)
        .or_else(|_| file.read_var(&upper_name, t))
        .map_err(|_| {
            WrfError::UnknownVar(format!(
                "{name} (not in computed registry and not found as raw variable in file)"
            ))
        })?;

    let shape = file
        .var_shape_no_time(name)
        .or_else(|_| file.var_shape_no_time(&upper_name))
        .ok()
        .filter(|shape| !shape.is_empty() && shape.iter().product::<usize>() == data.len())
        .unwrap_or_else(|| vec![data.len()]);

    // Raw variables don't have a known default unit, but if the user
    // requests a conversion from/to a specific pair we can try.
    // Common WRF raw variables and their units:
    let default_unit = match upper_name.as_str() {
        "RAINNC" | "RAINC" | "RAINSH" | "SNOWNC" | "GRAUPELNC" => "mm",
        "T2" | "TSK" | "SST" => "K",
        "PSFC" => "Pa",
        "PBLH" | "SNOWH" => "m",
        "HFX" | "LH" => "W/m2",
        "SWDOWN" | "GLW" | "OLR" => "W/m2",
        "UST" => "m/s",
        "U10" | "V10" => "m/s",
        _ => "",
    };

    let mut data = data;
    let actual_units = if let Some(ref req_units) = opts.units {
        if !default_unit.is_empty() {
            if let (Ok(from), Ok(to)) = (parse_units(default_unit), parse_units(req_units)) {
                let _ = convert_array(&mut data, from, to);
            }
        }
        req_units.clone()
    } else {
        default_unit.to_string()
    };

    Ok(VarOutput {
        data,
        shape,
        units: actual_units,
        description: format!("Raw WRF variable: {name}"),
    })
}
