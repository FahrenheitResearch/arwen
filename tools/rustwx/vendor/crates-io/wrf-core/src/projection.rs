use crate::error::{WrfError, WrfResult};
use crate::file::WrfFile;

/// WRF map projection type (MAP_PROJ attribute).
#[derive(Debug, Clone)]
pub enum WrfProjection {
    Lambert {
        truelat1: f64,
        truelat2: f64,
        stand_lon: f64,
        cen_lat: f64,
        cen_lon: f64,
        dx: f64,
        dy: f64,
    },
    PolarStereographic {
        truelat1: f64,
        stand_lon: f64,
        cen_lat: f64,
        cen_lon: f64,
        dx: f64,
        dy: f64,
    },
    Mercator {
        truelat1: f64,
        cen_lon: f64,
        dx: f64,
        dy: f64,
    },
    LatLon {
        cen_lat: f64,
        cen_lon: f64,
        dx: f64,
        dy: f64,
    },
}

impl WrfProjection {
    /// Extract projection parameters from a WRF file's global attributes.
    pub fn from_file(f: &WrfFile) -> WrfResult<Self> {
        let map_proj = f.global_attr_i32("MAP_PROJ")?;
        let dx = f.global_attr_f64("DX").unwrap_or(f.dx);
        let dy = f.global_attr_f64("DY").unwrap_or(f.dy);

        match map_proj {
            1 => {
                // Lambert Conformal
                let truelat1 = f.global_attr_f64("TRUELAT1")?;
                let truelat2 = f.global_attr_f64("TRUELAT2")?;
                let stand_lon = f.global_attr_f64("STAND_LON")?;
                let cen_lat = f.global_attr_f64("CEN_LAT")?;
                let cen_lon = f.global_attr_f64("CEN_LON")?;
                Ok(Self::Lambert {
                    truelat1,
                    truelat2,
                    stand_lon,
                    cen_lat,
                    cen_lon,
                    dx,
                    dy,
                })
            }
            2 => {
                // Polar Stereographic
                let truelat1 = f.global_attr_f64("TRUELAT1")?;
                let stand_lon = f.global_attr_f64("STAND_LON")?;
                let cen_lat = f.global_attr_f64("CEN_LAT")?;
                let cen_lon = f.global_attr_f64("CEN_LON")?;
                Ok(Self::PolarStereographic {
                    truelat1,
                    stand_lon,
                    cen_lat,
                    cen_lon,
                    dx,
                    dy,
                })
            }
            3 => {
                // Mercator
                let truelat1 = f.global_attr_f64("TRUELAT1")?;
                let cen_lon = f.global_attr_f64("CEN_LON")?;
                Ok(Self::Mercator {
                    truelat1,
                    cen_lon,
                    dx,
                    dy,
                })
            }
            6 => {
                // Lat-Lon (cylindrical equidistant)
                let cen_lat = f.global_attr_f64("CEN_LAT")?;
                let cen_lon = f.global_attr_f64("CEN_LON")?;
                Ok(Self::LatLon {
                    cen_lat,
                    cen_lon,
                    dx,
                    dy,
                })
            }
            other => Err(WrfError::InvalidParam(format!(
                "unsupported MAP_PROJ value: {other}"
            ))),
        }
    }

    /// Grid spacing in X (meters).
    pub fn dx(&self) -> f64 {
        match self {
            Self::Lambert { dx, .. }
            | Self::PolarStereographic { dx, .. }
            | Self::Mercator { dx, .. }
            | Self::LatLon { dx, .. } => *dx,
        }
    }

    /// Grid spacing in Y (meters).
    pub fn dy(&self) -> f64 {
        match self {
            Self::Lambert { dy, .. }
            | Self::PolarStereographic { dy, .. }
            | Self::Mercator { dy, .. }
            | Self::LatLon { dy, .. } => *dy,
        }
    }
}
