//! LANE 3.  Raster substrate for the high-resolution overlay path:
//! GeoTIFF decode/encode, CRS transforms, mosaic and the
//! area-average/nearest/bilinear warp that replaces
//! rasterio/GDAL+pyproj in `gpuwm/static/highres*.py`.
//!
//! Parity posture differs from the WPS path and is deliberate: GDAL's
//! warper is a black box, not a spec, so this substrate implements the
//! DEFINED behaviour (declared below and in
//! docs/dev/static-rust-port.md) and the parity harness gates on
//! quantitative tolerance against the rasterio output on pinned
//! footprints, with the divergence documented.  Everything downstream
//! of the resampled planes (USDA triangle, crosswalk, donor fill,
//! merges) is byte-parity Python-vs-Rust.
//!
//! Sources this must decode (from `highres_fetch.py`): staged 1x1-deg
//! elevation GeoTIFF tiles (f32, deflate, tiled), near-global DEM COGs
//! (f32, deflate, tiled, latitude-banded columns), the CONUS annual
//! land-cover raster (u8, deflate, tiled, Albers/NAD83), soil WCS
//! GeoTIFF windows (i16, Interrupted Goode Homolosine via the recorded
//! crs_override), plus the crate's own deflate-compressed GeoTIFF
//! output for derived windows.  Decode dependencies: `miniz_oxide`
//! (deflate), `weezl` (LZW), `sha2` (bound-raster verification) — all
//! lockfile-pinned workspace residents; see `tools/rustwx/VENDOR.md`.

pub mod geotiff;
pub mod proj;
pub mod warp;

use crate::error::{Result, StaticError};
use crate::raster::proj::{
    AeaEllipsoid, GeographicIdentity, Igh, LccSphere, MercSphere,
    PointProjection, StereSpherePolar, GRS80_A, GRS80_RF, IGH_WGS84_A,
};
use crate::projection::ProjectionKind;
use crate::EARTH_RADIUS_M;

/// One in-memory raster plane with georeferencing: f64 values (NaN =
/// nodata after masking), north-first rows exactly as the GeoTIFF
/// stores them, affine transform `(a, b, c, d, e, f)` mapping pixel
/// (col, row) centres via `x = c + a*(col+0.5) + b*(row+0.5)` etc.
#[derive(Debug, Clone)]
pub struct Raster {
    pub ny: usize,
    pub nx: usize,
    pub values: Vec<f64>,
    pub transform: [f64; 6],
    pub crs: Crs,
}

impl Raster {
    /// Pixel-centre projected coordinates of `(col, row)`.
    #[inline]
    pub fn centre(&self, col: usize, row: usize) -> (f64, f64) {
        let t = &self.transform;
        let c = col as f64 + 0.5;
        let r = row as f64 + 0.5;
        (t[2] + t[0] * c + t[1] * r, t[5] + t[3] * c + t[4] * r)
    }

    /// (west, south, east, north) bounds of the pixel grid, north-up
    /// rectilinear transforms only.
    pub fn bounds(&self) -> [f64; 4] {
        let t = &self.transform;
        let west = t[2];
        let north = t[5];
        let east = west + t[0] * self.nx as f64;
        let south = north + t[4] * self.ny as f64;
        [west, south, east, north]
    }
}

/// The CRS inventory the overlay path actually needs -- a closed set,
/// spelled as data, so a new source is a table row (the arbitrary
/// acceptance test), not a projection engine.
#[derive(Debug, Clone, PartialEq)]
pub enum Crs {
    /// EPSG:4326 lon/lat degrees.
    Geographic,
    /// The model grid's own spherical CRS (R = EARTH_RADIUS_M),
    /// parameterized by the grid spec -- lcc/merc/stere on the sphere.
    ModelSphere(crate::projection::GridSpec),
    /// Interrupted Goode Homolosine on WGS84 (the soil source's native
    /// plane; the delivered GeoTIFF carries no CRS tag, hence the
    /// override recorded on every bound raster).
    InterruptedGoodeHomolosine,
    /// Ellipsoidal Albers equal-area as parameterized by the source
    /// raster's own tags (the CONUS land-cover collection).
    AlbersConusNad83 {
        lat_1: f64,
        lat_2: f64,
        lat_0: f64,
        lon_0: f64,
        false_easting: f64,
        false_northing: f64,
    },
}

impl Crs {
    /// Parse a recorded `crs_override` string.  The inventory is
    /// closed: the one recorded override in the tree is the soil
    /// source's Interrupted Goode Homolosine proj string; anything
    /// else is refused by name rather than misread.
    pub fn parse_override(text: &str) -> Result<Crs> {
        let trimmed = text.trim();
        if trimmed == "igh"
            || (trimmed.contains("+proj=igh")
                && trimmed.contains("+datum=WGS84"))
        {
            return Ok(Crs::InterruptedGoodeHomolosine);
        }
        if trimmed.eq_ignore_ascii_case("EPSG:4326")
            || trimmed.eq_ignore_ascii_case("EPSG:4269")
        {
            return Ok(Crs::Geographic);
        }
        Err(StaticError::Invalid(format!(
            "crs_override {text:?} is not in the raster substrate's \
             closed CRS inventory (known: the SoilGrids Interrupted \
             Goode Homolosine proj string, EPSG:4326, EPSG:4269)"
        )))
    }

    /// The point transform for this CRS (degrees <-> metres).
    pub(crate) fn point_projection(&self) -> Result<Box<dyn PointProjection>> {
        Ok(match self {
            Crs::Geographic => Box::new(GeographicIdentity),
            Crs::InterruptedGoodeHomolosine => Box::new(Igh::new(IGH_WGS84_A)),
            Crs::AlbersConusNad83 {
                lat_1,
                lat_2,
                lat_0,
                lon_0,
                false_easting,
                false_northing,
            } => Box::new(AeaEllipsoid::new(
                *lat_1,
                *lat_2,
                *lat_0,
                *lon_0,
                *false_easting,
                *false_northing,
                GRS80_A,
                GRS80_RF,
            )),
            Crs::ModelSphere(spec) => match spec.kind {
                ProjectionKind::Lambert => Box::new(LccSphere::new(
                    spec.truelat1,
                    spec.truelat2,
                    spec.truelat1,
                    spec.stand_lon,
                    EARTH_RADIUS_M,
                )),
                // module_llxy Mercator is anchored at the known point's
                // longitude, not stand_lon (`_grid_crs`).
                ProjectionKind::Mercator => Box::new(MercSphere::new(
                    spec.truelat1,
                    spec.ref_lon,
                    EARTH_RADIUS_M,
                )),
                ProjectionKind::Polar => Box::new(StereSpherePolar::new(
                    spec.truelat1 >= 0.0,
                    spec.truelat1,
                    spec.stand_lon,
                    EARTH_RADIUS_M,
                )),
            },
        })
    }
}

/// Forward/inverse point transforms between any two [`Crs`] members.
/// Validated against the Python geodesy stack on fixture points to
/// <= 1e-6 m before any warp uses them (validate-the-instrument).
///
/// The hub is geographic degrees: `from` is inverted to lon/lat, `to`
/// applied forward — exactly the pipeline PROJ builds for these CRS
/// pairs (no datum shift exists between them, so PROJ's own pipeline
/// is the ballpark lon/lat pass-through this reproduces).
pub fn transform_points(
    from: &Crs,
    to: &Crs,
    x: &mut [f64],
    y: &mut [f64],
) -> Result<()> {
    if x.len() != y.len() {
        return Err(StaticError::Invalid(format!(
            "transform_points x/y length mismatch: {} vs {}",
            x.len(),
            y.len()
        )));
    }
    if from == to {
        return Ok(());
    }
    let inv = from.point_projection()?;
    let fwd = to.point_projection()?;
    use rayon::prelude::*;
    x.par_iter_mut()
        .zip(y.par_iter_mut())
        .with_min_len(4096)
        .for_each(|(px, py)| {
            let (lon, lat) = inv.inverse(*px, *py);
            let (ox, oy) = fwd.forward(lon, lat);
            *px = ox;
            *py = oy;
        });
    Ok(())
}
