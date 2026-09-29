//! LANE 1.  Explicit latitude rows on a uniform longitude ring (the
//! `rows` kind): the target-grid description a global spectral model's
//! Gaussian grid crosses the seam as.
//!
//! No WPS map projection describes a Gaussian grid, so this kind is
//! declared by its DATA rather than by parameters: the row latitudes
//! themselves (`GridSpec::lat_deg`, ascending row centres, handed over
//! by the caller that owns the Gauss-Legendre nodes -- nothing here
//! re-derives them), a first-column longitude and a uniform longitude
//! spacing.  The projection coordinate is the mass-grid cell index as
//! everywhere else in this crate: column `x` sits at
//! `lon0 + (x - 1) * dlon`, row `y` at the piecewise-linear latitude
//! between neighbouring row centres (extrapolated with the end spacing
//! beyond the first and last rows, clamped to the poles).
//!
//! The inverse puts a longitude into `[0.5, nlon + 0.5)` when the ring
//! closes (`nlon * dlon == 360`), so the half cell WEST of the first
//! column bins onto column 1 through the sampler's `nint` rule instead
//! of falling off the east edge; a latitude past the last row centre by
//! less than half a row spacing bins onto that row, further poleward
//! than that it is outside the grid (no cell owns the pole cap).
//!
//! Map factor is 1 and rotation is identity: the consumer of this grid
//! is a spectral model that carries its own metric, and the crate uses
//! neither while building static fields.

use super::{GridSpec, ProjectedGrid, Wps32Twin};
use crate::error::{Result, StaticError};

/// Derived state: the validated row table and ring parameters.
#[derive(Debug, Clone)]
pub struct RowsState {
    pub lat: Vec<f64>,
    pub lon0: f64,
    pub dlon: f64,
}

/// Validate the row table and ring parameters carried by the spec.
pub fn setup(spec: &GridSpec) -> Result<RowsState> {
    let ny = spec.e_sn - 1;
    if ny < 2 {
        return Err(StaticError::Invalid(format!(
            "RowsGrid needs at least two latitude rows, e_sn={} gives {ny}",
            spec.e_sn
        )));
    }
    if spec.lat_deg.len() as i64 != ny {
        return Err(StaticError::Invalid(format!(
            "RowsGrid declares {} latitude rows but e_sn={} implies {ny} \
             mass rows",
            spec.lat_deg.len(),
            spec.e_sn
        )));
    }
    for (k, &lat) in spec.lat_deg.iter().enumerate() {
        if !lat.is_finite() || !(-90.0..=90.0).contains(&lat) {
            return Err(StaticError::Invalid(format!(
                "RowsGrid latitude row {k} is {lat}, outside [-90, 90]"
            )));
        }
        if k > 0 && lat <= spec.lat_deg[k - 1] {
            return Err(StaticError::Invalid(format!(
                "RowsGrid latitude rows must ascend strictly; row {k} \
                 ({lat}) does not exceed row {} ({})",
                k - 1,
                spec.lat_deg[k - 1]
            )));
        }
    }
    if !spec.dlon_deg.is_finite() || spec.dlon_deg <= 0.0 {
        return Err(StaticError::Invalid(format!(
            "RowsGrid dlon_deg must be finite and positive, got {}",
            spec.dlon_deg
        )));
    }
    if !spec.lon0_deg.is_finite() {
        return Err(StaticError::Invalid(format!(
            "RowsGrid lon0_deg must be finite, got {}",
            spec.lon0_deg
        )));
    }
    Ok(RowsState {
        lat: spec.lat_deg.clone(),
        lon0: spec.lon0_deg,
        dlon: spec.dlon_deg,
    })
}

/// Longitude into (-180, 180].
fn wrap_lon(lon: f64) -> f64 {
    let mut out = (lon + 180.0).rem_euclid(360.0) - 180.0;
    if out <= -180.0 {
        out += 360.0;
    }
    out
}

/// Row latitude at fractional row coordinate `y` (1-based rows).
fn row_latitude(state: &RowsState, y: f64) -> f64 {
    let lat = &state.lat;
    let ny = lat.len();
    let value = if y <= 1.0 {
        lat[0] + (y - 1.0) * (lat[1] - lat[0])
    } else if y >= ny as f64 {
        lat[ny - 1] + (y - ny as f64) * (lat[ny - 1] - lat[ny - 2])
    } else {
        let j = y.floor() as usize; // 1-based lower row
        let frac = y - j as f64;
        lat[j - 1] + frac * (lat[j] - lat[j - 1])
    };
    value.clamp(-90.0, 90.0)
}

/// Fractional row coordinate of a latitude (inverse of `row_latitude`
/// inside the row table, end-spacing extrapolation beyond it).
fn row_coordinate(state: &RowsState, lat_deg: f64) -> f64 {
    let lat = &state.lat;
    let ny = lat.len();
    // rows strictly below lat_deg
    let below = lat.partition_point(|&v| v < lat_deg);
    if below == 0 {
        1.0 + (lat_deg - lat[0]) / (lat[1] - lat[0])
    } else if below == ny {
        ny as f64 + (lat_deg - lat[ny - 1]) / (lat[ny - 1] - lat[ny - 2])
    } else {
        below as f64 + (lat_deg - lat[below - 1]) / (lat[below] - lat[below - 1])
    }
}

/// Projection coordinate -> (lat, lon) degrees.
pub fn ij_to_latlon(state: &RowsState, x: f64, y: f64) -> (f64, f64) {
    let lon = wrap_lon(state.lon0 + (x - 1.0) * state.dlon);
    (row_latitude(state, y), lon)
}

/// (lat, lon) degrees -> projection coordinate.
pub fn latlon_to_ij(state: &RowsState, lat: f64, lon: f64) -> (f64, f64) {
    let mut d = (lon - state.lon0).rem_euclid(360.0);
    if d >= 360.0 - 0.5 * state.dlon {
        d -= 360.0;
    }
    let x = d / state.dlon + 1.0;
    (x, row_coordinate(state, lat))
}

/// The sampling twin: there is no WPS single-precision counterpart for
/// this kind, so the twin is the float64 transform rounded to f32 (the
/// sampler consumes it only for window bounds and the >= 1 km f64
/// interpolation path, never for the sub-kilometre stencil bands).
pub struct RowsTwin {
    grid: ProjectedGrid,
}

impl RowsTwin {
    pub fn new(grid: &ProjectedGrid) -> Self {
        RowsTwin { grid: grid.clone() }
    }
}

impl Wps32Twin for RowsTwin {
    fn ij_to_latlon32(&self, x: f32, y: f32) -> (f32, f32) {
        let (lat, lon) = self.grid.ij_to_latlon(x as f64, y as f64);
        (lat as f32, lon as f32)
    }

    fn latlon_to_ij32(&self, lat: f32, lon: f32) -> (f32, f32) {
        let (x, y) = self.grid.latlon_to_ij(lat as f64, lon as f64);
        (x as f32, y as f32)
    }

    fn adopt_public_pole(&mut self, _grid: &ProjectedGrid) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::projection::ProjectionKind;
    use crate::sampler::bin_grid_coords;

    fn spec(lat: Vec<f64>, nlon: i64) -> GridSpec {
        GridSpec {
            kind: ProjectionKind::Rows,
            ref_lat: 0.0,
            ref_lon: 0.0,
            truelat1: 0.0,
            truelat2: 0.0,
            stand_lon: 0.0,
            dx: 2.0 * std::f64::consts::PI * crate::EARTH_RADIUS_M / nlon as f64,
            dy: 2.0 * std::f64::consts::PI * crate::EARTH_RADIUS_M / nlon as f64,
            e_we: nlon + 1,
            e_sn: lat.len() as i64 + 1,
            known_x: 1.0,
            known_y: 1.0,
            moad_cen_lat: 0.0,
            moad_cen_lon: 0.0,
            lat_deg: lat,
            lon0_deg: 0.0,
            dlon_deg: 360.0 / nlon as f64,
        }
    }

    /// A small Gaussian-like row table (T3-ish: 8 rows, not uniform).
    fn rows() -> Vec<f64> {
        vec![-79.6, -60.2, -40.4, -20.5, 20.5, 40.4, 60.2, 79.6]
    }

    #[test]
    fn round_trips_row_centres_and_columns() {
        let grid = ProjectedGrid::new(spec(rows(), 16)).unwrap();
        for j in 1..=8 {
            for i in 1..=16 {
                let (lat, lon) = grid.ij_to_latlon(i as f64, j as f64);
                assert!((lat - rows()[j - 1]).abs() < 1e-12);
                let expect = (i - 1) as f64 * 22.5;
                let expect = if expect > 180.0 { expect - 360.0 } else { expect };
                assert!((lon - expect).abs() < 1e-12, "col {i}: {lon} vs {expect}");
                let (x, y) = grid.latlon_to_ij(lat, lon);
                assert!((x - i as f64).abs() < 1e-9, "x {x} vs {i}");
                assert!((y - j as f64).abs() < 1e-9, "y {y} vs {j}");
            }
        }
    }

    #[test]
    fn west_half_cell_bins_onto_column_one_and_poles_are_owned_by_end_rows() {
        let grid = ProjectedGrid::new(spec(rows(), 16)).unwrap();
        // 5 degrees west of the first column: inside column 1's half cell
        let (x, _) = grid.latlon_to_ij(0.0, -5.0);
        assert!(x >= 0.5 && x < 1.0, "x = {x}");
        // exactly half a cell west lands on the lower edge, not past nlon
        let (x, _) = grid.latlon_to_ij(0.0, -11.25);
        assert!((x - 0.5).abs() < 1e-9, "x = {x}");
        // just short of half a cell east of the last column stays inside
        let (x, _) = grid.latlon_to_ij(0.0, -11.26);
        assert!(x < 16.5 && x > 16.0, "x = {x}");
        let cells = bin_grid_coords(&[x], &[4.0], 16, 8, 0);
        assert_eq!(cells, vec![3 * 16 + 15]);
        // the pole cap within half a row spacing folds onto the end row
        let (_, y) = grid.latlon_to_ij(89.0, 10.0);
        assert!(y > 8.0 && y < 8.5, "y = {y}");
        let (_, y) = grid.latlon_to_ij(-90.0, 10.0);
        assert!(y < 1.0, "y = {y}");
        // row latitudes beyond the table clamp at the poles
        let (lat, _) = grid.ij_to_latlon(1.0, 9.0);
        assert_eq!(lat, 90.0);
        let (lat, _) = grid.ij_to_latlon(1.0, 0.0);
        assert_eq!(lat, -90.0);
    }

    #[test]
    fn a_translated_sector_bins_exactly_like_the_global_ring() {
        let global = ProjectedGrid::new(spec(rows(), 16)).unwrap();
        let sector = global.translated(8, 0, Some(9), None).unwrap();
        let (lat, lon) = (-20.5, 200.7);
        let (gx, gy) = global.latlon_to_ij(lat, lon);
        let (sx, sy) = sector.latlon_to_ij(lat, lon);
        assert_eq!(gx.to_bits(), (sx + 8.0).to_bits());
        assert_eq!(gy.to_bits(), sy.to_bits());
        let (slat, slon) = sector.ij_to_latlon(1.0, 4.0);
        let (glat, glon) = global.ij_to_latlon(9.0, 4.0);
        assert_eq!(slat.to_bits(), glat.to_bits());
        assert_eq!(slon.to_bits(), glon.to_bits());
    }

    #[test]
    fn refuses_a_row_table_that_does_not_match_the_extent() {
        let mut bad = spec(rows(), 16);
        bad.e_sn = 7;
        assert!(ProjectedGrid::new(bad).is_err());
        let mut unsorted = spec(rows(), 16);
        unsorted.lat_deg[3] = -50.0;
        assert!(ProjectedGrid::new(unsorted).is_err());
        let global = ProjectedGrid::new(spec(rows(), 16)).unwrap();
        assert!(global.nest(1, 1, 3, 10, 10, None, None).is_err());
    }
}
