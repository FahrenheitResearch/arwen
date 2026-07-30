/// Map projections for meteorological grids.
///
/// Core projection types and the `Projection` trait are defined in `wx-field`
/// and re-exported here. This module adds GRIB2-specific convenience constructors
/// that depend on wx-core's GridDefinition type.

// Re-export everything from wx-field::projection so existing code continues to work.
pub use wx_field::projection::*;

use crate::grib2::parser::GridDefinition;

// ============================================================
// GRIB2 GridDefinition constructors (depend on wx-core types)
// These are free functions because we cannot add inherent impls
// to types defined in wx-field.
// ============================================================

/// Build a PolarStereoProjection from a GRIB2 GridDefinition with template 20.
pub fn polar_stereo_from_grid_def(g: &GridDefinition) -> PolarStereoProjection {
    let south_pole = (g.projection_center_flag & 1) != 0;
    PolarStereoProjection::new(g.lov, g.lad, g.lat1, g.lon1, g.dx, g.dy, g.nx, g.ny, south_pole)
}

/// Build a MercatorProjection from a GRIB2 GridDefinition with template 10.
pub fn mercator_from_grid_def(g: &GridDefinition) -> MercatorProjection {
    MercatorProjection::new(g.lad, g.lat1, g.lon1, g.dx, g.dy, g.nx, g.ny)
}

/// Build a GaussianProjection from a GRIB2 GridDefinition with template 40.
pub fn gaussian_from_grid_def(g: &GridDefinition) -> GaussianProjection {
    GaussianProjection::new(g.lat1, g.lon1, g.lat2, g.lon2, g.nx, g.ny)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lambert_roundtrip() {
        let proj = LambertProjection::new(
            33.0, 45.0, -97.0,
            21.0, -122.0,
            3000.0, 3000.0,
            500, 400,
        );

        let (lat, lon) = proj.grid_to_latlon(0.0, 0.0);
        assert!((lat - 21.0).abs() < 0.1, "lat={}", lat);
        assert!((lon - (-122.0)).abs() < 0.1, "lon={}", lon);

        let (lat_mid, lon_mid) = proj.grid_to_latlon(250.0, 200.0);
        let (i, j) = proj.latlon_to_grid(lat_mid, lon_mid);
        assert!((i - 250.0).abs() < 0.01, "i={}", i);
        assert!((j - 200.0).abs() < 0.01, "j={}", j);
    }

    #[test]
    fn test_lambert_from_wrf() {
        let proj = LambertProjection::from_wrf(
            33.0, 45.0, -97.0,
            39.0, -97.0,
            3000.0, 3000.0,
            500, 400,
        );

        let (ci, cj) = proj.latlon_to_grid(39.0, -97.0);
        assert!((ci - 249.5).abs() < 1.0, "ci={}", ci);
        assert!((cj - 199.5).abs() < 1.0, "cj={}", cj);
    }

    #[test]
    fn test_latlon_roundtrip() {
        let proj = LatLonProjection::new(
            20.0, -130.0,  // SW corner
            55.0, -60.0,   // NE corner
            281, 141,
        );

        // Grid origin should map to SW corner
        let (lat, lon) = proj.grid_to_latlon(0.0, 0.0);
        assert!((lat - 20.0).abs() < 0.01);
        assert!((lon - (-130.0)).abs() < 0.01);

        // Roundtrip through center
        let mid_lat = 37.5;
        let mid_lon = -95.0;
        let (i, j) = proj.latlon_to_grid(mid_lat, mid_lon);
        let (lat2, lon2) = proj.grid_to_latlon(i, j);
        assert!((lat2 - mid_lat).abs() < 0.01);
        assert!((lon2 - mid_lon).abs() < 0.01);
    }

    #[test]
    fn test_projection_trait_dyn() {
        // Verify the trait object works
        let lambert: Box<dyn Projection> = Box::new(LambertProjection::new(
            33.0, 45.0, -97.0, 21.0, -122.0, 3000.0, 3000.0, 500, 400,
        ));
        let latlon: Box<dyn Projection> = Box::new(LatLonProjection::new(
            20.0, -130.0, 55.0, -60.0, 281, 141,
        ));

        assert_eq!(lambert.nx(), 500);
        assert_eq!(latlon.nx(), 281);

        let (lat, lon) = lambert.grid_to_latlon(0.0, 0.0);
        assert!(lat > 0.0);
        assert!(lon < 0.0);
    }

    #[test]
    fn test_polar_stereo_roundtrip() {
        // North pole projection -- approximate HRRR-Alaska-like setup
        let proj = PolarStereoProjection::new(
            -150.0,  // lov
            60.0,    // lad
            40.0,    // la1
            -170.0,  // lo1
            3000.0, 3000.0,
            200, 200,
            false,   // north pole
        );

        // Grid origin should map back to la1, lo1
        let (lat, lon) = proj.grid_to_latlon(0.0, 0.0);
        assert!((lat - 40.0).abs() < 0.1, "lat={}", lat);
        assert!((lon - (-170.0)).abs() < 0.1, "lon={}", lon);

        // Roundtrip through a middle point
        let (lat_mid, lon_mid) = proj.grid_to_latlon(100.0, 100.0);
        let (i, j) = proj.latlon_to_grid(lat_mid, lon_mid);
        assert!((i - 100.0).abs() < 0.01, "i={}", i);
        assert!((j - 100.0).abs() < 0.01, "j={}", j);
    }

    #[test]
    fn test_polar_stereo_south() {
        let proj = PolarStereoProjection::new(
            0.0,     // lov
            -60.0,   // lad
            -70.0,   // la1
            -30.0,   // lo1
            5000.0, 5000.0,
            100, 100,
            true,    // south pole
        );

        let (lat, _lon) = proj.grid_to_latlon(0.0, 0.0);
        assert!((lat - (-70.0)).abs() < 0.1, "lat={}", lat);

        let (lat2, lon2) = proj.grid_to_latlon(50.0, 50.0);
        let (i, j) = proj.latlon_to_grid(lat2, lon2);
        assert!((i - 50.0).abs() < 0.01, "i={}", i);
        assert!((j - 50.0).abs() < 0.01, "j={}", j);
    }

    #[test]
    fn test_mercator_roundtrip() {
        let proj = MercatorProjection::new(
            20.0,    // lad
            10.0,    // la1
            -80.0,   // lo1
            5000.0, 5000.0,
            200, 150,
        );

        let (lat, lon) = proj.grid_to_latlon(0.0, 0.0);
        assert!((lat - 10.0).abs() < 0.1, "lat={}", lat);
        assert!((lon - (-80.0)).abs() < 0.1, "lon={}", lon);

        let (lat_mid, lon_mid) = proj.grid_to_latlon(100.0, 75.0);
        let (i, j) = proj.latlon_to_grid(lat_mid, lon_mid);
        assert!((i - 100.0).abs() < 0.01, "i={}", i);
        assert!((j - 75.0).abs() < 0.01, "j={}", j);
    }

    #[test]
    fn test_gaussian_roundtrip() {
        let proj = GaussianProjection::new(
            90.0, 0.0,    // lat1, lon1 (north pole, dateline)
            -90.0, 359.5, // lat2, lon2
            720, 361,
        );

        let (lat, lon) = proj.grid_to_latlon(0.0, 0.0);
        assert!((lat - 90.0).abs() < 0.01);
        assert!((lon - 0.0).abs() < 0.01);

        let (i, j) = proj.latlon_to_grid(0.0, 180.0);
        let (lat2, lon2) = proj.grid_to_latlon(i, j);
        assert!((lat2 - 0.0).abs() < 0.01);
        assert!((lon2 - 180.0).abs() < 0.01);
    }

    #[test]
    fn test_new_projections_trait_dyn() {
        let polar: Box<dyn Projection> = Box::new(PolarStereoProjection::new(
            -150.0, 60.0, 40.0, -170.0, 3000.0, 3000.0, 200, 200, false,
        ));
        let mercator: Box<dyn Projection> = Box::new(MercatorProjection::new(
            20.0, 10.0, -80.0, 5000.0, 5000.0, 200, 150,
        ));
        let gaussian: Box<dyn Projection> = Box::new(GaussianProjection::new(
            90.0, 0.0, -90.0, 359.5, 720, 361,
        ));

        assert_eq!(polar.nx(), 200);
        assert_eq!(mercator.nx(), 200);
        assert_eq!(gaussian.nx(), 720);
    }
}
