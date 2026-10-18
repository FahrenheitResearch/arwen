#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SweepAngleAxis {
    X,
    Y,
}

impl SweepAngleAxis {
    pub fn parse(value: &str) -> Self {
        match value.trim().to_ascii_lowercase().as_str() {
            "y" => Self::Y,
            _ => Self::X,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::X => "x",
            Self::Y => "y",
        }
    }
}

pub fn scan_angles_to_lat_lon(
    perspective_point_height_m: f64,
    semi_major_axis_m: f64,
    semi_minor_axis_m: f64,
    longitude_of_projection_origin_deg: f64,
    sweep_angle_axis: SweepAngleAxis,
    x_rad: f64,
    y_rad: f64,
) -> Option<(f32, f32)> {
    if !perspective_point_height_m.is_finite()
        || !semi_major_axis_m.is_finite()
        || !semi_minor_axis_m.is_finite()
        || !longitude_of_projection_origin_deg.is_finite()
        || !x_rad.is_finite()
        || !y_rad.is_finite()
    {
        return None;
    }

    let h = perspective_point_height_m + semi_major_axis_m;
    let a = semi_major_axis_m;
    let b = semi_minor_axis_m;
    if h <= 0.0 || a <= 0.0 || b <= 0.0 {
        return None;
    }

    if sweep_angle_axis == SweepAngleAxis::Y {
        return scan_angles_to_lat_lon_sweep_y(
            h,
            a,
            b,
            longitude_of_projection_origin_deg,
            x_rad,
            y_rad,
        );
    }
    let (x, y) = (x_rad, y_rad);

    let sin_x = x.sin();
    let cos_x = x.cos();
    let sin_y = y.sin();
    let cos_y = y.cos();
    let eq_to_pol = (a * a) / (b * b);

    let a_var = sin_x * sin_x + cos_x * cos_x * (cos_y * cos_y + eq_to_pol * sin_y * sin_y);
    let b_var = -2.0 * h * cos_x * cos_y;
    let c_var = h * h - a * a;
    let discriminant = b_var * b_var - 4.0 * a_var * c_var;
    if discriminant < 0.0 {
        return None;
    }

    let r_s = (-b_var - discriminant.sqrt()) / (2.0 * a_var);
    if !r_s.is_finite() || r_s <= 0.0 {
        return None;
    }

    let s_x = r_s * cos_x * cos_y;
    let s_y = -r_s * sin_x;
    let s_z = r_s * cos_x * sin_y;

    let latitude = (eq_to_pol * (s_z / ((h - s_x).hypot(s_y)))).atan();
    let longitude = longitude_of_projection_origin_deg.to_radians() - (s_y / (h - s_x)).atan();
    let lat_deg = latitude.to_degrees();
    let lon_deg = normalize_longitude_deg(longitude.to_degrees());

    if !lat_deg.is_finite() || !lon_deg.is_finite() {
        return None;
    }
    Some((lat_deg as f32, lon_deg as f32))
}

pub fn lat_lon_to_scan_angles(
    perspective_point_height_m: f64,
    semi_major_axis_m: f64,
    semi_minor_axis_m: f64,
    longitude_of_projection_origin_deg: f64,
    sweep_angle_axis: SweepAngleAxis,
    latitude_deg: f64,
    longitude_deg: f64,
) -> Option<(f64, f64)> {
    if !perspective_point_height_m.is_finite()
        || !semi_major_axis_m.is_finite()
        || !semi_minor_axis_m.is_finite()
        || !longitude_of_projection_origin_deg.is_finite()
        || !latitude_deg.is_finite()
        || !longitude_deg.is_finite()
    {
        return None;
    }

    let h = perspective_point_height_m + semi_major_axis_m;
    let a = semi_major_axis_m;
    let b = semi_minor_axis_m;
    if h <= 0.0 || a <= 0.0 || b <= 0.0 {
        return None;
    }

    let lat = latitude_deg.to_radians();
    let lon_delta = (longitude_deg - longitude_of_projection_origin_deg).to_radians();
    let geocentric_lat = ((b * b) / (a * a) * lat.tan()).atan();
    let cos_geocentric_lat = geocentric_lat.cos();
    let sin_geocentric_lat = geocentric_lat.sin();
    let radius = b / (1.0 - (1.0 - (b * b) / (a * a)) * cos_geocentric_lat.powi(2)).sqrt();

    let earth_x = radius * cos_geocentric_lat * lon_delta.cos();
    let earth_y = -radius * cos_geocentric_lat * lon_delta.sin();
    let earth_z = radius * sin_geocentric_lat;

    let sat_x = h - earth_x;
    let sat_y = earth_y;
    let sat_z = earth_z;
    let sat_range = sat_x.hypot(sat_y).hypot(sat_z);
    if sat_range <= 0.0 || !sat_range.is_finite() {
        return None;
    }

    let (x, y) = match sweep_angle_axis {
        SweepAngleAxis::X => ((-sat_y / sat_range).asin(), (sat_z / sat_x).atan()),
        // Sweep about the y axis (the CGMS / Himawari / GK-2A / Meteosat convention): the
        // north-south angle is the outer gimbal, so y is the elevation of the look vector
        // and x its azimuth inside the plane that contains it.
        SweepAngleAxis::Y => ((-sat_y / sat_x).atan(), (sat_z / sat_range).asin()),
    };

    let (roundtrip_lat, roundtrip_lon) = scan_angles_to_lat_lon(
        perspective_point_height_m,
        semi_major_axis_m,
        semi_minor_axis_m,
        longitude_of_projection_origin_deg,
        sweep_angle_axis,
        x,
        y,
    )?;
    let lon_error = longitude_delta_deg(f64::from(roundtrip_lon), longitude_deg).abs();
    if (f64::from(roundtrip_lat) - latitude_deg).abs() > 1.0e-3 || lon_error > 1.0e-3 {
        return None;
    }

    if !x.is_finite() || !y.is_finite() {
        return None;
    }
    Some((x, y))
}

pub fn lat_lon_to_scan_angles_fast(
    perspective_point_height_m: f64,
    semi_major_axis_m: f64,
    semi_minor_axis_m: f64,
    longitude_of_projection_origin_deg: f64,
    sweep_angle_axis: SweepAngleAxis,
    latitude_deg: f64,
    longitude_deg: f64,
) -> Option<(f64, f64)> {
    if !perspective_point_height_m.is_finite()
        || !semi_major_axis_m.is_finite()
        || !semi_minor_axis_m.is_finite()
        || !longitude_of_projection_origin_deg.is_finite()
        || !latitude_deg.is_finite()
        || !longitude_deg.is_finite()
    {
        return None;
    }

    let h = perspective_point_height_m + semi_major_axis_m;
    let a = semi_major_axis_m;
    let b = semi_minor_axis_m;
    if h <= 0.0 || a <= 0.0 || b <= 0.0 {
        return None;
    }

    let lat = latitude_deg.to_radians();
    let lon_delta = (longitude_deg - longitude_of_projection_origin_deg).to_radians();
    let geocentric_lat = ((b * b) / (a * a) * lat.tan()).atan();
    let cos_geocentric_lat = geocentric_lat.cos();
    let sin_geocentric_lat = geocentric_lat.sin();
    let radius = b / (1.0 - (1.0 - (b * b) / (a * a)) * cos_geocentric_lat.powi(2)).sqrt();

    let earth_x = radius * cos_geocentric_lat * lon_delta.cos();
    let earth_y = -radius * cos_geocentric_lat * lon_delta.sin();
    let earth_z = radius * sin_geocentric_lat;

    let sat_x = h - earth_x;
    let sat_y = earth_y;
    let sat_z = earth_z;
    let sat_range = sat_x.hypot(sat_y).hypot(sat_z);
    if sat_range <= 0.0 || !sat_range.is_finite() || sat_x <= 0.0 {
        return None;
    }

    let (x, y) = match sweep_angle_axis {
        SweepAngleAxis::X => ((-sat_y / sat_range).asin(), (sat_z / sat_x).atan()),
        // Sweep about the y axis (the CGMS / Himawari / GK-2A / Meteosat convention): the
        // north-south angle is the outer gimbal, so y is the elevation of the look vector
        // and x its azimuth inside the plane that contains it.
        SweepAngleAxis::Y => ((-sat_y / sat_x).atan(), (sat_z / sat_range).asin()),
    };

    if !x.is_finite() || !y.is_finite() {
        return None;
    }
    Some((x, y))
}

/// Scan angles to latitude/longitude for an imager that sweeps about the y
/// axis (`sweep_angle_axis = "y"`: Himawari AHI, GK-2A AMI, Meteosat, the
/// CGMS LRIT/HRIT normalized geostationary projection).  `x_rad` is the
/// east-positive column angle and `y_rad` the north-positive line angle,
/// the same orientation the GOES fixed grid uses; only the gimbal order
/// differs.  The look vector is (cos x cos y, -sin x cos y, sin y), which is
/// JMA's HSD navigation (S1 = h - Sn cos x cos y, S2 = Sn sin x cos y,
/// S3 = -Sn sin y with its south-positive line angle).
///
/// Until 2026-10-16 the `Y` arm of [`scan_angles_to_lat_lon`] swapped x and
/// y and fed them to the x-sweep formula, which is a different projection
/// (tens of km wrong toward the full-disk corners); nothing in this
/// workspace used it before the AHI/AMI decoders.
fn scan_angles_to_lat_lon_sweep_y(
    h: f64,
    a: f64,
    b: f64,
    longitude_of_projection_origin_deg: f64,
    x: f64,
    y: f64,
) -> Option<(f32, f32)> {
    let (sin_x, cos_x, sin_y, cos_y) = (x.sin(), x.cos(), y.sin(), y.cos());
    let eq_to_pol = (a * a) / (b * b);
    let a_var = cos_y * cos_y + eq_to_pol * sin_y * sin_y;
    let b_var = -2.0 * h * cos_x * cos_y;
    let c_var = h * h - a * a;
    let discriminant = b_var * b_var - 4.0 * a_var * c_var;
    if discriminant < 0.0 {
        return None;
    }
    let r_s = (-b_var - discriminant.sqrt()) / (2.0 * a_var);
    if !r_s.is_finite() || r_s <= 0.0 {
        return None;
    }
    let s_x = r_s * cos_x * cos_y;
    let s_y = -r_s * sin_x * cos_y;
    let s_z = r_s * sin_y;
    let latitude = (eq_to_pol * (s_z / ((h - s_x).hypot(s_y)))).atan();
    let longitude = longitude_of_projection_origin_deg.to_radians() - (s_y / (h - s_x)).atan();
    let lat_deg = latitude.to_degrees();
    let lon_deg = normalize_longitude_deg(longitude.to_degrees());
    if !lat_deg.is_finite() || !lon_deg.is_finite() {
        return None;
    }
    Some((lat_deg as f32, lon_deg as f32))
}

fn normalize_longitude_deg(lon: f64) -> f64 {
    let mut value = (lon + 180.0).rem_euclid(360.0) - 180.0;
    if value == -180.0 {
        value = 180.0;
    }
    value
}

fn longitude_delta_deg(a: f64, b: f64) -> f64 {
    (a - b + 180.0).rem_euclid(360.0) - 180.0
}

#[cfg(test)]
mod tests {
    use super::*;

    const H: f64 = 35_786_023.0;
    const A: f64 = 6_378_137.0;
    const B: f64 = 6_356_752.314_14;
    const LON0: f64 = -137.0;

    #[test]
    fn nadir_maps_to_projection_origin() {
        let (lat, lon) = scan_angles_to_lat_lon(H, A, B, LON0, SweepAngleAxis::X, 0.0, 0.0)
            .expect("nadir should intersect earth");
        assert!(lat.abs() < 1.0e-4, "{lat}");
        assert!((f64::from(lon) - LON0).abs() < 1.0e-4, "{lon}");
    }

    #[test]
    fn far_limb_returns_none() {
        let point = scan_angles_to_lat_lon(H, A, B, LON0, SweepAngleAxis::X, 1.0, 1.0);
        assert!(point.is_none());
    }

    #[test]
    fn lat_lon_to_scan_angles_roundtrips_visible_points() {
        for (lat, lon) in [(0.0, LON0), (35.0, -120.0), (42.0, -100.0)] {
            let (x, y) = lat_lon_to_scan_angles(H, A, B, LON0, SweepAngleAxis::X, lat, lon)
                .expect("visible point should project to scan angles");
            let (actual_lat, actual_lon) =
                scan_angles_to_lat_lon(H, A, B, LON0, SweepAngleAxis::X, x, y)
                    .expect("scan angles should hit earth");
            assert!((f64::from(actual_lat) - lat).abs() < 1.0e-3, "{actual_lat}");
            assert!(
                longitude_delta_deg(f64::from(actual_lon), lon).abs() < 1.0e-3,
                "{actual_lon}"
            );
        }
    }

    #[test]
    fn lat_lon_to_scan_angles_rejects_far_side_points() {
        let point = lat_lon_to_scan_angles(H, A, B, LON0, SweepAngleAxis::X, 0.0, 40.0);
        assert!(point.is_none());
    }

    #[test]
    fn fast_lat_lon_to_scan_angles_matches_validated_path() {
        for (lat, lon) in [(0.0, LON0), (35.0, -120.0), (42.0, -100.0)] {
            let expected = lat_lon_to_scan_angles(H, A, B, LON0, SweepAngleAxis::X, lat, lon)
                .expect("visible point should project to scan angles");
            let actual = lat_lon_to_scan_angles_fast(H, A, B, LON0, SweepAngleAxis::X, lat, lon)
                .expect("visible point should project to scan angles");
            assert!((actual.0 - expected.0).abs() < 1.0e-12);
            assert!((actual.1 - expected.1).abs() < 1.0e-12);
        }
    }

    /// JMA's HSD navigation written out as the user guide states it
    /// (kilometres, south-positive line angle), independent of the code above.
    fn jma_hsd_navigation(x_deg: f64, y_line_deg: f64, sub_lon: f64) -> (f64, f64) {
        let (x, y) = (x_deg.to_radians(), y_line_deg.to_radians());
        let denom = y.cos().powi(2) + 1.006739501 * y.sin().powi(2);
        let sd = ((42164.0 * x.cos() * y.cos()).powi(2) - denom * 1737122264.0).sqrt();
        let sn = (42164.0 * x.cos() * y.cos() - sd) / denom;
        let s1 = 42164.0 - sn * x.cos() * y.cos();
        let s2 = sn * x.sin() * y.cos();
        let s3 = -sn * y.sin();
        let sxy = s1.hypot(s2);
        ((1.006739501 * s3 / sxy).atan().to_degrees(), (s2 / s1).atan().to_degrees() + sub_lon)
    }

    #[test]
    fn sweep_y_matches_the_jma_hsd_navigation() {
        // Himawari constants: 42164 km from the Earth centre, 6378.137 / 6356.7523 km radii.
        let (h, a, b, lon0) = (42_164_000.0 - 6_378_137.0, 6_378_137.0, 6_356_752.3, 140.7);
        for (xd, yd) in [(0.0, 0.0), (4.0, 3.0), (-5.5, 5.0), (5.5, -5.5), (-3.0, -7.0), (7.5, 1.0)] {
            let (lat_ref, lon_ref) = jma_hsd_navigation(xd, -yd, lon0);
            let (lat, lon) = scan_angles_to_lat_lon(
                h,
                a,
                b,
                lon0,
                SweepAngleAxis::Y,
                f64::to_radians(xd),
                f64::to_radians(yd),
            )
            .expect("on the disk");
            assert!((f64::from(lat) - lat_ref).abs() < 2.0e-4, "{xd},{yd}: {lat} vs {lat_ref}");
            assert!(longitude_delta_deg(f64::from(lon), lon_ref).abs() < 2.0e-4, "{xd},{yd}: {lon} vs {lon_ref}");
        }
    }

    #[test]
    fn sweep_y_round_trips_and_differs_from_sweep_x_off_axis() {
        let lon0 = 128.2;
        for (lat, lon) in [(0.0, 128.2), (37.5, 127.0), (-30.0, 150.0), (55.0, 100.0)] {
            let (x, y) = lat_lon_to_scan_angles(H, A, B, lon0, SweepAngleAxis::Y, lat, lon)
                .expect("visible");
            let (la, lo) = scan_angles_to_lat_lon(H, A, B, lon0, SweepAngleAxis::Y, x, y).unwrap();
            assert!((f64::from(la) - lat).abs() < 1.0e-3);
            assert!(longitude_delta_deg(f64::from(lo), lon).abs() < 1.0e-3);
            let fast = lat_lon_to_scan_angles_fast(H, A, B, lon0, SweepAngleAxis::Y, lat, lon).unwrap();
            assert!((fast.0 - x).abs() < 1.0e-12 && (fast.1 - y).abs() < 1.0e-12);
        }
        // On either axis the two gimbal orders see the same point ...
        let on_x = scan_angles_to_lat_lon(H, A, B, lon0, SweepAngleAxis::X, 0.1, 0.0).unwrap();
        let on_y = scan_angles_to_lat_lon(H, A, B, lon0, SweepAngleAxis::Y, 0.1, 0.0).unwrap();
        assert!((on_x.0 - on_y.0).abs() < 1.0e-5 && (on_x.1 - on_y.1).abs() < 1.0e-5);
        let on_x = scan_angles_to_lat_lon(H, A, B, lon0, SweepAngleAxis::X, 0.0, 0.1).unwrap();
        let on_y = scan_angles_to_lat_lon(H, A, B, lon0, SweepAngleAxis::Y, 0.0, 0.1).unwrap();
        assert!((on_x.0 - on_y.0).abs() < 1.0e-5 && (on_x.1 - on_y.1).abs() < 1.0e-5);
        // ... and toward a corner they do not.
        let c_x = scan_angles_to_lat_lon(H, A, B, lon0, SweepAngleAxis::X, 0.1, 0.1).unwrap();
        let c_y = scan_angles_to_lat_lon(H, A, B, lon0, SweepAngleAxis::Y, 0.1, 0.1).unwrap();
        assert!((c_x.0 - c_y.0).abs() + (c_x.1 - c_y.1).abs() > 0.05, "{c_x:?} {c_y:?}");
    }
}
