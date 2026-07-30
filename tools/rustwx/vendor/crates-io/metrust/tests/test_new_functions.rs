//! Integration tests for the newly added functions.

// ── angle_to_direction_ext ──

#[test]
fn test_ext_8_point() {
    assert_eq!(metrust::calc::angle_to_direction_ext(0.0, 8, false), "N");
    assert_eq!(metrust::calc::angle_to_direction_ext(45.0, 8, false), "NE");
    assert_eq!(metrust::calc::angle_to_direction_ext(90.0, 8, false), "E");
    assert_eq!(metrust::calc::angle_to_direction_ext(180.0, 8, false), "S");
    assert_eq!(metrust::calc::angle_to_direction_ext(270.0, 8, false), "W");
}

#[test]
fn test_ext_32_point() {
    assert_eq!(metrust::calc::angle_to_direction_ext(0.0, 32, false), "N");
    assert_eq!(metrust::calc::angle_to_direction_ext(11.25, 32, false), "NbE");
    assert_eq!(metrust::calc::angle_to_direction_ext(22.5, 32, false), "NNE");
    assert_eq!(metrust::calc::angle_to_direction_ext(90.0, 32, false), "E");
}

#[test]
fn test_ext_full_names() {
    assert_eq!(metrust::calc::angle_to_direction_ext(0.0, 8, true), "North");
    assert_eq!(metrust::calc::angle_to_direction_ext(90.0, 8, true), "East");
    assert_eq!(metrust::calc::angle_to_direction_ext(180.0, 8, true), "South");
    assert_eq!(metrust::calc::angle_to_direction_ext(225.0, 8, true), "Southwest");
}

#[test]
fn test_ext_16_full() {
    assert_eq!(metrust::calc::angle_to_direction_ext(0.0, 16, true), "North");
    assert_eq!(metrust::calc::angle_to_direction_ext(22.5, 16, true), "North-Northeast");
    assert_eq!(metrust::calc::angle_to_direction_ext(90.0, 16, true), "East");
}

#[test]
fn test_ext_32_full() {
    assert_eq!(metrust::calc::angle_to_direction_ext(0.0, 32, true), "North");
    assert_eq!(metrust::calc::angle_to_direction_ext(11.25, 32, true), "North by East");
    assert_eq!(metrust::calc::angle_to_direction_ext(90.0, 32, true), "East");
}

// ── find_peaks ──

#[test]
fn test_find_peaks_basic() {
    let data = vec![0.0, 5.0, 1.0, 3.0, 0.0, 10.0, 0.0];
    let peaks = metrust::calc::find_peaks(&data, true, 0.0);
    assert!(peaks.contains(&1));
    assert!(peaks.contains(&3));
    assert!(peaks.contains(&5));
}

#[test]
fn test_find_peaks_iqr_filter() {
    let data = vec![0.0, 1.0, 0.0, 1.0, 0.0, 10.0, 0.0];
    let peaks = metrust::calc::find_peaks(&data, true, 2.0);
    assert!(peaks.contains(&5));
    assert!(!peaks.contains(&1));
}

#[test]
fn test_find_peaks_troughs() {
    let data = vec![5.0, 1.0, 5.0, 3.0, 5.0, 0.0, 5.0];
    let peaks = metrust::calc::find_peaks(&data, false, 0.0);
    assert!(peaks.contains(&1));
    assert!(peaks.contains(&5));
}

#[test]
fn test_find_peaks_empty() {
    assert!(metrust::calc::find_peaks(&[], true, 0.0).is_empty());
    assert!(metrust::calc::find_peaks(&[1.0, 2.0], true, 0.0).is_empty());
}

// ── peak_persistence ──

#[test]
fn test_peak_persistence_single() {
    let data = vec![0.0, 5.0, 0.0];
    let peaks = metrust::calc::peak_persistence(&data, true);
    assert!(!peaks.is_empty());
    assert_eq!(peaks[0].0, 1);
    assert!((peaks[0].1 - 5.0).abs() < 1e-10);
}

#[test]
fn test_peak_persistence_two() {
    let data = vec![0.0, 10.0, 0.0, 5.0, 0.0];
    let peaks = metrust::calc::peak_persistence(&data, true);
    assert!(peaks.len() >= 2);
    assert_eq!(peaks[0].0, 1);
    assert!(peaks[0].1 > peaks[1].1);
}

#[test]
fn test_peak_persistence_minima() {
    let data = vec![10.0, 0.0, 10.0, 5.0, 10.0];
    let peaks = metrust::calc::peak_persistence(&data, false);
    assert!(!peaks.is_empty());
    assert_eq!(peaks[0].0, 1);
}

// ── azimuth_range_to_lat_lon ──

#[test]
fn test_az_range_origin() {
    let (lats, lons) = metrust::calc::azimuth_range_to_lat_lon(
        &[0.0], &[0.0], 35.0, -97.0,
    );
    assert!((lats[0] - 35.0).abs() < 1e-10);
    assert!((lons[0] + 97.0).abs() < 1e-10);
}

#[test]
fn test_az_range_north() {
    let (lats, lons) = metrust::calc::azimuth_range_to_lat_lon(
        &[0.0], &[111_194.0], 0.0, 0.0,
    );
    assert!((lats[0] - 1.0).abs() < 0.01);
    assert!(lons[0].abs() < 0.01);
}

#[test]
fn test_az_range_east() {
    let (lats, lons) = metrust::calc::azimuth_range_to_lat_lon(
        &[90.0], &[111_194.0], 0.0, 0.0,
    );
    assert!(lats[0].abs() < 0.01);
    assert!((lons[0] - 1.0).abs() < 0.01);
}

#[test]
fn test_az_range_shape() {
    let (lats, lons) = metrust::calc::azimuth_range_to_lat_lon(
        &[0.0, 90.0, 180.0], &[1000.0, 2000.0, 3000.0, 4000.0], 35.0, -97.0,
    );
    assert_eq!(lats.len(), 12);
    assert_eq!(lons.len(), 12);
}

// ── advection_3d ──

#[test]
fn test_advection_3d_uniform() {
    let nx = 4;
    let ny = 3;
    let nz = 3;
    let n = nx * ny * nz;
    let scalar = vec![10.0; n];
    let u = vec![5.0; n];
    let v = vec![3.0; n];
    let w = vec![1.0; n];
    let result = metrust::calc::advection_3d(&scalar, &u, &v, &w, nx, ny, nz, 1000.0, 1000.0, 500.0);
    for val in &result {
        assert!(val.abs() < 1e-10);
    }
}

#[test]
fn test_advection_3d_vertical_only() {
    let nx = 3;
    let ny = 3;
    let nz = 3;
    let nxy = nx * ny;
    let mut scalar = vec![0.0; nxy * nz];
    for k in 0..nz {
        for ij in 0..nxy {
            scalar[k * nxy + ij] = k as f64 * 10.0;
        }
    }
    let u = vec![0.0; nxy * nz];
    let v = vec![0.0; nxy * nz];
    let w = vec![1.0; nxy * nz];
    let dz = 100.0;
    let result = metrust::calc::advection_3d(&scalar, &u, &v, &w, nx, ny, nz, 1000.0, 1000.0, dz);
    let k = 1;
    for ij in 0..nxy {
        let val = result[k * nxy + ij];
        assert!((val + 0.1).abs() < 1e-10, "Expected -0.1, got {}", val);
    }
}

// ── specific_humidity_from_mixing_ratio ──

#[test]
fn test_specific_humidity_from_mixing_ratio() {
    // MetPy: specific_humidity_from_mixing_ratio(0.012 kg/kg) = 0.0118577075
    let q = metrust::calc::specific_humidity_from_mixing_ratio(0.012);
    assert!((q - 0.0118577075).abs() < 1e-8);
}

#[test]
fn test_specific_humidity_from_mixing_ratio_zero() {
    assert!((metrust::calc::specific_humidity_from_mixing_ratio(0.0)).abs() < 1e-15);
}

#[test]
fn test_specific_humidity_from_mixing_ratio_identity() {
    let w = 0.02;
    let q = metrust::calc::specific_humidity_from_mixing_ratio(w);
    assert!(q < w);
    assert!((q - w / (1.0 + w)).abs() < 1e-15);
}

// ── thickness_hydrostatic_from_relative_humidity ──

#[test]
fn test_thickness_hydrostatic_from_rh() {
    // MetPy: 5614.4389 m
    let p  = vec![1000.0, 900.0, 800.0, 700.0, 600.0, 500.0];
    let t  = vec![  25.0,  18.0,  10.0,   2.0,  -8.0, -18.0];
    let rh = vec![  80.0,  70.0,  60.0,  50.0,  40.0,  30.0];
    let dz = metrust::calc::thickness_hydrostatic_from_relative_humidity(&p, &t, &rh);
    assert!((dz - 5614.4).abs() < 5.0, "thickness = {dz}");
}

#[test]
fn test_thickness_hydrostatic_from_rh_moisture_increases_thickness() {
    let p = vec![1000.0, 500.0];
    let t = vec![25.0, -10.0];
    let dz_dry  = metrust::calc::thickness_hydrostatic_from_relative_humidity(&p, &t, &[20.0, 20.0]);
    let dz_moist = metrust::calc::thickness_hydrostatic_from_relative_humidity(&p, &t, &[90.0, 90.0]);
    assert!(dz_moist > dz_dry);
}

// ── friction_velocity ──

#[test]
fn test_friction_velocity_metpy() {
    // MetPy: friction_velocity([1,-1,1,-1,1], [0.5,-0.5,0.5,-0.5,0.5]) = 0.6928203230
    let u = vec![1.0, -1.0, 1.0, -1.0, 1.0];
    let w = vec![0.5, -0.5, 0.5, -0.5, 0.5];
    let ustar = metrust::calc::friction_velocity(&u, &w);
    assert!((ustar - 0.6928203230).abs() < 1e-8, "u* = {ustar}");
}

#[test]
fn test_friction_velocity_uncorrelated() {
    let u = vec![1.0, -1.0, 1.0, -1.0];
    let w = vec![1.0, 1.0, -1.0, -1.0];
    let ustar = metrust::calc::friction_velocity(&u, &w);
    assert!(ustar.abs() < 1e-10);
}

// ── tke ──

#[test]
fn test_tke_metpy() {
    // MetPy: tke([1,-1,1,-1], [2,-2,2,-2], [0.5,-0.5,0.5,-0.5]) = 2.625
    let u = vec![1.0, -1.0, 1.0, -1.0];
    let v = vec![2.0, -2.0, 2.0, -2.0];
    let w = vec![0.5, -0.5, 0.5, -0.5];
    let e = metrust::calc::tke(&u, &v, &w);
    assert!((e - 2.625).abs() < 1e-10, "TKE = {e}");
}

#[test]
fn test_tke_zero() {
    let u = vec![5.0, 5.0, 5.0];
    let v = vec![3.0, 3.0, 3.0];
    let w = vec![0.0, 0.0, 0.0];
    assert!(metrust::calc::tke(&u, &v, &w).abs() < 1e-10);
}

// ── gradient_richardson_number ──

#[test]
fn test_gradient_ri_metpy() {
    // MetPy reference values
    let z = vec![0.0, 100.0, 200.0, 300.0, 400.0];
    let theta = vec![300.0, 301.0, 302.5, 304.5, 307.0];
    let u = vec![2.0, 5.0, 8.0, 10.0, 12.0];
    let v = vec![1.0, 2.0, 3.5, 5.0, 6.0];
    let ri = metrust::calc::gradient_richardson_number(&z, &theta, &u, &v);
    let expected = [0.25638301, 0.38556488, 0.66744336, 1.30270438, 1.92536076];
    for i in 0..5 {
        assert!((ri[i] - expected[i]).abs() < 1e-4, "Ri[{i}] = {}, expected {}", ri[i], expected[i]);
    }
}

#[test]
fn test_gradient_ri_strongly_stable() {
    let z = vec![0.0, 100.0, 200.0];
    let theta = vec![300.0, 310.0, 320.0];
    let u = vec![5.0, 5.1, 5.2];
    let v = vec![0.0, 0.0, 0.0];
    let ri = metrust::calc::gradient_richardson_number(&z, &theta, &u, &v);
    for i in 0..3 {
        assert!(ri[i] > 10.0, "Ri[{i}] = {}", ri[i]);
    }
}

// ── interpolate_to_points ──

#[test]
fn test_interpolate_to_points_idw() {
    let src_lats = vec![30.0, 32.0];
    let src_lons = vec![-90.0, -90.0];
    let src_vals = vec![10.0, 20.0];
    let tgt_lats = vec![31.0];
    let tgt_lons = vec![-90.0];
    let result = metrust::interpolate::interpolate_to_points(
        &src_lats, &src_lons, &src_vals, &tgt_lats, &tgt_lons, "idw",
    );
    assert_eq!(result.len(), 1);
    assert!((result[0] - 15.0).abs() < 1e-10);
}

#[test]
fn test_interpolate_to_points_nn() {
    let src_lats = vec![30.0, 32.0];
    let src_lons = vec![-90.0, -90.0];
    let src_vals = vec![10.0, 20.0];
    let tgt_lats = vec![31.0];
    let tgt_lons = vec![-90.0];
    let result = metrust::interpolate::interpolate_to_points(
        &src_lats, &src_lons, &src_vals, &tgt_lats, &tgt_lons, "natural_neighbor",
    );
    assert_eq!(result.len(), 1);
    assert!((result[0] - 15.0).abs() < 1e-10);
}
