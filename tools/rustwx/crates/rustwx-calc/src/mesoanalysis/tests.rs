use super::*;

fn sample_background() -> (
    GridShape,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
) {
    let grid = GridShape::new(3, 1).unwrap();
    (
        grid,
        vec![35.0, 35.0, 35.0],
        vec![-98.2, -98.0, -97.8],
        vec![100000.0; 3],
        vec![293.15; 3],
        vec![0.010; 3],
        vec![0.0; 3],
        vec![0.0; 3],
    )
}

#[test]
fn mesoanalysis_applies_temperature_increment_near_observation() {
    let (grid, lat, lon, psfc, t2, q2, u10, v10) = sample_background();
    let obs = vec![
        MesoObservation::new("TEST", 35.0, -98.0)
            .with_source("unit")
            .with_temperature_c(25.0),
    ];
    let fields = compute_surface_mesoanalysis(
        SurfaceMesoBackground {
            grid,
            lat_deg: &lat,
            lon_deg: &lon,
            psfc_pa: &psfc,
            t2_k: &t2,
            q2_kgkg: &q2,
            u10_ms: &u10,
            v10_ms: &v10,
        },
        &obs,
        MesoanalysisConfig {
            barnes_radius_km: 35.0,
            barnes_kappa_km2: 100.0,
            ..MesoanalysisConfig::default()
        },
    )
    .unwrap();

    assert!(fields.temperature_2m_c[1] > 24.9);
    assert!(fields.temperature_increment_c[1].is_finite());
    assert_eq!(fields.diagnostics[0].accepted_observations, 1);
    assert!(fields.diagnostics[0].covered_grid_cells >= 1);
}

#[test]
fn optimal_interpolation_uses_source_error_weights_for_gain() {
    let (grid, lat, lon, psfc, t2, q2, u10, v10) = sample_background();
    let low_quality = vec![
        MesoObservation::new("LOW", 35.0, -98.0)
            .with_temperature_c(25.0)
            .with_quality_weight(0.2),
    ];
    let high_quality = vec![
        MesoObservation::new("HIGH", 35.0, -98.0)
            .with_temperature_c(25.0)
            .with_quality_weight(5.0),
    ];
    let config = MesoanalysisConfig {
        method: MesoanalysisMethod::OptimalInterpolation,
        barnes_radius_km: 30.0,
        oi_length_scale_km: 30.0,
        oi_background_error_temperature_c: 3.0,
        oi_observation_error_temperature_c: 1.0,
        oi_flow_anisotropy_ratio: 1.0,
        ..MesoanalysisConfig::default()
    };

    let low = compute_surface_mesoanalysis(
        SurfaceMesoBackground {
            grid,
            lat_deg: &lat,
            lon_deg: &lon,
            psfc_pa: &psfc,
            t2_k: &t2,
            q2_kgkg: &q2,
            u10_ms: &u10,
            v10_ms: &v10,
        },
        &low_quality,
        config,
    )
    .unwrap();
    let high = compute_surface_mesoanalysis(
        SurfaceMesoBackground {
            grid,
            lat_deg: &lat,
            lon_deg: &lon,
            psfc_pa: &psfc,
            t2_k: &t2,
            q2_kgkg: &q2,
            u10_ms: &u10,
            v10_ms: &v10,
        },
        &high_quality,
        config,
    )
    .unwrap();

    assert!(high.temperature_increment_c[1] > low.temperature_increment_c[1]);
    assert!(high.temperature_confidence[1] > low.temperature_confidence[1]);
    assert!(high.temperature_confidence[1] <= 1.0);
    assert!(high.diagnostics[0].mean_confidence.unwrap() > 0.0);
}

#[test]
fn optimal_interpolation_uses_variable_specific_observation_errors() {
    let (grid, lat, lon, psfc, t2, q2, u10, v10) = sample_background();
    let noisy = vec![
        MesoObservation::new("NOISY", 35.0, -98.0)
            .with_temperature_c(25.0)
            .with_temperature_error_c(5.0),
    ];
    let precise = vec![
        MesoObservation::new("PRECISE", 35.0, -98.0)
            .with_temperature_c(25.0)
            .with_temperature_error_c(0.25),
    ];
    let config = MesoanalysisConfig {
        method: MesoanalysisMethod::OptimalInterpolation,
        barnes_radius_km: 30.0,
        oi_length_scale_km: 30.0,
        oi_background_error_temperature_c: 3.0,
        oi_observation_error_temperature_c: 1.0,
        oi_flow_anisotropy_ratio: 1.0,
        ..MesoanalysisConfig::default()
    };

    let noisy = compute_surface_mesoanalysis(
        SurfaceMesoBackground {
            grid,
            lat_deg: &lat,
            lon_deg: &lon,
            psfc_pa: &psfc,
            t2_k: &t2,
            q2_kgkg: &q2,
            u10_ms: &u10,
            v10_ms: &v10,
        },
        &noisy,
        config,
    )
    .unwrap();
    let precise = compute_surface_mesoanalysis(
        SurfaceMesoBackground {
            grid,
            lat_deg: &lat,
            lon_deg: &lon,
            psfc_pa: &psfc,
            t2_k: &t2,
            q2_kgkg: &q2,
            u10_ms: &u10,
            v10_ms: &v10,
        },
        &precise,
        config,
    )
    .unwrap();

    assert!(precise.temperature_increment_c[1] > noisy.temperature_increment_c[1]);
    assert!(precise.temperature_confidence[1] > noisy.temperature_confidence[1]);
}

#[test]
fn optimal_interpolation_full_matrix_honors_colocated_obs_against_conflict() {
    let (grid, lat, lon, psfc, t2, q2, u10, v10) = sample_background();
    let obs = vec![
        MesoObservation::new("TARGET", 35.0, -98.0).with_temperature_c(25.0),
        MesoObservation::new("NEARBY_CONFLICT", 35.0, -97.8).with_temperature_c(15.0),
    ];
    let fields = compute_surface_mesoanalysis(
        SurfaceMesoBackground {
            grid,
            lat_deg: &lat,
            lon_deg: &lon,
            psfc_pa: &psfc,
            t2_k: &t2,
            q2_kgkg: &q2,
            u10_ms: &u10,
            v10_ms: &v10,
        },
        &obs,
        MesoanalysisConfig {
            method: MesoanalysisMethod::OptimalInterpolation,
            barnes_radius_km: 40.0,
            oi_length_scale_km: 30.0,
            oi_background_error_temperature_c: 3.0,
            oi_observation_error_temperature_c: 0.5,
            oi_flow_anisotropy_ratio: 1.0,
            oi_max_observations_per_grid_cell: 2,
            ..MesoanalysisConfig::default()
        },
    )
    .unwrap();

    assert!(fields.temperature_increment_c[1] > 3.0);
    assert!(fields.temperature_2m_c[1] > 23.0);
    assert_eq!(fields.neighbor_count[1], 2);
    assert!(fields.temperature_confidence[1] > 0.8);
}

#[test]
fn optimal_interpolation_can_follow_background_flow() {
    let grid = GridShape::new(3, 1).unwrap();
    let lat = vec![35.0, 35.0, 35.16];
    let lon = vec![-98.0, -97.805, -98.0];
    let psfc = vec![100000.0; 3];
    let t2 = vec![293.15; 3];
    let q2 = vec![0.010; 3];
    let u10 = vec![12.0; 3];
    let v10 = vec![0.0; 3];
    let obs = vec![MesoObservation::new("CENTER", 35.0, -98.0).with_temperature_c(25.0)];
    let fields = compute_surface_mesoanalysis(
        SurfaceMesoBackground {
            grid,
            lat_deg: &lat,
            lon_deg: &lon,
            psfc_pa: &psfc,
            t2_k: &t2,
            q2_kgkg: &q2,
            u10_ms: &u10,
            v10_ms: &v10,
        },
        &obs,
        MesoanalysisConfig {
            method: MesoanalysisMethod::OptimalInterpolation,
            barnes_radius_km: 40.0,
            oi_length_scale_km: 15.0,
            oi_flow_anisotropy_ratio: 4.0,
            ..MesoanalysisConfig::default()
        },
    )
    .unwrap();

    assert!(fields.temperature_increment_c[1] > fields.temperature_increment_c[2]);
    assert!(fields.temperature_confidence[1] > fields.temperature_confidence[2]);
}

#[test]
fn optimal_interpolation_damps_across_terrain_pressure_jumps() {
    let grid = GridShape::new(3, 1).unwrap();
    let lat = vec![35.0, 35.0, 35.1];
    let lon = vec![-98.0, -97.9, -98.0];
    let psfc = vec![100000.0, 100000.0, 85000.0];
    let t2 = vec![293.15; 3];
    let q2 = vec![0.010; 3];
    let u10 = vec![0.0; 3];
    let v10 = vec![0.0; 3];
    let obs = vec![MesoObservation::new("LOWLAND", 35.0, -98.0).with_temperature_c(25.0)];
    let fields = compute_surface_mesoanalysis(
        SurfaceMesoBackground {
            grid,
            lat_deg: &lat,
            lon_deg: &lon,
            psfc_pa: &psfc,
            t2_k: &t2,
            q2_kgkg: &q2,
            u10_ms: &u10,
            v10_ms: &v10,
        },
        &obs,
        MesoanalysisConfig {
            method: MesoanalysisMethod::OptimalInterpolation,
            barnes_radius_km: 40.0,
            oi_length_scale_km: 30.0,
            oi_flow_anisotropy_ratio: 1.0,
            oi_terrain_pressure_scale_hpa: 20.0,
            ..MesoanalysisConfig::default()
        },
    )
    .unwrap();

    assert!(fields.temperature_increment_c[1].is_finite());
    assert!(
        !fields.temperature_increment_c[2].is_finite()
            || fields.temperature_increment_c[1] > fields.temperature_increment_c[2] + 1.0
    );
    assert!(fields.temperature_confidence[1] > fields.temperature_confidence[2]);
}

#[test]
fn optimal_interpolation_rejects_normalized_gross_errors() {
    let (grid, lat, lon, psfc, t2, q2, u10, v10) = sample_background();
    let obs = vec![MesoObservation::new("SPIKE", 35.0, -98.0).with_temperature_c(45.0)];
    let fields = compute_surface_mesoanalysis(
        SurfaceMesoBackground {
            grid,
            lat_deg: &lat,
            lon_deg: &lon,
            psfc_pa: &psfc,
            t2_k: &t2,
            q2_kgkg: &q2,
            u10_ms: &u10,
            v10_ms: &v10,
        },
        &obs,
        MesoanalysisConfig {
            method: MesoanalysisMethod::OptimalInterpolation,
            max_temperature_increment_c: 50.0,
            oi_background_error_temperature_c: 1.0,
            oi_observation_error_temperature_c: 0.5,
            oi_gross_error_sigma: 3.0,
            ..MesoanalysisConfig::default()
        },
    )
    .unwrap();

    assert_eq!(fields.diagnostics[0].candidate_observations, 1);
    assert_eq!(fields.diagnostics[0].accepted_observations, 0);
    assert_eq!(fields.diagnostics[0].rejected_observations, 1);
    assert_eq!(fields.diagnostics[0].gross_error_rescued_observations, 0);
    assert_eq!(fields.temperature_2m_c, vec![20.0, 20.0, 20.0]);
}

#[test]
fn optimal_interpolation_rescues_buddy_supported_gross_errors() {
    let (grid, lat, lon, psfc, t2, q2, u10, v10) = sample_background();
    let obs = vec![
        MesoObservation::new("COLD_POOL_A", 35.0, -98.0).with_temperature_c(10.0),
        MesoObservation::new("COLD_POOL_B", 35.0, -98.05).with_temperature_c(10.5),
    ];
    let fields = compute_surface_mesoanalysis(
        SurfaceMesoBackground {
            grid,
            lat_deg: &lat,
            lon_deg: &lon,
            psfc_pa: &psfc,
            t2_k: &t2,
            q2_kgkg: &q2,
            u10_ms: &u10,
            v10_ms: &v10,
        },
        &obs,
        MesoanalysisConfig {
            method: MesoanalysisMethod::OptimalInterpolation,
            max_temperature_increment_c: 50.0,
            oi_background_error_temperature_c: 1.0,
            oi_observation_error_temperature_c: 0.5,
            oi_gross_error_sigma: 3.0,
            oi_gross_error_buddy_radius_km: 10.0,
            oi_gross_error_buddy_min_neighbors: 1,
            oi_gross_error_buddy_agreement_sigma: 1.0,
            oi_flow_anisotropy_ratio: 1.0,
            ..MesoanalysisConfig::default()
        },
    )
    .unwrap();

    assert_eq!(fields.diagnostics[0].candidate_observations, 2);
    assert_eq!(fields.diagnostics[0].accepted_observations, 2);
    assert_eq!(fields.diagnostics[0].gross_error_rescued_observations, 2);
    assert!(fields.temperature_2m_c[1] < 15.0);
    assert!(fields.temperature_confidence[1] > 0.0);
}

#[test]
fn optimal_interpolation_caps_local_matrix_and_reports_truncation() {
    let (grid, lat, lon, psfc, t2, q2, u10, v10) = sample_background();
    let obs = vec![
        MesoObservation::new("A", 35.0, -98.0).with_temperature_c(24.0),
        MesoObservation::new("B", 35.0, -98.05).with_temperature_c(25.0),
        MesoObservation::new("C", 35.0, -97.95).with_temperature_c(23.0),
    ];
    let fields = compute_surface_mesoanalysis(
        SurfaceMesoBackground {
            grid,
            lat_deg: &lat,
            lon_deg: &lon,
            psfc_pa: &psfc,
            t2_k: &t2,
            q2_kgkg: &q2,
            u10_ms: &u10,
            v10_ms: &v10,
        },
        &obs,
        MesoanalysisConfig {
            method: MesoanalysisMethod::OptimalInterpolation,
            barnes_radius_km: 40.0,
            oi_max_observations_per_grid_cell: 2,
            ..MesoanalysisConfig::default()
        },
    )
    .unwrap();

    assert!(fields.diagnostics[0].truncated_neighbor_grid_cells > 0);
    assert_eq!(fields.diagnostics[0].max_neighbor_count, 2);
    assert_eq!(fields.diagnostics[0].solver_failed_grid_cells, 0);
}

#[test]
fn optimal_interpolation_dense_network_stays_bounded_and_stable() {
    let grid = GridShape::new(10, 10).unwrap();
    let mut lat = Vec::new();
    let mut lon = Vec::new();
    for y in 0..10 {
        for x in 0..10 {
            lat.push(35.0 + y as f64 * 0.02);
            lon.push(-98.0 + x as f64 * 0.02);
        }
    }
    let psfc = vec![100000.0; 100];
    let t2 = vec![293.15; 100];
    let q2 = vec![0.010; 100];
    let u10 = vec![8.0; 100];
    let v10 = vec![2.0; 100];
    let mut obs = Vec::new();
    for y in 0..8 {
        for x in 0..8 {
            obs.push(
                MesoObservation::new(
                    format!("DENSE_{y}_{x}"),
                    35.01 + y as f64 * 0.022,
                    -97.99 + x as f64 * 0.022,
                )
                .with_temperature_c(21.0 + ((x + y) % 3) as f64),
            );
        }
    }

    let fields = compute_surface_mesoanalysis(
        SurfaceMesoBackground {
            grid,
            lat_deg: &lat,
            lon_deg: &lon,
            psfc_pa: &psfc,
            t2_k: &t2,
            q2_kgkg: &q2,
            u10_ms: &u10,
            v10_ms: &v10,
        },
        &obs,
        MesoanalysisConfig {
            method: MesoanalysisMethod::OptimalInterpolation,
            barnes_radius_km: 30.0,
            oi_max_observations_per_grid_cell: 8,
            ..MesoanalysisConfig::default()
        },
    )
    .unwrap();

    assert!(fields.diagnostics[0].covered_grid_cells > 90);
    assert_eq!(fields.diagnostics[0].max_neighbor_count, 8);
    assert!(fields.diagnostics[0].truncated_neighbor_grid_cells > 0);
    assert_eq!(fields.diagnostics[0].solver_failed_grid_cells, 0);
}

#[test]
fn covariance_kernel_changes_spatial_influence() {
    assert!(
        MesoanalysisCovarianceKernel::Gaussian.correlation(1.0)
            > MesoanalysisCovarianceKernel::Matern32.correlation(1.0)
    );
    assert!(
        MesoanalysisCovarianceKernel::Matern32.correlation(1.0)
            > MesoanalysisCovarianceKernel::Exponential.correlation(1.0)
    );
}

#[test]
fn spatial_bins_scan_without_allocating_candidate_lists_and_wraps_longitude() {
    let observations = vec![
        IncrementObservation {
            lat_deg: 0.0,
            lon_deg: 179.9,
            increment: 1.0,
            weight: 1.0,
            observation_error: 1.0,
            background_index: 0,
        },
        IncrementObservation {
            lat_deg: 0.0,
            lon_deg: -179.9,
            increment: 2.0,
            weight: 1.0,
            observation_error: 1.0,
            background_index: 1,
        },
    ];
    let bins = SpatialBins::new_observations(&observations, 1.0);
    let mut candidates = Vec::new();
    bins.for_each_candidate(0.0, 179.95, 30.0, |index| candidates.push(index));
    candidates.sort_unstable();

    assert_eq!(candidates, vec![0, 1]);
}

#[test]
fn cholesky_solver_solves_small_symmetric_positive_definite_system() {
    let mut factor = vec![4.0, 2.0, 2.0, 3.0];
    assert!(cholesky_decompose_lower(&mut factor, 2));
    let mut forward = Vec::new();
    let mut solution = Vec::new();
    assert!(cholesky_solve_into(
        &factor,
        2,
        &[6.0, 5.0],
        &mut forward,
        &mut solution
    ));

    assert!((solution[0] - 1.0).abs() < 1.0e-9);
    assert!((solution[1] - 1.0).abs() < 1.0e-9);
}

#[test]
fn mesoanalysis_second_barnes_pass_reduces_station_residual() {
    let (grid, lat, lon, psfc, t2, q2, u10, v10) = sample_background();
    let obs = vec![
        MesoObservation::new("WARM", 35.0, -98.2)
            .with_source("unit")
            .with_temperature_c(25.0),
        MesoObservation::new("COOL", 35.0, -97.8)
            .with_source("unit")
            .with_temperature_c(15.0),
    ];
    let base_config = MesoanalysisConfig {
        barnes_radius_km: 50.0,
        barnes_kappa_km2: 5000.0,
        barnes_second_pass_gamma: 0.3,
        ..MesoanalysisConfig::default()
    };
    let pass_one = compute_surface_mesoanalysis(
        SurfaceMesoBackground {
            grid,
            lat_deg: &lat,
            lon_deg: &lon,
            psfc_pa: &psfc,
            t2_k: &t2,
            q2_kgkg: &q2,
            u10_ms: &u10,
            v10_ms: &v10,
        },
        &obs,
        MesoanalysisConfig {
            barnes_passes: 1,
            ..base_config
        },
    )
    .unwrap();
    let pass_two = compute_surface_mesoanalysis(
        SurfaceMesoBackground {
            grid,
            lat_deg: &lat,
            lon_deg: &lon,
            psfc_pa: &psfc,
            t2_k: &t2,
            q2_kgkg: &q2,
            u10_ms: &u10,
            v10_ms: &v10,
        },
        &obs,
        MesoanalysisConfig {
            barnes_passes: 2,
            ..base_config
        },
    )
    .unwrap();

    let pass_one_warm_residual = (pass_one.temperature_2m_c[0] - 25.0).abs();
    let pass_two_warm_residual = (pass_two.temperature_2m_c[0] - 25.0).abs();
    let pass_one_cool_residual = (pass_one.temperature_2m_c[2] - 15.0).abs();
    let pass_two_cool_residual = (pass_two.temperature_2m_c[2] - 15.0).abs();

    assert!(pass_two_warm_residual < pass_one_warm_residual);
    assert!(pass_two_cool_residual < pass_one_cool_residual);
    assert!(
        pass_two
            .temperature_increment_c
            .iter()
            .filter(|value| value.is_finite())
            .all(|value| value.abs() <= base_config.max_temperature_increment_c + 1.0e-9)
    );
}

#[test]
fn mesoanalysis_converts_wind_direction_to_uv_increments() {
    let (grid, lat, lon, psfc, t2, q2, u10, v10) = sample_background();
    let obs = vec![MesoObservation::new("NORTH", 35.0, -98.0).with_wind(270.0, 10.0)];
    let fields = compute_surface_mesoanalysis(
        SurfaceMesoBackground {
            grid,
            lat_deg: &lat,
            lon_deg: &lon,
            psfc_pa: &psfc,
            t2_k: &t2,
            q2_kgkg: &q2,
            u10_ms: &u10,
            v10_ms: &v10,
        },
        &obs,
        MesoanalysisConfig {
            barnes_radius_km: 30.0,
            barnes_kappa_km2: 100.0,
            ..MesoanalysisConfig::default()
        },
    )
    .unwrap();

    assert!(fields.u10_ms[1] > 9.9);
    assert!(fields.v10_ms[1].abs() < 0.1);
    assert_eq!(fields.diagnostics[2].accepted_observations, 1);
}

#[test]
fn mesoanalysis_rejects_implausible_observation_increments() {
    let (grid, lat, lon, psfc, t2, q2, u10, v10) = sample_background();
    let obs = vec![MesoObservation::new("HOT", 35.0, -98.0).with_temperature_c(60.0)];
    let fields = compute_surface_mesoanalysis(
        SurfaceMesoBackground {
            grid,
            lat_deg: &lat,
            lon_deg: &lon,
            psfc_pa: &psfc,
            t2_k: &t2,
            q2_kgkg: &q2,
            u10_ms: &u10,
            v10_ms: &v10,
        },
        &obs,
        MesoanalysisConfig::default(),
    )
    .unwrap();

    assert_eq!(fields.diagnostics[0].candidate_observations, 1);
    assert_eq!(fields.diagnostics[0].accepted_observations, 0);
    assert_eq!(fields.temperature_2m_c, vec![20.0, 20.0, 20.0]);
}

#[test]
fn mesoanalysis_dewpoint_updates_specific_humidity() {
    let (grid, lat, lon, psfc, t2, q2, u10, v10) = sample_background();
    let obs = vec![MesoObservation::new("MOIST", 35.0, -98.0).with_dewpoint_c(18.0)];
    let fields = compute_surface_mesoanalysis(
        SurfaceMesoBackground {
            grid,
            lat_deg: &lat,
            lon_deg: &lon,
            psfc_pa: &psfc,
            t2_k: &t2,
            q2_kgkg: &q2,
            u10_ms: &u10,
            v10_ms: &v10,
        },
        &obs,
        MesoanalysisConfig {
            barnes_radius_km: 30.0,
            barnes_kappa_km2: 100.0,
            ..MesoanalysisConfig::default()
        },
    )
    .unwrap();

    assert!(fields.dewpoint_2m_c[1] > 17.9);
    assert!(fields.q2_kgkg[1] > q2[1]);
}

#[test]
fn mesoanalysis_applies_mean_sea_level_pressure_increment() {
    let (grid, lat, lon, psfc, t2, q2, u10, v10) = sample_background();
    let mean_sea_level_pressure_hpa = vec![1010.0; 3];
    let obs =
        vec![MesoObservation::new("PRESS", 35.0, -98.0).with_mean_sea_level_pressure_hpa(1014.0)];
    let fields = SurfaceMesoBackground {
        grid,
        lat_deg: &lat,
        lon_deg: &lon,
        psfc_pa: &psfc,
        t2_k: &t2,
        q2_kgkg: &q2,
        u10_ms: &u10,
        v10_ms: &v10,
    }
    .compute_with_mean_sea_level_pressure_hpa(
        &mean_sea_level_pressure_hpa,
        &obs,
        MesoanalysisConfig {
            barnes_radius_km: 30.0,
            barnes_kappa_km2: 100.0,
            ..MesoanalysisConfig::default()
        },
    )
    .unwrap();

    let analyzed_pressure = fields.mean_sea_level_pressure_hpa.as_ref().unwrap();
    let pressure_increment = fields
        .mean_sea_level_pressure_increment_hpa
        .as_ref()
        .unwrap();
    let diagnostics = fields
        .diagnostics
        .iter()
        .find(|diag| diag.variable == "mean_sea_level_pressure_hpa")
        .unwrap();

    assert!(analyzed_pressure[1] > 1013.9);
    assert!(pressure_increment[1].is_finite());
    assert_eq!(diagnostics.accepted_observations, 1);
    assert!(diagnostics.covered_grid_cells >= 1);
}

#[test]
fn mesoanalysis_leaves_missing_mean_sea_level_pressure_optional() {
    let (grid, lat, lon, psfc, t2, q2, u10, v10) = sample_background();
    let mean_sea_level_pressure_hpa = vec![1010.0, 1011.0, 1012.0];
    let obs = vec![MesoObservation::new("TEMP", 35.0, -98.0).with_temperature_c(22.0)];

    let surface_only = compute_surface_mesoanalysis(
        SurfaceMesoBackground {
            grid,
            lat_deg: &lat,
            lon_deg: &lon,
            psfc_pa: &psfc,
            t2_k: &t2,
            q2_kgkg: &q2,
            u10_ms: &u10,
            v10_ms: &v10,
        },
        &obs,
        MesoanalysisConfig::default(),
    )
    .unwrap();
    let pressure_aware = SurfaceMesoBackground {
        grid,
        lat_deg: &lat,
        lon_deg: &lon,
        psfc_pa: &psfc,
        t2_k: &t2,
        q2_kgkg: &q2,
        u10_ms: &u10,
        v10_ms: &v10,
    }
    .compute_with_mean_sea_level_pressure_hpa(
        &mean_sea_level_pressure_hpa,
        &obs,
        MesoanalysisConfig::default(),
    )
    .unwrap();

    let analyzed_pressure = pressure_aware.mean_sea_level_pressure_hpa.as_ref().unwrap();
    let pressure_increment = pressure_aware
        .mean_sea_level_pressure_increment_hpa
        .as_ref()
        .unwrap();
    let diagnostics = pressure_aware
        .diagnostics
        .iter()
        .find(|diag| diag.variable == "mean_sea_level_pressure_hpa")
        .unwrap();

    assert!(surface_only.mean_sea_level_pressure_hpa.is_none());
    assert_eq!(analyzed_pressure, &mean_sea_level_pressure_hpa);
    assert!(pressure_increment.iter().all(|value| value.is_nan()));
    assert_eq!(diagnostics.candidate_observations, 0);
    assert_eq!(diagnostics.accepted_observations, 0);
}
