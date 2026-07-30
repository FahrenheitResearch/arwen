use super::*;

/// Assert two f64 slices are bit-identical, NaN-aware.
fn assert_bits_eq(label: &str, left: &[f64], right: &[f64]) {
    assert_eq!(left.len(), right.len(), "{label}: length mismatch");
    for (i, (l, r)) in left.iter().zip(right.iter()).enumerate() {
        assert_eq!(
            l.to_bits(),
            r.to_bits(),
            "{label}[{i}]: left={l} right={r} differ in bit representation"
        );
    }
}

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1.0e-9,
        "expected {expected}, got {actual}"
    );
}

fn assert_vec_close(left: &[f64], right: &[f64]) {
    assert_eq!(left.len(), right.len(), "vector lengths differed");
    for (lhs, rhs) in left.iter().zip(right.iter()) {
        assert_close(*lhs, *rhs);
    }
}

#[test]
fn fixed_stp_matches_operational_lcl_and_shear_gates() {
    assert_close(fixed_stp_value(1500.0, 500.0, 150.0, 20.0), 1.0);
    assert_close(fixed_stp_value(1500.0, 1000.0, 150.0, 12.0), 0.0);
    assert_close(fixed_stp_value(1500.0, 1000.0, 150.0, 40.0), 1.5);
}

#[test]
fn effective_stp_applies_cin_and_ebwd_limits() {
    assert_close(effective_stp_value(1500.0, -50.0, 1000.0, 150.0, 10.0), 0.0);
    assert_close(effective_stp_value(1500.0, -50.0, 1000.0, 150.0, 20.0), 1.0);
    assert_close(
        effective_stp_value(1500.0, -250.0, 1000.0, 150.0, 20.0),
        0.0,
    );
    assert_close(effective_stp_value(1500.0, -50.0, 1000.0, 150.0, 40.0), 1.5);
}

#[test]
fn effective_scp_uses_ebwd_thresholds() {
    assert_close(scp_effective_value(3000.0, 150.0, 8.0), 0.0);
    assert_close(scp_effective_value(3000.0, 150.0, 20.0), 9.0);
    assert_close(scp_effective_value(3000.0, 150.0, 30.0), 9.0);
}

#[test]
fn tornadic_beta_matches_wrf_rust_formula_cases() {
    let outputs = compute_tornadic_beta(TornadicBetaInputs {
        grid: GridShape::new(3, 1).unwrap(),
        srh_1km_m2s2: &[200.0, 160.0, 200.0],
        mlcape_jkg: &[1000.0, 1600.0, 1000.0],
        mlcape_03km_jkg: &[100.0, 50.0, 100.0],
        shear_6km_ms: &[20.0, 20.0, 10.0],
        ml_lcl_m: &[1000.0, 1000.0, 1000.0],
        mlcin_jkg: &[-50.0, -50.0, -50.0],
        sbcin_jkg: &[-50.0, -50.0, -50.0],
    })
    .unwrap();

    assert_vec_close(&outputs.tehi, &[0.625, 1.6, 0.0]);

    let outputs = compute_tornadic_beta(TornadicBetaInputs {
        grid: GridShape::new(3, 1).unwrap(),
        srh_1km_m2s2: &[100.0, 100.0, -100.0],
        mlcape_jkg: &[2500.0, 1500.0, 2500.0],
        mlcape_03km_jkg: &[100.0, 200.0, 100.0],
        shear_6km_ms: &[20.0, 35.0, 20.0],
        ml_lcl_m: &[1000.0, 1000.0, 1000.0],
        mlcin_jkg: &[-50.0, -50.0, -50.0],
        sbcin_jkg: &[-50.0, -50.0, -50.0],
    })
    .unwrap();

    assert_vec_close(&outputs.tts, &[1.9230769230769231, 3.4615384615384617, 0.0]);
}

#[test]
fn vtp_mod_matches_wrf_rust_formula_cases() {
    let vtp = compute_vtp_mod(VtpModInputs {
        grid: GridShape::new(5, 1).unwrap(),
        mlcape_jkg: &[1700.0, 1700.0, 1700.0, 1700.0, 1700.0],
        effective_srh_m2s2: &[250.0, 250.0, 250.0, 250.0, 250.0],
        effective_bulk_wind_difference_ms: &[30.0, 20.0, 45.0, 30.0, 30.0],
        ml_lcl_m: &[1000.0, 1000.0, 1000.0, 1750.0, 1000.0],
        mlcin_jkg: &[-100.0, -50.0, -50.0, -50.0, -200.0],
        mlcape_03km_jkg: &[50.0, 50.0, 50.0, 50.0, 50.0],
        lapse_rate_700_500_cpkm: &[6.5, 6.5, 6.5, 6.5, 6.5],
    })
    .unwrap();

    assert_vec_close(&vtp, &[2.0 / 3.0, 0.0, 1.5, 0.0, 0.0]);
}

#[test]
fn effective_severe_bundle_matches_component_formulas() {
    let outputs = compute_effective_severe(EffectiveSevereInputs {
        grid: GridShape::new(4, 1).unwrap(),
        mlcape_jkg: &[1500.0, 1500.0, 1500.0, 1500.0],
        mlcin_jkg: &[-50.0, -50.0, -250.0, -50.0],
        ml_lcl_m: &[1000.0, 1000.0, 1000.0, 1000.0],
        mucape_jkg: &[3000.0, 3000.0, 3000.0, 3000.0],
        effective_srh_m2s2: &[150.0, 150.0, 150.0, 150.0],
        effective_bulk_wind_difference_ms: &[8.0, 20.0, 20.0, 40.0],
    })
    .unwrap();

    assert_eq!(outputs.stp_effective, vec![0.0, 1.0, 0.0, 1.5]);
    assert_eq!(outputs.scp_effective, vec![0.0, 9.0, 9.0, 9.0]);
}

#[test]
fn scp_ehi_bundle_matches_component_formulas() {
    let outputs = compute_scp_ehi(ScpEhiInputs {
        grid: GridShape::new(3, 1).unwrap(),
        scp_cape_jkg: &[3000.0, 3000.0, 3000.0],
        scp_srh_m2s2: &[150.0, 150.0, 150.0],
        scp_bulk_wind_difference_ms: &[8.0, 20.0, 30.0],
        ehi_cape_jkg: &[2000.0, 1600.0, 800.0],
        ehi_srh_m2s2: &[200.0, 100.0, 50.0],
    })
    .unwrap();

    assert_eq!(outputs.scp, vec![0.0, 9.0, 9.0]);
    assert_eq!(outputs.ehi, vec![2.5, 1.0, 0.25]);
}

#[test]
fn ship_matches_local_proxy_formula_and_low_cape_scaling() {
    assert_close(ship_value(2000.0, 20.0, -15.0, 7.0, 10.0), 1.0);
    assert_close(
        ship_value(1000.0, 20.0, -15.0, 7.0, 10.0),
        0.38461538461538464,
    );
    assert_close(ship_value(2000.0, 20.0, 5.0, 7.0, 10.0), 0.0);
}

#[test]
fn bri_uses_brn_shear_and_zeroes_degenerate_denominator() {
    assert_close(bri_value(2000.0, 20.0), 10.0);
    assert_close(bri_value(500.0, 30.0), 1.1111111111111112);
    assert_close(bri_value(1000.0, 0.1), 0.0);
}

#[test]
fn supported_severe_fields_reuse_fixed_and_proxy_component_math() {
    let grid = GridShape::new(1, 1).unwrap();
    let volume = EcapeVolumeInputs {
        pressure_pa: &[95_000.0, 90_000.0, 85_000.0, 70_000.0, 50_000.0, 30_000.0],
        temperature_c: &[26.0, 22.0, 18.0, 8.0, -10.0, -38.0],
        qvapor_kgkg: &[0.016, 0.013, 0.010, 0.005, 0.0015, 0.0003],
        height_agl_m: &[150.0, 800.0, 1500.0, 3000.0, 5600.0, 9200.0],
        u_ms: &[6.0, 9.0, 12.0, 18.0, 26.0, 33.0],
        v_ms: &[2.0, 5.0, 8.0, 13.0, 20.0, 28.0],
        nz: 6,
    };
    let surface = SurfaceInputs {
        psfc_pa: &[100_000.0],
        t2_k: &[303.15],
        q2_kgkg: &[0.018],
        u10_ms: &[5.0],
        v10_ms: &[1.5],
    };

    let supported = compute_supported_severe_fields(grid, volume, surface).unwrap();
    let sb = compute_cape_cin(grid, volume, surface, "sb", None).unwrap();
    let ml = compute_cape_cin(grid, volume, surface, "ml", None).unwrap();
    let ml_03km = compute_cape_cin(grid, volume, surface, "ml", Some(3000.0)).unwrap();
    let mu = compute_cape_cin(grid, volume, surface, "mu", None).unwrap();
    let wind = WindGridInputs {
        shape: VolumeShape::new(grid, volume.nz).unwrap(),
        u_3d_ms: volume.u_ms,
        v_3d_ms: volume.v_ms,
        height_agl_3d_m: volume.height_agl_m,
    };
    let srh_01km = compute_srh(wind, 1000.0).unwrap();
    let srh_03km = compute_srh(wind, 3000.0).unwrap();
    let shear_06km = compute_shear(wind, 0.0, 6000.0).unwrap();
    let stp_fixed = compute_stp_fixed(FixedStpInputs {
        grid,
        sbcape_jkg: &sb.cape_jkg,
        lcl_m: &sb.lcl_m,
        srh_1km_m2s2: &srh_01km,
        shear_6km_ms: &shear_06km,
    })
    .unwrap();
    let proxy = compute_scp_ehi(ScpEhiInputs {
        grid,
        scp_cape_jkg: &mu.cape_jkg,
        scp_srh_m2s2: &srh_03km,
        scp_bulk_wind_difference_ms: &shear_06km,
        ehi_cape_jkg: &sb.cape_jkg,
        ehi_srh_m2s2: &srh_01km,
    })
    .unwrap();
    let beta = compute_tornadic_beta(TornadicBetaInputs {
        grid,
        srh_1km_m2s2: &srh_01km,
        mlcape_jkg: &ml.cape_jkg,
        mlcape_03km_jkg: &ml_03km.cape_jkg,
        shear_6km_ms: &shear_06km,
        ml_lcl_m: &ml.lcl_m,
        mlcin_jkg: &ml.cin_jkg,
        sbcin_jkg: &sb.cin_jkg,
    })
    .unwrap();
    let effective_layer = compute_effective_layer_diagnostics(grid, volume, surface, None).unwrap();
    let lapse_rate_700_500 = lapse_rate_700_500_for_supported(grid, volume).unwrap();
    let vtp_mod = compute_vtp_mod(VtpModInputs {
        grid,
        mlcape_jkg: &ml.cape_jkg,
        effective_srh_m2s2: &effective_layer.effective_srh_m2s2,
        effective_bulk_wind_difference_ms: &effective_layer.effective_bulk_wind_difference_ms,
        ml_lcl_m: &ml.lcl_m,
        mlcin_jkg: &ml.cin_jkg,
        mlcape_03km_jkg: &ml_03km.cape_jkg,
        lapse_rate_700_500_cpkm: &lapse_rate_700_500,
    })
    .unwrap();

    assert_eq!(supported.sbcape_jkg, sb.cape_jkg);
    assert_eq!(supported.mlcin_jkg, ml.cin_jkg);
    assert_eq!(supported.mucape_jkg, mu.cape_jkg);
    assert_eq!(supported.srh_01km_m2s2, srh_01km);
    assert_eq!(supported.srh_03km_m2s2, srh_03km);
    assert_eq!(supported.shear_06km_ms, shear_06km);
    assert_eq!(supported.stp_fixed, stp_fixed);
    assert_eq!(supported.scp_mu_03km_06km_proxy, proxy.scp);
    assert_eq!(supported.ehi_sb_01km_proxy, proxy.ehi);
    assert_eq!(supported.tehi, beta.tehi);
    assert_eq!(supported.tts, beta.tts);
    assert_eq!(supported.vtp_mod, vtp_mod);
}

#[test]
fn cape_cin_levels_match_broadcast_pressure_path() {
    let grid = GridShape::new(2, 1).unwrap();
    let pressure_levels_pa = [95_000.0, 85_000.0, 70_000.0, 50_000.0];
    let volume_levels = EcapeVolumeInputs {
        pressure_pa: &pressure_levels_pa,
        temperature_c: &[26.0, 24.0, 18.0, 16.0, 8.0, 6.0, -10.0, -12.0],
        qvapor_kgkg: &[0.016, 0.015, 0.010, 0.009, 0.004, 0.0038, 0.0012, 0.0011],
        height_agl_m: &[150.0, 200.0, 1400.0, 1500.0, 3000.0, 3200.0, 5600.0, 5800.0],
        u_ms: &[6.0, 8.0, 12.0, 14.0, 20.0, 22.0, 28.0, 30.0],
        v_ms: &[2.0, 3.0, 8.0, 9.0, 13.0, 14.0, 20.0, 21.0],
        nz: 4,
    };
    let pressure_broadcast = [
        95_000.0, 95_000.0, 85_000.0, 85_000.0, 70_000.0, 70_000.0, 50_000.0, 50_000.0,
    ];
    let volume_broadcast = EcapeVolumeInputs {
        pressure_pa: &pressure_broadcast,
        ..volume_levels
    };
    let surface = SurfaceInputs {
        psfc_pa: &[100_000.0, 99_500.0],
        t2_k: &[303.15, 301.15],
        q2_kgkg: &[0.018, 0.017],
        u10_ms: &[5.0, 6.0],
        v10_ms: &[1.5, 2.0],
    };

    let levels = compute_cape_cin(grid, volume_levels, surface, "sb", None).unwrap();
    let broadcast = compute_cape_cin(grid, volume_broadcast, surface, "sb", None).unwrap();

    assert_vec_close(&levels.cape_jkg, &broadcast.cape_jkg);
    assert_vec_close(&levels.cin_jkg, &broadcast.cin_jkg);
    assert_vec_close(&levels.lcl_m, &broadcast.lcl_m);
    assert_vec_close(&levels.lfc_m, &broadcast.lfc_m);
}

#[test]
fn supported_severe_fields_levels_match_broadcast_pressure_path() {
    let grid = GridShape::new(2, 1).unwrap();
    let pressure_levels_pa = [95_000.0, 85_000.0, 70_000.0, 50_000.0];
    let volume_levels = EcapeVolumeInputs {
        pressure_pa: &pressure_levels_pa,
        temperature_c: &[26.0, 24.0, 18.0, 16.0, 8.0, 6.0, -10.0, -12.0],
        qvapor_kgkg: &[0.016, 0.015, 0.010, 0.009, 0.004, 0.0038, 0.0012, 0.0011],
        height_agl_m: &[150.0, 200.0, 1400.0, 1500.0, 3000.0, 3200.0, 5600.0, 5800.0],
        u_ms: &[6.0, 8.0, 12.0, 14.0, 20.0, 22.0, 28.0, 30.0],
        v_ms: &[2.0, 3.0, 8.0, 9.0, 13.0, 14.0, 20.0, 21.0],
        nz: 4,
    };
    let pressure_broadcast = [
        95_000.0, 95_000.0, 85_000.0, 85_000.0, 70_000.0, 70_000.0, 50_000.0, 50_000.0,
    ];
    let volume_broadcast = EcapeVolumeInputs {
        pressure_pa: &pressure_broadcast,
        ..volume_levels
    };
    let surface = SurfaceInputs {
        psfc_pa: &[100_000.0, 99_500.0],
        t2_k: &[303.15, 301.15],
        q2_kgkg: &[0.018, 0.017],
        u10_ms: &[5.0, 6.0],
        v10_ms: &[1.5, 2.0],
    };

    let levels = compute_supported_severe_fields(grid, volume_levels, surface).unwrap();
    let broadcast = compute_supported_severe_fields(grid, volume_broadcast, surface).unwrap();

    assert_vec_close(&levels.sbcape_jkg, &broadcast.sbcape_jkg);
    assert_vec_close(&levels.mlcin_jkg, &broadcast.mlcin_jkg);
    assert_vec_close(&levels.mucape_jkg, &broadcast.mucape_jkg);
    assert_vec_close(&levels.srh_01km_m2s2, &broadcast.srh_01km_m2s2);
    assert_vec_close(&levels.srh_03km_m2s2, &broadcast.srh_03km_m2s2);
    assert_vec_close(&levels.shear_06km_ms, &broadcast.shear_06km_ms);
    assert_vec_close(&levels.stp_fixed, &broadcast.stp_fixed);
    assert_vec_close(
        &levels.scp_mu_03km_06km_proxy,
        &broadcast.scp_mu_03km_06km_proxy,
    );
    assert_vec_close(&levels.ehi_sb_01km_proxy, &broadcast.ehi_sb_01km_proxy);
    assert_vec_close(&levels.tehi, &broadcast.tehi);
    assert_vec_close(&levels.tts, &broadcast.tts);
    assert_vec_close(&levels.vtp_mod, &broadcast.vtp_mod);
}

#[test]
fn fused_wind_diagnostics_bundle_matches_individual_wrappers() {
    let grid = GridShape::new(1, 1).unwrap();
    let wind = WindGridInputs {
        shape: VolumeShape::new(grid, 3).unwrap(),
        u_3d_ms: &[0.0, 10.0, 20.0],
        v_3d_ms: &[0.0, 0.0, 0.0],
        height_agl_3d_m: &[0.0, 3000.0, 6000.0],
    };

    let fused = compute_wind_diagnostics_bundle(wind).unwrap();
    assert_eq!(fused.srh_01km_m2s2, compute_srh(wind, 1000.0).unwrap());
    assert_eq!(fused.srh_03km_m2s2, compute_srh(wind, 3000.0).unwrap());
    assert_eq!(
        fused.shear_06km_ms,
        compute_shear(wind, 0.0, 6000.0).unwrap()
    );
}

/// Verify that `compute_cape_cin_triplet` is bit-identical to three separate
/// `compute_cape_cin` calls for all 12 output planes (sb/ml/mu × cape/cin/lcl/lfc),
/// exercised on a 7×5 grid with 9 pressure levels in both descending and
/// ascending level orderings.  Also asserts that sb/ml/mu differ somewhere,
/// guarding against a degenerate all-equal test.
#[test]
fn cape_cin_triplet_bit_identical_to_single_parcel_paths() {
    // Grid: 7 columns × 5 rows = 35 columns.
    let grid = GridShape::new(7, 5).unwrap();
    let n2d = grid.len(); // 35
    let nz: usize = 9;

    // Pressure levels (Pa) in descending order (surface-to-top).
    // Chosen so that for some columns psfc > level[0] (below-ground levels).
    let pressure_levels_desc_pa: [f64; 9] = [
        97_500.0, 92_500.0, 87_500.0, 82_500.0, 75_000.0, 65_000.0, 50_000.0, 40_000.0, 30_000.0,
    ];

    // psfc spread ~955–1005 hPa so that some columns have below-ground levels.
    // psfc[ij] cycles across 35 columns spanning 955–1005 hPa.
    let psfc_pa: Vec<f64> = (0..n2d)
        .map(|ij| 95_500.0 + (ij as f64) * (50_000.0 / (n2d as f64 - 1.0)))
        .collect();

    // 2-metre fields: mild variation across columns.
    let t2_k: Vec<f64> = (0..n2d).map(|ij| 295.0 + (ij as f64) * 0.3).collect();
    let q2_kgkg: Vec<f64> = (0..n2d).map(|ij| 0.010 + (ij as f64) * 0.0003).collect();
    let u10_ms: Vec<f64> = vec![5.0; n2d];
    let v10_ms: Vec<f64> = vec![2.0; n2d];

    // 3-D fields, level-major layout: index = k * n2d + ij.
    // Temperature decreases with altitude; moisture falls off; height grows.
    let mut temperature_c = vec![0.0f64; nz * n2d];
    let mut qvapor_kgkg = vec![0.0f64; nz * n2d];
    let mut height_agl_m = vec![0.0f64; nz * n2d];
    let mut u_ms = vec![0.0f64; nz * n2d];
    let mut v_ms = vec![0.0f64; nz * n2d];

    // Approximate heights for the pressure levels using hypsometric spacing.
    let approx_heights_m: [f64; 9] = [
        250.0, 750.0, 1300.0, 1900.0, 2800.0, 4000.0, 5600.0, 7200.0, 9500.0,
    ];

    for k in 0..nz {
        for ij in 0..n2d {
            let idx = k * n2d + ij;
            let col_frac = ij as f64 / (n2d as f64 - 1.0);
            // Surface temperature varies 18–28 °C; lapse ~6 °C/km.
            let t_sfc = 18.0 + col_frac * 10.0;
            let lapse = 6.0e-3 + col_frac * 1.5e-3; // 6–7.5 °C/km
            temperature_c[idx] = t_sfc - lapse * approx_heights_m[k];
            // Moisture: surface mixing ratio 8–16 g/kg, halves at mid-levels.
            let q_sfc = 0.008 + col_frac * 0.008;
            qvapor_kgkg[idx] = q_sfc * (-approx_heights_m[k] / 4500.0).exp();
            height_agl_m[idx] = approx_heights_m[k];
            u_ms[idx] = 5.0 + (k as f64) * 2.5 + col_frac * 3.0;
            v_ms[idx] = 1.0 + (k as f64) * 1.5;
        }
    }

    let surface = SurfaceInputs {
        psfc_pa: &psfc_pa,
        t2_k: &t2_k,
        q2_kgkg: &q2_kgkg,
        u10_ms: &u10_ms,
        v10_ms: &v10_ms,
    };

    // --- Helper: run triplet + three individual calls and compare all 12 planes ---
    let check_ordering = |label: &str, pressure_pa: &[f64]| {
        let volume = EcapeVolumeInputs {
            pressure_pa,
            temperature_c: &temperature_c,
            qvapor_kgkg: &qvapor_kgkg,
            height_agl_m: &height_agl_m,
            u_ms: &u_ms,
            v_ms: &v_ms,
            nz,
        };

        let triplet = compute_cape_cin_triplet(grid, volume, surface, None).unwrap();
        let sb = compute_cape_cin(grid, volume, surface, "sb", None).unwrap();
        let ml = compute_cape_cin(grid, volume, surface, "ml", None).unwrap();
        let mu = compute_cape_cin(grid, volume, surface, "mu", None).unwrap();

        // SB planes
        assert_bits_eq(
            &format!("{label}/sb.cape"),
            &triplet.sb.cape_jkg,
            &sb.cape_jkg,
        );
        assert_bits_eq(&format!("{label}/sb.cin"), &triplet.sb.cin_jkg, &sb.cin_jkg);
        assert_bits_eq(&format!("{label}/sb.lcl"), &triplet.sb.lcl_m, &sb.lcl_m);
        assert_bits_eq(&format!("{label}/sb.lfc"), &triplet.sb.lfc_m, &sb.lfc_m);
        // ML planes
        assert_bits_eq(
            &format!("{label}/ml.cape"),
            &triplet.ml.cape_jkg,
            &ml.cape_jkg,
        );
        assert_bits_eq(&format!("{label}/ml.cin"), &triplet.ml.cin_jkg, &ml.cin_jkg);
        assert_bits_eq(&format!("{label}/ml.lcl"), &triplet.ml.lcl_m, &ml.lcl_m);
        assert_bits_eq(&format!("{label}/ml.lfc"), &triplet.ml.lfc_m, &ml.lfc_m);
        // MU planes
        assert_bits_eq(
            &format!("{label}/mu.cape"),
            &triplet.mu.cape_jkg,
            &mu.cape_jkg,
        );
        assert_bits_eq(&format!("{label}/mu.cin"), &triplet.mu.cin_jkg, &mu.cin_jkg);
        assert_bits_eq(&format!("{label}/mu.lcl"), &triplet.mu.lcl_m, &mu.lcl_m);
        assert_bits_eq(&format!("{label}/mu.lfc"), &triplet.mu.lfc_m, &mu.lfc_m);

        // Return triplet for the non-degeneracy check
        triplet
    };

    // Descending pressure (surface-first): the normal NWP layout.
    let triplet_desc = check_ordering("descending", &pressure_levels_desc_pa);

    // Ascending pressure (top-first): the reversed layout the code must handle.
    let mut pressure_levels_asc_pa = pressure_levels_desc_pa;
    pressure_levels_asc_pa.reverse();
    check_ordering("ascending", &pressure_levels_asc_pa);

    // Non-degeneracy: sb, ml, and mu CAPE must differ somewhere across the grid,
    // proving the test exercises distinct parcel computations.
    let sb_cape = &triplet_desc.sb.cape_jkg;
    let ml_cape = &triplet_desc.ml.cape_jkg;
    let mu_cape = &triplet_desc.mu.cape_jkg;

    let sb_differs_from_ml = sb_cape.iter().zip(ml_cape.iter()).any(|(s, m)| s != m);
    let sb_differs_from_mu = sb_cape.iter().zip(mu_cape.iter()).any(|(s, m)| s != m);
    assert!(
        sb_differs_from_ml,
        "sb and ml CAPE are identical everywhere — test may be degenerate"
    );
    assert!(
        sb_differs_from_mu,
        "sb and mu CAPE are identical everywhere — test may be degenerate"
    );
}
