use super::*;
use rustwx_core::{GridShape, LatLonGrid};

fn tiny_grid() -> LatLonGrid {
    LatLonGrid::new(
        GridShape::new(2, 1).unwrap(),
        vec![40.0, 40.0],
        vec![-100.0, -99.0],
    )
    .unwrap()
}

#[test]
fn compute_qpf_prefers_direct_window_when_available() {
    let mut apcp = BTreeMap::new();
    apcp.insert(
        6,
        Ok(HrrrApcpDecode {
            windows: vec![
                WindowedFieldRecord {
                    hours: 1,
                    values: vec![0.5, 0.25],
                },
                WindowedFieldRecord {
                    hours: 6,
                    values: vec![12.7, 25.4],
                },
            ],
        }),
    );
    let computed = compute_qpf_product(HrrrWindowedProduct::Qpf6h, 6, &tiny_grid(), &apcp).unwrap();
    assert_eq!(computed.metadata.strategy, "direct APCP 6h accumulation");
    assert_eq!(computed.metadata.contributing_forecast_hours, vec![6]);
    assert_eq!(computed.field.values, vec![0.5_f32, 1.0_f32]);
}

#[test]
fn qpf_fallback_hours_only_when_direct_window_is_missing() {
    let mut apcp = BTreeMap::new();
    apcp.insert(
        12,
        Ok(HrrrApcpDecode {
            windows: vec![WindowedFieldRecord {
                hours: 12,
                values: vec![12.7, 25.4],
            }],
        }),
    );
    assert_eq!(
        qpf_fallback_hours_if_direct_missing(HrrrWindowedProduct::Qpf12h, 12, &apcp),
        None
    );
    assert_eq!(
        qpf_fallback_hours_if_direct_missing(HrrrWindowedProduct::Qpf6h, 12, &apcp),
        Some(vec![7, 8, 9, 10, 11, 12])
    );
    assert_eq!(
        qpf_fallback_hours_if_direct_missing(HrrrWindowedProduct::QpfTotal, 12, &apcp),
        None
    );
}

#[test]
fn compute_qpf_window_blocks_when_a_contributing_hour_is_missing() {
    // Partial-success regression: if the planner couldn't fetch one
    // hour inside a windowed QPF product's window, the compute
    // kernel must emit a blocker for *that* product - not abort the
    // whole windowed lane. The windowed lane's loader inserts
    // Err(reason) for missing hours; compute_qpf_product surfaces
    // the reason through the normal per-product blocker path.
    let mut apcp = BTreeMap::new();
    // Hour 1 and 3 loaded fine; hour 2 failed upstream.
    for hour in [1u16, 3u16] {
        apcp.insert(
            hour,
            Ok(HrrrApcpDecode {
                windows: vec![WindowedFieldRecord {
                    hours: 1,
                    values: vec![25.4, 12.7],
                }],
            }),
        );
    }
    apcp.insert(2, Err("hour 2 fetch failed: 404 Not Found".to_string()));

    // Qpf24h hitting forecast_hour 3 would want hours 1..=3 - the
    // missing hour 2 has to blocker this product. Use QpfTotal
    // (covers 1..=forecast_hour) which is more representative.
    let err = compute_qpf_product(HrrrWindowedProduct::QpfTotal, 3, &tiny_grid(), &apcp)
        .expect_err("compute must surface the missing-hour failure as a blocker");
    assert!(
        err.contains("hour 2") || err.contains("404"),
        "blocker should reference the missing hour or its upstream reason; got: {err}"
    );

    // Meanwhile a 1-hour QPF at forecast_hour 3 needs only hour 3 -
    // the missing hour 2 doesn't block it, and the product still
    // renders.
    let ok = compute_qpf_product(HrrrWindowedProduct::Qpf1h, 3, &tiny_grid(), &apcp)
        .expect("Qpf1h at f003 should render despite an unrelated missing hour");
    assert_eq!(ok.metadata.contributing_forecast_hours, vec![3]);
}

#[test]
fn compute_qpf_total_falls_back_to_hourly_sum() {
    let mut apcp = BTreeMap::new();
    for hour in 1..=3 {
        apcp.insert(
            hour,
            Ok(HrrrApcpDecode {
                windows: vec![WindowedFieldRecord {
                    hours: 1,
                    values: vec![25.4, 12.7],
                }],
            }),
        );
    }
    let computed =
        compute_qpf_product(HrrrWindowedProduct::QpfTotal, 3, &tiny_grid(), &apcp).unwrap();
    assert_eq!(
        computed.metadata.strategy,
        "sum of all available hourly APCP increments"
    );
    assert_eq!(computed.field.values, vec![3.0_f32, 1.5_f32]);
}

#[test]
fn compute_uh_run_max_takes_pointwise_maximum() {
    let mut uh = BTreeMap::new();
    uh.insert(
        1,
        Ok(HrrrUhDecode {
            windows: vec![WindowedFieldRecord {
                hours: 1,
                values: vec![50.0, 10.0],
            }],
        }),
    );
    uh.insert(
        2,
        Ok(HrrrUhDecode {
            windows: vec![WindowedFieldRecord {
                hours: 1,
                values: vec![25.0, 40.0],
            }],
        }),
    );
    let computed =
        compute_uh_product(HrrrWindowedProduct::Uh25kmRunMax, 2, &tiny_grid(), &uh).unwrap();
    assert_eq!(computed.field.values, vec![50.0_f32, 40.0_f32]);
    assert_eq!(
        computed.metadata.strategy,
        "run max of native hourly UH maxima"
    );
}

#[test]
fn compute_wind10m_run_max_takes_pointwise_maximum_and_converts_to_knots() {
    let mut wind = BTreeMap::new();
    wind.insert(
        1,
        Ok(HrrrWind10mMaxDecode {
            windows: vec![WindowedFieldRecord {
                hours: 1,
                values: vec![10.0, 5.0],
            }],
        }),
    );
    wind.insert(
        2,
        Ok(HrrrWind10mMaxDecode {
            windows: vec![WindowedFieldRecord {
                hours: 1,
                values: vec![8.0, 12.0],
            }],
        }),
    );
    let computed =
        compute_wind10m_product(HrrrWindowedProduct::Wind10mRunMax, 2, &tiny_grid(), &wind)
            .unwrap();
    assert_eq!(
        computed.field.values,
        vec![(10.0 * MS_TO_KT) as f32, (12.0 * MS_TO_KT) as f32]
    );
    assert_eq!(
        computed.metadata.strategy,
        "run max of native hourly 10 m wind maxima"
    );
    assert_eq!(computed.field.units, "kt");
}

#[test]
fn compute_temp2m_diurnal_windows_take_pointwise_extrema_and_convert_to_c() {
    let mut temp = BTreeMap::new();
    for hour in 1..=24 {
        temp.insert(
            hour,
            Ok(HrrrSurfaceSnapshotDecode {
                temp2m_k: Some(vec![273.15 + hour as f64, 310.15 - hour as f64]),
                rh2m_pct: None,
                dewpoint2m_k: None,
            }),
        );
    }

    let max =
        compute_surface_snapshot_product(HrrrWindowedProduct::Temp2m0to24hMax, &tiny_grid(), &temp)
            .unwrap();
    assert_eq!(max.field.values, vec![24.0_f32, 36.0_f32]);
    assert_eq!(max.field.units, "degC");
    assert_eq!(
        max.metadata.strategy,
        "pointwise max of hourly 2 m temperature snapshots across F001-F024"
    );

    let min =
        compute_surface_snapshot_product(HrrrWindowedProduct::Temp2m0to24hMin, &tiny_grid(), &temp)
            .unwrap();
    assert_eq!(min.field.values, vec![1.0_f32, 13.0_f32]);
    assert_eq!(min.field.units, "degC");
    assert_eq!(
        min.metadata.strategy,
        "pointwise min of hourly 2 m temperature snapshots across F001-F024"
    );

    let range = compute_surface_snapshot_product(
        HrrrWindowedProduct::Temp2m0to24hRange,
        &tiny_grid(),
        &temp,
    )
    .unwrap();
    assert_eq!(range.field.values, vec![23.0_f32, 23.0_f32]);
    assert_eq!(range.field.units, "degC");
    assert_eq!(
        range.metadata.strategy,
        "pointwise max-min range of hourly 2 m temperature snapshots across F001-F024"
    );
}

#[test]
fn compute_surface_snapshot_diurnal_windows_cover_rh_dewpoint_and_vpd() {
    let mut snapshots = BTreeMap::new();
    for hour in 1..=24 {
        snapshots.insert(
            hour,
            Ok(HrrrSurfaceSnapshotDecode {
                temp2m_k: Some(vec![303.15, 293.15]),
                rh2m_pct: Some(vec![20.0 + hour as f64, 80.0 - hour as f64]),
                dewpoint2m_k: Some(vec![283.15 + hour as f64 * 0.1, 273.15]),
            }),
        );
    }

    let rh_range = compute_surface_snapshot_product(
        HrrrWindowedProduct::Rh2m0to24hRange,
        &tiny_grid(),
        &snapshots,
    )
    .unwrap();
    assert_eq!(rh_range.field.units, "%");
    assert_eq!(rh_range.field.values, vec![23.0_f32, 23.0_f32]);

    let dewpoint_max = compute_surface_snapshot_product(
        HrrrWindowedProduct::Dewpoint2m0to24hMax,
        &tiny_grid(),
        &snapshots,
    )
    .unwrap();
    assert_eq!(dewpoint_max.field.units, "degC");
    assert!((dewpoint_max.field.values[0] - 12.4).abs() < 0.01);

    let vpd_max = compute_surface_snapshot_product(
        HrrrWindowedProduct::Vpd2m0to24hMax,
        &tiny_grid(),
        &snapshots,
    )
    .unwrap();
    assert_eq!(vpd_max.field.units, "hPa");
    assert!(vpd_max.field.values[0] > vpd_max.field.values[1]);
    assert!(vpd_max.metadata.strategy.contains("vapor pressure deficit"));
}
