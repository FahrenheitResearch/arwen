//! Actual native-history to classic-WRF conversion, including unsmoothed winds.
use std::path::{Path, PathBuf};

use rw_mpas::convert::{ConvertOptions, RADAR_NATIVE_WINDS, convert_frame};
use rw_mpas::history::{Timestamp, read_history};
use rw_mpas::weights::{AngleUnits, NearestCellWeights};
use rw_mpas::window::{LatLonWindow, Window};
use rw_store::netcdf_classic::{
    NcAttr, NcClassicWriter, NcData, NcDim, NcFormat, NcType, NcVarDef,
};

fn fixture(metadata: bool, meridional: bool) -> (PathBuf, NearestCellWeights) {
    // The test harness runs these cases on parallel threads of one process,
    // and a clock reading is not unique between threads (Windows' system
    // clock advances in coarse steps), so two fixtures could share a
    // directory and overwrite each other's history. A per-process counter
    // with the process id is unique, and `create_dir` refuses a collision.
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let serial = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let directory =
        std::env::temp_dir().join(format!("rw-mpas-radar-{}-{serial}", std::process::id()));
    if directory.exists() {
        // A directory left by an earlier process that reused this pid.
        std::fs::remove_dir_all(&directory).unwrap();
    }
    std::fs::create_dir(&directory).unwrap();
    let path = directory.join("native-history.nc");
    let attrs = if metadata {
        vec![
            NcAttr::int("MP_PHYSICS", 28),
            NcAttr::int("morr_rimed_ice", 1),
            NcAttr::text("RADAR_MIXING_RATIO_BASIS", "dry-air"),
            NcAttr::text("WOOF_MICROPHYSICS_SCHEME", "declared_scheme"),
            NcAttr::text("WOOF_ENGINE_MICROPHYSICS_SCHEME", "declared_engine_scheme"),
            NcAttr::text("WOOF_PHYSICS_BACKEND", "declared_backend"),
            NcAttr::text("WOOF_SCALAR_NAMES", "qv,qc,qr,qi,qs,qg"),
        ]
    } else {
        Vec::new()
    };
    let dims = vec![
        NcDim::record("Time"),
        NcDim::fixed("nCells", 6),
        NcDim::fixed("nVertLevels", 2),
    ];
    let mut vars = vec![
        NcVarDef::new("latCell", NcType::Double, vec![1]),
        NcVarDef::new("lonCell", NcType::Double, vec![1]),
        NcVarDef::new("u_zonal", NcType::Float, vec![0, 1, 2]),
        NcVarDef::new("qr", NcType::Float, vec![0, 1, 2]),
        NcVarDef::new("rainnc", NcType::Float, vec![0, 1]),
    ];
    if meridional {
        vars.push(NcVarDef::new("v_meridional", NcType::Float, vec![0, 1, 2]));
    }
    let mut writer =
        NcClassicWriter::create(&path, NcFormat::Offset64, dims, attrs, vars, 1).unwrap();
    writer.put("latCell", NcData::Doubles(&[0.0; 6])).unwrap();
    writer.put("lonCell", NcData::Doubles(&[0.0; 6])).unwrap();
    writer
        .put_record(
            "u_zonal",
            0,
            NcData::Floats(&[1., 101., 3., 103., 7., 107., 9., 109., 11., 111., 13., 113.]),
        )
        .unwrap();
    if meridional {
        writer
            .put_record(
                "v_meridional",
                0,
                NcData::Floats(&[
                    -2., 198., -4., 196., -8., 192., -10., 190., -12., 188., -14., 186.,
                ]),
            )
            .unwrap();
    }
    writer
        .put_record("qr", 0, NcData::Floats(&[0.0001; 12]))
        .unwrap();
    writer
        .put_record("rainnc", 0, NcData::Floats(&[1., 2., 3., 4., 5., 6.]))
        .unwrap();
    writer.finish().unwrap();
    let weights = NearestCellWeights {
        window_name: "test".into(),
        window: Window::LatLon(LatLonWindow {
            south: 30.,
            north: 31.,
            west: -100.,
            east: -98.,
            spacing_degrees: 1.,
            description: "test grid".into(),
        }),
        target_latitude: vec![30., 30., 30., 31., 31., 31.],
        target_longitude: vec![-100., -99., -98., -100., -99., -98.],
        cell_index: vec![2, 0, 1, 5, 3, 4],
        off_mesh: Vec::new(),
        distance_km: vec![0.; 6],
        mesh_sha256: "test-mesh".into(),
        mesh_path: "native-history.nc".into(),
        mesh_angle_units: AngleUnits::Declared,
        n_cells: 6,
        ny: 2,
        nx: 3,
        weights_sha256: "test-weights".into(),
        build_seconds: 0.,
    };
    (path, weights)
}

fn convert(
    path: &Path,
    weights: &NearestCellWeights,
    field_set: &str,
) -> (PathBuf, rw_mpas::convert::EmittedFrame) {
    let frame = read_history(
        path,
        None,
        Some(Timestamp::parse("2026-10-02_12:00:00").unwrap()),
        None,
        None,
    )
    .unwrap();
    let output = path.with_file_name(format!("converted-{field_set}.nc"));
    let result = convert_frame(
        &frame,
        weights,
        &output,
        &ConvertOptions {
            field_set: field_set.into(),
            ..Default::default()
        },
    )
    .unwrap();
    (output, result)
}

#[test]
fn full_conversion_preserves_declared_physics_and_mass_winds_before_staggering() {
    let (history, weights) = fixture(true, true);
    let (output, result) = convert(&history, &weights, "full");
    let native = netcrust::File::open(&history).unwrap();
    let actual = netcrust::File::open(&output).unwrap();
    for name in [
        "MP_PHYSICS",
        "WOOF_MICROPHYSICS_SCHEME",
        "WOOF_ENGINE_MICROPHYSICS_SCHEME",
        "WOOF_PHYSICS_BACKEND",
        "WOOF_SCALAR_NAMES",
        "RADAR_MIXING_RATIO_BASIS",
    ] {
        assert_eq!(
            actual.attribute(name).unwrap().value(),
            native.attribute(name).unwrap().value(),
            "{name}"
        );
    }
    assert_eq!(
        actual.attribute("MORR_RIMED_ICE").unwrap().as_f64(),
        Some(1.0)
    );
    assert_eq!(
        actual.attribute("RADAR_NATIVE_WINDS").unwrap().as_string(),
        Some(RADAR_NATIVE_WINDS)
    );
    assert_eq!(
        actual.variable("RADAR_U_EARTH").unwrap().shape(),
        vec![1, 2, 2, 3]
    );
    assert_eq!(
        actual.variable("RADAR_V_EARTH").unwrap().shape(),
        vec![1, 2, 2, 3]
    );
    assert_eq!(
        actual.read_f64("RADAR_U_EARTH").unwrap(),
        vec![7., 1., 3., 13., 9., 11., 107., 101., 103., 113., 109., 111.]
    );
    assert_eq!(
        actual.read_f64("RADAR_V_EARTH").unwrap(),
        vec![
            -8., -2., -4., -14., -10., -12., 192., 198., 196., 186., 190., 188.
        ]
    );
    // The established WRF face-grid variables keep their original values.
    assert_eq!(
        actual.read_f64("U").unwrap(),
        vec![
            7., 4., 2., 3., 13., 11., 10., 11., 107., 104., 102., 103., 113., 111., 110., 111.
        ]
    );
    assert_eq!(
        actual.read_f64("V").unwrap(),
        vec![
            -8., -2., -4., -11., -6., -8., -14., -10., -12., 192., 198., 196., 189., 194., 192.,
            186., 190., 188.
        ]
    );
    assert_eq!(
        actual.read_f64("RAINNC").unwrap(),
        vec![3., 1., 2., 6., 4., 5.]
    );
    assert_eq!(
        actual.read_f64("QRAIN").unwrap(),
        vec![f64::from(0.0001f32); 12]
    );
    assert!(result.written.contains(&"RADAR_U_EARTH".to_string()));
    assert!(result.written.contains(&"RADAR_V_EARTH".to_string()));
}

#[test]
fn absent_physics_is_not_invented_and_surface_conversion_has_no_radar_winds() {
    let (history, weights) = fixture(false, true);
    let (output, _) = convert(&history, &weights, "surface");
    let actual = netcrust::File::open(&output).unwrap();
    for name in [
        "MP_PHYSICS",
        "WOOF_MICROPHYSICS_SCHEME",
        "WOOF_ENGINE_MICROPHYSICS_SCHEME",
        "WOOF_PHYSICS_BACKEND",
        "WOOF_SCALAR_NAMES",
        "RADAR_NATIVE_WINDS",
    ] {
        assert!(actual.attribute(name).is_none(), "{name}");
    }
    assert!(actual.variable("RADAR_U_EARTH").is_none());
    assert!(actual.variable("RADAR_V_EARTH").is_none());
    assert_eq!(
        actual.read_f64("RAINNC").unwrap(),
        vec![3., 1., 2., 6., 4., 5.]
    );
}

#[test]
fn incomplete_wind_pair_does_not_claim_native_wind_contract() {
    let (history, weights) = fixture(false, false);
    let (output, _) = convert(&history, &weights, "full");
    let actual = netcrust::File::open(&output).unwrap();
    assert!(actual.attribute("RADAR_NATIVE_WINDS").is_none());
    assert!(actual.variable("RADAR_U_EARTH").is_none());
    assert!(actual.variable("RADAR_V_EARTH").is_none());
    assert!(actual.variable("U").is_some());
}
