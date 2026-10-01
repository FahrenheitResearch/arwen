//! The column planes the wrfout import derives (an isotherm's height, a
//! supercooled liquid water path, a hydrometeor's column maximum, the
//! simulated infrared brightness temperature), proven against a file
//! written the way this tree's own history stream writes one: the full
//! eta levels `ZNW` and no layer thicknesses `DNW`.
//!
//! The concrete breakage this gate prevents: on such a file the import
//! found no `DNW`, skipped every plane that integrates over a layer
//! mass, and `--list-products` answered `supercooled_water_path` and
//! `simulated_ir_satellite` with "not stored" while the file carried
//! everything the integral needs.  The values are checked, not only the
//! names: a plane stored under the right name with the wrong integral
//! would draw a confidently wrong map.

mod stored_plane_fixture;

use std::path::{Path, PathBuf};

use rw_wrfbatch::wrf_process::{WrfProcessMessage, WrfProcessOptions, spawn_process_paths};
use stored_plane_fixture as fixture;

/// wrf-core's own constants: the reader divides geopotential by this
/// gravity and raises the Exner function to this exponent.
const READER_G: f64 = 9.806_65;
const READER_KAPPA: f64 = 0.285_714_285_7;
/// The gravity the fixture multiplied its heights by and the import
/// divides its layer mass by.
const FIXTURE_G: f64 = 9.81;
/// The gravity `wrfcttcalc` divides a layer's pressure thickness by.
const REFERENCE_G: f64 = 9.81;
const T_FREEZE_K: f64 = 273.15;

struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Self {
        let dir = std::env::temp_dir().join(format!(
            "rw-wrfbatch-column-planes-{tag}-{}-{:?}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|value| value.as_nanos())
                .unwrap_or_default()
        ));
        std::fs::create_dir_all(&dir).expect("create scratch dir");
        Self(dir)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

struct Imported {
    variables: Vec<String>,
    notes: Vec<String>,
    hour: PathBuf,
}

fn import(wrfout: &Path, store_root: &Path) -> Imported {
    let task = spawn_process_paths(
        vec![wrfout.to_path_buf()],
        store_root.to_path_buf(),
        WrfProcessOptions::default(),
    );
    let summary = loop {
        match task.rx.recv().expect("the WRF processor answers") {
            WrfProcessMessage::Progress(_) => {}
            WrfProcessMessage::Done(result) => {
                break result.expect("the synthetic wrfout imports");
            }
        }
    };
    Imported {
        hour: store_root
            .join(&summary.model)
            .join(&summary.run)
            .join("f000.rws"),
        variables: summary.variables,
        notes: summary.notes,
    }
}

/// Temperature on mass level `k` as the reader computes it from the
/// fixture's theta and pressure.
fn temperature_k(level: usize) -> f64 {
    fixture::THETA_K * (fixture::pressure_pa(level) / 100_000.0).powf(READER_KAPPA)
}

/// Height of mass level `k` as the reader computes it: the fixture wrote
/// geopotential with one gravity and the reader divides by its own.
fn height_msl_m(level: usize) -> f64 {
    fixture::mass_level_height_msl_m(level) * FIXTURE_G / READER_G
}

/// The dry-air mass of every layer, kg m-2: the column mass over the
/// eta thickness of a layer, over gravity.
fn layer_mass_kg_m2() -> f64 {
    let thickness = f64::from(fixture::ETA_FULL_LEVELS[0] - fixture::ETA_FULL_LEVELS[1]);
    f64::from(fixture::DRY_COLUMN_MASS_PA) * thickness / FIXTURE_G
}

fn read_plane(hour: &Path, name: &str) -> Vec<f32> {
    let reader = rw_store::reader::HourReader::open(hour).expect("open the stored hour");
    let values = reader.read_full_2d(name).unwrap_or_else(|err| {
        panic!("read {name} back: {err}");
    });
    assert_eq!(values.len(), fixture::NX * fixture::NY, "{name} is one plane");
    values
}

fn assert_every_cell_near(values: &[f32], expected: f64, tolerance: f64, what: &str) {
    for (cell, value) in values.iter().enumerate() {
        assert!(
            (f64::from(*value) - expected).abs() <= tolerance,
            "{what}: cell {cell} reads {value}, expected {expected} within {tolerance}"
        );
    }
}

#[test]
fn a_file_with_eta_levels_and_no_thicknesses_stores_every_column_plane() {
    let scratch = Scratch::new("planes");
    let wrfout = fixture::write(scratch.path());
    let imported = import(&wrfout, &scratch.path().join("store"));

    for name in [
        "isotherm_height_0c",
        "isotherm_height_minus10c",
        "isotherm_height_minus20c",
        "supercooled_water_path",
        "supercooled_water_path_0_3km",
        "supercooled_water_path_3_6km",
        "cloud_water_column_max",
        "cloud_ice_column_max",
        "simulated_ir_brightness_temperature",
    ] {
        assert!(
            imported.variables.iter().any(|stored| stored == name),
            "{name} was not stored; notes: {:?}; stored: {:?}",
            imported.notes,
            imported.variables
        );
    }
    let skipped: Vec<&String> = imported
        .notes
        .iter()
        .filter(|note| note.contains("DNW") || note.contains("column planes skipped"))
        .collect();
    assert!(skipped.is_empty(), "a column plane was skipped: {skipped:?}");
}

#[test]
fn the_supercooled_path_integrates_the_cold_levels_inside_each_layer() {
    let scratch = Scratch::new("path");
    let wrfout = fixture::write(scratch.path());
    let imported = import(&wrfout, &scratch.path().join("store"));

    // Only the top mass level is below freezing, and it sits in the
    // 3 to 6 km layer above ground; the fixture pins both.
    let cold: Vec<usize> = (0..fixture::NZ)
        .filter(|level| temperature_k(*level) < T_FREEZE_K)
        .collect();
    assert_eq!(cold, vec![fixture::NZ - 1], "the fixture's cold levels moved");
    let cold_agl = height_msl_m(fixture::NZ - 1) - fixture::TERRAIN_M;
    assert!((3_000.0..6_000.0).contains(&cold_agl), "cold level at {cold_agl} m AGL");

    let expected_g_m2 = f64::from(fixture::CLOUD_WATER_KG_PER_KG) * layer_mass_kg_m2() * 1_000.0;
    assert_every_cell_near(
        &read_plane(&imported.hour, "supercooled_water_path"),
        expected_g_m2,
        0.05,
        "whole-column supercooled path",
    );
    assert_every_cell_near(
        &read_plane(&imported.hour, "supercooled_water_path_3_6km"),
        expected_g_m2,
        0.05,
        "3 to 6 km supercooled path",
    );
    assert_every_cell_near(
        &read_plane(&imported.hour, "supercooled_water_path_0_3km"),
        0.0,
        0.0,
        "0 to 3 km supercooled path",
    );
}

#[test]
fn the_isotherm_height_is_the_interpolated_crossing_and_nan_above_the_coldest_level() {
    let scratch = Scratch::new("isotherm");
    let wrfout = fixture::write(scratch.path());
    let imported = import(&wrfout, &scratch.path().join("store"));

    let (below, above) = (fixture::NZ - 2, fixture::NZ - 1);
    let (t_below, t_above) = (temperature_k(below), temperature_k(above));
    assert!(t_below > T_FREEZE_K && t_above < T_FREEZE_K, "the crossing moved");
    let fraction = (t_below - T_FREEZE_K) / (t_below - t_above);
    let expected = height_msl_m(below) + fraction * (height_msl_m(above) - height_msl_m(below));
    assert_every_cell_near(
        &read_plane(&imported.hour, "isotherm_height_0c"),
        expected,
        1.0,
        "0 C isotherm height",
    );

    // -20 C is colder than the coldest level, so no column crosses it.
    let never = read_plane(&imported.hour, "isotherm_height_minus20c");
    assert!(never.iter().all(|value| value.is_nan()), "-20 C: {:?}", &never[..4]);
}

/// The concrete breakage this test prevents: the brightness temperature
/// applied `wrfcttcalc`'s absorption coefficients, which are per gram of
/// condensate, to mixing ratios in kg kg-1, so every optical depth was a
/// thousand times too small; this column read the file's 290 K skin
/// temperature, and cirrus the model holds at -35 to -40 C drew 285 to
/// 300 K.
#[test]
fn the_brightness_temperature_is_the_reference_cloud_top_and_the_column_maximum_is_the_mixing_ratio() {
    let scratch = Scratch::new("opaque");
    let wrfout = fixture::write(scratch.path());
    let imported = import(&wrfout, &scratch.path().join("store"));

    // WRF-Python's wrfcttcalc on the fixture's column, which carries
    // 0.1 g kg-1 of cloud water on every level.  It integrates from one
    // level below the top, over a layer bounded by the full levels halfway
    // in pressure to its neighbours: 12,000 Pa, an optical depth of
    // 0.145 m2 g-1 * 0.1 g kg-1 * 12,000 Pa / 9.81 m s-2 = 17.7.  Depth one
    // is reached a seventeenth of the way down that layer, and the
    // brightness temperature is the temperature at that pressure,
    // interpolated in pressure between the two levels around it.
    let pressure = fixture::pressure_pa;
    let (top_level, below_top) = (fixture::NZ - 1, fixture::NZ - 2);
    let layer_top_pa = 0.5 * (pressure(top_level) + pressure(below_top));
    let layer_bottom_pa = 0.5 * (pressure(below_top) + pressure(below_top - 1));
    let cloud_water_g_per_kg = 1_000.0 * f64::from(fixture::CLOUD_WATER_KG_PER_KG);
    let depth = 0.145 * cloud_water_g_per_kg * (layer_bottom_pa - layer_top_pa) / REFERENCE_G;
    assert!(depth > 1.0, "the fixture's top layer is no longer opaque: {depth}");
    let cloud_top_pa = layer_top_pa + (layer_bottom_pa - layer_top_pa) / depth;
    assert!(
        (pressure(top_level)..=pressure(below_top)).contains(&cloud_top_pa),
        "the cloud top moved out of the top two levels: {cloud_top_pa} Pa"
    );
    let fraction = (cloud_top_pa - pressure(top_level)) / (pressure(below_top) - pressure(top_level));
    let expected = temperature_k(top_level) + fraction * (temperature_k(below_top) - temperature_k(top_level));
    assert!(
        (expected - f64::from(fixture::SKIN_TEMPERATURE_K)).abs() > 5.0,
        "the expected cloud top {expected} K cannot be told from the skin"
    );
    assert_every_cell_near(
        &read_plane(&imported.hour, "simulated_ir_brightness_temperature"),
        expected,
        1.0e-3,
        "brightness temperature",
    );
    assert_every_cell_near(
        &read_plane(&imported.hour, "cloud_water_column_max"),
        f64::from(fixture::CLOUD_WATER_KG_PER_KG),
        1.0e-9,
        "cloud water column maximum",
    );
    assert_every_cell_near(
        &read_plane(&imported.hour, "cloud_ice_column_max"),
        0.0,
        0.0,
        "cloud ice column maximum",
    );
}
