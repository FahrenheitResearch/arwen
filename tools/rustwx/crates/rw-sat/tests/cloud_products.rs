//! Fixture tests for the L2 cloud-product decode, DQF gating and CWP
//! derivation, against REAL GOES-19 granules from the public
//! `noaa-goes19` bucket (scan 2026-08-04 18:01 UTC, day 216). Every
//! asserted number below was computed independently with Python
//! (netCDF4/numpy) from the same granules' raw integers — see
//! `tests/fixtures/README.md` for provenance, checksums and the fetch
//! script.
//!
//! The fixtures are not committed with the crate. When the fixture
//! directory is absent the tests announce themselves loudly and pass
//! vacuously; run `tests/fixtures/fetch_fixtures.sh` (or `.ps1`) once to
//! arm them.

use std::path::PathBuf;

use rw_sat::cloud::{
    CloudProduct, DqfReport, read_cloud_product_field, read_cloud_product_field_window,
};
use rw_sat::cwp::{CwpCounts, cloud_water_path_plane};

const ACHAM_GRANULE: &str =
    "OR_ABI-L2-ACHAM1-M6_G19_s20262161801249_e20262161801336_c20262161801594.nc";
const CODC_GRANULE: &str =
    "OR_ABI-L2-CODC-M6_G19_s20262161801170_e20262161803545_c20262161805324.nc";
const CPSC_GRANULE: &str =
    "OR_ABI-L2-CPSC-M6_G19_s20262161801170_e20262161803545_c20262161805325.nc";
const ACTPC_GRANULE: &str =
    "OR_ABI-L2-ACTPC-M6_G19_s20262161801170_e20262161803545_c20262161804390.nc";

/// The shared CONUS test window (2 km fixed grid): x 1360..1520,
/// y 640..800.
const WIN_X: (usize, usize) = (1360, 160);
const WIN_Y: (usize, usize) = (640, 160);

fn fixture_dir() -> Option<PathBuf> {
    let dir = std::env::var_os("RW_SAT_CLOUD_FIXTURE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("tests")
                .join("fixtures")
        });
    if dir.is_dir() {
        Some(dir)
    } else {
        eprintln!(
            "SKIPPING cloud fixture test: fixture dir {} not found. \
             Run tests/fixtures/fetch_fixtures.sh (or .ps1), or point \
             RW_SAT_CLOUD_FIXTURE_DIR at the granules.",
            dir.display()
        );
        None
    }
}

fn fixture(dir: &std::path::Path, name: &str) -> PathBuf {
    let path = dir.join(name);
    assert!(
        path.is_file(),
        "fixture dir exists but granule {name} is missing — re-run the fetch script"
    );
    path
}

fn assert_close(actual: f32, expected: f32, what: &str) {
    let scale = expected.abs().max(1e-6);
    assert!(
        (actual - expected).abs() / scale < 1e-4,
        "{what}: {actual} != {expected}"
    );
}

#[test]
fn acham_height_decodes_and_gates_fail_closed() {
    let Some(dir) = fixture_dir() else { return };
    let read =
        read_cloud_product_field(fixture(&dir, ACHAM_GRANULE), CloudProduct::CloudTopHeight)
            .expect("decode ACHAM1 granule");

    let grid = &read.field.scene.fixed_grid;
    assert_eq!((grid.nx, grid.ny), (250, 250), "meso 2 km grid");
    assert_eq!(read.field.units.as_deref(), Some("m"));
    assert_eq!(read.field.variable_name, "HT");

    // Independently computed with Python netCDF4/numpy from raw integers:
    // every DQF != 0 pixel is fill in HT too, so the gate masks nothing
    // new but still condemns 36,648 of 62,500 pixels.
    assert_eq!(
        read.dqf,
        DqfReport {
            total: 62_500,
            primary_missing: 36_648,
            dqf_missing: 0,
            dqf_bad: 36_648,
            masked: 0,
            finite: 25_852,
        }
    );

    // Spot pixels (row-major, y*nx + x), decoded raw*0.30520370602607727:
    let at = |y: usize, x: usize| read.field.values[y * grid.nx + x];
    assert_close(at(0, 16), 2398.9011, "HT[0,16] (raw 7860, DQF 0)");
    assert_close(at(120, 120), 10553.028, "HT[120,120] (raw 34577, DQF 0)");
    assert_close(at(249, 249), 3128.338, "HT[249,249] (raw 10250, DQF 0)");

    let mean = mean_of_finite(&read.field.values);
    assert_close(mean as f32, 5733.527, "mean of gated HT");
}

#[test]
fn conus_window_decodes_cod_cps_phase_with_dqf_accounting() {
    let Some(dir) = fixture_dir() else { return };
    let (xs, xc) = WIN_X;
    let (ys, yc) = WIN_Y;

    let cod = read_cloud_product_field_window(
        fixture(&dir, CODC_GRANULE),
        CloudProduct::OpticalDepth,
        xs,
        xc,
        ys,
        yc,
    )
    .expect("decode CODC window");
    let cps = read_cloud_product_field_window(
        fixture(&dir, CPSC_GRANULE),
        CloudProduct::ParticleSize,
        xs,
        xc,
        ys,
        yc,
    )
    .expect("decode CPSC window");
    let phase = read_cloud_product_field_window(
        fixture(&dir, ACTPC_GRANULE),
        CloudProduct::CloudTopPhase,
        xs,
        xc,
        ys,
        yc,
    )
    .expect("decode ACTPC window");

    assert_eq!(cps.field.units.as_deref(), Some("um"), "CPS is µm");

    // Independently computed DQF accounting for the window. The DCOMP
    // (COD/CPS) DQF planes are bit-identical, so their gate counts match;
    // the primary planes differ (clear sky is COD 0.0 but CPS fill).
    assert_eq!(
        cod.dqf,
        DqfReport {
            total: 25_600,
            primary_missing: 30,
            dqf_missing: 0,
            dqf_bad: 850,
            masked: 840,
            finite: 24_730,
        }
    );
    assert_eq!(
        cps.dqf,
        DqfReport {
            total: 25_600,
            primary_missing: 8_028,
            dqf_missing: 0,
            dqf_bad: 850,
            masked: 840,
            finite: 16_732,
        }
    );
    assert_eq!(
        phase.dqf,
        DqfReport {
            total: 25_600,
            primary_missing: 0,
            dqf_missing: 0,
            dqf_bad: 1_640,
            masked: 1_640,
            finite: 23_960,
        }
    );

    // A sun-glint pixel (DQF 742 = glint bit 64 set): finite raw COD
    // (3.1765606) gated to NaN. Fail-closed, and the reason is a recorded
    // count rather than a lost bit.
    let nx = cod.field.scene.fixed_grid.nx;
    assert!(cod.field.values[147 * nx + 145].is_nan());
}

#[test]
fn cwp_from_real_granules_matches_independent_computation() {
    let Some(dir) = fixture_dir() else { return };
    let (xs, xc) = WIN_X;
    let (ys, yc) = WIN_Y;

    let cod = read_cloud_product_field_window(
        fixture(&dir, CODC_GRANULE),
        CloudProduct::OpticalDepth,
        xs,
        xc,
        ys,
        yc,
    )
    .unwrap();
    let cps = read_cloud_product_field_window(
        fixture(&dir, CPSC_GRANULE),
        CloudProduct::ParticleSize,
        xs,
        xc,
        ys,
        yc,
    )
    .unwrap();
    let phase = read_cloud_product_field_window(
        fixture(&dir, ACTPC_GRANULE),
        CloudProduct::CloudTopPhase,
        xs,
        xc,
        ys,
        yc,
    )
    .unwrap();

    // The three products must share one fixed grid, bit for bit —
    // otherwise combining their planes would be fabrication.
    assert_eq!(cod.field.scene.fixed_grid, cps.field.scene.fixed_grid);
    assert_eq!(cod.field.scene.fixed_grid, phase.field.scene.fixed_grid);

    let (cwp, counts) =
        cloud_water_path_plane(&cod.field.values, &cps.field.values, &phase.field.values)
            .expect("derive CWP plane");

    // Independently computed with Python netCDF4/numpy (same DQF gates,
    // same coefficients) over the window.
    assert_eq!(
        counts,
        CwpCounts {
            clear_zero: 3_857,
            liquid: 5_141,
            supercooled: 434,
            mixed: 531,
            ice: 10_303,
            unknown: 0,
            phase_missing: 1_640,
            input_missing: 3_694,
        }
    );
    assert_eq!(counts.finite(), 20_266);
    assert_eq!(cwp.len(), 25_600);

    let nx = cod.field.scene.fixed_grid.nx;
    let at = |y: usize, x: usize| cwp[y * nx + x];
    // Liquid pixel: COD 2.7810166, CPS 12.371739 µm -> 22.937342 g/m².
    assert_close(at(45, 20), 22.937342, "liquid CWP");
    // Ice pixel: COD 21.115215, CPS 34.402565 µm -> 444.0833 g/m².
    assert_close(at(108, 123), 444.0833, "ice CWP");
    // Clear-sky pixel: exact zero observation (CPS is fill there).
    assert_eq!(at(88, 33), 0.0, "clear-sky zero");

    let finite_count = cwp.iter().filter(|value| value.is_finite()).count();
    assert_eq!(finite_count, 20_266);
    assert_close(mean_of_finite(&cwp) as f32, 124.07099, "mean CWP");
    let max = cwp.iter().copied().fold(f32::NAN, f32::max);
    assert_close(max, 5541.37, "max CWP");
}

#[test]
fn fixture_filenames_parse_to_the_cloud_products() {
    // Pure filename checks — no fixture download required.
    use rw_sat::goes::parse_goes_abi_filename;
    let acham = parse_goes_abi_filename(ACHAM_GRANULE).unwrap();
    assert_eq!(acham.product, "ABI-L2-ACHAM1");
    assert_eq!(acham.mode, Some(6));
    assert_eq!(acham.channel, None, "cloud products carry no band token");
    let codc = parse_goes_abi_filename(CODC_GRANULE).unwrap();
    assert_eq!(codc.product, "ABI-L2-CODC");
    assert_eq!(
        codc.start_time_utc.to_rfc3339(),
        "2026-08-04T18:01:17+00:00"
    );
}

fn mean_of_finite(values: &[f32]) -> f64 {
    let mut sum = 0.0f64;
    let mut count = 0usize;
    for &value in values {
        if value.is_finite() {
            sum += f64::from(value);
            count += 1;
        }
    }
    sum / count.max(1) as f64
}
