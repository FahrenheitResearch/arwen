//! The non-ODIM readers on real files from the networks that publish them.
//!
//! The files are fetched on demand (`tools/radar_world_fixtures.sh DIR`),
//! pinned by SHA-256, and named by `RW_ODIM_WORLD_DATA=DIR`:
//!
//! ```text
//! RW_ODIM_WORLD_DATA=DIR cargo test --release -p rw-odim --test world_real_files -- --ignored
//! ```
//!
//! The numbers asserted for IRIS, UF and CfRadial are Py-ART 2.3's on the
//! same files (`read_sigmet`, `read_uf`, `read_cfradial`): measured-gate
//! counts per moment agreed exactly and every common gate's value agreed to
//! float32 rounding (worst 9.5e-7). The JMA numbers come from the files'
//! own level tables and PRF metadata; no independent reader decodes JMA's
//! polar GRIB2 (ecCodes lacks its local template 3.50120).

use std::path::PathBuf;

use rw_odim::censor;
use rw_odim::pack::{PackOptions, build_pack};
use rw_odim::volume::NyquistSource;
use rw_odim::world::{Format, merge_station, read_path, select_site};
use rw_odim::{DecodeOptions, PolarVolume};

fn data(name: &str) -> PathBuf {
    let dir = std::env::var("RW_ODIM_WORLD_DATA").unwrap_or_else(|_| {
        panic!(
            "set RW_ODIM_WORLD_DATA to a directory filled by tools/radar_world_fixtures.sh \
             (these tests read real files that are fetched on demand, not committed)"
        )
    });
    let path = PathBuf::from(dir).join(name);
    assert!(
        path.is_file(),
        "{} is missing; run tools/radar_world_fixtures.sh",
        path.display()
    );
    path
}

fn one(name: &str, format: Format) -> PolarVolume {
    let read = read_path(&data(name), None, &DecodeOptions::all()).expect("decodes");
    assert_eq!(read.format, format);
    assert_eq!(read.volumes.len(), 1);
    read.volumes.into_iter().next().unwrap()
}

fn measured(volume: &PolarVolume, sweep: usize, quantity: &str) -> usize {
    volume.sweeps[sweep]
        .moment(quantity)
        .unwrap_or_else(|| panic!("sweep {sweep} has no {quantity}"))
        .census
        .measured
}

/// The whole volume must pack and read back, as `rw_odim pack` does it.
fn packs(volume: &PolarVolume) {
    let (meta, payload) =
        build_pack(volume, &PackOptions::default(), "fixture", 1, "0", None).expect("packs");
    let json = serde_json::to_string(&meta).expect("serialises");
    let _: rw_nexrad::pack::PackMeta = serde_json::from_str(&json).expect("reads back");
    assert!(!payload.is_empty());
}

#[test]
#[ignore = "fetch-on-demand real files: tools/radar_world_fixtures.sh"]
fn jma_no_echo_is_clear_air_and_velocity_gets_the_dual_prf_interval() {
    let n5 = data("Z__C_RJTD_20191012090000_RDR_JMAGPV_N5_grib2.tar");
    let n6 = data("Z__C_RJTD_20191012090000_RDR_JMAGPV_N6_grib2.tar");

    // The shared decode, for the count of gates it reports as 0.00 dBZ.
    let raw = std::fs::read(&n5).unwrap();
    let shared = nexrad_io::jma::decode_jma_tar_volumes(&raw, Some("AKIT")).unwrap();
    let zero_dbz: usize = shared[0]
        .cuts
        .iter()
        .flat_map(|cut| cut.moments.values())
        .map(|grid| match &grid.storage {
            radar_core::MomentStorage::F32(v) => v.iter().filter(|x| **x == 0.0).count(),
            _ => 0,
        })
        .sum();
    assert!(
        zero_dbz > 1_000_000,
        "the shared decode called {zero_dbz} gates 0 dBZ"
    );

    let mut parts = Vec::new();
    for file in [&n5, &n6] {
        let read = read_path(file, Some("AKIT"), &DecodeOptions::all()).expect("decodes");
        assert_eq!(read.format, Format::JmaGrib2Tar);
        parts.extend(read.volumes);
    }
    let parts = select_site(parts, Some("AKIT"), "jma").unwrap();
    let volume = merge_station(parts, "jma").unwrap();
    assert_eq!(volume.source.label(), "AKIT");
    assert_eq!(volume.source.rad.as_deref(), Some("RS47582"));

    let mut undetect = 0usize;
    let mut dual = 0usize;
    for sweep in &volume.sweeps {
        if let Some(dbz) = sweep.moment("DBZH") {
            undetect += dbz.census.undetect;
            for (value, code) in dbz.values.iter().zip(&dbz.censor) {
                if *code == censor::MEASURED {
                    assert!(*value > 0.0, "a measured gate at {value} dBZ survived");
                }
            }
        }
        if let Some(vel) = sweep.moment("VRADH") {
            match sweep.nyquist.source {
                NyquistSource::DerivedDualPrf => {
                    dual += 1;
                    let interval = sweep.nyquist.interval_ms.unwrap();
                    assert!((interval - 52.53).abs() < 0.05, "{interval}");
                    let single = interval / 4.0; // 5:4 pair
                    let speeds: Vec<f64> = vel
                        .values
                        .iter()
                        .zip(&vel.censor)
                        .filter(|(_, c)| **c == censor::MEASURED)
                        .map(|(v, _)| v.abs())
                        .collect();
                    let beyond = speeds.iter().filter(|s| **s > interval + 0.01).count();
                    let folded_if_single = speeds.iter().filter(|s| **s > single).count();
                    // Typhoon winds: far beyond the single-PRF interval, and
                    // (all but a handful of outliers) inside the extended one.
                    assert!(
                        beyond * 10_000 <= speeds.len(),
                        "{beyond} of {} beyond",
                        speeds.len()
                    );
                    assert!(folded_if_single * 10 > speeds.len());
                }
                NyquistSource::Unavailable => {
                    assert!(sweep.nyquist.interval_ms.is_none());
                }
                other => panic!("unexpected Nyquist source {other:?}"),
            }
        }
    }
    // Every gate the shared decode called 0 dBZ is level 1, "No Echo".
    assert_eq!(undetect, zero_dbz);
    // 13 Doppler cuts in the file: 11 declare a 940/752 Hz pair, the two
    // long-range lowest cuts declare no PRF.
    assert_eq!(dual, 11, "dual-PRF velocity sweeps");
    packs(&volume);
}

#[test]
#[ignore = "fetch-on-demand real files: tools/radar_world_fixtures.sh"]
fn iris_raw_matches_pyart_on_ideam_tablazo() {
    let volume = one("TAB250601000003.RAWYSBR", Format::IrisRaw);
    assert_eq!(volume.sweeps.len(), 9);
    let elevations: Vec<f64> = volume.sweeps.iter().map(|s| s.elevation_deg).collect();
    for (got, want) in elevations
        .iter()
        .zip([0.0, 0.5, 1.0, 2.0, 4.5, 7.0, 10.0, 15.0, 25.0])
    {
        assert!((got - want).abs() < 0.01, "{elevations:?}");
    }
    assert!((volume.site.latitude_deg - 5.0118).abs() < 1e-4);
    assert!((volume.site.longitude_deg + 74.2037).abs() < 1e-4);
    // Antenna, not ground: 3523 m site plus a 21 m tower.
    assert_eq!(volume.site.height_m, 3544.0);
    let first = &volume.sweeps[0];
    assert_eq!(first.nbins, 664);
    assert_eq!(first.range_scale_m, 450.0);
    assert_eq!(first.gate_centre_range_m(0), 300.0);
    assert_eq!(first.nyquist.source, NyquistSource::DerivedSinglePrf);
    assert!((first.nyquist.interval_ms.unwrap() - 6.6625).abs() < 1e-3);
    // Py-ART's measured-gate counts.
    assert_eq!(measured(&volume, 0, "TH"), 119_293);
    assert_eq!(measured(&volume, 0, "DBZH"), 2_213);
    assert_eq!(measured(&volume, 0, "VRADH"), 2_328);
    assert_eq!(measured(&volume, 0, "KDP"), 94_322);
    assert_eq!(measured(&volume, 2, "DBZH"), 526);
    // Thresholded gates are clear air, not missing.
    assert!(volume.sweeps[0].moment("DBZH").unwrap().census.undetect > 200_000);
    packs(&volume);
}

#[test]
#[ignore = "fetch-on-demand real files: tools/radar_world_fixtures.sh"]
fn universal_format_matches_pyart_on_a_public_rvp8_file() {
    let volume = one(
        "20110427_164233_rvp8-rel_v001_SUR.uf.gz",
        Format::UniversalFormat,
    );
    assert_eq!(volume.sweeps.len(), 14);
    assert!((volume.site.latitude_deg - 34.93181).abs() < 1e-4);
    assert_eq!(volume.site.height_m, 226.0);
    let first = &volume.sweeps[0];
    assert_eq!(first.range_scale_m, 125.0);
    // UF words 3-4 are the range to the first gate's centre.
    assert_eq!(first.gate_centre_range_m(0), 501.0);
    assert_eq!(first.nyquist.source, NyquistSource::Declared);
    assert!((first.nyquist.interval_ms.unwrap() - 7.92).abs() < 1e-6);
    assert_eq!(measured(&volume, 0, "DBZ"), 27_598);
    assert_eq!(measured(&volume, 0, "VRAD"), 27_598);
    assert_eq!(measured(&volume, 1, "DBZ"), 41_669);
    assert_eq!(measured(&volume, 2, "VRAD"), 57_565);
    packs(&volume);
}

#[test]
#[ignore = "fetch-on-demand real files: tools/radar_world_fixtures.sh"]
fn cfradial_keeps_its_exact_gate_axis_and_its_velocity() {
    let volume = one(
        "1399BOG-20250601-000042-PPIVol-7550.nc.gz",
        Format::Cfradial1,
    );
    assert_eq!(volume.sweeps.len(), 1);
    let sweep = &volume.sweeps[0];
    // 62.5 m gates, not the shared model's whole-metre 63.
    assert_eq!(sweep.range_scale_m, 62.5);
    assert!((sweep.gate_centre_range_m(0) - 31.25).abs() < 1e-6);
    assert!((sweep.nyquist.interval_ms.unwrap() - 8.3903).abs() < 1e-3);
    assert_eq!(measured(&volume, 0, "DBZH"), 11_165);
    // VELH is the horizontal velocity and must claim the pack's VEL token.
    assert_eq!(measured(&volume, 0, "VRADH"), 35_385);
    packs(&volume);
}

#[test]
#[ignore = "fetch-on-demand real files: tools/radar_world_fixtures.sh"]
fn a_bom_volume_from_nci_packs_under_its_own_radar_id() {
    let volume = one("2_20261013_120000.pvol.h5", Format::OdimH5);
    assert_eq!(volume.source.rad.as_deref(), Some("AU02"));
    let (meta, _) =
        build_pack(&volume, &PackOptions::default(), "nci", 1, "0", None).expect("packs");
    assert_eq!(meta.site.id, "AU02");
    let json = serde_json::to_string(&meta).unwrap();
    let back: rw_nexrad::pack::PackMeta = serde_json::from_str(&json).expect("reads back");
    assert!(back.params.max_range_km.is_finite());
}
