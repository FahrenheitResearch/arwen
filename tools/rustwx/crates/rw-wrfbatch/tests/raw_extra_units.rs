//! A raw extra keeps the units its wrfout declares.
//!
//! The concrete breakage: the raw extras (`RAW_EXTRA_CATALOG`) are read
//! through wrf-core's raw fallback, whose unit table has no row for
//! HAILNC, UP_HELI_MAX, WSPD10MAX, W_UP_MAX or W_DN_MAX, so they reached
//! the store with no unit at all. Hail was then drawn as a unitless -1..1
//! generic field instead of in inches on the precipitation palette, on
//! every frame of a real 3 km forecast whose file declares HAILNC in `mm`
//! and UP_HELI_MAX in `m2 s-2`.

mod stored_plane_fixture;

use std::path::{Path, PathBuf};

use rustwx_core::ModelId;
use rustwx_products::viewer::curated_style_for_store_variable;
use rw_store::reader::HourReader;
use rw_wrfbatch::wrf_process::{WrfProcessMessage, WrfProcessOptions, spawn_process_paths};

struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Self {
        let dir = std::env::temp_dir().join(format!(
            "rw-wrfbatch-raw-units-{tag}-{}-{:?}",
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

/// `(file name, units the file declares, value, store name)`.
const PLANES: [(&str, &str, f32, &str); 5] = [
    ("HAILNC", "mm", 25.4, "wrf_hailnc"),
    ("UP_HELI_MAX", "m2 s-2", 400.0, "wrf_up_heli_max"),
    ("WSPD10MAX", "m s-1", 12.5, "wrf_wspd10max"),
    // The fallback table says mm for SNOWNC; the file is what the values
    // are in, so the file's cm wins.
    ("SNOWNC", "cm", 2.54, "wrf_snownc"),
    // The file names no unit, so the fallback's mm is kept.
    ("GRAUPELNC", "", 1.0, "wrf_graupelnc"),
];

fn import(scratch: &Scratch, options: WrfProcessOptions) -> HourReader {
    let extras: Vec<(&str, &str, f32)> = PLANES
        .iter()
        .map(|(name, units, value, _)| (*name, *units, *value))
        .collect();
    let wrfout = stored_plane_fixture::write_with_surface_planes(scratch.path(), &extras);
    let store_root = scratch.path().join("store");
    let task = spawn_process_paths(vec![wrfout], store_root.clone(), options);
    let summary = loop {
        match task.rx.recv().expect("the WRF processor answers") {
            WrfProcessMessage::Progress(_) => {}
            WrfProcessMessage::Done(result) => break result.expect("the synthetic wrfout imports"),
        }
    };
    HourReader::open(
        &store_root
            .join(&summary.model)
            .join(&summary.run)
            .join("f000.rws"),
    )
    .expect("the imported hour opens")
}

fn stored_units(reader: &HourReader, store_name: &str) -> String {
    reader
        .meta()
        .variables
        .iter()
        .find(|variable| variable.name == store_name)
        .unwrap_or_else(|| panic!("{store_name} is not in the store"))
        .units
        .clone()
}

fn assert_file_units(reader: &HourReader, route: &str) {
    for (name, units, value, store_name) in PLANES {
        let expected = if units.is_empty() { "mm" } else { units };
        assert_eq!(
            stored_units(reader, store_name),
            expected,
            "{name} on the {route} route"
        );
        let values = reader.read_full_2d(store_name).expect("the plane reads");
        assert!(
            values.iter().all(|stored| *stored == value),
            "{name} on the {route} route changed value"
        );
    }
}

#[test]
fn raw_extras_carry_the_units_their_file_declares() {
    let scratch = Scratch::new("default");
    let reader = import(&scratch, WrfProcessOptions::default());
    assert_file_units(&reader, "default");
}

#[test]
fn raw_extras_carry_file_units_with_the_stored_plane_pass_off() {
    // Turning the stored-plane pass off is a speed choice; it must not
    // cost the raw extras their units.
    let scratch = Scratch::new("raw-only");
    let reader = import(
        &scratch,
        WrfProcessOptions {
            stored_planes: false,
            ..WrfProcessOptions::default()
        },
    );
    assert_file_units(&reader, "raw-only");
}

#[test]
fn stored_hail_draws_in_inches_on_the_precipitation_palette() {
    let scratch = Scratch::new("hail-style");
    let reader = import(&scratch, WrfProcessOptions::default());
    let hail = reader
        .meta()
        .variables
        .iter()
        .find(|variable| variable.name == "wrf_hailnc")
        .expect("hail is stored")
        .clone();
    let style = curated_style_for_store_variable(
        &hail.name,
        &hail.selector,
        &hail.units,
        ModelId::WrfGdex,
    )
    .expect("stored hail wears the precipitation palette");
    assert_eq!(style.display_units, "in");
    assert!((style.convert.apply(25.4) - 1.0).abs() < 1e-6);
}
