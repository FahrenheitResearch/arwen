//! The renderer's raw-plane catalog agrees with the engine's precipitation
//! inventory.
//!
//! `RAW_EXTRA_CATALOG` is a hand-written list; the engine writes every
//! accumulator in `gpuwm/io/wrf_output_schema.py::PRECIPITATION_OUTPUT_FIELDS`
//! for every run, and per scheme the physics registry knows which of them a
//! scheme actually fills.  Nothing tied the two together, so HAILNC -- filled
//! by Milbrandt-Yau (mp=9) and NSSL-2 (mp=18) -- reached the file and never
//! the curated catalog or the QPF palette; it has both now (audit R-053) and
//! this test is what keeps the next one from being dropped.  The engine
//! exports the inventory
//! it derives from the registry as `gpuwm/physics_consumer_export_v1.json`
//! (`tools/build_registry.py`, byte-pinned by the engine's own tests), and
//! this test reads it: every scheme-filled accumulator is either in the
//! catalog or named below with the defect that keeps it out.  A citation
//! that stops being true fails here too, which is the retirement sweep.

use std::path::Path;

use rw_wrfbatch::wrf_process::RAW_EXTRA_CATALOG;

/// Scheme-filled accumulators deliberately outside the raw catalog, each
/// with the reason.
const CITED_ABSENCES: &[(&str, &str)] = &[
    (
        "RAINNC",
        "grid-scale rain is consumed by the wrf-core precipitation diagnostics \
         (total / hourly precipitation products), not offered as a raw plane",
    ),
];

fn export() -> serde_json::Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../../gpuwm/physics_consumer_export_v1.json");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|error| {
        panic!(
            "{} is the engine's consumer export and this test cannot run without it: {error}",
            path.display()
        )
    });
    let value: serde_json::Value = serde_json::from_str(&text).expect("consumer export parses");
    assert_eq!(
        value["schema"].as_str(),
        Some("gpuwm-physics-consumer-export-v1"),
        "unexpected consumer export schema"
    );
    value
}

#[test]
fn every_scheme_filled_accumulator_is_catalogued_or_cited() {
    let export = export();
    let fields = export["scheme_bound_precipitation_fields"]
        .as_object()
        .expect("scheme_bound_precipitation_fields is an object");
    let mut problems = Vec::new();
    for (name, schemes) in fields {
        let filled_by = schemes.as_array().expect("scheme list");
        let in_catalog = RAW_EXTRA_CATALOG.contains(&name.as_str());
        let cited = CITED_ABSENCES.iter().find(|(field, _)| field == name);
        if filled_by.is_empty() {
            // Nothing fills it (RAINC/RAINSH are cumulus accumulators, not
            // microphysics ones); the catalog owes it nothing here.
            continue;
        }
        match (in_catalog, cited) {
            (true, Some((_, reason))) => problems.push(format!(
                "{name} is in RAW_EXTRA_CATALOG and still cited as absent ({reason}); retire the citation"
            )),
            (false, None) => problems.push(format!(
                "{name} is filled by mp_physics {filled_by:?} and has neither a RAW_EXTRA_CATALOG row nor a cited reason"
            )),
            _ => {}
        }
    }
    for (field, _) in CITED_ABSENCES {
        if !fields.contains_key(*field) {
            problems.push(format!(
                "{field} is cited as absent but the engine export names no such accumulator; retire the citation"
            ));
        }
    }
    assert!(
        problems.is_empty(),
        "RAW_EXTRA_CATALOG disagrees with the engine's precipitation inventory:\n  {}",
        problems.join("\n  ")
    );
}

#[test]
fn the_three_catalogued_accumulators_are_the_engines() {
    let export = export();
    let names: Vec<&str> = export["precipitation_output_fields"]
        .as_array()
        .expect("precipitation_output_fields")
        .iter()
        .map(|value| value.as_str().expect("field name"))
        .collect();
    for catalogued in ["SNOWNC", "GRAUPELNC", "HAILNC"] {
        assert!(RAW_EXTRA_CATALOG.contains(&catalogued));
        assert!(
            names.contains(&catalogued),
            "{catalogued} is catalogued but the engine no longer writes it"
        );
    }
}
