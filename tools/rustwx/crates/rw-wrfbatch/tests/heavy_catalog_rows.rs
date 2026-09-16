//! Every excluded heavy (ECAPE) catalog row has to name its OWN missing
//! grid and its OWN way out.
//!
//! The concrete breakage this prevents: `--list-products` built the whole
//! heavy family's excluded rows from one string literal, so the three
//! `*_ecape_native_cape_ratio` pairs -- which divide by the source model's
//! own decoded CAPE plane and can never come off a wrfout, a permanent
//! exclusion -- printed the same sentence as the parcel grids a `--heavy`
//! import does produce. A reader of the catalog could not tell which rows
//! were worth re-running for, and the one cause the sentence named was the
//! wrong one for most of the family.
//!
//! This goes to the BINARY's own catalog transcript rather than to a
//! helper, because the transcript is what a caller reads
//! (`gpuwm.rustwx.list_products` parses exactly these `PRODUCT` lines).

mod stored_plane_fixture;

use std::path::{Path, PathBuf};
use std::process::Command;

struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Self {
        let dir = std::env::temp_dir().join(format!(
            "rw-wrfbatch-heavy-rows-{tag}-{}-{:?}",
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

/// `(slug, kind, status, detail)` for every `PRODUCT` row the catalog
/// transcript carries.
fn catalog_rows(heavy: bool) -> Vec<(String, String, String, String)> {
    let scratch = Scratch::new(if heavy { "heavy" } else { "light" });
    let wrfout = stored_plane_fixture::write(scratch.path());
    let store_root = scratch.path().join("store");
    let out_dir = scratch.path().join("out");
    let mut command = Command::new(env!("CARGO_BIN_EXE_rw_wrfbatch"));
    command
        .arg("--store-root")
        .arg(&store_root)
        .arg("--out-dir")
        .arg(&out_dir)
        .arg("--list-products");
    if heavy {
        command.arg("--heavy");
    }
    command.arg(&wrfout);
    let output = command.output().expect("the renderer runs");
    let transcript = String::from_utf8_lossy(&output.stdout).to_string();
    assert!(
        output.status.success(),
        "--list-products failed: {}\n{transcript}",
        String::from_utf8_lossy(&output.stderr)
    );
    let rows: Vec<(String, String, String, String)> = transcript
        .lines()
        .filter_map(|line| line.strip_prefix("PRODUCT\t"))
        .filter_map(|line| {
            let parts: Vec<&str> = line.split('\t').collect();
            // (slug, kind, status, detail, code): the fifth column is the
            // machine code beside the prose; these tests read the prose.
            (parts.len() == 5).then(|| {
                (
                    parts[0].to_string(),
                    parts[1].to_string(),
                    parts[2].to_string(),
                    parts[3].to_string(),
                )
            })
        })
        .collect();
    assert!(!rows.is_empty(), "no catalog rows: {transcript}");
    rows
}

#[test]
fn every_excluded_heavy_row_names_its_own_grid_and_its_own_way_out() {
    for heavy in [false, true] {
        let rows = catalog_rows(heavy);
        let excluded: Vec<(String, String)> = rows
            .iter()
            .filter(|(_, kind, status, _)| kind == "heavy" && status == "excluded")
            .map(|(slug, _, _, detail)| (slug.clone(), detail.clone()))
            .collect();
        assert!(
            !excluded.is_empty(),
            "heavy={heavy}: no excluded heavy rows to check"
        );
        let details: Vec<&str> = excluded.iter().map(|(_, detail)| detail.as_str()).collect();
        let distinct: std::collections::BTreeSet<&str> = details.iter().copied().collect();
        assert_eq!(
            distinct.len(),
            details.len(),
            "heavy={heavy}: {} excluded heavy row(s) share {} distinct reason(s); \
             one blanket sentence cannot be true of all of them",
            details.len(),
            distinct.len()
        );
        for (slug, detail) in &excluded {
            assert!(
                detail.contains(slug.as_str()),
                "heavy={heavy}: the row must name the grid it is missing: {slug} -> {detail}"
            );
            assert!(
                detail.contains("--heavy") || detail.contains("GRIB"),
                "heavy={heavy}: the row must name a way out: {slug} -> {detail}"
            );
        }
    }
}

/// The permanent exclusion and the per-hour gap must not read alike: the
/// native-CAPE ratio pairs point at the route that CAN compute them and
/// must not promise that a re-import will.
#[test]
fn the_native_cape_ratio_rows_are_a_named_permanent_exclusion() {
    let rows = catalog_rows(true);
    for slug in [
        "sb_ecape_native_cape_ratio",
        "ml_ecape_native_cape_ratio",
        "mu_ecape_native_cape_ratio",
    ] {
        let (_, _, status, detail) = rows
            .iter()
            .find(|(row_slug, _, _, _)| row_slug == slug)
            .unwrap_or_else(|| panic!("{slug} is not in the catalog"));
        assert_eq!(status, "excluded", "{slug}: {detail}");
        assert!(detail.contains("GRIB"), "{slug}: {detail}");
        assert!(
            detail.contains("native"),
            "{slug} must name the plane it divides by: {detail}"
        );
        assert!(
            !detail.contains("Re-import"),
            "{slug} is permanent here and must not promise a re-import: {detail}"
        );
    }
}

/// The other half of the same defect: the grids those rows were excusing.
/// wrf-core exposes no ml/mu parcel ECAPE diagnostic and no ECAPE /
/// derived-CAPE ratio at all, so on this lane those six products reached
/// no panel while running normally on the GRIB one. A `--heavy` import now
/// assembles this hour's own surface planes and isobaric volumes into the
/// products-side input pair and runs the SHARED heavy recipes over them,
/// so the catalog offers them.
#[test]
fn a_heavy_import_realizes_the_ml_and_mu_parcel_family_and_the_derived_ratios() {
    let rows = catalog_rows(true);
    for slug in [
        "mlecape",
        "muecape",
        "mlecin",
        "sb_ecape_derived_cape_ratio",
        "ml_ecape_derived_cape_ratio",
        "mu_ecape_derived_cape_ratio",
    ] {
        let (_, kind, status, detail) = rows
            .iter()
            .find(|(row_slug, _, _, _)| row_slug == slug)
            .unwrap_or_else(|| panic!("{slug} is not in the catalog"));
        assert_eq!(kind, "heavy", "{slug}");
        assert_eq!(
            status, "renderable",
            "a --heavy wrfout import must realize {slug}: {detail}"
        );
    }
}
