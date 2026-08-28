//! Byte-identity grade of the region culler against the native
//! MPAS-Limited-Area v2.2 reference culls.
//!
//! The oracle artifacts (pinned global parents and their native culls) are
//! hundreds of megabytes and live outside the tree. `RW_MPAS_LAM_ORACLE_MANIFEST`
//! names a JSON manifest describing them:
//!
//! ```json
//! {
//!   "base_dir": "…",
//!   "cases": [
//!     {"label": "x1-quick",
//!      "parent": "parents/x1.40962.grid.nc",
//!      "region": {"kind": "polygon", "vertices_deg": [[50,-129],[50,-65],[20,-65],[20,-129]]},
//!      "expect_nc": "…", "expect_graph": "…"}
//!   ]
//! }
//! ```
//!
//! With the variable unset the tests state loudly that they graded nothing
//! and pass, so an offline `cargo test` stays green while a release grade can
//! demand the real thing.

use std::path::{Path, PathBuf};

use rw_mpas::mesh::cull::cull_file;
use rw_mpas::mesh::density::Shape;

#[derive(serde::Deserialize)]
struct Manifest {
    base_dir: PathBuf,
    cases: Vec<Case>,
}

#[derive(serde::Deserialize)]
struct Case {
    label: String,
    parent: PathBuf,
    region: Shape,
    expect_nc: PathBuf,
    expect_graph: Option<PathBuf>,
}

fn manifest() -> Option<Manifest> {
    let path = match std::env::var("RW_MPAS_LAM_ORACLE_MANIFEST") {
        Ok(p) if !p.trim().is_empty() => p,
        _ => {
            eprintln!(
                "SKIPPED: RW_MPAS_LAM_ORACLE_MANIFEST is unset; the byte-identity \
                 grade against the native culls ran against NOTHING"
            );
            return None;
        }
    };
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("oracle manifest {path} could not be read: {e}"));
    Some(serde_json::from_str(&text).unwrap_or_else(|e| panic!("oracle manifest {path}: {e}")))
}

fn first_difference(a: &[u8], b: &[u8]) -> Option<usize> {
    if a == b {
        return None;
    }
    Some(
        a.iter()
            .zip(b.iter())
            .position(|(x, y)| x != y)
            .unwrap_or_else(|| a.len().min(b.len())),
    )
}

/// Every manifest case: cull the parent with the region row and demand byte
/// identity with the native output, netCDF and graph file both.
#[test]
fn the_rust_cull_byte_matches_the_native_cull_on_every_manifest_case() {
    let Some(m) = manifest() else { return };
    assert!(!m.cases.is_empty(), "oracle manifest lists no cases");
    let tmp = std::env::temp_dir().join(format!(
        "rw-mpas-cull-oracle-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&tmp).unwrap();

    for case in &m.cases {
        let parent = m.base_dir.join(&case.parent);
        let expect_nc = m.base_dir.join(&case.expect_nc);
        let out_nc = tmp.join(format!("{}.region.nc", case.label));
        let out_graph = tmp.join(format!("{}.graph.info", case.label));

        let receipt = cull_file(
            &parent,
            &case.region,
            &out_nc,
            case.expect_graph.as_ref().map(|_| out_graph.as_path()),
        )
        .unwrap_or_else(|e| panic!("{}: cull refused: {e}", case.label));

        assert_eq!(
            receipt.euler_v_minus_e_plus_f, 1,
            "{}: the region is not a disk",
            case.label
        );
        assert_eq!(
            receipt.native_wrap_divergence,
            [false, false, false],
            "{}: the parent's last element is inside the region, so the native \
             tool's numpy wrap makes byte identity impossible by construction; \
             this grade cannot be run on that parent",
            case.label
        );

        let got = std::fs::read(&out_nc).unwrap();
        let want = std::fs::read(&expect_nc)
            .unwrap_or_else(|e| panic!("{}: oracle {} unreadable: {e}", case.label, expect_nc.display()));
        if let Some(off) = first_difference(&got, &want) {
            panic!(
                "{}: regional netCDF diverges from the native cull: {} vs {} \
                 bytes, first difference at offset 0x{off:x}",
                case.label,
                got.len(),
                want.len()
            );
        }

        if let Some(eg) = &case.expect_graph {
            let expect_graph = m.base_dir.join(eg);
            let got = std::fs::read(&out_graph).unwrap();
            let want = std::fs::read(&expect_graph).unwrap();
            if let Some(off) = first_difference(&got, &want) {
                panic!(
                    "{}: graph.info diverges from the native tool's: {} vs {} \
                     bytes, first difference at offset 0x{off:x}",
                    case.label,
                    got.len(),
                    want.len()
                );
            }
        }
        eprintln!(
            "GRADED {}: {} cells / {} edges / {} vertices, byte-identical (sha256 {})",
            case.label,
            receipt.region_cells,
            receipt.region_edges,
            receipt.region_vertices,
            receipt.output_sha256
        );
    }
    let _ = std::fs::remove_dir_all(&tmp);
}

/// Fixture minter for the offline conventions test: derives
/// `tests/goldens/x1.window.cull.json` FROM THE NATIVE CULL by matching the
/// oracle's cell coordinates back to parent positions bit-for-bit. Run once,
/// by hand, with the oracle manifest set:
///
/// ```text
/// cargo test -p rw-mpas --test mesh_cull_oracle mint_window_fixture -- --ignored
/// ```
#[test]
#[ignore = "writes the committed fixture from the oracle; run by hand"]
fn mint_window_fixture_from_the_native_cull() {
    let m = manifest().expect("fixture minting needs the oracle manifest");
    let case = m
        .cases
        .iter()
        .find(|c| c.label == "x1-quick")
        .expect("manifest has no x1-quick case");
    let parent = netcrust::File::open(m.base_dir.join(&case.parent)).unwrap();
    let oracle = netcrust::File::open(m.base_dir.join(&case.expect_nc)).unwrap();

    let key = |lat: f64, lon: f64| (lat.to_bits(), lon.to_bits());
    let p_lat = parent.read_f64("latCell").unwrap();
    let p_lon = parent.read_f64("lonCell").unwrap();
    let mut by_coord = std::collections::HashMap::with_capacity(p_lat.len());
    for i in 0..p_lat.len() {
        assert!(
            by_coord.insert(key(p_lat[i], p_lon[i]), i).is_none(),
            "parent cell coordinates are not unique; coordinate matching is unusable"
        );
    }
    let o_lat = oracle.read_f64("latCell").unwrap();
    let o_lon = oracle.read_f64("lonCell").unwrap();
    let o_mask: Vec<i64> = oracle
        .read_f64("bdyMaskCell")
        .unwrap()
        .into_iter()
        .map(|v| v as i64)
        .collect();

    let mut rows: Vec<[i64; 2]> = Vec::with_capacity(o_lat.len());
    for k in 0..o_lat.len() {
        let idx = *by_coord
            .get(&key(o_lat[k], o_lon[k]))
            .unwrap_or_else(|| panic!("oracle cell {k} has no bit-identical parent coordinate"));
        rows.push([idx as i64, o_mask[k]]);
    }
    // The native cull preserves parent order; the fixture states it.
    for w in rows.windows(2) {
        assert!(w[0][0] < w[1][0], "oracle cells are not in parent order");
    }

    let fixture = serde_json::json!({
        "provenance": "derived from the native MPAS-Limited-Area v2.2 cull of the \
                       published x1.40962 mesh (regional oracle record set, 2026-08-25); \
                       [parent_index0, bdyMaskCell] per kept cell, parent order",
        "region": case.region,
        "expected_cells": rows,
    });
    let out = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("goldens")
        .join("x1.window.cull.json");
    std::fs::write(&out, serde_json::to_string(&fixture).unwrap()).unwrap();
    eprintln!("WROTE {} ({} cells)", out.display(), rows.len());
}
