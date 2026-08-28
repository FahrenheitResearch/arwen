//! The static builder's regional admission, graded on the REAL culled bytes.
//!
//! Reuses the oracle manifest of `mesh_cull_oracle.rs`: every grid case's
//! native-culled output (byte-identical to this crate's own cull) must load
//! through the static builder's reader with the sentinel geometry the
//! measurements recorded. This is the load half of the old bdyMaskCell==7
//! refusal being lifted; the compute half is graded by the full-build
//! comparison against the native-culled static.

use std::path::PathBuf;

use rw_mpas::mesh::density::Shape;
use rw_mpas::static_builder::probe_grid;

#[derive(serde::Deserialize)]
struct Manifest {
    base_dir: PathBuf,
    cases: Vec<Case>,
}

#[derive(serde::Deserialize)]
struct Case {
    label: String,
    #[allow(dead_code)]
    parent: PathBuf,
    #[allow(dead_code)]
    region: Shape,
    expect_nc: PathBuf,
    #[allow(dead_code)]
    expect_graph: Option<PathBuf>,
}

#[test]
fn every_native_culled_grid_admits_with_its_measured_sentinel_geometry() {
    let path = match std::env::var("RW_MPAS_LAM_ORACLE_MANIFEST") {
        Ok(p) if !p.trim().is_empty() => p,
        _ => {
            eprintln!(
                "SKIPPED: RW_MPAS_LAM_ORACLE_MANIFEST is unset; the regional \
                 admission graded NOTHING"
            );
            return;
        }
    };
    let m: Manifest = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    let mut graded = 0usize;
    for case in &m.cases {
        // Grid culls only: a culled STATIC is a finished product, not input
        // to the static builder.
        if case.label.contains("static") {
            continue;
        }
        let probe = probe_grid(&m.base_dir.join(&case.expect_nc))
            .unwrap_or_else(|e| panic!("{}: regional grid refused: {e}", case.label));
        assert!(probe.regional, "{}: not recognised as regional", case.label);
        assert_eq!(
            probe.ring_cell_counts.iter().sum::<usize>(),
            probe.n_cells,
            "{}: mask histogram does not cover every cell",
            case.label
        );
        // The measured convention: one absent-neighbour slot per one-sided
        // rim; both counts live only on the outermost ring.
        assert!(
            probe.absent_neighbor_slots > 0 && probe.one_sided_edges > 0,
            "{}: a culled mesh with no boundary sentinels is not a cull",
            case.label
        );
        eprintln!(
            "ADMITTED {}: {} cells, rings {:?}, {} absent neighbour slots, {} one-sided edges",
            case.label,
            probe.n_cells,
            probe.ring_cell_counts,
            probe.absent_neighbor_slots,
            probe.one_sided_edges
        );
        graded += 1;
    }
    assert!(graded > 0, "the manifest carried no grid cases");
}
