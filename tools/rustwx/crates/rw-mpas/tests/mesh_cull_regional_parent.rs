//! A cull of a culled file is the cull of the whole sphere, or it is refused.
//!
//! The hex mesh door computes a region's static and vertical grid over a
//! wider cut of the global parent and then cuts that down to the region. The
//! second cut reads a REGIONAL parent, whose outermost ring stores 0 where a
//! neighbour was cut away. These tests hold the two promises the culler makes
//! about such a parent:
//!
//! * when the region and its seven rings stay inside the parent's outermost
//!   ring, the file and the graph are byte-identical to culling the whole
//!   sphere with the same region;
//! * when they reach that ring, the cut is refused by name, because the rings
//!   there would stop where the file stops.

use std::path::{Path, PathBuf};

use rw_mpas::mesh::cull::cull_file;
use rw_mpas::mesh::density::{MeshSpec, Shape};
use rw_mpas::mesh::emit::{Provenance, write_grid};
use rw_mpas::mesh::{GenerateRequest, LloydOptions, generate};

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("rw_mpas_cull_regional_parent_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("scratch directory");
    let path = dir.join(name);
    let _ = std::fs::remove_file(&path);
    path
}

/// Each test writes its own parent: the two run on parallel threads in one
/// process and would otherwise replace one file under each other.
fn global_grid(label: &str) -> PathBuf {
    let request = GenerateRequest {
        spec: MeshSpec::uniform(480.0),
        target_cells: Some(2562),
        lloyd: LloydOptions {
            tolerance: 3e-3,
            max_sweeps: 200,
            ..Default::default()
        },
        sizing_samples: 20_000,
        ..Default::default()
    };
    let out = generate(&request, |_| {}).unwrap_or_else(|e| panic!("generate refused: {e}"));
    let path = scratch(&format!("{label}.global.grid.nc"));
    write_grid(&out.mesh, &path, &Provenance::default(), true).expect("write the global grid");
    path
}

fn cap(lat: f64, lon: f64, radius_km: f64) -> Shape {
    Shape::Cap {
        center_deg: [lat, lon],
        radius_km,
    }
}

fn bytes(path: &Path) -> Vec<u8> {
    std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

#[test]
fn a_cut_of_a_wider_cut_is_byte_identical_to_the_cut_of_the_whole_sphere() {
    let global = global_grid("nested");
    for (label, lat, lon) in [("mid", 37.7, -122.3), ("pole", 82.0, 15.0), ("dateline", -18.0, 179.9)] {
        let wide = scratch(&format!("{label}.wide.nc"));
        cull_file(&global, &cap(lat, lon, 5000.0), &wide, None).expect("the wide cut of the sphere");
        let direct = scratch(&format!("{label}.direct.nc"));
        let direct_graph = scratch(&format!("{label}.direct.graph.info"));
        let direct_receipt = cull_file(&global, &cap(lat, lon, 1500.0), &direct, Some(&direct_graph))
            .expect("the region cut from the sphere");
        let nested = scratch(&format!("{label}.nested.nc"));
        let nested_graph = scratch(&format!("{label}.nested.graph.info"));
        let nested_receipt = cull_file(&wide, &cap(lat, lon, 1500.0), &nested, Some(&nested_graph))
            .expect("the region cut from the wide cut");
        assert!(direct_receipt.region_cells > 20, "{label}: the region holds too few cells to test anything");
        assert!(
            nested_receipt.parent_cells < direct_receipt.parent_cells,
            "{label}: the wide cut is not narrower than the sphere"
        );
        assert_eq!(
            bytes(&direct),
            bytes(&nested),
            "{label}: cutting the region from a wider cut wrote different bytes than cutting it from the sphere"
        );
        assert_eq!(
            bytes(&direct_graph),
            bytes(&nested_graph),
            "{label}: the graph of the nested cut differs from the graph of the direct cut"
        );
        assert_eq!(direct_receipt.mark.ring_cell_counts, nested_receipt.mark.ring_cell_counts);
    }
}

#[test]
fn a_cut_that_reaches_the_parent_rim_is_refused_by_name() {
    let global = global_grid("rim");
    let wide = scratch("rim.wide.nc");
    cull_file(&global, &cap(10.0, 40.0, 2500.0), &wide, None).expect("the wide cut");
    // The same centre: the region's own seven rings (about 400 km each here)
    // reach past the wide cut's outermost ring.
    let out = scratch("rim.nested.nc");
    let err = cull_file(&wide, &cap(10.0, 40.0, 2400.0), &out, None)
        .expect_err("a cut reaching the parent's outermost ring must be refused")
        .to_string();
    assert!(err.contains("outermost ring"), "the refusal does not name the rim: {err}");
    assert!(!out.exists(), "a refused cut wrote a file");
}
