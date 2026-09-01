//! The published well-centredness claim, measured on the published meshes.
//!
//! Unstructured-mesh papers lead with two quality numbers this crate could not
//! answer until 2026-08-31: the count of OBTUSE Delaunay triangles, and whether
//! every Voronoi edge crosses exactly one Delaunay edge. `mesh::validate`
//! checks edge handedness and non-orthogonality, and NEITHER of those sees an
//! obtuse triangle -- it keeps its handedness and can sit inside the
//! orthogonality bound -- so a mesh could carry hundreds and validate clean.
//!
//! This file is the baseline that makes the new reading meaningful. It runs the
//! instrument on the two published NCAR meshes, whose numbers are the reference
//! every other reading is judged against, and it tests the instrument in BOTH
//! directions: the quasi-uniform mesh must read zero, the variable-resolution
//! mesh must read the three it actually carries, and a mesh deliberately given
//! an obtuse triangle must be caught. An instrument only ever shown to say
//! "clean" has not been shown to work.

mod common;

use common::{MESHES, load_mesh};
use rw_mpas::mesh::derive::{MpasMesh, Rings};
use rw_mpas::mesh::geom::{V3, unit};
use rw_mpas::mesh::validate::well_centredness;

fn published(tag: &str) -> MpasMesh {
    let g = load_mesh(tag);
    let x = g.f64("in/xCell");
    let y = g.f64("in/yCell");
    let z = g.f64("in/zCell");
    let cell_xyz: Vec<V3> = (0..x.len()).map(|i| [x[i], y[i], z[i]]).collect();
    let density = g.f64("in/meshDensity");
    let rings = Rings {
        offsets: g
            .i32("cell/cellsOnCell_offsets")
            .iter()
            .map(|&v| v as u32)
            .collect(),
        values: g
            .i32("cell/cellsOnCell_values")
            .iter()
            .map(|&v| v as u32)
            .collect(),
    };
    let nominal = g.f64("meta/nominalMinDc")[0];
    MpasMesh::derive(cell_xyz, density, &rings, nominal)
        .unwrap_or_else(|e| panic!("{tag}: derive refused the published mesh: {e}"))
}

/// THE BASELINE. Both published meshes, both directions of the instrument.
///
/// MEASURED 2026-08-31 on this tree, and these are the numbers every other
/// well-centredness reading in the project is quoted against:
///
/// | mesh | cells | obtuse | dual edges not crossing | max angle |
/// |---|---|---|---|---|
/// | x1.40962 (quasi-uniform, 120 km) | 40,962 | 0 | 0 | 72.0000 deg |
/// | x4.163842 (variable, 60 to 3 km) | 163,842 | 3 | 3 | 94.6798 deg |
///
/// The x4 row is the one worth stating plainly: NCAR's own published
/// variable-resolution mesh is NOT well-centred. That is why this metric is
/// REPORTED rather than gated -- a gate at zero would refuse the reference
/// mesh this crate validates itself against.
#[test]
fn the_two_published_meshes_read_their_measured_baseline() {
    for tag in MESHES {
        let mesh = published(tag);
        let wc = well_centredness(&mesh);
        eprintln!(
            "{tag}: {} cells, {} obtuse, {} dual edges not crossing, max angle {:.4} deg",
            mesh.n_cells, wc.obtuse_triangles, wc.non_crossing_dual_edges, wc.max_delaunay_angle_deg
        );
        let (want_obtuse, want_angle) = match tag {
            "x1.40962" => (0usize, 72.0),
            "x4.163842" => (3usize, 94.6798),
            other => panic!("no baseline recorded for {other}"),
        };
        assert_eq!(
            wc.obtuse_triangles, want_obtuse,
            "{tag} moved off its measured obtuse-triangle baseline"
        );
        assert!(
            (wc.max_delaunay_angle_deg - want_angle).abs() < 5e-4,
            "{tag} largest Delaunay angle {:.4} deg against a baseline of {want_angle:.4}",
            wc.max_delaunay_angle_deg
        );
    }
}

/// The two counts are ONE geometric fact: a triangle is obtuse exactly when its
/// circumcentre falls outside it, which is exactly when the dual edge opposite
/// the obtuse angle fails to cross its primal edge. They are computed by
/// completely separate arithmetic -- interior angles against segment-plane
/// straddling -- so their agreement is a check on both.
#[test]
fn obtuse_triangles_and_non_crossing_dual_edges_are_the_same_count() {
    for tag in MESHES {
        let wc = well_centredness(&published(tag));
        assert_eq!(
            wc.obtuse_triangles, wc.non_crossing_dual_edges,
            "{tag}: the two instruments disagree, so at least one is wrong"
        );
    }
}

/// THE INSTRUMENT FAILS WHEN IT SHOULD. A published mesh with one generator
/// dragged towards its neighbours' edge grows an obtuse triangle, and the
/// reading must move off the baseline. Without this, "0 obtuse" is not
/// evidence of a well-centred mesh -- it is equally consistent with an
/// instrument that always answers zero.
#[test]
fn a_deliberately_flattened_triangle_is_caught() {
    let g = load_mesh("x1.40962");
    let x = g.f64("in/xCell");
    let y = g.f64("in/yCell");
    let z = g.f64("in/zCell");
    let mut cell_xyz: Vec<V3> = (0..x.len()).map(|i| [x[i], y[i], z[i]]).collect();
    let density = g.f64("in/meshDensity");
    let rings = Rings {
        offsets: g
            .i32("cell/cellsOnCell_offsets")
            .iter()
            .map(|&v| v as u32)
            .collect(),
        values: g
            .i32("cell/cellsOnCell_values")
            .iter()
            .map(|&v| v as u32)
            .collect(),
    };
    let nominal = g.f64("meta/nominalMinDc")[0];

    // Cell 0 and two of its ring neighbours make a Delaunay triangle. Slide
    // cell 0 most of the way onto the chord between the other two: the angle
    // at cell 0 opens past a right angle and its circumcentre leaves the
    // triangle.
    let a = rings.values[rings.offsets[0] as usize] as usize;
    let b = rings.values[rings.offsets[0] as usize + 1] as usize;
    let chord_mid = unit([
        cell_xyz[a][0] + cell_xyz[b][0],
        cell_xyz[a][1] + cell_xyz[b][1],
        cell_xyz[a][2] + cell_xyz[b][2],
    ])
    .expect("chord midpoint");
    let p = cell_xyz[0];
    cell_xyz[0] = unit([
        0.02 * p[0] + 0.98 * chord_mid[0],
        0.02 * p[1] + 0.98 * chord_mid[1],
        0.02 * p[2] + 0.98 * chord_mid[2],
    ])
    .expect("flattened generator");

    let mesh = MpasMesh::derive(cell_xyz, density, &rings, nominal)
        .expect("derive on the perturbed centres");
    let wc = well_centredness(&mesh);
    eprintln!(
        "flattened x1.40962: {} obtuse, {} not crossing, max angle {:.4} deg",
        wc.obtuse_triangles, wc.non_crossing_dual_edges, wc.max_delaunay_angle_deg
    );
    assert!(
        wc.obtuse_triangles > 0,
        "a generator dragged onto its neighbours' chord produced no obtuse triangle: the instrument cannot fail"
    );
    assert!(
        wc.max_delaunay_angle_deg > 90.0,
        "largest angle {:.4} deg is not obtuse",
        wc.max_delaunay_angle_deg
    );
}
