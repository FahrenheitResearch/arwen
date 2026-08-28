//! Offline tests of the region culler's phases and conventions.
//!
//! The decision-heavy surface -- nearest-cell descent, boundary walk, flood
//! fill, ring growth -- is graded against a committed fixture DERIVED FROM THE
//! NATIVE CULL: `tests/goldens/x1.window.cull.json` records, for a polygon
//! window on the published x1.40962 mesh, exactly which parent cells the
//! native MPAS-Limited-Area v2.2 tool marked and with which mask values. The
//! cell topology comes from the x1 golden container already in the tree, so
//! the whole thing runs in an offline `cargo test --locked --offline`.
//!
//! The full-file byte-identity grade against the native culls of both pinned
//! parents lives in `tests/mesh_cull_oracle.rs` and needs the oracle
//! artifacts on disk.

mod common;

use rw_mpas::mesh::cull::{
    CellTopo, boundary_from_shape, edge_masks_from_cells, mark_region_cells, renumber_map,
    vertex_masks_from_cells,
};
use rw_mpas::mesh::density::Shape;

/// The x1 golden's cell arrays, reshaped to the row-major padded layout the
/// culler walks (1-based values, exactly how a grid file stores them).
struct GoldenCells {
    n_cells: usize,
    max_edges: usize,
    lat: Vec<f64>,
    lon: Vec<f64>,
    neoc: Vec<i32>,
    coc: Vec<i32>,
}

fn golden_cells() -> GoldenCells {
    let g = common::load_mesh("x1.40962");
    let lat = g.f64("cell/latCell");
    let lon = g.f64("cell/lonCell");
    let neoc = g.i32("cell/nEdgesOnCell");
    let offsets = g.i32("cell/cellsOnCell_offsets");
    let values = g.i32("cell/cellsOnCell_values");
    let n_cells = lat.len();
    assert_eq!(neoc.len(), n_cells);
    assert_eq!(offsets.len(), n_cells + 1);
    let max_edges = neoc.iter().copied().max().unwrap() as usize;
    let mut coc = vec![0i32; n_cells * max_edges];
    for c in 0..n_cells {
        let lo = offsets[c] as usize;
        let hi = offsets[c + 1] as usize;
        assert_eq!(hi - lo, neoc[c] as usize, "cell {c}: CSR row length");
        for (s, &v) in values[lo..hi].iter().enumerate() {
            // Golden values are 0-based; the file layout is 1-based.
            coc[c * max_edges + s] = v + 1;
        }
    }
    GoldenCells {
        n_cells,
        max_edges,
        lat,
        lon,
        neoc,
        coc,
    }
}

#[derive(serde::Deserialize)]
struct WindowFixture {
    provenance: String,
    region: Shape,
    /// `[parent_index0, output_mask]` for every marked cell, parent order.
    expected_cells: Vec<[i64; 2]>,
}

fn window_fixture() -> WindowFixture {
    let path = common::golden_dir().join("x1.window.cull.json");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "the native-derived window fixture {} is missing ({e}); without it \
             the mark phase is graded against nothing",
            path.display()
        )
    });
    serde_json::from_str(&text).expect("x1.window.cull.json parses")
}

/// The mark phase reproduces the native tool's marked set EXACTLY -- same
/// cells, same masks -- on the published x1 mesh.
#[test]
fn the_mark_phase_reproduces_the_native_cull_cell_for_cell() {
    let g = golden_cells();
    let fx = window_fixture();
    assert!(
        fx.provenance.contains("MPAS-Limited-Area"),
        "fixture provenance does not name the native tool: {}",
        fx.provenance
    );
    let topo = CellTopo {
        n_cells: g.n_cells,
        max_edges: g.max_edges,
        lat: &g.lat,
        lon: &g.lon,
        n_edges_on_cell: &g.neoc,
        cells_on_cell: &g.coc,
    };
    let (loops, seed) = boundary_from_shape(&fx.region).expect("region row yields a boundary");
    let (mask, stats) = mark_region_cells(&topo, &loops, seed).expect("mark phase completes");

    let got: Vec<[i64; 2]> = (0..g.n_cells)
        .filter(|&i| mask[i] != 0)
        .map(|i| [i as i64, (mask[i] - 1) as i64])
        .collect();
    assert_eq!(
        got.len(),
        fx.expected_cells.len(),
        "marked {} cells where the native tool marked {} (ring counts {:?})",
        got.len(),
        fx.expected_cells.len(),
        stats.ring_cell_counts
    );
    for (k, (g_row, e_row)) in got.iter().zip(&fx.expected_cells).enumerate() {
        assert_eq!(
            g_row, e_row,
            "marked-cell row {k} diverges from the native cull (got [index0, mask] {g_row:?}, native {e_row:?})"
        );
    }
}

/// The measured native histogram for the fixture window: interior plus seven
/// rings whose counts grow outward.
#[test]
fn the_ring_histogram_matches_the_native_cull() {
    let g = golden_cells();
    let fx = window_fixture();
    let topo = CellTopo {
        n_cells: g.n_cells,
        max_edges: g.max_edges,
        lat: &g.lat,
        lon: &g.lon,
        n_edges_on_cell: &g.neoc,
        cells_on_cell: &g.coc,
    };
    let (loops, seed) = boundary_from_shape(&fx.region).unwrap();
    let (_, stats) = mark_region_cells(&topo, &loops, seed).unwrap();

    let mut expected = [0usize; 8];
    for row in &fx.expected_cells {
        expected[row[1] as usize] += 1;
    }
    assert_eq!(
        stats.ring_cell_counts, expected,
        "ring histogram diverged from the native cull's"
    );
}

// ---------------------------------------------------------------------------
// Pure-phase conventions, on handcrafted arrays
// ---------------------------------------------------------------------------

/// Edge mask: MIN of two marked cells, else MAX -- so an interior edge takes
/// the inner ring and a rim edge takes its only marked cell's value.
#[test]
fn edge_masks_are_min_of_marked_else_max() {
    // Cells: masks 3, 4, 0 (internal values; cell 3 is outside).
    let cell_mask = vec![3, 4, 0];
    // Edges: (1,2) both marked -> min 3; (2,3) one marked -> max 4; (3,3)
    // hypothetical outside pair -> 0.
    let cells_on_edge = vec![1, 2, 2, 3, 3, 3];
    let masks = edge_masks_from_cells(&cell_mask, &cells_on_edge, 3).unwrap();
    assert_eq!(masks, vec![3, 4, 0]);
}

/// A cellsOnEdge slot of 0 is refused BY NAME: the native tool would read the
/// last cell's mask there, which is no rule at all.
#[test]
fn a_zero_cells_on_edge_slot_is_refused_with_the_breakage_named() {
    let err = edge_masks_from_cells(&[1, 1], &[1, 0], 1).unwrap_err().to_string();
    assert!(
        err.contains("LAST cell's mask"),
        "refusal does not name the native wrap it prevents: {err}"
    );
}

/// Vertex mask: same rule over vertexDegree adjacent cells.
#[test]
fn vertex_masks_are_min_of_marked_else_max() {
    let cell_mask = vec![2, 5, 0];
    // Vertex 0 touches (1,2,2): min 2. Vertex 1 touches (2,3,3): one marked -> max 5.
    let cells_on_vertex = vec![1, 2, 2, 2, 3, 3];
    let masks = vertex_masks_from_cells(&cell_mask, &cells_on_vertex, 2, 3).unwrap();
    assert_eq!(masks, vec![2, 5]);
}

/// The renumber map is 1..K over marked elements in PARENT order and 0
/// elsewhere -- the map the connectivity rewrite and the `indexTo*ID`
/// contiguity both come from.
#[test]
fn the_renumber_map_is_parent_ordered_and_contiguous() {
    let map = renumber_map(&[0, 3, 0, 1, 8, 0, 2]);
    assert_eq!(map, vec![0, 1, 0, 2, 3, 0, 4]);
}

/// A polygon row's boundary loop is its vertices verbatim, in radians, and
/// the seed sits strictly inside it.
#[test]
fn a_polygon_row_walks_its_vertices_verbatim() {
    let shape = Shape::Polygon {
        vertices_deg: vec![[50.0, -129.0], [50.0, -65.0], [20.0, -65.0], [20.0, -129.0]],
    };
    let (loops, seed) = boundary_from_shape(&shape).unwrap();
    assert_eq!(loops.len(), 1);
    assert_eq!(loops[0].len(), 4);
    assert!((loops[0][0].0 - 50f64.to_radians()).abs() < 1e-15);
    assert!((loops[0][0].1 - (-129f64).to_radians()).abs() < 1e-15);
    // The seed must be inside the window (signed distance negative).
    let p = rw_mpas::mesh::geom::from_lat_lon(seed.0, seed.1);
    assert!(
        shape.signed_distance(p) < 0.0,
        "interior seed is not inside the polygon"
    );
}

/// A cap row's traced circle carries the native tool's endpoint duplication:
/// numpy linspace(0, 2*pi, 100) begins and ends at the same point.
#[test]
fn a_cap_row_traces_a_closed_circle_of_100_points() {
    let shape = Shape::Cap {
        center_deg: [39.0, -98.0],
        radius_km: 1200.0,
    };
    let (loops, _) = boundary_from_shape(&shape).unwrap();
    assert_eq!(loops.len(), 1);
    assert_eq!(loops[0].len(), 100);
    let first = loops[0][0];
    let last = loops[0][99];
    assert!(
        (first.0 - last.0).abs() < 1e-12 && (first.1 - last.1).abs() < 1e-12,
        "circle endpoints diverge: {first:?} vs {last:?}"
    );
}

/// A region that swallows the whole parent is refused rather than written.
#[test]
fn a_region_covering_every_cell_is_refused() {
    let g = golden_cells();
    let topo = CellTopo {
        n_cells: g.n_cells,
        max_edges: g.max_edges,
        lat: &g.lat,
        lon: &g.lon,
        n_edges_on_cell: &g.neoc,
        cells_on_cell: &g.coc,
    };
    // A cap of nearly the whole sphere: every cell ends up marked (interior
    // or ring), which the file-level cull refuses. Here the mark phase is
    // asked directly and must mark everything, proving the refusal upstream
    // has something to refuse.
    let shape = Shape::Cap {
        center_deg: [0.0, 0.0],
        radius_km: 19_600.0,
    };
    let (loops, seed) = boundary_from_shape(&shape).unwrap();
    let (mask, _) = mark_region_cells(&topo, &loops, seed).unwrap();
    assert_eq!(
        mask.iter().filter(|&&m| m != 0).count(),
        g.n_cells,
        "a near-global cap left cells unmarked; the whole-sphere refusal would \
         never fire and a mis-specified region would write a broken file"
    );
}
