//! The field emit graded against the known answer.
//!
//! Both published MPAS meshes' own cell centres and neighbour rings are fed
//! straight into `mesh::derive`, and every field it produces is compared with
//! the value the published file carries. Nothing here is generated: this is the
//! grader running before any generator exists, which is the only order in which
//! a generator's output means anything.
//!
//! Residuals are printed with their resolution limits. Run with
//! `cargo test --release -p rw-mpas --test mesh_derive -- --nocapture` to read
//! them.

mod common;

use common::{MESHES, Rwg, load_mesh};
use rw_mpas::mesh::derive::{MpasMesh, Rings, nsum};
use rw_mpas::mesh::geom::{V3, chord, unit};
use rw_mpas::mesh::validate::{Limits, validate};

struct Case {
    tag: &'static str,
    g: Rwg,
    mesh: MpasMesh,
}

fn build(tag: &'static str) -> Case {
    let g = load_mesh(tag);
    let x = g.f64("in/xCell");
    let y = g.f64("in/yCell");
    let z = g.f64("in/zCell");
    let cell_xyz: Vec<V3> = (0..x.len()).map(|i| [x[i], y[i], z[i]]).collect();
    let density = g.f64("in/meshDensity");
    let rings = Rings {
        offsets: g.i32("cell/cellsOnCell_offsets").iter().map(|&v| v as u32).collect(),
        values: g.i32("cell/cellsOnCell_values").iter().map(|&v| v as u32).collect(),
    };
    let nominal = g.f64("meta/nominalMinDc")[0];
    let mesh = MpasMesh::derive(cell_xyz, density, &rings, nominal)
        .unwrap_or_else(|e| panic!("{tag}: derive refused the published mesh's own centres: {e}"));
    Case { tag, g, mesh }
}

fn cases() -> Vec<Case> {
    MESHES.iter().map(|t| build(t)).collect()
}

/// Relative gap, guarding the zero denominator.
fn rel(got: f64, want: f64) -> f64 {
    if want == 0.0 {
        got.abs()
    } else {
        (got - want).abs() / want.abs()
    }
}

fn max_of(v: impl Iterator<Item = f64>) -> f64 {
    v.fold(0.0f64, f64::max)
}

// ------------------------------------------------------------- spherical Delaunay

/// The generator's own triangulation, graded against the published one.
///
/// Feed each published mesh's cell centres -- and nothing else -- to the hull,
/// and it has to rebuild the neighbour ring the file carries. The ring's
/// STARTING SLOT is not something MPAS fixes, so the comparison is cyclic; the
/// order within the ring is not free and is compared exactly.
#[test]
fn the_published_cell_centres_rebuild_the_published_connectivity() {
    for tag in MESHES {
        let g = load_mesh(tag);
        let x = g.f64("in/xCell");
        let y = g.f64("in/yCell");
        let z = g.f64("in/zCell");
        let pts: Vec<V3> = (0..x.len()).map(|i| [x[i], y[i], z[i]]).collect();
        let pts = rw_mpas::mesh::hull::to_unit_sphere(&pts).expect("published centres are unit vectors");
        let started = std::time::Instant::now();
        let got = rw_mpas::mesh::hull::delaunay_rings(&pts)
            .unwrap_or_else(|e| panic!("{tag}: the hull refused the published cell centres: {e}"));
        let wall = started.elapsed();

        let offsets: Vec<i32> = g.i32("cell/cellsOnCell_offsets");
        let values: Vec<i32> = g.i32("cell/cellsOnCell_values");
        let n = x.len();
        let mut rotations: std::collections::BTreeMap<usize, usize> = Default::default();
        for i in 0..n {
            let want: Vec<u32> = values[offsets[i] as usize..offsets[i + 1] as usize]
                .iter()
                .map(|&v| v as u32)
                .collect();
            let mine = got.ring(i);
            assert_eq!(
                mine.len(),
                want.len(),
                "{tag}: cell {i} has {} neighbours, the file says {}",
                mine.len(),
                want.len()
            );
            let deg = want.len();
            let start = mine
                .iter()
                .position(|&v| v == want[0])
                .unwrap_or_else(|| panic!("{tag}: cell {i} ring {mine:?} does not contain {}", want[0]));
            for k in 0..deg {
                assert_eq!(
                    mine[(start + k) % deg],
                    want[k],
                    "{tag}: cell {i} ring differs at slot {k}: {mine:?} vs {want:?}"
                );
            }
            *rotations.entry(start).or_default() += 1;
        }
        eprintln!(
            "{tag}: {n} cell centres -> the published ring, exactly, in {:.2} s. Ring start-slot offsets: {rotations:?}",
            wall.as_secs_f64()
        );
    }
}

// ------------------------------------------------------------------- topology

#[test]
fn derived_dimensions_match_the_published_file() {
    for c in cases() {
        let d = c.g.i32("meta/dims");
        let (nc, ne, nv, me, me2, vd) = (
            d[0] as usize,
            d[1] as usize,
            d[2] as usize,
            d[3] as usize,
            d[4] as usize,
            d[5] as usize,
        );
        assert_eq!(c.mesh.n_cells, nc, "{}: nCells", c.tag);
        assert_eq!(c.mesh.n_edges, ne, "{}: nEdges", c.tag);
        assert_eq!(c.mesh.n_vertices, nv, "{}: nVertices", c.tag);
        assert_eq!(c.mesh.max_edges, me, "{}: maxEdges", c.tag);
        assert_eq!(c.mesh.max_edges2, me2, "{}: maxEdges2", c.tag);
        assert_eq!(c.mesh.vertex_degree, vd, "{}: vertexDegree", c.tag);
        let want = c.g.i32("cell/nEdgesOnCell");
        assert_eq!(c.mesh.n_edges_on_cell, want, "{}: nEdgesOnCell", c.tag);
    }
}

#[test]
fn cell_connectivity_reproduces_the_published_rings_and_padding() {
    for c in cases() {
        let idx = c.g.i32("sample/cell_index");
        let raw1 = c.g.i32("sample/cellsOnCell_raw1");
        let me = c.mesh.max_edges;
        // cellsOnCell carries the file's own cell numbering, so it compares
        // directly -- padding slots included. The published rule is
        // repeat-of-last-valid; a generator that wrote 0 there would look fine
        // to every validator and hand a real-looking neighbour index to any
        // reader that walks maxEdges slots without honouring the count.
        for (r, &ci) in idx.iter().enumerate() {
            let ci = ci as usize;
            for j in 0..me {
                let got = c.mesh.cells_on_cell[ci * me + j] + 1;
                let want = raw1[r * me + j];
                assert_eq!(got, want, "{}: cellsOnCell[{ci}][{j}]", c.tag);
            }
        }

        // verticesOnCell and edgesOnCell carry OUR numbering, so only the used
        // slots compare (against the canonical golden) and the padding rule is
        // checked against the published rule directly.
        let voc = c.g.i32("sample/verticesOnCell");
        let eoc = c.g.i32("sample/edgesOnCell");
        let eoc_raw = c.g.i32("sample/edgesOnCell_raw1");
        let voc_raw = c.g.i32("sample/verticesOnCell_raw1");
        for (r, &ci) in idx.iter().enumerate() {
            let ci = ci as usize;
            let deg = c.mesh.n_edges_on_cell[ci] as usize;
            for j in 0..deg {
                assert_eq!(
                    c.mesh.vertices_on_cell[ci * me + j],
                    voc[r * me + j],
                    "{}: verticesOnCell[{ci}][{j}]",
                    c.tag
                );
                assert_eq!(
                    c.mesh.edges_on_cell[ci * me + j],
                    eoc[r * me + j],
                    "{}: edgesOnCell[{ci}][{j}]",
                    c.tag
                );
            }
            let max_valid = (0..deg).map(|j| eoc_raw[r * me + j]).max().unwrap();
            for j in deg..me {
                // published rule, verified on the file itself in both directions
                assert_eq!(eoc_raw[r * me + j], max_valid, "{}: file edgesOnCell padding", c.tag);
                assert_eq!(voc_raw[r * me + j], 0, "{}: file verticesOnCell padding", c.tag);
                // and the same rule in our own arrays
                let mine_max = (0..deg).map(|j| c.mesh.edges_on_cell[ci * me + j]).max().unwrap();
                assert_eq!(c.mesh.edges_on_cell[ci * me + j], mine_max, "{}: our edgesOnCell padding", c.tag);
                assert_eq!(c.mesh.vertices_on_cell[ci * me + j], -1, "{}: our verticesOnCell padding", c.tag);
            }
        }
    }
}

#[test]
fn edge_and_vertex_connectivity_match_the_published_file() {
    for c in cases() {
        let e_idx = c.g.i32("sample/edge_index");
        let e_cells = c.g.i32("sample/edge_cells");
        let coe = c.g.i32("sample/cellsOnEdge");
        let orient = c.g.i32("sample/edge_orient");
        let voe = c.g.i32("sample/verticesOnEdge");

        // The published files order cellsOnEdge ascending on every sampled edge
        // of both meshes (4,115 edges, argmin/argmax tails included), which is
        // the convention this crate emits. If that ever stops being true the
        // signed weight comparison below is the check that fires.
        let flipped = orient.iter().filter(|&&o| o != 1).count();
        assert_eq!(flipped, 0, "{}: the published cellsOnEdge order is not ascending", c.tag);

        for (r, &e) in e_idx.iter().enumerate() {
            let e = e as usize;
            assert_eq!(c.mesh.cells_on_edge[e * 2], e_cells[r * 2], "{}: cellsOnEdge[{e}][0]", c.tag);
            assert_eq!(c.mesh.cells_on_edge[e * 2 + 1], e_cells[r * 2 + 1], "{}: cellsOnEdge[{e}][1]", c.tag);
            assert_eq!(c.mesh.cells_on_edge[e * 2], coe[r * 2], "{}: file cellsOnEdge[{e}][0]", c.tag);
            // THE ORIENTATION LOCK. verticesOnEdge order is not free once
            // cellsOnEdge order is fixed; the port's validator accepts a
            // swapped pair silently, so this is the only place it is graded.
            assert_eq!(
                c.mesh.vertices_on_edge[e * 2],
                voe[r * 2],
                "{}: verticesOnEdge[{e}][0] -- the edge's tangential direction runs backwards",
                c.tag
            );
            assert_eq!(c.mesh.vertices_on_edge[e * 2 + 1], voe[r * 2 + 1], "{}: verticesOnEdge[{e}][1]", c.tag);
        }

        let v_idx = c.g.i32("sample/vertex_index");
        let v_cells = c.g.i32("sample/vertex_cells");
        let cov = c.g.i32("sample/cellsOnVertex");
        let eov = c.g.i32("sample/edgesOnVertex");
        for (r, &v) in v_idx.iter().enumerate() {
            let v = v as usize;
            let mine: Vec<i32> = (0..3).map(|s| c.mesh.cells_on_vertex[v * 3 + s]).collect();
            assert_eq!(mine, v_cells[r * 3..r * 3 + 3].to_vec(), "{}: cellsOnVertex[{v}]", c.tag);
            // The published winding of cellsOnVertex is arbitrary (2009 CCW /
            // 1991 CW of 4000 measured), so it compares as a SET.
            let mut file_set: Vec<i32> = cov[r * 3..r * 3 + 3].to_vec();
            let mut mine_set = mine.clone();
            file_set.sort_unstable();
            mine_set.sort_unstable();
            assert_eq!(mine_set, file_set, "{}: cellsOnVertex[{v}] as a set", c.tag);
            // edgesOnVertex ordering is likewise free -- measured, there is no
            // index correspondence with cellsOnVertex in the published files --
            // so it too compares as a set.
            let mut mine_e: Vec<i32> = (0..3).map(|s| c.mesh.edges_on_vertex[v * 3 + s]).collect();
            let mut file_e: Vec<i32> = eov[r * 3..r * 3 + 3].to_vec();
            mine_e.sort_unstable();
            file_e.sort_unstable();
            assert_eq!(mine_e, file_e, "{}: edgesOnVertex[{v}] as a set", c.tag);
        }
    }
}

// ------------------------------------------------------------------- geometry

#[test]
fn edge_and_vertex_positions_match_the_published_file() {
    for c in cases() {
        let e_idx = c.g.i32("sample/edge_index");
        let (ex, ey, ez) = (c.g.f64("sample/xEdge"), c.g.f64("sample/yEdge"), c.g.f64("sample/zEdge"));
        let mut worst_e = 0.0f64;
        for (r, &e) in e_idx.iter().enumerate() {
            worst_e = worst_e.max(chord(c.mesh.edge_xyz[e as usize], [ex[r], ey[r], ez[r]]));
        }
        let v_idx = c.g.i32("sample/vertex_index");
        let (vx, vy, vz) = (c.g.f64("sample/xVertex"), c.g.f64("sample/yVertex"), c.g.f64("sample/zVertex"));
        let mut worst_v = 0.0f64;
        for (r, &v) in v_idx.iter().enumerate() {
            worst_v = worst_v.max(chord(c.mesh.vertex_xyz[v as usize], [vx[r], vy[r], vz[r]]));
        }
        eprintln!(
            "{}: xyzEdge chord max {worst_e:.3e}, xyzVertex chord max {worst_v:.3e} (unit sphere; x{:.0} m for metres)",
            c.tag,
            rw_mpas::mesh::geom::EARTH_RADIUS_M
        );
        // The published variable mesh carries a longitude defect from
        // MPAS-Tools' binary32 pi, but the CARTESIAN coordinates are clean on
        // both files, which is why this compares in xyz and not in lat/lon.
        assert!(worst_e < 5e-13, "{}: xyzEdge chord {worst_e:.3e}", c.tag);
        assert!(worst_v < 5e-13, "{}: xyzVertex chord {worst_v:.3e}", c.tag);

        // The alternative convention, decided by measurement rather than
        // assumed: xyzEdge is the CELL midpoint, not the VERTEX midpoint. The
        // two coincide exactly on a perfectly symmetric hexagon, so the
        // alternative is judged on its MEDIAN residual, not its minimum.
        let mut vertexmid: Vec<f64> = Vec::with_capacity(e_idx.len());
        for (r, &e) in e_idx.iter().enumerate() {
            let e = e as usize;
            let v0 = c.mesh.vertices_on_edge[e * 2] as usize;
            let v1 = c.mesh.vertices_on_edge[e * 2 + 1] as usize;
            let mid = unit([
                c.mesh.vertex_xyz[v0][0] + c.mesh.vertex_xyz[v1][0],
                c.mesh.vertex_xyz[v0][1] + c.mesh.vertex_xyz[v1][1],
                c.mesh.vertex_xyz[v0][2] + c.mesh.vertex_xyz[v1][2],
            ])
            .unwrap();
            vertexmid.push(chord(mid, [ex[r], ey[r], ez[r]]));
        }
        vertexmid.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_alt = vertexmid[vertexmid.len() / 2];
        eprintln!(
            "{}: vertex-midpoint convention residual median {median_alt:.3e} vs cell-midpoint max {worst_e:.3e}",
            c.tag
        );
        assert!(
            median_alt > 1e6 * worst_e.max(1e-16),
            "{}: the cell-midpoint and vertex-midpoint conventions have converged ({worst_e:.3e} vs median {median_alt:.3e}); the measurement that decided xyzEdge no longer decides it",
            c.tag
        );
    }
}

#[test]
fn edge_lengths_match_the_published_file() {
    for c in cases() {
        let e_idx = c.g.i32("sample/edge_index");
        let dc = c.g.f64("sample/dcEdge");
        let dv = c.g.f64("sample/dvEdge");
        let mut wdc = 0.0f64;
        let mut wdv = 0.0f64;
        for (r, &e) in e_idx.iter().enumerate() {
            wdc = wdc.max(rel(c.mesh.dc_edge[e as usize], dc[r]));
            wdv = wdv.max(rel(c.mesh.dv_edge[e as usize], dv[r]));
        }
        eprintln!("{}: dcEdge rel max {wdc:.3e}, dvEdge rel max {wdv:.3e}", c.tag);
        // RESOLUTION LIMIT: the published files computed these arcs with
        // acos(dot), which loses about half its digits on a short edge -- ~3e-11
        // relative at the finest published spacing. This crate uses
        // atan2(|a x b|, a.b), which does not, so the gap below is the FILE's
        // error, not this crate's.
        assert!(wdc < 1e-9, "{}: dcEdge rel {wdc:.3e}", c.tag);
        assert!(wdv < 1e-8, "{}: dvEdge rel {wdv:.3e}", c.tag);
    }
}

/// Cells whose published area cannot be the area of their own Voronoi polygon.
///
/// Measured, not assumed: the published variable-resolution mesh carries a
/// handful of dual vertices displaced from the circumcentre of their three
/// cells, and `the_published_variable_mesh_carries_displaced_dual_vertices`
/// below proves it by solving for the displacement. Their areas are reported
/// here so every other cell can be graded strictly.
fn cells_the_file_disagrees_on(c: &Case, tol: f64) -> Vec<(usize, f64)> {
    let want = c.g.f64("cell/areaCell");
    (0..c.mesh.n_cells)
        .filter(|&i| rel(c.mesh.area_cell[i], want[i]) > tol)
        .map(|i| (i, c.mesh.area_cell[i] - want[i]))
        .collect()
}

#[test]
fn areas_and_the_kite_partition_match_the_published_file() {
    for c in cases() {
        let want_area = c.g.f64("cell/areaCell");
        let odd = cells_the_file_disagrees_on(&c, 1e-9);
        let wa = max_of(
            (0..c.mesh.n_cells)
                .filter(|i| !odd.iter().any(|(j, _)| j == i))
                .map(|i| rel(c.mesh.area_cell[i], want_area[i])),
        );
        // The disagreements must CANCEL. A displaced shared vertex moves area
        // from one cell into its neighbours and the total is untouched; a cell
        // dropped, doubled or mis-shaped does not cancel. That is the check
        // that keeps the exemption from being a blanket one.
        let biggest = odd.iter().map(|(_, d)| d.abs()).fold(0.0f64, f64::max);
        let net: f64 = odd.iter().map(|(_, d)| *d).sum();
        eprintln!(
            "{}: areaCell rel max {wa:.3e} over {} graded cells; {} cells the file disagrees on, net {net:.3e} against biggest single {biggest:.3e}",
            c.tag,
            c.mesh.n_cells - odd.len(),
            odd.len()
        );
        assert!(odd.len() <= 32, "{}: {} cells disagree, more than the published defect accounts for", c.tag, odd.len());
        if !odd.is_empty() {
            assert!(
                net.abs() < 1e-6 * biggest,
                "{}: the disagreements do not cancel (net {net:.3e} of {biggest:.3e}); this is not a displaced shared vertex, it is missing or duplicated area",
                c.tag
            );
        }
        let v_idx = c.g.i32("sample/vertex_index");
        let at = c.g.f64("sample/areaTriangle");
        let kite = c.g.f64("sample/kiteAreasOnVertex");
        let cov = c.g.i32("sample/cellsOnVertex");
        let mut wt = 0.0f64;
        let mut wk = 0.0f64;
        for (r, &v) in v_idx.iter().enumerate() {
            let v = v as usize;
            wt = wt.max(rel(c.mesh.area_triangle[v], at[r]));
            // Slot i of kiteAreasOnVertex pairs with cellsOnVertex[v][i]; the
            // published winding is arbitrary, so the kites are matched through
            // the cell they belong to rather than by slot number.
            for s in 0..3 {
                let cell = cov[r * 3 + s];
                let mine_slot = (0..3)
                    .find(|&t| c.mesh.cells_on_vertex[v * 3 + t] == cell)
                    .expect("the two triples are the same set");
                wk = wk.max(rel(c.mesh.kite_areas_on_vertex[v * 3 + mine_slot], kite[r * 3 + s]));
            }
        }
        let four_pi = 4.0 * std::f64::consts::PI;
        eprintln!(
            "{}: areaCell rel max {wa:.3e} over ALL {} cells, areaTriangle rel max {wt:.3e}, kite rel max {wk:.3e}; sum(areaCell)/4pi = {:.16}",
            c.tag,
            c.mesh.n_cells,
            nsum(&c.mesh.area_cell) / four_pi
        );
        // RESOLUTION LIMIT: areaCell is the sum of a cell's kites, so it
        // inherits the kite accuracy, and the kites are being compared against
        // values the published files carry to about 1e-10 themselves. x1 reads
        // 1.4e-12 and x4 1.2e-10; the gate sits just above the worse of the two.
        assert!(wa < 5e-10, "{}: areaCell rel {wa:.3e}", c.tag);
        assert!(wt < 5e-10, "{}: areaTriangle rel {wt:.3e}", c.tag);
        assert!(wk < 5e-10, "{}: kiteAreasOnVertex rel {wk:.3e}", c.tag);
    }
}

#[test]
fn the_tangential_stencil_and_its_weights_match_the_published_file() {
    for c in cases() {
        let e_idx = c.g.i32("sample/edge_index");
        let neoe = c.g.i32("sample/nEdgesOnEdge");
        let eoe = c.g.i32("sample/edgesOnEdge");
        let w = c.g.f64("sample/weightsOnEdge");
        let me2 = c.mesh.max_edges2;
        let mut worst_w = 0.0f64;
        let mut worst_at = (0usize, 0usize);
        let mut scale = 0.0f64;
        // A weight is `(running kite sum / areaCell - 1/2) * dv/dc`, so an edge
        // on a cell whose published area is not its Voronoi area cannot match.
        // Those cells are identified by measurement, not by a hand-written list.
        let odd: Vec<usize> = cells_the_file_disagrees_on(&c, 1e-9)
            .into_iter()
            .map(|(i, _)| i)
            .collect();
        let mut skipped = 0usize;
        for (r, &e) in e_idx.iter().enumerate() {
            let e = e as usize;
            assert_eq!(c.mesh.n_edges_on_edge[e], neoe[r], "{}: nEdgesOnEdge[{e}]", c.tag);
            let n = neoe[r] as usize;
            let touches_odd = (0..n).any(|k| {
                let ep = c.mesh.edges_on_edge[e * me2 + k] as usize;
                odd.contains(&(c.mesh.cells_on_edge[ep * 2] as usize))
                    || odd.contains(&(c.mesh.cells_on_edge[ep * 2 + 1] as usize))
            }) || odd.contains(&(c.mesh.cells_on_edge[e * 2] as usize))
                || odd.contains(&(c.mesh.cells_on_edge[e * 2 + 1] as usize));
            if touches_odd {
                skipped += 1;
                continue;
            }
            for k in 0..n {
                assert_eq!(
                    c.mesh.edges_on_edge[e * me2 + k],
                    eoe[r * me2 + k],
                    "{}: edgesOnEdge[{e}][{k}] -- the blocked stencil order",
                    c.tag
                );
                let got = c.mesh.weights_on_edge[e * me2 + k];
                let want = w[r * me2 + k];
                scale = scale.max(want.abs());
                let gap = (got - want).abs();
                if gap > worst_w {
                    worst_w = gap;
                    worst_at = (e, k);
                }
            }
            for k in n..me2 {
                assert_eq!(c.mesh.edges_on_edge[e * me2 + k], -1, "{}: stencil padding", c.tag);
                assert_eq!(c.mesh.weights_on_edge[e * me2 + k], 0.0, "{}: weight padding", c.tag);
                assert_eq!(w[r * me2 + k], 0.0, "{}: the file's weight padding", c.tag);
            }
        }
        eprintln!(
            "{}: weightsOnEdge absolute max gap {worst_w:.3e} at edge {} slot {} over {} of {} sampled edges (weight scale {scale:.6}); {skipped} skipped for touching a cell whose published area is not its Voronoi area. This is a SIGNED comparison, so a flipped edge orientation anywhere in a stencil would show up here",
            c.tag,
            worst_at.0,
            worst_at.1,
            e_idx.len() - skipped,
            e_idx.len()
        );
        assert!(
            skipped * 20 < e_idx.len(),
            "{}: {skipped} of {} sampled edges skipped -- the exemption has stopped being a handful of cells",
            c.tag,
            e_idx.len()
        );
        // The weights are recomputed here from recomputed kite areas, cell
        // areas and arc lengths, so the residual is the propagated difference
        // of those, not the 1-ULP figure the closed form reaches when fed the
        // file's own inputs.
        assert!(worst_w < 1e-9, "{}: weightsOnEdge gap {worst_w:.3e}", c.tag);
    }
}

#[test]
fn angle_edge_matches_the_published_file_to_its_measured_limit() {
    for c in cases() {
        let e_idx = c.g.i32("sample/edge_index");
        let want = c.g.f64("sample/angleEdge");
        let mut residuals: Vec<f64> = Vec::with_capacity(e_idx.len());
        let mut worst = (0.0f64, 0usize);
        for (r, &e) in e_idx.iter().enumerate() {
            let mut d = c.mesh.angle_edge[e as usize] - want[r];
            while d > std::f64::consts::PI {
                d -= std::f64::consts::TAU;
            }
            while d < -std::f64::consts::PI {
                d += std::f64::consts::TAU;
            }
            residuals.push(d.abs());
            if d.abs() > worst.0 {
                worst = (d.abs(), e as usize);
            }
        }
        residuals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = residuals[residuals.len() / 2];
        let p99 = residuals[residuals.len() * 99 / 100];
        eprintln!(
            "{}: angleEdge residual median {median:.3e} rad, p99 {p99:.3e} rad, max {:.3e} rad at edge {} ({:.4} deg)",
            c.tag,
            worst.0,
            worst.1,
            worst.0.to_degrees()
        );
        // angleEdge is the ONE emitted field whose upstream closed form is not
        // settled. This crate uses atan2(n.north, n.east) with n the
        // tangent-projected c0 -> c1 normal at the edge point; three competing
        // forms were ruled out at ~pi/2 mean error. The gate is set at the
        // level this form actually reaches against the real files, so a change
        // that made it worse would fail rather than pass quietly. The port's
        // own validator checks only |a| <= pi and accepts a 0.5 rad error.
        assert!(median < 1e-5, "{}: angleEdge median residual {median:.3e} rad", c.tag);
        assert!(worst.0 < 3e-2, "{}: angleEdge max residual {:.3e} rad", c.tag, worst.0);
    }
}

// ----------------------------------------------------------------- validation

#[test]
fn the_published_meshes_pass_the_emit_gate() {
    for c in cases() {
        // The FP32 survival floor is the ONE check the published variable
        // mesh does not clear, and that is a statement about the mesh, not
        // about the gate: x4.163842 carries a 1,170.0 m dual edge (its
        // dv/dc floor of 0.0336 at a 34,770 m dcEdge), and a 1.17 km edge
        // recovered from f32 vertices at earth-radius magnitude misses the
        // port's rtol=2e-5/atol=0.0 load check by an order of magnitude --
        // the same check that makes fine meshes coin-flips at load. The
        // geometric checks are graded on BOTH meshes under limits that name
        // that floor away; the default gate's verdict on each is asserted
        // separately below.
        let geometric_limits = Limits {
            min_dv_edge_m: 0.0,
            min_dv_over_dc: 0.0,
            ..Limits::default()
        };
        let report = validate(&c.mesh, geometric_limits).unwrap_or_else(|e| {
            panic!("{}: the emit gate refused a published NCAR mesh: {e}", c.tag)
        });
        match c.tag {
            "x1.40962" => {
                validate(&c.mesh, Limits::default()).unwrap_or_else(|e| {
                    panic!("x1.40962 must clear the DEFAULT gate, FP32 floor included: {e}")
                });
            }
            "x4.163842" => {
                let err = validate(&c.mesh, Limits::default()).expect_err(
                    "x4.163842 carries a 1,170.0 m dual edge; a default gate that \
                     passes it would emit meshes the port refuses at load",
                );
                let text = err.to_string();
                assert!(text.contains("FP32 survival floor"), "{text}");
                assert!(text.contains("1170.0 m") || text.contains("1170 m"), "{text}");
            }
            other => panic!("unknown golden {other}"),
        }
        eprintln!(
            "{}: euler {} defect {} coord {:?}\n    sum(areaCell)/4pi {:.16} sum(areaTriangle)/4pi {:.16}\n    max nonorthogonality {:.3e}  min edge orientation {:.6}\n    max weight antisymmetry {:.3e} over {} pairs\n    dcEdge {:.3} .. {:.3} km   dv/dc {:.4} .. {:.4} (median {:.4})\n    spacing {:.3} .. {:.3} km  ratio {:.4}  worst adjacent {:.4}",
            c.tag,
            report.euler_characteristic,
            report.coordination_defect,
            report.coordination_histogram,
            report.sum_area_cell_over_4pi,
            report.sum_area_triangle_over_4pi,
            report.max_nonorthogonality,
            report.min_edge_orientation,
            report.max_weight_antisymmetry,
            report.weight_pairs_checked,
            report.min_dc_edge_m / 1000.0,
            report.max_dc_edge_m / 1000.0,
            report.min_dv_over_dc,
            report.max_dv_over_dc,
            report.median_dv_over_dc,
            report.min_spacing_m / 1000.0,
            report.max_spacing_m / 1000.0,
            report.spacing_ratio,
            report.max_adjacent_spacing_ratio,
        );
        assert_eq!(report.euler_characteristic, 2);
        assert_eq!(report.coordination_defect, 12);
        assert_eq!(report.nonzero_weight_padding_slots, 0);
    }
}

/// The gate has to say no as reliably as it says yes. Each injection below is a
/// defect that the port's own validator accepts silently, or one that costs a
/// forecast; every one must be refused here, and the refusal must name what it
/// prevents.
#[test]
fn the_emit_gate_refuses_each_defect_it_exists_to_catch() {
    let c = build("x1.40962");
    let base = &c.mesh;

    // 1. a swapped edge: the orientation lock
    let mut m = base.clone();
    m.vertices_on_edge.swap(11 * 2, 11 * 2 + 1);
    let err = validate(&m, Limits::default()).unwrap_err().to_string();
    assert!(err.contains("locked right-handed pair"), "swapped verticesOnEdge: {err}");

    // 2. a wrong weight value
    let mut m = base.clone();
    m.weights_on_edge[3 * m.max_edges2] += 1e-6;
    let err = validate(&m, Limits::default()).unwrap_err().to_string();
    assert!(err.contains("antisymmetric"), "perturbed weight: {err}");

    // 3. a nonzero weight padding slot
    let mut m = base.clone();
    let e = 5usize;
    let n = m.n_edges_on_edge[e] as usize;
    m.weights_on_edge[e * m.max_edges2 + n] = 1e-30;
    let err = validate(&m, Limits::default()).unwrap_err().to_string();
    assert!(err.contains("padding slots are not exactly 0.0"), "weight padding: {err}");

    // 4. an area that no longer tiles the sphere
    let mut m = base.clone();
    m.area_cell[7] *= 1.05;
    let err = validate(&m, Limits::default()).unwrap_err().to_string();
    assert!(err.contains("do not partition") || err.contains("4*pi"), "perturbed areaCell: {err}");

    // 5. a broken kite partition that leaves the global sum alone
    let mut m = base.clone();
    m.kite_areas_on_vertex[4 * 3] *= 1.0001;
    m.kite_areas_on_vertex[4 * 3 + 1] -= m.kite_areas_on_vertex[4 * 3] * 0.0001 / 1.0001;
    let err = validate(&m, Limits::default()).unwrap_err().to_string();
    assert!(err.contains("partition"), "perturbed kite: {err}");

    // 6. a neighbour relation that is not mutual
    let mut m = base.clone();
    let victim = m.cells_on_cell[0] as usize;
    let deg = m.n_edges_on_cell[victim] as usize;
    for k in 0..deg {
        if m.cells_on_cell[victim * m.max_edges + k] == 0 {
            m.cells_on_cell[victim * m.max_edges + k] = m.cells_on_cell[victim * m.max_edges];
        }
    }
    let err = validate(&m, Limits::default()).unwrap_err().to_string();
    assert!(err.contains("not mutual") || err.contains("disagree"), "broken reciprocity: {err}");

    // 7. a topology that is no longer a closed sphere
    let mut m = base.clone();
    m.n_vertices += 1;
    let err = validate(&m, Limits::default()).unwrap_err().to_string();
    assert!(err.contains("closed sphere"), "broken Euler: {err}");

    // 8. a cell whose vertex ring winds the wrong way
    let mut m = base.clone();
    let deg = m.n_edges_on_cell[9] as usize;
    let base9 = 9 * m.max_edges;
    m.vertices_on_cell[base9..base9 + deg].reverse();
    let err = validate(&m, Limits::default()).unwrap_err().to_string();
    assert!(
        err.contains("winds clockwise") || err.contains("locked right-handed pair"),
        "reversed ring: {err}"
    );

    // 9. a non-orthogonal mesh: move one cell centre far enough that its
    //    Voronoi edges stop bisecting the dual arcs
    let mut m = base.clone();
    m.cell_xyz[12] = unit([
        m.cell_xyz[12][0] + 1e-3,
        m.cell_xyz[12][1],
        m.cell_xyz[12][2],
    ])
    .unwrap();
    let err = validate(&m, Limits::default()).unwrap_err().to_string();
    assert!(
        err.contains("perpendicular") || err.contains("locked right-handed pair"),
        "displaced generator: {err}"
    );
}

/// Why a handful of published cells are exempted from the strict area grade.
///
/// The exemption is not a shrug. For each disagreeing cluster there is a SINGLE
/// dual-vertex position -- two degrees of freedom -- that reconciles the THREE
/// cells meeting at it. Three equations, two unknowns: if the fit closes to
/// machine precision it cannot be coincidence, and the only reading left is
/// that the published file carries that vertex somewhere other than the
/// circumcentre of its own three generators.
///
/// It also proves the mesh emitted here is the correct one, which is the reason
/// this test exists rather than a comment.
#[test]
fn the_published_variable_mesh_carries_displaced_dual_vertices() {
    let c = build("x4.163842");
    let want = c.g.f64("cell/areaCell");
    let me = c.mesh.max_edges;
    let odd: Vec<usize> = cells_the_file_disagrees_on(&c, 1e-9)
        .into_iter()
        .map(|(i, _)| i)
        .collect();
    assert!(!odd.is_empty(), "nothing to explain");

    // Every disagreeing cell must be Delaunay-correct: no fourth generator
    // inside any of its Voronoi vertices' circumcircles. If one were, the
    // published ring would be the wrong triangulation and this crate's reading
    // of it -- not the file's vertex placement -- would be what is off.
    for &i in odd.iter().take(3) {
        let deg = c.mesh.n_edges_on_cell[i] as usize;
        for j in 0..deg {
            let v = c.mesh.vertices_on_cell[i * me + j] as usize;
            let t: Vec<i32> = (0..3).map(|s| c.mesh.cells_on_vertex[v * 3 + s]).collect();
            let cc = c.mesh.vertex_xyz[v];
            let radius = rw_mpas::mesh::geom::arc(cc, c.mesh.cell_xyz[t[0] as usize]);
            let intruders = (0..c.mesh.n_cells)
                .filter(|d| !t.contains(&(*d as i32)))
                .filter(|&d| rw_mpas::mesh::geom::arc(cc, c.mesh.cell_xyz[d]) < radius * (1.0 - 1e-12))
                .count();
            assert_eq!(
                intruders, 0,
                "cell {i} vertex {v} triple {t:?}: the published ring is not Delaunay here"
            );
        }
    }

    // Find a vertex shared by three disagreeing cells and solve for it.
    let mut target: Option<usize> = None;
    'outer: for &i in &odd {
        let deg = c.mesh.n_edges_on_cell[i] as usize;
        for j in 0..deg {
            let v = c.mesh.vertices_on_cell[i * me + j] as usize;
            if (0..3).all(|s| odd.contains(&(c.mesh.cells_on_vertex[v * 3 + s] as usize))) {
                target = Some(v);
                break 'outer;
            }
        }
    }
    let target_v = target.expect("the disagreeing cells share a dual vertex");
    let trio: Vec<usize> = (0..3)
        .map(|s| c.mesh.cells_on_vertex[target_v * 3 + s] as usize)
        .collect();

    let area_with = |i: usize, p: V3| -> f64 {
        let deg = c.mesh.n_edges_on_cell[i] as usize;
        let ring: Vec<V3> = (0..deg)
            .map(|j| {
                let v = c.mesh.vertices_on_cell[i * me + j] as usize;
                if v == target_v { p } else { c.mesh.vertex_xyz[v] }
            })
            .collect();
        rw_mpas::mesh::geom::polygon_area(&ring)
    };
    let cc = c.mesh.vertex_xyz[target_v];
    let (e1, e2) = rw_mpas::mesh::geom::east_north(cc).expect("the vertex is not on a pole");
    let resid = |s: f64, t: f64| -> [f64; 3] {
        let p = unit([
            cc[0] + s * e1[0] + t * e2[0],
            cc[1] + s * e1[1] + t * e2[1],
            cc[2] + s * e1[2] + t * e2[2],
        ])
        .unwrap();
        [
            area_with(trio[0], p) - want[trio[0]],
            area_with(trio[1], p) - want[trio[1]],
            area_with(trio[2], p) - want[trio[2]],
        ]
    };
    let start = resid(0.0, 0.0);
    let (mut s, mut t) = (0.0f64, 0.0f64);
    let h = 1e-7;
    for _ in 0..40 {
        let f0 = resid(s, t);
        let fs = resid(s + h, t);
        let ft = resid(s, t + h);
        // normal equations of the over-determined 3x2 least-squares system
        let js: Vec<f64> = (0..3).map(|k| (fs[k] - f0[k]) / h).collect();
        let jt: Vec<f64> = (0..3).map(|k| (ft[k] - f0[k]) / h).collect();
        let (mut a, mut b, mut d, mut g1, mut g2) = (0.0, 0.0, 0.0, 0.0, 0.0);
        for k in 0..3 {
            a += js[k] * js[k];
            b += js[k] * jt[k];
            d += jt[k] * jt[k];
            g1 += js[k] * f0[k];
            g2 += jt[k] * f0[k];
        }
        let det = a * d - b * b;
        if det.abs() < 1e-30 {
            break;
        }
        s -= (d * g1 - b * g2) / det;
        t -= (a * g2 - b * g1) / det;
    }
    let end = resid(s, t);
    let disp = (s * s + t * t).sqrt();
    let area_scale = want[trio[0]];
    eprintln!(
        "x4.163842: {} cells disagree. Vertex {target_v} of cells {trio:?}: one displacement of {disp:.4e} rad ({:.1} m on Earth) takes the three residuals from {:.3e} {:.3e} {:.3e} to {:.3e} {:.3e} {:.3e}, against cell areas of {area_scale:.3e}",
        odd.len(),
        disp * rw_mpas::mesh::geom::EARTH_RADIUS_M,
        start[0], start[1], start[2],
        end[0], end[1], end[2]
    );
    let worst_start = start.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
    let worst_end = end.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
    assert!(
        worst_start > 1e-8 * area_scale,
        "the circumcentre already reproduces the file; there is nothing to explain"
    );
    assert!(
        worst_end < 1e-9 * worst_start,
        "no single vertex position reconciles the three cells (residual {worst_end:.3e} from {worst_start:.3e}); the disagreement is not a displaced dual vertex and needs a different explanation before any cell is exempted"
    );
    assert!(
        disp > 1e-6,
        "the fitted displacement {disp:.3e} rad is at the noise floor"
    );
}
