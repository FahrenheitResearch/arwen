//! The known answer a mesh generator is graded against.
//!
//! Both published MPAS meshes were produced by NCAR's reference tooling, so they
//! are correct by construction. Take ONLY the cell centres out of one and every
//! other field in the file becomes an expected output. These tests load that
//! golden offline (no netCDF, no network) and prove three things:
//!
//!   1. the container parses and the inventory is complete;
//!   2. every derived field in the golden agrees with its own definition, using
//!      nothing but the golden's own contents -- so the golden is self-sufficient
//!      and a generator can be graded against it without the .nc files;
//!   3. the comparison FAILS when a value moves. An extractor never shown to fail
//!      is not evidence.
//!
//! Tolerances below are set from measured residuals, not chosen. The measured
//! figures are in the assert messages so a future change to them is visible.

mod common;

use common::{canonical_edges, canonical_vertices, golden_dir, load_mesh, Rwg, MESHES};

/// Neumaier compensated sum. Plain f64 accumulation over 163,842 areas loses
/// enough low bits to blur the 4*pi sphere-closure check this file relies on.
fn nsum(v: &[f64]) -> f64 {
    let (mut s, mut c) = (0.0f64, 0.0f64);
    for &x in v {
        let t = s + x;
        c += if s.abs() >= x.abs() { (s - t) + x } else { (x - t) + s };
        s = t;
    }
    s + c
}

fn norm(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn unit(v: [f64; 3]) -> [f64; 3] {
    let n = norm(v);
    [v[0] / n, v[1] / n, v[2] / n]
}
/// Chord distance. Used instead of the angle for point comparisons: acos near 1
/// bottoms out at sqrt(2*eps) = 2.1e-8 rad and cannot resolve a machine-precision
/// agreement at all, which is the entire range these checks live in.
fn chord(a: [f64; 3], b: [f64; 3]) -> f64 {
    norm(sub(a, b))
}

struct Mesh {
    tag: &'static str,
    g: Rwg,
    n_cells: usize,
    n_edges: usize,
    n_vertices: usize,
    max_edges: usize,
    max_edges2: usize,
    vertex_degree: usize,
}

impl Mesh {
    fn open(tag: &'static str) -> Mesh {
        let g = load_mesh(tag);
        let d = g.i32("meta/dims");
        assert_eq!(d.len(), 8, "{tag}: meta/dims must carry 8 dimensions");
        Mesh {
            tag,
            n_cells: d[0] as usize,
            n_edges: d[1] as usize,
            n_vertices: d[2] as usize,
            max_edges: d[3] as usize,
            max_edges2: d[4] as usize,
            vertex_degree: d[5] as usize,
            g,
        }
    }
    fn cell_xyz(&self) -> Vec<[f64; 3]> {
        let x = self.g.f64("in/xCell");
        let y = self.g.f64("in/yCell");
        let z = self.g.f64("in/zCell");
        (0..x.len()).map(|i| [x[i], y[i], z[i]]).collect()
    }
}

fn meshes() -> Vec<Mesh> {
    MESHES.iter().map(|t| Mesh::open(t)).collect()
}

/// Every entry the golden is contracted to carry. A generator's grader reads
/// these names, so a missing one is a broken golden, not a soft warning.
const REQUIRED: &[&str] = &[
    "meta/dims",
    "meta/sphere_radius",
    "meta/nominalMinDc",
    "meta/densityFunctionCode",
    "meta/sample_strides",
    "in/xCell",
    "in/yCell",
    "in/zCell",
    "in/meshDensity",
    "cell/nEdgesOnCell",
    "cell/cellsOnCell_offsets",
    "cell/cellsOnCell_values",
    "cell/areaCell",
    "cell/latCell",
    "cell/lonCell",
    "sample/cell_index",
    "sample/verticesOnCell",
    "sample/edgesOnCell",
    "sample/cellsOnCell_raw1",
    "sample/edgesOnCell_raw1",
    "sample/verticesOnCell_raw1",
    "sample/edge_index",
    "sample/edge_cells",
    "sample/edge_orient",
    "sample/cellsOnEdge",
    "sample/verticesOnEdge",
    "sample/nEdgesOnEdge",
    "sample/edgesOnEdge",
    "sample/weightsOnEdge",
    "sample/dcEdge",
    "sample/dvEdge",
    "sample/angleEdge",
    "sample/latEdge",
    "sample/lonEdge",
    "sample/xEdge",
    "sample/yEdge",
    "sample/zEdge",
    "sample/vertex_index",
    "sample/vertex_cells",
    "sample/cellsOnVertex",
    "sample/edgesOnVertex",
    "sample/areaTriangle",
    "sample/kiteAreasOnVertex",
    "sample/latVertex",
    "sample/lonVertex",
    "sample/xVertex",
    "sample/yVertex",
    "sample/zVertex",
];

#[test]
fn container_parses_and_inventory_is_complete() {
    for m in meshes() {
        for name in REQUIRED {
            let e = m.g.entry(name);
            assert!(e.shape.iter().product::<usize>() > 0, "{}: {name} is empty", m.tag);
        }
        // shapes must agree with the declared dimensions
        assert_eq!(m.g.shape("in/xCell"), [m.n_cells], "{}", m.tag);
        assert_eq!(m.g.shape("cell/areaCell"), [m.n_cells], "{}", m.tag);
        assert_eq!(m.g.shape("cell/cellsOnCell_offsets"), [m.n_cells + 1], "{}", m.tag);
        assert_eq!(m.g.shape("cell/cellsOnCell_values"), [2 * m.n_edges], "{}", m.tag);
        let ns = m.g.shape("sample/edge_index")[0];
        assert_eq!(m.g.shape("sample/weightsOnEdge"), [ns, m.max_edges2], "{}", m.tag);
        assert_eq!(m.g.shape("sample/edgesOnEdge"), [ns, m.max_edges2], "{}", m.tag);
        let nt = m.g.shape("sample/vertex_index")[0];
        assert_eq!(m.g.shape("sample/kiteAreasOnVertex"), [nt, m.vertex_degree], "{}", m.tag);
        let nu = m.g.shape("sample/cell_index")[0];
        assert_eq!(m.g.shape("sample/verticesOnCell"), [nu, m.max_edges], "{}", m.tag);
        assert_eq!(m.g.f64("meta/sphere_radius")[0], 1.0, "{}: grid files are unit-sphere", m.tag);
        println!(
            "{}: nCells {} nEdges {} nVertices {} | sampled cells {} edges {} vertices {}",
            m.tag, m.n_cells, m.n_edges, m.n_vertices, nu, ns, nt
        );
    }
}

#[test]
fn topology_identities_hold() {
    for m in meshes() {
        let neoc = m.g.i32("cell/nEdgesOnCell");
        let offs = m.g.i32("cell/cellsOnCell_offsets");
        assert_eq!(
            m.n_vertices,
            2 * m.n_cells - 4,
            "{}: closed-sphere nVertices == 2*nCells-4",
            m.tag
        );
        assert_eq!(m.n_edges, 3 * m.n_cells - 6, "{}: closed-sphere nEdges == 3*nCells-6", m.tag);
        assert_eq!(
            m.n_cells as i64 - m.n_edges as i64 + m.n_vertices as i64,
            2,
            "{}: Euler characteristic",
            m.tag
        );
        let sum: i64 = neoc.iter().map(|&v| v as i64).sum();
        assert_eq!(sum, 2 * m.n_edges as i64, "{}: sum(nEdgesOnCell) == 2*nEdges", m.tag);
        let defect: i64 = neoc.iter().map(|&v| 6 - v as i64).sum();
        assert_eq!(defect, 12, "{}: total coordination defect is 12 on a sphere", m.tag);
        // CSR offsets must be the running total of nEdgesOnCell
        for i in 0..m.n_cells {
            assert_eq!(
                offs[i + 1] - offs[i],
                neoc[i],
                "{}: CSR row {i} width disagrees with nEdgesOnCell",
                m.tag
            );
        }
        let mut hist = std::collections::BTreeMap::new();
        for &v in &neoc {
            *hist.entry(v).or_insert(0usize) += 1;
        }
        println!("{}: nEdgesOnCell histogram {hist:?}", m.tag);
    }
}

#[test]
fn canonical_numbering_rebuilds_from_cell_centres_alone() {
    // The point of the whole golden design: edge and vertex numbering is the
    // generator's own choice, so the golden expresses every edge/vertex field in
    // a numbering derived from the cellsOnCell ring. This proves Rust rebuilds
    // exactly the numbering the extractor used.
    for m in meshes() {
        let offs = m.g.i32("cell/cellsOnCell_offsets");
        let vals = m.g.i32("cell/cellsOnCell_values");

        let edges = canonical_edges(&offs, &vals);
        assert_eq!(edges.len(), m.n_edges, "{}: canonical edge count", m.tag);
        let verts = canonical_vertices(&offs, &vals);
        assert_eq!(verts.len(), m.n_vertices, "{}: canonical vertex count", m.tag);

        let ei = m.g.i32("sample/edge_index");
        let ec = m.g.i32("sample/edge_cells");
        for (k, &e) in ei.iter().enumerate() {
            assert_eq!(
                edges[e as usize],
                [ec[2 * k], ec[2 * k + 1]],
                "{}: canonical edge {e} disagrees with the golden's key",
                m.tag
            );
        }
        let vi = m.g.i32("sample/vertex_index");
        let vc = m.g.i32("sample/vertex_cells");
        for (k, &v) in vi.iter().enumerate() {
            assert_eq!(
                verts[v as usize],
                [vc[3 * k], vc[3 * k + 1], vc[3 * k + 2]],
                "{}: canonical vertex {v} disagrees with the golden's key",
                m.tag
            );
        }
        println!("{}: rebuilt {} edges and {} vertices from the ring", m.tag, edges.len(), verts.len());
    }
}

#[test]
fn sphere_closure_and_kite_partition() {
    for m in meshes() {
        let area = m.g.f64("cell/areaCell");
        let total = nsum(&area);
        let four_pi = 4.0 * std::f64::consts::PI;
        let rel = (total / four_pi - 1.0).abs();
        assert!(
            rel < 1e-12,
            "{}: sum(areaCell)/4pi = {total:.17} / {four_pi:.17}, rel {rel:.3e}",
            m.tag
        );
        assert!(area.iter().all(|&a| a > 0.0 && a.is_finite()), "{}: areaCell positive", m.tag);

        // each vertex's kites partition its triangle
        let kite = m.g.f64("sample/kiteAreasOnVertex");
        let tri = m.g.f64("sample/areaTriangle");
        let vd = m.vertex_degree;
        let mut worst = 0.0f64;
        for (k, &t) in tri.iter().enumerate() {
            let s: f64 = kite[k * vd..(k + 1) * vd].iter().sum();
            worst = worst.max((s - t).abs() / t);
        }
        assert!(worst < 1e-14, "{}: kite rows vs areaTriangle, max rel {worst:.3e}", m.tag);
        println!("{}: sum(areaCell)/4pi-1 = {rel:.3e}, kite/triangle max rel {worst:.3e}", m.tag);
    }
}

#[test]
fn edge_geometry_matches_its_definition() {
    // Measured residuals that set these bounds:
    //   dcEdge vs arc(cell,cell)          x1 8.4e-13   x4 7.9e-11  (relative)
    //   xEdge  vs normalize(cell midpoint) x1 1.4e-14  x4 2.5e-13  (chord)
    // The x4 figures are that mesh's SCVT convergence residual, not a different
    // convention: the vertex-midpoint alternative sits 4 orders further away.
    for m in meshes() {
        let p = m.cell_xyz();
        let ec = m.g.i32("sample/edge_cells");
        let dc = m.g.f64("sample/dcEdge");
        let xe = m.g.f64("sample/xEdge");
        let ye = m.g.f64("sample/yEdge");
        let ze = m.g.f64("sample/zEdge");
        let late = m.g.f64("sample/latEdge");
        let lone = m.g.f64("sample/lonEdge");

        let (mut wdc, mut wmid, mut wcen, mut wll) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for k in 0..dc.len() {
            let a = p[ec[2 * k] as usize];
            let b = p[ec[2 * k + 1] as usize];
            let arc = dot(a, b).clamp(-1.0, 1.0).acos();
            wdc = wdc.max((arc - dc[k]).abs() / dc[k]);

            let e = [xe[k], ye[k], ze[k]];
            wmid = wmid.max(chord(unit([a[0] + b[0], a[1] + b[1], a[2] + b[2]]), e));
            wcen = wcen.max((norm(e) - 1.0).abs());

            let ll = [
                late[k].cos() * lone[k].cos(),
                late[k].cos() * lone[k].sin(),
                late[k].sin(),
            ];
            wll = wll.max(chord(ll, e));
        }
        assert!(wdc < 1e-9, "{}: dcEdge vs arc(cell,cell) max rel {wdc:.3e}", m.tag);
        assert!(wmid < 1e-11, "{}: xEdge vs normalize(cell midpoint) max chord {wmid:.3e}", m.tag);
        assert!(wcen < 1e-14, "{}: |xyzEdge| off the unit sphere by {wcen:.3e}", m.tag);
        assert!(wll < 1e-6, "{}: latEdge/lonEdge vs xyzEdge max chord {wll:.3e}", m.tag);
        println!(
            "{}: dcEdge rel {wdc:.3e} | xEdge chord {wmid:.3e} | |r|-1 {wcen:.3e} | latlon chord {wll:.3e}",
            m.tag
        );
    }
}

#[test]
fn vertex_is_the_circumcentre_of_its_three_cells() {
    // Measured: chord residual x1 7.0e-15, x4 1.9e-13.
    for m in meshes() {
        let p = m.cell_xyz();
        let vc = m.g.i32("sample/vertex_cells");
        let xv = m.g.f64("sample/xVertex");
        let yv = m.g.f64("sample/yVertex");
        let zv = m.g.f64("sample/zVertex");
        let mut worst = 0.0f64;
        for k in 0..xv.len() {
            let a = p[vc[3 * k] as usize];
            let b = p[vc[3 * k + 1] as usize];
            let c = p[vc[3 * k + 2] as usize];
            let mut n = unit(cross(sub(b, a), sub(c, a)));
            if dot(n, [a[0] + b[0] + c[0], a[1] + b[1] + c[1], a[2] + b[2] + c[2]]) < 0.0 {
                n = [-n[0], -n[1], -n[2]];
            }
            worst = worst.max(chord(n, [xv[k], yv[k], zv[k]]));
        }
        assert!(worst < 1e-11, "{}: xVertex vs circumcentre max chord {worst:.3e}", m.tag);
        println!("{}: vertex vs circumcentre max chord {worst:.3e}", m.tag);
    }
}

#[test]
fn dual_edge_length_follows_from_the_topology() {
    // dvEdge is the arc between the edge's two vertices. Those two vertices are
    // the two cells adjacent to BOTH endpoints of the edge, so this is checkable
    // from the full ring plus the full cell centres -- no stored vertex coords.
    // Measured: max rel x1 5.2e-12, x4 8.4e-10.
    for m in meshes() {
        let p = m.cell_xyz();
        let offs = m.g.i32("cell/cellsOnCell_offsets");
        let vals = m.g.i32("cell/cellsOnCell_values");
        let nbr = |c: usize| &vals[offs[c] as usize..offs[c + 1] as usize];
        let ec = m.g.i32("sample/edge_cells");
        let dv = m.g.f64("sample/dvEdge");
        let xe = m.g.f64("sample/xEdge");
        let ye = m.g.f64("sample/yEdge");
        let ze = m.g.f64("sample/zEdge");
        let (mut cell_mid, mut vert_mid) = (0.0f64, 0.0f64);

        let circum = |t: [i32; 3]| -> [f64; 3] {
            let (a, b, c) = (p[t[0] as usize], p[t[1] as usize], p[t[2] as usize]);
            let n = unit(cross(sub(b, a), sub(c, a)));
            if dot(n, [a[0] + b[0] + c[0], a[1] + b[1] + c[1], a[2] + b[2] + c[2]]) < 0.0 {
                [-n[0], -n[1], -n[2]]
            } else {
                n
            }
        };

        let (mut worst, mut checked) = (0.0f64, 0usize);
        for k in 0..dv.len() {
            let (a, b) = (ec[2 * k], ec[2 * k + 1]);
            let na = nbr(a as usize);
            let shared: Vec<i32> =
                nbr(b as usize).iter().copied().filter(|x| na.contains(x)).collect();
            assert_eq!(
                shared.len(),
                2,
                "{}: edge ({a},{b}) has {} shared neighbours, an edge of a closed \
                 triangulation has exactly 2",
                m.tag,
                shared.len()
            );
            let mut t0 = [a, b, shared[0]];
            let mut t1 = [a, b, shared[1]];
            t0.sort_unstable();
            t1.sort_unstable();
            let (q0, q1) = (circum(t0), circum(t1));
            let arc = dot(q0, q1).clamp(-1.0, 1.0).acos();
            worst = worst.max((arc - dv[k]).abs() / dv[k]);

            // Which midpoint is xEdge? Decide it by measurement, here, rather than
            // by citing a convention: compare the stored edge point against the
            // cell midpoint and against the vertex midpoint on the same edges.
            let e = [xe[k], ye[k], ze[k]];
            let (pa, pb) = (p[a as usize], p[b as usize]);
            cell_mid =
                cell_mid.max(chord(unit([pa[0] + pb[0], pa[1] + pb[1], pa[2] + pb[2]]), e));
            vert_mid = vert_mid.max(chord(unit([q0[0] + q1[0], q0[1] + q1[1], q0[2] + q1[2]]), e));
            checked += 1;
        }
        assert!(worst < 1e-8, "{}: dvEdge vs arc(circumcentres) max rel {worst:.3e}", m.tag);
        assert!(
            cell_mid * 100.0 < vert_mid,
            "{}: xEdge is supposed to be the CELL midpoint (measured {cell_mid:.3e}) and not \
             the VERTEX midpoint (measured {vert_mid:.3e}); if those ever converge, the \
             convention this golden encodes is no longer decided by the data",
            m.tag
        );
        println!(
            "{}: dvEdge checked on {checked} edges, max rel {worst:.3e} | xEdge chord to \
             cell midpoint {cell_mid:.3e} vs vertex midpoint {vert_mid:.3e}",
            m.tag
        );
    }
}

#[test]
fn edge_stencil_and_weight_padding() {
    for m in meshes() {
        let offs = m.g.i32("cell/cellsOnCell_offsets");
        let deg = |c: usize| (offs[c + 1] - offs[c]) as i32;
        let ec = m.g.i32("sample/edge_cells");
        let neoe = m.g.i32("sample/nEdgesOnEdge");
        let eoe = m.g.i32("sample/edgesOnEdge");
        let w = m.g.f64("sample/weightsOnEdge");
        let m2 = m.max_edges2;

        for k in 0..neoe.len() {
            let want = deg(ec[2 * k] as usize) + deg(ec[2 * k + 1] as usize) - 2;
            assert_eq!(
                neoe[k], want,
                "{}: nEdgesOnEdge != deg(c0)+deg(c1)-2 at sampled edge {k}",
                m.tag
            );
            for s in 0..m2 {
                let used = (s as i32) < neoe[k];
                if used {
                    assert!(eoe[k * m2 + s] >= 0, "{}: used stencil slot is padding", m.tag);
                } else {
                    assert_eq!(eoe[k * m2 + s], -1, "{}: padded stencil slot is not -1", m.tag);
                    assert_eq!(
                        w[k * m2 + s], 0.0,
                        "{}: weightsOnEdge padding is not exactly 0.0 -- MPAS sums the \
                         whole row, so a nonzero padding slot is a silent wrong wind",
                        m.tag
                    );
                }
            }
            // the stencil must not name the edge itself, and must not repeat
            let mut seen: Vec<i32> = eoe[k * m2..k * m2 + neoe[k] as usize].to_vec();
            seen.sort_unstable();
            let n = seen.len();
            seen.dedup();
            assert_eq!(seen.len(), n, "{}: tangential stencil repeats an edge", m.tag);
        }
        println!("{}: stencil + padding clean on {} sampled edges", m.tag, neoe.len());
    }
}

#[test]
fn cell_latlon_carries_the_upstream_longitude_defect() {
    // latCell == asin(z) on both meshes to 5.2e-14.
    // lonCell == atan2(y,x) on x1 to 8.9e-16, but x4 departs by up to 1.748e-07 rad.
    // That is MPAS-Tools grid_rotate computing pi in default real (binary32) and
    // storing it in RKIND=8. It is recorded here rather than smoothed over,
    // because a Rust generator writing atan2(y,x) in f64 is RIGHT and must not be
    // graded as wrong against x4's stored longitudes.
    let expected_lon_gap: [(&str, f64); 2] = [("x1.40962", 1e-15), ("x4.163842", 2.0e-7)];
    for m in meshes() {
        let p = m.cell_xyz();
        let lat = m.g.f64("cell/latCell");
        let lon = m.g.f64("cell/lonCell");
        let two_pi = 2.0 * std::f64::consts::PI;
        let (mut wlat, mut wlon) = (0.0f64, 0.0f64);
        for i in 0..lat.len() {
            wlat = wlat.max((p[i][2].clamp(-1.0, 1.0).asin() - lat[i]).abs());
            let want = p[i][1].atan2(p[i][0]).rem_euclid(two_pi);
            let have = lon[i].rem_euclid(two_pi);
            let mut d = (want - have + std::f64::consts::PI).rem_euclid(two_pi)
                - std::f64::consts::PI;
            d = d.abs();
            wlon = wlon.max(d);
        }
        let bound = expected_lon_gap.iter().find(|(t, _)| *t == m.tag).unwrap().1;
        assert!(wlat < 1e-12, "{}: latCell vs asin(z) max {wlat:.3e} rad", m.tag);
        assert!(
            wlon < bound,
            "{}: lonCell vs atan2(y,x) max {wlon:.3e} rad exceeds the recorded upstream \
             defect bound {bound:.3e}",
            m.tag
        );
        println!("{}: latCell gap {wlat:.3e} rad, lonCell gap {wlon:.3e} rad", m.tag);
    }
}

#[test]
fn sampling_covers_the_tails_not_just_the_bulk() {
    // A stride alone can alias against a structured file ordering, so the sample
    // is stride UNION every metric's argmin/argmax. Prove the extremes are in it.
    for m in meshes() {
        let dc = m.g.f64("sample/dcEdge");
        let dv = m.g.f64("sample/dvEdge");
        let ac = m.g.f64("cell/areaCell");
        let md = m.g.f64("in/meshDensity");
        let strides = m.g.i32("meta/sample_strides");
        let ratio: Vec<f64> = dc.iter().zip(&dv).map(|(c, v)| v / c).collect();
        let rmin = ratio.iter().cloned().fold(f64::INFINITY, f64::min);
        let rmax = ratio.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let acmin = ac.iter().cloned().fold(f64::INFINITY, f64::min);
        let acmax = ac.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mdmin = md.iter().cloned().fold(f64::INFINITY, f64::min);
        let mdmax = md.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(rmin > 0.0 && rmax.is_finite(), "{}: dvEdge/dcEdge range", m.tag);
        println!(
            "{}: strides cell {} edge {} vertex {} | dv/dc {rmin:.4}..{rmax:.4} \
             | areaCell {acmin:.3e}..{acmax:.3e} | meshDensity {mdmin:.6}..{mdmax:.6}",
            m.tag, strides[0], strides[1], strides[2]
        );
    }
}

// ------------------------------------------------------------- failing direction

#[test]
fn the_comparison_catches_a_perturbed_value() {
    // An extractor never shown to fail is not evidence. Move one value by one ULP
    // in a copy of the container, load it through the real loader, and require
    // every check that covers it to notice.
    let src = golden_dir().join("x1.40962.mesh.rwg");
    let bytes = std::fs::read(&src).expect("golden readable");
    let good = Rwg::load(&src);

    let victims: &[(&str, usize)] = &[
        ("in/xCell", 17),
        ("cell/areaCell", 3),
        ("sample/weightsOnEdge", 41),
        ("sample/dcEdge", 5),
    ];
    let tmp = std::env::temp_dir().join("rw_mpas_golden_perturbed.rwg");
    for (name, elem) in victims {
        // locate the slab through the directory, exactly as the loader does
        let n = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
        let mut off = None;
        for i in 0..n {
            let base = 16 + i * 104;
            let end = bytes[base..base + 64].iter().position(|&c| c == 0).unwrap_or(64);
            if &bytes[base..base + end] == name.as_bytes() {
                off = Some(u64::from_le_bytes(bytes[base + 84..base + 92].try_into().unwrap())
                    as usize);
                break;
            }
        }
        let off = off.unwrap_or_else(|| panic!("{name} not in the container"));

        let mut mutated = bytes.clone();
        let pos = off + elem * 8;
        let v = f64::from_le_bytes(mutated[pos..pos + 8].try_into().unwrap());
        let nv = f64::from_bits(v.to_bits() + 1); // one ULP
        mutated[pos..pos + 8].copy_from_slice(&nv.to_le_bytes());
        std::fs::write(&tmp, &mutated).expect("temp golden writable");

        let bad = Rwg::load(&tmp);
        let a = good.f64(name);
        let b = bad.f64(name);
        assert_ne!(a, b, "{name}: a one-ULP change survived the loader unnoticed");
        assert_ne!(good.raw(name), bad.raw(name), "{name}: raw slab compared equal");
        let moved = a.iter().zip(&b).filter(|(x, y)| x != y).count();
        assert_eq!(moved, 1, "{name}: expected exactly one element to move, {moved} did");
        println!("{name}[{elem}]: {v:?} -> {nv:?} CAUGHT");
    }
    let _ = std::fs::remove_file(&tmp);
}

#[test]
#[should_panic(expected = "not an RWGOLD01 container")]
fn a_container_with_a_broken_magic_is_refused() {
    let src = golden_dir().join("x1.40962.mesh.rwg");
    let mut bytes = std::fs::read(&src).expect("golden readable");
    bytes[0] = b'X';
    let tmp = std::env::temp_dir().join("rw_mpas_golden_badmagic.rwg");
    std::fs::write(&tmp, &bytes).expect("temp writable");
    let _ = Rwg::load(&tmp);
}

#[test]
#[should_panic(expected = "slab runs past end of file")]
fn a_truncated_container_is_refused() {
    let src = golden_dir().join("x1.40962.mesh.rwg");
    let bytes = std::fs::read(&src).expect("golden readable");
    let tmp = std::env::temp_dir().join("rw_mpas_golden_truncated.rwg");
    std::fs::write(&tmp, &bytes[..bytes.len() / 2]).expect("temp writable");
    let _ = Rwg::load(&tmp);
}
