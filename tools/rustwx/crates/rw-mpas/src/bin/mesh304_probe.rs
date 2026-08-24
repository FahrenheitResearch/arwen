//! Measurement instrument for the dislocation-scar defect class. It reads the
//! generator's own public modules and changes nothing: the same seed -> Lloyd
//! -> Delaunay -> derive pipeline `generate` runs, snapshotting min dvEdge at
//! chosen sweep counts and dumping the geometry of the worst dual edges so a
//! degenerate corner can be named rather than suspected.
//!
//! `--seed fibonacci` (default) reruns the pipeline that produced the defect;
//! `--seed goldberg` reruns the icosahedral seeding that replaced it for
//! uniform requests. The pair is the before/after evidence: at 2,000 cells,
//! sweep-104 min dvEdge 7,337.6 m REFUSED against 216,267.4 m PASS.

use rw_mpas::mesh::density::MeshSpec;
use rw_mpas::mesh::derive::MpasMesh;
use rw_mpas::mesh::emit::nominal_min_dc_from_m;
use rw_mpas::mesh::geom::{
    EARTH_RADIUS_M, V3, add, arc, circumcenter, lat_lon, scale, sub, tri_area, unit,
};
use rw_mpas::mesh::hull::delaunay_rings;
use rw_mpas::mesh::lloyd::seed_points;
use rw_mpas::mesh::validate::{Limits, validate};

/// One Lloyd step for one cell -- a verbatim re-statement of the private
/// `lloyd::cell_step` so the probe can run a fixed sweep count without
/// touching the generator.
fn cell_step(points: &[V3], rings: &rw_mpas::mesh::derive::Rings, spec: &MeshSpec, i: usize) -> (V3, f64) {
    let ring = rings.ring(i);
    let deg = ring.len();
    let centre = points[i];
    let mut verts: Vec<V3> = Vec::with_capacity(deg);
    for j in 0..deg {
        let prev = ring[(j + deg - 1) % deg] as usize;
        let cur = ring[j] as usize;
        match circumcenter(centre, points[prev], points[cur]) {
            Some(v) => verts.push(v),
            None => return (centre, 0.0),
        }
    }
    let mut moment: V3 = [0.0; 3];
    let mut area = 0.0f64;
    for j in 0..deg {
        let a = verts[j];
        let b = verts[(j + 1) % deg];
        let tri = tri_area(centre, a, b);
        area += tri;
        for (p, q) in [(centre, a), (a, b), (b, centre)] {
            if let Some(m) = unit(add(p, q)) {
                let w = tri / 3.0 * spec.density(m);
                moment = add(moment, scale(m, w));
            }
        }
    }
    let centroid = match unit(moment) {
        Some(c) => c,
        None => return (centre, 0.0),
    };
    let h = (2.0 * area.abs() / 3f64.sqrt()).sqrt();
    let delta = arc(centre, centroid);
    let ratio = if h > 0.0 { delta / h } else { 0.0 };
    (centroid, ratio)
}

struct Snap {
    sweep: usize,
    mean_res: f64,
    min_dv_m: f64,
    min_dv_over_dc: f64,
    under_1km: usize,
    under_100m: usize,
    under_10m: usize,
    worst_edge_cells: (usize, usize),
    worst_quad_extra: (usize, usize),
}

fn snapshot(points: &[V3], spec: &MeshSpec, sweep: usize, mean_res: f64, top: usize, dump: bool) -> Snap {
    let rings = delaunay_rings(&points.to_vec()).expect("delaunay");
    let density: Vec<f64> = points.iter().map(|&p| spec.density(p)).collect();
    let nominal = nominal_min_dc_from_m(spec.finest_km() * 1000.0);
    let mesh = MpasMesh::derive(points.to_vec(), density, &rings, nominal).expect("derive");
    let r = EARTH_RADIUS_M;
    let ne = mesh.n_edges;
    let mut order: Vec<usize> = (0..ne).collect();
    order.sort_by(|&a, &b| mesh.dv_edge[a].partial_cmp(&mesh.dv_edge[b]).unwrap());
    let dv_m = |e: usize| mesh.dv_edge[e] * r;
    let under = |t: f64| (0..ne).filter(|&e| dv_m(e) < t).count();
    let e0 = order[0];

    // For each of the top-K shortest dual edges: the quad that generates it.
    let mut worst_quad = (0usize, 0usize);
    if dump {
        println!("  -- worst {top} dual edges at sweep {sweep} --");
    }
    for &e in order.iter().take(top) {
        let (c1, c2) = (mesh.cells_on_edge[2 * e] as usize, mesh.cells_on_edge[2 * e + 1] as usize);
        let (v1, v2) = (
            mesh.vertices_on_edge[2 * e] as usize,
            mesh.vertices_on_edge[2 * e + 1] as usize,
        );
        let tri1: Vec<usize> = (0..3).map(|s| mesh.cells_on_vertex[v1 * 3 + s] as usize).collect();
        let tri2: Vec<usize> = (0..3).map(|s| mesh.cells_on_vertex[v2 * 3 + s] as usize).collect();
        let a = *tri1.iter().find(|c| !tri2.contains(c)).unwrap_or(&c1);
        let b = *tri2.iter().find(|c| !tri1.contains(c)).unwrap_or(&c2);
        if e == e0 {
            worst_quad = (a, b);
        }
        if dump {
            // Cocircularity: distance of b from the circumcircle through (c1,c2,a).
            let r1 = arc(mesh.vertex_xyz[v1], mesh.cell_xyz[c1]);
            let dev_b = (arc(mesh.vertex_xyz[v1], mesh.cell_xyz[b]) - r1).abs() * r;
            let r2 = arc(mesh.vertex_xyz[v2], mesh.cell_xyz[c1]);
            let dev_a = (arc(mesh.vertex_xyz[v2], mesh.cell_xyz[a]) - r2).abs() * r;
            let (lat, lon) = lat_lon(mesh.edge_xyz[e]);
            let degs: Vec<i32> = [c1, c2, a, b].iter().map(|&c| mesh.n_edges_on_cell[c]).collect();
            // Quad shape in the tangent plane at the edge point: the four sites.
            let quad_arcs = [
                arc(mesh.cell_xyz[c1], mesh.cell_xyz[a]) * r,
                arc(mesh.cell_xyz[a], mesh.cell_xyz[c2]) * r,
                arc(mesh.cell_xyz[c2], mesh.cell_xyz[b]) * r,
                arc(mesh.cell_xyz[b], mesh.cell_xyz[c1]) * r,
            ];
            let diag1 = arc(mesh.cell_xyz[c1], mesh.cell_xyz[c2]) * r;
            let diag2 = arc(mesh.cell_xyz[a], mesh.cell_xyz[b]) * r;
            println!(
                "  edge {e}: dv {:9.2} m  dc {:9.0} m  dv/dc {:.2e}  cells ({c1},{c2}) quad extras ({a},{b})  degs {:?}  lat/lon ({:6.1},{:6.1}) deg  cocirc-dev a {:8.2} m b {:8.2} m  quad sides [{:.0},{:.0},{:.0},{:.0}] m diags [{:.0},{:.0}] m  circumR {:8.0} m",
                dv_m(e),
                mesh.dc_edge[e] * r,
                mesh.dv_edge[e] / mesh.dc_edge[e],
                degs,
                lat.to_degrees(),
                lon.to_degrees(),
                dev_a,
                dev_b,
                quad_arcs[0], quad_arcs[1], quad_arcs[2], quad_arcs[3],
                diag1, diag2,
                r1 * r,
            );
        }
    }

    Snap {
        sweep,
        mean_res,
        min_dv_m: dv_m(e0),
        min_dv_over_dc: mesh.dv_edge[e0] / mesh.dc_edge[e0],
        under_1km: under(1000.0),
        under_100m: under(100.0),
        under_10m: under(10.0),
        worst_edge_cells: (
            mesh.cells_on_edge[2 * e0] as usize,
            mesh.cells_on_edge[2 * e0 + 1] as usize,
        ),
        worst_quad_extra: worst_quad,
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let get = |key: &str, dflt: &str| -> String {
        args.iter()
            .position(|a| a == key)
            .and_then(|i| args.get(i + 1))
            .cloned()
            .unwrap_or_else(|| dflt.to_string())
    };
    let n: usize = get("--cells", "2000").parse().unwrap();
    let bg: f64 = get("--bg-km", "120").parse().unwrap();
    let omega: f64 = get("--omega", "1.4").parse().unwrap();
    let top: usize = get("--top", "6").parse().unwrap();
    let snaps: Vec<usize> = get("--snapshots", "10,25,50,100,150,200,300,400")
        .split(',')
        .map(|s| s.parse().unwrap())
        .collect();
    let max_sweep = *snaps.iter().max().unwrap();

    let spec = MeshSpec::uniform(bg);
    let seed_kind = get("--seed", "fibonacci");
    let mut points = match seed_kind.as_str() {
        "goldberg" => {
            let choice = rw_mpas::mesh::icosa::snap_cells(n, false).expect("snap");
            println!(
                "seed: goldberg GP({},{}) -> {} cells (requested {n}, snap {:+.3}%)",
                choice.m,
                choice.n,
                choice.cells,
                (choice.cells as f64 / n as f64 - 1.0) * 100.0
            );
            rw_mpas::mesh::icosa::seed(choice.m, choice.n).expect("seed")
        }
        _ => seed_points(&spec, n).expect("seed"),
    };
    println!("probe: {} cells, bg {bg} km, omega {omega}, seed {seed_kind}, snapshots {snaps:?}", points.len());

    // sweep 0 = the raw seed
    let mut rows: Vec<Snap> = Vec::new();
    if snaps.contains(&0) {
        rows.push(snapshot(&points, &spec, 0, f64::NAN, top, false));
    }
    let mut rings = delaunay_rings(&points).expect("delaunay");
    let mut first_below_tol: Option<usize> = None;
    for sweep in 1..=max_sweep {
        let step: Vec<(V3, f64)> = (0..points.len())
            .map(|i| cell_step(&points, &rings, &spec, i))
            .collect();
        let mean = step.iter().map(|(_, r)| *r).sum::<f64>() / step.len() as f64;
        for (i, (c, _)) in step.iter().enumerate() {
            let d = sub(*c, points[i]);
            points[i] = unit(add(points[i], scale(d, omega))).unwrap_or(*c);
        }
        if mean < 1e-3 && first_below_tol.is_none() {
            first_below_tol = Some(sweep);
        }
        rings = delaunay_rings(&points).expect("delaunay");
        if snaps.contains(&sweep) {
            let dump = sweep == max_sweep;
            rows.push(snapshot(&points, &spec, sweep, mean, top, dump));
        }
    }

    println!("\nsweep  mean(d/h)   min_dv_m    min_dv/dc  <1km <100m <10m  worst-edge cells (quad extras)");
    for s in &rows {
        println!(
            "{:5}  {:9.3e}  {:10.2}  {:9.2e}  {:4} {:5} {:4}  ({},{}) ({},{})",
            s.sweep, s.mean_res, s.min_dv_m, s.min_dv_over_dc, s.under_1km, s.under_100m, s.under_10m,
            s.worst_edge_cells.0, s.worst_edge_cells.1, s.worst_quad_extra.0, s.worst_quad_extra.1
        );
    }
    if let Some(s) = first_below_tol {
        println!("shipped default (tolerance 1e-3) would have stopped at sweep {s}");
    }

    // Optional targeted-repair experiment: nudge one site of each offending
    // quad off cocircularity, relax a few sweeps, re-check. Measures whether
    // a min-dvEdge floor is reachable by local repair rather than by luck.
    let floor_m: f64 = get("--repair-floor-m", "0").parse().unwrap();
    if floor_m > 0.0 {
        let r = EARTH_RADIUS_M;
        println!("\nrepair loop: floor {floor_m} m, nudge 3% of local spacing, 5 sweeps per round");
        for round in 0..20 {
            let rg = delaunay_rings(&points).expect("delaunay");
            let dens: Vec<f64> = points.iter().map(|&p| spec.density(p)).collect();
            let nom = nominal_min_dc_from_m(spec.finest_km() * 1000.0);
            let mesh = MpasMesh::derive(points.clone(), dens, &rg, nom).expect("derive");
            let spacing = mesh.spacing_m();
            let offending: Vec<usize> = (0..mesh.n_edges)
                .filter(|&e| mesh.dv_edge[e] * r < floor_m)
                .collect();
            let min_dv = (0..mesh.n_edges)
                .map(|e| mesh.dv_edge[e] * r)
                .fold(f64::INFINITY, f64::min);
            println!("  round {round}: min dv {min_dv:9.2} m, {} edges under floor", offending.len());
            if offending.is_empty() {
                break;
            }
            let mut nudged: std::collections::BTreeSet<usize> = Default::default();
            for &e in &offending {
                let v1 = mesh.vertices_on_edge[2 * e] as usize;
                let tri1: Vec<usize> =
                    (0..3).map(|s| mesh.cells_on_vertex[v1 * 3 + s] as usize).collect();
                let tri2v = mesh.vertices_on_edge[2 * e + 1] as usize;
                let tri2: Vec<usize> =
                    (0..3).map(|s| mesh.cells_on_vertex[tri2v * 3 + s] as usize).collect();
                let site = *tri1
                    .iter()
                    .find(|c| !tri2.contains(c))
                    .unwrap_or(&(mesh.cells_on_edge[2 * e] as usize));
                if !nudged.insert(site) {
                    continue;
                }
                // Push the site away from the degenerate dual vertex, in the
                // tangent plane: a generic direction off the shared circle.
                let away = rw_mpas::mesh::geom::tangent_at(
                    points[site],
                    sub(points[site], mesh.vertex_xyz[v1]),
                );
                if let Some(dir) = unit(away) {
                    let amp = 0.03 * spacing[site] / r; // radians on the unit sphere
                    points[site] = unit(add(points[site], scale(dir, amp))).unwrap();
                }
            }
            let mut rg2 = delaunay_rings(&points).expect("delaunay");
            for _ in 0..5 {
                let step: Vec<(V3, f64)> = (0..points.len())
                    .map(|i| cell_step(&points, &rg2, &spec, i))
                    .collect();
                for (i, (c, _)) in step.iter().enumerate() {
                    let d = sub(*c, points[i]);
                    points[i] = unit(add(points[i], scale(d, omega))).unwrap_or(*c);
                }
                rg2 = delaunay_rings(&points).expect("delaunay");
            }
            rings = rg2;
        }
    }

    // Final validate verdict with the DEFAULT limits, same as generate().
    let density: Vec<f64> = points.iter().map(|&p| spec.density(p)).collect();
    let nominal = nominal_min_dc_from_m(spec.finest_km() * 1000.0);
    let mesh = MpasMesh::derive(points.clone(), density, &rings, nominal).expect("derive");
    match validate(&mesh, Limits::default()) {
        Ok(rep) => println!(
            "validate: PASS  min_dv_edge_m {:.1}  min_dv_over_dc {:.3e}  min_dc_edge_m {:.1}  nonorth {:.2e}  coordination {:?}",
            rep.min_dv_edge_m, rep.min_dv_over_dc, rep.min_dc_edge_m, rep.max_nonorthogonality,
            rep.coordination_histogram
        ),
        Err(e) => println!("validate: REFUSED: {e}"),
    }
}
