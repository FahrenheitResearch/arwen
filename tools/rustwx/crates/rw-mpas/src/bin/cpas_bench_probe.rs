//! `cpas_bench_probe` -- measure a resolution spec, and a grid built from it,
//! on the metrics the CPAS 200 m Hong Kong paper publishes.
//!
//! WHY THIS EXISTS. The published benchmark (Sze et al. 2022, Earth and Space
//! Science 9, e2022EA002342) reports two things `validate.rs` does not measure:
//! the count of OBTUSE Delaunay triangles (they claim zero) and whether every
//! Voronoi edge crosses exactly one Delaunay edge (well-centredness). The
//! crate's validator checks edge ORIENTATION -- (n x t).r > 0, a handedness
//! lock -- and non-orthogonality, and neither of those is well-centredness: an
//! obtuse triangle keeps its handedness and can sit inside the orthogonality
//! bound. So the published quality claim could not be answered from a receipt.
//!
//! It also measures the spec field with a RADIAL sampler, along the great
//! circle out of the refinement centre, where the field actually varies.
//!
//! THE METER THIS PROBE WAS WRITTEN AGAINST NO LONGER EXISTS. It called a
//! `steepest_gradient_per_cell` that walked a bare 50,000-point Fibonacci
//! lattice, whose points sit about 100 km apart, so a 200 m core inside a
//! 10 km cap and its ramp fit between two samples and the gate's own
//! instrument stepped clean over the transition it exists to police. The
//! release line replaced it with [`steepest_gradient_reading`], which unions
//! REGION-TARGETED variation probes onto the lattice and reports a
//! [`Coverage`] the ladder refuses on when it is partial. The rows below now
//! read that meter, and the coverage verdict is printed beside every one:
//! any gradient number measured with the old meter -- including the ones in
//! this benchmark's own write-up -- has to be re-measured before it is
//! quoted, because it was taken with an instrument that could miss the
//! feature entirely.
//!
//! Nothing here changes a gate, a floor or a threshold. It reads.
//!
use std::path::PathBuf;

use rw_mpas::mesh::density::{MeshSpec, fibonacci_lattice};
use rw_mpas::mesh::derive::MpasMesh;
use rw_mpas::mesh::geom::{EARTH_RADIUS_M, V3, arc, from_lat_lon, unit};
use rw_mpas::mesh::gridread::read_grid_generators;
use rw_mpas::mesh::hull::delaunay_rings;

/// Great-circle point at arc `theta` from `centre` along unit tangent `bearing`.
fn rotate_from(centre: V3, bearing: V3, theta: f64) -> V3 {
    let a = [
        centre[0] * theta.cos() + bearing[0] * theta.sin(),
        centre[1] * theta.cos() + bearing[1] * theta.sin(),
        centre[2] * theta.cos() + bearing[2] * theta.sin(),
    ];
    unit(a).expect("great-circle point")
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let mut spec_path: Option<PathBuf> = None;
    let mut grid_path: Option<PathBuf> = None;
    let mut centre_deg = (22.30f64, 114.2f64);
    let mut max_km = 3000.0f64;
    let mut transect = false;
    let mut brief = false;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--spec" => {
                spec_path = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--grid" => {
                grid_path = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--brief" => {
                brief = true;
                i += 1;
            }
            "--transect" => {
                transect = true;
                i += 1;
            }
            "--centre-deg" => {
                centre_deg = (
                    args[i + 1].parse().map_err(|e| format!("{e}"))?,
                    args[i + 2].parse().map_err(|e| format!("{e}"))?,
                );
                i += 3;
            }
            "--max-km" => {
                max_km = args[i + 1].parse().map_err(|e| format!("{e}"))?;
                i += 2;
            }
            other => return Err(format!("cpas_bench_probe: unknown argument {other}")),
        }
    }

    if let Some(p) = &spec_path {
        let text = std::fs::read_to_string(p).map_err(|e| format!("{}: {e}", p.display()))?;
        let spec = MeshSpec::from_json(&text).map_err(|e| format!("{e}"))?;
        let prepared = spec.prepared();
        let centre = from_lat_lon(centre_deg.0.to_radians(), centre_deg.1.to_radians());
        let (east, _north) = rw_mpas::mesh::geom::east_north(centre).ok_or("no tangent frame")?;

        // ---- radial transect: walk outward one CELL at a time -------------
        // Stepping by the local spacing is exactly what the gate's meter does;
        // doing it radially guarantees every cell of the ramp is visited.
        let mut r_m = 0.0f64;
        let mut worst_g = 0.0f64;
        let mut worst_r = 0.0f64;
        let mut worst_h = 0.0f64;
        let mut rows: Vec<(f64, f64, f64)> = Vec::new();
        let mut cells_out = 0.0f64;
        while r_m < max_km * 1000.0 {
            let p = rotate_from(centre, east, r_m / EARTH_RADIUS_M);
            let h = prepared.spacing_m(p);
            let q = rotate_from(centre, east, (r_m + h) / EARTH_RADIUS_M);
            let hq = prepared.spacing_m(q);
            let g = (hq / h - 1.0).abs();
            if g > worst_g {
                worst_g = g;
                worst_r = r_m;
                worst_h = h;
            }
            rows.push((r_m / 1000.0, h / 1000.0, g * 100.0));
            r_m += h;
            cells_out += 1.0;
        }
        println!(
            "# RADIAL TRANSECT from ({:.4}, {:.4}), one cell per step",
            centre_deg.0, centre_deg.1
        );
        println!("radial_steps_to_max_km\t{max_km:.0}\t{cells_out:.0}");
        println!(
            "radial_max_gradient_pct_per_cell\t{:.4}\tat_radius_km\t{:.3}\tat_spacing_km\t{:.4}",
            worst_g * 100.0,
            worst_r / 1000.0,
            worst_h / 1000.0
        );
        if transect {
            println!("# r_km\th_km\tgrad_pct_per_cell");
            for (k, (r, h, g)) in rows.iter().enumerate() {
                if *r < 400.0 || k % 20 == 0 {
                    println!("T\t{r:.3}\t{h:.4}\t{g:.3}");
                }
            }
        }

        // ---- what the GATE's own instrument reads -------------------------
        let lat_n: &[usize] = if brief { &[50_000] } else { &[20_000, 50_000, 200_000, 2_000_000, 20_000_000] };
        for &n in lat_n {
            let reading = rw_mpas::mesh::density::steepest_gradient_reading_of(&prepared, n);
            println!(
                "reading_gradient_pct_per_cell\tsamples\t{n}\t{:.4}\tcoverage\t{:?}\tprobes\t{}",
                reading.per_cell * 100.0,
                reading.coverage,
                reading.probe_points
            );
        }

        // ---- cell-count integral, lattice vs exact axisymmetric quadrature -
        let siz_n: &[usize] = if brief { &[200_000] } else { &[50_000, 200_000, 2_000_000, 20_000_000] };
        for &n in siz_n {
            let mut acc = 0.0f64;
            for p in fibonacci_lattice(n) {
                let h = prepared.spacing_m(p);
                acc += 1.0 / (h * h);
            }
            let mean = acc / n as f64;
            let area = 4.0 * std::f64::consts::PI * EARTH_RADIUS_M * EARTH_RADIUS_M;
            println!(
                "lattice_predicted_cells\tsamples\t{n}\t{:.0}",
                area * mean / (3f64.sqrt() / 2.0)
            );
        }
        // Axisymmetric: N = 2*pi*R^2 * int_0^pi sin(t)/h(t)^2 dt / (sqrt3/2).
        // Every region is a cap centred at the same point, so the field is a
        // function of theta alone and this quadrature is exact to its step.
        let steps = if brief { 2_000_000usize } else { 20_000_000usize };
        let mut acc = 0.0f64;
        for k in 0..steps {
            let t = (k as f64 + 0.5) * std::f64::consts::PI / steps as f64;
            let p = rotate_from(centre, east, t);
            let h = prepared.spacing_m(p);
            acc += t.sin() / (h * h);
        }
        acc *= std::f64::consts::PI / steps as f64;
        let n_exact =
            2.0 * std::f64::consts::PI * EARTH_RADIUS_M * EARTH_RADIUS_M * acc / (3f64.sqrt() / 2.0);
        println!("axisymmetric_predicted_cells\t{n_exact:.0}");

        // ---- the gate's arithmetic, reported not applied -------------------
        let gate_reading = spec.steepest_gradient_reading(50_000);
        let gate_g: f64 = gate_reading.per_cell;
        let band = (2f64).ln() / (1.0 + gate_g).ln();
        println!("gate_reads_gradient_pct_per_cell\t{:.4}", gate_g * 100.0);
        println!("gate_reads_coverage\t{:?}", gate_reading.coverage);
        println!("gate_reads_band_cells\t{band:.4}\tgate_requires\t6.0");
        let limit = (2f64.ln() / 6.0).exp() - 1.0;
        println!("gate_limit_pct_per_cell\t{:.4}", limit * 100.0);
        println!(
            "gate_would_bind\t{}",
            if gate_g > limit { "YES" } else { "NO" }
        );
        let radial_band = (2f64).ln() / (1.0 + worst_g).ln();
        println!("radial_band_cells\t{radial_band:.4}");
        println!(
            "gate_would_bind_on_radial_reading\t{}",
            if worst_g > limit { "YES" } else { "NO" }
        );
    }

    if let Some(p) = &grid_path {
        let grid = read_grid_generators(p)?;
        let n_cells = grid.points.len();
        println!("grid_cells\t{n_cells}");
        let rings = delaunay_rings(&grid.points).map_err(|e| format!("{e}"))?;
        let mesh = MpasMesh::derive(
            grid.points.clone(),
            grid.mesh_density.clone(),
            &rings,
            grid.nominal_min_dc,
        )
        .map_err(|e| format!("{e}"))?;

        // ---- obtuse Delaunay triangles ------------------------------------
        // ONE COPY OF THIS ARITHMETIC. It lives in `mesh::validate`, where it
        // now runs on every mesh the gate sees; this probe reads the same
        // function so the benchmark and the receipt can never disagree.
        let wc = rw_mpas::mesh::validate::well_centredness(&mesh);
        let (obtuse, worst_angle, worst_tri) = (
            wc.obtuse_triangles,
            wc.max_delaunay_angle_deg.to_radians(),
            wc.worst_triangle,
        );
        println!(
            "obtuse_delaunay_triangles\t{obtuse}\tof\t{}",
            mesh.n_vertices
        );
        println!(
            "max_delaunay_angle_deg\t{:.4}\tat_triangle\t{worst_tri}",
            worst_angle.to_degrees()
        );

        // ---- well-centredness: does each Voronoi edge cross its Delaunay edge
        let not_crossing = wc.non_crossing_dual_edges;
        let crossing = mesh.n_edges - not_crossing;
        println!(
            "voronoi_edges_crossing_their_delaunay_edge\t{crossing}\tof\t{}",
            mesh.n_edges
        );
        println!("voronoi_edges_NOT_crossing\t{not_crossing}");

        // ---- the crate's own quality numbers, on the same mesh -------------
        let mut min_dv = f64::INFINITY;
        let mut min_ratio = f64::INFINITY;
        let mut min_dc = f64::INFINITY;
        let mut max_dc = 0.0f64;
        for e in 0..mesh.n_edges {
            let dv = mesh.dv_edge[e];
            let dc = mesh.dc_edge[e];
            min_dv = min_dv.min(dv);
            min_dc = min_dc.min(dc);
            max_dc = max_dc.max(dc);
            if dc > 0.0 {
                min_ratio = min_ratio.min(dv / dc);
            }
        }
        println!("min_dv_edge_m\t{:.4}", min_dv * EARTH_RADIUS_M);
        println!("min_dv_over_dc\t{min_ratio:.6}");
        println!("min_dc_edge_km\t{:.5}", min_dc * EARTH_RADIUS_M / 1000.0);
        println!("max_dc_edge_km\t{:.5}", max_dc * EARTH_RADIUS_M / 1000.0);

        // ---- delivered adjacent-cell gradient, the thing the gate models ---
        let mut spacing = vec![0.0f64; n_cells];
        for i in 0..n_cells {
            let n = mesh.n_edges_on_cell[i] as usize;
            let mut s = 0.0;
            for k in 0..n {
                s += mesh.dc_edge[mesh.edges_on_cell[i * mesh.max_edges + k] as usize];
            }
            spacing[i] = s / n as f64;
        }
        let mut max_adj = 0.0f64;
        for e in 0..mesh.n_edges {
            let a = spacing[mesh.cells_on_edge[e * 2] as usize];
            let b = spacing[mesh.cells_on_edge[e * 2 + 1] as usize];
            let r = (a / b).max(b / a);
            if r > max_adj {
                max_adj = r;
            }
        }
        println!("delivered_max_adjacent_spacing_ratio\t{max_adj:.5}");
        println!(
            "delivered_max_adjacent_gradient_pct_per_cell\t{:.3}",
            (max_adj - 1.0) * 100.0
        );
        let centre = from_lat_lon(centre_deg.0.to_radians(), centre_deg.1.to_radians());
        let mut finest = f64::INFINITY;
        let mut finest_at = 0.0;
        for i in 0..n_cells {
            if spacing[i] < finest {
                finest = spacing[i];
                finest_at = arc(centre, mesh.cell_xyz[i]) * EARTH_RADIUS_M / 1000.0;
            }
        }
        println!(
            "delivered_finest_cell_km\t{:.5}\tat_radius_km\t{finest_at:.3}",
            finest * EARTH_RADIUS_M / 1000.0
        );
        for rk in [10.0f64, 50.0, 200.0, 800.0, 1400.0] {
            let mut n = 0usize;
            let mut hmax = 0.0f64;
            for i in 0..n_cells {
                if arc(centre, mesh.cell_xyz[i]) * EARTH_RADIUS_M / 1000.0 <= rk {
                    n += 1;
                    hmax = hmax.max(spacing[i] * EARTH_RADIUS_M / 1000.0);
                }
            }
            println!("cells_within_km\t{rk:.0}\t{n}\tcoarsest_km_there\t{hmax:.4}");
        }
    }
    Ok(())
}
