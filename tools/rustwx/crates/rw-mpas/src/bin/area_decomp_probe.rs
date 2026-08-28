//! `area_decomp_probe` -- what the area-decomposition gate is actually reading.
//!
//! The gate in `mesh::validate` compares two triangulations of every Voronoi
//! cell: the spherical polygon fanned from the cell's vertex ring, against the
//! sum of the kites that tile the same cell. It refuses on the worst RELATIVE
//! disagreement over the mesh. That refusal fired on a five-level graded ladder
//! at 1.023e-9 against a 1.000e-9 limit while two-level ladders on MORE cells
//! read 5.7e-11, and no ladder anywhere is a correctness statement -- so the
//! question this probe exists to answer is: **is the disagreement rounding, or
//! is it the ring genuinely not being the cell's boundary?**
//!
//! It runs the real pipeline with the area-decomposition limit lifted to
//! infinity and NOTHING else changed, then writes:
//!
//! * a mesh-level summary (the readings the receipt carries, plus the
//!   quantities a tolerance could be derived from);
//! * one row per cell -- the two areas, their gap, the cell's own spacing,
//!   degree and dual-edge conditioning -- so the gap can be regressed against
//!   cell geometry instead of against a mesh label;
//! * the raw coordinates of the worst cells, to 17 significant digits, so the
//!   same two areas can be recomputed in arbitrary precision OUTSIDE this
//!   program. That last one is the decisive measurement: if the two areas agree
//!   to 30 digits when the arithmetic is exact, the f64 gap is the arithmetic's
//!   own noise and not a property of the mesh.
//!
//! This is an EVIDENCE instrument. It writes no grid file and it is not part of
//! any shipped door.

use std::path::PathBuf;
use std::process::ExitCode;

use rw_mpas::mesh::density::MeshSpec;
use rw_mpas::mesh::geom::{V3, arc};
use rw_mpas::mesh::{GenerateRequest, Limits, LloydOptions, generate};

fn usage() -> String {
    "usage: area_decomp_probe --out-json SUMMARY.json [--out-cells CELLS.csv] \
     [--out-worst WORST.json] [--spec SPEC.json | --background-km KM] \
     [--finest-km KM] [--cap-radius-km KM] [--cap-centre-deg LAT,LON] \
     [--transition-cells N] [--cells N] [--sweeps N] [--tolerance X] [--label TEXT]\n\n\
     Builds the mesh the request names with the area-decomposition limit lifted, \
     and measures what that gate would have read. Every other limit is left at \
     its shipped value.\n\n\
     --spec and the cap flags are alternatives: the cap flags synthesise a \
     one-region spec so a depth sweep is a command line, not eight spec files."
        .to_string()
}

struct Args {
    map: std::collections::BTreeMap<String, String>,
}

impl Args {
    fn parse(argv: Vec<String>) -> Result<Args, String> {
        let mut map = std::collections::BTreeMap::new();
        let mut it = argv.into_iter();
        while let Some(token) = it.next() {
            if !token.starts_with("--") {
                return Err(format!("unexpected argument \"{token}\"\n\n{}", usage()));
            }
            let key = token.trim_start_matches("--").to_string();
            let value = it
                .next()
                .ok_or_else(|| format!("--{key} needs a value\n\n{}", usage()))?;
            map.insert(key, value);
        }
        Ok(Args { map })
    }
    fn get(&self, key: &str) -> Option<&str> {
        self.map.get(key).map(String::as_str)
    }
    fn number<T: std::str::FromStr>(&self, key: &str) -> Result<Option<T>, String> {
        match self.get(key) {
            None => Ok(None),
            Some(v) => v
                .parse::<T>()
                .map(Some)
                .map_err(|_| format!("--{key} is not a number: \"{v}\"")),
        }
    }
}

fn run() -> Result<String, String> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    if argv.is_empty() || argv.iter().any(|a| a == "--help" || a == "-h") {
        return Err(usage());
    }
    let args = Args::parse(argv)?;

    // --- the request --------------------------------------------------------
    //
    // `--from-grid` takes the mesh as given, so it needs no resolution request
    // at all; a uniform placeholder stands in for the fields the generate
    // route would have filled and is never used to build anything.
    let spec: MeshSpec = match args.get("spec") {
        _ if args.get("from-grid").is_some() => MeshSpec::uniform(120.0),
        Some(path) => {
            let text = std::fs::read_to_string(path)
                .map_err(|e| format!("cannot read the resolution spec {path}: {e}"))?;
            MeshSpec::from_json(&text).map_err(|e| e.to_string())?
        }
        None => {
            let bg: f64 = args
                .number("background-km")?
                .ok_or("give --spec or --background-km")?;
            match args.number::<f64>("finest-km")? {
                None => MeshSpec::uniform(bg),
                Some(fine) => {
                    let radius: f64 = args.number("cap-radius-km")?.unwrap_or(600.0);
                    let cells: f64 = args.number("transition-cells")?.unwrap_or(81.0);
                    let centre = args.get("cap-centre-deg").unwrap_or("17.0,-57.0");
                    let (lat, lon) = centre
                        .split_once(',')
                        .ok_or("--cap-centre-deg is LAT,LON")?;
                    let lat: f64 = lat.trim().parse().map_err(|_| "bad --cap-centre-deg lat")?;
                    let lon: f64 = lon.trim().parse().map_err(|_| "bad --cap-centre-deg lon")?;
                    let json = serde_json::json!({
                        "background_km": bg,
                        "regions": [{
                            "shape": {"kind": "cap", "center_deg": [lat, lon], "radius_km": radius},
                            "spacing_km": fine,
                            "transition_cells": cells,
                        }],
                    });
                    MeshSpec::from_json(&json.to_string()).map_err(|e| e.to_string())?
                }
            }
        }
    };

    let mut lloyd = LloydOptions::default();
    if let Some(v) = args.number::<usize>("sweeps")? {
        lloyd.max_sweeps = v;
    }
    if let Some(v) = args.number::<f64>("tolerance")? {
        lloyd.tolerance = v;
    }

    // The ONE thing this probe changes. Every other gate keeps its shipped
    // value, so a mesh that would have been refused for any other reason is
    // still refused here and the probe reports the refusal rather than a
    // measurement taken on a mesh nobody would ship.
    let mut limits = Limits::default();
    limits.area_decomposition_ulps = f64::INFINITY;

    let request = GenerateRequest {
        spec: spec.clone(),
        target_cells: args.number::<usize>("cells")?,
        lloyd,
        limits,
        ..Default::default()
    };

    let t0 = std::time::Instant::now();
    // `--from-grid` takes a written mesh and rebuilds every derived field from
    // its cell centres, which is the same derivation the generator runs and
    // skips the relaxation. It exists so the breakage sweep below can be taken
    // on the ACTUAL grid a run produced rather than on a stand-in.
    let rebuilt;
    let generated;
    let (mesh, seeding_levels, receipt_mesh, spec_out, sweeps, mean_delta): (
        &rw_mpas::mesh::MpasMesh,
        usize,
        &rw_mpas::mesh::MeshReport,
        serde_json::Value,
        usize,
        f64,
    ) = match args.get("from-grid") {
        Some(path) => {
            let src = rw_mpas::mesh::gridread::read_grid_generators(std::path::Path::new(path))?;
            rebuilt = rw_mpas::mesh::rebuild(
                src.points,
                src.mesh_density,
                src.nominal_min_dc,
                limits,
                |line| eprintln!("{line}"),
            )
            .map_err(|e| e.to_string())?;
            (
                &rebuilt.mesh,
                0,
                &rebuilt.receipt.mesh,
                serde_json::json!({"route": "from-grid", "source": path}),
                0,
                f64::NAN,
            )
        }
        None => {
            generated =
                generate(&request, |line| eprintln!("{line}")).map_err(|e| e.to_string())?;
            let levels = match &generated.receipt.seeding {
                rw_mpas::mesh::Seeding::HierarchicalGoldberg { levels, .. } => *levels,
                _ => 0,
            };
            (
                &generated.mesh,
                levels,
                &generated.receipt.mesh,
                serde_json::to_value(&generated.spec).map_err(|e| e.to_string())?,
                generated.receipt.relaxation_sweeps,
                generated.receipt.relaxation_mean_delta_over_h,
            )
        }
    };
    let build_seconds = t0.elapsed().as_secs_f64();

    // --- the two decompositions, cell by cell -------------------------------
    let polygon = mesh.polygon_areas();
    let spacing = mesh.spacing_m();
    let r = rw_mpas::mesh::geom::EARTH_RADIUS_M;

    // Per-cell dual-edge conditioning: the smallest dvEdge/dcEdge among the
    // cell's own edges. This is the quantity the near-cocircular story blames
    // the gate reading on, so it is measured per cell rather than asserted.
    let mut min_ratio_of_cell = vec![f64::INFINITY; mesh.n_cells];
    let mut min_dv_of_cell = vec![f64::INFINITY; mesh.n_cells];
    for e in 0..mesh.n_edges {
        let ratio = mesh.dv_edge[e] / mesh.dc_edge[e];
        for s in 0..2 {
            let c = mesh.cells_on_edge[e * 2 + s] as usize;
            if ratio < min_ratio_of_cell[c] {
                min_ratio_of_cell[c] = ratio;
            }
            if mesh.dv_edge[e] < min_dv_of_cell[c] {
                min_dv_of_cell[c] = mesh.dv_edge[e];
            }
        }
    }

    #[derive(Clone, Copy)]
    struct Row {
        cell: usize,
        rel: f64,
        gap: f64,
        area_cell: f64,
        polygon: f64,
        spacing_m: f64,
        degree: i32,
        min_dv_over_dc: f64,
        min_dv_m: f64,
        fan_leg_max: f64,
    }

    let mut rows: Vec<Row> = Vec::with_capacity(mesh.n_cells);
    for c in 0..mesh.n_cells {
        let deg = mesh.n_edges_on_cell[c] as usize;
        let base = c * mesh.max_edges;
        let ring: Vec<V3> = (0..deg)
            .map(|j| mesh.vertex_xyz[mesh.vertices_on_cell[base + j] as usize])
            .collect();
        // The longest chord out of the fan apex: the scale the fan's rounding
        // error is proportional to, as against the cell's own spacing.
        let fan_leg_max = ring
            .iter()
            .skip(1)
            .map(|&p| arc(ring[0], p))
            .fold(0.0f64, f64::max);
        let gap = polygon[c] - mesh.area_cell[c];
        rows.push(Row {
            cell: c,
            rel: (gap / mesh.area_cell[c]).abs(),
            gap,
            area_cell: mesh.area_cell[c],
            polygon: polygon[c],
            spacing_m: spacing[c],
            degree: mesh.n_edges_on_cell[c],
            min_dv_over_dc: min_ratio_of_cell[c],
            min_dv_m: min_dv_of_cell[c] * r,
            fan_leg_max: fan_leg_max * r,
        });
    }

    if let Some(path) = args.get("out-cells") {
        let mut text = String::with_capacity(rows.len() * 96);
        text.push_str(
            "cell,rel,gap,area_cell,polygon,spacing_m,degree,min_dv_over_dc,min_dv_m,fan_leg_max_m\n",
        );
        for row in &rows {
            text.push_str(&format!(
                "{},{:.17e},{:.17e},{:.17e},{:.17e},{:.6},{},{:.9},{:.6},{:.6}\n",
                row.cell,
                row.rel,
                row.gap,
                row.area_cell,
                row.polygon,
                row.spacing_m,
                row.degree,
                row.min_dv_over_dc,
                row.min_dv_m,
                row.fan_leg_max
            ));
        }
        std::fs::write(path, text).map_err(|e| format!("cannot write {path}: {e}"))?;
    }

    // --- the worst cells, in full, to 17 digits -----------------------------
    let mut worst = rows.clone();
    worst.sort_by(|a, b| b.rel.partial_cmp(&a.rel).unwrap_or(std::cmp::Ordering::Equal));
    let n_worst: usize = args.number("worst")?.unwrap_or(24);
    if let Some(path) = args.get("out-worst") {
        // The worst cells are what the gate reads, but a dump of only the
        // worst cannot say whether the worst is a DIFFERENT animal from the
        // rest or just the tail of one distribution. Quantile picks ride
        // along so the arbitrary-precision recomputation covers both.
        let mut picks: Vec<Row> = worst.iter().take(n_worst).cloned().collect();
        let ascending = {
            let mut a = rows.clone();
            a.sort_by(|x, y| x.rel.partial_cmp(&y.rel).unwrap_or(std::cmp::Ordering::Equal));
            a
        };
        for f in [0.5f64, 0.9, 0.99, 0.999] {
            let idx = ((ascending.len() - 1) as f64 * f).round() as usize;
            picks.push(ascending[idx]);
        }
        let dump: Vec<serde_json::Value> = picks
            .iter()
            .map(|row| {
                let c = row.cell;
                let deg = mesh.n_edges_on_cell[c] as usize;
                let base = c * mesh.max_edges;
                serde_json::json!({
                    "cell": c,
                    "rel": row.rel,
                    "gap": row.gap,
                    "area_cell": row.area_cell,
                    "polygon_area": row.polygon,
                    "spacing_m": row.spacing_m,
                    "degree": deg,
                    "min_dv_over_dc": row.min_dv_over_dc,
                    "centre": mesh.cell_xyz[c],
                    "ring_vertices": (0..deg)
                        .map(|j| mesh.vertex_xyz[mesh.vertices_on_cell[base + j] as usize])
                        .collect::<Vec<_>>(),
                    "ring_edge_points": (0..deg)
                        .map(|j| mesh.edge_xyz[mesh.edges_on_cell[base + j] as usize])
                        .collect::<Vec<_>>(),
                    "kite_areas": (0..deg)
                        .map(|j| {
                            let v = mesh.vertices_on_cell[base + j] as usize;
                            let slot = (0..3)
                                .find(|&s| mesh.cells_on_vertex[v * 3 + s] == c as i32)
                                .unwrap_or(0);
                            mesh.kite_areas_on_vertex[v * 3 + slot]
                        })
                        .collect::<Vec<_>>(),
                    "dv_edge_m": (0..deg)
                        .map(|j| mesh.dv_edge[mesh.edges_on_cell[base + j] as usize] * r)
                        .collect::<Vec<_>>(),
                    "dc_edge_m": (0..deg)
                        .map(|j| mesh.dc_edge[mesh.edges_on_cell[base + j] as usize] * r)
                        .collect::<Vec<_>>(),
                    // The GENERATORS every ring point is a pure function of.
                    // With these an outside reader can rebuild the vertex ring
                    // and the edge points in arbitrary precision and ask
                    // whether the two decompositions still disagree -- which
                    // is the only way to tell the arithmetic's own noise from
                    // a ring that is genuinely not the cell's boundary.
                    "vertex_generators": (0..deg)
                        .map(|j| {
                            let v = mesh.vertices_on_cell[base + j] as usize;
                            (0..3)
                                .map(|s| mesh.cell_xyz[mesh.cells_on_vertex[v * 3 + s] as usize])
                                .collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>(),
                    "edge_generators": (0..deg)
                        .map(|j| {
                            let e = mesh.edges_on_cell[base + j] as usize;
                            (0..2)
                                .map(|s| mesh.cell_xyz[mesh.cells_on_edge[e * 2 + s] as usize])
                                .collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>(),
                })
            })
            .collect();
        let text = serde_json::to_string_pretty(&dump).map_err(|e| e.to_string())?;
        std::fs::write(path, text).map_err(|e| format!("cannot write {path}: {e}"))?;
    }

    // --- the mesh-level summary ---------------------------------------------
    let mut by_rel = rows.clone();
    by_rel.sort_by(|a, b| a.rel.partial_cmp(&b.rel).unwrap_or(std::cmp::Ordering::Equal));
    let q = |f: f64| by_rel[((by_rel.len() - 1) as f64 * f).round() as usize].rel;
    let levels = seeding_levels;
    let min_spacing = receipt_mesh.min_spacing_m;

    // --- the breakage the gate exists for, on THIS mesh ---------------------
    //
    // Every adjacent transposition of every cell's vertex ring: the defect
    // class is "the ring is not the boundary of the region the kites tile",
    // and a transposed pair is the mildest way to produce it. The MINIMUM over
    // the sweep is the number the floor has to sit under, and it is measured
    // here rather than quoted from another mesh.
    let breakage = if args.get("break-rings").map(|v| v != "no").unwrap_or(false) {
        let mut mildest = f64::INFINITY;
        let mut worst = 0.0f64;
        let mut ring_buf: Vec<V3> = Vec::with_capacity(mesh.max_edges);
        for c in 0..mesh.n_cells {
            let deg = mesh.n_edges_on_cell[c] as usize;
            let base = c * mesh.max_edges;
            for j in 0..deg {
                ring_buf.clear();
                for k in 0..deg {
                    ring_buf.push(mesh.vertex_xyz[mesh.vertices_on_cell[base + k] as usize]);
                }
                ring_buf.swap(j, (j + 1) % deg);
                let broken = rw_mpas::mesh::geom::polygon_area(&ring_buf);
                let fraction = ((broken - polygon[c]) / mesh.area_cell[c]).abs();
                mildest = mildest.min(fraction);
                worst = worst.max(fraction);
            }
        }
        let a_min = mesh.area_cell.iter().cloned().fold(f64::INFINITY, f64::min);
        serde_json::json!({
            "corruption": "one adjacent transposition of verticesOnCell, every cell, every slot",
            "mildest_fraction_of_cell_area": mildest,
            "worst_fraction_of_cell_area": worst,
            "smallest_cell_area_sr": a_min,
            "mildest_absolute_sr_on_the_smallest_cell": mildest * a_min,
            "mildest_in_machine_epsilons": mildest * a_min / f64::EPSILON,
        })
    } else {
        serde_json::Value::Null
    };
    let summary = serde_json::json!({
        "schema": "rw-mpas.area-decomposition-probe.v1",
        "label": args.get("label").unwrap_or("unlabelled"),
        "engine": concat!("rw-mpas ", env!("CARGO_PKG_VERSION"), " (rust)"),
        "build_seconds": build_seconds,
        "spec": spec_out,
        "ladder_levels": levels,
        "n_cells": mesh.n_cells,
        "background_km": receipt_mesh.max_spacing_m / 1000.0,
        "finest_requested_km": receipt_mesh.min_spacing_m / 1000.0,
        "min_spacing_m": min_spacing,
        "max_spacing_m": receipt_mesh.max_spacing_m,
        "spacing_ratio": receipt_mesh.spacing_ratio,
        "min_dv_over_dc": receipt_mesh.min_dv_over_dc,
        "min_dv_edge_m": receipt_mesh.min_dv_edge_m,
        "coordination_histogram": receipt_mesh.coordination_histogram.clone(),
        "max_area_decomposition_rel": receipt_mesh.max_area_decomposition_rel,
        "max_area_decomposition_abs": receipt_mesh.max_area_decomposition_abs,
        "max_area_decomposition_ulps": receipt_mesh.max_area_decomposition_ulps,
        "breakage_sweep": breakage,
        "worst_cell": worst.first().map(|r| r.cell),
        "worst_cell_spacing_m": worst.first().map(|r| r.spacing_m),
        "worst_cell_min_dv_over_dc": worst.first().map(|r| r.min_dv_over_dc),
        "rel_quantiles": {
            "p50": q(0.50), "p90": q(0.90), "p99": q(0.99),
            "p999": q(0.999), "p9999": q(0.9999), "max": by_rel.last().map(|r| r.rel),
        },
        // The two candidate laws, evaluated. `rel * h^2` is constant if the
        // ABSOLUTE area gap is size-independent; `rel * h` is constant if the
        // gap scales with the cell. Both are printed so neither is assumed.
        "max_rel_times_min_spacing_rad": worst.first().map(|r| r.rel * (r.spacing_m / r_earth())),
        "max_rel_times_min_spacing_rad_sq": worst.first().map(|r| {
            let h = r.spacing_m / r_earth();
            r.rel * h * h
        }),
        "relaxation_sweeps": sweeps,
        "relaxation_mean_delta_over_h": mean_delta,
    });
    let text = serde_json::to_string_pretty(&summary).map_err(|e| e.to_string())?;
    if let Some(path) = args.get("out-json") {
        std::fs::write(path, &text).map_err(|e| format!("cannot write {path}: {e}"))?;
    }
    Ok(text)
}

fn r_earth() -> f64 {
    rw_mpas::mesh::geom::EARTH_RADIUS_M
}

fn main() -> ExitCode {
    match run() {
        Ok(text) => {
            println!("{text}");
            ExitCode::SUCCESS
        }
        Err(message) => {
            eprintln!("area_decomp_probe: {message}");
            ExitCode::FAILURE
        }
    }
}

// Silence the unused-import lint when the PathBuf helper is not needed on a
// given build configuration.
#[allow(dead_code)]
fn _unused(_: PathBuf) {}
