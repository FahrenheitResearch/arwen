//! `fp32_floor_probe` -- what coordinate quantisation costs the TRiSK weights,
//! as a function of dual-edge length.
//!
//! WHY IT EXISTS. `mesh::validate`'s `Limits::min_dv_edge_m` is a RULING with a
//! plausible justification, not a measurement: it says that below a couple
//! hundred metres a stored dual length is "materially quantisation noise". The
//! quantity that sentence rests on -- how wrong the TRiSK weights actually get
//! when the mesh's coordinates are stored at a finite quantum -- had never been
//! measured. This probe measures it.
//!
//! METHOD. The reference is the real `MpasMesh::derive` run on the float64 cell
//! centres of a published grid file: exact topology, exact geometry, the
//! weights the crate ships. Then the SAME centres are round-tripped through a
//! coordinate quantum `q` metres at Earth radius -- the storage the static and
//! init files actually use is binary32 at `sphere_radius = 6371229`, whose
//! coordinate spacing is exactly 0.5 m -- renormalised onto the unit sphere the
//! way every MPAS reader does, and the mesh is re-derived on the UNCHANGED
//! topology, because connectivity is stored as integers and cannot round. Every
//! difference is then attributable to the quantum alone.
//!
//! WHAT IT REPORTS, binned by the REFERENCE dvEdge in metres so the answer is a
//! curve and not one aggregate:
//!   * `dvEdge` absolute and relative error,
//!   * the dimensionless TRiSK weight `R[e,e'] = w * dcEdge[e] / dvEdge[e']`,
//!     which is bounded by 1/2 -- this is the number a tolerance should be
//!     stated on, because it is what the tangential reconstruction multiplies
//!     the neighbouring normal wind by,
//!   * the stored-weight relative error,
//!   * the SCVT orthogonality defect `|cos(primal, dual)|`,
//!   * the Thuburn antisymmetry residual of the quantised mesh.
//!
//! It writes no grid file and is not part of any shipped door.
//!
//! Usage:
//!   fp32_floor_probe --grid X.grid.nc --out-csv BINS.csv --out-json SUM.json
//!                    [--quanta 0.5,1,2] [--radius 6371229]

use std::collections::BTreeMap;
use std::process::ExitCode;

use rw_mpas::mesh::density::MeshSpec;
use rw_mpas::mesh::derive::{MpasMesh, Rings, edge_nonorthogonality};
use rw_mpas::mesh::geom::{EARTH_RADIUS_M, V3, from_lat_lon, unit};
use rw_mpas::mesh::gridread::read_grid_generators;
use rw_mpas::mesh::hierarchy::{DEFAULT_BETA, generate_graded};
use rw_mpas::mesh::hull::delaunay_rings;
use rw_mpas::mesh::lloyd::LloydOptions;
use rw_mpas::mesh::surgery::SurgeryOptions;

/// Round one Earth-centred coordinate to a multiple of `q` metres, then hand
/// back the unit-sphere component. `q = 0.5` is binary32 at 6371229 m exactly.
fn quantise(p: V3, q: f64, radius: f64) -> V3 {
    // A uniform grid of spacing `q`, offset by HALF A CELL so no coordinate can
    // land on exactly zero. Without the offset, any generator within q/2 of a
    // coordinate axis snaps onto it and `derive` correctly refuses an edge
    // sitting on the pole -- which killed every quantum on a graded mesh whose
    // icosahedral seed carries a polar generator. The offset is the
    // instrument's fix, not the mesh's: the rounding error stays uniform on
    // [-q/2, q/2]. Scaling the radius by a power of two is NOT an alternative:
    // binary32 rounding is exactly invariant under it, so every quantum would
    // read the same number (measured, 2026-08-29).
    let mut out = [0.0f64; 3];
    for k in 0..3 {
        let m = p[k] * radius;
        out[k] = ((m / q).floor() + 0.5) * q / radius;
    }
    unit(out).expect("a quantised generator still has a direction")
}

/// binary32 storage at `radius`: the coordinate spacing is `ulp(radius)`.
fn f32_spacing(radius: f64) -> f64 {
    let r = radius as f32;
    let next = f32::from_bits(r.to_bits() + 1);
    (next - r) as f64
}

/// A LOCAL frame: coordinates are stored as the offset from `origin`, in
/// metres, at binary32. The quantum a point then carries is `ulp(|offset|)`,
/// which for a 100 km domain is 2^-7 m instead of the 0.5 m that binary32 at
/// Earth radius forces. Far from the origin the offset is larger than the
/// Earth radius and the frame is WORSE, which is why it is only ever proposed
/// for a mesh that has a local origin at all.
fn quantise_local(p: V3, origin_m: V3, radius: f64) -> V3 {
    let mut out = [0.0f64; 3];
    for k in 0..3 {
        let d = p[k] * radius - origin_m[k];
        out[k] = ((d as f32) as f64 + origin_m[k]) / radius;
    }
    unit(out).expect("a quantised generator still has a direction")
}

struct EdgeStat {
    dv_ref: f64,
    dc_ref: f64,
    h_ref: f64,
    origin_km: f64,
    dv_abs: f64,
    r_abs_max: f64,
    w_rel_max: f64,
    nonorth: f64,
    anti_max: f64,
}

fn r_weight(m: &MpasMesh, e: usize, k: usize) -> (usize, f64) {
    let ep = m.edges_on_edge[e * m.max_edges2 + k] as usize;
    (ep, m.weights_on_edge[e * m.max_edges2 + k] * m.dc_edge[e] / m.dv_edge[ep])
}

fn compare(
    reference: &MpasMesh,
    test: &MpasMesh,
    radius: f64,
    spacing: &[f64],
    origin: Option<V3>,
) -> Vec<EdgeStat> {
    let mut out = Vec::with_capacity(reference.n_edges);
    for e in 0..reference.n_edges {
        let n = reference.n_edges_on_edge[e] as usize;
        let mut r_abs_max = 0.0f64;
        let mut w_rel_max = 0.0f64;
        let mut anti_max = 0.0f64;
        for k in 0..n {
            let (ep_r, r_ref) = r_weight(reference, e, k);
            let (ep_t, r_test) = r_weight(test, e, k);
            assert_eq!(ep_r, ep_t, "topology moved under quantisation");
            r_abs_max = r_abs_max.max((r_test - r_ref).abs());
            let wr = reference.weights_on_edge[e * reference.max_edges2 + k];
            let wt = test.weights_on_edge[e * test.max_edges2 + k];
            if wr != 0.0 {
                w_rel_max = w_rel_max.max(((wt - wr) / wr).abs());
            }
            // Antisymmetry of the TEST mesh: find e back inside e''s stencil.
            let np = test.n_edges_on_edge[ep_t] as usize;
            let back = (0..np).find(|&t| test.edges_on_edge[ep_t * test.max_edges2 + t] == e as i32);
            if let Some(t) = back {
                let (_, r_back) = r_weight(test, ep_t, t);
                anti_max = anti_max.max((r_test + r_back).abs());
            }
        }
        let c0 = reference.cells_on_edge[e * 2] as usize;
        let c1 = reference.cells_on_edge[e * 2 + 1] as usize;
        let h = 0.5 * (spacing[c0] + spacing[c1]);
        out.push(EdgeStat {
            dv_ref: reference.dv_edge[e] * radius,
            dc_ref: reference.dc_edge[e] * radius,
            h_ref: h,
            origin_km: origin
                .map(|o| {
                    let d = reference.edge_xyz[e];
                    let dot = (d[0] * o[0] + d[1] * o[1] + d[2] * o[2]).clamp(-1.0, 1.0);
                    dot.acos() * radius / 1000.0
                })
                .unwrap_or(f64::NAN),
            dv_abs: (test.dv_edge[e] - reference.dv_edge[e]).abs() * radius,
            r_abs_max,
            w_rel_max,
            nonorth: edge_nonorthogonality(test, e).abs(),
            anti_max,
        });
    }
    out
}

fn pct(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let i = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[i]
}

fn sorted_by(rows: &[&EdgeStat], sel: impl Fn(&EdgeStat) -> f64) -> Vec<f64> {
    let mut v: Vec<f64> = rows.iter().map(|r| sel(r)).collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let mut map: BTreeMap<String, String> = BTreeMap::new();
    let mut i = 1;
    while i + 1 < args.len() {
        map.insert(args[i].trim_start_matches("--").to_string(), args[i + 1].clone());
        i += 2;
    }
    if map.get("grid").is_none() && map.get("spec").is_none() {
        eprintln!(
            "usage: fp32_floor_probe (--grid X.grid.nc | --spec SPEC.json) --out-csv B.csv --out-json S.json [--quanta 0.5,1,2] [--radius 6371229] [--out-edges E.csv --edges-quantum 0.5] [--local-origin LAT,LON]"
        );
        return ExitCode::FAILURE;
    }
    let grid = map
        .get("grid")
        .cloned()
        .unwrap_or_else(|| map.get("spec").cloned().unwrap());
    let radius: f64 = map
        .get("radius")
        .and_then(|s| s.parse().ok())
        .unwrap_or(EARTH_RADIUS_M);
    let quanta: Vec<f64> = match map.get("quanta") {
        Some(s) => s.split(',').filter_map(|t| t.trim().parse().ok()).collect(),
        None => {
            let base = f32_spacing(radius);
            vec![
                base / 64.0,
                base / 8.0,
                base,
                base * 4.0,
                base * 16.0,
                base * 64.0,
                base * 256.0,
                base * 1024.0,
            ]
        }
    };

    let t0 = std::time::Instant::now();
    let (points, density, nominal, rings): (Vec<V3>, Vec<f64>, f64, Rings) =
        if let Some(spec_path) = map.get("spec") {
            // The GENERATOR arm: a real graded mesh, so the fine end of the
            // curve is measured on cells the shipped generator actually made
            // rather than on a synthetic point set that can be degenerate in
            // ways a real mesh is not.
            let text = std::fs::read_to_string(spec_path).expect("spec file");
            let spec = match MeshSpec::from_json(&text) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("REFUSED: {e}");
                    return ExitCode::FAILURE;
                }
            };
            let (pts, rings, _outcome, choice, reports) = match generate_graded(
                &spec,
                50_000,
                &LloydOptions::default(),
                &SurgeryOptions::default(),
                DEFAULT_BETA,
                |s| eprintln!("  {s}"),
            ) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("GENERATION REFUSED: {e}");
                    return ExitCode::FAILURE;
                }
            };
            eprintln!(
                "generated {} cells, GP({},{}), {} levels, {:.1}s",
                pts.len(),
                choice.m,
                choice.n,
                reports.len(),
                t0.elapsed().as_secs_f64()
            );
            let n = pts.len();
            (pts, vec![1.0; n], 0.0, rings)
        } else {
            let generators = match read_grid_generators(std::path::Path::new(&grid)) {
                Ok(g) => g,
                Err(e) => {
                    eprintln!("REFUSED: {e}");
                    return ExitCode::FAILURE;
                }
            };
            eprintln!(
                "read {} centres, sphere_radius {}, max radius departure {:.3e}, {:.1}s",
                generators.points.len(),
                generators.sphere_radius,
                generators.max_radius_departure,
                t0.elapsed().as_secs_f64()
            );
            let rings: Rings = match delaunay_rings(&generators.points) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("REFUSED: {e}");
                    return ExitCode::FAILURE;
                }
            };
            eprintln!("rings built, {:.1}s", t0.elapsed().as_secs_f64());
            (
                generators.points,
                generators.mesh_density,
                generators.nominal_min_dc,
                rings,
            )
        };
    // A fixed irrational rotation, applied to the generators BEFORE anything
    // else. A graded mesh's icosahedral seed carries a generator exactly on the
    // polar axis; quantisation can then put an EDGE MIDPOINT exactly on the
    // pole, where local east and north are undefined and `derive` correctly
    // refuses. That is the seed's alignment to the coordinate frame, not a
    // property of the mesh: a rigid rotation leaves every arc, area, kite and
    // weight unchanged and removes the coincidence. Reference and test are
    // rotated identically, so no error is created or hidden.
    let points: Vec<V3> = if map.contains_key("rotate") {
        let (ca, sa) = (0.317_845_1f64.cos(), 0.317_845_1f64.sin());
        let (cb, sb) = (0.618_033_9f64.cos(), 0.618_033_9f64.sin());
        points
            .into_iter()
            .map(|p| {
                let y = p[1] * ca - p[2] * sa;
                let z = p[1] * sa + p[2] * ca;
                [p[0] * cb - y * sb, p[0] * sb + y * cb, z]
            })
            .collect()
    } else {
        points
    };
    let origin: Option<V3> = map.get("local-origin").map(|s| {
        let parts: Vec<f64> = s
            .split(',')
            .map(|t| t.trim().parse().expect("lat,lon"))
            .collect();
        from_lat_lon(parts[0].to_radians(), parts[1].to_radians())
    });
    let reference = match MpasMesh::derive(points.clone(), density.clone(), &rings, nominal) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("REFUSED: {e}");
            return ExitCode::FAILURE;
        }
    };
    eprintln!("reference derived, {:.1}s", t0.elapsed().as_secs_f64());

    // The instrument's own zero, stated rather than assumed.
    let mut ref_nonorth = 0.0f64;
    for e in 0..reference.n_edges {
        ref_nonorth = ref_nonorth.max(edge_nonorthogonality(&reference, e).abs());
    }
    let spacing = reference.spacing_m();
    let self_stats = compare(&reference, &reference, radius, &spacing, origin);
    let self_refs: Vec<&EdgeStat> = self_stats.iter().collect();
    let ref_anti = sorted_by(&self_refs, |s| s.anti_max);
    let self_r = sorted_by(&self_refs, |s| s.r_abs_max);

    let bin_of = |dv: f64| -> i32 { dv.log2().floor() as i32 };

    let mut csv = String::from(
        "scheme,quantum_m,dv_bin_lo_m,dv_bin_hi_m,n_edges,dv_med_m,dv_abs_med_m,dv_abs_p99_m,dv_rel_med,dv_rel_p99,R_abs_med,R_abs_p99,R_abs_max,w_rel_med,w_rel_p99,nonorth_med,nonorth_p99,nonorth_max,anti_med,anti_max\n",
    );
    let mut summary = String::from("{\n");
    summary.push_str(&format!("  \"grid\": {:?},\n", grid));
    summary.push_str(&format!("  \"radius_m\": {},\n", radius));
    summary.push_str(&format!(
        "  \"f32_coordinate_spacing_m\": {},\n",
        f32_spacing(radius)
    ));
    summary.push_str(&format!("  \"n_cells\": {},\n", reference.n_cells));
    summary.push_str(&format!("  \"n_edges\": {},\n", reference.n_edges));
    summary.push_str(&format!(
        "  \"reference_max_nonorthogonality\": {:.6e},\n",
        ref_nonorth
    ));
    summary.push_str(&format!(
        "  \"reference_antisymmetry_med\": {:.6e},\n  \"reference_antisymmetry_max\": {:.6e},\n  \"reference_self_R_max\": {:.6e},\n",
        pct(&ref_anti, 0.5),
        ref_anti.last().copied().unwrap_or(f64::NAN),
        self_r.last().copied().unwrap_or(f64::NAN)
    ));
    let mut dv_sorted: Vec<f64> = reference.dv_edge.iter().map(|d| d * radius).collect();
    dv_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    summary.push_str(&format!(
        "  \"dv_min_m\": {:.4}, \"dv_med_m\": {:.4}, \"dv_max_m\": {:.4},\n",
        dv_sorted[0],
        pct(&dv_sorted, 0.5),
        dv_sorted[dv_sorted.len() - 1]
    ));
    summary.push_str("  \"schemes\": [\n");

    for (qi, &q) in quanta.iter().enumerate() {
        let pq: Vec<V3> = points.iter().map(|&p| quantise(p, q, radius)).collect();
        let test = match MpasMesh::derive(pq, density.clone(), &rings, nominal) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("q={q}: derive REFUSED: {e}");
                continue;
            }
        };
        let stats = compare(&reference, &test, radius, &spacing, origin);
        if map.get("edges-quantum").and_then(|s| s.parse::<f64>().ok()) == Some(q) {
            if let Some(path) = map.get("out-edges") {
                let mut e_csv = String::from("edge,dv_m,dc_m,h_m,origin_km,dv_abs_m,R_abs,w_rel,nonorth,anti\n");
                for (e, s) in stats.iter().enumerate() {
                    e_csv.push_str(&format!(
                        "{e},{:.4},{:.4},{:.4},{:.4},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e}\n",
                        s.dv_ref,
                        s.dc_ref,
                        s.h_ref,
                        s.origin_km,
                        s.dv_abs,
                        s.r_abs_max,
                        s.w_rel_max,
                        s.nonorth,
                        s.anti_max
                    ));
                }
                std::fs::write(path, e_csv).expect("write edges csv");
                eprintln!("per-edge rows for q={q} written to {path}");
            }
        }
        let mut by_bin: BTreeMap<i32, Vec<&EdgeStat>> = BTreeMap::new();
        for s in &stats {
            by_bin.entry(bin_of(s.dv_ref)).or_default().push(s);
        }
        let all: Vec<&EdgeStat> = stats.iter().collect();
        let all_r = sorted_by(&all, |s| s.r_abs_max);
        let all_no = sorted_by(&all, |s| s.nonorth);
        let all_anti = sorted_by(&all, |s| s.anti_max);
        summary.push_str(&format!(
            "    {{\"quantum_m\": {q}, \"R_abs_med\": {:.4e}, \"R_abs_p99\": {:.4e}, \"R_abs_max\": {:.4e}, \"nonorth_max\": {:.4e}, \"anti_max\": {:.4e}}}{}\n",
            pct(&all_r, 0.5),
            pct(&all_r, 0.99),
            all_r.last().unwrap(),
            all_no.last().unwrap(),
            all_anti.last().unwrap(),
            if qi + 1 == quanta.len() { "" } else { "," }
        ));
        for (b, rows) in &by_bin {
            let dv = sorted_by(rows, |s| s.dv_ref);
            let dva = sorted_by(rows, |s| s.dv_abs);
            let dvr = sorted_by(rows, |s| s.dv_abs / s.dv_ref);
            let ra = sorted_by(rows, |s| s.r_abs_max);
            let wr = sorted_by(rows, |s| s.w_rel_max);
            let no = sorted_by(rows, |s| s.nonorth);
            let an = sorted_by(rows, |s| s.anti_max);
            csv.push_str(&format!(
                "q{q},{q},{:.4},{:.4},{},{:.4},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e}\n",
                (*b as f64).exp2(),
                ((*b + 1) as f64).exp2(),
                rows.len(),
                pct(&dv, 0.5),
                pct(&dva, 0.5),
                pct(&dva, 0.99),
                pct(&dvr, 0.5),
                pct(&dvr, 0.99),
                pct(&ra, 0.5),
                pct(&ra, 0.99),
                ra.last().unwrap(),
                pct(&wr, 0.5),
                pct(&wr, 0.99),
                pct(&no, 0.5),
                pct(&no, 0.99),
                no.last().unwrap(),
                pct(&an, 0.5),
                an.last().unwrap()
            ));
        }
        eprintln!(
            "q={q:>12.6} m: R_abs med {:.3e} p99 {:.3e} max {:.3e}  nonorth max {:.3e}  anti max {:.3e}  ({:.1}s)",
            pct(&all_r, 0.5),
            pct(&all_r, 0.99),
            all_r.last().unwrap(),
            all_no.last().unwrap(),
            all_anti.last().unwrap(),
            t0.elapsed().as_secs_f64()
        );
    }
    summary.push_str("  ]");

    if let Some(o) = origin {
        // The LOCAL FRAME arm. The same binary32 storage, a different origin:
        // coordinates become offsets from a point inside the mesh, so their
        // magnitude is the domain size and not the Earth radius.
        let origin_m = [o[0] * radius, o[1] * radius, o[2] * radius];
        let pq: Vec<V3> = points
            .iter()
            .map(|&p| quantise_local(p, origin_m, radius))
            .collect();
        match MpasMesh::derive(pq, density.clone(), &rings, nominal) {
            Ok(test) => {
                let stats = compare(&reference, &test, radius, &spacing, origin);
                if let Some(path) = map.get("out-edges-local") {
                    let mut e_csv = String::from("edge,dv_m,dc_m,h_m,origin_km,dv_abs_m,R_abs,w_rel,nonorth,anti\n");
                    for (e, s) in stats.iter().enumerate() {
                        e_csv.push_str(&format!(
                            "{e},{:.4},{:.4},{:.4},{:.4},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e}\n",
                            s.dv_ref,
                            s.dc_ref,
                            s.h_ref,
                            s.origin_km,
                            s.dv_abs,
                            s.r_abs_max,
                            s.w_rel_max,
                            s.nonorth,
                            s.anti_max
                        ));
                    }
                    std::fs::write(path, e_csv).expect("write local edges csv");
                }
                summary.push_str(",\n  \"local_frame\": [\n");
                for &cap_km in &[25.0f64, 50.0, 100.0, 200.0, 400.0, 800.0, 1600.0] {
                    let rows: Vec<&EdgeStat> =
                        stats.iter().filter(|s| s.origin_km <= cap_km).collect();
                    if rows.is_empty() {
                        continue;
                    }
                    let ra = sorted_by(&rows, |s| s.r_abs_max);
                    let dva = sorted_by(&rows, |s| s.dv_abs);
                    let no = sorted_by(&rows, |s| s.nonorth);
                    summary.push_str(&format!(
                        "    {{\"cap_km\": {}, \"n_edges\": {}, \"dv_abs_med_m\": {:.4e}, \"dv_abs_max_m\": {:.4e}, \"R_abs_med\": {:.4e}, \"R_abs_max\": {:.4e}, \"nonorth_max\": {:.4e}}},\n",
                        cap_km,
                        rows.len(),
                        pct(&dva, 0.5),
                        dva.last().unwrap(),
                        pct(&ra, 0.5),
                        ra.last().unwrap(),
                        no.last().unwrap()
                    ));
                }
                summary.push_str("    null\n  ]");
            }
            Err(e) => eprintln!("local frame: derive REFUSED: {e}"),
        }
    }
    summary.push_str("\n}\n");

    if let Some(p) = map.get("out-csv") {
        std::fs::write(p, &csv).expect("write csv");
    }
    if let Some(p) = map.get("out-json") {
        std::fs::write(p, &summary).expect("write json");
    }
    print!("{summary}");
    ExitCode::SUCCESS
}
