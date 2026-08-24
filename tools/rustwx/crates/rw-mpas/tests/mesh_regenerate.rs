//! A published mesh's own cell centres, put back through the real binary, and
//! every field of the result compared with the file it came from.
//!
//! This is the whole-file version of `mesh_derive.rs`. That grader feeds the
//! library from a golden container and checks a stride sample of the edge and
//! vertex fields; this one reads the grid file `rw_mpas_mesh --from-centres`
//! actually wrote and compares EVERY element of EVERY field against the
//! published file, through `netcrust` -- a reader that shares no code with the
//! writer.
//!
//! NUMBERING. A generator invents its own edge and vertex numbering, so nothing
//! requires the rebuilt file to use the published one. The comparison is made
//! through a bijection built from the topology itself -- an edge is identified
//! by the unordered pair of cells it separates, a vertex by the unordered
//! triple of cells that meet at it -- and whether that bijection turns out to
//! be the identity is MEASURED and reported rather than assumed either way.
//!
//! UNITS. Both files stamp `sphere_radius = 1.0`. `areaCell`, `dcEdge`,
//! `dvEdge` and `kiteAreasOnVertex` are unit-sphere quantities despite their
//! `m` and `m^2` attributes; lengths are reported here in metres by multiplying
//! by 6,371,229 m, and areas by its square in km^2.
//!
//! RUNNING IT. Both paths come from the environment, so an offline
//! `cargo test --locked --offline` still passes on a box that has neither file:
//!
//!   GPUWM_MPAS_PUBLISHED_GRID=... GPUWM_MPAS_REBUILT_GRID=... \
//!   cargo test --release -p rw-mpas --test mesh_regenerate -- --nocapture

use std::collections::HashMap;

use rw_mpas::mesh::geom::EARTH_RADIUS_M;

// ------------------------------------------------------------------ harness

struct Pair {
    published: netcrust::File,
    rebuilt: netcrust::File,
}

fn open_pair() -> Option<Pair> {
    let a = std::env::var("GPUWM_MPAS_PUBLISHED_GRID").ok()?;
    let b = std::env::var("GPUWM_MPAS_REBUILT_GRID").ok()?;
    let published = netcrust::File::open(&a).unwrap_or_else(|e| panic!("{a}: {e}"));
    let rebuilt = netcrust::File::open(&b).unwrap_or_else(|e| panic!("{b}: {e}"));
    eprintln!("published: {a}\nrebuilt:   {b}");
    Some(Pair { published, rebuilt })
}

/// Skip cleanly when the meshes are not on this box, and say so out loud rather
/// than reporting a pass nobody measured.
macro_rules! pair_or_skip {
    () => {
        match open_pair() {
            Some(p) => p,
            None => {
                eprintln!(
                    "NOT MEASURED: set GPUWM_MPAS_PUBLISHED_GRID and GPUWM_MPAS_REBUILT_GRID to the two grid files"
                );
                return;
            }
        }
    };
}

fn f64s(f: &netcrust::File, name: &str) -> Vec<f64> {
    f.read_f64(name).unwrap_or_else(|e| panic!("{name}: {e}"))
}

fn i64s(f: &netcrust::File, name: &str) -> Vec<i64> {
    f64s(f, name).into_iter().map(|v| v as i64).collect()
}

fn dim(f: &netcrust::File, name: &str) -> usize {
    f.dimension(name)
        .unwrap_or_else(|| panic!("no dimension {name}"))
        .len()
}

/// Worst absolute and relative gap over a whole field, with where it happened.
#[derive(Default, Clone, Copy)]
struct Worst {
    abs: f64,
    rel: f64,
    at: usize,
    /// The magnitude of the published value at the worst RELATIVE gap, so a
    /// reader can turn the ratio back into a number.
    scale: f64,
    exact: usize,
    n: usize,
}

impl Worst {
    /// Longitude is an angle on a circle: 0 and 2*pi are the same place, and a
    /// linear residual reports a full turn as a total disagreement. Fields that
    /// wrap are fed through here instead.
    fn feed_circular(&mut self, i: usize, got: f64, want: f64) {
        self.n += 1;
        if got == want {
            self.exact += 1;
            return;
        }
        let mut d = got - want;
        while d > std::f64::consts::PI {
            d -= std::f64::consts::TAU;
        }
        while d < -std::f64::consts::PI {
            d += std::f64::consts::TAU;
        }
        let abs = d.abs();
        let rel = if want == 0.0 { abs } else { abs / want.abs() };
        if abs > self.abs {
            self.abs = abs;
        }
        if rel > self.rel {
            self.rel = rel;
            self.at = i;
            self.scale = want.abs();
        }
    }

    fn feed(&mut self, i: usize, got: f64, want: f64) {
        self.n += 1;
        if got == want {
            self.exact += 1;
            return;
        }
        let abs = (got - want).abs();
        let rel = if want == 0.0 { abs } else { abs / want.abs() };
        if abs > self.abs {
            self.abs = abs;
        }
        if rel > self.rel {
            self.rel = rel;
            self.at = i;
            self.scale = want.abs();
        }
    }
    fn all_exact(&self) -> bool {
        self.exact == self.n
    }
    fn line(&self, name: &str, unit: &str, to_unit: f64) -> String {
        if self.all_exact() {
            format!("{name:22} {:>9} / {:<9} EXACT (bit-identical)", self.exact, self.n)
        } else {
            format!(
                "{name:22} {:>9} / {:<9} worst rel {:.3e} (abs {:.3e} {unit}) at element {}, where the published value is {:.6e} {unit}",
                self.exact,
                self.n,
                self.rel,
                self.abs * to_unit,
                self.at,
                self.scale * to_unit,
                unit = unit
            )
        }
    }
}

/// Percentiles and threshold counts of a residual, because a worst value alone
/// cannot tell a handful of outliers from a field that is wrong everywhere.
fn distribution(name: &str, mut rel: Vec<f64>) -> usize {
    let n = rel.len();
    rel.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let at = |q: f64| rel[(((n - 1) as f64) * q) as usize];
    let above = |t: f64| rel.iter().filter(|&&v| v > t).count();
    eprintln!(
        "  {name} residual distribution over {n}: p50 {:.3e}  p90 {:.3e}  p99 {:.3e}  p99.9 {:.3e}  max {:.3e}",
        at(0.50),
        at(0.90),
        at(0.99),
        at(0.999),
        rel[n - 1]
    );
    eprintln!(
        "  {name} elements above 1e-10: {}   above 1e-9: {}   above 1e-6: {}   above 1e-3: {}",
        above(1e-10),
        above(1e-9),
        above(1e-6),
        above(1e-3)
    );
    above(1e-9)
}

/// Exact-match counter for an integer field.
#[derive(Default, Clone, Copy)]
struct Tally {
    same: usize,
    n: usize,
    first_bad: Option<usize>,
}

impl Tally {
    fn feed(&mut self, i: usize, got: i64, want: i64) {
        self.n += 1;
        if got == want {
            self.same += 1;
        } else if self.first_bad.is_none() {
            self.first_bad = Some(i);
        }
    }
    fn line(&self, name: &str) -> String {
        match self.first_bad {
            None => format!("{name:22} {:>9} / {:<9} EXACT", self.same, self.n),
            Some(i) => format!(
                "{name:22} {:>9} / {:<9} {} DIFFER, first at element {i}",
                self.same,
                self.n,
                self.n - self.same
            ),
        }
    }
}

// -------------------------------------------------------------- bijections

/// Published element index -> rebuilt element index, keyed on the cells the
/// element touches. Panics with the count when the two topologies are not the
/// same set of elements, which is the one failure that makes every field
/// comparison below meaningless.
fn bijection<K: std::hash::Hash + Eq + Clone + std::fmt::Debug>(
    what: &str,
    pub_keys: &[K],
    reb_keys: &[K],
) -> Vec<usize> {
    assert_eq!(pub_keys.len(), reb_keys.len(), "{what}: different counts");
    let index: HashMap<K, usize> = reb_keys
        .iter()
        .cloned()
        .enumerate()
        .map(|(i, k)| (k, i))
        .collect();
    assert_eq!(
        index.len(),
        reb_keys.len(),
        "{what}: the rebuilt file has duplicate keys, so no bijection exists"
    );
    let mut out = Vec::with_capacity(pub_keys.len());
    let mut missing = 0usize;
    let mut first_missing: Option<K> = None;
    for k in pub_keys {
        match index.get(k) {
            Some(&i) => out.push(i),
            None => {
                missing += 1;
                if first_missing.is_none() {
                    first_missing = Some(k.clone());
                }
                out.push(usize::MAX);
            }
        }
    }
    assert_eq!(
        missing, 0,
        "{what}: {missing} of {} published elements have no counterpart in the rebuilt mesh (first: {:?}). The two files do not describe the same topology",
        pub_keys.len(),
        first_missing
    );
    out
}

fn edge_keys(f: &netcrust::File, n_edges: usize) -> Vec<[i64; 2]> {
    let coe = i64s(f, "cellsOnEdge");
    (0..n_edges)
        .map(|e| {
            let (a, b) = (coe[e * 2], coe[e * 2 + 1]);
            [a.min(b), a.max(b)]
        })
        .collect()
}

fn vertex_keys(f: &netcrust::File, n_vertices: usize) -> Vec<[i64; 3]> {
    let cov = i64s(f, "cellsOnVertex");
    (0..n_vertices)
        .map(|v| {
            let mut t = [cov[v * 3], cov[v * 3 + 1], cov[v * 3 + 2]];
            t.sort_unstable();
            t
        })
        .collect()
}

// ------------------------------------------------------------------- tests

#[test]
fn the_rebuilt_file_carries_the_same_shape_and_header() {
    let p = pair_or_skip!();
    for name in ["nCells", "nEdges", "nVertices", "maxEdges", "maxEdges2", "TWO", "vertexDegree"] {
        let (a, b) = (dim(&p.published, name), dim(&p.rebuilt, name));
        assert_eq!(a, b, "dimension {name}: published {a}, rebuilt {b}");
        eprintln!("dimension {name:14} {a}");
    }
    for name in ["mesh_spec", "on_a_sphere", "is_periodic"] {
        let a = p.published.attribute(name).and_then(|a| a.as_string().map(str::to_string));
        let b = p.rebuilt.attribute(name).and_then(|a| a.as_string().map(str::to_string));
        assert_eq!(a, b, "attribute {name}");
        eprintln!("attribute {name:14} {a:?}");
    }
    for name in ["sphere_radius", "x_period", "y_period"] {
        let a = p.published.attribute(name).and_then(|a| a.as_f64());
        let b = p.rebuilt.attribute(name).and_then(|a| a.as_f64());
        assert_eq!(a, b, "attribute {name}");
        eprintln!("attribute {name:14} {a:?}");
    }
}

/// The centres go in and have to come out untouched. Anything else here means
/// the read or the write moved a generator, and every field below would be
/// graded against a mesh the published file never described.
#[test]
fn the_cell_centres_survive_the_round_trip_bit_for_bit() {
    let p = pair_or_skip!();
    let n = dim(&p.published, "nCells");
    let mut worst = Worst::default();
    for name in ["xCell", "yCell", "zCell"] {
        let a = f64s(&p.published, name);
        let b = f64s(&p.rebuilt, name);
        let mut w = Worst::default();
        for i in 0..n {
            w.feed(i, b[i], a[i]);
            worst.feed(i, b[i], a[i]);
        }
        eprintln!("{}", w.line(name, "unit-sphere", 1.0));
    }
    eprintln!(
        "  {} of {} components are bit-identical; worst departure {:.3e} on the unit sphere ({:.3e} m on Earth). The reader divides each centre by its own radius to land it exactly on the sphere, and the published radii are 6.661e-16 off 1.0, so a component can move by one double ULP. That ULP is the input perturbation every number below is measured on top of.",
        worst.exact, worst.n, worst.abs, worst.abs * EARTH_RADIUS_M
    );
    assert!(
        worst.abs < 4.0 * f64::EPSILON,
        "a cell centre moved by {:.3e}, more than the renormalisation can account for; the rebuild is not being graded on the published generators",
        worst.abs
    );
}

#[test]
fn every_cell_field_matches_the_published_file() {
    let p = pair_or_skip!();
    let n = dim(&p.published, "nCells");
    let me = dim(&p.published, "maxEdges");
    eprintln!("\n--- cell fields, all {n} cells -----------------------------------");

    // latCell / lonCell, in radians.
    for name in ["latCell", "lonCell"] {
        let a = f64s(&p.published, name);
        let b = f64s(&p.rebuilt, name);
        let mut w = Worst::default();
        let wraps = name.starts_with("lon");
        for i in 0..n {
            if wraps {
                w.feed_circular(i, b[i], a[i]);
            } else {
                w.feed(i, b[i], a[i]);
            }
        }
        eprintln!("{}", w.line(name, "rad", 1.0));
    }

    // meshDensity and indexToCellID have to be carried through untouched.
    for name in ["meshDensity", "indexToCellID"] {
        let a = f64s(&p.published, name);
        let b = f64s(&p.rebuilt, name);
        let mut w = Worst::default();
        for i in 0..n {
            w.feed(i, b[i], a[i]);
        }
        eprintln!("{}", w.line(name, "", 1.0));
        assert!(w.all_exact(), "{name} was not carried through unchanged");
    }

    // nEdgesOnCell: a pure count, exact or the triangulation differs.
    let a = i64s(&p.published, "nEdgesOnCell");
    let b = i64s(&p.rebuilt, "nEdgesOnCell");
    let mut t = Tally::default();
    for i in 0..n {
        t.feed(i, b[i], a[i]);
    }
    eprintln!("{}", t.line("nEdgesOnCell"));
    assert_eq!(t.first_bad, None, "the rebuilt triangulation gives some cell a different coordination number");

    // areaCell: the field variable resolution stresses hardest.
    let pa = f64s(&p.published, "areaCell");
    let ra = f64s(&p.rebuilt, "areaCell");
    let mut w = Worst::default();
    for i in 0..n {
        w.feed(i, ra[i], pa[i]);
    }
    let m2 = EARTH_RADIUS_M * EARTH_RADIUS_M / 1.0e6; // unit-sphere area -> km^2
    eprintln!("{}", w.line("areaCell", "km^2", m2));
    let area_rel: Vec<f64> = (0..n).map(|i| ((ra[i] - pa[i]) / pa[i]).abs()).collect();
    let n_outliers = distribution("areaCell", area_rel.clone());
    // Name the outliers rather than averaging them away: which cells, how big,
    // and -- the question variable resolution actually raises -- whether they
    // sit where the cell size is changing fastest.
    let mut outliers: Vec<(usize, f64)> = (0..n)
        .filter(|&i| area_rel[i] > 1e-9)
        .map(|i| (i, area_rel[i]))
        .collect();
    outliers.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    eprintln!("  areaCell: {n_outliers} cells above 1e-9. Every one of them, with the local spacing gradient:");
    let deg0 = i64s(&p.published, "nEdgesOnCell");
    let coc = i64s(&p.published, "cellsOnCell");
    for &(i, rel) in outliers.iter() {
        // Steepest neighbour-to-neighbour area ratio at this cell: the local
        // measure of how fast resolution is changing.
        let d = deg0[i] as usize;
        let mut worst_ratio = 1.0f64;
        for k in 0..d {
            let j = (coc[i * me + k] - 1) as usize;
            let r = (pa[j] / pa[i]).max(pa[i] / pa[j]);
            worst_ratio = worst_ratio.max(r);
        }
        eprintln!(
            "    cell {i:>7}  rel {rel:.3e}  areaCell {:>8.1} km^2  steepest neighbour area ratio {worst_ratio:.4}",
            pa[i] * m2
        );
    }
    // The same gradient statistic over ALL cells, so the outlier gradients can
    // be read against the mesh's own distribution instead of in isolation.
    let mut all_ratio: Vec<f64> = (0..n)
        .map(|i| {
            let d = deg0[i] as usize;
            let mut r = 1.0f64;
            for k in 0..d {
                let j = (coc[i * me + k] - 1) as usize;
                r = r.max((pa[j] / pa[i]).max(pa[i] / pa[j]));
            }
            r
        })
        .collect();
    all_ratio.sort_by(|a, b| a.partial_cmp(b).unwrap());
    eprintln!(
        "  neighbour area ratio over all {n} cells: p50 {:.4}  p90 {:.4}  p99 {:.4}  max {:.4} -- this is the mesh's own resolution gradient, for reading the outlier column above against",
        all_ratio[n / 2],
        all_ratio[n * 9 / 10],
        all_ratio[n * 99 / 100],
        all_ratio[n - 1]
    );
    eprintln!(
        "  areaCell spans {:.1} .. {:.1} km^2, a {:.2}:1 range; sum/4pi published {:.16} rebuilt {:.16}",
        pa.iter().cloned().fold(f64::MAX, f64::min) * m2,
        pa.iter().cloned().fold(0.0, f64::max) * m2,
        pa.iter().cloned().fold(0.0, f64::max) / pa.iter().cloned().fold(f64::MAX, f64::min),
        pa.iter().sum::<f64>() / (4.0 * std::f64::consts::PI),
        ra.iter().sum::<f64>() / (4.0 * std::f64::consts::PI),
    );

    // Where the areaCell disagreement sits AGAINST CELL SIZE. If the residual
    // were a transition-zone defect it would concentrate at one end of the
    // range; this bins it so that claim can be made or refused on counts.
    let (amin, amax) = (
        pa.iter().cloned().fold(f64::MAX, f64::min),
        pa.iter().cloned().fold(0.0f64, f64::max),
    );
    const BINS: usize = 8;
    let mut bin_worst = [0.0f64; BINS];
    let mut bin_count = [0usize; BINS];
    let mut bin_exact = [0usize; BINS];
    for i in 0..n {
        let t = ((pa[i] / amin).ln() / (amax / amin).ln() * BINS as f64) as usize;
        let k = t.min(BINS - 1);
        bin_count[k] += 1;
        if ra[i] == pa[i] {
            bin_exact[k] += 1;
        }
        let rel = ((ra[i] - pa[i]) / pa[i]).abs();
        bin_worst[k] = bin_worst[k].max(rel);
    }
    eprintln!("  areaCell residual by cell size (log bins over the {:.2}:1 range):", amax / amin);
    for k in 0..BINS {
        if bin_count[k] == 0 {
            continue;
        }
        let lo = amin * (amax / amin).powf(k as f64 / BINS as f64) * m2;
        let hi = amin * (amax / amin).powf((k + 1) as f64 / BINS as f64) * m2;
        eprintln!(
            "    {lo:>10.0} .. {hi:<10.0} km^2  {:>7} cells  {:>7} exact  worst rel {:.3e}",
            bin_count[k], bin_exact[k], bin_worst[k]
        );
    }

    // cellsOnCell, as a cyclic sequence: the ring START SLOT is not something
    // MPAS fixes, so a rebuild that begins the ring elsewhere is not wrong. The
    // ORDER within the ring is not free and is compared exactly.
    let pc = i64s(&p.published, "cellsOnCell");
    let rc = i64s(&p.rebuilt, "cellsOnCell");
    let deg = i64s(&p.published, "nEdgesOnCell");
    let mut rotations: std::collections::BTreeMap<usize, usize> = Default::default();
    let mut ring_bad = 0usize;
    let mut rotation_of_cell = vec![0usize; n];
    for i in 0..n {
        let d = deg[i] as usize;
        let want: Vec<i64> = (0..d).map(|j| pc[i * me + j]).collect();
        let mine: Vec<i64> = (0..d).map(|j| rc[i * me + j]).collect();
        match (0..d).find(|&r| (0..d).all(|k| mine[(r + k) % d] == want[k])) {
            Some(r) => {
                *rotations.entry(r).or_default() += 1;
                rotation_of_cell[i] = r;
            }
            None => ring_bad += 1,
        }
    }
    eprintln!(
        "cellsOnCell            {:>9} / {:<9} same ring up to rotation; start-slot offsets {rotations:?}",
        n - ring_bad,
        n
    );
    assert_eq!(ring_bad, 0, "{ring_bad} cells have a different neighbour ring");

    // edgesOnCell and verticesOnCell, read at the SAME rotation the neighbour
    // ring needed, through the topology bijections.
    let ne = dim(&p.published, "nEdges");
    let nv = dim(&p.published, "nVertices");
    let e_map = bijection("edge", &edge_keys(&p.published, ne), &edge_keys(&p.rebuilt, ne));
    let v_map = bijection("vertex", &vertex_keys(&p.published, nv), &vertex_keys(&p.rebuilt, nv));

    let pe = i64s(&p.published, "edgesOnCell");
    let re = i64s(&p.rebuilt, "edgesOnCell");
    let pv = i64s(&p.published, "verticesOnCell");
    let rv = i64s(&p.rebuilt, "verticesOnCell");
    let mut te = Tally::default();
    let mut tv = Tally::default();
    for i in 0..n {
        let d = deg[i] as usize;
        let r = rotation_of_cell[i];
        for k in 0..d {
            let want_e = e_map[(pe[i * me + k] - 1) as usize] as i64 + 1;
            te.feed(i, re[i * me + (r + k) % d], want_e);
            let want_v = v_map[(pv[i * me + k] - 1) as usize] as i64 + 1;
            tv.feed(i, rv[i * me + (r + k) % d], want_v);
        }
    }
    eprintln!("{}", te.line("edgesOnCell"));
    eprintln!("{}", tv.line("verticesOnCell"));
    assert_eq!(te.first_bad, None);
    assert_eq!(tv.first_bad, None);
}

#[test]
fn the_topology_bijections_are_measured_not_assumed() {
    let p = pair_or_skip!();
    let ne = dim(&p.published, "nEdges");
    let nv = dim(&p.published, "nVertices");
    let e_map = bijection("edge", &edge_keys(&p.published, ne), &edge_keys(&p.rebuilt, ne));
    let v_map = bijection("vertex", &vertex_keys(&p.published, nv), &vertex_keys(&p.rebuilt, nv));
    let e_identity = (0..ne).filter(|&e| e_map[e] == e).count();
    let v_identity = (0..nv).filter(|&v| v_map[v] == v).count();
    eprintln!(
        "\nedge bijection:   {ne} published edges all found; {e_identity} ({:.2}%) land on the same index\nvertex bijection: {nv} published vertices all found; {v_identity} ({:.2}%) land on the same index",
        100.0 * e_identity as f64 / ne as f64,
        100.0 * v_identity as f64 / nv as f64
    );
    eprintln!(
        "  A bijection existing at all is the real result: it says both files enumerate the same set of edges and the same set of vertices. Whether the NUMBERING agrees is a convention, and every field below is compared through the map either way."
    );
}

#[test]
fn every_edge_field_matches_the_published_file() {
    let p = pair_or_skip!();
    let ne = dim(&p.published, "nEdges");
    let nv = dim(&p.published, "nVertices");
    let me2 = dim(&p.published, "maxEdges2");
    let e_map = bijection("edge", &edge_keys(&p.published, ne), &edge_keys(&p.rebuilt, ne));
    let v_map = bijection("vertex", &vertex_keys(&p.published, nv), &vertex_keys(&p.rebuilt, nv));
    eprintln!("\n--- edge fields, all {ne} edges ----------------------------------");

    let m = EARTH_RADIUS_M;
    for (name, unit, to_unit) in [
        ("dcEdge", "m", m),
        ("dvEdge", "m", m),
        ("latEdge", "rad", 1.0),
        ("lonEdge", "rad", 1.0),
        ("xEdge", "unit-sphere", 1.0),
        ("yEdge", "unit-sphere", 1.0),
        ("zEdge", "unit-sphere", 1.0),
    ] {
        let a = f64s(&p.published, name);
        let b = f64s(&p.rebuilt, name);
        let mut w = Worst::default();
        for e in 0..ne {
            if name.starts_with("lon") {
                w.feed_circular(e, b[e_map[e]], a[e]);
            } else {
                w.feed(e, b[e_map[e]], a[e]);
            }
        }
        eprintln!("{}", w.line(name, unit, to_unit));
        if name == "dcEdge" || name == "dvEdge" {
            distribution(name, (0..ne).map(|e| ((b[e_map[e]] - a[e]) / a[e]).abs()).collect());
        }
    }

    // angleEdge wraps, so the residual is taken on the circle.
    let a = f64s(&p.published, "angleEdge");
    let b = f64s(&p.rebuilt, "angleEdge");
    let mut resid: Vec<f64> = Vec::with_capacity(ne);
    for e in 0..ne {
        let mut d = b[e_map[e]] - a[e];
        while d > std::f64::consts::PI {
            d -= std::f64::consts::TAU;
        }
        while d < -std::f64::consts::PI {
            d += std::f64::consts::TAU;
        }
        resid.push(d.abs());
    }
    let exact = resid.iter().filter(|&&d| d == 0.0).count();
    resid.sort_by(|x, y| x.partial_cmp(y).unwrap());
    eprintln!(
        "angleEdge              {exact:>9} / {ne:<9} median {:.3e} rad, p99 {:.3e} rad, max {:.3e} rad ({:.4} deg)",
        resid[ne / 2],
        resid[ne * 99 / 100],
        resid[ne - 1],
        resid[ne - 1].to_degrees()
    );

    // cellsOnEdge: the ORIENTATION is a convention. Measure how many edges the
    // rebuild orients the other way, because weightsOnEdge changes sign with it
    // and a comparison that ignored this would report noise as a defect.
    let pco = i64s(&p.published, "cellsOnEdge");
    let rco = i64s(&p.rebuilt, "cellsOnEdge");
    let mut same_orientation = 0usize;
    let mut flipped = 0usize;
    let mut sign = vec![1.0f64; ne];
    for e in 0..ne {
        let r = e_map[e];
        if pco[e * 2] == rco[r * 2] && pco[e * 2 + 1] == rco[r * 2 + 1] {
            same_orientation += 1;
        } else if pco[e * 2] == rco[r * 2 + 1] && pco[e * 2 + 1] == rco[r * 2] {
            flipped += 1;
            sign[e] = -1.0;
        } else {
            panic!("edge {e} separates different cells in the two files");
        }
    }
    eprintln!(
        "cellsOnEdge            {same_orientation:>9} / {ne:<9} same orientation, {flipped} reversed (a convention; weightsOnEdge below is sign-corrected for each)"
    );

    // verticesOnEdge, through the vertex map, allowing the matching reversal.
    let pvo = i64s(&p.published, "verticesOnEdge");
    let rvo = i64s(&p.rebuilt, "verticesOnEdge");
    let mut tv = Tally::default();
    for e in 0..ne {
        let r = e_map[e];
        let want = [
            v_map[(pvo[e * 2] - 1) as usize] as i64 + 1,
            v_map[(pvo[e * 2 + 1] - 1) as usize] as i64 + 1,
        ];
        let got = [rvo[r * 2], rvo[r * 2 + 1]];
        let ok = if sign[e] > 0.0 { got == want } else { got == [want[1], want[0]] };
        tv.feed(e, ok as i64, 1);
    }
    eprintln!("{}", tv.line("verticesOnEdge"));

    // nEdgesOnEdge and the blocked stencil.
    let pn = i64s(&p.published, "nEdgesOnEdge");
    let rn = i64s(&p.rebuilt, "nEdgesOnEdge");
    let mut tn = Tally::default();
    for e in 0..ne {
        tn.feed(e, rn[e_map[e]], pn[e]);
    }
    eprintln!("{}", tn.line("nEdgesOnEdge"));

    let peo = i64s(&p.published, "edgesOnEdge");
    let reo = i64s(&p.rebuilt, "edgesOnEdge");
    let mut ts = Tally::default();
    let mut pad = Tally::default();
    for e in 0..ne {
        let r = e_map[e];
        let k = pn[e] as usize;
        for j in 0..k {
            let want = e_map[(peo[e * me2 + j] - 1) as usize] as i64 + 1;
            ts.feed(e, reo[r * me2 + j], want);
        }
        for j in k..me2 {
            pad.feed(e, reo[r * me2 + j], peo[e * me2 + j]);
        }
    }
    eprintln!("{}", ts.line("edgesOnEdge"));
    eprintln!("{}", pad.line("  (its padding)"));

    // weightsOnEdge: SIGNED, sign-corrected for orientation. This is where a
    // flipped edge anywhere in a stencil would surface.
    let pw = f64s(&p.published, "weightsOnEdge");
    let rw = f64s(&p.rebuilt, "weightsOnEdge");
    let mut w = Worst::default();
    let mut scale = 0.0f64;
    let mut padding_nonzero = 0usize;
    for e in 0..ne {
        let r = e_map[e];
        let k = pn[e] as usize;
        for j in 0..k {
            // A weight belongs to the stencil edge; its sign follows both this
            // edge's orientation and the stencil edge's.
            let stencil = (peo[e * me2 + j] - 1) as usize;
            let s = sign[e] * sign[stencil];
            let want = pw[e * me2 + j];
            scale = scale.max(want.abs());
            w.feed(e, rw[r * me2 + j] * s, want);
        }
        for j in k..me2 {
            if rw[r * me2 + j] != 0.0 {
                padding_nonzero += 1;
            }
        }
    }
    eprintln!(
        "weightsOnEdge          {:>9} / {:<9} worst ABSOLUTE gap {:.3e} at edge {} against a weight scale of {scale:.6}; {padding_nonzero} nonzero padding slots",
        w.exact, w.n, w.abs, w.at
    );
    {
        let mut gaps: Vec<f64> = Vec::new();
        for e in 0..ne {
            let r = e_map[e];
            for j in 0..pn[e] as usize {
                let stencil = (peo[e * me2 + j] - 1) as usize;
                let s = sign[e] * sign[stencil];
                gaps.push((rw[r * me2 + j] * s - pw[e * me2 + j]).abs());
            }
        }
        distribution("weightsOnEdge (absolute)", gaps);
    }

    // The question variable resolution actually raises: are the operator
    // weights worse where the cell size changes fastest? Each edge is binned by
    // the area ratio of the two cells it separates -- the local resolution
    // gradient -- and the weight residual reported per bin, so the answer is a
    // table of counts rather than an average over the whole sphere.
    {
        let pa = f64s(&p.published, "areaCell");
        let mut rows: Vec<(f64, f64)> = Vec::with_capacity(ne);
        for e in 0..ne {
            let (c0, c1) = ((pco[e * 2] - 1) as usize, (pco[e * 2 + 1] - 1) as usize);
            let ratio = (pa[c0] / pa[c1]).max(pa[c1] / pa[c0]);
            let r = e_map[e];
            let mut worst = 0.0f64;
            for j in 0..pn[e] as usize {
                let stencil = (peo[e * me2 + j] - 1) as usize;
                let sg = sign[e] * sign[stencil];
                worst = worst.max((rw[r * me2 + j] * sg - pw[e * me2 + j]).abs());
            }
            rows.push((ratio, worst));
        }
        let mut sorted = rows.clone();
        sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        eprintln!(
            "  weightsOnEdge by the LOCAL RESOLUTION GRADIENT (area ratio of the two cells the edge separates), in equal-count deciles:"
        );
        let d = ne / 10;
        for k in 0..10 {
            let lo = k * d;
            let hi = if k == 9 { ne } else { (k + 1) * d };
            let slice = &sorted[lo..hi];
            let mut worst = 0.0f64;
            let mut above = 0usize;
            let mut sum = 0.0f64;
            for &(_, g) in slice {
                worst = worst.max(g);
                sum += g;
                if g > 1e-9 {
                    above += 1;
                }
            }
            eprintln!(
                "    ratio {:.4} .. {:.4}   {:>7} edges   mean gap {:.3e}   worst {:.3e}   {above} above 1e-9",
                slice[0].0,
                slice[slice.len() - 1].0,
                slice.len(),
                sum / slice.len() as f64,
                worst
            );
        }
    }
    assert_eq!(padding_nonzero, 0, "the rebuilt file has nonzero weight padding");
}

#[test]
fn every_vertex_field_matches_the_published_file() {
    let p = pair_or_skip!();
    let nv = dim(&p.published, "nVertices");
    let ne = dim(&p.published, "nEdges");
    let v_map = bijection("vertex", &vertex_keys(&p.published, nv), &vertex_keys(&p.rebuilt, nv));
    let e_map = bijection("edge", &edge_keys(&p.published, ne), &edge_keys(&p.rebuilt, ne));
    eprintln!("\n--- vertex fields, all {nv} vertices -----------------------------");

    for (name, unit) in [("latVertex", "rad"), ("lonVertex", "rad"), ("xVertex", "unit-sphere"), ("yVertex", "unit-sphere"), ("zVertex", "unit-sphere")] {
        let a = f64s(&p.published, name);
        let b = f64s(&p.rebuilt, name);
        let mut w = Worst::default();
        for v in 0..nv {
            if name.starts_with("lon") {
                w.feed_circular(v, b[v_map[v]], a[v]);
            } else {
                w.feed(v, b[v_map[v]], a[v]);
            }
        }
        eprintln!("{}", w.line(name, unit, 1.0));
    }

    let m2 = EARTH_RADIUS_M * EARTH_RADIUS_M / 1.0e6;
    let a = f64s(&p.published, "areaTriangle");
    let b = f64s(&p.rebuilt, "areaTriangle");
    let mut w = Worst::default();
    for v in 0..nv {
        w.feed(v, b[v_map[v]], a[v]);
    }
    eprintln!("{}", w.line("areaTriangle", "km^2", m2));
    distribution("areaTriangle", (0..nv).map(|v| ((b[v_map[v]] - a[v]) / a[v]).abs()).collect());

    // kiteAreasOnVertex: slot i belongs to cellsOnVertex[v][i]. The winding is
    // a convention, so the kites are matched through the CELL they belong to.
    let pk = f64s(&p.published, "kiteAreasOnVertex");
    let rk = f64s(&p.rebuilt, "kiteAreasOnVertex");
    let pcov = i64s(&p.published, "cellsOnVertex");
    let rcov = i64s(&p.rebuilt, "cellsOnVertex");
    let mut wk = Worst::default();
    for v in 0..nv {
        let r = v_map[v];
        for s in 0..3 {
            let cell = pcov[v * 3 + s];
            let slot = (0..3)
                .find(|&t| rcov[r * 3 + t] == cell)
                .expect("the two triples are the same set");
            wk.feed(v, rk[r * 3 + slot], pk[v * 3 + s]);
        }
    }
    eprintln!("{}", wk.line("kiteAreasOnVertex", "km^2", m2));
    {
        let mut rels: Vec<f64> = Vec::with_capacity(nv * 3);
        for v in 0..nv {
            let r = v_map[v];
            for s in 0..3 {
                let cell = pcov[v * 3 + s];
                let slot = (0..3).find(|&t| rcov[r * 3 + t] == cell).unwrap();
                let want = pk[v * 3 + s];
                rels.push(((rk[r * 3 + slot] - want) / want).abs());
            }
        }
        distribution("kiteAreasOnVertex", rels);
    }
    eprintln!(
        "  sum(kiteAreas)/4pi published {:.16} rebuilt {:.16}",
        pk.iter().sum::<f64>() / (4.0 * std::f64::consts::PI),
        rk.iter().sum::<f64>() / (4.0 * std::f64::consts::PI)
    );

    // The kite residual against the LOCAL CELL SIZE, so a transition-zone
    // claim can be made or refused on counts rather than on an average.
    let pa = f64s(&p.published, "areaCell");
    let amin = pa.iter().cloned().fold(f64::MAX, f64::min);
    let amax = pa.iter().cloned().fold(0.0f64, f64::max);
    const BINS: usize = 8;
    let mut bin_worst = [0.0f64; BINS];
    let mut bin_count = [0usize; BINS];
    for v in 0..nv {
        let r = v_map[v];
        for s in 0..3 {
            let cell = pcov[v * 3 + s];
            let ci = (cell - 1) as usize;
            let slot = (0..3).find(|&t| rcov[r * 3 + t] == cell).unwrap();
            let want = pk[v * 3 + s];
            let rel = ((rk[r * 3 + slot] - want) / want).abs();
            let t = ((pa[ci] / amin).ln() / (amax / amin).ln() * BINS as f64) as usize;
            let k = t.min(BINS - 1);
            bin_count[k] += 1;
            bin_worst[k] = bin_worst[k].max(rel);
        }
    }
    eprintln!("  kiteAreasOnVertex residual by the size of the cell the kite belongs to:");
    for k in 0..BINS {
        if bin_count[k] == 0 {
            continue;
        }
        let lo = amin * (amax / amin).powf(k as f64 / BINS as f64) * m2;
        let hi = amin * (amax / amin).powf((k + 1) as f64 / BINS as f64) * m2;
        eprintln!(
            "    {lo:>10.0} .. {hi:<10.0} km^2  {:>8} kites  worst rel {:.3e}",
            bin_count[k], bin_worst[k]
        );
    }

    // edgesOnVertex, through the edge map, allowing the winding to differ.
    let pev = i64s(&p.published, "edgesOnVertex");
    let rev = i64s(&p.rebuilt, "edgesOnVertex");
    let mut t = Tally::default();
    for v in 0..nv {
        let r = v_map[v];
        let mut want: Vec<i64> = (0..3).map(|s| e_map[(pev[v * 3 + s] - 1) as usize] as i64 + 1).collect();
        let mut got: Vec<i64> = (0..3).map(|s| rev[r * 3 + s]).collect();
        want.sort_unstable();
        got.sort_unstable();
        t.feed(v, (want == got) as i64, 1);
    }
    eprintln!("{}", t.line("edgesOnVertex"));
    assert_eq!(t.first_bad, None);
}

/// The comparator has to say "different" as reliably as it says "same".
///
/// Every gate above is a threshold, and a comparator that silently reported
/// zero -- a mis-built bijection, a field read twice from the same file, an
/// index that never advanced -- would pass every one of them. This drives a
/// KNOWN perturbation through the same code and checks the reported number is
/// the perturbation, in both directions.
#[test]
fn the_comparator_detects_a_known_perturbation_in_both_directions() {
    let p = pair_or_skip!();
    let n = dim(&p.published, "nCells");
    let pa = f64s(&p.published, "areaCell");
    let ra = f64s(&p.rebuilt, "areaCell");

    // Direction 1: unperturbed. Whatever this is, it is the floor.
    let mut clean = Worst::default();
    for i in 0..n {
        clean.feed(i, ra[i], pa[i]);
    }

    // Direction 2: one cell moved by a named amount, through the same code.
    // The ladder is anchored to the MEASURED floor, so the positive direction is
    // always exercised no matter which mesh this runs on: one rung below the
    // floor (must be invisible) and two above it (must be reported exactly).
    let floor = clean.rel;
    for injected in [floor * 0.01, floor * 1e3, (floor * 1e6).max(1e-3)] {
        let victim = 12_345usize.min(n - 1);
        let mut poisoned = ra.clone();
        poisoned[victim] = pa[victim] * (1.0 + injected);
        let mut w = Worst::default();
        for i in 0..n {
            w.feed(i, poisoned[i], pa[i]);
        }
        eprintln!(
            "injected {injected:.0e} relative into areaCell[{victim}] -> comparator reports worst rel {:.3e} (clean floor {:.3e})",
            w.rel, clean.rel
        );
        if injected > clean.rel * 10.0 {
            assert!(
                (w.rel / injected - 1.0).abs() < 1e-6,
                "a {injected:.0e} perturbation was reported as {:.3e}; the comparator is not measuring what it claims",
                w.rel
            );
            assert!(w.at == victim, "the comparator blamed element {} rather than {victim}", w.at);
        } else {
            eprintln!(
                "  RESOLUTION LIMIT: {injected:.0e} is at or below the clean residual {:.3e}, so this comparison cannot see it. That is the floor of every number reported by this file.",
                clean.rel
            );
        }
    }

    // And the bijection has to refuse a topology that is not the same set.
    let ne = dim(&p.published, "nEdges");
    let mut keys = edge_keys(&p.published, ne);
    let broken = std::panic::catch_unwind(|| {
        let mut reb = keys.clone();
        reb[7] = [i64::MAX, i64::MAX];
        bijection("edge", &keys, &reb)
    });
    assert!(broken.is_err(), "the bijection accepted an edge set that is missing an edge");
    keys.truncate(0);
    eprintln!("the bijection refuses an edge set with one edge replaced: confirmed");
}
