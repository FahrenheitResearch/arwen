//! Stage attribution for the mesh generator, off unless asked for.
//!
//! WHY THIS EXISTS. Two mesh speed hypotheses have died on a guess about
//! where the time went; the only thing that settles it is a reading. The
//! stage totals a caller can see today are the progress lines, and those
//! attribute a whole anneal to "RELAXED" -- which is the parallel centroid
//! sweep and the SERIAL Delaunay hull added together, and the two need
//! completely different remedies.
//!
//! It is off by default and costs one `Instant::now()` pair per instrumented
//! block. Nothing here reads a number back into the mesh, so a profiled run
//! and an unprofiled one produce the same bytes.
//!
//! Turn it on with `GPUWM_MESH_PROFILE=1`; the report goes to stderr at the
//! end of the run, never to stdout, which carries the receipt JSON.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// One instrumented block: nanoseconds, call count, and a work unit whose
/// meaning is the block's own (points triangulated, cells stepped, ...).
pub struct Counter {
    pub name: &'static str,
    nanos: AtomicU64,
    calls: AtomicU64,
    work: AtomicU64,
}

impl Counter {
    const fn new(name: &'static str) -> Counter {
        Counter {
            name,
            nanos: AtomicU64::new(0),
            calls: AtomicU64::new(0),
            work: AtomicU64::new(0),
        }
    }

    pub fn add(&self, nanos: u64, work: u64) {
        self.nanos.fetch_add(nanos, Ordering::Relaxed);
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.work.fetch_add(work, Ordering::Relaxed);
    }

    pub fn seconds(&self) -> f64 {
        self.nanos.load(Ordering::Relaxed) as f64 * 1e-9
    }

    pub fn calls(&self) -> u64 {
        self.calls.load(Ordering::Relaxed)
    }

    pub fn work(&self) -> u64 {
        self.work.load(Ordering::Relaxed)
    }
}

/// The spherical Delaunay hull. SERIAL; work is points triangulated.
pub static HULL: Counter = Counter::new("hull.delaunay_rings");
/// The Lloyd centroid sweep. rayon-parallel; work is cells stepped.
pub static LLOYD_STEP: Counter = Counter::new("lloyd.centroid_sweep");
/// Applying the step and the per-sweep dv/dc monitor.
pub static LLOYD_MONITOR: Counter = Counter::new("lloyd.dv_over_dc_monitor");
/// The seed lattice.
pub static SEED: Counter = Counter::new("seed");
/// Midpoint insertion between ladder rungs. Work is edges considered.
pub static INSERT: Counter = Counter::new("hierarchy.insert");
/// Count-changing surgery, whole rounds. Work is cells at entry.
pub static SURGERY: Counter = Counter::new("surgery.drain");
/// The local polish inside surgery, which re-triangulates.
pub static SURGERY_POLISH: Counter = Counter::new("surgery.local_polish");
/// The whole-point-set nearest-generator scan the polish runs per seed.
pub static NEAREST: Counter = Counter::new("surgery.nearest_generator");
/// Detecting flagged quads, once per surgery round over every edge.
pub static QUAD_SCAN: Counter = Counter::new("surgery.quad_scan");
/// Deriving the MPAS field set from the rings.
pub static DERIVE: Counter = Counter::new("derive");
/// The emit gate.
pub static VALIDATE: Counter = Counter::new("validate");
/// The density field, evaluated per point outside the relaxation.
pub static DENSITY: Counter = Counter::new("density.bulk_eval");
/// Writing the netCDF.
pub static EMIT: Counter = Counter::new("emit.write");

// --- inside the hull ----------------------------------------------------
//
// COUNTS, not times. The three sites below run 10^9 times between them, and
// an `Instant::now()` pair at that granularity would cost more than the work
// it measures. Each hull build accumulates plain locals -- which stay in
// registers -- and flushes them here once, so the counting is free.
/// `visible` calls made walking the visible region out from the seed facet.
pub static VIS_FLOOD: Counter = Counter::new("hull.visible.flood_fill");
/// `visible` calls made re-homing orphaned points onto the new facets.
pub static VIS_ORPHAN: Counter = Counter::new("hull.visible.orphan_rehome");
/// `visible` calls made assigning the initial conflict lists.
pub static VIS_SEED: Counter = Counter::new("hull.visible.initial_assign");
/// Facets created, live and retired. Work is the peak live count.
pub static FACES: Counter = Counter::new("hull.facets_created");
/// Orphaned points re-homed. Work is horizon edges walked.
pub static ORPHANS: Counter = Counter::new("hull.orphans_rehomed");
/// The duplicate-generator sort, once per hull build.
pub static HULL_DUP: Counter = Counter::new("hull.exact_duplicate");
/// Choosing the seed tetrahedron, once per hull build.
pub static HULL_TET: Counter = Counter::new("hull.initial_tetrahedron");
/// The incremental insertion loop itself.
pub static HULL_INSERT: Counter = Counter::new("hull.insertion_loop");
/// Turning the finished facets into neighbour rings.
pub static HULL_RINGS: Counter = Counter::new("hull.rings_from_faces");

/// HOW MUCH the triangulation actually moves between two Lloyd sweeps.
///
/// The relaxation rebuilds the whole spherical Delaunay from scratch once per
/// sweep. Whether that is redundant work or necessary work is decided by one
/// number: how many cells' neighbour rings differ from the previous sweep's.
/// `calls` counts sweep-to-sweep comparisons, `work` the cells whose ring
/// changed as a SET, and `nanos` -- misused here as a plain tally, this
/// counter records no time -- the cells whose ring changed only by rotation.
pub static RING_CHURN: Counter = Counter::new("lloyd.ring_churn");

/// WHAT AN INCREMENTAL UPDATE WOULD COST INSTEAD, measured on the real data.
///
/// One Lawson pass: every ring step of the PREVIOUS triangulation tested for
/// the local Delaunay property at the MOVED positions, with the same exact
/// predicate the hull uses. `work` counts the tests, `nanos` the time, and
/// `calls` the sweeps. The violations are reported separately.
pub static FLIP_PROBE: Counter = Counter::new("lloyd.lawson_pass_probe");
/// Ring steps the pass found non-Delaunay, i.e. flips one pass would do.
pub static FLIP_VIOLATIONS: Counter = Counter::new("lloyd.lawson_violations");

// --- the class-B incremental arm ----------------------------------------
/// Lawson repair of a moved triangulation. `work` counts edge tests.
pub static LAWSON: Counter = Counter::new("hull.lawson_repair");
/// Flips the repair performed. `work` is flips, `calls` is repairs.
pub static LAWSON_FLIPS: Counter = Counter::new("hull.lawson_flips");
/// Repairs that gave up and fell back to a full rebuild BY NAME. Every call
/// here is a hull build the incremental arm did not save, and a non-zero
/// count is a reportable event, not a tuning knob.
pub static LAWSON_FALLBACK: Counter = Counter::new("hull.lawson_fallback_to_rebuild");
/// Rings re-derived from a maintained facet list, rather than from a build.
pub static TRI_RINGS: Counter = Counter::new("hull.rings_from_maintained_facets");

static ALL: &[&Counter] = &[
    &SEED,
    &LLOYD_STEP,
    &HULL,
    &LLOYD_MONITOR,
    &INSERT,
    &SURGERY,
    &SURGERY_POLISH,
    &VIS_FLOOD,
    &VIS_ORPHAN,
    &VIS_SEED,
    &FACES,
    &ORPHANS,
    &HULL_DUP,
    &HULL_TET,
    &HULL_INSERT,
    &HULL_RINGS,
    &RING_CHURN,
    &FLIP_PROBE,
    &FLIP_VIOLATIONS,
    &LAWSON,
    &LAWSON_FLIPS,
    &LAWSON_FALLBACK,
    &TRI_RINGS,
    &NEAREST,
    &QUAD_SCAN,
    &DERIVE,
    &VALIDATE,
    &DENSITY,
    &EMIT,
];

static CHECKED: AtomicBool = AtomicBool::new(false);
static ON: AtomicBool = AtomicBool::new(false);

/// Is profiling asked for? Read once, from `GPUWM_MESH_PROFILE`.
#[inline]
pub fn on() -> bool {
    if !CHECKED.load(Ordering::Relaxed) {
        let want = std::env::var("GPUWM_MESH_PROFILE")
            .map(|v| v != "0" && !v.is_empty())
            .unwrap_or(false);
        ON.store(want, Ordering::Relaxed);
        CHECKED.store(true, Ordering::Relaxed);
    }
    ON.load(Ordering::Relaxed)
}

/// Time `body` into `counter` when profiling is on, and call it plainly when
/// it is not. The closure runs exactly once either way.
#[inline]
pub fn timed<T>(counter: &Counter, work: u64, body: impl FnOnce() -> T) -> T {
    if !on() {
        return body();
    }
    let t = std::time::Instant::now();
    let out = body();
    counter.add(t.elapsed().as_nanos() as u64, work);
    out
}

/// A scope timer, for a block whose body has early returns (`?`) and so
/// cannot be a closure without changing what it returns.
pub struct Span {
    counter: &'static Counter,
    work: u64,
    start: Option<std::time::Instant>,
}

impl Span {
    pub fn new(counter: &'static Counter, work: u64) -> Span {
        Span {
            counter,
            work,
            start: if on() {
                Some(std::time::Instant::now())
            } else {
                None
            },
        }
    }
}

impl Drop for Span {
    fn drop(&mut self) {
        if let Some(t) = self.start {
            self.counter.add(t.elapsed().as_nanos() as u64, self.work);
        }
    }
}

/// The report, tab separated, one row per instrumented block.
pub fn report(total_wall: f64) -> String {
    let mut lines = vec![format!(
        "PROFILE\tstage\tseconds\tpercent_of_wall\tcalls\twork_units"
    )];
    let mut accounted = 0.0;
    for c in ALL {
        if c.calls() == 0 {
            continue;
        }
        // The hull's time is inside the relaxation's wall time on the graded
        // arm, so these rows OVERLAP by construction; the accounted sum below
        // adds only the leaves.
        lines.push(format!(
            "PROFILE\t{}\t{:.3}\t{:.1}\t{}\t{}",
            c.name,
            c.seconds(),
            100.0 * c.seconds() / total_wall,
            c.calls(),
            c.work()
        ));
    }
    for c in [
        &SEED,
        &LLOYD_STEP,
        &HULL,
        &LLOYD_MONITOR,
        &INSERT,
        &DERIVE,
        &VALIDATE,
        &DENSITY,
        &EMIT,
    ] {
        accounted += c.seconds();
    }
    // Surgery's own time minus the hull calls it makes is not separable here,
    // so the leaf sum states what it covers rather than pretending to total.
    lines.push(format!(
        "PROFILE\tleaf_sum_excl_surgery\t{:.3}\t{:.1}\t-\t-",
        accounted,
        100.0 * accounted / total_wall
    ));
    lines.push(format!(
        "PROFILE\tsurgery_incl_its_hull_calls\t{:.3}\t{:.1}\t{}\t{}",
        SURGERY.seconds(),
        100.0 * SURGERY.seconds() / total_wall,
        SURGERY.calls(),
        SURGERY.work()
    ));
    // The raw integers as well, because three of these counters carry TALLIES
    // in the nanosecond field rather than a duration, and a rounded seconds
    // column cannot report a count.
    for c in ALL {
        if c.calls() == 0 {
            continue;
        }
        lines.push(format!(
            "PROFILE_RAW\t{}\t{}\t{}\t{}",
            c.name,
            c.nanos.load(Ordering::Relaxed),
            c.calls(),
            c.work()
        ));
    }
    lines.push(format!("PROFILE\twall\t{total_wall:.3}\t100.0\t-\t-"));
    lines.join("\n")
}
