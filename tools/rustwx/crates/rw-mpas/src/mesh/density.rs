//! The resolution spec: what the user asks for, expressed as DATA.
//!
//! A spec is a background spacing plus a list of refinement regions. Every
//! region names a shape, a spacing and a transition width. Adding a new place
//! to refine -- any box, any cap, any polygon, anywhere, at any spacing, at any
//! ratio -- is a row in this file's `regions` array and touches no code. The
//! only per-shape code in the crate is one `signed_distance` method of about
//! ten lines, and a new shape kind is the only thing that has ever needed one.
//!
//! Example:
//!
//! ```json
//! {
//!   "background_km": 120.0,
//!   "regions": [
//!     { "shape": { "kind": "cap", "center_deg": [39.0, -98.0], "radius_km": 1200 },
//!       "spacing_km": 20.0, "transition_km": 900.0 }
//!   ]
//! }
//! ```

use serde::{Deserialize, Serialize};

use crate::error::{MpasError, MpasResult};
use crate::mesh::geom::{EARTH_RADIUS_M, V3, arc, cross, dot, from_lat_lon, lat_lon, sub, unit};

// THE DEVICE FOOTPRINT MODEL DOES NOT LIVE HERE ANY MORE.
//
// It used to: two file-scope constants, `9798 MiB + cells * 86630 B`, applied
// to every caller. Those are ONE CARD's numbers -- an RTX 5090's -- and the
// help text above them called the result "the measured footprint model"
// without ever saying which card was measured. A 16 GiB owner was told
// 79,717 cells and a 10 GiB owner 5,350, when the 70 SM part that owners
// actually have measures 5,384 MiB fixed instead of 9,798 and holds 133,144
// and 58,777 at the same budgets: 1.6x and 11x out, in opposite directions.
//
// The fixed term is a property of the PART, because CUDA sizes the
// per-context local-memory backing store from the card's resident-thread
// capacity. It is therefore derived per card in [`crate::mesh::footprint`],
// and a card nobody has measured is refused by name rather than handed a
// neighbour's number.

/// Where a region refines. One method, `signed_distance`, negative inside.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Shape {
    /// A spherical cap: everything within `radius_km` of a point.
    Cap {
        /// `[latitude, longitude]` in degrees.
        center_deg: [f64; 2],
        radius_km: f64,
    },
    /// A latitude/longitude box. `lon_deg` may wrap through the antimeridian.
    LatLonBox {
        lat_deg: [f64; 2],
        lon_deg: [f64; 2],
    },
    /// A spherical polygon, vertices `[latitude, longitude]` in degrees, wound
    /// either way.
    ///
    /// A closed ring on a sphere bounds TWO discs and the winding cannot pick
    /// between them once "either way" is promised, so the region is the
    /// SMALLER of the two -- which is what a refinement window is. A polygon
    /// covering more than half the globe is therefore not expressible, and the
    /// complement is what such a request would deliver; `region_attainment`
    /// prints the `interior_depth_km` it actually resolved, so the swap is
    /// visible before anything is generated.
    Polygon { vertices_deg: Vec<[f64; 2]> },
}

impl Shape {
    /// Signed great-circle distance from `p` to the shape's boundary, in
    /// radians: negative inside, positive outside.
    ///
    /// This is a CONVENIENCE DOOR. The arithmetic lives once, in
    /// [`PreparedShape`], and this method prepares per call -- which is
    /// exactly the work it did inline before that type existed (a polygon
    /// rebuilt its unit-vector ring and re-solved its own winding on every
    /// evaluation). A caller in a hot loop prepares once; a caller asking one
    /// question keeps this spelling. There is no second definition to drift.
    pub fn signed_distance(&self, p: V3) -> f64 {
        self.prepare().signed_distance(p)
    }

    /// Everything about this shape that does not depend on the query point.
    pub fn prepare(&self) -> PreparedShape {
        match self {
            Shape::Cap {
                center_deg,
                radius_km,
            } => PreparedShape::Cap {
                centre: from_lat_lon(center_deg[0].to_radians(), center_deg[1].to_radians()),
                radius_rad: radius_km * 1000.0 / EARTH_RADIUS_M,
            },
            Shape::LatLonBox { lat_deg, lon_deg } => {
                let (a, b) = (lat_deg[0].to_radians(), lat_deg[1].to_radians());
                PreparedShape::LatLonBox {
                    lat0: a.min(b),
                    lat1: a.max(b),
                    lon_a: lon_deg[0].to_radians(),
                    lon_b: lon_deg[1].to_radians(),
                }
            }
            Shape::Polygon { vertices_deg } => {
                let ring: Vec<V3> = vertices_deg
                    .iter()
                    .map(|v| from_lat_lon(v[0].to_radians(), v[1].to_radians()))
                    .collect();
                // The ring's own geometry: the great-circle normal and length
                // of every edge (what `arc_segment_distance` used to rebuild
                // per call), the midpoint and half-length that bound each
                // edge, and the winding verdict that decides which of the two
                // discs the ring bounds is the region.
                let n = ring.len();
                let mut edges = Vec::with_capacity(n);
                let mut widest_half = 0.0f64;
                for k in 0..n {
                    let (a, b) = (ring[k], ring[(k + 1) % n]);
                    let normal = unit(cross(a, b));
                    let seg = arc(a, b);
                    let half = 0.5 * seg;
                    let mid = unit(crate::mesh::geom::add(a, b)).unwrap_or(a);
                    widest_half = widest_half.max(half);
                    edges.push(PreparedEdge {
                        a,
                        b,
                        normal,
                        seg,
                        mid,
                    });
                }
                let left_is_region = ring_left_area(&ring) <= 2.0 * std::f64::consts::PI;
                PreparedShape::Polygon {
                    ring,
                    edges,
                    widest_half,
                    left_is_region,
                }
            }
        }
    }

    /// A representative DEEPEST-INSIDE point of the shape.
    ///
    /// The ramp is a `tanh` centred on the boundary, so the requested spacing
    /// is approached asymptotically inwards and the point that gets closest to
    /// it is the one furthest inside. That point is what
    /// [`MeshSpec::region_attainment`] evaluates the field at, so a caller can
    /// be told the spacing the request will ACTUALLY reach before the
    /// relaxation is spent rather than after.
    ///
    /// Exact for a cap (its centre) and for a box (its centre). For a polygon
    /// it is the normalised vertex centroid, which is inside a convex ring and
    /// is only representative otherwise; `region_attainment` therefore takes
    /// the deeper of this point and a lattice search, so a non-convex ring
    /// cannot make the reported attainment look worse than the field's.
    pub fn interior_point(&self) -> V3 {
        match self {
            Shape::Cap { center_deg, .. } => {
                from_lat_lon(center_deg[0].to_radians(), center_deg[1].to_radians())
            }
            Shape::LatLonBox { lat_deg, lon_deg } => {
                let lat = 0.5 * (lat_deg[0] + lat_deg[1]);
                let tau = std::f64::consts::TAU;
                let wrap = |x: f64| ((x % tau) + tau) % tau;
                let (a, b) = (lon_deg[0].to_radians(), lon_deg[1].to_radians());
                let lon = a + 0.5 * wrap(b - a);
                from_lat_lon(lat.to_radians(), lon)
            }
            Shape::Polygon { vertices_deg } => {
                let mut acc = [0.0f64; 3];
                for v in vertices_deg {
                    let p = from_lat_lon(v[0].to_radians(), v[1].to_radians());
                    acc[0] += p[0];
                    acc[1] += p[1];
                    acc[2] += p[2];
                }
                let n = (acc[0] * acc[0] + acc[1] * acc[1] + acc[2] * acc[2]).sqrt();
                if n <= 0.0 {
                    // Vertices that cancel to the centre of the sphere carry no
                    // direction; the lattice search is then the whole answer.
                    return [0.0, 0.0, 1.0];
                }
                [acc[0] / n, acc[1] / n, acc[2] / n]
            }
        }
    }
}

/// What one region's request will ACTUALLY deliver, reported before anything
/// is generated.
///
/// `spacing_km` in a spec is the ASYMPTOTE of a ramp centred on the region
/// boundary: a region several transition widths across reads its request to
/// one part in a thousand, and a region narrower than its own ramp never gets
/// there. That is a property of the ramp and not a defect -- but a door that
/// prints "15 km" and hands back 22 km without saying so IS one, and this
/// structure is what lets it say so.
///
/// `widest_transition_km` is solved against the FIELD, by bisection on the
/// real evaluation rather than by inverting the ramp in closed form, so it
/// stays true if the ramp's shape is ever changed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegionAttainment {
    /// The spacing the region asked for, in km.
    pub requested_spacing_km: f64,
    /// The finest spacing the field actually reaches inside it, in km.
    pub attained_spacing_km: f64,
    /// `attained / requested`. 1.0 means the request is met.
    pub attained_over_requested: f64,
    /// How far inside the region its deepest point sits, in km.
    pub interior_depth_km: f64,
    /// The transition width this region asked for, in km.
    pub transition_km: f64,
    /// The WIDEST ramp that would still meet the request to within
    /// `tolerance`, in km, or `None` when no ramp does (the region is too
    /// small for the requested spacing to be resolved at all).
    pub widest_transition_km: Option<f64>,
    /// The fraction the request is considered met within.
    pub tolerance: f64,
}

/// One polygon edge with everything that does not depend on the query point.
#[derive(Debug, Clone)]
pub struct PreparedEdge {
    a: V3,
    b: V3,
    /// `unit(cross(a, b))`, the edge's great-circle normal. `None` for a
    /// degenerate edge, which the distance falls back on exactly as before.
    normal: Option<V3>,
    /// `arc(a, b)`.
    seg: f64,
    /// Great-circle midpoint of the edge, the centre of its own bound.
    mid: V3,
}

/// A [`Shape`] with its per-shape constants solved once.
///
/// This is the ONE definition of the signed distance field; [`Shape`] delegates
/// to it. Nothing here changes an arithmetic result -- every value is the same
/// expression on the same inputs, hoisted out of the query loop. The polygon
/// arm additionally carries a PROVEN enclosing cap (see
/// [`PreparedShape::enclosing_cap`]) that lets a resolution field skip a region
/// whose ramp has already saturated, which is a statement about the ramp and
/// not an approximation of it.
#[derive(Debug, Clone)]
pub enum PreparedShape {
    Cap {
        centre: V3,
        radius_rad: f64,
    },
    LatLonBox {
        lat0: f64,
        lat1: f64,
        lon_a: f64,
        lon_b: f64,
    },
    Polygon {
        ring: Vec<V3>,
        edges: Vec<PreparedEdge>,
        /// The longest edge's half-length: the slack a midpoint bound needs.
        widest_half: f64,
        /// Whether the disc to the LEFT of the ring's traversal is the region.
        left_is_region: bool,
    },
}

impl PreparedShape {
    /// Signed great-circle distance from `p` to the shape's boundary, in
    /// radians: negative inside, positive outside.
    pub fn signed_distance(&self, p: V3) -> f64 {
        match self {
            PreparedShape::Cap { centre, radius_rad } => arc(*centre, p) - *radius_rad,
            PreparedShape::LatLonBox {
                lat0,
                lat1,
                lon_a,
                lon_b,
            } => {
                // Standard box distance field, evaluated in the locally scaled
                // latitude/longitude plane. Exact along the meridians and
                // accurate to the sphere's curvature over the parallels, which
                // is what a refinement ramp needs; it is not a geodesic
                // distance to a rhumb boundary and is not used as one.
                let (lat, lon) = lat_lon(p);
                let d_lat = (*lat0 - lat).max(lat - *lat1);
                let d_lon =
                    lon_interval_distance(lon, *lon_a, *lon_b) * lat.cos().abs().max(1e-12);
                let (qx, qy) = (d_lat.max(0.0), d_lon.max(0.0));
                (qx * qx + qy * qy).sqrt() + d_lat.max(d_lon).min(0.0)
            }
            PreparedShape::Polygon {
                ring,
                edges,
                widest_half,
                left_is_region,
            } => {
                // The minimum over the edges, with a MIDPOINT BOUND that skips
                // edges which cannot hold it. `dist(p, edge) >= arc(p, mid) -
                // half` by the triangle inequality, so an edge whose bound
                // already exceeds the running best cannot lower the minimum --
                // and the minimum of a set is unchanged by dropping members
                // that are provably not the minimum. The running best only
                // falls, so a skip stays valid for the rest of the loop. The
                // bound is tested as a DOT PRODUCT against `cos(best + half)`,
                // which is recomputed only when `best` improves.
                let mut best = edge_distance(p, &edges[0]);
                let mut cos_gate = cos_gate_for(best, *widest_half);
                for e in &edges[1..] {
                    if dot(p, e.mid) < cos_gate {
                        continue;
                    }
                    let d = edge_distance(p, e);
                    if d < best {
                        best = d;
                        cos_gate = cos_gate_for(best, *widest_half);
                    }
                }
                if polygon_contains_wound(ring, *left_is_region, p) {
                    -best
                } else {
                    best
                }
            }
        }
    }

    /// A spherical cap that provably CONTAINS the whole shape, or `None` when
    /// no cap under a quarter turn does.
    ///
    /// The containment is what makes the caller's saturation skip sound: for
    /// `p` outside the cap, `p` is outside the shape and
    /// `signed_distance(p) >= arc(centre, p) - radius`. For a polygon the cap
    /// is centred on the normalised vertex sum with the radius set by the
    /// farthest vertex; a cap under a quarter turn is convex, so every edge
    /// arc stays inside it, and the disc the ring bounds INSIDE such a cap has
    /// area under 2*pi -- so it is the smaller disc, which is the region.
    pub fn enclosing_cap(&self) -> Option<(V3, f64)> {
        let quarter = std::f64::consts::FRAC_PI_2;
        match self {
            // A cap and a box are already one cheap evaluation; bounding them
            // would cost what it saves, so they decline rather than pretend.
            PreparedShape::Cap { .. } | PreparedShape::LatLonBox { .. } => None,
            PreparedShape::Polygon { ring, .. } => {
                let mut acc = [0.0f64; 3];
                for &v in ring {
                    acc = crate::mesh::geom::add(acc, v);
                }
                let centre = unit(acc)?;
                let mut radius = 0.0f64;
                for &v in ring {
                    radius = radius.max(arc(centre, v));
                }
                if radius < quarter { Some((centre, radius)) } else { None }
            }
        }
    }

    /// A representative DEEPEST-INSIDE point, the prepared twin of
    /// [`Shape::interior_point`]. Exact for a cap and a box; the normalised
    /// vertex centroid for a polygon, which is inside a convex ring and only
    /// representative otherwise. It is one probe among many here, so a
    /// non-convex ring costs nothing but a redundant point.
    pub fn interior_point(&self) -> V3 {
        match self {
            PreparedShape::Cap { centre, .. } => *centre,
            PreparedShape::LatLonBox {
                lat0,
                lat1,
                lon_a,
                lon_b,
            } => box_centre(*lat0, *lat1, *lon_a, *lon_b),
            PreparedShape::Polygon { ring, .. } => {
                let mut acc = [0.0f64; 3];
                for &v in ring {
                    acc = crate::mesh::geom::add(acc, v);
                }
                unit(acc).unwrap_or([0.0, 0.0, 1.0])
            }
        }
    }

    /// A spherical cap that provably contains this shape's BOUNDARY, or `None`
    /// when no cap under a quarter turn does.
    ///
    /// NOT [`PreparedShape::enclosing_cap`], and the difference is
    /// load-bearing rather than stylistic. That one exists to underwrite the
    /// evaluation-time saturation skip, which needs
    /// `arc(c, p) - radius <= signed_distance(p)` -- a statement about the
    /// FIELD -- and it declines for a cap and a box because "bounding them
    /// would cost what it saves": both are already one cheap evaluation. This
    /// contract is weaker and purely geometric, so all three shapes can honour
    /// it, and `None` here is not a failure -- it only means the disc shortcut
    /// in [`PreparedShape::boundary_shell_probes`] is unavailable and the
    /// per-shape walk is used instead.
    ///
    /// THE BOX ARM IS EXACT, not sampled. With the box centre at latitude
    /// `lc`, `cos(arc)` to a point at `(lat, lon)` is
    /// `sin(lc) sin(lat) + cos(lc) cos(lat) cos(dlon)`. In `dlon` that falls
    /// monotonically out to half a turn, so the extreme longitude offset is
    /// `min(width/2, pi)`; at any fixed `dlon` the expression is
    /// `R cos(lat - phi)`, which is unimodal in `lat`, so its minimum over a
    /// latitude interval is at an endpoint. The farthest point of the whole
    /// parameter rectangle -- boundary and interior alike -- is therefore one
    /// of those two corner candidates.
    pub fn probe_bound(&self) -> Option<(V3, f64)> {
        let quarter = std::f64::consts::FRAC_PI_2;
        match self {
            PreparedShape::Cap { centre, radius_rad } => Some((*centre, *radius_rad)),
            PreparedShape::LatLonBox {
                lat0,
                lat1,
                lon_a,
                lon_b,
            } => {
                let centre = box_centre(*lat0, *lat1, *lon_a, *lon_b);
                let tau = std::f64::consts::TAU;
                let wrap = |x: f64| ((x % tau) + tau) % tau;
                let width = wrap(*lon_b - *lon_a);
                let dmax = (0.5 * width).min(std::f64::consts::PI);
                let (_, lon_c) = lat_lon(centre);
                let mut radius = 0.0f64;
                for lat in [*lat0, *lat1] {
                    for sign in [1.0f64, -1.0] {
                        radius = radius.max(arc(centre, from_lat_lon(lat, lon_c + sign * dmax)));
                    }
                }
                if radius < quarter { Some((centre, radius)) } else { None }
            }
            PreparedShape::Polygon { ring, .. } => {
                // A cap under a quarter turn is convex, so every edge arc
                // between two contained vertices stays inside it.
                let mut acc = [0.0f64; 3];
                for &v in ring {
                    acc = crate::mesh::geom::add(acc, v);
                }
                let centre = unit(acc)?;
                let mut radius = 0.0f64;
                for &v in ring {
                    radius = radius.max(arc(centre, v));
                }
                if radius < quarter { Some((centre, radius)) } else { None }
            }
        }
    }

    /// Probe points covering the TRANSITION SHELL
    /// `{ p : |signed_distance(p)| <= half_width_rad }`, spaced `gn` across the
    /// boundary and `gt` along it, appended to `out`.
    ///
    /// `align_rad` is the signed distance at which this region's ramp is
    /// steepest -- `W ln(rho) / 2`, known in closed form. It is placed on the
    /// across-boundary grid EXPLICITLY, on both sides, so the peak surface is
    /// hit rather than approached; the grid around it is what catches the
    /// composite peak of a nested ladder, where a rung's real neighbour is the
    /// next rung and not the global background.
    ///
    /// `false` means the budget was exhausted. Whatever was generated STAYS in
    /// `out` -- more points can only raise a `max` -- but a caller must never
    /// report a partial set as complete.
    ///
    /// THE CONTRACT IS SHAPE-AGNOSTIC, and so is the test that proves it:
    /// draw an independent point cloud FROM THE SHAPE'S OWN BOUND (never from
    /// a globe lattice, which would reproduce inside the guard the very defect
    /// the guard exists for), keep the points inside the shell, and demand
    /// each lie within `sqrt(gn^2 + gt^2)` of a probe. The test names no
    /// shape, so a new arm inherits the obligation instead of being trusted.
    pub fn boundary_shell_probes(
        &self,
        half_width_rad: f64,
        align_rad: f64,
        gn: f64,
        gt: f64,
        budget: &mut usize,
        out: &mut Vec<V3>,
    ) -> bool {
        let mut probes = ShellProbes {
            shape: self,
            keep: half_width_rad + gn,
            budget: *budget,
            out,
            exhausted: false,
        };
        probes.offer(self.interior_point());

        // THE DISC SHORTCUT, open to every shape: when a cap containing the
        // boundary is no wider than the shell, the whole shell lies inside one
        // cap and filling that cap beats tracing the boundary. It is what
        // keeps a many-vertex swath polygon affordable -- its ramp is wider
        // than the polygon itself, so the per-edge bands would overlap dozens
        // of times over.
        let mut done = false;
        if let Some((c, rb)) = self.probe_bound() {
            if rb <= half_width_rad {
                let r_hi = (rb + half_width_rad).min(std::f64::consts::PI);
                probes.annulus(c, 0.0, r_hi, &[rb + align_rad, rb - align_rad], gn, gt);
                done = true;
            }
        }
        if !done {
            match self {
                PreparedShape::Cap { centre, radius_rad } => {
                    let lo = (radius_rad - half_width_rad).max(0.0);
                    let hi = (radius_rad + half_width_rad).min(std::f64::consts::PI);
                    probes.annulus(
                        *centre,
                        lo,
                        hi,
                        &[radius_rad + align_rad, radius_rad - align_rad],
                        gn,
                        gt,
                    );
                }
                PreparedShape::LatLonBox {
                    lat0,
                    lat1,
                    lon_a,
                    lon_b,
                } => probes.box_shell(
                    *lat0,
                    *lat1,
                    *lon_a,
                    *lon_b,
                    half_width_rad,
                    align_rad,
                    gn,
                    gt,
                ),
                PreparedShape::Polygon { edges, .. } => {
                    probes.polygon_shell(edges, half_width_rad, align_rad, gn, gt)
                }
            }
        }
        let exhausted = probes.exhausted;
        *budget = probes.budget;
        !exhausted
    }
}

/// The great-circle midpoint of a latitude/longitude box, honouring a wrap
/// through the antimeridian.
fn box_centre(lat0: f64, lat1: f64, lon_a: f64, lon_b: f64) -> V3 {
    let tau = std::f64::consts::TAU;
    let wrap = |x: f64| ((x % tau) + tau) % tau;
    from_lat_lon(0.5 * (lat0 + lat1), lon_a + 0.5 * wrap(lon_b - lon_a))
}

/// Two orthonormal tangents at `c`. Unlike [`crate::mesh::geom::east_north`]
/// this never declines: a probe generator that dropped a whole region because
/// its centre sat on the polar axis would be a coverage hole with no
/// signature, so the frame is built from whichever axis `c` is least aligned
/// with.
fn orthonormal_frame(c: V3) -> (V3, V3) {
    let axis = if c[2].abs() < 0.9 {
        [0.0, 0.0, 1.0]
    } else {
        [1.0, 0.0, 0.0]
    };
    let e = unit(cross(axis, c)).unwrap_or([1.0, 0.0, 0.0]);
    let n = cross(c, e);
    (e, n)
}

/// The across-boundary offsets: a grid of step `gn` over `[-half, half]`, plus
/// the two closed-form peak offsets placed EXACTLY.
///
/// `None` when the grid would be larger than the budget allows, which the
/// caller turns into `Coverage::Partial` rather than a truncated reading.
fn across_offsets(half: f64, align: f64, gn: f64, budget: usize) -> Option<Vec<f64>> {
    if !(gn.is_finite() && gn > 0.0 && half.is_finite() && half >= 0.0) {
        return None;
    }
    let count = (2.0 * half / gn).ceil();
    if !count.is_finite() || count > budget as f64 {
        return None;
    }
    let n = count as usize;
    let mut v = Vec::with_capacity(n + 4);
    for k in 0..=n {
        v.push(-half + 2.0 * half * (k as f64 / n.max(1) as f64));
    }
    for a in [align, -align] {
        if a.is_finite() && a.abs() <= half {
            v.push(a);
        }
    }
    Some(v)
}

/// The probe generator's shared state: a budget that stops a pathological
/// request instead of truncating its answer, and the shell filter that keeps
/// the set defined against the REAL signed distance rather than against a
/// model of it.
struct ShellProbes<'a> {
    shape: &'a PreparedShape,
    /// `half_width + gn`: a candidate this close to the boundary is inside the
    /// shell, or close enough to it that keeping it can only help.
    keep: f64,
    budget: usize,
    out: &'a mut Vec<V3>,
    exhausted: bool,
}

impl ShellProbes<'_> {
    fn offer(&mut self, p: V3) {
        if self.exhausted {
            return;
        }
        if self.budget == 0 {
            self.exhausted = true;
            return;
        }
        self.budget -= 1;
        if self.shape.signed_distance(p).abs() <= self.keep {
            self.out.push(p);
        }
    }

    /// Rings about `centre` from `r_lo` to `r_hi`: radial step `gn`, arc step
    /// `gt`, with every radius in `exact` placed on the grid as well.
    fn annulus(&mut self, centre: V3, r_lo: f64, r_hi: f64, exact: &[f64], gn: f64, gt: f64) {
        let half = 0.5 * (r_hi - r_lo);
        let mid = 0.5 * (r_hi + r_lo);
        let Some(offsets) = across_offsets(half, 0.0, gn, self.budget) else {
            self.exhausted = true;
            return;
        };
        let (e, n) = orthonormal_frame(centre);
        let mut radii: Vec<f64> = offsets.iter().map(|d| mid + d).collect();
        for &r in exact {
            if r >= r_lo && r <= r_hi {
                radii.push(r);
            }
        }
        for rr in radii {
            if self.exhausted {
                return;
            }
            if rr <= 0.0 {
                self.offer(centre);
                continue;
            }
            let circumference = std::f64::consts::TAU * rr.sin().abs();
            let m = ((circumference / gt).ceil().max(1.0)).min(self.budget as f64 + 1.0) as usize;
            for j in 0..m.max(1) {
                let az = std::f64::consts::TAU * j as f64 / m.max(1) as f64;
                let dir = crate::mesh::geom::add(
                    crate::mesh::geom::scale(e, az.cos()),
                    crate::mesh::geom::scale(n, az.sin()),
                );
                self.offer(crate::mesh::geom::add(
                    crate::mesh::geom::scale(centre, rr.cos()),
                    crate::mesh::geom::scale(dir, rr.sin()),
                ));
                if self.exhausted {
                    return;
                }
            }
        }
    }

    /// FOUR STRIPS, not a filled sweep: the interior of a large box is
    /// saturated and carries nothing.
    ///
    /// The parameter box is EXACT rather than padded by guess. The box's
    /// signed distance is built from `d_lat` and `d_lon * cos(lat)`: outside,
    /// `s = sqrt(qx^2 + qy^2) >= max(qx, qy)`, so `|s| <= half` forces BOTH to
    /// be within `half`; inside, `s = max(d_lat, d_lon cos lat)` with both
    /// negative, so `|s| <= half` forces at least ONE of them to be. Either
    /// way the shell lies within `half` of a latitude boundary or within
    /// `half` of a longitude boundary, which is what these four strips are.
    /// Every candidate is then kept only if `signed_distance` agrees.
    #[allow(clippy::too_many_arguments)]
    fn box_shell(
        &mut self,
        lat0: f64,
        lat1: f64,
        lon_a: f64,
        lon_b: f64,
        half: f64,
        align: f64,
        gn: f64,
        gt: f64,
    ) {
        const COS_FLOOR: f64 = 1e-6;
        let tau = std::f64::consts::TAU;
        let quarter = std::f64::consts::FRAC_PI_2;
        let wrap = |x: f64| ((x % tau) + tau) % tau;
        let width = wrap(lon_b - lon_a);
        let Some(offsets) = across_offsets(half, align, gn, self.budget) else {
            self.exhausted = true;
            return;
        };

        // The two parallels: fine in latitude, coarse in longitude.
        for blat in [lat0, lat1] {
            for &d in &offsets {
                let lat = (blat + d).clamp(-quarter, quarter);
                let c = lat.cos().abs().max(COS_FLOOR);
                let ext = half / c;
                let dlon = gt / c;
                let (lo, hi) = if width + 2.0 * ext >= tau {
                    (0.0, tau)
                } else {
                    (-ext, width + ext)
                };
                let steps = ((hi - lo) / dlon).ceil();
                if !steps.is_finite() || steps > self.budget as f64 {
                    self.exhausted = true;
                    return;
                }
                let steps = (steps as usize).max(1);
                for k in 0..=steps {
                    let t = lo + (hi - lo) * (k as f64 / steps as f64);
                    self.offer(from_lat_lon(lat, lon_a + t));
                    if self.exhausted {
                        return;
                    }
                }
            }
        }

        // The two meridian edges: fine in longitude, coarse in latitude. A box
        // spanning every longitude has no meridian edge to walk.
        if width >= tau - 1e-12 {
            return;
        }
        let lat_lo = (lat0 - half).max(-quarter);
        let lat_hi = (lat1 + half).min(quarter);
        let steps = ((lat_hi - lat_lo) / gt).ceil();
        if !steps.is_finite() || steps > self.budget as f64 {
            self.exhausted = true;
            return;
        }
        let steps = (steps as usize).max(1);
        for blon in [lon_a, lon_b] {
            for k in 0..=steps {
                let lat = lat_lo + (lat_hi - lat_lo) * (k as f64 / steps as f64);
                let c = lat.cos().abs().max(COS_FLOOR);
                for &d in &offsets {
                    self.offer(from_lat_lon(lat, blon + d / c));
                    if self.exhausted {
                        return;
                    }
                }
            }
        }
    }

    /// Edge bands, each walked from `half` BEFORE its start to `half` past its
    /// end, offset by an exact geodesic rotation about the edge's own
    /// great-circle normal.
    ///
    /// The over-walk is what closes the corner wedges, with no separate vertex
    /// discs to size: a point within `half` of a vertex `v` has both its
    /// along-edge and its across-edge coordinate within `half` of `v`, so it
    /// lies in the extended band of either incident edge. `q cos(b) + n sin(b)`
    /// is an exact geodesic offset -- no tangent-plane approximation, at any
    /// width.
    fn polygon_shell(
        &mut self,
        edges: &[PreparedEdge],
        half: f64,
        align: f64,
        gn: f64,
        gt: f64,
    ) {
        let Some(offsets) = across_offsets(half, align, gn, self.budget) else {
            self.exhausted = true;
            return;
        };
        for e in edges {
            let Some(n) = e.normal else {
                // A degenerate edge has no direction to offset along; its
                // endpoint is a vertex of the two neighbours, whose own
                // over-walked bands cover the disc around it.
                continue;
            };
            let Some(d) = unit(sub(e.b, crate::mesh::geom::scale(e.a, dot(e.a, e.b)))) else {
                continue;
            };
            let (t_lo, t_hi) = (-half, e.seg + half);
            let steps = ((t_hi - t_lo) / gt).ceil();
            if !steps.is_finite() || steps > self.budget as f64 {
                self.exhausted = true;
                return;
            }
            let steps = (steps as usize).max(1);
            for i in 0..=steps {
                let t = t_lo + (t_hi - t_lo) * (i as f64 / steps as f64);
                let q = crate::mesh::geom::add(
                    crate::mesh::geom::scale(e.a, t.cos()),
                    crate::mesh::geom::scale(d, t.sin()),
                );
                for &b in &offsets {
                    self.offer(crate::mesh::geom::add(
                        crate::mesh::geom::scale(q, b.cos()),
                        crate::mesh::geom::scale(n, b.sin()),
                    ));
                    if self.exhausted {
                        return;
                    }
                }
            }
        }
    }
}

/// `cos(best + half)`, the dot-product threshold below which an edge whose
/// midpoint is that far away cannot beat `best`. Saturates to -1 (accept
/// everything) when the sum passes half a turn, where the cosine stops being
/// monotone in the arc.
#[inline]
fn cos_gate_for(best: f64, half: f64) -> f64 {
    let limit = best + half;
    if limit >= std::f64::consts::PI {
        -1.0
    } else {
        limit.cos()
    }
}

/// Great-circle distance from `p` to one prepared edge, in radians. The same
/// arithmetic as [`arc_segment_distance`] with the edge's own constants read
/// rather than rebuilt.
#[inline]
fn edge_distance(p: V3, e: &PreparedEdge) -> f64 {
    let Some(n) = e.normal else {
        return arc(p, e.a);
    };
    let along = sub(p, crate::mesh::geom::scale(n, dot(n, p)));
    let Some(foot) = unit(along) else {
        return std::f64::consts::FRAC_PI_2;
    };
    if arc(e.a, foot) <= e.seg && arc(foot, e.b) <= e.seg {
        arc(p, foot)
    } else {
        arc(p, e.a).min(arc(p, e.b))
    }
}

/// Angular distance from `lon` to the interval `[a, b]`, following the shorter
/// way round and honouring a wrap through the antimeridian. Negative inside.
fn lon_interval_distance(lon: f64, a: f64, b: f64) -> f64 {
    let tau = std::f64::consts::TAU;
    let wrap = |x: f64| ((x % tau) + tau) % tau;
    let (a, b, lon) = (wrap(a), wrap(b), wrap(lon));
    let width = wrap(b - a);
    let offset = wrap(lon - a);
    if offset <= width {
        // inside: distance to the nearer end, negative
        -offset.min(width - offset)
    } else {
        (offset - width).min(tau - offset)
    }
}

/// Great-circle distance from `p` to the arc segment `a -> b`, in radians.
///
/// RETIRED FROM THE PRODUCTION PATH and kept as the test oracle: it rebuilds
/// the edge's normal and length on every call, which
/// [`PreparedShape::signed_distance`] now reads from the prepared ring
/// instead. `the_prepared_field_is_bit_identical_to_the_per_call_arithmetic`
/// pins the two together, so the hoist cannot become a divergence.
#[cfg(test)]
fn arc_segment_distance(p: V3, a: V3, b: V3) -> f64 {
    let Some(n) = unit(cross(a, b)) else {
        return arc(p, a);
    };
    // Distance to the full great circle, and where along it the foot lies.
    let along = sub(p, crate::mesh::geom::scale(n, dot(n, p)));
    let Some(foot) = unit(along) else {
        // p is a pole of the arc's great circle: equidistant from every point
        return std::f64::consts::FRAC_PI_2;
    };
    let seg = arc(a, b);
    if arc(a, foot) <= seg && arc(foot, b) <= seg {
        arc(p, foot)
    } else {
        arc(p, a).min(arc(p, b))
    }
}

/// Area of the disc lying to the LEFT of a closed ring's traversal, in
/// steradians, by Gauss-Bonnet on a geodesic polygon: `A = 2*pi - sum(turning
/// angles)`. The result is in `(0, 4*pi)`, and reversing the ring's winding
/// maps `A -> 4*pi - A` -- which is exactly the symmetry
/// [`polygon_contains`] needs to honour `Shape::Polygon`'s "wound either way".
fn ring_left_area(ring: &[V3]) -> f64 {
    let n = ring.len();
    let mut turn = 0.0f64;
    for k in 0..n {
        let prev = ring[(k + n - 1) % n];
        let cur = ring[k];
        let next = ring[(k + 1) % n];
        // Direction of travel arriving at `cur`, and leaving it.
        let arrive = unit(crate::mesh::geom::tangent_at(cur, sub(cur, prev)));
        let leave = unit(crate::mesh::geom::tangent_at(cur, sub(next, cur)));
        let (Some(arrive), Some(leave)) = (arrive, leave) else {
            // A repeated or antipodal vertex carries no turn to measure.
            continue;
        };
        turn += dot(cross(arrive, leave), cur).atan2(dot(arrive, leave));
    }
    2.0 * std::f64::consts::PI - turn
}

/// Spherical point-in-polygon.
///
/// THE BREAKAGE THIS FIXES, MEASURED (ledger #367, 2026-08-26). This test used
/// to accept on `total.abs() > PI`, and the magnitude alone cannot tell a
/// point inside the ring from a point whose ANTIPODE is inside it: the azimuth
/// of the ring seen from `p` winds once in both cases, positively for the
/// first and negatively for the second. Every polygon region therefore refined
/// a GHOST COPY OF ITSELF on the far side of the globe, and that ghost's
/// boundary is a step and not a ramp -- the signed distance jumps from
/// -19,900 km to +19,900 km across it, both ends of a `tanh` that saturated
/// thousands of widths ago, so the spacing field falls straight from the
/// region's spacing to the background across a single cell. Measured on the
/// swath layer's own emitted spec: `rw_mpas_mesh` refused it on the gradient
/// gate at **1775 %/cell** at 4 km, 525 % at 12 km, 275 % at 20 km, 150 % at
/// 30 km -- exactly `background/spacing - 1` every time, the signature of a
/// field with no ramp at all -- while the equivalent cap at the same place,
/// background and spacing cleared the gate and built. Every storm-following
/// swath this program can place was unbuildable. `--dry-run` did not catch it
/// because it does not apply the gradient gate; what it reported instead was
/// the same ghost seen from the reporting side, `interior_depth_km` near
/// 19,900 km and the request "exactly met" at every size.
///
/// WHAT IT DOES NOW. `total` is `+2*pi` when `p` lies in the disc to the LEFT
/// of the traversal, `-2*pi` when `p`'s antipode does, and `0` when neither.
/// [`Shape::Polygon`] promises the ring may be "wound either way", so the
/// orientation cannot be the thing that picks a side; the SMALLER of the two
/// discs the ring bounds is the region, which is the same answer for both
/// windings and is what a refinement window means.
#[cfg(test)]
fn polygon_contains(ring: &[V3], p: V3) -> bool {
    polygon_contains_wound(ring, ring_left_area(ring) <= 2.0 * std::f64::consts::PI, p)
}

/// [`polygon_contains`] with the ring's winding verdict already solved.
///
/// `ring_left_area` depends on the RING ALONE, never on `p`, so a caller that
/// evaluates one polygon many times solves it once. Same comparison, same
/// arithmetic; the split exists only so the loop stops recomputing a constant.
fn polygon_contains_wound(ring: &[V3], left_is_region: bool, p: V3) -> bool {
    let mut total = 0.0f64;
    for k in 0..ring.len() {
        let a = crate::mesh::geom::tangent_at(p, sub(ring[k], p));
        let b = crate::mesh::geom::tangent_at(p, sub(ring[(k + 1) % ring.len()], p));
        let (Some(a), Some(b)) = (unit(a), unit(b)) else {
            return true; // p is a vertex
        };
        total += dot(cross(a, b), p).atan2(dot(a, b));
    }
    let pi = std::f64::consts::PI;
    if left_is_region { total > pi } else { total < -pi }
}

/// How wide the ramp from a region's spacing out to the background is.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Transition {
    /// Half-width in kilometres.
    Km(f64),
    /// Half-width counted in cells of the REGION's own spacing.
    Cells(f64),
}

/// One refinement region.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Region {
    pub shape: Shape,
    /// Target cell spacing inside the region, hexagon across-flats, in km.
    pub spacing_km: f64,
    #[serde(flatten)]
    pub transition: TransitionField,
}

/// `transition_km` or `transition_cells`, whichever the spec writes.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum TransitionField {
    #[serde(rename = "transition_km")]
    Km(f64),
    #[serde(rename = "transition_cells")]
    Cells(f64),
}

impl TransitionField {
    fn width_rad(&self, spacing_km: f64) -> f64 {
        match self {
            TransitionField::Km(km) => km * 1000.0 / EARTH_RADIUS_M,
            TransitionField::Cells(n) => n * spacing_km * 1000.0 / EARTH_RADIUS_M,
        }
    }
}

/// The whole resolution request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeshSpec {
    /// Spacing far from every region, hexagon across-flats, in km.
    pub background_km: f64,
    #[serde(default)]
    pub regions: Vec<Region>,
    /// Optional human label carried into the file's provenance attributes.
    #[serde(default)]
    pub name: Option<String>,
}

/// The production transition width, for reference in a receipt: the published
/// variable-resolution mesh ramps from 90% to 10% of its density over 81 cells.
/// A spec asking for fewer is steeper than anything NCAR publishes, and the
/// receipt says so with the number rather than refusing a legitimate request.
pub const PUBLISHED_RAMP_CELLS: f64 = 81.0;

/// Whether a probe set covered every locus where a field varies.
///
/// THE BREAKAGE THIS PREVENTS. [`steepest_gradient_reading_of`] reports a
/// `max`, so a probe set that never touched a feature reports 0.0 -- and
/// [`crate::mesh::hierarchy`] maps a zero gradient to a band of
/// `f64::INFINITY`, the most permissive verdict it has. Zero is not evidence
/// of a flat field; it is equally the signature of an instrument that never
/// looked, and the two used to be spelled the same. MEASURED on the shipped
/// sampler: a 3 km cap at 0.15 km spacing on a 76.8 km background -- a 512:1
/// refinement -- read exactly 0.0000 %/cell and was ADMITTED, because none of
/// the 50,000 lattice points landed inside its 24 km detection radius. A
/// `Complete` coverage is what earns a zero the right to mean flat.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Coverage {
    /// Every locus where the field varies was covered to the stated radius.
    Complete,
    /// Loci this field could not cover inside its probe budget, named for the
    /// refusal text.
    Partial(Vec<String>),
}

impl Coverage {
    pub fn is_complete(&self) -> bool {
        matches!(self, Coverage::Complete)
    }
    /// The single word a receipt carries, so a reader -- and a downstream gate
    /// -- can tell a measured reading from an unmeasured one.
    pub fn as_receipt_word(&self) -> &'static str {
        match self {
            Coverage::Complete => "complete",
            Coverage::Partial(_) => "partial",
        }
    }
    pub fn missed(&self) -> &[String] {
        match self {
            Coverage::Complete => &[],
            Coverage::Partial(v) => v,
        }
    }
}

/// One reading of the steepest per-cell spacing change, WITH the certificate
/// that says whether it was taken.
///
/// `per_cell` is a `max` over the probe set, hence always a LOWER bound on the
/// field's true peak: adding probe points can only raise it, never lower it.
/// That one-sidedness is what makes a wider probe set incapable of loosening a
/// gate built on this number.
#[derive(Debug, Clone)]
pub struct GradientReading {
    /// Steepest per-cell spacing change found, as a fraction.
    pub per_cell: f64,
    /// Whether the probe set reached every locus where the field varies.
    pub coverage: Coverage,
    /// How many region-targeted probes were spent. The global lattice is
    /// unioned on top of these and is not counted here.
    pub probe_points: usize,
    /// The finest spacing the FIELD declares, recovered exactly from one
    /// probe -- see [`declared_finest_m_of`].
    pub declared_finest_m: f64,
    /// The finest spacing any probe actually READ.
    pub visited_finest_m: f64,
    /// The coarsest spacing any probe read.
    pub background_m: f64,
}

impl GradientReading {
    /// Did the probe set actually reach the refinement the field asks for?
    ///
    /// An independent check on [`Coverage`], failing differently: coverage
    /// says "I probed every locus I know about", this says "the probes reached
    /// the resolution the spec asks for". A probe set that never left the
    /// background while the field declares something finer measured the
    /// background and not the request, and `per_cell` is then a reading of the
    /// wrong field. Only the WEAK form is asserted -- something deviated at
    /// all -- because the strict form (probes reached within a few percent of
    /// the declared finest) false-alarms on regions that legitimately cannot
    /// attain their own request, which `region_attainment` already reports.
    pub fn saw_the_refinement(&self) -> bool {
        self.declared_finest_m >= self.background_m * (1.0 - 1e-12)
            || self.visited_finest_m < self.background_m * (1.0 - 1e-9)
    }
}

/// The finest spacing a field DECLARES, recovered exactly from one point.
///
/// `density` is `(finest / h)^4` for a spec and `(level_spacing / h)^4` for a
/// level clamp, so `h * density^(1/4)` inverts to the field's own infimum with
/// no second arithmetic and no extra trait method to keep in step. It is exact
/// at every point, which is what lets a reading state what it was LOOKING for
/// beside what it found.
pub fn declared_finest_m_of(field: &impl DensityField, p: V3) -> f64 {
    field.spacing_m(p) * field.density(p).powf(0.25)
}

/// The ceiling on region-targeted probe points one reading may spend.
///
/// A cost ceiling, not a resolution choice: the largest spec in either tree
/// spends well under a tenth of it, and a request that cannot be covered
/// inside it is reported as UNMEASURED ([`Coverage::Partial`]) rather than
/// silently truncated to a max over whatever was reached. A truncated max is a
/// second way to under-read with no signature at all.
pub const PROBE_BUDGET: usize = 4_000_000;

/// How much of a ramp's own peak gradient the shell is allowed to omit. One
/// ramp's peak per-cell gradient sits at signed distance `W ln(rho) / 2`; the
/// tail at `K` widths is `(sqrt(rho)+1)^2 sech^2(K) / 4` of it, which is what
/// [`PreparedSpec::variation_probes`] solves `K` from.
const SHELL_TAIL_FRACTION: f64 = 1e-3;
/// The shell is never thinner than this many transition widths.
const SHELL_MIN_WIDTHS: f64 = 6.0;
/// And never thicker, because it cannot need to be: `tanh(x) == 1.0` exactly
/// in f64 for `x >= 19.07`, which [`PreparedSpec::spacing_m`]'s saturation
/// skip already exploits, so beyond this many widths the field is BIT
/// IDENTICAL to the background and there is nothing left to measure.
const SHELL_SATURATION_WIDTHS: f64 = 19.07;
/// The across-boundary probe step resolves the smaller of the ramp's own width
/// and the step the instrument itself takes inside the refinement.
const ACROSS_STEP_DIVISOR: f64 = 2.0;
/// ...but never coarser than this fraction of the ramp width.
const ACROSS_STEP_FLOOR_DIVISOR: f64 = 64.0;
/// ALONG the boundary the field is the same `tanh` of the same signed
/// distance, so a quarter of the ramp width is enough. MEASURED: sweeping the
/// azimuths per ray from 1 to 256 moved a reading by one part in a million,
/// while sweeping the across-ramp step over a comparable range moved it by
/// 13%. Resolution is needed across the ramp, not along it, and that
/// anisotropy is what makes the cost affordable.
const ALONG_STEP_DIVISOR: f64 = 4.0;

/// What the relaxation actually reads off a resolution request.
///
/// [`MeshSpec`] is the USER's request and stays the only thing a receipt or a
/// stamped provenance ever serialises. The hierarchical (graded) generator
/// relaxes each refinement level under the same request CLAMPED at that
/// level's spacing, and this trait is the seam that lets it do so without a
/// hidden field on `MeshSpec` -- a clamp carried inside the spec would leak
/// into `rw_mesh_spec_json` and every receipt, claiming a request nobody made.
pub trait DensityField {
    /// `meshDensity` at a point; only RATIOS of this matter to the
    /// relaxation (the centroid moment normalises a global scale away).
    fn density(&self, p: V3) -> f64;
    /// The requested spacing at a point, in metres.
    fn spacing_m(&self, p: V3) -> f64;

    /// WHERE this field varies. Appends probe points to `out` and reports
    /// whether the budget covered every locus.
    ///
    /// NO DEFAULT BODY, deliberately. THE BREAKAGE THIS PREVENTS: the gradient
    /// meter steps a distance `h(p)` from each probe, which is right, but it
    /// used to take its probes from a lattice uniform over the WHOLE SPHERE
    /// while every feature it measures lives in a shell a few transition
    /// widths wide around a region boundary. At the 50,000 samples every
    /// caller passes, those points sit 101 km apart; a CPAS-class ladder's
    /// innermost ramp is 3.3 km wide, ZERO lattice points landed within three
    /// widths of it, and the build gate read 12.1 %/cell for a field that does
    /// 38. Two gates in two repositories then agreed, to the last bit, on a
    /// number three times too low.
    ///
    /// RAISING THE SAMPLE COUNT IS NOT THE FIX AND CANNOT BE MADE ONE. Let A
    /// be any procedure that reads a field only through this trait's other two
    /// methods and makes at most N calls. Run A on a uniform field: every
    /// answer is the same constant, so A's adaptive choices are determined and
    /// the point set it queries is fixed. Any N points on the sphere leave an
    /// empty cap of angular radius at least `c / sqrt(N)`, and a legal spec
    /// can put a whole region inside it -- [`MeshSpec::check`] puts no
    /// positive floor on a radius or a transition width, and `tanh` saturates
    /// to exactly 1.0 in f64 beyond 19.07 widths, so that region is compactly
    /// supported and agrees with the uniform field BIT FOR BIT at every point
    /// A looked at. A returns the uniform answer, which is zero. That holds
    /// for every finite N, adaptive or not, randomised or not. The escape is
    /// not more samples: it is the field naming its own loci, which is what
    /// this method is. A field that cannot answer this question has no
    /// business being gated on, so it is a compile error not to answer it.
    fn variation_probes(&self, budget: usize, out: &mut Vec<V3>) -> Coverage;
}

impl DensityField for MeshSpec {
    fn density(&self, p: V3) -> f64 {
        MeshSpec::density(self, p)
    }
    fn spacing_m(&self, p: V3) -> f64 {
        MeshSpec::spacing_m(self, p)
    }
    fn variation_probes(&self, budget: usize, out: &mut Vec<V3>) -> Coverage {
        self.prepared().variation_probes(budget, out)
    }
}

/// A [`MeshSpec`] with every per-region constant solved once.
///
/// THE ONE DEFINITION of the resolution field. `MeshSpec::spacing_m` and
/// `MeshSpec::density` delegate here after preparing, so there is no second
/// arithmetic to drift; what this type removes is the per-evaluation rebuild
/// of things that never depended on the query point -- a polygon's unit-vector
/// ring, its edge normals and lengths, and its winding verdict.
#[derive(Debug, Clone)]
pub struct PreparedSpec {
    background_inv_m: f64,
    finest_m: f64,
    regions: Vec<PreparedRegion>,
}

#[derive(Debug, Clone)]
struct PreparedRegion {
    shape: PreparedShape,
    /// A cap that provably contains the shape, when one under a quarter turn
    /// exists. See [`PreparedSpec::spacing_m`] for what it is used for.
    bound: Option<(V3, f64)>,
    width_rad: f64,
    fine_inv_m: f64,
    back_inv_m: f64,
}

impl PreparedSpec {
    /// The requested spacing at a point, in metres. Same field, same numbers,
    /// as [`MeshSpec::spacing_m`].
    pub fn spacing_m(&self, p: V3) -> f64 {
        let mut inv = self.background_inv_m;
        for r in &self.regions {
            // THE SATURATION SKIP, and why it cannot move a bit.
            //
            // `blend` falls monotonically with the signed distance and is
            // never negative, so if the blend of a PROVEN LOWER BOUND on that
            // distance evaluates to exactly 0.0, the blend of the true
            // distance is exactly 0.0 as well. `here` is then exactly
            // `back_inv_m` -- the same expression `inv` starts at, and `inv`
            // only ever rises -- so `here > inv` is false and the region
            // contributes nothing. Skipping is that statement, not an
            // approximation of it: no threshold, no tolerance, and the test is
            // made on the evaluated blend rather than on an assumed
            // saturation point of `tanh`.
            if let Some((c, radius)) = r.bound {
                let s_lower = arc(c, p) - radius;
                if 0.5 * (1.0 - (s_lower / r.width_rad).tanh()) == 0.0 {
                    continue;
                }
            }
            let s = r.shape.signed_distance(p);
            let blend = 0.5 * (1.0 - (s / r.width_rad).tanh());
            let here = r.back_inv_m + (r.fine_inv_m - r.back_inv_m) * blend;
            if here > inv {
                inv = here;
            }
        }
        1.0 / inv
    }

    /// `meshDensity` at a point. Same field as [`MeshSpec::density`].
    pub fn density(&self, p: V3) -> f64 {
        let ratio = self.finest_m / self.spacing_m(p);
        ratio.powi(4)
    }

    /// This field clamped below at one refinement level's spacing.
    pub fn clamped(&self, level_spacing_m: f64) -> LevelClamp<'_> {
        LevelClamp {
            spec: self,
            level_spacing_m,
        }
    }

    /// WHERE THIS FIELD VARIES -- one transition shell per region.
    ///
    /// `inv(p) = max(background, max_i inv_i(p))`, and outside region `i`'s
    /// shell its `tanh` has saturated so `inv_i` is a constant there. A point
    /// outside EVERY shell therefore has every term locally constant, so `inv`
    /// is, so `h` is: the field's variation is contained in the union of the
    /// shells, up to the tail the `K` rule bounds. Two corollaries this design
    /// leans on. A CROSSOVER between two regions -- where the `max` switches
    /// and the field has a kink -- needs at least one of them unsaturated, so
    /// it lies inside at least one shell and is covered; that is the case a
    /// nested rung stack lives in, and it is why the answer is found there.
    /// And [`PreparedSpec::spacing_m`]'s saturation skip already makes the
    /// field BIT IDENTICAL to the background beyond about 19 widths, so a `K`
    /// in `[6, 19.07]` genuinely brackets every non-constant point.
    ///
    /// The cost follows the REGIONS, not the globe. Per region it is about
    /// `(2 K W / gn) * (L / gt) = C K (L / W)` with `L` the boundary length --
    /// the boundary measured in ramp widths, a dimensionless number the spec
    /// writes. Shrink every length in a request by any factor and `L / W` does
    /// not move, so the probe count does not move either: the same request at
    /// one sixteen-thousandth the size costs the same and reads the same.
    /// Nothing here depends on the radius of the Earth.
    pub fn variation_probes(&self, budget: usize, out: &mut Vec<V3>) -> Coverage {
        let mut missed: Vec<String> = Vec::new();
        let mut left = budget;
        for (i, r) in self.regions.iter().enumerate() {
            // rho = background / region spacing, the refinement ratio.
            let ratio = (r.fine_inv_m / r.back_inv_m).max(1.0);
            if ratio <= 1.0 {
                // A region no finer than the background contributes nothing to
                // the field; `spacing_m` never lets it raise `inv`.
                continue;
            }
            // K SHELL WIDTHS. One ramp's peak per-cell gradient sits at
            // `s = W ln(rho) / 2`; the tail at K widths is
            // `(sqrt(rho) + 1)^2 sech^2(K) / 4` of that peak, so this K holds
            // the omission under SHELL_TAIL_FRACTION of it.
            let k = ((ratio.sqrt() + 1.0) / SHELL_TAIL_FRACTION.sqrt())
                .ln()
                .clamp(SHELL_MIN_WIDTHS, SHELL_SATURATION_WIDTHS);
            // TWO SCALES SET THE ACROSS-RAMP STEP. The ramp's own width, and
            // the step length the instrument itself takes -- because the
            // measure is already adaptive (`step = h / EARTH_RADIUS_M`), and
            // that is the deep reason this fix works: a probe placed INSIDE a
            // refinement takes a short step and can resolve a short ramp,
            // while a globe lattice's points sit almost entirely on the
            // background, where the step is tens of kilometres and nothing
            // narrow is resolvable however many of them there are.
            let h_fine_rad = 1.0 / (r.fine_inv_m * EARTH_RADIUS_M);
            let gn = (r.width_rad.min(h_fine_rad) / ACROSS_STEP_DIVISOR)
                .max(r.width_rad / ACROSS_STEP_FLOOR_DIVISOR);
            let gt = r.width_rad / ALONG_STEP_DIVISOR;
            // The closed-form peak offset, placed exactly rather than
            // approached: s* = W ln(rho) / 2.
            let align = 0.5 * r.width_rad * ratio.ln();
            if !r
                .shape
                .boundary_shell_probes(k * r.width_rad, align, gn, gt, &mut left, out)
            {
                missed.push(format!("region {i}"));
            }
        }
        if missed.is_empty() {
            Coverage::Complete
        } else {
            Coverage::Partial(missed)
        }
    }
}

impl DensityField for PreparedSpec {
    fn density(&self, p: V3) -> f64 {
        PreparedSpec::density(self, p)
    }
    fn spacing_m(&self, p: V3) -> f64 {
        PreparedSpec::spacing_m(self, p)
    }
    fn variation_probes(&self, budget: usize, out: &mut Vec<V3>) -> Coverage {
        PreparedSpec::variation_probes(self, budget, out)
    }
}

/// A resolution request clamped below at one refinement level's spacing:
/// `h_bar_l(x) = max(h_spec(x), h_l)`.
///
/// This is the level field of the hierarchical ladder. It is a VIEW over a
/// borrowed spec, deliberately unserialisable: `nominal_min_dc`, the receipt
/// and the stamped provenance are all computed from the UNCLAMPED spec, and
/// the borrow makes reaching this type from any serialisation path a compile
/// error rather than a discipline.
pub struct LevelClamp<'a> {
    pub spec: &'a PreparedSpec,
    /// The level spacing `h_l` in metres; no spacing below it is asked for.
    pub level_spacing_m: f64,
}

impl DensityField for LevelClamp<'_> {
    fn spacing_m(&self, p: V3) -> f64 {
        self.spec.spacing_m(p).max(self.level_spacing_m)
    }
    /// VERBATIM DELEGATION to the spec, and it is a theorem rather than a
    /// shortcut. `h_bar = max(h, h_l)` is a pointwise monotone map of `h`, so
    /// wherever the spec's field is locally constant the clamped one is too:
    /// the spec's loci are a SUPERSET of the clamp's, and probing the spec's
    /// loci covers the clamp's. The clamp also cannot read HIGHER than the
    /// spec -- for `a >= b > 0`, `max(a,c) / max(b,c) <= a / b` in all three
    /// orderings of `c` -- so nothing is lost by measuring where the spec
    /// varies.
    fn variation_probes(&self, budget: usize, out: &mut Vec<V3>) -> Coverage {
        self.spec.variation_probes(budget, out)
    }
    fn density(&self, p: V3) -> f64 {
        // Normalised so the level's own spacing reads 1.0 -- the same
        // `(h_min / h)^4` law as the spec, with the level floor as h_min.
        let ratio = self.level_spacing_m / self.spacing_m(p);
        ratio.powi(4)
    }
}

impl MeshSpec {
    /// A uniform mesh at one spacing.
    pub fn uniform(background_km: f64) -> MeshSpec {
        MeshSpec {
            background_km,
            regions: Vec::new(),
            name: None,
        }
    }

    /// Parse a spec from JSON, refusing anything that cannot describe a mesh.
    pub fn from_json(text: &str) -> MpasResult<MeshSpec> {
        let spec: MeshSpec = serde_json::from_str(text)
            .map_err(|e| MpasError::Refusal(format!("the resolution spec is not valid JSON: {e}")))?;
        spec.check()?;
        Ok(spec)
    }

    pub fn check(&self) -> MpasResult<()> {
        if !(self.background_km.is_finite() && self.background_km > 0.0) {
            return Err(MpasError::Refusal(format!(
                "background_km is {}; the background spacing sets the cell size over most of the sphere, and a non-positive one makes the target cell count infinite",
                self.background_km
            )));
        }
        for (i, r) in self.regions.iter().enumerate() {
            if !(r.spacing_km.is_finite() && r.spacing_km > 0.0) {
                return Err(MpasError::Refusal(format!(
                    "region {i} asks for a spacing of {} km; a non-positive spacing has no cell count",
                    r.spacing_km
                )));
            }
            let w = r.transition.width_rad(r.spacing_km);
            if !(w.is_finite() && w > 0.0) {
                return Err(MpasError::Refusal(format!(
                    "region {i} has a transition width of {w} rad; a zero-width ramp puts a resolution jump between two adjacent cells, and the time step is set globally by the finest edge on the mesh"
                )));
            }
            if let Shape::Polygon { vertices_deg } = &r.shape {
                if vertices_deg.len() < 3 {
                    return Err(MpasError::Refusal(format!(
                        "region {i} is a polygon with {} vertices; fewer than three enclose no area to refine",
                        vertices_deg.len()
                    )));
                }
            }
            if let Shape::Cap { radius_km, .. } = &r.shape {
                if !(radius_km.is_finite() && *radius_km > 0.0) {
                    return Err(MpasError::Refusal(format!(
                        "region {i} is a cap of radius {radius_km} km, which encloses nothing"
                    )));
                }
            }
        }
        Ok(())
    }

    /// The finest spacing anywhere in the spec, in km.
    pub fn finest_km(&self) -> f64 {
        self.regions
            .iter()
            .map(|r| r.spacing_km)
            .fold(self.background_km, f64::min)
    }

    /// The requested spacing at a point, in metres.
    ///
    /// Regions blend into the background LINEARLY IN 1/h, which is linear in
    /// `meshDensity^(1/4)` -- the form the published variable mesh actually
    /// uses. The blend is a `tanh` ramp, and where two regions overlap the
    /// finer one wins.
    ///
    /// A CONVENIENCE DOOR, like [`Shape::signed_distance`]: the arithmetic is
    /// [`PreparedSpec::spacing_m`] and this prepares per call. Anything asking
    /// the field more than a handful of times should call [`MeshSpec::prepared`]
    /// once -- the relaxation asks it about 10^9 times per mesh.
    pub fn spacing_m(&self, p: V3) -> f64 {
        self.prepared().spacing_m(p)
    }

    /// This request with every per-region constant solved once.
    pub fn prepared(&self) -> PreparedSpec {
        let back_inv_m = 1.0 / (self.background_km * 1000.0);
        PreparedSpec {
            background_inv_m: back_inv_m,
            finest_m: self.finest_km() * 1000.0,
            regions: self
                .regions
                .iter()
                .map(|r| {
                    let shape = r.shape.prepare();
                    let bound = shape.enclosing_cap();
                    PreparedRegion {
                        shape,
                        bound,
                        width_rad: r.transition.width_rad(r.spacing_km),
                        fine_inv_m: 1.0 / (r.spacing_km * 1000.0),
                        back_inv_m,
                    }
                })
                .collect(),
        }
    }

    /// What each region's request will actually deliver -- see
    /// [`RegionAttainment`].
    ///
    /// The deepest interior point is the deeper of the shape's own
    /// representative point and the best of a `samples`-point golden-ratio
    /// lattice, so a non-convex polygon cannot make the answer pessimistic.
    /// The field is evaluated WHOLE at that point, so an overlapping finer
    /// region is credited rather than ignored.
    pub fn region_attainment(&self, samples: usize) -> Vec<RegionAttainment> {
        const TOLERANCE: f64 = 0.05;
        let lattice: Vec<V3> = fibonacci_lattice(samples).collect();
        let mut out = Vec::with_capacity(self.regions.len());
        for (i, r) in self.regions.iter().enumerate() {
            let shape = r.shape.prepare();
            let mut deep = r.shape.interior_point();
            let mut depth = shape.signed_distance(deep);
            for &p in &lattice {
                let s = shape.signed_distance(p);
                if s < depth {
                    depth = s;
                    deep = p;
                }
            }
            let attained_m = self.spacing_m(deep);
            let requested_m = r.spacing_km * 1000.0;
            out.push(RegionAttainment {
                requested_spacing_km: r.spacing_km,
                attained_spacing_km: attained_m / 1000.0,
                attained_over_requested: attained_m / requested_m,
                interior_depth_km: -depth * EARTH_RADIUS_M / 1000.0,
                transition_km: r.transition.width_rad(r.spacing_km) * EARTH_RADIUS_M / 1000.0,
                widest_transition_km: self.widest_transition_km(i, deep, TOLERANCE),
                tolerance: TOLERANCE,
            });
        }
        out
    }

    /// Bisect for the widest ramp on region `i` that still meets its request.
    ///
    /// Monotone by construction -- a wider ramp reaches less far in -- so
    /// bisection is exact to the bracket. Solved against the real field
    /// evaluation, never against an inverted closed form, so this stays
    /// correct if the ramp's shape changes.
    fn widest_transition_km(&self, i: usize, deep: V3, tolerance: f64) -> Option<f64> {
        let target_m = self.regions[i].spacing_km * 1000.0 * (1.0 + tolerance);
        let attained_with = |km: f64| -> f64 {
            let mut trial = self.clone();
            trial.regions[i].transition = TransitionField::Km(km);
            trial.spacing_m(deep)
        };

        // A ramp narrower than the requested spacing itself puts the whole
        // jump inside one cell, which the spec check refuses; if even that
        // does not meet the request, no ramp does.
        let mut lo = self.regions[i].spacing_km;
        if attained_with(lo) > target_m {
            return None;
        }
        // One Earth circumference: a ramp no spec can exceed, so a bracket
        // that still meets the request means the ramp is not the constraint.
        let mut hi = 40_075.0_f64;
        if attained_with(hi) <= target_m {
            return Some(hi);
        }
        for _ in 0..80 {
            let mid = 0.5 * (lo + hi);
            if attained_with(mid) <= target_m {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        Some(lo)
    }

    /// `meshDensity` at a point: the field MPAS scales its horizontal mixing by.
    ///
    /// `rho = (h_min / h(x))^4`, normalised so the finest requested spacing
    /// reads 1.0. The fourth power is the CVT density law `h ~ rho^(-1/(d+2))`
    /// with d = 2; the published variable mesh's density spans exactly
    /// `(1/4)^4 = 1/256` for its designed 4x refinement, which is what pins the
    /// exponent rather than a fit.
    ///
    /// A convenience door; the arithmetic is [`PreparedSpec::density`].
    pub fn density(&self, p: V3) -> f64 {
        self.prepared().density(p)
    }

    /// Predicted cell count for this spec.
    ///
    /// `N = integral over the sphere of dA / A_cell`, with `A_cell` the area of
    /// a regular hexagon of the requested across-flats spacing. Quadrature is a
    /// golden-ratio lattice of `samples` points, which has no polar clustering
    /// to bias a mid-latitude region.
    ///
    /// INSTRUMENT VALIDATION: on the published uniform mesh this predicts
    /// 40,972 against an actual 40,962 (+0.02%); on the published variable mesh
    /// it predicts within 0.4%. RESOLUTION LIMIT: about +/-0.4% on a variable
    /// spec, which is why the generator reports both the target and the
    /// delivered count rather than only the target.
    pub fn predicted_cells(&self, samples: usize) -> f64 {
        predicted_cells_of(&self.prepared(), samples)
    }

    /// Scale every spacing by `k` so the mesh comes out at a chosen cell count.
    /// Ratios between regions are preserved: the SHAPE of the request is the
    /// user's, the size is the card's.
    pub fn scaled(&self, k: f64) -> MeshSpec {
        let mut out = self.clone();
        out.background_km *= k;
        for r in &mut out.regions {
            r.spacing_km *= k;
            if let TransitionField::Km(km) = &mut r.transition {
                *km *= k;
            }
        }
        out
    }

    /// Rescale the spec so it predicts `target` cells, then return it with the
    /// factor applied. Solved by bisection on the scale factor: `N` falls as
    /// `k^-2`, so the bracket is easy and the solve is exact to a cell.
    pub fn fitted_to(&self, target: usize, samples: usize) -> MpasResult<(MeshSpec, f64)> {
        if target < 12 {
            return Err(MpasError::Refusal(format!(
                "a target of {target} cells cannot carry the twelve pentagons every triangulated sphere must have"
            )));
        }
        let (mut lo, mut hi) = (1e-3f64, 1e3f64);
        if self.scaled(hi).predicted_cells(samples) > target as f64 {
            return Err(MpasError::Refusal(format!(
                "even at a thousand times the requested spacing this spec needs more than {target} cells; the region geometry, not the resolution, is what does not fit"
            )));
        }
        for _ in 0..200 {
            let mid = (lo * hi).sqrt();
            if self.scaled(mid).predicted_cells(samples) > target as f64 {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let k = (lo * hi).sqrt();
        Ok((self.scaled(k), k))
    }

    /// The steepest per-cell spacing change the spec asks for, with the
    /// certificate that says whether it was measured. The published variable
    /// mesh runs 1.53% per cell; a receipt quotes this beside that number so a
    /// steeper request is a stated fact rather than a surprise in the output.
    ///
    /// `lattice_samples` now sizes only the GLOBAL arm of the probe set. The
    /// region-targeted arm is sized by the request's own geometry and is not a
    /// caller's choice, because it was never a choice a caller could make
    /// correctly -- see [`steepest_gradient_reading_of`].
    pub fn steepest_gradient_reading(&self, lattice_samples: usize) -> GradientReading {
        steepest_gradient_reading_of(&self.prepared(), lattice_samples)
    }
}

/// [`MeshSpec::predicted_cells`] over any [`DensityField`]: the sizing
/// integral evaluated on the field a relaxation actually runs under, so a
/// ladder level can hold its own delivered count to its own field.
pub fn predicted_cells_of(field: &impl DensityField, samples: usize) -> f64 {
    let mut acc = 0.0f64;
    for p in fibonacci_lattice(samples) {
        let h = field.spacing_m(p);
        acc += 1.0 / (h * h);
    }
    let mean_inv_h2 = acc / samples as f64;
    let sphere_area = 4.0 * std::f64::consts::PI * EARTH_RADIUS_M * EARTH_RADIUS_M;
    sphere_area * mean_inv_h2 / (3f64.sqrt() / 2.0)
}

/// [`MeshSpec::steepest_gradient_reading`] over any [`DensityField`], so the
/// relaxation's oscillation refusal can quote the gradient of the field it
/// actually relaxed under (a level clamp's, not only a spec's).
///
/// WHAT IT MEASURES, unchanged: at each probe `p` it reads `h = spacing_m(p)`,
/// steps that same distance east, north and back, and takes the largest
/// `|h(q)/h(p) - 1|`. The step is the local requested spacing, which is what
/// turns a spatial derivative into the PER-CELL change the band arithmetic
/// `ln 2 / ln(1+g)` needs.
///
/// WHERE IT LOOKS, corrected. The probe set is the field's own declared loci
/// ([`DensityField::variation_probes`]) UNIONED WITH the global lattice, never
/// replacing it. The union is arithmetic and not tidiness: the reading is a
/// `max`, so a strict superset of the old probe set can only raise it, and
/// keeping the lattice is what makes this change structurally incapable of
/// loosening any gate built on the number. It is also not redundant -- on
/// several real specs the region-targeted set ALONE reads slightly LOWER than
/// the lattice, because a lattice point happened to land on a rung crossover
/// where no region boundary is.
///
/// WHAT IT STILL UNDER-READS, so nobody has to rediscover it. The measure is a
/// MEAN of `|dh/dx|` over a step of length `h(p)`, therefore bounded above by
/// the supremum over that step: with sampling eliminated entirely it reads
/// 15-22% below the continuum peak. That bias is one-sided (it can admit, it
/// can never wrongly refuse), it is independent of where the probes are, and
/// this function does not address it. Neither do the two other instruments on
/// the same receipt: `predicted_cells_of` and `region_attainment` still walk
/// the global lattice alone.
pub fn steepest_gradient_reading_of(
    field: &impl DensityField,
    lattice_samples: usize,
) -> GradientReading {
    let mut points: Vec<V3> = Vec::new();
    let coverage = field.variation_probes(PROBE_BUDGET, &mut points);
    let probe_points = points.len();
    points.extend(fibonacci_lattice(lattice_samples));

    let declared_finest_m = declared_finest_m_of(field, [0.0, 0.0, 1.0]);
    let mut background_m = 0.0f64;
    let mut visited_finest_m = f64::INFINITY;
    let mut worst = 0.0f64;
    for p in points {
        let h = field.spacing_m(p);
        background_m = background_m.max(h);
        visited_finest_m = visited_finest_m.min(h);
        let step = h / EARTH_RADIUS_M;
        let (east, north) = match crate::mesh::geom::east_north(p) {
            Some(v) => v,
            None => continue,
        };
        for dir in [east, north] {
            for sign in [1.0f64, -1.0] {
                let q = unit(crate::mesh::geom::add(
                    p,
                    crate::mesh::geom::scale(dir, sign * step),
                ));
                if let Some(q) = q {
                    let hq = field.spacing_m(q);
                    worst = worst.max((hq / h - 1.0).abs());
                }
            }
        }
    }
    GradientReading {
        per_cell: worst,
        coverage,
        probe_points,
        declared_finest_m,
        visited_finest_m,
        background_m,
    }
}

/// Golden-ratio lattice on the sphere: near-uniform, no polar clustering, and
/// no exact symmetry for a degenerate quadruple to hide in.
pub fn fibonacci_lattice(n: usize) -> impl Iterator<Item = V3> {
    let ga = std::f64::consts::PI * (3.0 - 5f64.sqrt());
    (0..n).map(move |k| {
        let z = 1.0 - (2 * k + 1) as f64 / n as f64;
        let r = (1.0 - z * z).max(0.0).sqrt();
        let t = ga * k as f64;
        [r * t.cos(), r * t.sin(), z]
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The RETIRED per-call arithmetic, copied verbatim from the body
    /// `Shape::signed_distance` carried before it was hoisted into
    /// [`PreparedShape`]. It exists only here, only as an oracle, and it is
    /// what makes the hoist provable rather than asserted.
    fn reference_signed_distance(shape: &Shape, p: V3) -> f64 {
        match shape {
            Shape::Cap {
                center_deg,
                radius_km,
            } => {
                let c = from_lat_lon(center_deg[0].to_radians(), center_deg[1].to_radians());
                arc(c, p) - radius_km * 1000.0 / EARTH_RADIUS_M
            }
            Shape::LatLonBox { lat_deg, lon_deg } => {
                let (lat, lon) = lat_lon(p);
                let (lat0, lat1) = (lat_deg[0].to_radians(), lat_deg[1].to_radians());
                let (lat0, lat1) = (lat0.min(lat1), lat0.max(lat1));
                let d_lat = (lat0 - lat).max(lat - lat1);
                let d_lon =
                    lon_interval_distance(lon, lon_deg[0].to_radians(), lon_deg[1].to_radians())
                        * lat.cos().abs().max(1e-12);
                let (qx, qy) = (d_lat.max(0.0), d_lon.max(0.0));
                (qx * qx + qy * qy).sqrt() + d_lat.max(d_lon).min(0.0)
            }
            Shape::Polygon { vertices_deg } => {
                let ring: Vec<V3> = vertices_deg
                    .iter()
                    .map(|v| from_lat_lon(v[0].to_radians(), v[1].to_radians()))
                    .collect();
                let mut best = f64::INFINITY;
                for k in 0..ring.len() {
                    best = best.min(arc_segment_distance(p, ring[k], ring[(k + 1) % ring.len()]));
                }
                if polygon_contains(&ring, p) { -best } else { best }
            }
        }
    }

    /// The RETIRED per-call spacing, likewise verbatim.
    fn reference_spacing_m(spec: &MeshSpec, p: V3) -> f64 {
        let mut inv = 1.0 / (spec.background_km * 1000.0);
        for r in &spec.regions {
            let s = reference_signed_distance(&r.shape, p);
            let w = r.transition.width_rad(r.spacing_km);
            let fine = 1.0 / (r.spacing_km * 1000.0);
            let back = 1.0 / (spec.background_km * 1000.0);
            let blend = 0.5 * (1.0 - (s / w).tanh());
            let here = back + (fine - back) * blend;
            if here > inv {
                inv = here;
            }
        }
        1.0 / inv
    }

    /// A 38-vertex corridor of the shape the swath layer actually emits.
    /// Deliberately lobed rather than convex-and-tidy: this is the shape whose
    /// evaluation cost the hoist exists to remove, so it is the shape the
    /// hoist is proved on.
    fn corridor_polygon() -> Shape {
        let mut vertices_deg = Vec::new();
        for k in 0..38 {
            let t = k as f64 / 38.0 * std::f64::consts::TAU;
            let r = 1.8 + 0.8 * (3.0 * t).sin().abs();
            vertices_deg.push([14.4 + r * t.sin(), -141.0 + r * t.cos() / 0.97]);
        }
        Shape::Polygon { vertices_deg }
    }

    /// THE BREAKAGE THIS PREVENTS: the resolution field was hoisted out of the
    /// query loop -- the unit-vector ring, its edge normals and lengths, its
    /// winding verdict, and a saturation skip on a proven enclosing cap. A
    /// mesh registry pins grid files by SHA-256, so one moved bit anywhere in
    /// this field makes every registered digest downstream unreproducible, and
    /// the move would be invisible because a resolution field has no oracle of
    /// its own. Bit equality, not a tolerance.
    #[test]
    fn the_prepared_field_is_bit_identical_to_the_per_call_arithmetic() {
        let specs = [
            MeshSpec::uniform(120.0),
            MeshSpec {
                background_km: 75.0,
                regions: vec![Region {
                    shape: corridor_polygon(),
                    spacing_km: 6.0,
                    transition: TransitionField::Cells(81.0),
                }],
                name: None,
            },
            MeshSpec {
                background_km: 200.0,
                regions: vec![
                    Region {
                        shape: Shape::Cap {
                            center_deg: [39.0, -98.0],
                            radius_km: 1200.0,
                        },
                        spacing_km: 20.0,
                        transition: TransitionField::Km(900.0),
                    },
                    Region {
                        shape: Shape::LatLonBox {
                            lat_deg: [-40.0, -20.0],
                            lon_deg: [170.0, -170.0],
                        },
                        spacing_km: 50.0,
                        transition: TransitionField::Km(600.0),
                    },
                    Region {
                        shape: corridor_polygon(),
                        spacing_km: 8.0,
                        transition: TransitionField::Cells(40.0),
                    },
                ],
                name: None,
            },
        ];
        for (n, spec) in specs.iter().enumerate() {
            let prepared = spec.prepared();
            let mut skipped = 0usize;
            for p in fibonacci_lattice(60_000) {
                let want = reference_spacing_m(spec, p);
                let got = prepared.spacing_m(p);
                assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "spec {n}: the prepared field answers {got} where the per-call arithmetic answers {want}"
                );
                assert_eq!(spec.density(p).to_bits(), prepared.density(p).to_bits());
                for r in &spec.regions {
                    let a = reference_signed_distance(&r.shape, p);
                    let b = r.shape.prepare().signed_distance(p);
                    assert_eq!(a.to_bits(), b.to_bits(), "spec {n}: a signed distance moved");
                }
                if let Some((c, rad)) = prepared.regions.first().and_then(|r| r.bound) {
                    if arc(c, p) - rad > 0.0 {
                        skipped += 1;
                    }
                }
            }
            if n == 1 {
                // The skip has to be REACHED as well as correct: a bound that
                // never fires would leave this test proving nothing about it.
                assert!(
                    skipped > 1_000,
                    "only {skipped} of 60,000 lattice points sit outside the enclosing cap, so the saturation skip is untested"
                );
            }
        }
    }

    /// A cap that did not contain its shape would let the saturation skip drop
    /// a live region, so the containment is checked directly rather than
    /// argued: every polygon vertex and every edge midpoint is inside it.
    #[test]
    fn the_enclosing_cap_actually_encloses_the_polygon() {
        let shape = corridor_polygon();
        let prepared = shape.prepare();
        let (centre, radius) = prepared.enclosing_cap().expect("a corridor has a cap");
        assert!(radius < std::f64::consts::FRAC_PI_2, "cap radius {radius}");
        let PreparedShape::Polygon { ring, edges, .. } = &prepared else {
            panic!("a polygon prepared into something else");
        };
        for (k, &v) in ring.iter().enumerate() {
            assert!(arc(centre, v) <= radius, "vertex {k} is outside the cap");
        }
        for (k, e) in edges.iter().enumerate() {
            assert!(
                arc(centre, e.mid) <= radius,
                "edge {k} midpoint is outside the cap"
            );
        }
    }

    // The footprint tests moved with the model, to
    // `crate::mesh::footprint`, where they are pinned PER CARD: both
    // measured parts' anchors, the null-launch derivation of the fixed
    // term, and the refusal for a part nobody has run.

    #[test]
    fn the_cell_count_model_predicts_the_published_uniform_mesh() {
        // The published uniform mesh's own delivered plateau spacing is 119.90
        // km across flats and it carries 40,962 cells. Predicting that count
        // from the spacing alone is the instrument check in the passing
        // direction.
        let spec = MeshSpec::uniform(119.90);
        let n = spec.predicted_cells(200_000);
        let err = (n / 40_962.0 - 1.0).abs();
        assert!(
            err < 0.01,
            "predicted {n:.0} cells against an actual 40,962, off by {:.2}%",
            err * 100.0
        );
    }

    #[test]
    fn the_cell_count_model_moves_the_right_way_when_the_spec_does() {
        // The failing direction: halving the spacing must quadruple the count.
        let coarse = MeshSpec::uniform(120.0).predicted_cells(100_000);
        let fine = MeshSpec::uniform(60.0).predicted_cells(100_000);
        let ratio = fine / coarse;
        assert!(
            (ratio - 4.0).abs() < 0.01,
            "halving the spacing changed the count by {ratio:.4}x, not 4x"
        );
    }

    #[test]
    fn fitting_to_a_card_hits_the_target() {
        let spec = MeshSpec {
            background_km: 120.0,
            regions: vec![Region {
                shape: Shape::Cap {
                    center_deg: [39.0, -98.0],
                    radius_km: 1200.0,
                },
                spacing_km: 20.0,
                transition: TransitionField::Km(900.0),
            }],
            name: None,
        };
        let target = crate::mesh::footprint::card("rtx-5070-ti")
            .unwrap()
            .cells_that_fit(10.0 * 1024.0)
            .unwrap();
        let (fitted, k) = spec.fitted_to(target, 100_000).unwrap();
        let got = fitted.predicted_cells(100_000);
        assert!(
            (got / target as f64 - 1.0).abs() < 1e-3,
            "fitted spec predicts {got:.0} against a target of {target}"
        );
        // the SHAPE of the request survives the rescale
        let want_ratio = spec.background_km / spec.regions[0].spacing_km;
        let got_ratio = fitted.background_km / fitted.regions[0].spacing_km;
        assert!((want_ratio - got_ratio).abs() < 1e-12, "the refinement ratio moved");
        assert!(k > 0.0);
    }

    /// The instrument that tells a caller the request will NOT be met.
    ///
    /// Validated in BOTH directions in one test, because a reporter that
    /// always says "met" and a reporter that always says "missed" are
    /// equally useless and look identical from one arm.
    #[test]
    fn attainment_reports_the_spacing_a_region_actually_reaches() {
        let region = |radius_km: f64, transition_km: f64| MeshSpec {
            background_km: 200.0,
            regions: vec![Region {
                shape: Shape::Cap {
                    center_deg: [37.0, -97.0],
                    radius_km,
                },
                spacing_km: 15.0,
                transition: TransitionField::Km(transition_km),
            }],
            name: None,
        };

        // MISSED: a 700 km cap under a 2,800 km ramp. The tanh is centred on
        // the boundary, so the centre sits 0.25 ramp widths in and the field
        // never gets near 15 km.
        let missed = region(700.0, 2800.0).region_attainment(50_000);
        assert_eq!(missed.len(), 1);
        assert!(
            missed[0].attained_over_requested > 1.5,
            "a ramp four times wider than the region reads {:.3}x, which \
             would let a door promise 15 km and hand back 23",
            missed[0].attained_over_requested
        );
        assert!((missed[0].interior_depth_km - 700.0).abs() < 1.0);
        let widest = missed[0]
            .widest_transition_km
            .expect("a 700 km cap can be served by SOME ramp");
        assert!(
            widest < 2800.0,
            "the remedy is not narrower than the ramp that failed"
        );

        // MET: taking the remedy the same call reported.
        let met = region(700.0, widest).region_attainment(50_000);
        assert!(
            met[0].attained_over_requested <= 1.0 + met[0].tolerance,
            "the reported remedy does not deliver the request: {:.4}x",
            met[0].attained_over_requested
        );

        // MET the other way: a region wide enough for the ramp it asked for.
        let wide = region(3000.0, 400.0).region_attainment(50_000);
        assert!(
            (wide[0].attained_spacing_km - 15.0).abs() < 0.05,
            "a cap fifteen ramp widths across reads {} km, not 15",
            wide[0].attained_spacing_km
        );
    }

    #[test]
    fn a_cap_refines_inside_and_relaxes_outside() {
        let spec = MeshSpec {
            background_km: 120.0,
            regions: vec![Region {
                shape: Shape::Cap {
                    center_deg: [39.0, -98.0],
                    radius_km: 1000.0,
                },
                spacing_km: 15.0,
                transition: TransitionField::Km(200.0),
            }],
            name: None,
        };
        let centre = from_lat_lon(39f64.to_radians(), (-98f64).to_radians());
        let far = from_lat_lon((-39f64).to_radians(), 82f64.to_radians());
        // The ramp is a tanh centred ON the boundary, so the requested spacing
        // is reached asymptotically inside, not at the edge. A cap five
        // transition widths across reads its request to 1 part in 1000; a
        // narrower one reads coarser, which is a property of the ramp and not a
        // defect -- the receipt reports the DELIVERED spacing for this reason.
        assert!(
            (spec.spacing_m(centre) / 15_000.0 - 1.0).abs() < 1e-3,
            "the cap centre reads {} m, not 15 km",
            spec.spacing_m(centre)
        );
        assert!(
            (spec.spacing_m(far) / 120_000.0 - 1.0).abs() < 1e-3,
            "the far side reads {} m, not 120 km",
            spec.spacing_m(far)
        );
        assert!((spec.density(centre) - 1.0).abs() < 1e-3);
        assert!(spec.density(far) < 1e-3);
    }

    #[test]
    fn a_lat_lon_box_is_inside_where_it_says_it_is() {
        let shape = Shape::LatLonBox {
            lat_deg: [30.0, 45.0],
            lon_deg: [-105.0, -90.0],
        };
        let inside = from_lat_lon(37f64.to_radians(), (-97f64).to_radians());
        let outside = from_lat_lon(37f64.to_radians(), (-60f64).to_radians());
        assert!(shape.signed_distance(inside) < 0.0, "the box centre reads outside");
        assert!(shape.signed_distance(outside) > 0.0, "a point 30 deg east reads inside");
        // and it wraps through the antimeridian
        let wrapped = Shape::LatLonBox {
            lat_deg: [-10.0, 10.0],
            lon_deg: [170.0, -170.0],
        };
        assert!(
            wrapped.signed_distance(from_lat_lon(0.0, 180f64.to_radians())) < 0.0,
            "a box straddling the antimeridian excludes the antimeridian"
        );
        assert!(wrapped.signed_distance(from_lat_lon(0.0, 0.0)) > 0.0);
    }

    #[test]
    fn a_polygon_is_inside_where_it_says_it_is() {
        let shape = Shape::Polygon {
            vertices_deg: vec![[30.0, -105.0], [30.0, -90.0], [45.0, -90.0], [45.0, -105.0]],
        };
        assert!(shape.signed_distance(from_lat_lon(37f64.to_radians(), (-97f64).to_radians())) < 0.0);
        assert!(shape.signed_distance(from_lat_lon(37f64.to_radians(), (-60f64).to_radians())) > 0.0);
        assert!(shape.signed_distance(from_lat_lon((-37f64).to_radians(), (-97f64).to_radians())) > 0.0);
    }

    /// A square whose inscribed circle has the given half-width, about a
    /// centre -- the same figure the swath layer's own probe builds.
    fn square_about(centre_deg: [f64; 2], half_width_km: f64) -> Shape {
        let c = from_lat_lon(centre_deg[0].to_radians(), centre_deg[1].to_radians());
        let (east, north) = crate::mesh::geom::east_north(c).expect("frame");
        let r = half_width_km * std::f64::consts::SQRT_2 * 1000.0 / EARTH_RADIUS_M;
        let mut vertices_deg = Vec::new();
        for bearing in [45.0f64, 135.0, 225.0, 315.0] {
            let b = bearing.to_radians();
            let dir = crate::mesh::geom::add(
                crate::mesh::geom::scale(north, b.cos()),
                crate::mesh::geom::scale(east, b.sin()),
            );
            let p = unit(crate::mesh::geom::add(
                crate::mesh::geom::scale(c, r.cos()),
                crate::mesh::geom::scale(unit(dir).expect("dir"), r.sin()),
            ))
            .expect("vertex");
            let (lat, lon) = crate::mesh::geom::lat_lon(p);
            vertices_deg.push([lat.to_degrees(), lon.to_degrees()]);
        }
        Shape::Polygon { vertices_deg }
    }

    /// LEDGER #367. The winding magnitude cannot tell "inside the ring" from
    /// "the antipode is inside the ring", so every polygon region used to
    /// refine a ghost copy of itself on the far side of the globe.
    #[test]
    fn a_polygon_does_not_contain_its_own_antipode() {
        let centre = [0.2f64, 0.2];
        let shape = square_about(centre, 150.0);
        let inside = from_lat_lon(centre[0].to_radians(), centre[1].to_radians());
        assert!(
            shape.signed_distance(inside) < 0.0,
            "the polygon does not contain its own centre"
        );
        let antipode = [-inside[0], -inside[1], -inside[2]];
        let d = shape.signed_distance(antipode) * EARTH_RADIUS_M / 1000.0;
        assert!(
            d > 0.0,
            "the polygon claims its own ANTIPODE is inside it, {d:.1} km deep"
        );
        // And it is not marginally outside: the antipode is half a world away.
        assert!(d > 19_000.0, "antipodal distance reads {d:.1} km");
    }

    /// `Shape::Polygon` promises "wound either way". It has to stay true, and
    /// it is the reason the SIGN of the winding cannot be the interior test.
    #[test]
    fn polygon_containment_is_the_same_for_both_windings() {
        let centre = [0.2f64, 0.2];
        let Shape::Polygon { vertices_deg } = square_about(centre, 150.0) else {
            unreachable!()
        };
        let mut reversed = vertices_deg.clone();
        reversed.reverse();
        let forward = Shape::Polygon { vertices_deg };
        let backward = Shape::Polygon {
            vertices_deg: reversed,
        };
        for p in fibonacci_lattice(4_000) {
            assert_eq!(
                forward.signed_distance(p) < 0.0,
                backward.signed_distance(p) < 0.0,
                "the two windings disagree about a point"
            );
        }
    }

    /// The reporting half of #367: `--dry-run`'s `region_attainment` read the
    /// ghost's depth, so it printed the request "exactly met" at every size
    /// with an interior depth near half the Earth's circumference.
    #[test]
    fn a_polygon_reports_its_own_interior_depth_not_the_antipodes() {
        let centre = [0.2f64, 0.2];
        for half_width in [22.0f64, 60.0, 150.0, 300.0] {
            let spec = MeshSpec {
                background_km: 75.0,
                regions: vec![Region {
                    shape: square_about(centre, half_width),
                    spacing_km: 4.0,
                    transition: TransitionField::Km(100.0),
                }],
                name: None,
            };
            let row = &spec.region_attainment(20_000)[0];
            assert!(
                row.interior_depth_km < 1.5 * half_width,
                "a {half_width} km half-width polygon reports {:.1} km of interior depth",
                row.interior_depth_km
            );
            // A region narrower than its own ramp cannot reach its request,
            // and the report has to say so instead of printing the request
            // back. The cap arm of the same sweep reads 6.358 km at 22 km.
            if half_width <= 60.0 {
                assert!(
                    row.attained_spacing_km > 4.05,
                    "a {half_width} km half-width polygon claims it reaches its 4 km request exactly ({:.4} km)",
                    row.attained_spacing_km
                );
            }
        }
    }

    /// The generation half of #367, and the one that made it blocking. The
    /// ghost region's boundary is a STEP: signed distance jumps from
    /// -19,900 km to +19,900 km across it, both ends of a saturated `tanh`, so
    /// the spacing field falls from the region's spacing to the background in
    /// one cell. The generator's gradient gate read exactly
    /// `background/spacing - 1` -- 1775 %/cell at 4 km in 75 km, measured on
    /// the swath layer's own emitted spec -- and refused every fine spacing,
    /// while the equivalent cap built.
    #[test]
    fn a_polygon_ramp_is_a_ramp_and_not_a_step() {
        let centre = [0.2f64, 0.2];
        let polygon = MeshSpec {
            background_km: 75.0,
            regions: vec![Region {
                shape: square_about(centre, 300.0),
                spacing_km: 4.0,
                transition: TransitionField::Cells(81.0),
            }],
            name: None,
        };
        let cap = MeshSpec {
            background_km: 75.0,
            regions: vec![Region {
                shape: Shape::Cap {
                    center_deg: centre,
                    radius_km: 300.0,
                },
                spacing_km: 4.0,
                transition: TransitionField::Cells(81.0),
            }],
            name: None,
        };
        let g_polygon = polygon.steepest_gradient_reading(50_000).per_cell;
        let g_cap = cap.steepest_gradient_reading(50_000).per_cell;
        let step = 75.0 / 4.0 - 1.0; // what a field with no ramp reads
        assert!(
            g_polygon < 0.5 * step,
            "the polygon field reads {:.3} per cell against a no-ramp step of {step:.3}",
            g_polygon
        );
        assert!(
            g_polygon < 4.0 * g_cap.max(1e-6),
            "the polygon field is {:.3} per cell against the equivalent cap's {:.3}",
            g_polygon,
            g_cap
        );
    }

    /// The sizing half: a ghost region is a second refined patch, and it shows
    /// up as cells nobody asked for.
    #[test]
    fn a_polygon_does_not_size_for_a_second_refined_region() {
        let centre = [0.2f64, 0.2];
        let polygon = MeshSpec {
            background_km: 75.0,
            regions: vec![Region {
                shape: square_about(centre, 600.0),
                spacing_km: 4.0,
                transition: TransitionField::Km(100.0),
            }],
            name: None,
        };
        let cap = MeshSpec {
            background_km: 75.0,
            regions: vec![Region {
                shape: Shape::Cap {
                    center_deg: centre,
                    radius_km: 600.0,
                },
                spacing_km: 4.0,
                transition: TransitionField::Km(100.0),
            }],
            name: None,
        };
        let n_polygon = polygon.predicted_cells(200_000);
        let n_cap = cap.predicted_cells(200_000);
        // A square of half-width 600 km inscribes that cap, so it asks for a
        // few percent MORE cells -- not the 1.70x the ghost cost.
        let ratio = n_polygon / n_cap;
        assert!(
            (1.0..1.35).contains(&ratio),
            "polygon predicts {n_polygon:.0} cells against the equivalent cap's {n_cap:.0} ({ratio:.3}x)"
        );
    }

    #[test]
    fn a_spec_round_trips_through_json_as_data() {
        let text = r#"{
            "name": "two regions",
            "background_km": 120.0,
            "regions": [
              { "shape": { "kind": "cap", "center_deg": [39.0, -98.0], "radius_km": 1200 },
                "spacing_km": 20.0, "transition_km": 900.0 },
              { "shape": { "kind": "lat_lon_box", "lat_deg": [30, 45], "lon_deg": [-105, -90] },
                "spacing_km": 12.0, "transition_cells": 30 }
            ]
        }"#;
        let spec = MeshSpec::from_json(text).unwrap();
        assert_eq!(spec.regions.len(), 2);
        assert!((spec.finest_km() - 12.0).abs() < 1e-12);
        let back = serde_json::to_string(&spec).unwrap();
        let again = MeshSpec::from_json(&back).unwrap();
        assert!((again.finest_km() - 12.0).abs() < 1e-12);
        assert_eq!(again.regions.len(), 2);
    }

    #[test]
    fn a_spec_that_cannot_describe_a_mesh_is_refused_by_name() {
        let err = MeshSpec::from_json(r#"{"background_km": 0.0}"#).unwrap_err().to_string();
        assert!(err.contains("background spacing"), "{err}");

        let err = MeshSpec::from_json(
            r#"{"background_km": 120, "regions":[{"shape":{"kind":"cap","center_deg":[0,0],"radius_km":100},"spacing_km":-3,"transition_km":50}]}"#,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("non-positive spacing"), "{err}");

        let err = MeshSpec::from_json(
            r#"{"background_km": 120, "regions":[{"shape":{"kind":"cap","center_deg":[0,0],"radius_km":100},"spacing_km":30,"transition_km":0}]}"#,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("zero-width ramp"), "{err}");

        let err = MeshSpec::from_json(
            r#"{"background_km": 120, "regions":[{"shape":{"kind":"polygon","vertices_deg":[[0,0],[1,1]]},"spacing_km":30,"transition_km":50}]}"#,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("enclose no area"), "{err}");

        let err = MeshSpec::from_json("not json at all").unwrap_err().to_string();
        assert!(err.contains("not valid JSON"), "{err}");
    }

    #[test]
    fn the_level_clamp_floors_the_spacing_and_leaves_the_spec_untouched() {
        let spec = MeshSpec {
            background_km: 60.0,
            regions: vec![Region {
                shape: Shape::Cap {
                    center_deg: [39.0, -98.0],
                    radius_km: 1500.0,
                },
                spacing_km: 15.0,
                transition: TransitionField::Km(3000.0),
            }],
            name: None,
        };
        let centre = from_lat_lon(39f64.to_radians(), (-98f64).to_radians());
        let far = from_lat_lon((-39f64).to_radians(), 82f64.to_radians());

        // Level 1 of the 15 -> 60 ladder: h_1 = 30 km.
        let prepared = spec.prepared();
        let clamp = prepared.clamped(30_000.0);
        // Inside the fine cap the clamp floors at the level spacing...
        assert!(
            (DensityField::spacing_m(&clamp, centre) - 30_000.0).abs() < 1e-9,
            "the clamp does not floor the cap centre at h_l"
        );
        // ...and its density normalises to 1.0 there, so Lloyd relaxes toward
        // the LEVEL's equilibrium, not the final one.
        assert!((DensityField::density(&clamp, centre) - 1.0).abs() < 1e-12);
        // Far away the clamp is inert: the field is the spec's own.
        assert!(
            (DensityField::spacing_m(&clamp, far) - spec.spacing_m(far)).abs() < 1e-9,
            "the clamp changed the background"
        );
        // Level 0 (h_0 = background) is the uniform field IDENTICALLY --
        // the whole reason level 0 is the shipped uniform arm byte for byte.
        let level0 = prepared.clamped(spec.background_km * 1000.0);
        for p in fibonacci_lattice(2_000) {
            assert_eq!(
                DensityField::spacing_m(&level0, p),
                spec.background_km * 1000.0,
                "level 0 is not the uniform background field"
            );
        }
        // The spec itself is byte-identical through serialisation: the clamp
        // is a borrow-only view and cannot reach rw_mesh_spec_json or a
        // receipt. (The type has no Serialize impl -- this asserts the spec's
        // own JSON is unchanged by having been viewed through a clamp.)
        let before = serde_json::to_string(&spec).unwrap();
        let _ = DensityField::density(&clamp, centre);
        assert_eq!(before, serde_json::to_string(&spec).unwrap());
    }

    #[test]
    fn the_gradient_meter_reads_zero_on_a_uniform_spec_and_more_on_a_steep_one() {
        let flat = MeshSpec::uniform(120.0).steepest_gradient_reading(20_000);
        assert!(
            flat.per_cell < 1e-12,
            "a uniform spec has a gradient of {}",
            flat.per_cell
        );
        // A zero reading is only meaningful with a coverage word beside it,
        // and a spec with no regions has nothing to cover.
        assert!(flat.coverage.is_complete());
        assert_eq!(flat.probe_points, 0);
        assert!(flat.saw_the_refinement());
        let gentle = MeshSpec {
            background_km: 120.0,
            regions: vec![Region {
                shape: Shape::Cap {
                    center_deg: [39.0, -98.0],
                    radius_km: 2100.0,
                },
                spacing_km: 30.0,
                transition: TransitionField::Km(1200.0),
            }],
            name: None,
        };
        let steep = MeshSpec {
            background_km: 120.0,
            regions: vec![Region {
                shape: Shape::Cap {
                    center_deg: [39.0, -98.0],
                    radius_km: 2100.0,
                },
                spacing_km: 30.0,
                transition: TransitionField::Km(200.0),
            }],
            name: None,
        };
        let g = gentle.steepest_gradient_reading(50_000).per_cell;
        let s = steep.steepest_gradient_reading(50_000).per_cell;
        assert!(s > 3.0 * g, "a 6x narrower ramp read {s:.4} against {g:.4}");
    }

    // ---------------------------------------------------------------------
    // WHERE THE INSTRUMENT LOOKS
    //
    // Every assertion below fails on the retired sampler, which took its
    // probes from a lattice uniform over the whole sphere. The retired body is
    // kept verbatim as `lattice_only_gradient` so the two can be compared
    // rather than argued about, exactly as `arc_segment_distance` is kept as
    // the oracle for the prepared signed distance.
    // ---------------------------------------------------------------------

    /// THE RETIRED SAMPLER, copied verbatim from the body
    /// `steepest_gradient_per_cell_of` carried before the probe set became the
    /// field's own. It exists only here, only as an oracle, and it is what
    /// makes "the corrected reading can never be lower" a measurement rather
    /// than a claim.
    fn lattice_only_gradient(field: &impl DensityField, samples: usize) -> f64 {
        let mut worst = 0.0f64;
        for p in fibonacci_lattice(samples) {
            let h = field.spacing_m(p);
            let step = h / EARTH_RADIUS_M;
            let (east, north) = match crate::mesh::geom::east_north(p) {
                Some(v) => v,
                None => continue,
            };
            for dir in [east, north] {
                for sign in [1.0f64, -1.0] {
                    let q = unit(crate::mesh::geom::add(
                        p,
                        crate::mesh::geom::scale(dir, sign * step),
                    ));
                    if let Some(q) = q {
                        let hq = field.spacing_m(q);
                        worst = worst.max((hq / h - 1.0).abs());
                    }
                }
            }
        }
        worst
    }

    /// One cap region, spelled once for the tests that vary it.
    fn one_cap(background_km: f64, at: [f64; 2], radius_km: f64, spacing_km: f64, w_km: f64) -> MeshSpec {
        MeshSpec {
            background_km,
            regions: vec![Region {
                shape: Shape::Cap {
                    center_deg: at,
                    radius_km,
                },
                spacing_km,
                transition: TransitionField::Km(w_km),
            }],
            name: None,
        }
    }

    /// The closed-form peak per-cell gradient of ONE isolated ramp:
    /// `|dh/dx|` peaks at `h = (h_bg + h_i)/2` with the value
    /// `L (h_bg - h_i) / (2 W)`. Derived, not fitted: with `u = 1/h` linear in
    /// `phi = (1 - tanh(s/W))/2`, `|dh/dx| = 2 (a-b) h^2 phi (1-phi) / W`,
    /// whose maximum over `phi` sits at `phi = 1/(1+rho)`. `L` is the shape's
    /// `sup|grad s|`, which is exactly 1 for a cap (arc distance) and for a
    /// polygon (a corner FANS the level sets, it does not compress them).
    fn isolated_peak_per_cell(background_km: f64, spacing_km: f64, w_km: f64) -> f64 {
        (background_km - spacing_km) / (2.0 * w_km)
    }

    /// REQUIREMENT: the instrument must find a transition narrower than the
    /// global lattice's own point spacing.
    ///
    /// 50,000 Fibonacci points sit 101 km apart. This ramp is 3.253 km wide
    /// around a 10 km cap -- the CPAS-class innermost rung -- so the retired
    /// sampler stepped over it entirely and read the flat background it landed
    /// on. MEASURED: 0.0000 %/cell from the lattice alone, which the ladder's
    /// band arithmetic turns into `f64::INFINITY` cells and admits, against a
    /// closed-form peak of 1177.4 %/cell and a band of a fifth of a cell.
    #[test]
    fn a_transition_narrower_than_the_global_lattice_is_still_measured() {
        let spec = one_cap(76.8, [-60.0, 0.0], 10.0, 0.2, 3.253);
        let truth = isolated_peak_per_cell(76.8, 0.2, 3.253);
        let blind = lattice_only_gradient(&spec.prepared(), 50_000);
        let reading = spec.steepest_gradient_reading(50_000);
        assert!(
            blind < 1e-4 * truth,
            "the global lattice was supposed to be blind here and read {blind:.6} against a peak of {truth:.4}"
        );
        // WHAT IS STILL MISSING, stated as a number rather than left to be
        // rediscovered. The measure is a MEAN of |dh/dx| over a step of length
        // h(p), so it is bounded above by the continuum peak and reads about
        // three quarters of it here. That bias is one-sided -- it can admit,
        // it can never wrongly refuse -- it does not depend on where the
        // probes are, and this change does not address it.
        assert!(
            reading.per_cell > 0.5 * truth,
            "the corrected reading is {:.4} against a closed-form peak of {truth:.4}",
            reading.per_cell
        );
        assert!(
            reading.per_cell <= truth,
            "a secant over a step of length h cannot exceed the continuum peak: {:.4} against {truth:.4}",
            reading.per_cell
        );
        // The operational statement: a band this narrow is what the surgery
        // locality gate exists to refuse, and it now gets the chance to.
        let band = (2f64).ln() / (1.0 + reading.per_cell).ln();
        assert!(
            band < 1.0,
            "a 384:1 refinement over a 3.253 km ramp reports a band of {band:.3} cells"
        );
        assert!(reading.coverage.is_complete());
        assert!(reading.saw_the_refinement());
    }

    /// The same request shrunk is GEOMETRICALLY SIMILAR, so its steepest
    /// per-cell spacing change is a dimensionless constant and the instrument
    /// must return the same number at every size. The retired sampler read
    /// 9.23 %/cell at k = 1 and exactly 0.0000 %/cell at k = 1/128 -- a
    /// receipt printing "this is a uniform mesh" for a 16:1 refinement.
    #[test]
    fn the_gradient_meter_is_scale_invariant() {
        let truth = isolated_peak_per_cell(76.8, 4.8, 388.8);
        let mut readings = Vec::new();
        for k in [1.0f64, 1.0 / 8.0, 1.0 / 128.0, 1.0 / 2048.0, 1.0 / 16384.0] {
            let spec = one_cap(76.8 * k, [21.0, 137.0], 307.2 * k, 4.8 * k, 388.8 * k);
            let r = spec.steepest_gradient_reading(50_000);
            assert!(r.coverage.is_complete());
            readings.push(r.per_cell);
        }
        let lo = readings.iter().cloned().fold(f64::INFINITY, f64::min);
        let hi = readings.iter().cloned().fold(0.0f64, f64::max);
        assert!(
            hi / lo < 1.001,
            "one request read {lo:.6} at its smallest size and {hi:.6} at its largest: {readings:?}"
        );
        assert!(
            lo > 0.99 * truth && hi <= truth,
            "the scale-free reading is {lo:.6}..{hi:.6} against a closed-form {truth:.6}"
        );
    }

    /// A gate whose answer depends on the longitude of the storm is not a
    /// function of the spec at all. The retired sampler read one request
    /// anywhere between 0.66 %/cell and 325 %/cell depending only on where the
    /// cap was placed, and admitted it at some placements and refused it at
    /// others.
    #[test]
    fn the_gradient_meter_does_not_depend_on_where_on_earth_the_region_sits() {
        let mut lo = f64::INFINITY;
        let mut hi = 0.0f64;
        let mut blind_lo = f64::INFINITY;
        let mut blind_hi = 0.0f64;
        for k in 0..24 {
            let lat = -80.0 + 160.0 * (k as f64 / 23.0);
            let lon = -180.0 + 359.0 * ((k * 7 % 24) as f64 / 24.0);
            let spec = one_cap(51.2, [lat, lon], 10.0, 0.2, 6.685);
            let g = spec.steepest_gradient_reading(50_000).per_cell;
            lo = lo.min(g);
            hi = hi.max(g);
            let b = lattice_only_gradient(&spec.prepared(), 50_000);
            blind_lo = blind_lo.min(b);
            blind_hi = blind_hi.max(b);
        }
        assert!(
            hi / lo < 1.001,
            "one request read {lo:.4}..{hi:.4} over 24 placements"
        );
        // And say plainly what it used to do, so the guard is not mistaken for
        // a tautology if the sampler is ever changed back.
        assert!(
            blind_hi / blind_lo.max(1e-12) > 10.0,
            "the global lattice was supposed to be placement-dependent here: {blind_lo:.4}..{blind_hi:.4}"
        );
    }

    /// NON-LOOSENING, BY ARITHMETIC AND BY MEASUREMENT. The reading is a `max`
    /// over a probe set that is a strict SUPERSET of the retired one -- the
    /// global lattice is kept in full and the region-targeted points are
    /// unioned on top -- so the corrected reading is >= the old one for every
    /// field, unconditionally. No spec that is refused today can pass.
    ///
    /// The union is arithmetic and not tidiness: on real specs the
    /// region-targeted set ALONE sometimes reads a shade lower than the
    /// lattice, because a lattice point landed on a rung crossover where no
    /// region boundary is. Replacing rather than unioning would loosen the
    /// gate on exactly those.
    #[test]
    fn the_corrected_reading_never_falls_below_the_lattice_alone() {
        for spec in fixture_corpus() {
            let prepared = spec.prepared();
            let old = lattice_only_gradient(&prepared, 50_000);
            let new = steepest_gradient_reading_of(&prepared, 50_000).per_cell;
            assert!(
                new >= old,
                "{:?}: the corrected reading {new:.6} is below the lattice-only {old:.6}",
                spec.name
            );
        }
    }

    /// THE ANTI-DRIFT GUARD. A sampled maximum is a mean of `|dh/dx|` over a
    /// step of length `h(p)`, so it can never exceed the field's continuum
    /// supremum; and `h` is a MIN of Lipschitz branches, so that supremum is
    /// bounded by the steepest single branch,
    /// `max_i L_i (h_bg - h_i) / (2 W_i)`. If a probe generator ever starts
    /// reporting above this, either the generator or the field it claims to
    /// describe has changed, and this is where that shows up.
    ///
    /// Caps and polygons only: `L = 1` for both, exactly. A `LatLonBox`'s
    /// signed distance is NOT a distance -- it scales the longitude term by
    /// the cosine of the QUERY point's latitude, so `|grad s|` exceeds 1 and
    /// the box's delivered ramp is narrower than the `transition_km` its
    /// author wrote. That is a defect in the FIELD, not in this instrument,
    /// and deriving the box's Lipschitz constant is not this guard's job.
    #[test]
    fn the_reading_never_exceeds_the_steepest_single_branch() {
        for spec in fixture_corpus() {
            if spec
                .regions
                .iter()
                .any(|r| matches!(r.shape, Shape::LatLonBox { .. }))
            {
                continue;
            }
            let bound = spec
                .regions
                .iter()
                .map(|r| {
                    let w_km = r.transition.width_rad(r.spacing_km) * EARTH_RADIUS_M / 1000.0;
                    isolated_peak_per_cell(spec.background_km, r.spacing_km, w_km)
                })
                .fold(0.0f64, f64::max);
            let read = spec.steepest_gradient_reading(50_000).per_cell;
            assert!(
                read <= bound * (1.0 + 1e-9),
                "{:?} read {read:.6} against a closed-form ceiling of {bound:.6}",
                spec.name
            );
        }
    }

    /// The receipt carries `gradient_probe_points`, so two runs of one spec
    /// must agree exactly or the receipt stops being reproducible and the
    /// grid registry's digests stop being stable. No RNG, fixed azimuths,
    /// fixed traversal order.
    #[test]
    fn the_gradient_meter_is_deterministic() {
        for spec in fixture_corpus() {
            let a = spec.steepest_gradient_reading(50_000);
            let b = spec.steepest_gradient_reading(50_000);
            assert_eq!(a.per_cell.to_bits(), b.per_cell.to_bits(), "{:?}", spec.name);
            assert_eq!(a.probe_points, b.probe_points, "{:?}", spec.name);
        }
    }

    /// THE COVERING CONTRACT, and it names no shape.
    ///
    /// Draw an independent point cloud FROM THE SHAPE'S OWN BOUND -- never
    /// from a globe lattice, which on a thin shell would land almost nothing
    /// inside it and reproduce, inside the guard, the very defect the guard
    /// exists for. Keep the cloud points that are in the shell, and demand
    /// each lies within one probe cell `sqrt(gn^2 + gt^2)` of a probe. A new
    /// `Shape` arm inherits this obligation instead of being trusted.
    #[test]
    fn the_transition_shell_is_covered_for_every_shape_kind() {
        for spec in covering_corpus() {
            let prepared = spec.prepared();
            let mut probes: Vec<V3> = Vec::new();
            let coverage = prepared.variation_probes(PROBE_BUDGET, &mut probes);
            assert!(
                coverage.is_complete(),
                "{:?} could not be covered: {:?}",
                spec.name,
                coverage.missed()
            );
            // The same constants `variation_probes` derives, re-derived here
            // so the test is checking the contract and not echoing the code.
            let r = &prepared.regions[0];
            let ratio = r.fine_inv_m / r.back_inv_m;
            let k = ((ratio.sqrt() + 1.0) / SHELL_TAIL_FRACTION.sqrt())
                .ln()
                .clamp(SHELL_MIN_WIDTHS, SHELL_SATURATION_WIDTHS);
            let half = k * r.width_rad;
            let h_fine_rad = 1.0 / (r.fine_inv_m * EARTH_RADIUS_M);
            let gn = (r.width_rad.min(h_fine_rad) / ACROSS_STEP_DIVISOR)
                .max(r.width_rad / ACROSS_STEP_FLOOR_DIVISOR);
            let gt = r.width_rad / ALONG_STEP_DIVISOR;
            let cell = (gn * gn + gt * gt).sqrt();

            // z-sorted probes: |z_p - z_q| <= arc(p, q), so a window on z
            // prunes without ever excluding a true neighbour.
            let mut by_z = probes.clone();
            by_z.sort_by(|a, b| a[2].total_cmp(&b[2]));
            let zs: Vec<f64> = by_z.iter().map(|p| p[2]).collect();

            let mut checked = 0usize;
            let mut worst = 0.0f64;
            for q in cloud_over(&r.shape, half, 900) {
                let s = r.shape.signed_distance(q);
                if s.abs() > half {
                    continue;
                }
                checked += 1;
                let lo = zs.partition_point(|z| *z < q[2] - cell);
                let hi = zs.partition_point(|z| *z <= q[2] + cell);
                let mut best = f64::INFINITY;
                for p in &by_z[lo..hi] {
                    best = best.min(arc(*p, q));
                }
                worst = worst.max(best);
            }
            assert!(
                checked > 50,
                "{:?}: only {checked} cloud points landed in the shell, so this proved nothing",
                spec.name
            );
            assert!(
                worst <= cell,
                "{:?}: a shell point sits {:.6} rad from the nearest probe, against a covering radius of {cell:.6}",
                spec.name,
                worst
            );
        }
    }

    /// A cloud drawn uniformly over the cap that contains the shape's shell,
    /// or over the sphere when the shape declines to bound itself. It shares
    /// no arithmetic with the probe generator -- different lattice, different
    /// frame, different count -- which is what makes it a check and not an
    /// echo.
    fn cloud_over(shape: &PreparedShape, half: f64, n: usize) -> Vec<V3> {
        let ga = std::f64::consts::PI * (3.0 - 5f64.sqrt());
        let (centre, radius) = match shape.probe_bound() {
            Some((c, rb)) => (c, (rb + half).min(std::f64::consts::PI)),
            None => ([0.0, 0.0, 1.0], std::f64::consts::PI),
        };
        let axis = if centre[0].abs() < 0.9 {
            [1.0, 0.0, 0.0]
        } else {
            [0.0, 1.0, 0.0]
        };
        let e = unit(cross(axis, centre)).unwrap_or([0.0, 1.0, 0.0]);
        let f = cross(centre, e);
        let z_min = radius.cos();
        (0..n)
            .map(|k| {
                let z = 1.0 - (1.0 - z_min) * ((2 * k + 1) as f64 / (2 * n) as f64);
                let r = (1.0 - z * z).max(0.0).sqrt();
                let t = ga * k as f64 + 0.37;
                let dir = crate::mesh::geom::add(
                    crate::mesh::geom::scale(e, t.cos()),
                    crate::mesh::geom::scale(f, t.sin()),
                );
                crate::mesh::geom::add(
                    crate::mesh::geom::scale(centre, z),
                    crate::mesh::geom::scale(dir, r),
                )
            })
            .collect()
    }

    /// The specs the non-loosening, drift and determinism guards run over: one
    /// of every shape kind, a nested ladder, the request the shipped example
    /// makes, and the two sizes at which the retired sampler went blind.
    fn fixture_corpus() -> Vec<MeshSpec> {
        let named = |name: &str, spec: MeshSpec| MeshSpec {
            name: Some(name.to_string()),
            ..spec
        };
        let mut out = vec![
            named("uniform", MeshSpec::uniform(120.0)),
            named("wide-cap", one_cap(120.0, [39.0, -98.0], 1200.0, 20.0, 900.0)),
            named("narrow-cap", one_cap(76.8, [-60.0, 0.0], 10.0, 0.2, 3.253)),
            named("tiny-cap", one_cap(76.8, [12.0, 44.0], 3.0, 0.15, 1.125)),
            named(
                "cells-ramp-cap",
                MeshSpec {
                    background_km: 75.0,
                    regions: vec![Region {
                        shape: Shape::Cap {
                            center_deg: [0.2, 0.2],
                            radius_km: 300.0,
                        },
                        spacing_km: 4.0,
                        transition: TransitionField::Cells(81.0),
                    }],
                    name: None,
                },
            ),
            named(
                "box",
                MeshSpec {
                    background_km: 120.0,
                    regions: vec![Region {
                        shape: Shape::LatLonBox {
                            lat_deg: [33.0, 43.0],
                            lon_deg: [-104.0, -88.0],
                        },
                        spacing_km: 15.0,
                        transition: TransitionField::Cells(30.0),
                    }],
                    name: None,
                },
            ),
            named(
                "polygon",
                MeshSpec {
                    background_km: 75.0,
                    regions: vec![Region {
                        shape: Shape::Polygon {
                            vertices_deg: vec![
                                [36.0, -100.0],
                                [36.0, -94.0],
                                [41.0, -92.0],
                                [43.0, -98.0],
                                [39.0, -103.0],
                            ],
                        },
                        spacing_km: 4.0,
                        transition: TransitionField::Cells(81.0),
                    }],
                    name: None,
                },
            ),
        ];
        // A nested ladder: the class where the answer actually lives, and the
        // one an isolated per-region argument cannot decide on its own.
        let mut rungs = Vec::new();
        let mut h = 25.6f64;
        let mut radius = 1600.0f64;
        while h >= 0.2 {
            rungs.push(Region {
                shape: Shape::Cap {
                    center_deg: [39.0, -98.0],
                    radius_km: radius,
                },
                spacing_km: h,
                transition: TransitionField::Cells(16.4),
            });
            h *= 0.5;
            radius *= 0.5;
        }
        out.push(named(
            "nested-ladder",
            MeshSpec {
                background_km: 51.2,
                regions: rungs,
                name: None,
            },
        ));
        out
    }

    /// One region per shape kind, plus the awkward ones: a cap far wider than
    /// its shell, a box straddling the pole and every longitude, a non-convex
    /// ring, and a ring whose vertices cancel so it has no bound at all.
    fn covering_corpus() -> Vec<MeshSpec> {
        let named = |name: &str, spec: MeshSpec| MeshSpec {
            name: Some(name.to_string()),
            ..spec
        };
        vec![
            named("small-cap", one_cap(76.8, [-60.0, 0.0], 10.0, 0.2, 3.253)),
            named("big-cap", one_cap(120.0, [39.0, -98.0], 2200.0, 20.0, 400.0)),
            named(
                "mid-latitude-box",
                MeshSpec {
                    background_km: 120.0,
                    regions: vec![Region {
                        shape: Shape::LatLonBox {
                            lat_deg: [33.0, 43.0],
                            lon_deg: [-104.0, -88.0],
                        },
                        spacing_km: 15.0,
                        transition: TransitionField::Cells(30.0),
                    }],
                    name: None,
                },
            ),
            named(
                "polar-box",
                MeshSpec {
                    background_km: 240.0,
                    regions: vec![Region {
                        shape: Shape::LatLonBox {
                            lat_deg: [72.0, 89.0],
                            lon_deg: [0.0, 359.9],
                        },
                        spacing_km: 60.0,
                        transition: TransitionField::Km(600.0),
                    }],
                    name: None,
                },
            ),
            named(
                "triangle",
                MeshSpec {
                    background_km: 120.0,
                    regions: vec![Region {
                        shape: Shape::Polygon {
                            vertices_deg: vec![[10.0, 20.0], [10.0, 32.0], [20.0, 26.0]],
                        },
                        spacing_km: 20.0,
                        transition: TransitionField::Km(300.0),
                    }],
                    name: None,
                },
            ),
            named(
                "crescent",
                MeshSpec {
                    background_km: 120.0,
                    regions: vec![Region {
                        shape: Shape::Polygon {
                            vertices_deg: vec![
                                [0.0, 0.0],
                                [6.0, 3.0],
                                [12.0, 0.0],
                                [9.0, 8.0],
                                [12.0, 16.0],
                                [6.0, 13.0],
                                [0.0, 16.0],
                                [3.0, 8.0],
                            ],
                        },
                        spacing_km: 20.0,
                        transition: TransitionField::Km(300.0),
                    }],
                    name: None,
                },
            ),
            named(
                "cancelling-ring",
                MeshSpec {
                    background_km: 240.0,
                    regions: vec![Region {
                        shape: Shape::Polygon {
                            vertices_deg: vec![
                                [0.0, 0.0],
                                [0.0, 90.0],
                                [0.0, 180.0],
                                [0.0, 270.0],
                            ],
                        },
                        spacing_km: 60.0,
                        transition: TransitionField::Km(900.0),
                    }],
                    name: None,
                },
            ),
        ]
    }
}
