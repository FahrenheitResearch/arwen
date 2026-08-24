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
    Polygon { vertices_deg: Vec<[f64; 2]> },
}

impl Shape {
    /// Signed great-circle distance from `p` to the shape's boundary, in
    /// radians: negative inside, positive outside.
    pub fn signed_distance(&self, p: V3) -> f64 {
        match self {
            Shape::Cap {
                center_deg,
                radius_km,
            } => {
                let c = from_lat_lon(center_deg[0].to_radians(), center_deg[1].to_radians());
                arc(c, p) - radius_km * 1000.0 / EARTH_RADIUS_M
            }
            Shape::LatLonBox { lat_deg, lon_deg } => {
                // Standard box distance field, evaluated in the locally scaled
                // latitude/longitude plane. Exact along the meridians and
                // accurate to the sphere's curvature over the parallels, which
                // is what a refinement ramp needs; it is not a geodesic
                // distance to a rhumb boundary and is not used as one.
                let (lat, lon) = lat_lon(p);
                let (lat0, lat1) = (lat_deg[0].to_radians(), lat_deg[1].to_radians());
                let (lat0, lat1) = (lat0.min(lat1), lat0.max(lat1));
                let d_lat = (lat0 - lat).max(lat - lat1);
                let d_lon = lon_interval_distance(lon, lon_deg[0].to_radians(), lon_deg[1].to_radians())
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

/// Spherical winding test: the signed angles a closed ring subtends at `p` sum
/// to +/-2*pi inside and 0 outside.
fn polygon_contains(ring: &[V3], p: V3) -> bool {
    let mut total = 0.0f64;
    for k in 0..ring.len() {
        let a = crate::mesh::geom::tangent_at(p, sub(ring[k], p));
        let b = crate::mesh::geom::tangent_at(p, sub(ring[(k + 1) % ring.len()], p));
        let (Some(a), Some(b)) = (unit(a), unit(b)) else {
            return true; // p is a vertex
        };
        total += dot(cross(a, b), p).atan2(dot(a, b));
    }
    total.abs() > std::f64::consts::PI
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
    pub fn spacing_m(&self, p: V3) -> f64 {
        let mut inv = 1.0 / (self.background_km * 1000.0);
        for r in &self.regions {
            let s = r.shape.signed_distance(p);
            let w = r.transition.width_rad(r.spacing_km);
            let fine = 1.0 / (r.spacing_km * 1000.0);
            let back = 1.0 / (self.background_km * 1000.0);
            let blend = 0.5 * (1.0 - (s / w).tanh());
            let here = back + (fine - back) * blend;
            if here > inv {
                inv = here;
            }
        }
        1.0 / inv
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
            let mut deep = r.shape.interior_point();
            let mut depth = r.shape.signed_distance(deep);
            for &p in &lattice {
                let s = r.shape.signed_distance(p);
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
    pub fn density(&self, p: V3) -> f64 {
        let ratio = self.finest_km() * 1000.0 / self.spacing_m(p);
        ratio.powi(4)
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
        let mut acc = 0.0f64;
        for p in fibonacci_lattice(samples) {
            let h = self.spacing_m(p);
            acc += 1.0 / (h * h);
        }
        let mean_inv_h2 = acc / samples as f64;
        let sphere_area = 4.0 * std::f64::consts::PI * EARTH_RADIUS_M * EARTH_RADIUS_M;
        sphere_area * mean_inv_h2 / (3f64.sqrt() / 2.0)
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

    /// The steepest per-cell spacing change the spec asks for, as a fraction.
    /// The published variable mesh runs 1.53% per cell; a receipt quotes this
    /// beside that number so a steeper request is a stated fact rather than a
    /// surprise in the output.
    pub fn steepest_gradient_per_cell(&self, samples: usize) -> f64 {
        let mut worst = 0.0f64;
        for p in fibonacci_lattice(samples) {
            let h = self.spacing_m(p);
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
                        let hq = self.spacing_m(q);
                        worst = worst.max((hq / h - 1.0).abs());
                    }
                }
            }
        }
        worst
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
    fn the_gradient_meter_reads_zero_on_a_uniform_spec_and_more_on_a_steep_one() {
        let flat = MeshSpec::uniform(120.0).steepest_gradient_per_cell(20_000);
        assert!(flat < 1e-12, "a uniform spec has a gradient of {flat}");
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
        let g = gentle.steepest_gradient_per_cell(50_000);
        let s = steep.steepest_gradient_per_cell(50_000);
        assert!(s > 3.0 * g, "a 6x narrower ramp read {s:.4} against {g:.4}");
    }
}
