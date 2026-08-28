//! Spherical geometry primitives for mesh generation.
//!
//! Everything here works on the UNIT SPHERE, because that is what an MPAS grid
//! file carries: `sphere_radius = 1.0`, and `areaCell`, `dcEdge`, `dvEdge` and
//! `nominalMinDc` are all unit-sphere quantities despite their `units` attributes
//! reading `"m"` and `"m^2"`. Multiply by [`EARTH_RADIUS_M`] to get metres.
//! Computing a spacing without that normalisation prints 0.0 and has already
//! cost this program two attempts.

/// MPAS-A's Earth radius. The same constant `crate::window` uses for the render
/// bridge, repeated here so a mesh length can be quoted in metres.
pub const EARTH_RADIUS_M: f64 = 6_371_229.0;

/// A point or direction in R^3.
pub type V3 = [f64; 3];

#[inline]
pub fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
pub fn cross(a: V3, b: V3) -> V3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
pub fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
pub fn add(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

#[inline]
pub fn scale(a: V3, s: f64) -> V3 {
    [a[0] * s, a[1] * s, a[2] * s]
}

#[inline]
pub fn norm(a: V3) -> f64 {
    dot(a, a).sqrt()
}

/// Radial projection back onto the unit sphere. Returns `None` for a vector too
/// short to have a direction -- the caller decides whether that is a refusal.
#[inline]
pub fn unit(a: V3) -> Option<V3> {
    let n = norm(a);
    if n > 0.0 && n.is_finite() {
        Some(scale(a, 1.0 / n))
    } else {
        None
    }
}

/// Chord distance between two unit vectors.
///
/// Point agreement is measured in chord, never in angle: `acos(dot)` bottoms out
/// at `sqrt(2*eps) = 2.1e-8 rad` and cannot resolve machine-precision agreement
/// at all, so an angle-based check silently reports its own floor as a residual.
#[inline]
pub fn chord(a: V3, b: V3) -> f64 {
    norm(sub(a, b))
}

/// Great-circle arc between two unit vectors, in radians.
///
/// `atan2(|a x b|, a.b)` rather than `acos(a.b)`: acos loses half its digits for
/// small arcs, which is exactly the range every `dcEdge` lives in. The published
/// meshes were written with the acos form, so a mesh emitted here disagrees with
/// them at the ~1e-10 relative level -- and this form is the more accurate one.
#[inline]
pub fn arc(a: V3, b: V3) -> f64 {
    norm(cross(a, b)).atan2(dot(a, b))
}

/// Latitude and longitude of a unit vector, in radians. Longitude is wrapped
/// into `[0, 2*pi)`, which is where the published files carry `lonCell`.
///
/// Latitude is `atan2(z, hypot(x, y))`, NOT `asin(z)`, and the difference is
/// a fixed defect, not taste: near a pole, `z` alone carries the latitude at
/// half precision (`sin` flattens there) while `hypot(x, y)` carries it in
/// full, so an `asin(z)` latitude written beside its own Cartesian point
/// fails to reproduce that point. MEASURED: a graded 224k-cell mesh rolled
/// an edge midpoint 7.1e-8 rad from the south pole, its emitted asin
/// latitude reconstructed the stored xyz only to 1.27e-9 on the unit
/// sphere, and the port's binary64 load gate (5e-10) refused the pair.
/// The atan2 form reconstructs to machine precision at every latitude.
#[inline]
pub fn lat_lon(v: V3) -> (f64, f64) {
    let lat = v[2].atan2(v[0].hypot(v[1]));
    let mut lon = v[1].atan2(v[0]);
    if lon < 0.0 {
        lon += std::f64::consts::TAU;
    }
    if lon >= std::f64::consts::TAU {
        lon -= std::f64::consts::TAU;
    }
    (lat, lon)
}

/// Unit vector from latitude and longitude in radians.
#[inline]
pub fn from_lat_lon(lat: f64, lon: f64) -> V3 {
    let (sla, cla) = lat.sin_cos();
    let (slo, clo) = lon.sin_cos();
    [cla * clo, cla * slo, sla]
}

/// SIGNED area of the spherical triangle `(a, b, c)` on the unit sphere.
///
/// Half-tangent (Van Oosterom & Strackee) form. The excess-of-angles form loses
/// all its digits on the small triangles a kite decomposes into; this one is
/// stable there. The result is positive when `(a, b, c)` winds
/// counter-clockwise seen from outside the sphere and negative when it does not.
///
/// The sign is not decoration. Taking `|numerator|` instead makes every fan
/// triangle add, and a Voronoi cell that is even slightly non-convex then
/// reports an area too large by twice its reflex sliver -- measured at 2.2e-02
/// relative on the published variable-resolution mesh, which then propagates
/// into every tangential weight on that cell.
/// The numerator is evaluated as `a . ((b-a) x (c-a))`, which is algebraically
/// the same determinant as `a . (b x c)` but numerically a different animal: on
/// a triangle of side 1e-3 the direct form is a difference of order-1 products
/// that cancels down to 1e-6, losing ten digits, while the differenced form
/// keeps them. Measured on the published variable-resolution mesh, that change
/// moves the kite-partition closure from 7.9e-11 to machine precision.
#[inline]
pub fn tri_area(a: V3, b: V3, c: V3) -> f64 {
    let num = dot(a, cross(sub(b, a), sub(c, a)));
    let den = 1.0 + dot(a, b) + dot(b, c) + dot(c, a);
    2.0 * num.atan2(den)
}

/// Area of the spherical polygon `ring` on the unit sphere, fanned from
/// `ring[0]` with signed sub-triangles. Positive for a ring wound
/// counter-clockwise seen from outside. Correct for any simple polygon, convex
/// or not.
pub fn polygon_area(ring: &[V3]) -> f64 {
    if ring.len() < 3 {
        return 0.0;
    }
    let mut total = 0.0;
    for k in 1..ring.len() - 1 {
        total += tri_area(ring[0], ring[k], ring[k + 1]);
    }
    total
}

/// Circumcentre of the spherical triangle `(a, b, c)`, on the unit sphere.
///
/// This is the Voronoi vertex of the three generators. It is the normalised
/// normal of the plane through the three points, signed onto the same side as
/// the triangle. A circumcentre outside its own triangle is legal and rare
/// (3 of 327,680 on the published variable-resolution mesh) and is not an error.
pub fn circumcenter(a: V3, b: V3, c: V3) -> Option<V3> {
    let n = cross(sub(b, a), sub(c, a));
    let u = unit(n)?;
    let side = add(add(a, b), c);
    if dot(u, side) < 0.0 {
        Some(scale(u, -1.0))
    } else {
        Some(u)
    }
}

/// Component of `d` relative to the tangent plane of the sphere at `at`.
/// Returns the tangent part of `d`, not normalised.
#[inline]
pub fn tangent_at(at: V3, d: V3) -> V3 {
    sub(d, scale(at, dot(at, d)))
}

/// Local east and north unit vectors at a point on the sphere.
///
/// Degenerate exactly at the poles, where east and north are undefined; the
/// caller gets `None` there rather than a silently wrong frame.
pub fn east_north(at: V3) -> Option<(V3, V3)> {
    let east = unit(cross([0.0, 0.0, 1.0], at))?;
    let north = unit(tangent_at(at, [0.0, 0.0, 1.0]))?;
    Some((east, north))
}

// --------------------------------------------------------------- exact predicate

/// Error-free transformation: `a + b` as an unevaluated double-double.
#[inline]
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let bb = s - a;
    (s, (a - (s - bb)) + (b - bb))
}

/// Error-free transformation: `a * b` as an unevaluated double-double.
#[inline]
fn two_product(a: f64, b: f64) -> (f64, f64) {
    let p = a * b;
    (p, f64::mul_add(a, b, -p))
}

/// Double-double sum.
#[inline]
fn dd_add(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    let (s, e) = two_sum(a.0, b.0);
    let e = e + a.1 + b.1;
    let (s, e2) = two_sum(s, e);
    (s, e2)
}

/// Double-double product of two doubles.
#[inline]
fn dd_mul(a: f64, b: f64) -> (f64, f64) {
    let (p, e) = two_product(a, b);
    (p, e)
}

/// Double-double product of a double-double and a double.
#[inline]
fn dd_mul_d(a: (f64, f64), b: f64) -> (f64, f64) {
    let (p, e) = two_product(a.0, b);
    let e = e + a.1 * b;
    let (s, e2) = two_sum(p, e);
    (s, e2)
}

/// Signed volume of the tetrahedron `(a, b, c, d)`, times six.
///
/// Positive means `d` lies on the outward side of the plane through `a, b, c`
/// when `(a, b, c)` winds counter-clockwise seen from that side. This is the
/// only predicate spherical Delaunay needs: for COSPHERICAL points the classical
/// `insphere` test returns exactly zero for every quadruple, so a lifted-
/// paraboloid Delaunay library hands back a zero-information answer here. The
/// empty-circumcircle test on a sphere IS the convex-hull visibility test.
///
/// Evaluated in double-double arithmetic. The differences are formed in plain
/// f64 first, which is where the only rounding enters; from there the products
/// and sums carry ~32 significant digits. RESOLUTION LIMIT: absolute error is
/// ~1e-30 on unit-sphere inputs, against a near-degeneracy scale of ~1e-13 for a
/// 1 km mesh and ~1e-11 for a 3 km mesh -- 17 decades of margin. A plain f64
/// determinant has an error bound of ~1e-15 and only four decades of margin at
/// 3 km, which is inside the range this door will be asked for.
pub fn orient3d(a: V3, b: V3, c: V3, d: V3) -> f64 {
    let bx = b[0] - a[0];
    let by = b[1] - a[1];
    let bz = b[2] - a[2];
    let cx = c[0] - a[0];
    let cy = c[1] - a[1];
    let cz = c[2] - a[2];
    let dx = d[0] - a[0];
    let dy = d[1] - a[1];
    let dz = d[2] - a[2];

    // det = bx*(cy*dz - cz*dy) - by*(cx*dz - cz*dx) + bz*(cx*dy - cy*dx)
    let m0 = dd_add(dd_mul(cy, dz), dd_mul(-cz, dy));
    let m1 = dd_add(dd_mul(cx, dz), dd_mul(-cz, dx));
    let m2 = dd_add(dd_mul(cx, dy), dd_mul(-cy, dx));

    let t0 = dd_mul_d(m0, bx);
    let t1 = dd_mul_d(m1, -by);
    let t2 = dd_mul_d(m2, bz);

    let s = dd_add(dd_add(t0, t1), t2);
    s.0 + s.1
}

/// The floating-point filter's error bound coefficient, `7 * eps`.
///
/// Shewchuk's `o3derrboundA` for the 3x3 determinant of differences: with the
/// `permanent` below (the sum of the absolute values of the six products the
/// determinant is built from), `|det| > o3derrboundA * permanent` proves the
/// rounded `det` carries the sign of the EXACT determinant of those same
/// differences -- which is precisely what [`orient3d`] returns, to about 1e-30.
/// `eps` here is the unit roundoff `2^-53`, which is HALF of Rust's
/// `f64::EPSILON` (`2^-52`, the gap above 1.0) -- getting that factor wrong in
/// the optimistic direction is how a filter starts returning wrong signs.
const ORIENT3D_FILTER_BOUND: f64 = {
    let eps = f64::EPSILON / 2.0;
    (7.0 + 56.0 * eps) * eps
};

/// The SIGN of [`orient3d`], by a filter that falls back to it.
///
/// Callers that only compare the determinant against zero -- the hull's
/// visibility test is the one that runs 10^8 times -- get the same answer from
/// a plain f64 determinant whenever that determinant is provably larger than
/// its own rounding error, and pay for the double-double evaluation only on
/// the near-degenerate quadruples that need it.
///
/// THE BREAKAGE THIS AVOIDS: the returned VALUE is not [`orient3d`]'s and must
/// never be used as one. `initial_tetrahedron` ranks candidates by
/// `orient3d(..).abs()`, and a filtered magnitude there would pick a different
/// seed tetrahedron, which reorders the facets, which rotates every emitted
/// neighbour ring -- a different grid file from the same generators. Sign only.
#[inline]
pub fn orient3d_sign(a: V3, b: V3, c: V3, d: V3) -> f64 {
    let bx = b[0] - a[0];
    let by = b[1] - a[1];
    let bz = b[2] - a[2];
    let cx = c[0] - a[0];
    let cy = c[1] - a[1];
    let cz = c[2] - a[2];
    let dx = d[0] - a[0];
    let dy = d[1] - a[1];
    let dz = d[2] - a[2];

    let m0 = cy * dz - cz * dy;
    let m1 = cx * dz - cz * dx;
    let m2 = cx * dy - cy * dx;
    let det = bx * m0 - by * m1 + bz * m2;

    // The permanent: the same six products with every sign made positive, each
    // weighted by the row it multiplies -- formed exactly the way the
    // determinant is, which is what makes it a bound on that determinant's
    // rounding error rather than on some other quantity.
    let permanent = bx.abs() * ((cy * dz).abs() + (cz * dy).abs())
        + by.abs() * ((cx * dz).abs() + (cz * dx).abs())
        + bz.abs() * ((cx * dy).abs() + (cy * dx).abs());
    let bound = ORIENT3D_FILTER_BOUND * permanent;
    if det > bound || -det > bound {
        return det;
    }
    orient3d(a, b, c, d)
}

/// `orient3d` with a deterministic tie-break for the exactly-degenerate case.
///
/// Four exactly-cospherical-and-coplanar generators do occur: a Fibonacci or
/// icosahedral seed contains exactly symmetric configurations, and a zero there
/// would let the hull build two different triangulations of the same quad
/// depending on visit order, which is a mesh that changes between runs. The
/// tie-break is Simulation of Simplicity on the point indices: the quadruple is
/// resolved as if each point were perturbed by a distinct infinitesimal, which
/// is consistent across every test that quadruple appears in.
pub fn orient3d_sos(a: V3, b: V3, c: V3, d: V3, ia: u32, ib: u32, ic: u32, id: u32) -> f64 {
    let det = orient3d(a, b, c, d);
    if det != 0.0 {
        return det;
    }
    // Sort the four indices, counting the transpositions; the sign of the
    // symbolic perturbation flips with each swap.
    let mut idx = [ia, ib, ic, id];
    let mut sign = 1.0f64;
    for i in 0..4 {
        for j in 0..3 - i {
            if idx[j] > idx[j + 1] {
                idx.swap(j, j + 1);
                sign = -sign;
            }
        }
    }
    sign
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sphere_triangle_area_matches_the_octant() {
        // One eighth of the sphere: area should be 4*pi/8 = pi/2.
        let a = tri_area([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]);
        assert!(
            (a - std::f64::consts::FRAC_PI_2).abs() < 1e-15,
            "octant area {a} is not pi/2"
        );
    }

    #[test]
    fn tiny_triangle_area_stays_accurate() {
        // A near-planar triangle of side s has area ~ sqrt(3)/4 * s^2. The
        // excess-of-angles form loses every digit here; the half-tangent form
        // must not.
        let s = 1e-6;
        let a = [1.0, 0.0, 0.0];
        let b = unit([1.0, s, 0.0]).unwrap();
        let c = unit([1.0, s / 2.0, s * 0.8660254037844386]).unwrap();
        let got = tri_area(a, b, c).abs();
        let want = 3f64.sqrt() / 4.0 * s * s;
        assert!(
            (got - want).abs() / want < 1e-6,
            "tiny triangle area {got} vs planar {want}"
        );
    }

    #[test]
    fn polygon_area_of_a_hemisphere_square() {
        // Four points on the equator-plus-pole octant boundary: the polygon
        // covering one octant, given as a 3-gon and as a fanned 4-gon, must
        // agree.
        let a = [1.0, 0.0, 0.0];
        let b = [0.0, 1.0, 0.0];
        let c = [0.0, 0.0, 1.0];
        let mid = unit(add(a, b)).unwrap();
        let three = polygon_area(&[a, b, c]);
        let four = polygon_area(&[a, mid, b, c]);
        assert!((three - four).abs() < 1e-15, "{three} vs {four}");
        // reversing the ring negates the signed area, which is what makes a
        // reflex sliver subtract instead of add
        assert!(
            (polygon_area(&[c, b, mid, a]) + four).abs() < 1e-15,
            "the polygon area is not signed"
        );
    }

    #[test]
    fn a_non_convex_ring_does_not_over_count_its_reflex_sliver() {
        // A quadrilateral with one vertex pushed inside the triangle of the
        // other three: the true area is the triangle MINUS the sliver, and an
        // unsigned fan reports the triangle PLUS it.
        let a = unit([1.0, 0.0, 0.0]).unwrap();
        let b = unit([1.0, 0.20, 0.0]).unwrap();
        let c = unit([1.0, 0.10, 0.20]).unwrap();
        // midpoint of a..c pulled toward the interior
        let reflex = unit([1.0, 0.075, 0.085]).unwrap();
        let triangle = polygon_area(&[a, b, c]);
        let quad = polygon_area(&[a, b, c, reflex]);
        assert!(
            quad < triangle,
            "a reflex vertex must remove area: quad {quad} vs triangle {triangle}"
        );
        let unsigned: f64 = [(a, b, c), (a, c, reflex)]
            .iter()
            .map(|&(p, q, r)| tri_area(p, q, r).abs())
            .sum();
        assert!(
            unsigned > triangle,
            "the unsigned fan should be the failure mode being guarded against: {unsigned} vs {triangle}"
        );
    }

    #[test]
    fn circumcentre_of_an_octant_is_the_diagonal() {
        let cc = circumcenter([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]).unwrap();
        let want = unit([1.0, 1.0, 1.0]).unwrap();
        assert!(chord(cc, want) < 1e-15, "{cc:?} vs {want:?}");
    }

    #[test]
    fn circumcentre_is_equidistant_from_its_three_cells() {
        let a = unit([1.0, 0.10, 0.05]).unwrap();
        let b = unit([1.0, -0.07, 0.11]).unwrap();
        let c = unit([1.0, 0.02, -0.09]).unwrap();
        let cc = circumcenter(a, b, c).unwrap();
        let (ra, rb, rc) = (arc(cc, a), arc(cc, b), arc(cc, c));
        assert!(
            (ra - rb).abs() < 1e-15 && (rb - rc).abs() < 1e-15,
            "circumradii {ra} {rb} {rc}"
        );
    }

    #[test]
    fn orient3d_agrees_with_the_naive_determinant_away_from_degeneracy() {
        let a = [1.0, 0.0, 0.0];
        let b = [0.0, 1.0, 0.0];
        let c = [0.0, 0.0, 1.0];
        let d = [0.0, 0.0, 0.0];
        // origin is on the negative side of the (a,b,c) plane with this winding
        assert!(orient3d(a, b, c, d) < 0.0);
        assert!(orient3d(a, b, c, [1.0, 1.0, 1.0]) > 0.0);
        // swapping two arguments flips the sign
        assert_eq!(orient3d(a, b, c, d), -orient3d(b, a, c, d));
    }

    #[test]
    fn orient3d_resolves_a_case_the_naive_determinant_cannot() {
        // Three points 3 km apart on a unit sphere, and a fourth displaced off
        // their plane by 1e-14 of a radius (0.06 mm on Earth). The naive
        // determinant's own rounding is ~1e-15 of the term magnitude here.
        let s = 3000.0 / EARTH_RADIUS_M; // ~4.7e-4 rad
        let a = [1.0, 0.0, 0.0];
        let b = unit([1.0, s, 0.0]).unwrap();
        let c = unit([1.0, 0.0, s]).unwrap();
        // A point exactly IN the plane of the three (their centroid, not their
        // projection onto the sphere -- that sits a sagitta away), then nudged
        // to each side by a hair.
        let on = scale(add(add(a, b), c), 1.0 / 3.0);
        let plane = unit(cross(sub(b, a), sub(c, a))).unwrap();
        let above = add(on, scale(plane, 1e-14));
        let below = add(on, scale(plane, -1e-14));
        assert!(
            orient3d(a, b, c, above) > 0.0,
            "a point 1e-14 above the plane read as not-above"
        );
        assert!(
            orient3d(a, b, c, below) < 0.0,
            "a point 1e-14 below the plane read as not-below"
        );
    }

    /// THE BREAKAGE THIS PREVENTS: `hull::visible` takes the FILTERED sign, and
    /// a filter that answers wrongly on a near-degenerate quadruple builds a
    /// different triangulation -- which is a different neighbour ring, a
    /// different `cellsOnCell`, and a different grid file from the same
    /// generators. The two predicates are held to the same sign over the
    /// configurations the generator actually produces: a golden-ratio sphere
    /// (the seed lattice), quadruples squeezed onto their own plane down to
    /// 1e-18 of a radius, and exactly-coplanar ties where both must read 0.
    #[test]
    fn the_filtered_sign_never_disagrees_with_the_exact_predicate() {
        let ga = std::f64::consts::PI * (3.0 - 5f64.sqrt());
        let pts: Vec<V3> = (0..400)
            .map(|k| {
                let z = 1.0 - (2 * k + 1) as f64 / 400.0;
                let r = (1.0 - z * z).max(0.0).sqrt();
                let t = ga * k as f64;
                unit([r * t.cos(), r * t.sin(), z]).unwrap()
            })
            .collect();
        let mut filtered_took_the_fast_path = 0usize;
        let mut checked = 0usize;
        for i in 0..pts.len() {
            for j in (i + 1)..(i + 9).min(pts.len()) {
                for k in (j + 1)..(j + 5).min(pts.len()) {
                    for d in 0..pts.len() {
                        if d == i || d == j || d == k {
                            continue;
                        }
                        let exact = orient3d(pts[i], pts[j], pts[k], pts[d]);
                        let fast = orient3d_sign(pts[i], pts[j], pts[k], pts[d]);
                        assert_eq!(
                            exact.partial_cmp(&0.0),
                            fast.partial_cmp(&0.0),
                            "the filter read {fast} where the exact predicate reads {exact}"
                        );
                        if fast != exact {
                            filtered_took_the_fast_path += 1;
                        }
                        checked += 1;
                    }
                }
            }
        }
        assert!(checked > 100_000, "only {checked} quadruples were checked");
        // A filter that never fired would make the assertion above vacuous, so
        // the fast path has to be REACHED here. The bar is a third and the
        // measured share on this deliberately degenerate sample -- slivers
        // from three near-neighbours on the seed spiral against every other
        // point -- is about half; a real hull's facets are far rounder and
        // fall back far less.
        assert!(
            filtered_took_the_fast_path * 3 > checked,
            "the filter fell back to the exact predicate on {} of {checked} quadruples, so the fast path is barely exercised",
            checked - filtered_took_the_fast_path
        );

        // Squeezed onto the plane: the range where a naive determinant starts
        // being wrong and the filter has to hand back to the exact one.
        let s = 3000.0 / EARTH_RADIUS_M;
        let a = [1.0, 0.0, 0.0];
        let b = unit([1.0, s, 0.0]).unwrap();
        let c = unit([1.0, 0.0, s]).unwrap();
        let on = scale(add(add(a, b), c), 1.0 / 3.0);
        let plane = unit(cross(sub(b, a), sub(c, a))).unwrap();
        for mag in [1e-12, 1e-14, 1e-16, 1e-18, 1e-20, 0.0] {
            for sign in [1.0f64, -1.0] {
                let d = add(on, scale(plane, sign * mag));
                let exact = orient3d(a, b, c, d);
                let fast = orient3d_sign(a, b, c, d);
                assert_eq!(
                    exact.partial_cmp(&0.0),
                    fast.partial_cmp(&0.0),
                    "at {mag:.0e} off the plane the filter read {fast} against the exact {exact}"
                );
            }
        }

        // An exactly coplanar quadruple: both must read 0 so the SoS tie-break
        // is the thing that decides, on both paths.
        let (q0, q1, q2, q3) = (
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        );
        assert_eq!(orient3d(q0, q1, q2, q3), 0.0);
        assert_eq!(orient3d_sign(q0, q1, q2, q3), 0.0);
    }

    #[test]
    fn sos_breaks_an_exactly_coplanar_tie_consistently() {
        // Four points on a common great circle: exactly coplanar with the
        // origin, so the determinant is exactly zero and only the tie-break
        // decides.
        let a = [1.0, 0.0, 0.0];
        let b = [0.0, 1.0, 0.0];
        let c = [-1.0, 0.0, 0.0];
        let d = [0.0, -1.0, 0.0];
        assert_eq!(orient3d(a, b, c, d), 0.0, "the plain determinant must be 0");
        let s1 = orient3d_sos(a, b, c, d, 3, 7, 11, 19);
        let s2 = orient3d_sos(a, b, c, d, 3, 7, 11, 19);
        assert_eq!(s1, s2, "the tie-break is not deterministic");
        // one transposition of the index labels flips the sign
        let s3 = orient3d_sos(b, a, c, d, 7, 3, 11, 19);
        assert_eq!(s1, -s3, "the tie-break does not follow the permutation sign");
    }

    #[test]
    fn lat_lon_round_trips() {
        for &(lat, lon) in &[(0.0, 0.0), (0.5, 1.2), (-1.1, 5.9), (1.4, 3.0)] {
            let v = from_lat_lon(lat, lon);
            let (la, lo) = lat_lon(v);
            assert!((la - lat).abs() < 1e-15, "lat {la} vs {lat}");
            assert!((lo - lon).abs() < 1e-15, "lon {lo} vs {lon}");
        }
    }

    #[test]
    fn the_latitude_round_trip_is_pole_stable() {
        // This test used to RECORD a resolution limit: the asin(z) latitude
        // lost half its digits near a pole, and its own failure message said
        // that if the loss ever stopped, the limit was gone. It stopped, on
        // a measurement: a graded 224k-cell mesh rolled an edge midpoint
        // 7.1e-8 rad from the south pole, the emitted asin latitude
        // reconstructed the stored Cartesian point only to 1.27e-9, and the
        // port's binary64 load gate (5e-10) refused the whole pair. The
        // atan2(z, hypot(x, y)) form retires the limit; this test now
        // GUARDS the retirement at the measured failure's own distance and
        // closer. (Chord remains the crate's comparison convention: lat/lon
        // is still an emitted field, merely no longer a lossy one.)
        for &off in &[1e-7, 1e-9, 1e-12, 0.0] {
            let lat = std::f64::consts::FRAC_PI_2 - off;
            for sign in [1.0f64, -1.0] {
                let v = from_lat_lon(sign * lat, 3.0);
                let (la, lo) = lat_lon(v);
                let point_gap = chord(from_lat_lon(la, lo), v);
                assert!(
                    point_gap < 5e-16,
                    "the pole round trip lost digits again at {off:.0e} off the pole (chord {point_gap:.3e}); the port's 5e-10 load gate would refuse an emitted mesh with an edge there"
                );
            }
        }
        // Mid-latitude stays exact too.
        let mid = from_lat_lon(0.7, 3.0);
        let (mla, mlo) = lat_lon(mid);
        assert!(chord(from_lat_lon(mla, mlo), mid) < 1e-16, "the mid-latitude round trip is not exact");
    }

    #[test]
    fn arc_beats_acos_on_a_short_edge() {
        // A 3 km edge: acos(dot) has ~1e-11 relative error there, atan2 has
        // ~1e-16. Print both so the resolution limit is on the record.
        let s = 3000.0 / EARTH_RADIUS_M;
        let a = [1.0, 0.0, 0.0];
        let b = unit([s.cos(), s.sin(), 0.0]).unwrap();
        let atan = arc(a, b);
        let acos = dot(a, b).clamp(-1.0, 1.0).acos();
        assert!(
            (atan - s).abs() / s < 1e-15,
            "atan2 arc {atan} vs exact {s}, rel {}",
            (atan - s).abs() / s
        );
        // The acos form is allowed to be worse; this records by how much.
        let acos_rel = (acos - s).abs() / s;
        assert!(acos_rel < 1e-8, "acos arc relative error {acos_rel}");
    }
}
