//! Does this static survive being written in FP32?
//!
//! # The defect this measures
//!
//! A static stores every coordinate and length as `f32`, at earth-radius
//! magnitude. At 6,371,229 m an `f32` ULP is 0.5 m, so a length RECOVERED from
//! two stored vertices carries roughly half a metre of error however exactly it
//! was computed. On a 45 km dual edge that is 1.2e-05 relative; on a 75 m one it
//! is 7e-03.
//!
//! The MPAS GPU port recomputes `dvEdge` from the stored vertices and compares,
//! with `rtol = 2.0e-5` and `atol = 0.0` for an FP32 mesh
//! (`mpas_port/mesh.py:808`). A pair that fails is refused whole:
//!
//! ```text
//! MeshValidationError: MPAS mesh validation failed:
//!  - dvEdge disagrees with spherical vertex arc length
//! ```
//!
//! MEASURED on generated meshes at a 120 km background, shortest dual edge:
//!
//! | cells   | shortest dvEdge | edges under 1 km |
//! |--------:|----------------:|-----------------:|
//! |   2,000 |       7,337.6 m |                0 |
//! |  12,000 |          75.0 m |                3 |
//! |  40,962 |           7.2 m |               28 |
//! | 127,051 |           3.4 m |              147 |
//!
//! against the PUBLISHED `x1.40962` at the same 40,962 cells: shortest dual edge
//! 45,016.7 m, nothing under 5 km. So this is not a property of the resolution
//! and not a tolerance that wants loosening -- it is a handful of near-degenerate
//! Voronoi vertices, and the published family does not have them.
//!
//! It is also not under-relaxation, which was the first guess. MEASURED on the
//! same 12,000-cell request at `--sweeps` 200, 600 and 2000: the shortest edge
//! is 75.04 m in all three, and three edges sit under a kilometre in all three.
//! The relaxation has converged; this is the converged mesh.
//!
//! # Why this is measured rather than modelled
//!
//! `0.5 m / L` predicts the published file's 1.19e-05 well, and a prediction is
//! still not the file. This walks the bytes that were actually written and
//! recomputes what the consumer will recompute, so the number in the receipt is
//! the number the consumer is going to get. Nothing here refuses: the engine
//! reports, and the door turns the reading into the verdict, because a static
//! that a different consumer would accept is not the engine's to reject.

use serde::Serialize;

/// The consumer's relative tolerance on an FP32 mesh's recomputed metrics.
///
/// `mpas_port/mesh.py:808`: `metric_rtol = 2.0e-5 if has_float32 else 5.0e-10`,
/// applied with `atol = 0.0`. Named here so the reading has a bound to be read
/// against instead of being a number nobody can act on.
pub const CONSUMER_METRIC_RTOL: f64 = 2.0e-5;

/// What FP32 storage did to this file's own edge lengths.
#[derive(Debug, Clone, Serialize)]
pub struct Fp32MetricAgreement {
    /// Worst relative disagreement between the stored `dvEdge` and the arc
    /// recomputed from the stored vertices.
    pub max_dv_edge_relative: f64,
    /// The edge that produced it.
    pub worst_edge: usize,
    /// That edge's length in metres, which is what makes it the worst one.
    pub worst_edge_length_m: f64,
    /// The shortest dual edge in the mesh, in metres.
    pub min_dv_edge_m: f64,
    /// How many edges are individually past the consumer's tolerance.
    pub edges_past_consumer_tolerance: usize,
    /// The bound the reading is judged against.
    pub consumer_metric_rtol: f64,
}

impl Fp32MetricAgreement {
    /// True when the consumer's `Mesh.validate` will accept these lengths.
    pub fn within_consumer_tolerance(&self) -> bool {
        self.max_dv_edge_relative <= self.consumer_metric_rtol
    }
}

/// Recompute `dvEdge` from the vertices exactly as the consumer does.
///
/// `vertices_on_edge` is 1-BASED, as the file stores it. Coordinates are the
/// metre-scale values that will be written, already rounded to `f32`.
pub fn measure(
    dv_edge_f32: &[f32],
    x_vertex_f32: &[f32],
    y_vertex_f32: &[f32],
    z_vertex_f32: &[f32],
    vertices_on_edge: &[i64],
    sphere_radius: f64,
) -> Fp32MetricAgreement {
    let mut worst = 0.0f64;
    let mut worst_edge = 0usize;
    let mut worst_len = 0.0f64;
    let mut min_dv = f64::INFINITY;
    let mut past = 0usize;
    for e in 0..dv_edge_f32.len() {
        let v1 = vertices_on_edge[2 * e] - 1;
        let v2 = vertices_on_edge[2 * e + 1] - 1;
        if v1 < 0 || v2 < 0 {
            continue;
        }
        let (v1, v2) = (v1 as usize, v2 as usize);
        let stored = dv_edge_f32[e] as f64;
        if stored.is_finite() && stored > 0.0 && stored < min_dv {
            min_dv = stored;
        }
        let dx = x_vertex_f32[v2] as f64 - x_vertex_f32[v1] as f64;
        let dy = y_vertex_f32[v2] as f64 - y_vertex_f32[v1] as f64;
        let dz = z_vertex_f32[v2] as f64 - z_vertex_f32[v1] as f64;
        let chord = (dx * dx + dy * dy + dz * dz).sqrt();
        let arc = sphere_radius * 2.0 * (chord / (2.0 * sphere_radius)).clamp(-1.0, 1.0).asin();
        if !(arc.is_finite() && arc > 0.0) {
            continue;
        }
        let rel = (stored - arc).abs() / arc;
        if rel > CONSUMER_METRIC_RTOL {
            past += 1;
        }
        if rel > worst {
            worst = rel;
            worst_edge = e;
            worst_len = arc;
        }
    }
    Fp32MetricAgreement {
        max_dv_edge_relative: worst,
        worst_edge,
        worst_edge_length_m: worst_len,
        min_dv_edge_m: if min_dv.is_finite() { min_dv } else { 0.0 },
        edges_past_consumer_tolerance: past,
        consumer_metric_rtol: CONSUMER_METRIC_RTOL,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two vertices a long way apart survive FP32; two very close together do
    /// not. This is the whole mechanism, in the smallest mesh that shows it.
    ///
    /// The pair sits at a GENERIC point on the sphere, not on an axis. That is
    /// not decoration: put a short edge along the equator and one endpoint
    /// lands on `(R, 0, 0)` exactly while the other's x-component rounds back
    /// to the same `R`, the error cancels, and a 75 m edge measures clean. Real
    /// vertices have all three components at ~R magnitude and no such luck,
    /// which is why the reading in the receipt is taken from the file's own
    /// bytes rather than from a length.
    #[test]
    fn a_short_dual_edge_fails_what_a_long_one_passes() {
        let r = 6_371_229.0f64;
        let (lat, lon) = (0.61_f64, 2.37_f64); // nothing special, and not an axis
        let centre = [
            r * lat.cos() * lon.cos(),
            r * lat.cos() * lon.sin(),
            r * lat.sin(),
        ];
        // A tangent direction at that point, normalised.
        let east = [-lon.sin(), lon.cos(), 0.0];

        let make = |arc_m: f64| -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
            let half = arc_m / 2.0;
            let mut xs = Vec::new();
            let mut ys = Vec::new();
            let mut zs = Vec::new();
            for sign in [-1.0f64, 1.0] {
                let p = [
                    centre[0] + sign * half * east[0],
                    centre[1] + sign * half * east[1],
                    centre[2] + sign * half * east[2],
                ];
                // Put it back on the sphere so the arc is the arc.
                let n = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
                xs.push((r * p[0] / n) as f32);
                ys.push((r * p[1] / n) as f32);
                zs.push((r * p[2] / n) as f32);
            }
            (vec![arc_m as f32], xs, ys, zs)
        };
        let voe = vec![1i64, 2];

        let (dv, x, y, z) = make(45_000.0);
        let long = measure(&dv, &x, &y, &z, &voe, r);
        assert!(
            long.within_consumer_tolerance(),
            "a 45 km dual edge should survive FP32: {long:?}"
        );
        assert_eq!(long.edges_past_consumer_tolerance, 0);

        let (dv, x, y, z) = make(75.0);
        let short = measure(&dv, &x, &y, &z, &voe, r);
        assert!(
            !short.within_consumer_tolerance(),
            "a 75 m dual edge cannot survive FP32 at earth radius: {short:?}"
        );
        assert_eq!(short.edges_past_consumer_tolerance, 1);
        assert!(short.min_dv_edge_m < 100.0);
    }
}
