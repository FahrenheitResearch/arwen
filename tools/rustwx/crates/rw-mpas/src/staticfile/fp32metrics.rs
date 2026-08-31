//! Does this static survive the MPAS port's LIVE load contract?
//!
//! # The two gates this measures
//!
//! A static stores every coordinate and length as `f32`, at earth-radius
//! magnitude. At 6,371,229 m an `f32` ULP is 0.5 m, so a length RECOVERED from
//! two stored vertices carries up to `sqrt(3)` coordinate ULPs (~0.87 m) of
//! quantisation however exactly it was computed.
//!
//! The MPAS port judges the pair with TWO gates (contract of 2026-08-23;
//! before that it compared at `rtol = 2e-5, atol = 0.0`, a premise retired by
//! the stale-guard audit of 2026-08-25, finding 3):
//!
//! * STORAGE (`mpas_port/mesh.py`, `spherical_arc_tolerance`): the recomputed
//!   arc must agree with the stored `dvEdge` within an ABSOLUTE floor of
//!   `2*sqrt(3)` coordinate ULPs -- 1.73 m for f32 at earth radius -- plus an
//!   `8 * f32::EPSILON` relative term. A generated pair sits under ONE
//!   coordinate rounding by construction, so a reading past the floor means
//!   the file was rescaled or edited after generation.
//! * ADMISSION (`mpas_port/dual_edge_admission.py`, `DualEdgePolicy`): every
//!   edge must carry `dvEdge/dcEdge >= 0.02`, because the TRiSK tangential
//!   terms divide by `dvEdge` and a near-degenerate dual edge amplifies the
//!   tangential gradient by `dc/dv`. Refused before any CUDA allocation with
//!   `DualEdgeAdmissionError`.
//!
//! MEASURED 2026-08-25, against the live loader itself: a generated
//! 654,432-cell uniform 30 km pair carrying 12,732 edges past the RETIRED
//! 2e-5 relative bound (worst 4.64e-5 relative, 0.634 m absolute, shortest
//! dual edge 9,588.6 m, min dv/dc 0.368) is ACCEPTED whole by
//! `Mesh.from_netcdf(validate=True)`; the same loader refuses a
//! ratio-1e-4 edge through the admission gate.
//!
//! # Why this is measured rather than modelled
//!
//! `0.5 m / L` predicts the quantisation well, and a prediction is still not
//! the file. This walks the bytes that were actually written and recomputes
//! what the consumer will recompute, so the numbers in the receipt are the
//! numbers the consumer is going to get. Nothing here refuses: the engine
//! reports, and the door turns the reading into the verdict, because a static
//! that a different consumer would accept is not the engine's to reject.

use serde::Serialize;

use crate::staticfile::coordframe::CoordinateRepresentation;

/// The relative term of the port's storage gate: `_SPHERICAL_ARC_MAX_RTOL`,
/// eight `f32` ULP of relative headroom (9.5367431640625e-7).
pub const PORT_ARC_RTOL: f64 = 8.0 * (f32::EPSILON as f64);

/// The port's dual-edge admission floor: `DualEdgePolicy.minimum_dv_over_dc`.
/// Below it the TRiSK tangential terms amplify a gradient by more than 50x
/// and the port refuses the pair before any CUDA allocation.
pub const PORT_MIN_DV_OVER_DC: f64 = 0.02;

/// The absolute term of the port's storage gate at this radius: `2*sqrt(3)`
/// coordinate ULPs of the `f32` the coordinates are stored in -- 1.73 m at
/// earth radius. One rounding moves an endpoint by at most `sqrt(3)/2` ULPs;
/// two endpoints in opposite directions carry `sqrt(3)`, and the factor of
/// two on top covers a pair rescaled between two sphere radii (a second
/// rounding), exactly as `mpas_port.mesh.spherical_arc_tolerance` derives it.
pub fn port_arc_atol_m(sphere_radius: f64) -> f64 {
    port_arc_atol_m_for(sphere_radius, CoordinateRepresentation::Binary32EarthCentred)
}

/// The same bound for a file that declares a different coordinate
/// representation. The port reads the floor off the dtype of
/// `xCell`/`yCell`/`zCell`, so this has to as well or the two disagree:
/// 1.73 m at binary32, 3.2e-9 m at binary64.
pub fn port_arc_atol_m_for(
    sphere_radius: f64,
    coordinates: CoordinateRepresentation,
) -> f64 {
    2.0 * 3.0f64.sqrt() * coordinates.quantum_m(sphere_radius)
}

/// What FP32 storage did to this file's own edge lengths, judged against the
/// port's live contract.
#[derive(Debug, Clone, Serialize)]
pub struct Fp32MetricAgreement {
    /// Worst relative disagreement between the stored `dvEdge` and the arc
    /// recomputed from the stored vertices. Informative: the port no longer
    /// judges storage relatively, but the reading dates the file's regime.
    pub max_dv_edge_relative: f64,
    /// The edge that produced it.
    pub worst_edge: usize,
    /// That edge's length in metres, which is what makes it the worst one.
    pub worst_edge_length_m: f64,
    /// The shortest dual edge in the mesh, in metres.
    pub min_dv_edge_m: f64,
    /// Worst ABSOLUTE disagreement in metres -- the number the port's
    /// storage gate actually judges.
    pub max_dv_edge_absolute_m: f64,
    /// Edges whose disagreement exceeds `atol + rtol * arc`. Non-zero means
    /// the file was rescaled or edited after generation, not that the mesh
    /// is too fine.
    pub edges_past_port_storage_tolerance: usize,
    /// The absolute bound the reading is judged against (1.73 m at earth
    /// radius for f32 coordinates).
    pub port_arc_atol_m: f64,
    /// The relative term beside it.
    pub port_arc_rtol: f64,
    /// Smallest `dvEdge/dcEdge` in the file's own stored metrics.
    pub min_dv_over_dc: f64,
    /// Edges below the port's 0.02 admission floor. Non-zero means the port
    /// refuses the pair with `DualEdgeAdmissionError`.
    pub edges_below_admission_floor: usize,
    /// The admission floor the ratio is judged against.
    pub port_min_dv_over_dc: f64,
}

impl Fp32MetricAgreement {
    /// True when the port's live load contract will accept these metrics:
    /// no edge past the storage tolerance, none below the admission floor.
    pub fn port_accepts(&self) -> bool {
        self.edges_past_port_storage_tolerance == 0 && self.edges_below_admission_floor == 0
    }
}

/// Recompute `dvEdge` from the vertices exactly as the consumer does, and
/// take the dv/dc reading its admission gate takes.
///
/// `vertices_on_edge` is 1-BASED, as the file stores it. The METRICS are the
/// metre-scale values that will be written, already rounded to `f32` -- they
/// are binary32 in every coordinate representation. The VERTICES are the
/// metre-scale values that will be written, already rounded to whatever
/// `coordinates` says, and the tolerance is derived from that same
/// representation so this reading and the consumer's are the same comparison.
pub fn measure(
    dv_edge_f32: &[f32],
    x_vertex_f32: &[f64],
    y_vertex_f32: &[f64],
    z_vertex_f32: &[f64],
    vertices_on_edge: &[i64],
    dc_edge_f32: &[f32],
    sphere_radius: f64,
    coordinates: CoordinateRepresentation,
) -> Fp32MetricAgreement {
    let atol = port_arc_atol_m_for(sphere_radius, coordinates);
    let mut worst = 0.0f64;
    let mut worst_edge = 0usize;
    let mut worst_len = 0.0f64;
    let mut min_dv = f64::INFINITY;
    let mut max_abs = 0.0f64;
    let mut past_storage = 0usize;
    let mut min_ratio = f64::INFINITY;
    let mut below_floor = 0usize;
    for e in 0..dv_edge_f32.len() {
        let stored = dv_edge_f32[e] as f64;
        let dc = dc_edge_f32.get(e).copied().unwrap_or(0.0) as f64;
        if stored.is_finite() && stored > 0.0 {
            if stored < min_dv {
                min_dv = stored;
            }
            if dc.is_finite() && dc > 0.0 {
                let ratio = stored / dc;
                if ratio < min_ratio {
                    min_ratio = ratio;
                }
                if ratio < PORT_MIN_DV_OVER_DC {
                    below_floor += 1;
                }
            }
        }
        let v1 = vertices_on_edge[2 * e] - 1;
        let v2 = vertices_on_edge[2 * e + 1] - 1;
        if v1 < 0 || v2 < 0 {
            continue;
        }
        let (v1, v2) = (v1 as usize, v2 as usize);
        let dx = x_vertex_f32[v2] - x_vertex_f32[v1];
        let dy = y_vertex_f32[v2] - y_vertex_f32[v1];
        let dz = z_vertex_f32[v2] - z_vertex_f32[v1];
        let chord = (dx * dx + dy * dy + dz * dz).sqrt();
        let arc = sphere_radius * 2.0 * (chord / (2.0 * sphere_radius)).clamp(-1.0, 1.0).asin();
        if !(arc.is_finite() && arc > 0.0) {
            continue;
        }
        let abs = (stored - arc).abs();
        if abs > max_abs {
            max_abs = abs;
        }
        if abs > atol + PORT_ARC_RTOL * arc {
            past_storage += 1;
        }
        let rel = abs / arc;
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
        max_dv_edge_absolute_m: max_abs,
        edges_past_port_storage_tolerance: past_storage,
        port_arc_atol_m: atol,
        port_arc_rtol: PORT_ARC_RTOL,
        min_dv_over_dc: if min_ratio.is_finite() { min_ratio } else { 0.0 },
        edges_below_admission_floor: below_floor,
        port_min_dv_over_dc: PORT_MIN_DV_OVER_DC,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const B32: CoordinateRepresentation =
        CoordinateRepresentation::Binary32EarthCentred;
    const B64: CoordinateRepresentation =
        CoordinateRepresentation::Binary64EarthCentred;

    /// The pair sits at a GENERIC point on the sphere, not on an axis. That is
    /// not decoration: put a short edge along the equator and one endpoint
    /// lands on `(R, 0, 0)` exactly while the other's x-component rounds back
    /// to the same `R`, the error cancels, and the reading measures clean.
    /// Real vertices have all three components at ~R magnitude and no such
    /// luck, which is why the reading in the receipt is taken from the file's
    /// own bytes rather than from a length.
    fn edge_at(arc_m: f64, r: f64) -> (Vec<f32>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let (lat, lon) = (0.61_f64, 2.37_f64); // nothing special, and not an axis
        let centre = [
            r * lat.cos() * lon.cos(),
            r * lat.cos() * lon.sin(),
            r * lat.sin(),
        ];
        let east = [-lon.sin(), lon.cos(), 0.0];
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
            xs.push(((r * p[0] / n) as f32) as f64);
            ys.push(((r * p[1] / n) as f32) as f64);
            zs.push(((r * p[2] / n) as f32) as f64);
        }
        (vec![arc_m as f32], xs, ys, zs)
    }

    /// The finding-3 regime, in the smallest mesh that shows it: a 75 m dual
    /// edge quantises to a RELATIVE disagreement orders past the retired
    /// 2e-5 bound while staying inside the live 1.73 m storage atol.  Under
    /// the retired contract this file read as unloadable; under the live one
    /// storage PASSES and what gates it is the 0.02 admission floor.
    #[test]
    fn a_short_dual_edge_is_gated_by_admission_not_storage() {
        let r = 6_371_229.0f64;
        let voe = vec![1i64, 2];

        let (dv, x, y, z) = edge_at(75.0, r);
        let short = measure(&dv, &x, &y, &z, &voe, &[45_000.0f32], r, B32);
        assert!(
            short.max_dv_edge_relative > 2.0e-5,
            "a 75 m edge at earth radius must quantise past the retired \
             relative bound, or this test lost its regime: {short:?}"
        );
        assert_eq!(
            short.edges_past_port_storage_tolerance, 0,
            "quantisation alone cannot exceed the 1.73 m storage atol: {short:?}"
        );
        assert_eq!(
            short.edges_below_admission_floor, 1,
            "75 m over 45 km is ratio 1.7e-3, below the 0.02 floor: {short:?}"
        );
        assert!(!short.port_accepts(), "{short:?}");

        // A healthy edge passes both gates.
        let (dv, x, y, z) = edge_at(45_000.0, r);
        let long = measure(&dv, &x, &y, &z, &voe, &[78_000.0f32], r, B32);
        assert_eq!(long.edges_past_port_storage_tolerance, 0, "{long:?}");
        assert_eq!(long.edges_below_admission_floor, 0, "{long:?}");
        assert!(long.port_accepts(), "{long:?}");
        assert!(long.max_dv_edge_absolute_m < long.port_arc_atol_m, "{long:?}");
    }

    /// The storage gate still exists and still catches what it is FOR: a
    /// length that disagrees with its own vertices by more than rounding --
    /// a file rescaled or edited after generation.
    #[test]
    fn a_corrupted_length_is_caught_by_the_storage_gate() {
        let r = 6_371_229.0f64;
        let voe = vec![1i64, 2];
        let (mut dv, x, y, z) = edge_at(45_000.0, r);
        dv[0] += 5.0; // five metres of disagreement no rounding produces
        let bad = measure(&dv, &x, &y, &z, &voe, &[78_000.0f32], r, B32);
        assert_eq!(bad.edges_past_port_storage_tolerance, 1, "{bad:?}");
        assert!(bad.max_dv_edge_absolute_m > bad.port_arc_atol_m, "{bad:?}");
        assert!(!bad.port_accepts(), "{bad:?}");
    }

    /// The SAME 75 m dual edge, stored at binary64 coordinates. The storage
    /// disagreement collapses by the ratio of the two quanta and the reading
    /// stops being about storage at all -- what still refuses the edge is the
    /// 0.02 admission floor, which is about mesh SHAPE and moves for no
    /// representation.
    #[test]
    fn binary64_coordinates_take_storage_out_of_the_question() {
        let r = 6_371_229.0f64;
        let voe = vec![1i64, 2];
        let (dv, x, y, z) = edge_at_exact(75.0, r);
        let b64 = measure(&dv, &x, &y, &z, &voe, &[45_000.0f32], r, B64);
        assert!(
            b64.port_arc_atol_m < 4.0e-9 && b64.port_arc_atol_m > 3.0e-9,
            "binary64 atol at earth radius is 3.2e-9 m: {:e}",
            b64.port_arc_atol_m
        );
        // dvEdge itself is still binary32, so the disagreement is that array's
        // own rounding (6e-8 relative), not the coordinates'.
        assert!(
            b64.max_dv_edge_relative < 1.0e-6,
            "binary64 vertices leave only the f32 dvEdge rounding: {b64:?}"
        );
        assert_eq!(
            b64.edges_below_admission_floor, 1,
            "75 m over 45 km is still ratio 1.7e-3: {b64:?}"
        );
    }

    /// `edge_at` rounds its vertices to f32 on purpose; this is the same
    /// construction left in f64 so a binary64 file can be measured.
    fn edge_at_exact(arc_m: f64, r: f64) -> (Vec<f32>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let (lat, lon) = (0.61_f64, 2.37_f64);
        let centre = [
            r * lat.cos() * lon.cos(),
            r * lat.cos() * lon.sin(),
            r * lat.sin(),
        ];
        let east = [-lon.sin(), lon.cos(), 0.0];
        let half = arc_m / 2.0;
        let (mut xs, mut ys, mut zs) = (Vec::new(), Vec::new(), Vec::new());
        for sign in [-1.0f64, 1.0] {
            let p = [
                centre[0] + sign * half * east[0],
                centre[1] + sign * half * east[1],
                centre[2] + sign * half * east[2],
            ];
            let n = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
            xs.push(r * p[0] / n);
            ys.push(r * p[1] / n);
            zs.push(r * p[2] / n);
        }
        (vec![arc_m as f32], xs, ys, zs)
    }

    /// The atol is a property of the coordinate dtype at the file's own
    /// radius, exactly as the port derives it: 1.73 m at earth radius.
    #[test]
    fn the_storage_atol_is_the_ports_own() {
        let atol = port_arc_atol_m(6_371_229.0);
        assert!((atol - 1.732_050_8).abs() < 1e-6, "{atol}");
    }
}
