//! MPAS static operator construction.
//!
//! `cell_gradient_coef_x/y` follows MPAS-A v8.4.1
//! `atm_initialize_deformation_weights`.
//! `deriv_two` follows MPAS-A v8.4.1 `atm_initialize_advection_rk`.
//! `defc_a/b` follows the MPAS v8.4.1 deformation operator from
//! `core_sw/mpas_sw_advection.F`; the atmospheric port consumes these
//! coefficients for the Smagorinsky strain operator, so they are generated
//! rather than silently zero-filled.

use crate::error::{MpasError, MpasResult};
use crate::static_memory::checked_vec;

pub const FIFTEEN: usize = 15;

pub struct OperatorMesh<'a> {
    pub n_cells: usize,
    pub n_edges: usize,
    pub max_edges: usize,
    pub sphere_radius: f64,
    pub n_edges_on_cell: &'a [usize],
    pub cells_on_cell: &'a [usize],    // row-major [cell,max_edges], 0-based
    pub edges_on_cell: &'a [usize],    // row-major [cell,max_edges], 0-based
    pub cells_on_edge: &'a [[usize; 2]], // 0-based
    pub vertices_on_cell: &'a [usize], // row-major [cell,max_edges], 0-based
    pub vertices_on_edge: &'a [[usize; 2]], // 0-based
    pub x_cell: &'a [f64],
    pub y_cell: &'a [f64],
    pub z_cell: &'a [f64],
    pub x_vertex: &'a [f64],
    pub y_vertex: &'a [f64],
    pub z_vertex: &'a [f64],
    pub angle_edge: &'a [f64],
    pub dc_edge: &'a [f64],
}

#[derive(Debug)]
pub struct OperatorFields {
    pub cell_gradient_coef_x: Vec<f32>,
    pub cell_gradient_coef_y: Vec<f32>,
    pub defc_a: Vec<f32>,
    pub defc_b: Vec<f32>,
    pub deriv_two: Vec<f32>, // [edge,side,15]
    /// Regional meshes only: cells whose neighbour ring reaches a culled
    /// cell, so their side of `deriv_two` is left 0 -- exactly what native
    /// `atm_initialize_advection_rk` does on a limited-area mesh
    /// (`cell_list(i) > nCells -> cycle`). 0 on a global mesh.
    pub deriv_two_zero_stencil_cells: usize,
}

fn validate(mesh: &OperatorMesh<'_>) -> MpasResult<()> {
    if mesh.sphere_radius <= 0.0 || !mesh.sphere_radius.is_finite() {
        return Err(MpasError::Refusal("operator sphere radius must be finite and positive".to_string()));
    }
    if mesh.n_edges_on_cell.len() != mesh.n_cells
        || mesh.cells_on_cell.len() != mesh.n_cells * mesh.max_edges
        || mesh.edges_on_cell.len() != mesh.n_cells * mesh.max_edges
        || mesh.vertices_on_cell.len() != mesh.n_cells * mesh.max_edges
        || mesh.cells_on_edge.len() != mesh.n_edges
        || mesh.vertices_on_edge.len() != mesh.n_edges
        || mesh.angle_edge.len() != mesh.n_edges
        || mesh.dc_edge.len() != mesh.n_edges
    {
        return Err(MpasError::Refusal(
            "operator topology shapes are inconsistent with nCells/nEdges/maxEdges".to_string(),
        ));
    }
    if mesh.max_edges + 1 > FIFTEEN {
        return Err(MpasError::Refusal(format!(
            "operator mesh maxEdges={} needs {} polynomial entries, past FIFTEEN={FIFTEEN}",
            mesh.max_edges,
            mesh.max_edges + 1
        )));
    }
    for c in 0..mesh.n_cells {
        let n = mesh.n_edges_on_cell[c];
        if n < 3 || n > mesh.max_edges {
            return Err(MpasError::Refusal(format!(
                "cell {c} has nEdgesOnCell={n}, outside 3..={}",
                mesh.max_edges
            )));
        }
        for s in 0..n {
            let e = mesh.edges_on_cell[c * mesh.max_edges + s];
            let cn = mesh.cells_on_cell[c * mesh.max_edges + s];
            let v = mesh.vertices_on_cell[c * mesh.max_edges + s];
            // `usize::MAX` is the regional absent-neighbour sentinel: a
            // culled cell beyond the outermost ring. Legitimate in
            // cellsOnCell only; the deriv_two stencil skips such cells.
            if e >= mesh.n_edges
                || (cn >= mesh.n_cells && cn != usize::MAX)
                || v >= mesh.x_vertex.len()
            {
                return Err(MpasError::Refusal(format!(
                    "corrupt topology at cell {c} slot {s}: edge={e}, neighbor={cn}, vertex={v}"
                )));
            }
            if mesh.cells_on_edge[e][0] != c && mesh.cells_on_edge[e][1] != c {
                return Err(MpasError::Refusal(format!(
                    "corrupt topology: cell {c} claims edge {e}, but cellsOnEdge={:?}",
                    mesh.cells_on_edge[e]
                )));
            }
        }
    }
    Ok(())
}

#[inline]
fn arc_length(a: [f64; 3], b: [f64; 3]) -> f64 {
    let r = norm(a);
    let c = norm([b[0] - a[0], b[1] - a[1], b[2] - a[2]]);
    r * 2.0 * (c / (2.0 * r)).clamp(-1.0, 1.0).asin()
}

#[inline]
fn norm(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

pub(crate) fn sphere_angle(a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> f64 {
    let side_a = arc_length(b, c);
    let side_b = arc_length(a, c);
    let side_c = arc_length(a, b);
    let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let cross = [
        ab[1] * ac[2] - ab[2] * ac[1],
        -(ab[0] * ac[2] - ab[2] * ac[0]),
        ab[0] * ac[1] - ab[1] * ac[0],
    ];
    let s = 0.5 * (side_a + side_b + side_c);
    let denom = side_b.sin() * side_c.sin();
    let ratio = if denom == 0.0 {
        0.0
    } else {
        ((s - side_b).sin() * (s - side_c).sin() / denom).clamp(0.0, 1.0)
    };
    let angle = 2.0 * ratio.sqrt().clamp(-1.0, 1.0).asin();
    if cross[0] * a[0] + cross[1] * a[1] + cross[2] * a[2] >= 0.0 {
        angle
    } else {
        -angle
    }
}

fn arc_bisect(a: [f64; 3], b: [f64; 3]) -> MpasResult<[f64; 3]> {
    let r = norm(a);
    let mut c = [0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]), 0.5 * (a[2] + b[2])];
    let d = norm(c);
    if d == 0.0 {
        return Err(MpasError::Refusal(
            "operator arc_bisect received diametrically opposite vertices".to_string(),
        ));
    }
    for v in &mut c {
        *v = r * *v / d;
    }
    Ok(c)
}

fn cell_point(mesh: &OperatorMesh<'_>, c: usize) -> [f64; 3] {
    [mesh.x_cell[c], mesh.y_cell[c], mesh.z_cell[c]]
}

fn vertex_point(mesh: &OperatorMesh<'_>, v: usize) -> [f64; 3] {
    [mesh.x_vertex[v], mesh.y_vertex[v], mesh.z_vertex[v]]
}

/// Partial-pivoting inverse of a 6x6 matrix.  The arithmetic order is fixed,
/// deterministic, and mirrors the small dense solve in MPAS's `poly_fit_2`.
fn inv6(mut a: [[f64; 6]; 6]) -> MpasResult<[[f64; 6]; 6]> {
    let mut inv = [[0.0f64; 6]; 6];
    for i in 0..6 {
        inv[i][i] = 1.0;
    }
    for col in 0..6 {
        let mut pivot = col;
        let mut best = a[col][col].abs();
        for r in col + 1..6 {
            if a[r][col].abs() > best {
                best = a[r][col].abs();
                pivot = r;
            }
        }
        if !best.is_finite() || best <= 1.0e-30 {
            return Err(MpasError::Refusal(format!(
                "operator polynomial normal matrix is singular at column {col}"
            )));
        }
        if pivot != col {
            a.swap(pivot, col);
            inv.swap(pivot, col);
        }
        let d = a[col][col];
        for j in 0..6 {
            a[col][j] /= d;
            inv[col][j] /= d;
        }
        for r in 0..6 {
            if r == col {
                continue;
            }
            let f = a[r][col];
            if f == 0.0 {
                continue;
            }
            for j in 0..6 {
                a[r][j] -= f * a[col][j];
                inv[r][j] -= f * inv[col][j];
            }
        }
    }
    Ok(inv)
}

fn polynomial_pseudoinverse(x: &[f64], y: &[f64]) -> MpasResult<Vec<[f64; FIFTEEN]>> {
    // Return six rows of B=(A^T A)^-1 A^T, padded to FIFTEEN columns.
    let m = x.len() + 1;
    if m > FIFTEEN {
        return Err(MpasError::Refusal(format!(
            "polynomial stencil has {m} points, past FIFTEEN={FIFTEEN}"
        )));
    }
    let mut a = [[0.0f64; 6]; FIFTEEN];
    a[0][0] = 1.0;
    for row in 1..m {
        let xp = x[row - 1];
        let yp = y[row - 1];
        a[row] = [1.0, xp, yp, xp * xp, xp * yp, yp * yp];
    }
    let mut ata = [[0.0f64; 6]; 6];
    for i in 0..6 {
        for j in 0..6 {
            let mut sum = 0.0;
            for r in 0..m {
                sum += a[r][i] * a[r][j];
            }
            ata[i][j] = sum;
        }
    }
    let inv = inv6(ata)?;
    let mut b = vec![[0.0f64; FIFTEEN]; 6];
    for i in 0..6 {
        for r in 0..m {
            let mut sum = 0.0;
            for j in 0..6 {
                sum += inv[i][j] * a[r][j];
            }
            b[i][r] = sum;
        }
    }
    Ok(b)
}

pub fn build_operator_fields(mesh: &OperatorMesh<'_>) -> MpasResult<OperatorFields> {
    validate(mesh)?;
    let slots = mesh.n_cells * mesh.max_edges;
    let mut gx = checked_vec::<f32>(slots, "operators", "cell_gradient_coef_x")?;
    let mut gy = checked_vec::<f32>(slots, "operators", "cell_gradient_coef_y")?;
    let mut defa = checked_vec::<f32>(slots, "operators", "defc_a")?;
    let mut defb = checked_vec::<f32>(slots, "operators", "defc_b")?;
    let mut d2 = checked_vec::<f32>(
        mesh.n_edges * 2 * FIFTEEN,
        "operators",
        "deriv_two",
    )?;

    let unit_north = [0.0, 0.0, 1.0];
    let pi = 2.0 * 1.0f64.asin();
    let mut deriv_two_zero_stencil_cells = 0usize;

    for cell in 0..mesh.n_cells {
        let ne = mesh.n_edges_on_cell[cell];
        let center_phys = cell_point(mesh, cell);
        let center = [
            center_phys[0] / mesh.sphere_radius,
            center_phys[1] / mesh.sphere_radius,
            center_phys[2] / mesh.sphere_radius,
        ];

        // --- deformation/gradient tangent polygon from cell vertices ---
        let mut xp = [0.0f64; FIFTEEN];
        let mut yp = [0.0f64; FIFTEEN];
        let first_v = mesh.vertices_on_cell[cell * mesh.max_edges];
        let v0p = vertex_point(mesh, first_v);
        let v0 = [
            v0p[0] / mesh.sphere_radius,
            v0p[1] / mesh.sphere_radius,
            v0p[2] / mesh.sphere_radius,
        ];
        let theta_abs = if center[2] == 1.0 {
            pi / 2.0
        } else {
            pi / 2.0 - sphere_angle(center, v0, unit_north)
        };
        let mut theta = [0.0f64; FIFTEEN];
        theta[0] = theta_abs;
        for i in 0..ne {
            let va = mesh.vertices_on_cell[cell * mesh.max_edges + i];
            let vb = mesh.vertices_on_cell[cell * mesh.max_edges + ((i + 1) % ne)];
            let ap = vertex_point(mesh, va);
            let bp = vertex_point(mesh, vb);
            let a = [ap[0] / mesh.sphere_radius, ap[1] / mesh.sphere_radius, ap[2] / mesh.sphere_radius];
            let b = [bp[0] / mesh.sphere_radius, bp[1] / mesh.sphere_radius, bp[2] / mesh.sphere_radius];
            let d = mesh.sphere_radius * arc_length(center, a);
            if i > 0 {
                let prev = mesh.vertices_on_cell[cell * mesh.max_edges + i - 1];
                let pp = vertex_point(mesh, prev);
                let p = [pp[0] / mesh.sphere_radius, pp[1] / mesh.sphere_radius, pp[2] / mesh.sphere_radius];
                theta[i] = theta[i - 1] + sphere_angle(center, p, a);
            }
            xp[i] = theta[i].cos() * d;
            yp[i] = theta[i].sin() * d;
            let _ = b; // b participates below through polygon differences
        }

        let mut area = 0.0f64;
        let mut normal_angle = [0.0f64; FIFTEEN];
        for i in 0..ne {
            let j = (i + 1) % ne;
            let dx = xp[j] - xp[i];
            let dy = yp[j] - yp[i];
            area += 0.25 * (xp[i] + xp[j]) * (yp[j] - yp[i])
                - 0.25 * (yp[i] + yp[j]) * (xp[j] - xp[i]);
            normal_angle[i] = dy.atan2(dx) - pi / 2.0;
        }
        if !area.is_finite() || area.abs() <= 1.0e-20 {
            return Err(MpasError::Refusal(format!(
                "cell {cell} has degenerate tangent-plane area {area}; refusing operator construction"
            )));
        }
        for i in 0..ne {
            let j = (i + 1) % ne;
            let dl = ((xp[j] - xp[i]).powi(2) + (yp[j] - yp[i]).powi(2)).sqrt();
            gx[cell * mesh.max_edges + i] = (dl * normal_angle[i].cos() / area) as f32;
            gy[cell * mesh.max_edges + i] = (dl * normal_angle[i].sin() / area) as f32;

            // MPAS deformation tensor coefficients.
            let s = normal_angle[i].sin();
            let c = normal_angle[i].cos();
            let edge = mesh.edges_on_cell[cell * mesh.max_edges + i];
            let sign = if mesh.cells_on_edge[edge][0] == cell { 1.0 } else { -1.0 };
            defa[cell * mesh.max_edges + i] = (sign * dl * (c * c - s * s) / area) as f32;
            defb[cell * mesh.max_edges + i] = (sign * dl * (2.0 * s * c) / area) as f32;
        }

        // --- deriv_two polynomial fit over cell center + neighboring cells ---
        //
        // The regional branch, exactly native `atm_initialize_advection_rk`:
        // a cell whose neighbour ring reaches a culled cell is skipped and its
        // side of deriv_two stays 0 (`if (cell_list(i) > nCells) ... cycle`).
        // The absent centre is not in the file, so there is nothing to fit.
        if (0..ne).any(|i| mesh.cells_on_cell[cell * mesh.max_edges + i] >= mesh.n_cells) {
            deriv_two_zero_stencil_cells += 1;
            continue;
        }
        let mut nx = [0.0f64; FIFTEEN];
        let mut ny = [0.0f64; FIFTEEN];
        let mut neighbor_unit = [[0.0f64; 3]; FIFTEEN];
        for i in 0..ne {
            let cn = mesh.cells_on_cell[cell * mesh.max_edges + i];
            let p = cell_point(mesh, cn);
            neighbor_unit[i] = [
                p[0] / mesh.sphere_radius,
                p[1] / mesh.sphere_radius,
                p[2] / mesh.sphere_radius,
            ];
        }
        let first = neighbor_unit[0];
        let theta0 = if center[2] == 1.0 {
            pi / 2.0
        } else {
            pi / 2.0 - sphere_angle(center, first, unit_north)
        };
        let mut t = theta0;
        for i in 0..ne {
            if i > 0 {
                t += sphere_angle(center, neighbor_unit[i - 1], neighbor_unit[i]);
            }
            let dist = mesh.sphere_radius * arc_length(center, neighbor_unit[i]);
            nx[i] = t.cos() * dist;
            ny[i] = t.sin() * dist;
        }
        let b = polynomial_pseudoinverse(&nx[..ne], &ny[..ne])?;

        for i in 0..ne {
            let edge = mesh.edges_on_cell[cell * mesh.max_edges + i];
            let verts = mesh.vertices_on_edge[edge];
            let va = vertex_point(mesh, verts[0]);
            let vb = vertex_point(mesh, verts[1]);
            let va = [va[0] / mesh.sphere_radius, va[1] / mesh.sphere_radius, va[2] / mesh.sphere_radius];
            let vb = [vb[0] / mesh.sphere_radius, vb[1] / mesh.sphere_radius, vb[2] / mesh.sphere_radius];
            let mid = arc_bisect(va, vb)?;
            let edge_theta = sphere_angle(center, neighbor_unit[i], mid) + {
                let mut tt = theta0;
                for k in 1..=i {
                    tt += sphere_angle(center, neighbor_unit[k - 1], neighbor_unit[k]);
                }
                tt
            };
            let c = edge_theta.cos();
            let s = edge_theta.sin();
            let side = if mesh.cells_on_edge[edge][0] == cell { 0 } else { 1 };
            let base = (edge * 2 + side) * FIFTEEN;
            for j in 0..=ne {
                let value = 2.0 * c * c * b[3][j]
                    + 2.0 * c * s * b[4][j]
                    + 2.0 * s * s * b[5][j];
                d2[base + j] = value as f32;
            }
        }
    }

    Ok(OperatorFields {
        cell_gradient_coef_x: gx,
        cell_gradient_coef_y: gy,
        defc_a: defa,
        defc_b: defb,
        deriv_two: d2,
        deriv_two_zero_stencil_cells,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inverse_identity() {
        let mut a = [[0.0; 6]; 6];
        for i in 0..6 {
            a[i][i] = 1.0;
        }
        assert_eq!(inv6(a).unwrap(), a);
    }

    #[test]
    fn polynomial_constant_row_closes() {
        let x = [1.0, 0.5, -0.5, -1.0, -0.5, 0.5];
        let y = [0.0, 0.866, 0.866, 0.0, -0.866, -0.866];
        let b = polynomial_pseudoinverse(&x, &y).unwrap();
        // A constant field's quadratic derivative must vanish.
        let mut dxx = 0.0;
        for j in 0..7 {
            dxx += 2.0 * b[3][j];
        }
        assert!(dxx.abs() < 1.0e-10, "{dxx}");
    }
}
