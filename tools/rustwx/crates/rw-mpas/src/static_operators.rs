//! MPAS static operator construction.
//!
//! `cell_gradient_coef_x/y` follows MPAS-A v8.4.2
//! `atm_initialize_deformation_weights`.
//! `deriv_two` follows MPAS-A v8.4.2 `atm_initialize_advection_rk`.
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
    pub on_a_sphere: bool,
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
    for c in 0..mesh.n_cells {
        let n = mesh.n_edges_on_cell[c];
        if n < 3 || n > mesh.max_edges {
            return Err(MpasError::Refusal(format!(
                "cell {c} has nEdgesOnCell={n}, outside 3..={}",
                mesh.max_edges
            )));
        }
        // MPAS: mpas_atm_advection.F:94-98,335-345. FIFTEEN bounds
        // the active stencil (centre plus neighbours), not row padding.
        if n >= FIFTEEN {
            return Err(MpasError::Refusal(format!(
                "cell {c} needs {} polynomial entries, past FIFTEEN={FIFTEEN}", n + 1
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

use crate::static_native::{arc_length,sphere_angle};

#[inline]
fn norm(v: [f32; 3]) -> f32 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

/// MPAS: core_init_atmosphere/mpas_atm_advection.F:539-564.
fn arc_bisect(a: [f32; 3], b: [f32; 3]) -> MpasResult<[f32; 3]> {
    let r = norm(a);
    let mut c = [0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]), 0.5 * (a[2] + b[2])];
    if c.iter().all(|&v| v == 0.0) {
        return Err(MpasError::Refusal(
            "operator arc_bisect received diametrically opposite vertices".to_string(),
        ));
    }
    // Native only diagnoses an exactly zero midpoint. A nonzero midpoint
    // whose squared norm underflows still proceeds through IEEE division.
    let d = norm(c);
    for v in &mut c {
        *v = r * *v / d;
    }
    Ok(c)
}

fn cell_point(mesh: &OperatorMesh<'_>, c: usize) -> [f32; 3] {
    [mesh.x_cell[c] as f32, mesh.y_cell[c] as f32, mesh.z_cell[c] as f32]
}

fn vertex_point(mesh: &OperatorMesh<'_>, v: usize) -> [f32; 3] {
    [mesh.x_vertex[v] as f32, mesh.y_vertex[v] as f32, mesh.z_vertex[v] as f32]
}

/// MPAS: core_init_atmosphere/mpas_atm_advection.F:641-743 (MIGS/ELGS).
fn inv6(mut a: [[f32; 6]; 6]) -> MpasResult<[[f32; 6]; 6]> {
    let mut index=[0,1,2,3,4,5];
    let scale=a.map(|row| row.into_iter().fold(0.0f32,|s,v|s.max(v.abs())));
    for j in 0..5 {
        let mut largest=0.0f32;
        let mut pivot=None;
        for i in j..6 {
            let ratio=a[index[i]][j].abs()/scale[index[i]];
            if ratio>largest { largest=ratio; pivot=Some(i); }
        }
        let k=pivot.ok_or_else(||MpasError::Refusal(format!("native ELGS has no defined pivot in column {j}; MPAS would use an uninitialized index")))?;
        index.swap(j,k);
        for i in j+1..6 {
            let ratio=a[index[i]][j]/a[index[j]][j];
            a[index[i]][j]=ratio;
            for k in j+1..6 { a[index[i]][k]=a[index[i]][k]-ratio*a[index[j]][k]; }
        }
    }
    // MPAS: mpas_atm_advection.F:666-675. A zero final pivot proceeds
    // through IEEE division, preserving native infinity and NaN words.
    let mut b=[[0.0f32;6];6];
    for i in 0..6 { b[i][i]=1.0; }
    for i in 0..5 { for j in i+1..6 { for k in 0..6 {
        b[index[j]][k]=b[index[j]][k]-a[index[j]][i]*b[index[i]][k];
    } } }
    let mut inverse=[[0.0f32;6];6];
    for i in 0..6 {
        inverse[5][i]=b[index[5]][i]/a[index[5]][5];
        for j in (0..5).rev() {
            inverse[j][i]=b[index[j]][i];
            for k in j+1..6 { inverse[j][i]=inverse[j][i]-a[index[j]][k]*inverse[k][i]; }
            inverse[j][i]=inverse[j][i]/a[index[j]][j];
        }
    }
    Ok(inverse)
}

/// MPAS: core_init_atmosphere/mpas_atm_advection.F:191-212,567-614.
fn polynomial_pseudoinverse(x: &[f32], y: &[f32]) -> MpasResult<Vec<[f32; FIFTEEN]>> {
    // Return six rows of B=(A^T A)^-1 A^T, padded to FIFTEEN columns.
    let m = x.len() + 1;
    if m > FIFTEEN {
        return Err(MpasError::Refusal(format!(
            "polynomial stencil has {m} points, past FIFTEEN={FIFTEEN}"
        )));
    }
    let mut a = [[0.0f32; 6]; FIFTEEN];
    a[0][0] = 1.0;
    for row in 1..m {
        let xp = x[row - 1];
        let yp = y[row - 1];
        a[row] = [1.0, xp, yp, xp * xp, xp * yp, yp * yp];
    }
    let mut ata = [[0.0f32; 6]; 6];
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
    let mut b = vec![[0.0f32; FIFTEEN]; 6];
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

/// MPAS: mpas_atm_advection.F:890-917. Native divides by the computed
/// tangent-plane area even when it is zero or underflows; retain IEEE words.
fn tangent_gradient(xp: &[f32], yp: &[f32]) -> ([f32; FIFTEEN], [f32; FIFTEEN], [f32; FIFTEEN], f32) {
    let ne = xp.len();
    let pi = 2.0 * 1.0f32.asin();
    let mut area = 0.0f32;
    let mut angle = [0.0f32; FIFTEEN];
    let mut gx = [0.0f32; FIFTEEN];
    let mut gy = [0.0f32; FIFTEEN];
    for i in 0..ne {
        let j = (i + 1) % ne;
        let dx = xp[j] - xp[i];
        let dy = yp[j] - yp[i];
        area = area + 0.25 * (xp[i] + xp[j]) * (yp[j] - yp[i])
            - 0.25 * (yp[i] + yp[j]) * (xp[j] - xp[i]);
        angle[i] = dy.atan2(dx) - pi / 2.0;
    }
    for i in 0..ne {
        let j = (i + 1) % ne;
        let dl = ((xp[j] - xp[i]).powi(2) + (yp[j] - yp[i]).powi(2)).sqrt();
        gx[i] = dl * angle[i].cos() / area;
        gy[i] = dl * angle[i].sin() / area;
    }
    (gx, gy, angle, area)
}

/// MPAS: core_init_atmosphere/mpas_atm_advection.F:21-394,745-925.
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

    let radius=mesh.sphere_radius as f32;
    let unit_north = [0.0, 0.0, 1.0];
    let pi = 2.0 * 1.0f32.asin();
    let mut deriv_two_zero_stencil_cells = 0usize;

    for cell in 0..mesh.n_cells {
        let ne = mesh.n_edges_on_cell[cell];
        if (0..ne).any(|i| mesh.cells_on_cell[cell * mesh.max_edges + i] >= mesh.n_cells) {
            deriv_two_zero_stencil_cells += 1;
            continue;
        }
        let center_phys = cell_point(mesh, cell);
        let center = [
            center_phys[0] / radius,
            center_phys[1] / radius,
            center_phys[2] / radius,
        ];

        // --- deformation/gradient tangent polygon from cell vertices ---
        let mut xp = [0.0f32; FIFTEEN];
        let mut yp = [0.0f32; FIFTEEN];
        if mesh.on_a_sphere {
            let first_v = mesh.vertices_on_cell[cell * mesh.max_edges];
            let v0p = vertex_point(mesh, first_v);
            let v0 = [
                v0p[0] / radius,
                v0p[1] / radius,
                v0p[2] / radius,
            ];
            let theta_abs = if center[2] == 1.0 {
                pi / 2.0
            } else {
                pi / 2.0 - sphere_angle(center, v0, unit_north)
            };
            let mut theta = [0.0f32; FIFTEEN];
            theta[0] = theta_abs;
            for i in 0..ne {
                let va = mesh.vertices_on_cell[cell * mesh.max_edges + i];
                let vb = mesh.vertices_on_cell[cell * mesh.max_edges + ((i + 1) % ne)];
                let ap = vertex_point(mesh, va);
                let bp = vertex_point(mesh, vb);
                let a = [ap[0] / radius, ap[1] / radius, ap[2] / radius];
                let b = [bp[0] / radius, bp[1] / radius, bp[2] / radius];
                let d = radius * arc_length(center, a);
                if i > 0 {
                    let prev = mesh.vertices_on_cell[cell * mesh.max_edges + i - 1];
                    let pp = vertex_point(mesh, prev);
                    let p = [pp[0] / radius, pp[1] / radius, pp[2] / radius];
                    theta[i] = theta[i - 1] + sphere_angle(center, p, a);
                }
                xp[i] = theta[i].cos() * d;
                yp[i] = theta[i].sin() * d;
                let _ = b; // b participates below through polygon differences
            }

        } else {
            // MPAS: mpas_atm_advection.F:883-895. Native includes the cell
            // centre as the first polygon point and uses the following
            // nEdgesOnCell-1 vertices; preserve that indexing verbatim.
            xp[0] = center_phys[0];
            yp[0] = center_phys[1];
            for i in 1..ne {
                let vertex = vertex_point(mesh, mesh.vertices_on_cell[cell * mesh.max_edges + i - 1]);
                xp[i] = vertex[0];
                yp[i] = vertex[1];
            }
        }

        let (gradient_x, gradient_y, normal_angle, area) = tangent_gradient(&xp[..ne], &yp[..ne]);
        for i in 0..ne {
            let j = (i + 1) % ne;
            let dl = ((xp[j] - xp[i]).powi(2) + (yp[j] - yp[i]).powi(2)).sqrt();
            gx[cell * mesh.max_edges + i] = gradient_x[i];
            gy[cell * mesh.max_edges + i] = gradient_y[i];

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
        let mut nx = [0.0f32; FIFTEEN];
        let mut ny = [0.0f32; FIFTEEN];
        let mut neighbor_unit = [[0.0f32; 3]; FIFTEEN];
        for i in 0..ne {
            let cn = mesh.cells_on_cell[cell * mesh.max_edges + i];
            let p = cell_point(mesh, cn);
            neighbor_unit[i] = [
                p[0] / radius,
                p[1] / radius,
                p[2] / radius,
            ];
        }
        let mut theta0 = 0.0f32;
        let mut planar_angles = [0.0f32; FIFTEEN];
        if mesh.on_a_sphere {
            let first = neighbor_unit[0];
            theta0 = if center[2] == 1.0 {
                pi / 2.0
            } else {
                pi / 2.0 - sphere_angle(center, first, unit_north)
            };
            let mut t = theta0;
            for i in 0..ne {
                if i > 0 {
                    t += sphere_angle(center, neighbor_unit[i - 1], neighbor_unit[i]);
                }
                let dist = radius * arc_length(center, neighbor_unit[i]);
                nx[i] = t.cos() * dist;
                ny[i] = t.sin() * dist;
            }
        } else {
            // MPAS: mpas_atm_advection.F:184-200. The directed edge angle
            // determines the Cartesian neighbour locations, including the
            // subtraction of native RKIND pi on the second edge side.
            for i in 0..ne {
                let edge = mesh.edges_on_cell[cell * mesh.max_edges + i];
                let mut angle = mesh.angle_edge[edge] as f32;
                if mesh.cells_on_edge[edge][0] != cell { angle -= pi; }
                planar_angles[i] = angle;
                nx[i] = mesh.dc_edge[edge] as f32 * angle.cos();
                ny[i] = mesh.dc_edge[edge] as f32 * angle.sin();
            }
        }

        let b = polynomial_pseudoinverse(&nx[..ne], &ny[..ne])?;

        for i in 0..ne {
            let edge = mesh.edges_on_cell[cell * mesh.max_edges + i];
            let edge_theta = if mesh.on_a_sphere {
                let verts = mesh.vertices_on_edge[edge];
                let va = vertex_point(mesh, verts[0]);
                let vb = vertex_point(mesh, verts[1]);
                let va = [va[0] / radius, va[1] / radius, va[2] / radius];
                let vb = [vb[0] / radius, vb[1] / radius, vb[2] / radius];
                let mid = arc_bisect(va, vb)?;
                 sphere_angle(center, neighbor_unit[i], mid) + {
                    let mut tt = theta0;
                    for k in 1..=i {
                        tt += sphere_angle(center, neighbor_unit[k - 1], neighbor_unit[k]);
                    }
                    tt
                }
            } else {
                // MPAS: mpas_atm_advection.F:360-389.
                planar_angles[i]
            };
            let c = edge_theta.cos();
            let s = edge_theta.sin();
            let cs = c*s;
            let c2 = c*c;
            let s2 = s*s;
            let side = if mesh.cells_on_edge[edge][0] == cell { 0 } else { 1 };
            let base = (edge * 2 + side) * FIFTEEN;
            for j in 0..=ne {
                let value = 2.0 * c2 * b[3][j]
                    + 2.0 * cs * b[4][j]
                    + 2.0 * s2 * b[5][j];
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
    fn arc_midpoint_underflow_matches_pristine_native_ieee_words() {
        let mut count = 0;
        for line in include_str!("../tests/native_static/arc_bisect.words").lines() {
            let fields: Vec<_> = line.split_whitespace().collect();
            let value = |i: usize| f32::from_bits(u32::from_str_radix(fields[i], 16).unwrap());
            let actual = arc_bisect([value(1), value(2), value(3)], [value(4), value(5), value(6)]).unwrap();
            for i in 0..3 {
                assert_eq!(actual[i].to_bits(), value(i + 7).to_bits(), "native arc {}/{i}", fields[0]);
                count += 1;
            }
        }
        assert_eq!(count, 672);
    }

    #[test]
    fn inverse_identity() {
        let mut a = [[0.0; 6]; 6];
        for i in 0..6 {
            a[i][i] = 1.0;
        }
        assert_eq!(inv6(a).unwrap(), a);
    }

    #[test]
    fn zero_final_pivot_matches_pristine_native_ieee_words() {
        let fits: Vec<_> = (0..32).map(|q| {
            let mut a = [[0.0;6];6];
            for i in 1..=5 {
                a[i-1][i-1] = (i+q+1) as f32;
                for j in i+1..=6 { a[i-1][j-1] = (((i*7+j*11+q)%13) as i32-6) as f32*0.125; }
            }
            inv6(a).unwrap()
        }).collect();
        let mut count = 0;
        for line in include_str!("../tests/native_static/inverse.words").lines() {
            let fields: Vec<_> = line.split_whitespace().collect();
            let q: usize = fields[0].parse().unwrap();
            let i: usize = fields[1].parse().unwrap();
            let j: usize = fields[2].parse().unwrap();
            let expected = u32::from_str_radix(fields[3],16).unwrap();
            assert_eq!(fits[q][i-1][j-1].to_bits(), expected, "native inverse {q}/{i}/{j}");
            count += 1;
        }
        assert_eq!(count,1152);
    }

    #[test]
    fn polynomial_fit_matches_pristine_native_words() {
        let mut fits=Vec::new();
        for q in 0..256 {
            let edges=5+q%4;
            let mut x=Vec::new();
            let mut y=Vec::new();
            for i in 1..=edges {
                let t=(i-1) as f32*std::f32::consts::TAU/edges as f32+q as f32*0.013;
                let r=1000.0+((q*17+i*31)%700) as f32;
                x.push(t.cos()*r); y.push(t.sin()*r);
            }
            fits.push(polynomial_pseudoinverse(&x,&y).unwrap());
        }
        let mut count=0;
        let mut differences=Vec::new();
        for line in include_str!("../tests/native_static/polynomial.words").lines() {
            let fields:Vec<_>=line.split_whitespace().collect();
            let q:usize=fields[0].parse().unwrap();
            let i:usize=fields[1].parse().unwrap();
            let j:usize=fields[2].parse().unwrap();
            let expected=u32::from_str_radix(fields[3],16).unwrap();
            let actual=fits[q][i-1][j-1].to_bits();
            if expected!=actual && differences.len()<20 { differences.push((q,i,j,expected,actual)); }
            count+=1;
        }
        assert_eq!(count,11520);
        assert!(differences.is_empty(),"native polynomial words differ: {differences:?}");
    }
    #[test]
    fn tangent_gradient_retains_native_tiny_and_zero_area_words() {
        let mut results = Vec::new();
        for q in 0..32 {
            let scale = [1.0f32, 1.0e-11, 1.0e-23, 0.0][q % 4];
            let xp = [0.0, 2.0, 3.0, 2.0, 0.0, -1.0].map(|v| v * scale);
            let mut yp = [0.0, 0.0, 1.0, 3.0, 2.0, 1.0].map(|v| v * scale);
            if q % 8 >= 4 { yp = [0.0; 6]; }
            results.push(tangent_gradient(&xp, &yp));
        }
        let mut count = 0;
        for line in include_str!("../tests/native_static/gradient.words").lines() {
            let fields: Vec<_> = line.split_whitespace().collect();
            let q: usize = fields[0].parse().unwrap();
            let i: usize = fields[1].parse::<usize>().unwrap() - 1;
            for (axis, actual) in [results[q].0[i], results[q].1[i]].into_iter().enumerate() {
                let expected = u32::from_str_radix(fields[axis + 2], 16).unwrap();
                assert_eq!(actual.to_bits(), expected, "native gradient q={q} i={i} axis={axis}");
                count += 1;
            }
        }
        assert_eq!(count, 384);
    }
}
