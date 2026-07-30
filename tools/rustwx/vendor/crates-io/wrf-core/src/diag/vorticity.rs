//! Vorticity diagnostic variables: avo, pvo

use crate::compute::ComputeOpts;
use crate::error::WrfResult;
use crate::file::WrfFile;

const OMEGA: f64 = 7.2921159e-5; // Earth's angular velocity (rad/s)

/// Absolute vorticity (s^-1). `[nz, ny, nx]`
///
/// AVO = relative_vorticity + coriolis_parameter
/// = (dv/dx - du/dy) + 2*Omega*sin(lat)
pub fn compute_avo(f: &WrfFile, t: usize, _opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let u = f.u_destag(t)?;
    let v = f.v_destag(t)?;
    let lat = f.xlat(t)?;

    let nx = f.nx;
    let ny = f.ny;
    let nz = f.nz;
    let nxy = nx * ny;
    let dx = f.dx;
    let dy = f.dy;

    let mut avo = vec![0.0f64; nz * nxy];

    avo.chunks_mut(nxy).enumerate().for_each(|(k, plane)| {
        let u_plane = &u[k * nxy..(k + 1) * nxy];
        let v_plane = &v[k * nxy..(k + 1) * nxy];

        // Compute relative vorticity using central differences
        let rel_vort = crate::met::dynamics::vorticity(v_plane, u_plane, nx, ny, dx, dy);

        for ij in 0..nxy {
            let f_cor = 2.0 * OMEGA * (lat[ij].to_radians()).sin();
            plane[ij] = rel_vort[ij] + f_cor;
        }
    });

    Ok(avo)
}

/// Potential vorticity (PVU, 1 PVU = 10^-6 K m^2 kg^-1 s^-1). `[nz, ny, nx]`
///
/// PVO = -g * (f + zeta) * (dtheta/dp)
pub fn compute_pvo(f: &WrfFile, t: usize, _opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let u = f.u_destag(t)?;
    let v = f.v_destag(t)?;
    let theta = f.full_theta(t)?;
    let pres = f.full_pressure(t)?;
    let lat = f.xlat(t)?;

    let nx = f.nx;
    let ny = f.ny;
    let nz = f.nz;
    let nxy = nx * ny;
    let dx = f.dx;
    let dy = f.dy;

    let g = 9.80665;

    let mut pvo = vec![0.0f64; nz * nxy];

    pvo.chunks_mut(nxy).enumerate().for_each(|(k, plane)| {
        let u_plane = &u[k * nxy..(k + 1) * nxy];
        let v_plane = &v[k * nxy..(k + 1) * nxy];

        let rel_vort = crate::met::dynamics::vorticity(v_plane, u_plane, nx, ny, dx, dy);

        for ij in 0..nxy {
            let f_cor = 2.0 * OMEGA * (lat[ij].to_radians()).sin();
            let abs_vort = rel_vort[ij] + f_cor;

            // dtheta/dp using centered differences in the vertical
            let dtheta_dp = if k == 0 {
                let idx0 = ij;
                let idx1 = nxy + ij;
                (theta[idx1] - theta[idx0]) / (pres[idx1] - pres[idx0])
            } else if k == nz - 1 {
                let idx0 = (k - 1) * nxy + ij;
                let idx1 = k * nxy + ij;
                (theta[idx1] - theta[idx0]) / (pres[idx1] - pres[idx0])
            } else {
                let idx_b = (k - 1) * nxy + ij;
                let idx_t = (k + 1) * nxy + ij;
                (theta[idx_t] - theta[idx_b]) / (pres[idx_t] - pres[idx_b])
            };

            // PVO = -g * abs_vort * dtheta/dp, convert to PVU (*1e6)
            plane[ij] = -g * abs_vort * dtheta_dp * 1e6;
        }
    });

    Ok(pvo)
}
