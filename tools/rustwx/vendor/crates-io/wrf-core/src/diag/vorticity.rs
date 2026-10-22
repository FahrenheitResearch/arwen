//! Vorticity diagnostic variables: avo, pvo

use crate::compute::ComputeOpts;
use crate::error::WrfResult;
use crate::file::WrfFile;

const OMEGA: f64 = 7.2921159e-5; // Earth's angular velocity (rad/s)

/// Absolute vorticity (s^-1). `[nz, ny, nx]`
///
/// wrf-python's `avo` (fortran/wrf_pvo.f90 DCOMPUTEABSVORT, Apache-2.0),
/// in s^-1 rather than its 10^-5 s^-1: on the native staggered winds, each
/// face wind divided by its own map factor, centred differences (one-sided
/// at the domain edge, as wrf-python clamps), times the mass-point map
/// factor squared, plus the file's Coriolis parameter F (2 Omega sin(lat)
/// when the file carries no F):
///
///   dvdx = 0.5 (v(i+1,j)/mv + v(i+1,j+1)/mv - v(i-1,j)/mv - v(i-1,j+1)/mv) / dsx * m^2
///   dudy = 0.5 (u(i,j+1)/mu + u(i+1,j+1)/mu - u(i,j-1)/mu - u(i+1,j-1)/mu) / dsy * m^2
///   avo  = dvdx - dudy + F
///
/// Breakage it fixes (2026-10-21, WOOF vs HRRR ABSV at the same analysis):
/// the earlier body passed (v, u) to `met::dynamics::vorticity(u, v, ..)`,
/// so it returned du/dx - dv/dy -- zero for solid rotation -- and every
/// closed low vanished from the 500 hPa vorticity map while jet shear
/// zones stayed (corr 0.23 with NOAA's ABSV at 500 hPa, 0.06 at 850 hPa,
/// with the winds themselves matching to 0.3 kt).
pub fn compute_avo(f: &WrfFile, t: usize, _opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let u = f.read_var("U", t)?;
    let v = f.read_var("V", t)?;
    let (nx, ny, nz) = (f.nx, f.ny, f.nz);
    let nxy = nx * ny;
    let ones_u = vec![1.0; (nx + 1) * ny];
    let ones_v = vec![1.0; nx * (ny + 1)];
    let ones_m = vec![1.0; nxy];
    let msfu = f.read_var("MAPFAC_U", t).unwrap_or(ones_u);
    let msfv = f.read_var("MAPFAC_V", t).unwrap_or(ones_v);
    let msfm = f.read_var("MAPFAC_M", t).unwrap_or(ones_m);
    let cor = match f.read_var("F", t) {
        Ok(cor) if cor.len() == nxy => cor,
        _ => f.xlat(t)?.iter().map(|lat| 2.0 * OMEGA * lat.to_radians().sin()).collect(),
    };
    absolute_vorticity_staggered(&u, &v, &msfu, &msfv, &msfm, &cor, nx, ny, nz, f.dx, f.dy)
}

/// The arithmetic of [`compute_avo`] on plain arrays, row-major
/// `[k][j][i]` with j south to north: `u` is `[nz][ny][nx+1]`, `v`
/// `[nz][ny+1][nx]`, `msfu` `[ny][nx+1]`, `msfv` `[ny+1][nx]`, `msfm` and
/// `cor` `[ny][nx]`.
#[allow(clippy::too_many_arguments)]
pub fn absolute_vorticity_staggered(
    u: &[f64],
    v: &[f64],
    msfu: &[f64],
    msfv: &[f64],
    msfm: &[f64],
    cor: &[f64],
    nx: usize,
    ny: usize,
    nz: usize,
    dx: f64,
    dy: f64,
) -> WrfResult<Vec<f64>> {
    let nxu = nx + 1;
    let nxy = nx * ny;
    if u.len() != nz * ny * nxu || v.len() != nz * (ny + 1) * nx || msfu.len() != ny * nxu
        || msfv.len() != (ny + 1) * nx || msfm.len() != nxy || cor.len() != nxy
        || nx < 2 || ny < 2
    {
        return Err(crate::error::WrfError::DimMismatch(format!(
            "avo needs U [{nz},{ny},{nxu}], V [{nz},{},{nx}] and their map factors",
            ny + 1
        )));
    }
    let mut av = vec![0.0f64; nz * nxy];
    for k in 0..nz {
        let uk = &u[k * ny * nxu..(k + 1) * ny * nxu];
        let vk = &v[k * (ny + 1) * nx..(k + 1) * (ny + 1) * nx];
        for j in 0..ny {
            let jp1 = (j + 1).min(ny - 1);
            let jm1 = j.saturating_sub(1);
            let dsy = (jp1 - jm1) as f64 * dy;
            for i in 0..nx {
                let ip1 = (i + 1).min(nx - 1);
                let im1 = i.saturating_sub(1);
                let dsx = (ip1 - im1) as f64 * dx;
                let mm = msfm[j * nx + i] * msfm[j * nx + i];
                let uu = |jj: usize, ii: usize| uk[jj * nxu + ii] / msfu[jj * nxu + ii];
                let vv = |jj: usize, ii: usize| vk[jj * nx + ii] / msfv[jj * nx + ii];
                let dudy = 0.5 * (uu(jp1, i) + uu(jp1, i + 1) - uu(jm1, i) - uu(jm1, i + 1)) / dsy * mm;
                let dvdx = 0.5 * (vv(j, ip1) + vv(j + 1, ip1) - vv(j, im1) - vv(j + 1, im1)) / dsx * mm;
                av[k * nxy + j * nx + i] = dvdx - dudy + cor[j * nx + i];
            }
        }
    }
    Ok(av)
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

        // (u, v) in that order: vorticity(u, v) is dv/dx - du/dy.  The
        // earlier (v, u) gave du/dx - dv/dy (see compute_avo).
        let rel_vort = crate::met::dynamics::vorticity(u_plane, v_plane, nx, ny, dx, dy);

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
