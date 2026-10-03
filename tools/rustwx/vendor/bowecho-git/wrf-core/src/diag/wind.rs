//! Wind diagnostic variables:
//! ua, va, wa, wspd, wdir, uvmet, uvmet10, wspd10, wdir10

use crate::compute::ComputeOpts;
use crate::error::WrfResult;
use crate::file::WrfFile;

/// U-wind destaggered (m/s). `[nz, ny, nx]`
pub fn compute_ua(f: &WrfFile, t: usize, _opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    f.u_destag(t).map(|v| v.to_vec())
}

/// V-wind destaggered (m/s). `[nz, ny, nx]`
pub fn compute_va(f: &WrfFile, t: usize, _opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    f.v_destag(t).map(|v| v.to_vec())
}

/// W-wind destaggered (m/s). `[nz, ny, nx]`
pub fn compute_wa(f: &WrfFile, t: usize, _opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    f.w_destag(t).map(|v| v.to_vec())
}

/// Latitude (degrees). `[ny, nx]`
pub fn compute_lat(f: &WrfFile, t: usize, _opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    f.xlat(t).map(|v| v.to_vec())
}

/// Longitude (degrees). `[ny, nx]`
pub fn compute_lon(f: &WrfFile, t: usize, _opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    f.xlong(t).map(|v| v.to_vec())
}

/// Rotate grid-relative (u, v) to earth-relative using SINALPHA/COSALPHA.
fn rotate_to_earth(
    u: &[f64],
    v: &[f64],
    sina: &[f64],
    cosa: &[f64],
    nxy: usize,
) -> (Vec<f64>, Vec<f64>) {
    let mut u_earth = vec![0.0; u.len()];
    let mut v_earth = vec![0.0; v.len()];

    u_earth
        .iter_mut()
        .zip(v_earth.iter_mut())
        .enumerate()
        .for_each(|(idx, (ue, ve))| {
            let ij = idx % nxy;
            let ca = cosa[ij];
            let sa = sina[ij];
            *ue = u[idx] * ca - v[idx] * sa;
            *ve = u[idx] * sa + v[idx] * ca;
        });

    (u_earth, v_earth)
}

/// Wind speed from (u, v). `[nz, ny, nx]`
pub fn compute_wspd(f: &WrfFile, t: usize, _opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let u = f.u_destag(t)?;
    let v = f.v_destag(t)?;

    Ok(u.iter()
        .zip(v.iter())
        .map(|(u, v)| (u * u + v * v).sqrt())
        .collect())
}

/// Wind direction (degrees, meteorological convention). `[nz, ny, nx]`
pub fn compute_wdir(f: &WrfFile, t: usize, _opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let u = f.u_destag(t)?;
    let v = f.v_destag(t)?;
    let sina = f.sinalpha(t)?;
    let cosa = f.cosalpha(t)?;

    let (ue, ve) = rotate_to_earth(&u, &v, &sina, &cosa, f.nxy());

    Ok(ue
        .iter()
        .zip(ve.iter())
        .map(|(u, v)| {
            let dir = 270.0 - v.atan2(*u).to_degrees();
            if dir < 0.0 {
                dir + 360.0
            } else if dir >= 360.0 {
                dir - 360.0
            } else {
                dir
            }
        })
        .collect())
}

/// Earth-rotated U/V wind. Returns interleaved `[u_earth..., v_earth...]` (2 * nxyz).
pub fn compute_uvmet(f: &WrfFile, t: usize, _opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let u = f.u_destag(t)?;
    let v = f.v_destag(t)?;
    let sina = f.sinalpha(t)?;
    let cosa = f.cosalpha(t)?;

    let (ue, ve) = rotate_to_earth(&u, &v, &sina, &cosa, f.nxy());

    let mut out = ue;
    out.extend(ve);
    Ok(out)
}

/// 10-m earth-rotated U/V wind. Returns `[u_earth..., v_earth...]` (2 * nxy).
pub fn compute_uvmet10(f: &WrfFile, t: usize, _opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let u10 = f.u10(t)?;
    let v10 = f.v10(t)?;
    let sina = f.sinalpha(t)?;
    let cosa = f.cosalpha(t)?;

    let (ue, ve) = rotate_to_earth(&u10, &v10, &sina, &cosa, f.nxy());

    let mut out = ue;
    out.extend(ve);
    Ok(out)
}

/// 10-m wind speed (m/s). `[ny, nx]`
pub fn compute_wspd10(f: &WrfFile, t: usize, _opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let u = f.u10(t)?;
    let v = f.v10(t)?;
    Ok(u.iter()
        .zip(v.iter())
        .map(|(u, v)| (u * u + v * v).sqrt())
        .collect())
}

/// 10-m wind direction (degrees). `[ny, nx]`
pub fn compute_wdir10(f: &WrfFile, t: usize, _opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let u = f.u10(t)?;
    let v = f.v10(t)?;
    let sina = f.sinalpha(t)?;
    let cosa = f.cosalpha(t)?;

    let (ue, ve) = rotate_to_earth(&u, &v, &sina, &cosa, f.nxy());

    Ok(ue
        .iter()
        .zip(ve.iter())
        .map(|(u, v)| {
            let dir = 270.0 - v.atan2(*u).to_degrees();
            if dir < 0.0 {
                dir + 360.0
            } else if dir >= 360.0 {
                dir - 360.0
            } else {
                dir
            }
        })
        .collect())
}
