//! Native-column IVT, kg m-1 s-1. No isobaric remapping before integration.
//! q = r/(1+r), with r the dry-air vapour mixing ratio. Integrate q*(u,v)*dp/g
//! over the model column: PSFC to extrapolated top half-level, clipped at 0 Pa.
//! Interior pressure interfaces are midpoints. The lowest/top values are held
//! constant to their interfaces. This is model-column IVT, not a 1000-300 hPa
//! truncation or a claim of transport above the model top. Missing, negative
//! humidity, nonfinite, or nonmonotone columns produce NaN in all three planes.
use rayon::prelude::*;
use wrf_core::{WrfFile, ComputeOpts, getvar};
const G: f64 = 9.80665;

pub(crate) fn integrate(p: &[f64], r: &[f64], u: &[f64], v: &[f64], ps: &[f64], nz: usize, cells: usize)
    -> Result<[Vec<f32>; 3], String> {
    let n = nz.checked_mul(cells).ok_or("IVT column shape overflow")?;
    if nz < 2 || ps.len() != cells || [p.len(),r.len(),u.len(),v.len()].iter().any(|&len| len != n) {
        return Err("IVT needs at least two pressure levels, matching qv/earth-relative u/v, and PSFC".into());
    }
    let columns: Vec<[f32;3]> = (0..cells).into_par_iter().map(|c| {
        let invalid = [f32::NAN;3];
        if !ps[c].is_finite() || ps[c] < p[c] { return invalid; }
        let mut east = 0.0;
        let mut north = 0.0;
        for k in 0..nz {
            let i = k*cells+c;
            if !p[i].is_finite() || p[i] <= 0.0 || !r[i].is_finite() || r[i] < 0.0
                || !u[i].is_finite() || !v[i].is_finite()
                || (k > 0 && p[i-cells] <= p[i]) { return invalid; }
            let bottom = if k == 0 { ps[c] } else { (p[i-cells]+p[i])*0.5 };
            let top = if k+1 == nz { (p[i] - (p[i-cells]-p[i])*0.5).max(0.0) }
                      else { (p[i]+p[i+cells])*0.5 };
            let mass = r[i]/(1.0+r[i]) * (bottom-top)/G;
            east += mass*u[i]; north += mass*v[i];
        }
        [east.hypot(north) as f32, east as f32, north as f32]
    }).collect();
    Ok(std::array::from_fn(|i| columns.iter().map(|c| c[i]).collect()))
}

pub(crate) fn read(file: &WrfFile, timeidx: usize) -> Result<[Vec<f32>;3], String> {
    let cells = file.nx*file.ny;
    let nz = file.nz;
    let p = file.full_pressure(timeidx).map_err(|e| format!("IVT pressure: {e}"))?;
    let r = file.read_var("QVAPOR",timeidx).map_err(|e| format!("IVT QVAPOR: {e}"))?;
    let ps = file.read_var("PSFC",timeidx).map_err(|e| format!("IVT PSFC: {e}"))?;
    let wind = getvar(file,"uvmet",Some(timeidx),&ComputeOpts::default()).map_err(|e| format!("IVT uvmet: {e}"))?;
    let n = nz*cells;
    let wind = crate::wrf_volumes::validate_earth_relative_uvmet(wind,nz,file.ny,file.nx,n)?;
    integrate(&p,&r,&wind[..n],&wind[n..],&ps,nz,cells)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn analytic_constant_humidity_and_wind_uses_specific_humidity_and_pa() {
        let result=integrate(&[90000.,70000.,50000.],&[0.01;3],&[3.;3],&[4.;3],&[100000.],3,1).unwrap();
        let mass=0.01/1.01*60000./G;
        for (plane,expected) in result.iter().zip([5.*mass,3.*mass,4.*mass]) {
            assert!((plane[0] as f64-expected).abs()<0.0001);
        }
    }
    #[test]
    fn opposing_layers_cancel_before_magnitude_and_invalid_columns_stay_missing() {
        let result=integrate(&[90000.,70000.],&[0.01;2],&[10.,-10.],&[0.;2],&[100000.],2,1).unwrap();
        assert_eq!(result[0][0],0.);
        for bad in [vec![90000.,95000.],vec![90000.,f64::NAN]] {
            assert!(integrate(&bad,&[0.01;2],&[1.;2],&[0.;2],&[100000.],2,1).unwrap().iter().all(|p|p[0].is_nan()));
        }
        assert!(integrate(&[90000.,70000.],&[0.01,-0.1],&[1.;2],&[0.;2],&[100000.],2,1).unwrap()[0][0].is_nan());
    }
    #[test]
    fn parallel_columns_keep_components_and_reject_bad_shape() {
        let result=integrate(&[90000.,90000.,70000.,70000.],&[0.01;4],&[3.,-3.,3.,-3.],&[4.,4.,4.,4.],&[100000.;2],2,2).unwrap();
        assert_eq!(result[0][0],result[0][1]); assert_eq!(result[1][0],-result[1][1]);
        assert!(integrate(&[],&[],&[],&[],&[],1,1).is_err());
    }
}
