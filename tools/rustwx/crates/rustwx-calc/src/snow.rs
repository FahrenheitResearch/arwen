//! Snow diagnostics in mm. SLR uses saved interval ends, not every model step.
use rayon::prelude::*;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SnowMethod {
    LiquidEquivalent,
    TenToOne,
    Kuchera,
}
pub const EXCEEDANCE_MM: [f64; 4] = [25.4, 101.6, 203.2, 304.8];
/// NOAA forecast-guide Kuchera diagnostic with nonnegative floor.
pub fn kuchera_slr(t: f32) -> f32 {
    if !t.is_finite() {
        return f32::NAN;
    }
    let d = 271.16 - t;
    (if t > 271.16 { 12.0 + 2.0 * d } else { 12.0 + d }).max(0.0)
}
/// Level-major native mass levels, surface through 500 hPa inclusive.
pub fn column_max(t: &[f64], p: &[f64], nz: usize, cells: usize) -> Result<Vec<f32>, String> {
    if nz == 0 || cells == 0 || nz.checked_mul(cells) != Some(t.len()) || p.len() != t.len() {
        return Err("temperature/pressure must have the same nonempty [nz,ny,nx] shape".into());
    }
    Ok((0..cells)
        .into_par_iter()
        .map(|i| {
            let mut best = f64::NEG_INFINITY;
            for k in 0..nz {
                let p = p[k * cells + i];
                if !p.is_finite() || p <= 0.0 || p > 1.0e30 {
                    return f32::NAN;
                }
                if p < 50000.0 {
                    break;
                }
                let t = t[k * cells + i];
                if !t.is_finite() || t <= 0.0 || t > 1.0e30 {
                    return f32::NAN;
                }
                best = best.max(t);
            }
            if best.is_finite() {
                best as f32
            } else {
                f32::NAN
            }
        })
        .collect())
}
pub fn accumulate(
    a: &mut [f32],
    prev: &[f32],
    curr: &[f32],
    t: &[f32],
    method: SnowMethod,
) -> Result<(), String> {
    if prev.len() != a.len()
        || curr.len() != a.len()
        || (method == SnowMethod::Kuchera && t.len() != a.len())
    {
        return Err("snowfall interval planes have different lengths".into());
    }
    a.par_iter_mut().enumerate().for_each(|(i, a)| {
        let d = curr[i] - prev[i];
        let ratio = match method {
            SnowMethod::LiquidEquivalent => 1.0,
            SnowMethod::TenToOne => 10.0,
            SnowMethod::Kuchera => kuchera_slr(t[i]),
        };
        if prev[i].is_finite()
            && prev[i] >= 0.0
            && d.is_finite()
            && d >= 0.0
            && ratio.is_finite()
            && a.is_finite()
        {
            *a += d * ratio;
        } else {
            *a = f32::NAN;
        }
    });
    Ok(())
}
/// SNOWH is standing depth in m. Negative changes retain melt and settling.
pub fn depth_change_mm(c: &[f32], b: &[f32]) -> Result<Vec<f32>, String> {
    if c.len() != b.len() {
        return Err("snow depth planes have different lengths".into());
    }
    Ok(c.par_iter()
        .zip(b)
        .map(|(&c, &b)| {
            if c.is_finite() && b.is_finite() && c >= 0.0 && b >= 0.0 {
                (c - b) * 1000.0
            } else {
                f32::NAN
            }
        })
        .collect())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn intervals_and_resets() {
        let mut a = vec![0.; 3];
        accumulate(
            &mut a,
            &[2.; 3],
            &[3., 1., f32::NAN],
            &[272.16; 3],
            SnowMethod::Kuchera,
        )
        .unwrap();
        assert!((a[0] - 10.).abs() < 0.001);
        assert!(a[1].is_nan() && a[2].is_nan());
        accumulate(
            &mut a,
            &[3., 1., 0.],
            &[4., 2., 1.],
            &[269.16; 3],
            SnowMethod::Kuchera,
        )
        .unwrap();
        assert!((a[0] - 24.).abs() < 0.001);
        assert!(a[1].is_nan() && a[2].is_nan());
    }
    #[test]
    fn signed_depth() {
        let a = depth_change_mm(&[0.15, 0.3, f32::NAN], &[0.2, 0.1, 0.]).unwrap();
        assert!((a[0] + 50.).abs() < 0.001 && (a[1] - 200.).abs() < 0.001);
        assert!(a[2].is_nan());
    }
}
