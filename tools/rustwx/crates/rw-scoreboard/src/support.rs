use crate::{Result, fail};
use obs_regrid::{KdTree, unit_vectors};

pub struct Locator {
    pub nx: usize,
    pub ny: usize,
    pub lat: Vec<f64>,
    pub lon: Vec<f64>,
    tree: KdTree,
}
#[derive(Clone, Debug)]
pub struct Stencil {
    pub indices: [usize; 4],
    pub weights: [f64; 4],
}
fn wrap(d: f64) -> f64 {
    (d + 180.0).rem_euclid(360.0) - 180.0
}
impl Locator {
    pub fn new(nx: usize, ny: usize, lat: Vec<f64>, lon: Vec<f64>) -> Result<Self> {
        if nx < 2 || ny < 2 || nx.checked_mul(ny) != Some(lat.len()) || lat.len() != lon.len() {
            return Err(fail(
                "bilinear interpolation requires a rectangular grid with at least 2 x 2 cells",
            ));
        }
        if lat.iter().chain(lon.iter()).any(|v| !v.is_finite()) {
            return Err(fail("nonfinite native grid coordinates"));
        }
        let tree = KdTree::build(unit_vectors(&lat, &lon)?);
        Ok(Self {
            nx,
            ny,
            lat,
            lon,
            tree,
        })
    }
    pub fn locate(&self, lat: f64, lon: f64) -> Option<Stencil> {
        let q = unit_vectors(&[lat], &[lon]).ok()?[0];
        let near = self.tree.nearest(q, 4.1)?.index;
        let ni = near % self.nx;
        let nj = near / self.nx;
        for j in nj.saturating_sub(1)..=nj.min(self.ny - 2) {
            for i in ni.saturating_sub(1)..=ni.min(self.nx - 2) {
                let ids = [
                    j * self.nx + i,
                    j * self.nx + i + 1,
                    (j + 1) * self.nx + i,
                    (j + 1) * self.nx + i + 1,
                ];
                // Invert the native latitude/longitude quadrilateral. Longitude
                // differences are local wrapped values, including dateline cells.
                let x = ids.map(|k| wrap(self.lon[k] - lon));
                let y = ids.map(|k| self.lat[k] - lat);
                let (mut u, mut v) = (0.5, 0.5);
                for _ in 0..12 {
                    let weights = [(1.0 - u) * (1.0 - v), u * (1.0 - v), (1.0 - u) * v, u * v];
                    let fx = (0..4).map(|k| weights[k] * x[k]).sum::<f64>();
                    let fy = (0..4).map(|k| weights[k] * y[k]).sum::<f64>();
                    let xu = (x[1] - x[0]) * (1.0 - v) + (x[3] - x[2]) * v;
                    let xv = (x[2] - x[0]) * (1.0 - u) + (x[3] - x[1]) * u;
                    let yu = (y[1] - y[0]) * (1.0 - v) + (y[3] - y[2]) * v;
                    let yv = (y[2] - y[0]) * (1.0 - u) + (y[3] - y[1]) * u;
                    let det = xu * yv - xv * yu;
                    if det.abs() < 1e-16 {
                        break;
                    }
                    let du = (fx * yv - fy * xv) / det;
                    let dv = (fy * xu - fx * yu) / det;
                    u -= du;
                    v -= dv;
                    if du.abs() + dv.abs() < 1e-12 {
                        break;
                    }
                }
                if u >= -1e-7 && u <= 1.0 + 1e-7 && v >= -1e-7 && v <= 1.0 + 1e-7 {
                    u = u.clamp(0.0, 1.0);
                    v = v.clamp(0.0, 1.0);
                    let weights = [(1.0 - u) * (1.0 - v), u * (1.0 - v), (1.0 - u) * v, u * v];
                    let errx = (0..4).map(|k| weights[k] * x[k]).sum::<f64>();
                    let erry = (0..4).map(|k| weights[k] * y[k]).sum::<f64>();
                    if errx.abs() + erry.abs() < 1e-6 {
                        return Some(Stencil {
                            indices: ids,
                            weights,
                        });
                    }
                }
            }
        }
        None
    }
}
pub fn sample(values: &[f64], s: &Stencil) -> Option<f64> {
    let mut out = 0.0;
    for k in 0..4 {
        let v = *values.get(s.indices[k])?;
        if !v.is_finite() {
            return None;
        }
        out += v * s.weights[k];
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bilinear_and_domain_are_exact() {
        let l = Locator::new(2, 2, vec![0., 0., 1., 1.], vec![179., -179., 179., -179.]).unwrap();
        let s = l.locate(0.25, 179.5).unwrap();
        assert!((sample(&[0., 2., 4., 6.], &s).unwrap() - 1.5).abs() < 1e-10);
        assert!(l.locate(2., 179.).is_none());
    }
    #[test]
    fn missing_is_not_zero() {
        let s = Stencil {
            indices: [0, 1, 2, 3],
            weights: [0.25; 4],
        };
        assert!(sample(&[1., 2., f64::NAN, 3.], &s).is_none());
    }
}
