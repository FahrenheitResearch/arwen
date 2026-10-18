//! MPAS v8.4.2 mpas_init_atm_llxy.F:1914-2222: the complete Gaussian
//! projection used by init_atm_case_lbc (mpas_init_atm_cases.F:5741-5752).
//! The latitude solver uses REAL64, but several unsuffixed constants and
//! the normalization square roots are REAL(RKIND). Preserve those conversions.

use crate::error::{MpasError, MpasResult};

#[derive(Debug)]
pub struct Gaussian {
    latitudes: Vec<f32>,
    lon1: f32,
    nx: usize,
    ny: usize,
}

// The unsuffixed Fortran parameter is rounded to REAL(4) before REAL64.
const PI64: f64 = std::f32::consts::PI as f64;

/// mpas_init_atm_llxy.F:2095-2140 (lgord).
fn lgord(cosc: f64, n: usize) -> f64 {
    let colat = cosc.acos();
    let mut c1 = 2.0f64.sqrt();
    for k in 1..=n {
        c1 *= (1.0f32 - 1.0f32 / (4 * k * k) as f32).sqrt() as f64;
    }
    let nf = n as f64;
    let mut ang = nf * colat;
    let (mut s1, mut c4, mut a, mut b) = (0.0, 1.0, -1.0, 0.0);
    for k in (0..=n).step_by(2) {
        if k == n {
            c4 = 0.5 * c4;
        }
        s1 += c4 * ang.cos();
        a += 2.0;
        b += 1.0;
        ang = colat * (nf - k as f64 - 2.0);
        c4 = (a * (nf - b + 1.0) / (b * (nf + nf - a))) * c4;
    }
    s1 * c1
}

/// mpas_init_atm_llxy.F:1956-2092 (gausll and lggaus).
pub fn gaussian_latitudes(nlat: usize) -> Vec<f32> {
    let fi = nlat as f64;
    let fi1 = fi + 1.0;
    let a = fi * fi1 / (4.0 * fi1 * fi1 - 1.0).sqrt();
    let b = fi1 * fi / (4.0 * fi * fi - 1.0).sqrt();
    let mut latitudes = vec![0.0; nlat];
    for i in 1..=nlat / 2 {
        let mut cosc = (((i as f32 - 0.5) as f64) * PI64 / fi + PI64 * 0.5).sin();
        loop {
            let g = lgord(cosc, nlat);
            let gm = lgord(cosc, nlat - 1);
            let gp = lgord(cosc, nlat + 1);
            let gt = (cosc * cosc - 1.0) / (a * gp - b * gm);
            let delta = g * gt;
            cosc -= delta;
            if delta.abs() <= 1.0e-14f32 as f64 {
                break;
            }
        }
        let sinc = cosc.acos().sin();
        let lat = (sinc.acos() * 180.0 / PI64) as f32;
        latitudes[i - 1] = lat;
        latitudes[nlat - i] = -lat;
    }
    latitudes
}

impl Gaussian {
    /// mpas_init_atm_llxy.F:1914-1953 (set_gauss).
    pub fn new(north_count: f32, lat1: f32, lon1: f32, nx: usize, ny: usize) -> MpasResult<Self> {
        let nlat = (north_count.round() as usize).checked_mul(2).unwrap_or(0);
        // mpas_init_atm_cases.F:5813-5822 reads columns nx-2 and 3
        // for periodic halos. A narrower slab would index outside its data.
        if nlat < 2 || nlat != ny || nx < 3 {
            return Err(MpasError::Refusal(format!(
                "Gaussian slab has {ny} rows and hemisphere count {north_count}; \
                 case 9 requires two hemispheres and at least three longitude halo columns"
            )));
        }
        let mut latitudes = gaussian_latitudes(nlat);
        if (latitudes[0] - lat1).abs() > 0.01 {
            for v in &mut latitudes {
                *v = -1.0 * *v;
            }
        }
        if (latitudes[0] - lat1).abs() > 0.01 {
            return Err(MpasError::Refusal(format!(
                "Gaussian starting latitude {lat1} differs from computed {} by more than \
                 MPAS set_gauss's 0.01 degrees; sampling would use the wrong latitude grid",
                latitudes[0]
            )));
        }
        Ok(Self {
            latitudes,
            lon1,
            nx,
            ny,
        })
    }

    /// mpas_init_atm_llxy.F:2143-2222 (llij_gauss), followed by
    /// mpas_init_atm_cases.F:5826-5841 (case-9 wrap and polar clamp).
    pub fn locate(&self, lat: f32, mut lon: f32) -> (f32, f32) {
        let loninc = 360.0 / self.nx as f32;
        let mut x = (lon - self.lon1) / loninc + 1.0;
        if x < 0.5 {
            lon += 360.0;
            x = (lon - self.lon1) / loninc + 1.0;
        } else if x >= self.nx as f32 + 0.5 {
            lon -= 360.0;
            x = (lon - self.lon1) / loninc + 1.0;
        }
        let mut y = if lat.abs() > self.latitudes[0].abs() {
            if (lat - self.latitudes[0]).abs() < (lat - self.latitudes[self.ny - 1]).abs() {
                1.0
            } else {
                self.ny as f32
            }
        } else {
            let n = self
                .latitudes
                .windows(2)
                .position(|p| (p[0] - lat) * (p[1] - lat) <= 0.0)
                .expect("Gaussian bounding latitudes");
            ((self.latitudes[n] - lat) * (n + 2) as f32
                + (lat - self.latitudes[n + 1]) * (n + 1) as f32)
                / (self.latitudes[n] - self.latitudes[n + 1])
        };
        if y < 0.5 {
            y = 1.0;
        } else if y >= self.ny as f32 + 0.5 {
            y = self.ny as f32;
        }
        (x, y)
    }
}
