//! Thompson GOCART initialization, including the LBC reuse path.
//! MPAS: src/core_init_atmosphere/mpas_init_atm_thompson_aerosols.F:36-865.
//! RKIND is binary32; only Gaussian latitude construction uses binary64.

use super::{
    hinterp::{self, InterpContext, Method, Slab, Underflow},
    metfile::{MetReader, MetSlab},
    soil,
    vinterp::{vertical_interp, Extrap},
    DEG_PER_RAD,
};
use crate::error::{MpasError, MpasResult};
use std::cell::Cell;
use std::path::Path;

pub const NATIVE_FILENAME: &str = "QNWFA_QNIFA_SIGMA_MONTHLY.dat";
const MONTHS: [&str; 12] = [
    "JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC",
];
const METHODS: [Method; 4] = [
    Method::FourPoint,
    Method::WtAverage4,
    Method::WtAverage16,
    Method::Search,
];

#[derive(Debug, Default, serde::Serialize)]
pub struct AerosolReceipt {
    pub source: String,
    pub source_present: bool,
    pub records_read: usize,
    pub records_interpolated: usize,
    pub masked_cells: usize,
    pub n_gocart_levels: usize,
    pub values_computed: usize,
}

/// Layout: month outermost, then cell, with GOCART level fastest.
#[derive(Debug, Clone)]
pub struct Climatology {
    pub n_cells: usize,
    pub n_levels: usize,
    pub nifa: Vec<Vec<f32>>,
    pub nwfa: Vec<Vec<f32>>,
    pub pressure: Vec<Vec<f32>>,
    // get_julgmt DATA mmd keeps February=29 after the first leap-year call.
    february_is_29: Cell<bool>,
}

#[derive(Debug)]
pub struct AerosolState {
    /// Cell-major, model vertical level fastest, as the init stream carries it.
    pub nifa: Vec<f32>,
    pub nwfa: Vec<f32>,
}

/// MPAS: mpas_init_atm_thompson_aerosols.F:360-383,673-689.
/// Aerosols wrap with strict `>` and do NOT clamp latitude like case 7.
fn locate(field: &MetSlab, lat: f32, lon: f32, gaussian: Option<&[f32]>) -> MpasResult<(f32, f32)> {
    let loninc = if gaussian.is_some() {
        360.0 / field.nx as f32
    } else {
        field.delta_lon
    };
    let ij = |lon: f32| -> MpasResult<(f32, f32)> {
        let x = (lon - field.start_lon) / loninc + 1.0;
        let y = if let Some(lats) = gaussian {
            gaussian_j(lats, lat)?
        } else {
            (lat - field.start_lat) / field.delta_lat + 1.0
        };
        Ok((x, y))
    };
    let (x, y) = ij(lon)?;
    if x < 0.5 {
        ij(lon + 360.0)
    } else if x > field.nx as f32 + 0.5 {
        ij(lon - 360.0)
    } else {
        Ok((x, y))
    }
}

/// MPAS: mpas_init_atm_llxy.F:2095-2140 (lgord).
fn lgord(cosc: f64, n: usize) -> f64 {
    let colat = cosc.acos();
    let mut c1 = 2.0_f64.sqrt();
    for k in 1..=n {
        // This subexpression is default REAL in MPAS, even in the REAL64 routine.
        c1 *= (1.0_f32 - 1.0_f32 / (4 * k * k) as f32).sqrt() as f64;
    }
    let fn_ = n as f64;
    let mut ang = fn_ * colat;
    let (mut s1, mut c4, mut a, mut b) = (0.0, 1.0, -1.0, 0.0);
    for k in (0..=n).step_by(2) {
        if k == n {
            c4 *= 0.5;
        }
        s1 += c4 * ang.cos();
        a += 2.0;
        b += 1.0;
        ang = colat * (fn_ - k as f64 - 2.0);
        c4 = (a * (fn_ - b + 1.0) / (b * (fn_ + fn_ - a))) * c4;
    }
    s1 * c1
}

/// MPAS: mpas_init_atm_llxy.F:1914-2092 (set_gauss, gausll, lggaus).
/// Weight/colat outputs unused by gausll are omitted; root operations keep order.
pub fn gaussian_latitudes(nlat: usize, start_lat: f32) -> MpasResult<Vec<f32>> {
    if nlat < 2 || nlat % 2 != 0 {
        return Err(MpasError::Refusal(
            "Gaussian input requires two hemispheres of latitude rows".into(),
        ));
    }
    // Both MPAS REAL64 pi parameters are initialized by a default REAL literal.
    let fi = nlat as f64;
    let fi1 = fi + 1.0;
    let a = fi * fi1 / (4.0 * fi1 * fi1 - 1.0).sqrt();
    let b = fi1 * fi / (4.0 * fi * fi - 1.0).sqrt();
    let mut lats = vec![0.0; nlat];
    for i in 1..=nlat / 2 {
        // i-0.5 is default REAL before multiplication by pi.
        let mut root = (((i as f32 - 0.5) as f64) * (std::f32::consts::PI as f64) / fi
            + (std::f32::consts::PI as f64) * 0.5)
            .sin();
        loop {
            let g = lgord(root, nlat);
            let gm = lgord(root, nlat - 1);
            let gp = lgord(root, nlat + 1);
            let gt = (root * root - 1.0) / (a * gp - b * gm);
            let delta = g * gt;
            root -= delta;
            if delta.abs() <= 1.0e-14_f32 as f64 {
                break;
            }
        }
        let sinc = root.acos().sin();
        let lat = (sinc.acos() * 180.0 / (std::f32::consts::PI as f64)) as f32;
        lats[i - 1] = lat;
        lats[nlat - i] = -lat;
    }
    if (lats[0] - start_lat).abs() > 0.01 {
        for lat in &mut lats {
            *lat *= -1.0;
        }
    }
    if (lats[0] - start_lat).abs() > 0.01 {
        return Err(MpasError::Refusal(format!("Gaussian starting latitude {start_lat} disagrees with MPAS-computed {} by more than MPAS's 0.01 degree check", lats[0])));
    }
    Ok(lats)
}

/// MPAS: mpas_init_atm_llxy.F:2143-2222 (llij_gauss).
fn gaussian_j(lats: &[f32], lat: f32) -> MpasResult<f32> {
    if lat.abs() > lats[0].abs() {
        return Ok(
            if (lat - lats[0]).abs() < (lat - lats[lats.len() - 1]).abs() {
                1.0
            } else {
                lats.len() as f32
            },
        );
    }
    for n in 0..lats.len() - 1 {
        if (lats[n] - lat) * (lats[n + 1] - lat) <= 0.0 {
            return Ok(
                ((lats[n] - lat) * (n + 2) as f32 + (lat - lats[n + 1]) * (n + 1) as f32)
                    / (lats[n] - lats[n + 1]),
            );
        }
    }
    Err(MpasError::Refusal(
        "MPAS Gaussian grid has no bounding latitudes for this cell".into(),
    ))
}

/// MPAS: mpas_init_atm_thompson_aerosols.F:317-359,389-662.
fn destination(name: &str) -> Option<(usize, usize)> {
    for (family, prefix) in ["QNIFA_", "QNWFA_", "P_WIF_"].iter().enumerate() {
        for (month, suffix) in MONTHS.iter().enumerate() {
            if name.contains(&format!("{prefix}{suffix}")) {
                return Some((family, month));
            }
        }
    }
    None
}

/// MPAS: mpas_init_atm_thompson_aerosols.F:209-719 (init_hinterp_gocart).
/// LANDSEA is read by native but maskval=-1 means it never affects sampling.
pub fn horizontal(
    path: &Path,
    lat: &[f32],
    lon: &[f32],
    landmask: &[i32],
    n_levels: usize,
) -> MpasResult<(Climatology, AerosolReceipt)> {
    let n_cells = lat.len();
    if lon.len() != n_cells || landmask.len() != n_cells || n_levels < 2 {
        return Err(MpasError::Refusal(
            "GOCART mesh array shapes or level count are inconsistent".into(),
        ));
    }
    let mut clim = Climatology {
        n_cells,
        n_levels,
        nifa: vec![vec![0.0; n_cells * n_levels]; 12],
        nwfa: vec![vec![0.0; n_cells * n_levels]; 12],
        pressure: vec![vec![0.0; n_cells * n_levels]; 12],
        february_is_29: Cell::new(false),
    };
    let mut receipt = AerosolReceipt {
        source: path.display().to_string(),
        source_present: true,
        n_gocart_levels: n_levels,
        masked_cells: landmask.iter().filter(|&&m| m == -1).count(),
        ..Default::default()
    };
    let mut reader = MetReader::open(path, true)?;
    while let Some((field, projection)) = reader.next_field()? {
        receipt.records_read += 1;
        let Some((family, month)) = destination(&field.field) else {
            continue;
        };
        let k = field.xlvl as i32; // Fortran REAL-to-INTEGER truncates.
        if k < 1 || k as usize > n_levels || field.nx < 3 {
            return Err(MpasError::Refusal(format!("GOCART field {} has level {} or grid width {} outside its allocated level/halo bounds", field.field, field.xlvl, field.nx)));
        }
        let gaussian = if projection == 4 {
            let nlat = field.delta_lat.round() as usize * 2;
            if nlat != field.ny {
                return Err(MpasError::Refusal(
                    "Gaussian header nlat does not match slab rows".into(),
                ));
            }
            Some(gaussian_latitudes(nlat, field.start_lat)?)
        } else {
            None
        };
        let slab = Slab::from_met(&field.values, field.nx, field.ny);
        let ctx = InterpContext {
            array: &slab,
            msgval: 0.0,
            maskval: None,
            mask: None,
            underflow: Underflow::Preserve,
            underflow_hits: None,
        };
        let dest = match family {
            0 => &mut clim.nifa,
            1 => &mut clim.nwfa,
            _ => &mut clim.pressure,
        };
        for c in 0..n_cells {
            let value = if landmask[c] == -1 {
                0.0
            } else {
                let (x, y) = locate(
                    &field,
                    lat[c] * DEG_PER_RAD,
                    lon[c] * DEG_PER_RAD,
                    gaussian.as_deref(),
                )?;
                hinterp::interp_sequence(x, y, &ctx, &METHODS, 0)
            };
            dest[month][c * n_levels + k as usize - 1] = value;
        }
        receipt.records_interpolated += 1;
    }
    Ok((clim, receipt))
}

/// MPAS: mpas_init_atm_thompson_aerosols.F:87-205 (init_vinterp_gocart).
/// Also the identical interpolation slice of init_atm_thompson_aerosols_lbc:790-844.
impl Climatology {
    pub fn at_date(
        &self,
        date: &str,
        pressure_base: &[f32],
        n_vert: usize,
    ) -> MpasResult<AerosolState> {
        let parse = |range: std::ops::Range<usize>| -> MpasResult<u32> {
            date.get(range).and_then(|s| s.parse().ok()).ok_or_else(|| {
                MpasError::Refusal(format!("invalid GOCART interpolation date {date}"))
            })
        };
        let (year, month, day) = (parse(0..4)? as i32, parse(5..7)?, parse(8..10)?);
        let leap = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
        let days = [
            31,
            if leap { 29 } else { 28 },
            31,
            30,
            31,
            30,
            31,
            31,
            30,
            31,
            30,
            31,
        ];
        if !(1..=12).contains(&month) || day < 1 || day > days[(month - 1) as usize] {
            return Err(MpasError::Refusal(format!(
                "invalid GOCART interpolation date {date}"
            )));
        }
        let size = self.n_cells * self.n_levels;
        if self.n_levels < 2
            || n_vert == 0
            || pressure_base.len() != self.n_cells * n_vert
            || [&self.nifa, &self.nwfa, &self.pressure]
                .iter()
                .any(|a| a.len() != 12 || a.iter().any(|v| v.len() != size))
        {
            return Err(MpasError::Refusal(
                "GOCART climatology and pressure_base have inconsistent array shapes".into(),
            ));
        }
        self.february_is_29.set(self.february_is_29.get() || leap);
        let native_leap = self.february_is_29.get();
        let p = soil::monthly_interp_with_february(&self.pressure, year, month, day, native_leap);
        let interpolate = |monthly: &[Vec<f32>]| {
            let v = soil::monthly_interp_with_february(monthly, year, month, day, native_leap);
            let mut out = vec![0.0; pressure_base.len()];
            for c in 0..self.n_cells {
                let offset = c * self.n_levels;
                let coords: Vec<_> = p[offset + 1..offset + self.n_levels]
                    .iter()
                    .rev()
                    .copied()
                    .collect();
                let values: Vec<_> = v[offset + 1..offset + self.n_levels]
                    .iter()
                    .rev()
                    .copied()
                    .collect();
                for k in (0..n_vert).rev() {
                    let target = pressure_base[c * n_vert + k];
                    // Constant extrapolation also supports a one-level retained column.
                    out[c * n_vert + k] = if coords.len() == 1 {
                        values[0]
                    } else {
                        vertical_interp(target, &coords, &values, Extrap::Constant)
                            .expect("constant extrapolation cannot fail")
                    };
                    if target >= p[offset] && k + 1 < n_vert {
                        out[c * n_vert + k] = out[c * n_vert + k + 1];
                    }
                }
            }
            out
        };
        Ok(AerosolState {
            nifa: interpolate(&self.nifa),
            nwfa: interpolate(&self.nwfa),
        })
    }
}

/// MPAS: mpas_init_atm_thompson_aerosols.F:36-81.
/// No file means zero cold-start scalars, never the synthetic aerosol profile.
pub fn initialize(
    path: &Path,
    lat: &[f32],
    lon: &[f32],
    landmask: &[i32],
    n_levels: usize,
    date: &str,
    pressure_base: &[f32],
    n_vert: usize,
) -> MpasResult<(AerosolState, AerosolReceipt)> {
    if !path.try_exists()? {
        return Ok((
            AerosolState {
                nifa: vec![0.0; lat.len() * n_vert],
                nwfa: vec![0.0; lat.len() * n_vert],
            },
            AerosolReceipt {
                source: path.display().to_string(),
                ..Default::default()
            },
        ));
    }
    let (clim, mut receipt) = horizontal(path, lat, lon, landmask, n_levels)?;
    let state = clim.at_date(date, pressure_base, n_vert)?;
    receipt.values_computed = state.nifa.len() + state.nwfa.len();
    Ok((state, receipt))
}

/// MPAS: mpas_init_atm_thompson_aerosols.F:722-862.
/// Caller-owned cache mirrors mesh climatology arrays: reload at timestart,
/// reuse thereafter; a missing file leaves the caller's LBC state untouched.
pub fn initialize_lbc(
    path: &Path,
    timestamp: &str,
    timestart: &str,
    lat: &[f32],
    lon: &[f32],
    landmask: &[i32],
    n_levels: usize,
    pressure_base: &[f32],
    n_vert: usize,
    cache: &mut Option<Climatology>,
) -> MpasResult<Option<AerosolState>> {
    if !path.try_exists()? {
        return Ok(None);
    }
    if timestamp == timestart {
        *cache = Some(horizontal(path, lat, lon, landmask, n_levels)?.0);
    }
    let clim = cache.as_ref().ok_or_else(|| {
        MpasError::Refusal(
            "GOCART LBC requested after timestart without initialized mesh climatology".into(),
        )
    })?;
    Ok(Some(clim.at_date(timestamp, pressure_base, n_vert)?))
}
