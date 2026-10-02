//! The operator catalogue: the only code a variables-table row can name.
//!
//! A row says which operator makes it and which history fields it needs;
//! a new variable that an existing operator can make (any raw history field
//! with a scale, any of these derived quantities) is a new row and nothing
//! here changes.
//!
//! | operator | makes | from |
//! |---|---|---|
//! | `geopotential` | geopotential on mass levels (m2 s-2) | PH + PHB averaged onto mass levels |
//! | `temperature` | temperature (K) | (T + 300) (p / 1e5)^(2/7) |
//! | `earth-wind:u`, `earth-wind:v` | earth-relative wind (m s-1) | U, V unstaggered, rotated by SINALPHA/COSALPHA |
//! | `specific-humidity` | vapour per kg of moist air (kg kg-1) | QVAPOR / (1 + QVAPOR + condensate) |
//! | `omega` | vertical velocity in pressure (Pa s-1) | -g rho w, wrf-core's `omega` |
//! | `rh-ifs` | relative humidity (%), IFS mixed phase | T, p, QVAPOR |
//! | `pressure` | pressure (Pa), model levels | P + PB |
//! | `raw:NAME` | a history field times the row's scale | NAME |
//! | `earth-wind10:u`, `earth-wind10:v` | earth-relative 10 m wind | U10, V10 rotated |
//! | `slp` | mean sea-level pressure (Pa) | wrf-core's `slp` (Shuell 1995, the wrf-python routine), x 100 |
//! | `accumulation:interval`, `accumulation:6h` | precipitation over the period (m) | RAINNC, RAINC, RAINSH and their buckets |
//! | `column-vapour` | total column water vapour (kg m-2) | q integrated over the column |
//! | `surface-geopotential` | the surface's geopotential (m2 s-2) | PH + PHB at the bottom face |

use wrf_core::WrfFile;

use crate::error::{fail, refuse, Result};

/// Gravity for column integrals (ECMWF's value; tcwv is an ERA5 quantity).
pub const G_COLUMN: f64 = 9.80665;

/// Condensate mixing ratios counted into the moist-air mass, when present.
pub const CONDENSATES: [&str; 6] = ["QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP", "QHAIL"];

fn nonnegative(value: f64) -> f64 {
    if value.is_finite() { value.max(0.0) } else { f64::NAN }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Component {
    U,
    V,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Period {
    Interval,
    Hours(i64),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Op {
    Geopotential,
    Temperature,
    EarthWind(Component),
    SpecificHumidity,
    Omega,
    RhIfs,
    Pressure,
    Raw(String),
    EarthWind10(Component),
    Slp,
    Accumulation(Period),
    ColumnVapour,
    SurfaceGeopotential,
}

impl Op {
    pub fn parse(text: &str) -> Result<Op> {
        let unknown = || {
            refuse(format!(
                "operator '{text}' is not one this exporter has, so the table row naming it cannot be made; the operators are geopotential, temperature, earth-wind:u|v, specific-humidity, omega, rh-ifs, pressure, raw:NAME, earth-wind10:u|v, slp, accumulation:interval|NNh, column-vapour, surface-geopotential"
            ))
        };
        let component = |c: &str| match c {
            "u" => Ok(Component::U),
            "v" => Ok(Component::V),
            _ => Err(unknown()),
        };
        Ok(match text.split_once(':') {
            None => match text {
                "geopotential" => Op::Geopotential,
                "temperature" => Op::Temperature,
                "specific-humidity" => Op::SpecificHumidity,
                "omega" => Op::Omega,
                "rh-ifs" => Op::RhIfs,
                "pressure" => Op::Pressure,
                "slp" => Op::Slp,
                "column-vapour" => Op::ColumnVapour,
                "surface-geopotential" => Op::SurfaceGeopotential,
                _ => return Err(unknown()),
            },
            Some(("earth-wind", c)) => Op::EarthWind(component(c)?),
            Some(("earth-wind10", c)) => Op::EarthWind10(component(c)?),
            Some(("raw", name)) if !name.is_empty() => Op::Raw(name.to_string()),
            Some(("accumulation", "interval")) => Op::Accumulation(Period::Interval),
            Some(("accumulation", hours)) => {
                let h = hours
                    .strip_suffix('h')
                    .and_then(|h| h.parse::<i64>().ok())
                    .filter(|&h| h > 0)
                    .ok_or_else(unknown)?;
                Op::Accumulation(Period::Hours(h))
            }
            _ => return Err(unknown()),
        })
    }
}

/// Read a raw field for one time, naming the history file on failure.
pub fn read(file: &WrfFile, t: usize, name: &str) -> Result<Vec<f64>> {
    file.read_var(name, t)
        .map_err(|e| fail(format!("reading {name}: {e}")))
}

fn shared(result: wrf_core::WrfResult<wrf_core::file::SharedField>, what: &str) -> Result<std::sync::Arc<[f64]>> {
    result.map_err(|e| fail(format!("computing {what}: {e}")))
}

/// Earth-relative wind from grid-relative components: u_e = u cos a - v sin
/// a, v_e = u sin a + v cos a (wrf-python's `uvmet`).  `cells` is one plane.
pub fn rotate(u: &[f64], v: &[f64], sin: &[f64], cos: &[f64], cells: usize, which: Component) -> Vec<f64> {
    u.iter()
        .zip(v)
        .enumerate()
        .map(|(i, (&u, &v))| {
            let c = i % cells;
            match which {
                Component::U => u * cos[c] - v * sin[c],
                Component::V => u * sin[c] + v * cos[c],
            }
        })
        .collect()
}

/// The moist-air denominator 1 + QVAPOR + condensate, per mass point.
pub fn moist_denominator(file: &WrfFile, t: usize) -> Result<Vec<f64>> {
    let qv = shared(file.qvapor(t), "QVAPOR")?;
    let mut denominator: Vec<f64> = qv.iter().map(|&q| 1.0 + nonnegative(q)).collect();
    for name in CONDENSATES {
        if file.has_var(name) {
            let q = read(file, t, name)?;
            for (d, q) in denominator.iter_mut().zip(&q) {
                *d += nonnegative(*q);
            }
        }
    }
    Ok(denominator)
}

/// Specific humidity, vapour per kg of moist air.
pub fn specific_humidity(file: &WrfFile, t: usize) -> Result<Vec<f64>> {
    let qv = shared(file.qvapor(t), "QVAPOR")?;
    let denominator = moist_denominator(file, t)?;
    Ok(qv.iter().zip(&denominator).map(|(&q, d)| nonnegative(q) / d).collect())
}

/// IFS saturation vapour pressure (Pa): over water at or above 273.16 K,
/// over ice at or below 250.16 K, weighted by ((T - 250.16) / 23)^2 between
/// (the Tetens form the IFS documentation, Part IV, gives for its
/// saturation curves).
pub fn es_ifs(t: f64) -> f64 {
    const T0: f64 = 273.16;
    const TICE: f64 = 250.16;
    let water = 611.21 * (17.502 * (t - T0) / (t - 32.19)).exp();
    let ice = 611.21 * (22.587 * (t - T0) / (t + 0.7)).exp();
    if t >= T0 {
        water
    } else if t <= TICE {
        ice
    } else {
        let alpha = ((t - TICE) / (T0 - TICE)).powi(2);
        alpha * water + (1.0 - alpha) * ice
    }
}

/// Relative humidity (%), not capped at 100 (ERA5 is not).
pub fn rh_ifs(p: f64, t: f64, qv: f64) -> f64 {
    let qv = nonnegative(qv);
    let e = p * qv / (0.62198 + qv);
    100.0 * e / es_ifs(t)
}

/// A 3-D field on mass levels, `[nz, ny, nx]`.
pub fn level_field(file: &WrfFile, t: usize, op: &Op, scale: f64) -> Result<Vec<f64>> {
    let cells = file.nxy();
    Ok(match op {
        Op::Geopotential => shared(file.full_geopotential(t), "geopotential")?.to_vec(),
        Op::Temperature => shared(file.temperature(t), "temperature")?.to_vec(),
        Op::EarthWind(which) => {
            let u = shared(file.u_destag(t), "U")?;
            let v = shared(file.v_destag(t), "V")?;
            let sin = shared(file.sinalpha(t), "SINALPHA")?;
            let cos = shared(file.cosalpha(t), "COSALPHA")?;
            rotate(&u, &v, &sin, &cos, cells, *which)
        }
        Op::SpecificHumidity => specific_humidity(file, t)?,
        Op::Omega => wrf_core::diag::pressure::compute_omega(file, t, &Default::default())
            .map_err(|e| fail(format!("computing omega: {e}")))?,
        Op::RhIfs => {
            let p = shared(file.full_pressure(t), "pressure")?;
            let tk = shared(file.temperature(t), "temperature")?;
            let qv = shared(file.qvapor(t), "QVAPOR")?;
            p.iter().zip(tk.iter()).zip(qv.iter()).map(|((&p, &t), &q)| rh_ifs(p, t, q)).collect()
        }
        Op::Pressure => shared(file.full_pressure(t), "pressure")?.to_vec(),
        Op::Raw(name) => {
            let values = read(file, t, name)?;
            if values.len() != file.nxyz() {
                return Err(refuse(format!(
                    "{name} is not a mass-level field ({} values, the grid has {}), so it cannot be put on levels",
                    values.len(),
                    file.nxyz()
                )));
            }
            values.into_iter().map(|v| v * scale).collect()
        }
        other => {
            return Err(refuse(format!(
                "operator {other:?} makes a surface field, so a level variable cannot use it"
            )))
        }
    })
}

/// Total column water vapour (kg m-2): sum over mass levels of q (p_lower -
/// p_upper) / g.  Interfaces: PSFC at the bottom, the geometric mean of the
/// two mass-level pressures inside, and at the top P_TOP when the file
/// states it, else half a layer above the top mass level in ln p.
pub fn column_vapour(file: &WrfFile, t: usize, p_top: Option<f64>) -> Result<Vec<f64>> {
    let (nz, cells) = (file.nz, file.nxy());
    let q = specific_humidity(file, t)?;
    let p = shared(file.full_pressure(t), "pressure")?;
    let psfc = read(file, t, "PSFC")?;
    let mut out = vec![0.0f64; cells];
    for c in 0..cells {
        let mut lower = psfc[c];
        let mut sum = 0.0;
        for k in 0..nz {
            let upper = if k + 1 < nz {
                (p[k * cells + c] * p[(k + 1) * cells + c]).sqrt()
            } else {
                match p_top {
                    Some(top) => top,
                    None if nz >= 2 => {
                        let pk = p[k * cells + c];
                        let below = p[(k - 1) * cells + c];
                        pk * (pk / below).sqrt()
                    }
                    None => 0.0,
                }
            };
            sum += q[k * cells + c] * nonnegative(lower - upper);
            lower = upper;
        }
        out[c] = sum / G_COLUMN;
    }
    Ok(out)
}

/// The accumulated precipitation (mm) a history file carries: RAINNC,
/// RAINC and RAINSH when present, plus their bucket counts times
/// BUCKET_MM.
pub fn accumulated_precipitation(file: &WrfFile, t: usize, bucket_mm: Option<f64>) -> Result<Vec<f64>> {
    let cells = file.nxy();
    let mut total = vec![0.0f64; cells];
    let mut any = false;
    for (field, bucket) in [("RAINNC", "I_RAINNC"), ("RAINC", "I_RAINC"), ("RAINSH", "")] {
        if !file.has_var(field) {
            continue;
        }
        any = true;
        let values = read(file, t, field)?;
        for (a, v) in total.iter_mut().zip(&values) {
            *a += v;
        }
        if !bucket.is_empty() && file.has_var(bucket) {
            let size = bucket_mm.ok_or_else(|| {
                refuse(format!(
                    "the history file carries {bucket} but no BUCKET_MM, so the emptied buckets cannot be added back and the precipitation would be wrong"
                ))
            })?;
            let counts = read(file, t, bucket)?;
            for (a, n) in total.iter_mut().zip(&counts) {
                *a += n * size;
            }
        }
    }
    if !any {
        return Err(refuse(
            "the history file carries no RAINNC, so total precipitation cannot be made",
        ));
    }
    Ok(total)
}

/// A 2-D field, `[ny, nx]`, for a surface or static row.  Accumulations
/// are handled by the caller, which holds the earlier frames.
pub fn surface_field(file: &WrfFile, t: usize, op: &Op, scale: f64, p_top: Option<f64>) -> Result<Vec<f64>> {
    let cells = file.nxy();
    Ok(match op {
        Op::Raw(name) => {
            let values = read(file, t, name)?;
            if values.len() != cells {
                return Err(refuse(format!(
                    "{name} is not a surface field ({} values, a plane has {cells}), so it cannot be written as one",
                    values.len()
                )));
            }
            values.into_iter().map(|v| v * scale).collect()
        }
        Op::EarthWind10(which) => {
            let u = shared(file.u10(t), "U10")?;
            let v = shared(file.v10(t), "V10")?;
            let sin = shared(file.sinalpha(t), "SINALPHA")?;
            let cos = shared(file.cosalpha(t), "COSALPHA")?;
            rotate(&u, &v, &sin, &cos, cells, *which)
        }
        Op::Slp => wrf_core::diag::pressure::compute_slp(file, t, &Default::default())
            .map_err(|e| fail(format!("computing sea-level pressure: {e}")))?
            .into_iter()
            .map(|hpa| hpa * 100.0)
            .collect(),
        Op::ColumnVapour => column_vapour(file, t, p_top)?,
        Op::SurfaceGeopotential => {
            let stag = shared(file.geopotential_stag(t), "geopotential")?;
            stag[..cells].iter().map(|v| v * scale).collect()
        }
        other => {
            return Err(refuse(format!(
                "operator {other:?} does not make a surface field"
            )))
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn operators_parse_and_unknown_ones_refuse() {
        assert_eq!(Op::parse("earth-wind:v").unwrap(), Op::EarthWind(Component::V));
        assert_eq!(Op::parse("accumulation:6h").unwrap(), Op::Accumulation(Period::Hours(6)));
        assert_eq!(Op::parse("raw:T2").unwrap(), Op::Raw("T2".into()));
        assert!(Op::parse("vorticity").unwrap_err().is_refusal());
        assert!(Op::parse("earth-wind:w").is_err());
        assert!(Op::parse("accumulation:0h").is_err());
    }

    #[test]
    fn rotation_is_a_proper_rotation() {
        let (sin, cos) = (0.3f64.sin(), 0.3f64.cos());
        let u = rotate(&[3.0], &[4.0], &[sin], &[cos], 1, Component::U)[0];
        let v = rotate(&[3.0], &[4.0], &[sin], &[cos], 1, Component::V)[0];
        assert!(((u * u + v * v).sqrt() - 5.0).abs() < 1e-12);
        assert!((u - (3.0 * cos - 4.0 * sin)).abs() < 1e-12);
        // No rotation is the identity.
        assert_eq!(rotate(&[3.0], &[4.0], &[0.0], &[1.0], 1, Component::U)[0], 3.0);
    }

    #[test]
    fn ifs_saturation_blends_between_ice_and_water() {
        // At the triple point both curves give 611.21 Pa.
        assert!((es_ifs(273.16) - 611.21).abs() < 1e-9);
        // Below 250.16 K: ice; above 273.16 K: water.
        let ice = 611.21 * (22.587 * (240.0 - 273.16) / (240.0 + 0.7f64)).exp();
        assert!((es_ifs(240.0) - ice).abs() < 1e-12);
        let water = 611.21 * (17.502 * (300.0 - 273.16) / (300.0 - 32.19f64)).exp();
        assert!((es_ifs(300.0) - water).abs() < 1e-12);
        // Halfway in T is a quarter of the way to water in weight.
        let t: f64 = 250.16 + 11.5;
        let w = 611.21 * (17.502 * (t - 273.16) / (t - 32.19)).exp();
        let i = 611.21 * (22.587 * (t - 273.16) / (t + 0.7)).exp();
        assert!((es_ifs(t) - (0.25 * w + 0.75 * i)).abs() < 1e-9);
        // RH is not capped.
        assert!(rh_ifs(100_000.0, 280.0, 0.02) > 100.0);
    }
}
