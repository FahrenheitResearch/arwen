//! Inverse map projections for the fabric CRSs the contract allows, so divide
//! polygons can be laid onto the lon/lat parameter grids:
//! - EPSG:5070 (NAD83 / Conus Albers), ellipsoidal Albers equal-area conic;
//! - the contract's global row CRS: Lambert azimuthal equal-area on WGS84,
//!   centred on the domain (organization ARWEN, WKT2 definition);
//! - EPSG:4326 / EPSG:4269 (geographic, identity).
//! Formulas: J. P. Snyder (1987), "Map Projections: A Working Manual", USGS
//! Professional Paper 1395: Albers eqs. (3-12), (14-3)..(14-19) and (3-16);
//! Lambert azimuthal equal-area eqs. (24-11)..(24-31). Datum shifts between
//! NAD83 and WGS84 (about 1 m) are below the 90 m grid and are ignored.
use crate::Result;

const A: f64 = 6378137.0;
const F_GRS80: f64 = 1.0 / 298.257222101;
const F_WGS84: f64 = 1.0 / 298.257223563;

#[derive(Clone, Debug)]
pub enum Crs {
    Geographic,
    Albers { e: f64, n: f64, c: f64, rho0: f64, lon0: f64, fe: f64, fn_: f64 },
    Laea { e: f64, qp: f64, rq: f64, d: f64, sinb1: f64, cosb1: f64, lat0: f64, lon0: f64, fe: f64, fn_: f64 },
}

fn q_of(e: f64, phi: f64) -> f64 {
    let s = phi.sin();
    let es = e * s;
    (1.0 - e * e) * (s / (1.0 - es * es) - (1.0 / (2.0 * e)) * ((1.0 - es) / (1.0 + es)).ln())
}
fn m_of(e: f64, phi: f64) -> f64 {
    let s = phi.sin();
    phi.cos() / (1.0 - e * e * s * s).sqrt()
}
/// latitude from authalic q by Snyder eq. (3-16), Newton iteration
fn phi_of_q(e: f64, q: f64) -> f64 {
    let mut phi = (q / 2.0).clamp(-1.0, 1.0).asin();
    for _ in 0..30 {
        let s = phi.sin();
        let es = e * s;
        let one = 1.0 - es * es;
        let d = one * one / (2.0 * phi.cos()) * (q / (1.0 - e * e) - s / one + (1.0 / (2.0 * e)) * ((1.0 - es) / (1.0 + es)).ln());
        phi += d;
        if d.abs() < 1e-14 {
            break;
        }
    }
    phi
}

impl Crs {
    pub fn albers(f: f64, lat1: f64, lat2: f64, lat0: f64, lon0: f64, fe: f64, fn_: f64) -> Crs {
        let e = (2.0 * f - f * f).sqrt();
        let (p1, p2, p0) = (lat1.to_radians(), lat2.to_radians(), lat0.to_radians());
        let (m1, m2) = (m_of(e, p1), m_of(e, p2));
        let (q1, q2, q0) = (q_of(e, p1), q_of(e, p2), q_of(e, p0));
        let n = (m1 * m1 - m2 * m2) / (q2 - q1);
        let c = m1 * m1 + n * q1;
        let rho0 = A * (c - n * q0).sqrt() / n;
        Crs::Albers { e, n, c, rho0, lon0: lon0.to_radians(), fe, fn_ }
    }
    pub fn laea(f: f64, lat0: f64, lon0: f64, fe: f64, fn_: f64) -> Crs {
        let e = (2.0 * f - f * f).sqrt();
        let p1 = lat0.to_radians();
        let qp = q_of(e, std::f64::consts::FRAC_PI_2);
        let b1 = (q_of(e, p1) / qp).asin();
        let rq = A * (qp / 2.0).sqrt();
        let d = A * m_of(e, p1) / (rq * b1.cos());
        Crs::Laea { e, qp, rq, d, sinb1: b1.sin(), cosb1: b1.cos(), lat0, lon0: lon0.to_radians(), fe, fn_ }
    }

    /// The CRS of a contract fabric from its gpkg_spatial_ref_sys row.
    pub fn from_srs(org: &str, code: i64, wkt: &str) -> Result<Crs> {
        if org.eq_ignore_ascii_case("EPSG") {
            return match code {
                4326 | 4269 => Ok(Crs::Geographic),
                5070 => Ok(Crs::albers(F_GRS80, 29.5, 45.5, 23.0, -96.0, 0.0, 0.0)),
                _ => Err(format!("EPSG:{code} is not a contract CRS (EPSG:5070 or an ARWEN LAEA row)").into()),
            };
        }
        let w = wkt.to_ascii_lowercase();
        if w.contains("lambert azimuthal equal area") || w.contains("lambert_azimuthal_equal_area") {
            let get = |names: &[&str]| -> Option<f64> {
                for n in names {
                    if let Some(i) = w.find(n) {
                        let rest = &w[i + n.len()..];
                        let rest = rest.trim_start_matches(|c: char| c == '"' || c == ',' || c.is_whitespace());
                        let num: String = rest.chars().take_while(|c| c.is_ascii_digit() || *c == '-' || *c == '.' || *c == 'e' || *c == '+').collect();
                        if let Ok(v) = num.parse::<f64>() {
                            return Some(v);
                        }
                    }
                }
                None
            };
            let lat0 = get(&["latitude of natural origin\"", "latitude_of_center\"", "latitude_of_origin\""]).ok_or("LAEA WKT without latitude of natural origin")?;
            let lon0 = get(&["longitude of natural origin\"", "longitude_of_center\"", "central_meridian\""]).ok_or("LAEA WKT without longitude of natural origin")?;
            let fe = get(&["false easting\"", "false_easting\""]).unwrap_or(0.0);
            let fn_ = get(&["false northing\"", "false_northing\""]).unwrap_or(0.0);
            let f = if w.contains("grs 1980") || w.contains("grs_1980") { F_GRS80 } else { F_WGS84 };
            return Ok(Crs::laea(f, lat0, lon0, fe, fn_));
        }
        Err(format!("unsupported fabric CRS {org}:{code}").into())
    }

    /// projected metres -> (lon, lat) degrees
    pub fn inverse(&self, x: f64, y: f64) -> (f64, f64) {
        match *self {
            Crs::Geographic => (x, y),
            Crs::Albers { e, n, c, rho0, lon0, fe, fn_ } => {
                let (x, y) = (x - fe, y - fn_);
                let rho = (x * x + (rho0 - y) * (rho0 - y)).sqrt();
                let theta = if n >= 0.0 { x.atan2(rho0 - y) } else { (-x).atan2(y - rho0) };
                let q = (c - rho * rho * n * n / (A * A)) / n;
                ((lon0 + theta / n).to_degrees(), phi_of_q(e, q).to_degrees())
            }
            Crs::Laea { e, qp, rq, d, sinb1, cosb1, lat0, lon0, fe, fn_ } => {
                let (x, y) = (x - fe, y - fn_);
                let rho = ((x / d).powi(2) + (d * y).powi(2)).sqrt();
                if rho < 1e-9 {
                    return (lon0.to_degrees(), lat0);
                }
                let ce = 2.0 * (rho / (2.0 * rq)).clamp(-1.0, 1.0).asin();
                let (sc, cc) = ce.sin_cos();
                let q = qp * (cc * sinb1 + d * y * sc * cosb1 / rho);
                let lam = lon0 + (x * sc).atan2(d * rho * cosb1 * cc - d * d * y * sinb1 * sc);
                (lam.to_degrees(), phi_of_q(e, q).to_degrees())
            }
        }
    }

    /// (lon, lat) degrees -> projected metres (used by the self-test only)
    pub fn forward(&self, lon: f64, lat: f64) -> (f64, f64) {
        match *self {
            Crs::Geographic => (lon, lat),
            Crs::Albers { e, n, c, rho0, lon0, fe, fn_ } => {
                let q = q_of(e, lat.to_radians());
                let rho = A * (c - n * q).sqrt() / n;
                let th = n * (lon.to_radians() - lon0);
                (fe + rho * th.sin(), fn_ + rho0 - rho * th.cos())
            }
            Crs::Laea { e, qp, rq, d, sinb1, cosb1, lon0, fe, fn_, .. } => {
                let b = (q_of(e, lat.to_radians()) / qp).asin();
                let dl = lon.to_radians() - lon0;
                let bb = rq * (2.0 / (1.0 + sinb1 * b.sin() + cosb1 * b.cos() * dl.cos())).sqrt();
                (fe + bb * d * b.cos() * dl.sin(), fn_ + (bb / d) * (cosb1 * b.sin() - sinb1 * b.cos() * dl.cos()))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn albers_5070_known_point_and_roundtrip() {
        let c = Crs::albers(F_GRS80, 29.5, 45.5, 23.0, -96.0, 0.0, 0.0);
        // the projection origin maps to (0, 0)
        let (x, y) = c.forward(-96.0, 23.0);
        assert!(x.abs() < 1e-6 && y.abs() < 1e-6);
        // reference values from PROJ (gdaltransform -s_srs EPSG:4269 -t_srs EPSG:5070)
        for (lon, lat, x, y) in [(-70.97, 43.15, 2001933.48393881, 2504627.98212282), (-123.0, 38.8, -2291893.36416209, 2080277.51835114)] {
            let (px, py) = c.forward(lon, lat);
            assert!((px - x).abs() < 1e-3 && (py - y).abs() < 1e-3, "{lon} {lat}: {px} {py} vs PROJ {x} {y}");
            let (l2, p2) = c.inverse(x, y);
            assert!((l2 - lon).abs() < 1e-9 && (p2 - lat).abs() < 1e-9);
        }
        for (lon, lat) in [(-70.97, 43.15), (-123.0, 38.8), (-80.0, 26.0), (-100.0, 48.0)] {
            let (x, y) = c.forward(lon, lat);
            let (l2, p2) = c.inverse(x, y);
            assert!((l2 - lon).abs() < 1e-9 && (p2 - lat).abs() < 1e-9, "{lon} {lat} -> {l2} {p2}");
        }
    }
    #[test]
    fn laea_roundtrip() {
        let c = Crs::laea(F_WGS84, 49.5, 7.75, 0.0, 0.0);
        // PROJ: +proj=laea +lat_0=49.5 +lon_0=7.75 +ellps=WGS84
        let (x, y) = c.forward(4.0, 52.0);
        assert!((x + 257487.851616899).abs() < 1e-3 && (y - 284531.234452094).abs() < 1e-3, "{x} {y}");
        for (lon, lat) in [(7.75, 49.5), (4.0, 52.0), (11.9, 46.0), (6.0, 53.3)] {
            let (x, y) = c.forward(lon, lat);
            let (l2, p2) = c.inverse(x, y);
            assert!((l2 - lon).abs() < 1e-9 && (p2 - lat).abs() < 1e-9, "{lon} {lat} -> {l2} {p2}");
        }
    }
}
