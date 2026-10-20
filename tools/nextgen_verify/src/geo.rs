// SPDX-License-Identifier: Apache-2.0
//! Geometry helpers for gauge matching: GeoPackage binary geometry, the
//! equal-area projections the HYDRO-WORLD contract allows (Albers conic for
//! CONUS EPSG:5070, Lambert azimuthal for EPSG:3035 and the domain-centred
//! ARWEN rows), read from the fabric's own CRS definition. Formulas: Snyder,
//! "Map Projections: A Working Manual", USGS PP 1395 (1987), ellipsoidal forms.
use crate::Result;

pub type Line = Vec<[f64; 2]>;

/// Decode a GeoPackage binary LINESTRING / MULTILINESTRING into lines.
pub fn gpkg_lines(b: &[u8]) -> Result<Vec<Line>> {
    let wkb = gpkg_body(b)?;
    let mut at = 0;
    let mut out = Vec::new();
    wkb_lines(wkb, &mut at, &mut out)?;
    Ok(out)
}
/// Decode a GeoPackage binary POINT.
pub fn gpkg_point(b: &[u8]) -> Result<[f64; 2]> {
    let wkb = gpkg_body(b)?;
    let le = *wkb.first().ok_or("empty wkb")? == 1;
    let t = rd_u32(wkb, 1, le)? & 0xffff;
    if t % 1000 != 1 {
        return Err(format!("expected POINT wkb, found type {t}").into());
    }
    Ok([rd_f64(wkb, 5, le)?, rd_f64(wkb, 13, le)?])
}
/// Encode a GeoPackage binary POINT (version 0, little endian, no envelope).
pub fn gpkg_point_bytes(srs_id: i32, p: [f64; 2]) -> Vec<u8> {
    let mut v = vec![b'G', b'P', 0, 0b0000_0001];
    v.extend(srs_id.to_le_bytes());
    v.push(1);
    v.extend(1u32.to_le_bytes());
    v.extend(p[0].to_le_bytes());
    v.extend(p[1].to_le_bytes());
    v
}
/// Encode a GeoPackage binary LINESTRING with an XY envelope (contract section 4).
pub fn gpkg_line_bytes(srs_id: i32, line: &[[f64; 2]]) -> Vec<u8> {
    let (mut x0, mut y0, mut x1, mut y1) = (f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
    for p in line {
        x0 = x0.min(p[0]);
        y0 = y0.min(p[1]);
        x1 = x1.max(p[0]);
        y1 = y1.max(p[1]);
    }
    let mut v = vec![b'G', b'P', 0, 0b0000_0011];
    v.extend(srs_id.to_le_bytes());
    for e in [x0, x1, y0, y1] {
        v.extend(e.to_le_bytes());
    }
    v.push(1);
    v.extend(2u32.to_le_bytes());
    v.extend((line.len() as u32).to_le_bytes());
    for p in line {
        v.extend(p[0].to_le_bytes());
        v.extend(p[1].to_le_bytes());
    }
    v
}
fn gpkg_body(b: &[u8]) -> Result<&[u8]> {
    if b.len() < 8 || &b[..2] != b"GP" {
        return Err("not GeoPackage binary geometry".into());
    }
    let flags = b[3];
    let env = match (flags >> 1) & 0b111 {
        0 => 0,
        1 => 32,
        2 | 3 => 48,
        4 => 64,
        e => return Err(format!("bad envelope code {e}").into()),
    };
    b.get(8 + env..).ok_or_else(|| "truncated geometry".into())
}
fn rd_u32(b: &[u8], at: usize, le: bool) -> Result<u32> {
    let s: [u8; 4] = b.get(at..at + 4).ok_or("truncated wkb")?.try_into().unwrap();
    Ok(if le { u32::from_le_bytes(s) } else { u32::from_be_bytes(s) })
}
fn rd_f64(b: &[u8], at: usize, le: bool) -> Result<f64> {
    let s: [u8; 8] = b.get(at..at + 8).ok_or("truncated wkb")?.try_into().unwrap();
    Ok(if le { f64::from_le_bytes(s) } else { f64::from_be_bytes(s) })
}
fn wkb_lines(b: &[u8], at: &mut usize, out: &mut Vec<Line>) -> Result<()> {
    let le = *b.get(*at).ok_or("truncated wkb")? == 1;
    let t = rd_u32(b, *at + 1, le)?;
    let (base, dims) = match t {
        t if t > 0x2000_0000 => return Err("EWKB flags are not GeoPackage WKB".into()),
        t if t >= 3000 => (t - 3000, 4),
        t if t >= 2000 => (t - 2000, 3),
        t if t >= 1000 => (t - 1000, 3),
        t => (t, 2),
    };
    *at += 5;
    match base {
        2 => {
            let n = rd_u32(b, *at, le)? as usize;
            *at += 4;
            let mut line = Vec::with_capacity(n);
            for _ in 0..n {
                line.push([rd_f64(b, *at, le)?, rd_f64(b, *at + 8, le)?]);
                *at += 8 * dims;
            }
            out.push(line);
        }
        5 => {
            let n = rd_u32(b, *at, le)? as usize;
            *at += 4;
            for _ in 0..n {
                wkb_lines(b, at, out)?;
            }
        }
        other => return Err(format!("expected (MULTI)LINESTRING wkb, found type {other}").into()),
    }
    Ok(())
}

/// Forward projection from geographic degrees to the fabric CRS metres.
#[derive(Clone, Debug, PartialEq)]
pub enum Projection {
    Albers { a: f64, e: f64, lat0: f64, lon0: f64, lat1: f64, lat2: f64, fe: f64, fn_: f64 },
    Laea { a: f64, e: f64, lat0: f64, lon0: f64, fe: f64, fn_: f64 },
}

fn wkt_number(wkt: &str, names: &[&str]) -> Option<f64> {
    let lower = wkt.to_ascii_lowercase();
    for n in names {
        let key = format!("\"{}\"", n.to_ascii_lowercase());
        if let Some(i) = lower.find(&key) {
            let rest = &wkt[i + key.len()..];
            let rest = rest.trim_start_matches([',', ' ']);
            let end = rest.find([',', ']', ' ']).unwrap_or(rest.len());
            if let Ok(v) = rest[..end].trim().parse::<f64>() {
                return Some(v);
            }
        }
    }
    None
}
fn ellipsoid(wkt: &str) -> (f64, f64) {
    // SPHEROID["GRS 1980",6378137,298.257222101] or ELLIPSOID["WGS 84",6378137,298.257223563,...]
    let upper = wkt.to_ascii_uppercase();
    for tag in ["SPHEROID[", "ELLIPSOID["] {
        if let Some(i) = upper.find(tag) {
            let parts: Vec<&str> = wkt[i + tag.len()..].splitn(4, ',').collect();
            if parts.len() >= 3 {
                let a = parts[1].trim().parse::<f64>();
                let rf = parts[2].trim().trim_end_matches(']').parse::<f64>();
                if let (Ok(a), Ok(rf)) = (a, rf) {
                    let f = 1.0 / rf;
                    return (a, (2.0 * f - f * f).sqrt());
                }
            }
        }
    }
    let f = 1.0_f64 / 298.257_223_563;
    (6_378_137.0, (2.0 * f - f * f).sqrt())
}

impl Projection {
    /// Read the method and parameters from a WKT1 or WKT2 definition.
    pub fn from_wkt(wkt: &str) -> Result<Projection> {
        let (a, e) = ellipsoid(wkt);
        let l = wkt.to_ascii_lowercase();
        let fe = wkt_number(wkt, &["false_easting", "False easting", "Easting at false origin"]).unwrap_or(0.0);
        let fn_ = wkt_number(wkt, &["false_northing", "False northing", "Northing at false origin"]).unwrap_or(0.0);
        if l.contains("albers") {
            Ok(Projection::Albers {
                a,
                e,
                lat0: wkt_number(wkt, &["latitude_of_center", "latitude_of_origin", "Latitude of false origin"]).ok_or("albers: no origin latitude")?,
                lon0: wkt_number(wkt, &["longitude_of_center", "central_meridian", "Longitude of false origin"]).ok_or("albers: no origin longitude")?,
                lat1: wkt_number(wkt, &["standard_parallel_1", "Latitude of 1st standard parallel"]).ok_or("albers: no parallel 1")?,
                lat2: wkt_number(wkt, &["standard_parallel_2", "Latitude of 2nd standard parallel"]).ok_or("albers: no parallel 2")?,
                fe,
                fn_,
            })
        } else if l.contains("lambert_azimuthal_equal_area") || l.contains("lambert azimuthal equal area") {
            Ok(Projection::Laea {
                a,
                e,
                lat0: wkt_number(wkt, &["latitude_of_center", "latitude_of_origin", "Latitude of natural origin"]).ok_or("laea: no origin latitude")?,
                lon0: wkt_number(wkt, &["longitude_of_center", "central_meridian", "Longitude of natural origin"]).ok_or("laea: no origin longitude")?,
                fe,
                fn_,
            })
        } else {
            Err("fabric CRS is neither Albers nor Lambert azimuthal equal area".into())
        }
    }
    /// WKT2 definition of a Lambert azimuthal equal-area CRS on WGS 84 centred at (lat0, lon0).
    pub fn laea_wkt2(name: &str, lat0: f64, lon0: f64) -> String {
        format!(
            "PROJCRS[\"{name}\",BASEGEOGCRS[\"WGS 84\",DATUM[\"World Geodetic System 1984\",ELLIPSOID[\"WGS 84\",6378137,298.257223563,LENGTHUNIT[\"metre\",1]]],PRIMEM[\"Greenwich\",0,ANGLEUNIT[\"degree\",0.0174532925199433]]],CONVERSION[\"Lambert Azimuthal Equal Area\",METHOD[\"Lambert Azimuthal Equal Area\",ID[\"EPSG\",9820]],PARAMETER[\"Latitude of natural origin\",{lat0},ANGLEUNIT[\"degree\",0.0174532925199433]],PARAMETER[\"Longitude of natural origin\",{lon0},ANGLEUNIT[\"degree\",0.0174532925199433]],PARAMETER[\"False easting\",0,LENGTHUNIT[\"metre\",1]],PARAMETER[\"False northing\",0,LENGTHUNIT[\"metre\",1]]],CS[Cartesian,2],AXIS[\"easting (X)\",east,ORDER[1],LENGTHUNIT[\"metre\",1]],AXIS[\"northing (Y)\",north,ORDER[2],LENGTHUNIT[\"metre\",1]]]"
        )
    }
    pub fn forward(&self, lon: f64, lat: f64) -> [f64; 2] {
        let q = |e: f64, phi: f64| {
            let s = phi.sin();
            (1.0 - e * e) * (s / (1.0 - e * e * s * s) - (1.0 / (2.0 * e)) * ((1.0 - e * s) / (1.0 + e * s)).ln())
        };
        let m = |e: f64, phi: f64| phi.cos() / (1.0 - e * e * phi.sin().powi(2)).sqrt();
        let r = f64::to_radians;
        match *self {
            Projection::Albers { a, e, lat0, lon0, lat1, lat2, fe, fn_ } => {
                let (m1, m2) = (m(e, r(lat1)), m(e, r(lat2)));
                let (q0, q1, q2) = (q(e, r(lat0)), q(e, r(lat1)), q(e, r(lat2)));
                let n = (m1 * m1 - m2 * m2) / (q2 - q1);
                let c = m1 * m1 + n * q1;
                let rho0 = a * (c - n * q0).sqrt() / n;
                let rho = a * (c - n * q(e, r(lat))).sqrt() / n;
                let theta = n * r(lon - lon0);
                [fe + rho * theta.sin(), fn_ + rho0 - rho * theta.cos()]
            }
            Projection::Laea { a, e, lat0, lon0, fe, fn_ } => {
                let qp = q(e, std::f64::consts::FRAC_PI_2);
                let beta = (q(e, r(lat)) / qp).clamp(-1.0, 1.0).asin();
                let beta1 = (q(e, r(lat0)) / qp).clamp(-1.0, 1.0).asin();
                let rq = a * (qp / 2.0).sqrt();
                let d = a * m(e, r(lat0)) / (rq * beta1.cos());
                let dl = r(lon - lon0);
                let b = rq * (2.0 / (1.0 + beta1.sin() * beta.sin() + beta1.cos() * beta.cos() * dl.cos())).sqrt();
                [
                    fe + b * d * beta.cos() * dl.sin(),
                    fn_ + (b / d) * (beta1.cos() * beta.sin() - beta1.sin() * beta.cos() * dl.cos()),
                ]
            }
        }
    }
}

/// Great-circle distance on the WGS 84 mean sphere, metres.
pub fn haversine_m(lon1: f64, lat1: f64, lon2: f64, lat2: f64) -> f64 {
    let r = 6_371_008.8;
    let (p1, p2) = (lat1.to_radians(), lat2.to_radians());
    let dp = p2 - p1;
    let dl = (lon2 - lon1).to_radians();
    let h = (dp / 2.0).sin().powi(2) + p1.cos() * p2.cos() * (dl / 2.0).sin().powi(2);
    2.0 * r * h.sqrt().asin()
}

/// Nearest point on a polyline: (distance, fraction of polyline length from its start).
pub fn nearest_on_line(line: &[[f64; 2]], p: [f64; 2]) -> (f64, f64) {
    let total: f64 = line.windows(2).map(|w| (w[1][0] - w[0][0]).hypot(w[1][1] - w[0][1])).sum();
    if line.len() == 1 || total == 0.0 {
        let d = (line[0][0] - p[0]).hypot(line[0][1] - p[1]);
        return (d, 0.0);
    }
    let (mut best, mut best_s, mut walked) = (f64::INFINITY, 0.0, 0.0);
    for w in line.windows(2) {
        let (dx, dy) = (w[1][0] - w[0][0], w[1][1] - w[0][1]);
        let len2 = dx * dx + dy * dy;
        let t = if len2 > 0.0 { (((p[0] - w[0][0]) * dx + (p[1] - w[0][1]) * dy) / len2).clamp(0.0, 1.0) } else { 0.0 };
        let (x, y) = (w[0][0] + t * dx, w[0][1] + t * dy);
        let d = (x - p[0]).hypot(y - p[1]);
        let seg = len2.sqrt();
        if d < best {
            best = d;
            best_s = (walked + t * seg) / total;
        }
        walked += seg;
    }
    (best, best_s)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn epsg_3035_guidance_example() {
        // IOGP Guidance Note 7-2, EPSG 9820 worked example: 50N 5E -> E 3962799.45, N 2999718.85.
        let f = 1.0_f64 / 298.257_222_101; // GRS 1980
        let p = Projection::Laea { a: 6_378_137.0, e: (2.0 * f - f * f).sqrt(), lat0: 52.0, lon0: 10.0, fe: 4_321_000.0, fn_: 3_210_000.0 };
        let [x, y] = p.forward(5.0, 50.0);
        assert!((x - 3_962_799.45).abs() < 0.01, "{x}");
        assert!((y - 2_999_718.85).abs() < 0.01, "{y}");
    }
    #[test]
    fn albers_guidance_example() {
        // IOGP Guidance Note 7-2, EPSG 9822 worked example (Clarke 1866):
        // lat0 23 N, lon0 96 W, parallels 29.5/45.5; 35N 75W (Snyder p. 292 values) -> x 1885472.7, y 1535925.0.
        let f = 1.0_f64 / 294.978_698_2;
        let p = Projection::Albers { a: 6_378_206.4, e: (2.0 * f - f * f).sqrt(), lat0: 23.0, lon0: -96.0, lat1: 29.5, lat2: 45.5, fe: 0.0, fn_: 0.0 };
        let [x, y] = p.forward(-75.0, 35.0);
        assert!((x - 1_885_472.7).abs() < 0.1, "{x}");
        assert!((y - 1_535_925.0).abs() < 0.1, "{y}");
    }
    #[test]
    fn wkt_parsing_reads_parameters() {
        let w = Projection::laea_wkt2("t", 49.5, 8.25);
        match Projection::from_wkt(&w).unwrap() {
            Projection::Laea { lat0, lon0, fe, fn_, a, .. } => {
                assert_eq!((lat0, lon0, fe, fn_, a), (49.5, 8.25, 0.0, 0.0, 6_378_137.0));
            }
            p => panic!("{p:?}"),
        }
        let w1 = r#"PROJCS["NAD83 / Conus Albers",GEOGCS["NAD83",DATUM["North_American_Datum_1983",SPHEROID["GRS 1980",6378137,298.257222101]]],PROJECTION["Albers_Conic_Equal_Area"],PARAMETER["latitude_of_center",23],PARAMETER["longitude_of_center",-96],PARAMETER["standard_parallel_1",29.5],PARAMETER["standard_parallel_2",45.5],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1]]"#;
        assert!(matches!(Projection::from_wkt(w1).unwrap(), Projection::Albers { lat1, .. } if lat1 == 29.5));
    }
    #[test]
    fn geometry_round_trip() {
        let line = vec![[0.0, 0.0], [3.0, 4.0], [3.0, 10.0]];
        assert_eq!(gpkg_lines(&gpkg_line_bytes(5, &line)).unwrap(), vec![line.clone()]);
        assert_eq!(gpkg_point(&gpkg_point_bytes(5, [1.5, -2.0])).unwrap(), [1.5, -2.0]);
        let (d, s) = nearest_on_line(&line, [3.0, 7.0]);
        assert_eq!((d, s), (0.0, 8.0 / 11.0));
    }
}
