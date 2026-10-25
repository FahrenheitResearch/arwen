//! Geodesy and geometry encoding for the builder: WGS84 ellipsoid constants,
//! geodesic distance (Vincenty inverse), ellipsoidal Lambert Azimuthal Equal
//! Area (EPSG method 9820, Snyder 1987 eqs. 24-xx / EPSG Guidance Note 7-2),
//! exact ellipsoidal cell areas, the WKT2 text of the fabric CRS, and
//! GeoPackage binary encoders (spec 2.1.3, little-endian, XY envelope for
//! LINESTRING and POLYGON, none for POINT).

pub const A: f64 = 6_378_137.0;
pub const INV_F: f64 = 298.257_223_563;
pub const F: f64 = 1.0 / INV_F;
pub const B: f64 = A * (1.0 - F);
pub const E2: f64 = F * (2.0 - F);

/// Snyder q(phi) (eq. 3-12), the authalic helper used by LAEA and areas.
pub fn q(phi: f64) -> f64 {
    let e = E2.sqrt();
    let s = phi.sin();
    (1.0 - E2) * (s / (1.0 - E2 * s * s) - (1.0 / (2.0 * e)) * ((1.0 - e * s) / (1.0 + e * s)).ln())
}

/// Area in m2 of the ellipsoidal quadrangle between latitudes lat1 < lat2 and
/// a longitude span dlon (degrees). Sum over the globe = 2 pi a^2 q_p.
pub fn quad_area_m2(lat1: f64, lat2: f64, dlon_deg: f64) -> f64 {
    A * A * dlon_deg.to_radians() / 2.0 * (q(lat2.to_radians()) - q(lat1.to_radians()))
}

/// Geodesic distance in metres on WGS84 (Vincenty 1975 inverse). Builder
/// segments are short (cell to cell, reach vertices), far from the antipodal
/// non-convergence region; a non-converging pair falls back to the
/// mean-radius great circle and is never silent (counted by the caller via
/// the return flag).
pub fn geodesic_m(lon1: f64, lat1: f64, lon2: f64, lat2: f64) -> f64 {
    if lon1 == lon2 && lat1 == lat2 {
        return 0.0;
    }
    let l = (lon2 - lon1).to_radians();
    let u1 = ((1.0 - F) * lat1.to_radians().tan()).atan();
    let u2 = ((1.0 - F) * lat2.to_radians().tan()).atan();
    let (su1, cu1) = u1.sin_cos();
    let (su2, cu2) = u2.sin_cos();
    let mut lambda = l;
    for _ in 0..200 {
        let (sl, cl) = lambda.sin_cos();
        let sin_sigma = ((cu2 * sl).powi(2) + (cu1 * su2 - su1 * cu2 * cl).powi(2)).sqrt();
        if sin_sigma == 0.0 {
            return 0.0;
        }
        let cos_sigma = su1 * su2 + cu1 * cu2 * cl;
        let sigma = sin_sigma.atan2(cos_sigma);
        let sin_alpha = cu1 * cu2 * sl / sin_sigma;
        let cos2_alpha = 1.0 - sin_alpha * sin_alpha;
        let cos_2sm = if cos2_alpha != 0.0 { cos_sigma - 2.0 * su1 * su2 / cos2_alpha } else { 0.0 };
        let c = F / 16.0 * cos2_alpha * (4.0 + F * (4.0 - 3.0 * cos2_alpha));
        let prev = lambda;
        lambda = l + (1.0 - c) * F * sin_alpha * (sigma + c * sin_sigma * (cos_2sm + c * cos_sigma * (-1.0 + 2.0 * cos_2sm * cos_2sm)));
        if (lambda - prev).abs() < 1e-12 {
            let u_sq = cos2_alpha * (A * A - B * B) / (B * B);
            let a_ = 1.0 + u_sq / 16384.0 * (4096.0 + u_sq * (-768.0 + u_sq * (320.0 - 175.0 * u_sq)));
            let b_ = u_sq / 1024.0 * (256.0 + u_sq * (-128.0 + u_sq * (74.0 - 47.0 * u_sq)));
            let ds = b_ * sin_sigma * (cos_2sm + b_ / 4.0 * (cos_sigma * (-1.0 + 2.0 * cos_2sm * cos_2sm) - b_ / 6.0 * cos_2sm * (-3.0 + 4.0 * sin_sigma * sin_sigma) * (-3.0 + 4.0 * cos_2sm * cos_2sm)));
            return B * a_ * (sigma - ds);
        }
    }
    // Antipodal non-convergence: great circle on the mean radius.
    let r = (2.0 * A + B) / 3.0;
    let (p1, p2) = (lat1.to_radians(), lat2.to_radians());
    let h = ((p2 - p1) / 2.0).sin().powi(2) + p1.cos() * p2.cos() * (l / 2.0).sin().powi(2);
    2.0 * r * h.sqrt().asin()
}

pub fn polyline_geodesic_m(pts: &[(f64, f64)]) -> f64 {
    pts.windows(2).map(|w| geodesic_m(w[0].0, w[0].1, w[1].0, w[1].1)).sum()
}

/// Ellipsoidal Lambert Azimuthal Equal Area, oblique aspect, on WGS84.
#[derive(Clone, Copy, Debug)]
pub struct Laea {
    pub lat0: f64,
    pub lon0: f64,
    qp: f64,
    rq: f64,
    sb1: f64,
    cb1: f64,
    d: f64,
}

impl Laea {
    pub fn new(lat0: f64, lon0: f64) -> Laea {
        let qp = q(std::f64::consts::FRAC_PI_2);
        let rq = A * (qp / 2.0).sqrt();
        let p1 = lat0.to_radians();
        let b1 = (q(p1) / qp).asin();
        let m1 = p1.cos() / (1.0 - E2 * p1.sin().powi(2)).sqrt();
        let d = A * m1 / (rq * b1.cos());
        Laea { lat0, lon0, qp, rq, sb1: b1.sin(), cb1: b1.cos(), d }
    }
    pub fn forward(&self, lon: f64, lat: f64) -> (f64, f64) {
        let beta = (q(lat.to_radians()) / self.qp).clamp(-1.0, 1.0).asin();
        let (sb, cb) = beta.sin_cos();
        let dl = (lon - self.lon0).to_radians();
        let (sdl, cdl) = dl.sin_cos();
        let bb = self.rq * (2.0 / (1.0 + self.sb1 * sb + self.cb1 * cb * cdl)).sqrt();
        (bb * self.d * cb * sdl, (bb / self.d) * (self.cb1 * sb - self.sb1 * cb * cdl))
    }
    /// WKT2 (ISO 19162:2019) definition written to gpkg_spatial_ref_sys.
    pub fn wkt2(&self, name: &str) -> String {
        format!(
            "PROJCRS[\"{name}\",BASEGEOGCRS[\"WGS 84\",DATUM[\"World Geodetic System 1984\",ELLIPSOID[\"WGS 84\",6378137,298.257223563,LENGTHUNIT[\"metre\",1]]],PRIMEM[\"Greenwich\",0,ANGLEUNIT[\"degree\",0.0174532925199433]],ID[\"EPSG\",4326]],CONVERSION[\"Lambert Azimuthal Equal Area\",METHOD[\"Lambert Azimuthal Equal Area\",ID[\"EPSG\",9820]],PARAMETER[\"Latitude of natural origin\",{},ANGLEUNIT[\"degree\",0.0174532925199433],ID[\"EPSG\",8801]],PARAMETER[\"Longitude of natural origin\",{},ANGLEUNIT[\"degree\",0.0174532925199433],ID[\"EPSG\",8802]],PARAMETER[\"False easting\",0,LENGTHUNIT[\"metre\",1],ID[\"EPSG\",8806]],PARAMETER[\"False northing\",0,LENGTHUNIT[\"metre\",1],ID[\"EPSG\",8807]]],CS[Cartesian,2],AXIS[\"easting (E)\",east,ORDER[1],LENGTHUNIT[\"metre\",1]],AXIS[\"northing (N)\",north,ORDER[2],LENGTHUNIT[\"metre\",1]]]",
            fmt_num(self.lat0),
            fmt_num(self.lon0)
        )
    }
}

/// Shortest decimal that round-trips (Rust's Display for f64).
pub fn fmt_num(x: f64) -> String {
    format!("{x}")
}

/// Signed planar area (shoelace), positive for counter-clockwise rings.
pub fn ring_area(r: &[(f64, f64)]) -> f64 {
    let n = r.len();
    if n < 3 {
        return 0.0;
    }
    let mut s = 0.0;
    for i in 0..n {
        let (x1, y1) = r[i];
        let (x2, y2) = r[(i + 1) % n];
        s += x1 * y2 - x2 * y1;
    }
    s / 2.0
}

fn header(buf: &mut Vec<u8>, srs: i32, env: Option<[f64; 4]>) {
    buf.extend_from_slice(b"GP");
    buf.push(0);
    buf.push(if env.is_some() { 0b0000_0011 } else { 0b0000_0001 });
    buf.extend_from_slice(&srs.to_le_bytes());
    if let Some(e) = env {
        for v in e {
            buf.extend_from_slice(&v.to_le_bytes());
        }
    }
}

fn envelope(pts: impl Iterator<Item = (f64, f64)>) -> [f64; 4] {
    let mut e = [f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::NEG_INFINITY];
    for (x, y) in pts {
        e[0] = e[0].min(x);
        e[1] = e[1].max(x);
        e[2] = e[2].min(y);
        e[3] = e[3].max(y);
    }
    e
}

fn nz(v: f64) -> f64 {
    if v == 0.0 { 0.0 } else { v }
}

pub fn gpkg_point(srs: i32, x: f64, y: f64) -> Vec<u8> {
    let mut b = Vec::with_capacity(29);
    header(&mut b, srs, None);
    b.push(1);
    b.extend_from_slice(&1u32.to_le_bytes());
    b.extend_from_slice(&nz(x).to_le_bytes());
    b.extend_from_slice(&nz(y).to_le_bytes());
    b
}

pub fn gpkg_linestring(srs: i32, pts: &[(f64, f64)]) -> Vec<u8> {
    let mut b = Vec::with_capacity(48 + 9 + 16 * pts.len());
    header(&mut b, srs, Some(envelope(pts.iter().copied())));
    b.push(1);
    b.extend_from_slice(&2u32.to_le_bytes());
    b.extend_from_slice(&(pts.len() as u32).to_le_bytes());
    for &(x, y) in pts {
        b.extend_from_slice(&nz(x).to_le_bytes());
        b.extend_from_slice(&nz(y).to_le_bytes());
    }
    b
}

/// Rings are closed here (first vertex repeated last). The exterior is made
/// counter-clockwise and holes clockwise, as the contract requires.
pub fn gpkg_polygon(srs: i32, exterior: &[(f64, f64)], holes: &[Vec<(f64, f64)>]) -> Vec<u8> {
    let mut rings: Vec<Vec<(f64, f64)>> = Vec::with_capacity(1 + holes.len());
    let orient = |r: &[(f64, f64)], ccw: bool| {
        let mut v = r.to_vec();
        if v.len() > 1 && v.first() == v.last() {
            v.pop();
        }
        if (ring_area(&v) > 0.0) != ccw {
            v.reverse();
        }
        v.push(v[0]);
        v
    };
    rings.push(orient(exterior, true));
    for h in holes {
        rings.push(orient(h, false));
    }
    let n: usize = rings.iter().map(|r| r.len()).sum();
    let mut b = Vec::with_capacity(48 + 9 + 4 * rings.len() + 16 * n);
    header(&mut b, srs, Some(envelope(rings[0].iter().copied())));
    b.push(1);
    b.extend_from_slice(&3u32.to_le_bytes());
    b.extend_from_slice(&(rings.len() as u32).to_le_bytes());
    for r in &rings {
        b.extend_from_slice(&(r.len() as u32).to_le_bytes());
        for &(x, y) in r {
            b.extend_from_slice(&nz(x).to_le_bytes());
            b.extend_from_slice(&nz(y).to_le_bytes());
        }
    }
    b
}

/// Encode all ESRI parts, grouping positive (hole) rings under their negative
/// exterior. Disconnected exteriors become MULTIPOLYGON, never discarded.
pub fn gpkg_esri_parts(srs: i32, rings: &[Vec<(f64, f64)>]) -> Vec<u8> {
    let exteriors: Vec<_> = rings.iter().filter(|r| ring_area(r) < 0.0).collect();
    let contains = |r: &[(f64, f64)], p: (f64, f64)| {
        let mut inside = false;
        for i in 0..r.len() {
            let (a,b) = (r[i],r[(i+1)%r.len()]);
            if (a.1 > p.1) != (b.1 > p.1) && p.0 < (b.0-a.0)*(p.1-a.1)/(b.1-a.1)+a.0 { inside = !inside; }
        }
        inside
    };
    let polys: Vec<_> = exteriors.iter().map(|e| {
        let holes = rings.iter().filter(|h| ring_area(h) > 0.0 && contains(e, h[0])).cloned().collect::<Vec<_>>();
        gpkg_polygon(srs,e,&holes)
    }).collect();
    if polys.len() == 1 { return polys.into_iter().next().unwrap(); }
    let mut out = Vec::new();
    header(&mut out,srs,Some(envelope(rings.iter().flatten().copied())));
    out.push(1); out.extend_from_slice(&6u32.to_le_bytes());
    out.extend_from_slice(&(polys.len() as u32).to_le_bytes());
    for p in polys { out.extend_from_slice(&p[40..]); }
    out
}

/// Decode the 2D vertices of a contract GPKG geometry (point, linestring or
/// polygon rings), used by the comparison tool and tests.
pub fn gpkg_rings(b: &[u8]) -> Option<Vec<Vec<(f64, f64)>>> {
    if b.len() < 8 || &b[..2] != b"GP" {
        return None;
    }
    let env = match (b[3] >> 1) & 7 {
        0 => 0,
        1 => 32,
        2 | 3 => 48,
        4 => 64,
        _ => return None,
    };
    let w = &b[8 + env..];
    let le = w[0] == 1;
    let u32at = |o: usize| -> Option<u32> {
        let s: [u8; 4] = w.get(o..o + 4)?.try_into().ok()?;
        Some(if le { u32::from_le_bytes(s) } else { u32::from_be_bytes(s) })
    };
    let f64at = |o: usize| -> Option<f64> {
        let s: [u8; 8] = w.get(o..o + 8)?.try_into().ok()?;
        Some(if le { f64::from_le_bytes(s) } else { f64::from_be_bytes(s) })
    };
    let ty = u32at(1)? % 1000;
    let mut o = 5;
    let ring = |o: &mut usize| -> Option<Vec<(f64, f64)>> {
        let n = u32at(*o)? as usize;
        *o += 4;
        let mut v = Vec::with_capacity(n);
        for _ in 0..n {
            v.push((f64at(*o)?, f64at(*o + 8)?));
            *o += 16;
        }
        Some(v)
    };
    match ty {
        1 => Some(vec![vec![(f64at(5)?, f64at(13)?)]]),
        2 => Some(vec![ring(&mut o)?]),
        3 => {
            let nr = u32at(o)? as usize;
            o += 4;
            (0..nr).map(|_| ring(&mut o)).collect()
        }
        5 => {
            // MULTILINESTRING: parts joined in order into one vertex list.
            let np = u32at(o)? as usize;
            o += 4;
            let mut all = Vec::new();
            for _ in 0..np {
                o += 5;
                all.extend(ring(&mut o)?);
            }
            Some(vec![all])
        }
        6 => {
            // MULTIPOLYGON: concatenate every part's rings.
            let np = u32at(o)? as usize;
            o += 4;
            let mut all = Vec::new();
            for _ in 0..np {
                o += 5;
                let nr = u32at(o)? as usize;
                o += 4;
                for _ in 0..nr {
                    all.push(ring(&mut o)?);
                }
            }
            Some(all)
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn native_catchment_preserves_islands_and_holes() {
        let outer = vec![(0.,0.),(0.,4.),(4.,4.),(4.,0.),(0.,0.)];
        let hole = vec![(1.,1.),(2.,1.),(2.,2.),(1.,2.),(1.,1.)];
        let island = vec![(5.,0.),(5.,1.),(6.,1.),(6.,0.),(5.,0.)];
        let parts = vec![outer, hole, island];
        let encoded = gpkg_esri_parts(9999, &parts);
        let g=crate::contract::Geometry(encoded.clone());
        let json=serde_json::to_string(&g).unwrap();
        assert_eq!(serde_json::from_str::<crate::contract::Geometry>(&json).unwrap(),g);
        assert!(serde_json::from_str::<crate::contract::Geometry>("\"é\"").is_err());
        assert_eq!(u32::from_le_bytes(encoded[41..45].try_into().unwrap()), 6);
        let decoded = gpkg_rings(&encoded).unwrap();
        assert_eq!(decoded.len(), 3);
        // Polygon writer normalizes winding and the starting vertex.
        for (actual, expected) in decoded.iter().zip(parts.iter()) {
            assert_eq!(actual.len(), expected.len());
            assert!(expected.iter().all(|v| actual.contains(v)));
            assert_eq!(ring_area(actual).abs(), ring_area(expected).abs());
        }
    }
    #[test]
    fn ellipsoid_total_area() {
        let total = quad_area_m2(-90.0, 90.0, 360.0);
        // 510,065,621.7 km2 for WGS84.
        assert!((total / 1e6 - 510_065_621.72).abs() < 1.0, "{total}");
    }
    #[test]
    fn vincenty_reference() {
        // Flinders Peak -> Buninyong (Vincenty 1975 test line), 54972.271 m.
        let d = geodesic_m(144.0 + 25.0 / 60.0 + 29.52440 / 3600.0, -(37.0 + 57.0 / 60.0 + 3.72030 / 3600.0), 143.0 + 55.0 / 60.0 + 35.38390 / 3600.0, -(37.0 + 39.0 / 60.0 + 10.15610 / 3600.0));
        assert!((d - 54972.271).abs() < 0.01, "{d}");
    }
    #[test]
    fn laea_origin_and_symmetry() {
        let p = Laea::new(49.0, 8.0);
        let (x, y) = p.forward(8.0, 49.0);
        assert!(x.abs() < 1e-9 && y.abs() < 1e-9);
        let (x1, _) = p.forward(9.0, 49.0);
        let (x2, _) = p.forward(7.0, 49.0);
        assert!((x1 + x2).abs() < 1e-6);
    }
}
