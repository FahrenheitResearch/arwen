//! GeoPackage polygon decoding, ring clipping and ring areas.
//!
//! Areas are signed (counter-clockwise positive). A polygon's area is the sum
//! over its rings with the exterior's orientation taken as positive, so holes
//! subtract whatever way the source wound them.
use crate::Result;

pub type Pt = [f64; 2];
pub type Ring = Vec<Pt>;
/// Exterior ring first, then holes.
pub type Polygon = Vec<Ring>;

struct Cursor<'a> {
    b: &'a [u8],
    at: usize,
    le: bool,
}
impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let s = self.b.get(self.at..self.at + n).ok_or("truncated WKB")?;
        self.at += n;
        Ok(s)
    }
    fn u32(&mut self) -> Result<u32> {
        let s: [u8; 4] = self.take(4)?.try_into()?;
        Ok(if self.le { u32::from_le_bytes(s) } else { u32::from_be_bytes(s) })
    }
    fn f64(&mut self) -> Result<f64> {
        let s: [u8; 8] = self.take(8)?.try_into()?;
        Ok(if self.le { f64::from_le_bytes(s) } else { f64::from_be_bytes(s) })
    }
}

/// Decode a GeoPackage binary geometry holding POLYGON or MULTIPOLYGON.
pub fn parse_gpkg(blob: &[u8]) -> Result<Vec<Polygon>> {
    if blob.len() < 8 || &blob[0..2] != b"GP" {
        return Err("not a GeoPackage geometry blob".into());
    }
    let flags = blob[3];
    if flags & 0x10 != 0 {
        return Ok(Vec::new()); // empty geometry
    }
    let env = match (flags >> 1) & 0x07 {
        0 => 0,
        1 => 32,
        2 | 3 => 48,
        4 => 64,
        e => return Err(format!("invalid GeoPackage envelope code {e}").into()),
    };
    let mut c = Cursor { b: &blob[8 + env..], at: 0, le: true };
    read_geometry(&mut c)
}

fn read_geometry(c: &mut Cursor) -> Result<Vec<Polygon>> {
    c.le = match c.take(1)?[0] {
        1 => true,
        0 => false,
        o => return Err(format!("bad WKB byte order {o}").into()),
    };
    let raw = c.u32()?;
    // ISO (1000/2000/3000 offsets) and EWKB (high flag bits) dimension markers.
    let ewkb_z = raw & 0x8000_0000 != 0;
    let ewkb_m = raw & 0x4000_0000 != 0;
    let base = raw & 0x0FFF_FFFF;
    let (kind, iso) = (base % 1000, base / 1000);
    let dims = 2 + usize::from(ewkb_z || iso == 1 || iso == 3) + usize::from(ewkb_m || iso == 2 || iso == 3);
    if raw & 0x2000_0000 != 0 {
        c.take(4)?; // EWKB SRID
    }
    match kind {
        3 => Ok(vec![read_polygon(c, dims)?]),
        6 => {
            let n = c.u32()? as usize;
            let mut out = Vec::with_capacity(n);
            for _ in 0..n {
                out.extend(read_geometry(c)?);
            }
            Ok(out)
        }
        k => Err(format!("divide geometry type {k} is not POLYGON or MULTIPOLYGON").into()),
    }
}

fn read_polygon(c: &mut Cursor, dims: usize) -> Result<Polygon> {
    let rings = c.u32()? as usize;
    let mut poly = Vec::with_capacity(rings);
    for _ in 0..rings {
        let n = c.u32()? as usize;
        let mut ring = Vec::with_capacity(n);
        for _ in 0..n {
            let x = c.f64()?;
            let y = c.f64()?;
            for _ in 2..dims {
                c.f64()?;
            }
            ring.push([x, y]);
        }
        poly.push(ring);
    }
    Ok(poly)
}

/// Encode one polygon as a GeoPackage blob (little-endian, XY envelope).
pub fn encode_gpkg_polygon(poly: &Polygon, srs_id: i32) -> Vec<u8> {
    let (mut x0, mut y0, mut x1, mut y1) = (f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
    for p in poly.iter().flatten() {
        x0 = x0.min(p[0]);
        x1 = x1.max(p[0]);
        y0 = y0.min(p[1]);
        y1 = y1.max(p[1]);
    }
    let mut b = vec![b'G', b'P', 0, 0b0000_0011];
    b.extend_from_slice(&srs_id.to_le_bytes());
    for v in [x0, x1, y0, y1] {
        b.extend_from_slice(&v.to_le_bytes());
    }
    b.push(1);
    b.extend_from_slice(&3u32.to_le_bytes());
    b.extend_from_slice(&(poly.len() as u32).to_le_bytes());
    for ring in poly {
        b.extend_from_slice(&(ring.len() as u32).to_le_bytes());
        for p in ring {
            b.extend_from_slice(&p[0].to_le_bytes());
            b.extend_from_slice(&p[1].to_le_bytes());
        }
    }
    b
}

/// Planar signed shoelace area (counter-clockwise positive).
pub fn planar_area(ring: &[Pt]) -> f64 {
    let n = ring.len();
    if n < 3 {
        return 0.0;
    }
    let mut s = 0.0;
    for i in 0..n {
        let a = ring[i];
        let b = ring[(i + 1) % n];
        s += a[0] * b[1] - b[0] * a[1];
    }
    0.5 * s
}

/// Mean radius of the authalic sphere for WGS84 (m), so lat/lon areas are in m2.
pub const AUTHALIC_RADIUS_M: f64 = 6_371_007.181;

/// WGS84 first eccentricity squared.
const E2: f64 = 0.006_694_379_990_141_317;

fn q_of(sin_phi: f64) -> f64 {
    let e = E2.sqrt();
    (1.0 - E2) * (sin_phi / (1.0 - E2 * sin_phi * sin_phi) - (1.0 / (2.0 * e)) * ((1.0 - e * sin_phi) / (1.0 + e * sin_phi)).ln())
}

/// Authalic latitude (degrees) of a WGS84 geodetic latitude: on the authalic
/// sphere, areas between authalic parallels equal the ellipsoid's.
pub fn authalic_lat_deg(lat_deg: f64) -> f64 {
    let d = std::f64::consts::PI / 180.0;
    let qp = q_of(1.0);
    ((q_of((lat_deg * d).sin()) / qp).clamp(-1.0, 1.0)).asin() / d
}

/// Signed area on the WGS84 ellipsoid of a ring whose edges are straight in
/// (longitude, geodetic latitude) degrees, through the authalic sphere:
/// A = -R_q^2 * sum of the exact integral of sin(beta) d(lon) along each edge.
/// Exact for cell-clipped edges (constant longitude or latitude); polygon
/// edges are densified first, so their beta curvature is negligible.
pub fn lonlat_area(ring: &[Pt]) -> f64 {
    let r: Vec<Pt> = ring.iter().map(|p| [p[0], authalic_lat_deg(p[1])]).collect();
    sphere_lonlat_area(&r)
}

/// Same on the authalic sphere for coordinates already in authalic latitude.
pub fn sphere_lonlat_area(ring: &[Pt]) -> f64 {
    let n = ring.len();
    if n < 3 {
        return 0.0;
    }
    let d = std::f64::consts::PI / 180.0;
    let mut s = 0.0;
    for i in 0..n {
        let (l1, p1) = (ring[i][0] * d, ring[i][1] * d);
        let (l2, p2) = (ring[(i + 1) % n][0] * d, ring[(i + 1) % n][1] * d);
        let dp = p2 - p1;
        let half = 0.5 * dp;
        let sinc = if half.abs() < 1e-8 { 1.0 - half * half / 6.0 } else { half.sin() / half };
        s -= (l2 - l1) * (0.5 * (p1 + p2)).sin() * sinc;
    }
    s * AUTHALIC_RADIUS_M * AUTHALIC_RADIUS_M
}

/// Sutherland-Hodgman clip of a ring to an axis-aligned half plane.
/// axis 0 = x, 1 = y; keep values >= bound when keep_above, else <= bound.
fn clip_half(ring: &[Pt], axis: usize, bound: f64, keep_above: bool) -> Ring {
    let inside = |p: &Pt| if keep_above { p[axis] >= bound } else { p[axis] <= bound };
    let mut out = Vec::with_capacity(ring.len() + 4);
    let n = ring.len();
    for i in 0..n {
        let cur = ring[i];
        let prev = ring[(i + n - 1) % n];
        let (ci, pi) = (inside(&cur), inside(&prev));
        if ci != pi {
            let t = (bound - prev[axis]) / (cur[axis] - prev[axis]);
            let mut q = [prev[0] + t * (cur[0] - prev[0]), prev[1] + t * (cur[1] - prev[1])];
            q[axis] = bound;
            out.push(q);
        }
        if ci {
            out.push(cur);
        }
    }
    out
}

pub fn clip_band(ring: &[Pt], axis: usize, lo: f64, hi: f64) -> Ring {
    let a = clip_half(ring, axis, lo, true);
    if a.len() < 3 {
        return Vec::new();
    }
    clip_half(&a, axis, hi, false)
}

/// Drop the closing duplicate vertex GeoPackage rings carry.
pub fn open_ring(mut r: Ring) -> Ring {
    if r.len() > 1 && r.first() == r.last() {
        r.pop();
    }
    r
}

/// Insert vertices so no edge is longer than max_len (source units).
pub fn densify(ring: &[Pt], max_len: f64) -> Ring {
    let n = ring.len();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let a = ring[i];
        let b = ring[(i + 1) % n];
        out.push(a);
        let len = ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2)).sqrt();
        let k = (len / max_len).ceil() as usize;
        for j in 1..k {
            let t = j as f64 / k as f64;
            out.push([a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn lonlat_area_matches_band_formula() {
        // 1 x 1 degree cell at 45-46N: R^2 * dlon * (sin 46 - sin 45)
        let r = vec![[10.0, 45.0], [11.0, 45.0], [11.0, 46.0], [10.0, 46.0]];
        let d = std::f64::consts::PI / 180.0;
        let exact = AUTHALIC_RADIUS_M.powi(2) * d * ((46.0 * d).sin() - (45.0 * d).sin());
        assert!((sphere_lonlat_area(&r) - exact).abs() / exact < 1e-12);
        let cw: Vec<_> = r.iter().rev().copied().collect();
        assert!((sphere_lonlat_area(&cw) + exact).abs() / exact < 1e-12);
        // ellipsoid: R_q^2 * dlon * (sin beta(46) - sin beta(45))
        let b = |x: f64| (authalic_lat_deg(x) * d).sin();
        let ell = AUTHALIC_RADIUS_M.powi(2) * d * (b(46.0) - b(45.0));
        assert!((lonlat_area(&r) - ell).abs() / ell < 1e-12);
        assert!((authalic_lat_deg(90.0) - 90.0).abs() < 1e-9 && authalic_lat_deg(0.0).abs() < 1e-12);
    }
    #[test]
    fn clip_square_into_quarters() {
        let r = vec![[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]];
        let mut total = 0.0;
        for (xl, xh) in [(0.0, 1.0), (1.0, 2.0)] {
            for (yl, yh) in [(0.0, 1.0), (1.0, 2.0)] {
                let a = clip_band(&clip_band(&r, 1, yl, yh), 0, xl, xh);
                assert!((planar_area(&a) - 1.0).abs() < 1e-15);
                total += planar_area(&a);
            }
        }
        assert_eq!(total, 4.0);
    }
    #[test]
    fn gpkg_roundtrip() {
        let p = vec![vec![[0.0, 0.0], [3.0, 0.0], [3.0, 2.0], [0.0, 2.0], [0.0, 0.0]]];
        let b = encode_gpkg_polygon(&p, 100001);
        assert_eq!(parse_gpkg(&b).unwrap(), vec![p]);
    }
}
