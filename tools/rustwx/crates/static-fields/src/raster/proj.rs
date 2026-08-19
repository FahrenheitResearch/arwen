//! LANE 3.  Map-projection transforms for the raster substrate,
//! transcribed from PROJ 9.5.1 (the version the Python geodesy stack
//! links on this box), spherical/ellipsoidal branches exactly as PROJ
//! selects them for the CRS strings the overlay path constructs:
//!
//! * `+proj=lcc +lat_0=lat_1 +R=6370000` — `lcc.cpp` sphere branch;
//! * `+proj=merc +lat_ts +R=6370000` — `merc.cpp` sphere branch;
//! * `+proj=stere +lat_0=±90 +lat_ts +R=6370000` — `stere.cpp` sphere
//!   polar modes;
//! * `+proj=igh +datum=WGS84` — `igh.cpp` (which forces `es = 0`, so
//!   the twelve sinusoidal/Mollweide zones run SPHERICAL on
//!   a = 6 378 137 m, WGS84's semimajor axis);
//! * the CONUS land-cover Albers — `aea.cpp` ellipsoidal branch on
//!   GRS80/NAD83.
//!
//! These are NOT the WPS `module_llxy` grid transforms (lane 1 owns
//! those); they replicate what `pyproj`/rasterio do when
//! `gpuwm/static/highres.py` builds a CRS with `_grid_crs` and warps
//! through it.  Validated against pyproj on committed fixture point
//! sets to <= 1e-6 m before any warp uses them
//! (`tests/highres_parity.rs`, validate-the-instrument).
//!
//! Every function takes/returns degrees on the geographic side and
//! metres on the projected side.  Points outside a projection's domain
//! come back as NaN (PROJ's HUGE_VAL discipline), and the warp treats
//! NaN as "does not participate".

use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};

const TWO_PI: f64 = 2.0 * PI;
const DEG_TO_RAD: f64 = PI / 180.0;
const RAD_TO_DEG: f64 = 180.0 / PI;
const EPS10: f64 = 1.0e-10;

/// `adjlon` (PROJ `adjlon.cpp`): reduce longitude to [-pi, pi].
fn adjlon(mut lon: f64) -> f64 {
    if lon.abs() <= PI {
        return lon;
    }
    lon += PI;
    lon -= TWO_PI * (lon / TWO_PI).floor();
    lon -= PI;
    lon
}

/// `aasin` (PROJ `aasincos.cpp`): arcsine clamped at |v| >= 1.
fn aasin(v: f64) -> f64 {
    if v.abs() >= 1.0 {
        return if v < 0.0 { -FRAC_PI_2 } else { FRAC_PI_2 };
    }
    v.asin()
}

/// One projected coordinate operation: geographic degrees <-> metres.
pub trait PointProjection: Sync {
    /// (lon_deg, lat_deg) -> (x_m, y_m); NaN outside the domain.
    fn forward(&self, lon_deg: f64, lat_deg: f64) -> (f64, f64);
    /// (x_m, y_m) -> (lon_deg, lat_deg); NaN outside the domain.
    fn inverse(&self, x_m: f64, y_m: f64) -> (f64, f64);
}

// ---------------------------------------------------------------------------
// Lambert conformal conic, sphere (PROJ lcc.cpp, es == 0, k0 == 1).
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct LccSphere {
    lam0: f64,
    n: f64,
    c: f64,
    rho0: f64,
    r: f64,
}

impl LccSphere {
    /// `+proj=lcc +lat_1 +lat_2 +lat_0 +lon_0 +R` (lat_0 == lat_1 in
    /// the overlay path's CRS strings, but lat_0 is honored as given).
    pub fn new(lat_1: f64, lat_2: f64, lat_0: f64, lon_0: f64, r: f64) -> Self {
        let phi1 = lat_1 * DEG_TO_RAD;
        let phi2 = lat_2 * DEG_TO_RAD;
        let phi0 = lat_0 * DEG_TO_RAD;
        let sinphi = phi1.sin();
        let cosphi = phi1.cos();
        let secant = (phi1 - phi2).abs() >= EPS10;
        let n = if secant {
            (cosphi / phi2.cos()).ln()
                / ((FRAC_PI_4 + 0.5 * phi2).tan()
                    / (FRAC_PI_4 + 0.5 * phi1).tan())
                    .ln()
        } else {
            sinphi
        };
        let c = cosphi * (FRAC_PI_4 + 0.5 * phi1).tan().powf(n) / n;
        let rho0 = if (phi0.abs() - FRAC_PI_2).abs() < EPS10 {
            0.0
        } else {
            c * (FRAC_PI_4 + 0.5 * phi0).tan().powf(-n)
        };
        LccSphere { lam0: lon_0 * DEG_TO_RAD, n, c, rho0, r }
    }
}

impl PointProjection for LccSphere {
    fn forward(&self, lon_deg: f64, lat_deg: f64) -> (f64, f64) {
        let phi = lat_deg * DEG_TO_RAD;
        let mut lam = adjlon(lon_deg * DEG_TO_RAD - self.lam0);
        let rho = if (phi.abs() - FRAC_PI_2).abs() < EPS10 {
            if phi * self.n <= 0.0 {
                return (f64::NAN, f64::NAN);
            }
            0.0
        } else {
            self.c * (FRAC_PI_4 + 0.5 * phi).tan().powf(-self.n)
        };
        lam *= self.n;
        (self.r * (rho * lam.sin()), self.r * (self.rho0 - rho * lam.cos()))
    }

    fn inverse(&self, x_m: f64, y_m: f64) -> (f64, f64) {
        let mut x = x_m / self.r;
        let mut y = self.rho0 - y_m / self.r;
        let mut rho = x.hypot(y);
        let (phi, lam);
        if rho != 0.0 {
            if self.n < 0.0 {
                rho = -rho;
                x = -x;
                y = -y;
            }
            phi = 2.0 * (self.c / rho).powf(1.0 / self.n).atan() - FRAC_PI_2;
            lam = x.atan2(y) / self.n;
        } else {
            lam = 0.0;
            phi = if self.n > 0.0 { FRAC_PI_2 } else { -FRAC_PI_2 };
        }
        (adjlon(lam + self.lam0) * RAD_TO_DEG, phi * RAD_TO_DEG)
    }
}

// ---------------------------------------------------------------------------
// Mercator, sphere with lat_ts (PROJ merc.cpp, es == 0).
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct MercSphere {
    lam0: f64,
    k0: f64,
    r: f64,
}

impl MercSphere {
    pub fn new(lat_ts: f64, lon_0: f64, r: f64) -> Self {
        MercSphere {
            lam0: lon_0 * DEG_TO_RAD,
            k0: (lat_ts.abs() * DEG_TO_RAD).cos(),
            r,
        }
    }
}

impl PointProjection for MercSphere {
    fn forward(&self, lon_deg: f64, lat_deg: f64) -> (f64, f64) {
        let phi = lat_deg * DEG_TO_RAD;
        if (phi.abs() - FRAC_PI_2).abs() <= 1.0e-10 {
            return (f64::NAN, f64::NAN);
        }
        let lam = adjlon(lon_deg * DEG_TO_RAD - self.lam0);
        (self.r * self.k0 * lam, self.r * self.k0 * phi.tan().asinh())
    }

    fn inverse(&self, x_m: f64, y_m: f64) -> (f64, f64) {
        let phi = (y_m / self.r / self.k0).sinh().atan();
        let lam = x_m / self.r / self.k0;
        (adjlon(lam + self.lam0) * RAD_TO_DEG, phi * RAD_TO_DEG)
    }
}

// ---------------------------------------------------------------------------
// Polar stereographic, sphere with lat_ts (PROJ stere.cpp, es == 0).
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct StereSpherePolar {
    lam0: f64,
    akm1: f64,
    north: bool,
    r: f64,
}

impl StereSpherePolar {
    /// `+proj=stere +lat_0=±90 +lat_ts +lon_0 +R` (k0 == 1).
    pub fn new(north: bool, lat_ts: f64, lon_0: f64, r: f64) -> Self {
        let phits = (lat_ts * DEG_TO_RAD).abs();
        let akm1 = if (phits - FRAC_PI_2).abs() >= EPS10 {
            phits.cos() / (FRAC_PI_4 - 0.5 * phits).tan()
        } else {
            2.0
        };
        StereSpherePolar { lam0: lon_0 * DEG_TO_RAD, akm1, north, r }
    }
}

impl PointProjection for StereSpherePolar {
    fn forward(&self, lon_deg: f64, lat_deg: f64) -> (f64, f64) {
        let lam = adjlon(lon_deg * DEG_TO_RAD - self.lam0);
        let mut phi = lat_deg * DEG_TO_RAD;
        let mut coslam = lam.cos();
        let sinlam = lam.sin();
        if self.north {
            coslam = -coslam;
            phi = -phi;
        }
        if (phi - FRAC_PI_2).abs() < 1.0e-8 {
            return (f64::NAN, f64::NAN);
        }
        let t = self.akm1 * (FRAC_PI_4 + 0.5 * phi).tan();
        (self.r * (sinlam * t), self.r * (t * coslam))
    }

    fn inverse(&self, x_m: f64, y_m: f64) -> (f64, f64) {
        let x = x_m / self.r;
        let mut y = y_m / self.r;
        let rh = x.hypot(y);
        let c = 2.0 * (rh / self.akm1).atan();
        let cosc = c.cos();
        if self.north {
            y = -y;
        }
        let phi = if rh.abs() <= EPS10 {
            if self.north { FRAC_PI_2 } else { -FRAC_PI_2 }
        } else if self.north {
            aasin(cosc)
        } else {
            aasin(-cosc)
        };
        let lam =
            if x == 0.0 && y == 0.0 { 0.0 } else { x.atan2(y) };
        (adjlon(lam + self.lam0) * RAD_TO_DEG, phi * RAD_TO_DEG)
    }
}

// ---------------------------------------------------------------------------
// Interrupted Goode Homolosine (PROJ igh.cpp: es forced 0, twelve
// sinusoidal/Mollweide zones on the sphere; a = WGS84 semimajor for
// the `+datum=WGS84` string the soil source records).
// ---------------------------------------------------------------------------

/// WGS84 semimajor axis: the sphere radius `+proj=igh +datum=WGS84`
/// runs on after igh's own `es = 0` override.
pub const IGH_WGS84_A: f64 = 6_378_137.0;

/// Mollweide constants for p = pi/2, computed with PROJ moll.cpp's own
/// `setup` arithmetic so every bit matches the reference (`sp = sin(p)`,
/// `r = sqrt(2*pi*sp / (2p + sin(2p)))`, `C_x = 2r/pi`, `C_y = r/sp`,
/// `C_p = 2p + sin(2p)`).
fn moll_constants() -> (f64, f64, f64) {
    let p = FRAC_PI_2;
    let p2 = p + p;
    let sp = p.sin();
    let r = (TWO_PI * sp / (p2 + p2.sin())).sqrt();
    (2.0 * r / PI, r / sp, p2 + p2.sin())
}

fn moll_forward(lam: f64, phi: f64) -> (f64, f64) {
    let (c_x, c_y, c_p) = moll_constants();
    let k = c_p * phi.sin();
    let mut theta = phi;
    let mut converged = false;
    for _ in 0..30 {
        let v = (theta + theta.sin() - k) / (1.0 + theta.cos());
        theta -= v;
        if v.abs() < 1.0e-7 {
            converged = true;
            break;
        }
    }
    if !converged {
        theta = if theta < 0.0 { -FRAC_PI_2 } else { FRAC_PI_2 };
    } else {
        theta *= 0.5;
    }
    (c_x * lam * theta.cos(), c_y * theta.sin())
}

fn moll_inverse(x: f64, y: f64) -> (f64, f64) {
    let (c_x, c_y, c_p) = moll_constants();
    let mut theta = aasin(y / c_y);
    let lam = x / (c_x * theta.cos());
    if lam.abs() < PI {
        theta += theta;
        let phi = aasin((theta + theta.sin()) / c_p);
        (lam, phi)
    } else {
        (f64::NAN, f64::NAN)
    }
}

/// Sinusoidal sphere with n = 1, m = 0 (PROJ gn_sinu.cpp via sinu).
fn sinu_forward(lam: f64, phi: f64) -> (f64, f64) {
    (lam * phi.cos(), phi)
}

fn sinu_inverse(x: f64, y: f64) -> (f64, f64) {
    (x / y.cos(), y)
}

/// Zone latitude boundary: 40 deg 44' 11.8".
const IGH_PHI_BOUNDARY: f64 =
    (40.0 + 44.0 / 60.0 + 11.8 / 3600.0) * DEG_TO_RAD;
const IGH_EPSLN: f64 = 1.0e-10;

const D10: f64 = 10.0 * DEG_TO_RAD;
const D20: f64 = 20.0 * DEG_TO_RAD;
const D30: f64 = 30.0 * DEG_TO_RAD;
const D40: f64 = 40.0 * DEG_TO_RAD;
const D50: f64 = 50.0 * DEG_TO_RAD;
const D60: f64 = 60.0 * DEG_TO_RAD;
const D80: f64 = 80.0 * DEG_TO_RAD;
const D90: f64 = 90.0 * DEG_TO_RAD;
const D100: f64 = 100.0 * DEG_TO_RAD;
const D140: f64 = 140.0 * DEG_TO_RAD;
const D160: f64 = 160.0 * DEG_TO_RAD;
const D180: f64 = 180.0 * DEG_TO_RAD;

#[derive(Debug, Clone, Copy)]
struct IghZone {
    moll: bool,
    lam0: f64,
    x0: f64,
    y0: f64,
}

#[derive(Debug, Clone)]
pub struct Igh {
    zones: [IghZone; 12],
    dy0: f64,
    a: f64,
}

impl Igh {
    pub fn new(a: f64) -> Self {
        // dy0: moll and sinu meet at the boundary latitude (igh.cpp).
        let (_, y_moll) = moll_forward(0.0, IGH_PHI_BOUNDARY);
        let (_, y_sinu) = sinu_forward(0.0, IGH_PHI_BOUNDARY);
        let dy0 = y_sinu - y_moll;
        let z = |moll: bool, lam0: f64, x0: f64, y0: f64| IghZone {
            moll,
            lam0,
            x0,
            y0,
        };
        Igh {
            zones: [
                z(true, -D100, -D100, dy0),  // 1
                z(true, D30, D30, dy0),      // 2
                z(false, -D100, -D100, 0.0), // 3
                z(false, D30, D30, 0.0),     // 4
                z(false, -D160, -D160, 0.0), // 5
                z(false, -D60, -D60, 0.0),   // 6
                z(false, D20, D20, 0.0),     // 7
                z(false, D140, D140, 0.0),   // 8
                z(true, -D160, -D160, -dy0), // 9
                z(true, -D60, -D60, -dy0),   // 10
                z(true, D20, D20, -dy0),     // 11
                z(true, D140, D140, -dy0),   // 12
            ],
            dy0,
            a,
        }
    }
}

impl PointProjection for Igh {
    fn forward(&self, lon_deg: f64, lat_deg: f64) -> (f64, f64) {
        let lam = adjlon(lon_deg * DEG_TO_RAD);
        let phi = lat_deg * DEG_TO_RAD;
        let z = if phi >= IGH_PHI_BOUNDARY {
            if lam <= -D40 { 1 } else { 2 }
        } else if phi >= 0.0 {
            if lam <= -D40 { 3 } else { 4 }
        } else if phi >= -IGH_PHI_BOUNDARY {
            if lam <= -D100 {
                5
            } else if lam <= -D20 {
                6
            } else if lam <= D80 {
                7
            } else {
                8
            }
        } else if lam <= -D100 {
            9
        } else if lam <= -D20 {
            10
        } else if lam <= D80 {
            11
        } else {
            12
        };
        let zone = self.zones[z - 1];
        let lam_z = lam - zone.lam0;
        let (x, y) = if zone.moll {
            moll_forward(lam_z, phi)
        } else {
            sinu_forward(lam_z, phi)
        };
        (self.a * (x + zone.x0), self.a * (y + zone.y0))
    }

    fn inverse(&self, x_m: f64, y_m: f64) -> (f64, f64) {
        let x = x_m / self.a;
        let y = y_m / self.a;
        let y90 = self.dy0 + std::f64::consts::SQRT_2;
        let z: usize = if y > y90 + IGH_EPSLN || y < -y90 + IGH_EPSLN {
            0
        } else if y >= IGH_PHI_BOUNDARY {
            if x <= -D40 { 1 } else { 2 }
        } else if y >= 0.0 {
            if x <= -D40 { 3 } else { 4 }
        } else if y >= -IGH_PHI_BOUNDARY {
            if x <= -D100 {
                5
            } else if x <= -D20 {
                6
            } else if x <= D80 {
                7
            } else {
                8
            }
        } else if x <= -D100 {
            9
        } else if x <= -D20 {
            10
        } else if x <= D80 {
            11
        } else {
            12
        };
        if z == 0 {
            return (f64::NAN, f64::NAN);
        }
        let zone = self.zones[z - 1];
        let (mut lam, phi) = if zone.moll {
            moll_inverse(x - zone.x0, y - zone.y0)
        } else {
            sinu_inverse(x - zone.x0, y - zone.y0)
        };
        if lam.is_nan() || phi.is_nan() {
            return (f64::NAN, f64::NAN);
        }
        lam += zone.lam0;
        let e = IGH_EPSLN;
        let ok = match z {
            1 => {
                (lam >= -D180 - e && lam <= -D40 + e)
                    || ((lam >= -D40 - e && lam <= -D10 + e)
                        && (phi >= D60 - e && phi <= D90 + e))
            }
            2 => {
                (lam >= -D40 - e && lam <= D180 + e)
                    || ((lam >= -D180 - e && lam <= -D160 + e)
                        && (phi >= D50 - e && phi <= D90 + e))
                    || ((lam >= -D50 - e && lam <= -D40 + e)
                        && (phi >= D60 - e && phi <= D90 + e))
            }
            3 => lam >= -D180 - e && lam <= -D40 + e,
            4 => lam >= -D40 - e && lam <= D180 + e,
            5 | 9 => lam >= -D180 - e && lam <= -D100 + e,
            6 | 10 => lam >= -D100 - e && lam <= -D20 + e,
            7 | 11 => lam >= -D20 - e && lam <= D80 + e,
            8 | 12 => lam >= D80 - e && lam <= D180 + e,
            _ => false,
        };
        if !ok {
            return (f64::NAN, f64::NAN);
        }
        (lam * RAD_TO_DEG, phi * RAD_TO_DEG)
    }
}

// ---------------------------------------------------------------------------
// Albers equal area, ellipsoid (PROJ aea.cpp): the CONUS land-cover
// collection's projection on GRS80/NAD83.
// ---------------------------------------------------------------------------

/// GRS80 semimajor axis / inverse flattening (NAD83's ellipsoid).
pub const GRS80_A: f64 = 6_378_137.0;
pub const GRS80_RF: f64 = 298.257_222_101;

fn msfn(sinphi: f64, cosphi: f64, es: f64) -> f64 {
    cosphi / (1.0 - es * sinphi * sinphi).sqrt()
}

fn qsfn(sinphi: f64, e: f64, one_es: f64) -> f64 {
    if e >= 1.0e-7 {
        let con = e * sinphi;
        let div1 = 1.0 - con * con;
        let div2 = 1.0 + con;
        if div1 == 0.0 || div2 == 0.0 {
            return f64::NAN;
        }
        one_es * (sinphi / div1 - (0.5 / e) * ((1.0 - con) / div2).ln())
    } else {
        sinphi + sinphi
    }
}

/// `phi1_` iteration from aea.cpp.
fn aea_phi1(qs: f64, te: f64, tone_es: f64) -> f64 {
    let mut phi = (0.5 * qs).asin();
    if te < 1.0e-7 {
        return phi;
    }
    for _ in 0..=15 {
        let sinpi = phi.sin();
        let cospi = phi.cos();
        let con = te * sinpi;
        let com = 1.0 - con * con;
        let dphi = 0.5 * com * com / cospi
            * (qs / tone_es - sinpi / com
                + 0.5 / te * ((1.0 - con) / (1.0 + con)).ln());
        phi += dphi;
        if dphi.abs() <= 1.0e-10 {
            return phi;
        }
    }
    f64::NAN
}

#[derive(Debug, Clone)]
pub struct AeaEllipsoid {
    lam0: f64,
    e: f64,
    one_es: f64,
    n: f64,
    c: f64,
    dd: f64,
    ec: f64,
    rho0: f64,
    a: f64,
    x0: f64,
    y0: f64,
}

impl AeaEllipsoid {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        lat_1: f64,
        lat_2: f64,
        lat_0: f64,
        lon_0: f64,
        false_easting: f64,
        false_northing: f64,
        a: f64,
        rf: f64,
    ) -> Self {
        let f = 1.0 / rf;
        let es = f * (2.0 - f);
        let e = es.sqrt();
        let one_es = 1.0 - es;
        let phi1 = lat_1 * DEG_TO_RAD;
        let phi2 = lat_2 * DEG_TO_RAD;
        let phi0 = lat_0 * DEG_TO_RAD;
        let mut sinphi = phi1.sin();
        let cosphi = phi1.cos();
        let mut n = sinphi;
        let secant = (phi1 - phi2).abs() >= EPS10;
        let m1 = msfn(sinphi, cosphi, es);
        let ml1 = qsfn(sinphi, e, one_es);
        if secant {
            sinphi = phi2.sin();
            let cosphi2 = phi2.cos();
            let m2 = msfn(sinphi, cosphi2, es);
            let ml2 = qsfn(sinphi, e, one_es);
            n = (m1 * m1 - m2 * m2) / (ml2 - ml1);
        }
        let ec = 1.0 - 0.5 * one_es * ((1.0 - e) / (1.0 + e)).ln() / e;
        let c = m1 * m1 + n * ml1;
        let dd = 1.0 / n;
        let rho0 = dd * (c - n * qsfn(phi0.sin(), e, one_es)).sqrt();
        AeaEllipsoid {
            lam0: lon_0 * DEG_TO_RAD,
            e,
            one_es,
            n,
            c,
            dd,
            ec,
            rho0,
            a,
            x0: false_easting,
            y0: false_northing,
        }
    }
}

impl PointProjection for AeaEllipsoid {
    fn forward(&self, lon_deg: f64, lat_deg: f64) -> (f64, f64) {
        let phi = lat_deg * DEG_TO_RAD;
        let mut lam = adjlon(lon_deg * DEG_TO_RAD - self.lam0);
        let mut rho = self.c - self.n * qsfn(phi.sin(), self.e, self.one_es);
        if rho < 0.0 {
            return (f64::NAN, f64::NAN);
        }
        rho = self.dd * rho.sqrt();
        lam *= self.n;
        (
            self.a * (rho * lam.sin()) + self.x0,
            self.a * (self.rho0 - rho * lam.cos()) + self.y0,
        )
    }

    fn inverse(&self, x_m: f64, y_m: f64) -> (f64, f64) {
        let mut x = (x_m - self.x0) / self.a;
        let mut y = self.rho0 - (y_m - self.y0) / self.a;
        let mut rho = x.hypot(y);
        let (phi, lam);
        if rho != 0.0 {
            if self.n < 0.0 {
                rho = -rho;
                x = -x;
                y = -y;
            }
            let mut p = rho / self.dd;
            p = (self.c - p * p) / self.n;
            phi = if (self.ec - p.abs()).abs() > 1.0e-7 {
                if p.abs() > 2.0 {
                    return (f64::NAN, f64::NAN);
                }
                let it = aea_phi1(p, self.e, self.one_es);
                if it.is_nan() {
                    return (f64::NAN, f64::NAN);
                }
                it
            } else if p < 0.0 {
                -FRAC_PI_2
            } else {
                FRAC_PI_2
            };
            lam = x.atan2(y) / self.n;
        } else {
            lam = 0.0;
            phi = if self.n > 0.0 { FRAC_PI_2 } else { -FRAC_PI_2 };
        }
        (adjlon(lam + self.lam0) * RAD_TO_DEG, phi * RAD_TO_DEG)
    }
}

/// The identity "projection": geographic degrees on both sides.
#[derive(Debug, Clone, Copy)]
pub struct GeographicIdentity;

impl PointProjection for GeographicIdentity {
    fn forward(&self, lon_deg: f64, lat_deg: f64) -> (f64, f64) {
        (lon_deg, lat_deg)
    }
    fn inverse(&self, x: f64, y: f64) -> (f64, f64) {
        (x, y)
    }
}
