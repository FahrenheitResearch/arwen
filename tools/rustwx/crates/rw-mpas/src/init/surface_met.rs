//! Case-8 intermediate decoder and projections, seeded from init/metfile.rs.
//! `mpas_init_atm_read_met.F:97-412`, `mpas_init_atm_surface.F:320-349`.
//! Formats 3, 4 and 5; lat/lon, Gaussian and polar stereographic grids.
use super::metfile::MetSlab;
use crate::error::{MpasError, MpasResult};
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

#[derive(Debug)]
pub(super) struct SurfaceSlab {
    pub met: MetSlab,
    pub projection: Projection,
}
#[derive(Debug)]
pub(super) enum Projection {
    LatLon,
    Gaussian { nlat: usize },
    Polar { dx: f32, stdlon: f32, truelat1: f32 },
    Unmapped(i32),
}
fn refusal(s: impl Into<String>) -> MpasError {
    MpasError::Refusal(s.into())
}
/// `mpas_init_atm_read_met.F:115-411`: big-endian sequential records.
fn record(r: &mut impl Read) -> MpasResult<Option<Vec<u8>>> {
    let mut head = [0; 4];
    let n = r.read(&mut head[..1])?;
    if n == 0 {
        return Ok(None);
    }
    r.read_exact(&mut head[1..])?;
    let size = u32::from_be_bytes(head) as usize;
    let mut data = vec![0; size];
    r.read_exact(&mut data)?;
    let mut tail = [0; 4];
    r.read_exact(&mut tail)?;
    if head != tail {
        return Err(refusal(
            "surface intermediate record markers disagree; the slab framing is corrupt",
        ));
    }
    Ok(Some(data))
}
struct Cursor<'a> {
    b: &'a [u8],
    at: usize,
}
impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> MpasResult<&'a [u8]> {
        if self.at + n > self.b.len() {
            return Err(refusal("truncated surface intermediate header"));
        }
        let b = &self.b[self.at..self.at + n];
        self.at += n;
        Ok(b)
    }
    fn text(&mut self, n: usize) -> MpasResult<String> {
        Ok(String::from_utf8_lossy(self.take(n)?).trim().into())
    }
    fn f32(&mut self) -> MpasResult<f32> {
        Ok(f32::from_be_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn i32(&mut self) -> MpasResult<i32> {
        Ok(i32::from_be_bytes(self.take(4)?.try_into().unwrap()))
    }
}
fn next(r: &mut impl Read) -> MpasResult<Vec<u8>> {
    record(r)?.ok_or_else(|| refusal("surface intermediate ends inside a field"))
}
/// `mpas_init_atm_read_met.F:97-412`: version-specific header layout,
/// projection-code translation, km-to-m conversion and longitude normalization.
pub(super) fn read_surface_file(path: &Path) -> MpasResult<Vec<SurfaceSlab>> {
    let mut r = BufReader::new(File::open(path)?);
    let mut out = Vec::new();
    while let Some(v) = record(&mut r)? {
        let version = Cursor { b: &v, at: 0 }.i32()?;
        if !(3..=5).contains(&version) {
            return Err(refusal(format!(
                "surface intermediate version {version} is not native format 3, 4 or 5"
            )));
        }
        let header = next(&mut r)?;
        let mut h = Cursor { b: &header, at: 0 };
        let hdate = h.text(24)?;
        let xfcst = h.f32()?;
        let map_source = if version >= 4 {
            h.text(32)?
        } else {
            String::new()
        };
        let mut field = h.text(9)?;
        if field == "HGT" {
            field = "GHT".into();
        }
        let units = h.text(25)?;
        let desc = h.text(46)?;
        let xlvl = h.f32()?;
        let nx = h.i32()?;
        let ny = h.i32()?;
        let code = h.i32()?;
        if nx < 1 || ny < 1 {
            return Err(refusal("surface slabs need positive dimensions"));
        }
        let nx = nx as usize;
        let ny = ny as usize;
        let pr = next(&mut r)?;
        let mut p = Cursor { b: &pr, at: 0 };
        if version >= 4 {
            let _startloc = p.text(8)?;
        }
        let mut start_lat = p.f32()?;
        let mut start_lon = p.f32()?;
        let mut delta_lat = 0.0;
        let mut delta_lon = 0.0;
        let projection = match code {
            0 => {
                delta_lat = p.f32()?;
                delta_lon = p.f32()?;
                Projection::LatLon
            }
            4 if version == 5 => {
                delta_lat = p.f32()?;
                delta_lon = p.f32()?;
                Projection::Gaussian { nlat: delta_lat.round() as usize }
            }
            5 => {
                let dx = p.f32()? * 1000.0;
                let _dy = p.f32()?;
                let mut stdlon = p.f32()?;
                let truelat1 = p.f32()?;
                if stdlon > 180.0 { stdlon -= 360.0; }
                Projection::Polar { dx, stdlon, truelat1 }
            }
            1 | 3 => {
                for _ in 0..if code == 1 { 3 } else { 5 } { let _ = p.f32()?; }
                Projection::Unmapped(code)
            }
            _ => return Err(refusal(format!("projection {code} is not recognized by the native intermediate reader for format {version}"))),
        };
        let earth_radius_km = if version == 5 { p.f32()? } else { 6370.0 };
        if start_lon > 180.0 {
            start_lon -= 360.0;
        }
        if start_lat < -90.0 {
            start_lat = -90.0;
        }
        if start_lat > 90.0 {
            start_lat = 90.0;
        }
        let mut wind_grid_rel = false;
        if version == 5 {
            let wind = next(&mut r)?;
            if wind.len() != 4 {
                return Err(refusal("invalid surface wind-relative logical record"));
            }
            wind_grid_rel = i32::from_be_bytes(wind.try_into().unwrap()) != 0;
        }
        let data = next(&mut r)?;
        let want = nx
            .checked_mul(ny)
            .and_then(|v| v.checked_mul(4))
            .ok_or_else(|| refusal("surface dimensions overflow slab length"))?;
        if data.len() != want {
            return Err(refusal(format!(
                "surface field {field} has {} slab bytes, expected {want}",
                data.len()
            )));
        }
        let values = data
            .chunks_exact(4)
            .map(|b| f32::from_be_bytes(b.try_into().unwrap()))
            .collect();
        out.push(SurfaceSlab {
            met: MetSlab {
                // Case 8 uses SurfaceSlab::projection, not the case-7 projection
                // metadata. These additional i1 fields do not enter its data path.
                format_version: version,
                xfcst,
                start_i: 1.0,
                start_j: 1.0,
                wind_grid_rel,
                dx: 0.0,
                dy: 0.0,
                stdlon: 0.0,
                truelat1: 0.0,
                truelat2: 0.0,
                iproj: code,
                polar: None,
                hdate,
                map_source,
                field,
                units,
                desc,
                xlvl,
                nx,
                ny,
                start_lat,
                start_lon,
                delta_lat,
                delta_lon,
                earth_radius_km,
                values,
            },
            projection,
        });
    }
    Ok(out)
}

// Host data path uses the host's glibc routines, matching strict gfortran.
#[cfg_attr(unix, link(name = "m"))]
unsafe extern "C" {
    fn sinf(x: f32) -> f32;
    fn cosf(x: f32) -> f32;
    fn sin(x: f64) -> f64;
    fn cos(x: f64) -> f64;
    fn acos(x: f64) -> f64;
}
const RAD: f32 = std::f32::consts::PI / 180.0;

pub(super) enum Map {
    LatLon {
        lat1: f32,
        lon1: f32,
        latinc: f32,
        loninc: f32,
    },
    Gaussian {
        latitudes: Vec<f32>,
        lon1: f32,
        loninc: f32,
    },
    Polar {
        hemi: f32,
        rebydx: f32,
        scale: f32,
        reflon: f32,
        polei: f32,
        polej: f32,
    },
}
/// `mpas_init_atm_llxy.F:445-567`: map_set wraps origin and standard
/// longitudes with at most ten turns and fixes the Earth radius at 6370000 m.
fn normalize(mut lon: f32) -> MpasResult<f32> {
    for _ in 0..10 {
        if lon.abs() <= 180.0 {
            break;
        }
        if lon < -180.0 {
            lon += 360.0;
        }
        if lon > 180.0 {
            lon -= 360.0;
        }
    }
    if lon.abs() > 180.0 {
        return Err(refusal(
            "projection longitude exceeds native ten-turn normalization",
        ));
    }
    Ok(lon)
}
impl Map {
    /// `mpas_init_atm_surface.F:320-349`; native set_ps and set_gauss.
    pub(super) fn new(field: &SurfaceSlab) -> MpasResult<Self> {
        let s = &field.met;
        let lon1 = normalize(s.start_lon)?;
        match field.projection {
            Projection::LatLon=>{
                if s.delta_lat==0.0 || s.delta_lon==0.0{return Err(refusal("zero lat/lon grid increment divides coordinates by zero"));}
                Ok(Self::LatLon{lat1:s.start_lat,lon1,latinc:s.delta_lat,loninc:s.delta_lon})
            }
            Projection::Gaussian{nlat}=>{
                if nlat==0 || nlat.checked_mul(2)!=Some(s.ny) {return Err(refusal("Gaussian nlat*2 must equal slab rows; otherwise latitude indices address different rows"));}
                let mut latitudes=gaussian(nlat*2);
                if (latitudes[0]-s.start_lat).abs()>0.01 {for lat in &mut latitudes{*lat= -1.0 * *lat;}}
                if (latitudes[0]-s.start_lat).abs()>0.01 {return Err(refusal("Gaussian start latitude fails native set_gauss 0.01 degree check (mpas_init_atm_llxy.F:1940-1950)"));}
                Ok(Self::Gaussian{latitudes,lon1,loninc:360.0/s.nx as f32})
            }
            Projection::Polar{dx,stdlon,truelat1}=>{
                if dx<=0.0 || truelat1.abs()>90.0{return Err(refusal("polar projection needs positive dx and a true latitude within 90 degrees (native map_set)"));}
                let hemi=if truelat1<0.0{-1.0}else{1.0};let rebydx=6370000.0/dx;let reflon=normalize(stdlon)?+90.0;
                let scale=1.0+hemi*unsafe{sinf(truelat1*RAD)};
                let ala=s.start_lat*RAD;let rsw=rebydx*unsafe{cosf(ala)}*scale/(1.0+hemi*unsafe{sinf(ala)});
                let alo=(lon1-reflon)*RAD;
                let polei=s.nx as f32/2.0-rsw*unsafe{cosf(alo)};
                let polej=s.ny as f32/2.0-hemi*rsw*unsafe{sinf(alo)};
                Ok(Self::Polar{hemi,rebydx,scale,reflon,polei,polej})
            }
            Projection::Unmapped(code)=>Err(refusal(format!("surface projection {code} is not initialized by native case 8; native map_init sets init=false and latlon_to_ij explicitly aborts before interpolation (mpas_init_atm_surface.F:324-349; mpas_init_atm_llxy.F:601-604)"))),
        }
    }
    /// `mpas_init_atm_llxy.F:729-765,1389-1416,2143-2222`.
    pub(super) fn ij(&self, lat: f32, lon: f32) -> (f32, f32) {
        match self {
            Self::LatLon {
                lat1,
                lon1,
                latinc,
                loninc,
            } => ((lon - lon1) / loninc + 1.0, (lat - lat1) / latinc + 1.0),
            Self::Polar {
                hemi,
                rebydx,
                scale,
                reflon,
                polei,
                polej,
            } => {
                let ala = lat * RAD;
                let rm =
                    rebydx * unsafe { cosf(ala) } * scale / (1.0 + hemi * unsafe { sinf(ala) });
                let alo = (lon - reflon) * RAD;
                (
                    polei + rm * unsafe { cosf(alo) },
                    polej + hemi * rm * unsafe { sinf(alo) },
                )
            }
            Self::Gaussian {
                latitudes: g,
                lon1,
                loninc,
            } => {
                let i = (lon - lon1) / loninc + 1.0;
                let n = g.len();
                let j = if lat.abs() > g[0].abs() {
                    if (lat - g[0]).abs() < (lat - g[n - 1]).abs() {
                        1.0
                    } else {
                        n as f32
                    }
                } else if let Some(k) = (0..n - 1).find(|&k| (g[k] - lat) * (g[k + 1] - lat) <= 0.0)
                {
                    ((g[k] - lat) * (k + 2) as f32 + (lat - g[k + 1]) * (k + 1) as f32)
                        / (g[k] - g[k + 1])
                } else {
                    f32::NAN
                };
                (i, j)
            }
        }
    }
}
/// `mpas_init_atm_llxy.F:2100-2140`: normalized Legendre polynomial.
fn lgord(cosc: f64, n: usize) -> f64 {
    let colat = unsafe { acos(cosc) };
    let mut c1 = 2.0f64.sqrt();
    for k in 1..=n {
        c1 *= (1.0f32 - 1.0f32 / (4 * k * k) as f32).sqrt() as f64;
    }
    let nf = n as f64;
    let mut ang = nf * colat;
    let mut s1 = 0.0;
    let mut c4 = 1.0;
    let mut a = -1.0;
    let mut b = 0.0;
    for k in (0..=n).step_by(2) {
        if k == n {
            c4 = 0.5 * c4;
        }
        s1 += c4 * unsafe { cos(ang) };
        a += 2.0;
        b += 1.0;
        ang = colat * (nf - k as f64 - 2.0);
        c4 = (a * (nf - b + 1.0) / (b * (nf + nf - a))) * c4;
    }
    s1 * c1
}
/// `mpas_init_atm_llxy.F:1959-2097`: binary64 Newton iteration, with
/// default-real literals rounded in binary32 before conversion to REAL64.
fn gaussian(n: usize) -> Vec<f32> {
    let pi = std::f32::consts::PI as f64;
    let fi = n as f64;
    let fi1 = fi + 1.0;
    let a = fi * fi1 / (4.0 * fi1 * fi1 - 1.0).sqrt();
    let b = fi1 * fi / (4.0 * fi * fi - 1.0).sqrt();
    let mut lat = vec![0.0; n];
    for i in 1..=n / 2 {
        let mut root = unsafe { sin((i as f32 - 0.5) as f64 * pi / fi + pi * 0.5) };
        loop {
            let g = lgord(root, n);
            let gm = lgord(root, n - 1);
            let gp = lgord(root, n + 1);
            let gt = (root * root - 1.0) / (a * gp - b * gm);
            let delta = g * gt;
            root -= delta;
            if delta.abs() <= 1e-14f32 as f64 {
                break;
            }
        }
        let colat = unsafe { acos(root) };
        let sinc = unsafe { sin(colat) };
        let degree = unsafe { acos(sinc) } * 180.0 / pi;
        lat[i - 1] = degree as f32;
        lat[n - i] = (-degree) as f32;
    }
    lat
}
