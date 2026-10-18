//! MPAS-A v8.4.2 case 8, `mpas_init_atm_surface.F:29-394`.
//! The surface stream contains only xtime, Time, sst and xice. No vertical
//! capsule, atmosphere interpolation, or case-7 sea-ice conversion applies.

use super::hinterp::{interp_sequence, InterpContext, Method, Slab, Underflow};
use super::surface_met::{read_surface_file, Map, SurfaceSlab};
use super::DEG_PER_RAD;
use crate::error::{MpasError, MpasResult};
use rw_store::netcdf_classic::{
    NcAttr, NcClassicWriter, NcData, NcDim, NcFormat, NcType, NcVarDef,
};
use serde::Serialize;
use std::path::{Path, PathBuf};

pub const SURFACE_ABI: &str = "rw_mpas_init --mode surface-update --sfc-prefix PREFIX --static STATIC.nc --out SURFACE.nc --start-time YYYY-MM-DD_HH:MM:SS --stop-time YYYY-MM-DD_HH:MM:SS --fg-interval SECONDS --surface-interval SECONDS [--receipt JSON]";
const MISSING: f32 = -1.0e30;

#[derive(Debug)]
pub struct SurfaceConfig {
    pub sfc_prefix: PathBuf,
    pub static_path: PathBuf,
    pub out_path: PathBuf,
    pub start_time: String,
    pub stop_time: String,
    pub fg_interval: i64,
    pub surface_interval: i64,
}

#[derive(Debug, Serialize)]
pub struct SurfaceReceipt {
    pub schema: &'static str,
    pub mpas_source: &'static str,
    pub n_cells: usize,
    pub records: Vec<SurfaceRecord>,
    pub output_bytes: u64,
}

#[derive(Debug, Serialize)]
pub struct SurfaceRecord {
    pub xtime: String,
    pub elapsed_seconds: f32,
    pub intermediate: PathBuf,
    pub fields_used: Vec<String>,
    pub have_landsea: bool,
}

/// `mpas_init_atm_surface.F:108-170`: read only the three mesh arrays case 8
/// needs; the native max(landmask)==0 refusal distinguishes grid from static.
fn read_mesh(path: &Path) -> MpasResult<(Vec<f32>, Vec<f32>, Vec<i32>)> {
    let file = netcrust::File::open(path)?;
    let n = file
        .dimension("nCells")
        .ok_or_else(|| MpasError::Refusal("surface input has no nCells dimension".into()))?
        .len();
    let read = |name: &str| -> MpasResult<Vec<f64>> {
        let values = file.read_array_f64_first_record_or_all(name)?.into_values();
        if values.len() != n || n == 0 {
            return Err(MpasError::Refusal(format!("surface {name} has {} values for {n} cells; a different shape samples the wrong mesh", values.len())));
        }
        Ok(values)
    };
    let lat = read("latCell")?.into_iter().map(|v| v as f32).collect();
    let lon = read("lonCell")?.into_iter().map(|v| v as f32).collect();
    let landmask: Vec<i32> = read("landmask")?.into_iter().map(|v| v as i32).collect();
    if landmask.iter().copied().max() == Some(0) {
        return Err(MpasError::Refusal("the global maximum of landmask is zero; case 8 needs a static or initial conditions file to mask surface interpolation (mpas_init_atm_surface.F:156-169)".into()));
    }
    Ok((lat, lon, landmask))
}

/// `mpas_init_atm_surface.F:289-394`. The native surface routine clamps y
/// BEFORE wrapping x. Reprojection overwrites y, even at a pole; case 7's
/// locate helper clamps AFTER wrapping and cannot be substituted here.
fn interp_to_mpas(
    source: &SurfaceSlab,
    lat: &[f32],
    lon: &[f32],
    landmask: &[i32],
    mask: Option<&Slab>,
    ice: bool,
    dest: &mut [f32],
) -> MpasResult<()> {
    let field = &source.met;
    if field.nx < 3 || field.ny == 0 || field.values.len() != field.nx * field.ny {
        return Err(MpasError::Refusal(
            "surface source needs at least three columns for the native periodic halo".into(),
        ));
    }
    if let Some(m) = mask {
        if m.end_x != field.nx as i32 + 3 || m.end_y != field.ny as i32 {
            return Err(MpasError::Refusal("LANDSEA and surface slabs differ in shape; the native shared mask would index the wrong source points".into()));
        }
    }
    let proj = Map::new(source)?;
    let array = Slab::from_met(&field.values, field.nx, field.ny);
    let ctx = InterpContext {
        array: &array,
        msgval: MISSING,
        maskval: Some(if ice { 1.0 } else { -1.0 }),
        mask,
        underflow: Underflow::Preserve,
        underflow_hits: None,
    };
    let methods: &[Method] = if ice {
        &[Method::FourPoint, Method::WtAverage4, Method::Search]
    } else {
        &[Method::FourPoint, Method::Search]
    };
    for i in 0..dest.len() {
        if landmask[i] == if ice { 1 } else { -1 } {
            dest[i] = 0.0;
            continue;
        }
        let lat = lat[i] * DEG_PER_RAD;
        let lon = lon[i] * DEG_PER_RAD;
        let (mut x, mut y) = proj.ij(lat, lon);
        if y <= 0.5 {
            y = 1.0;
        } else if y >= field.ny as f32 + 0.5 {
            y = field.ny as f32;
        }
        if x < 0.5 {
            (x, y) = proj.ij(lat, lon + 360.0);
        } else if x >= field.nx as f32 + 0.5 {
            (x, y) = proj.ij(lat, lon - 360.0);
        }
        dest[i] = interp_sequence(x, y, &ctx, methods, 0);
    }
    Ok(())
}

/// `mpas_init_atm_surface.F:104-286`. Scan the entire file for LANDSEA
/// first, then process fields in their original order. Last LANDSEA and last
/// SST/SKINTEMP win. A field absent at this time retains the prior time's data.
fn interp_sfc(
    slabs: &[SurfaceSlab],
    lat: &[f32],
    lon: &[f32],
    landmask: &[i32],
    sst: &mut [f32],
    xice: &mut [f32],
) -> MpasResult<(Vec<String>, bool)> {
    let landsea = slabs
        .iter()
        .rev()
        .map(|s| &s.met)
        .find(|s| s.field == "LANDSEA");
    let mask = match landsea {
        Some(s) if s.nx >= 3 => Some(Slab::from_met(&s.values, s.nx, s.ny)),
        Some(_) => {
            return Err(MpasError::Refusal(
                "LANDSEA needs three columns for the native periodic halo".into(),
            ))
        }
        None => None,
    };
    let mut used = Vec::new();
    for s in slabs {
        match s.met.field.as_str() {
            "SST" | "SKINTEMP" => {
                sst.fill(0.0);
                interp_to_mpas(s, lat, lon, landmask, mask.as_ref(), false, sst)?;
            }
            "SEAICE" => {
                xice.fill(0.0);
                interp_to_mpas(s, lat, lon, landmask, mask.as_ref(), true, xice)?;
            }
            _ => continue,
        }
        used.push(s.met.field.clone());
    }
    for i in 0..sst.len() {
        if sst[i] < 271.0 && landmask[i] == 0 {
            xice[i] = 1.0;
        }
        // Two comparisons, preserving native NaN behaviour.
        if xice[i] < 0.0 {
            xice[i] = 0.0;
        }
        if xice[i] > 1.0 {
            xice[i] = 1.0;
        }
    }
    Ok((used, mask.is_some()))
}

fn leap(y: i64) -> bool {
    y % 4 == 0 && (y % 100 != 0 || y % 400 == 0)
}
fn month_days(y: i64, m: usize) -> i64 {
    [
        31,
        if leap(y) { 29 } else { 28 },
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
    ][m - 1]
}

/// MPAS Gregorian clock dates (`mpas_init_atm_surface.F:69-99`). Integer
/// seconds keep the calendar independent of floating Time's RKIND rounding.
fn seconds(stamp: &str) -> MpasResult<i64> {
    let bad = || {
        MpasError::Refusal(format!(
            "invalid Gregorian time {stamp:?}; expected YYYY-MM-DD_HH:MM:SS"
        ))
    };
    let b = stamp.as_bytes();
    if b.len() != 19
        || b[4] != b'-'
        || b[7] != b'-'
        || b[10] != b'_'
        || b[13] != b':'
        || b[16] != b':'
    {
        return Err(bad());
    }
    let number = |a: usize, z: usize| -> MpasResult<i64> {
        if !b[a..z].iter().all(u8::is_ascii_digit) {
            return Err(bad());
        }
        std::str::from_utf8(&b[a..z])
            .map_err(|_| bad())?
            .parse()
            .map_err(|_| bad())
    };
    let (y, m, d, h, mi, s) = (
        number(0, 4)?,
        number(5, 7)?,
        number(8, 10)?,
        number(11, 13)?,
        number(14, 16)?,
        number(17, 19)?,
    );
    if y < 1
        || !(1..=12).contains(&m)
        || d < 1
        || d > month_days(y, m as usize)
        || h > 23
        || mi > 59
        || s > 59
    {
        return Err(bad());
    }
    let days: i64 = (1..y).map(|v| if leap(v) { 366 } else { 365 }).sum::<i64>()
        + (1..m as usize).map(|v| month_days(y, v)).sum::<i64>()
        + d
        - 1;
    Ok(days * 86400 + h * 3600 + mi * 60 + s)
}

fn stamp(t: i64) -> String {
    let mut days = t / 86400;
    let daysec = t % 86400;
    let mut y = 1;
    while days >= if leap(y) { 366 } else { 365 } {
        days -= if leap(y) { 366 } else { 365 };
        y += 1;
    }
    let mut m = 1;
    while days >= month_days(y, m) {
        days -= month_days(y, m);
        m += 1;
    }
    format!(
        "{y:04}-{m:02}-{:02}_{:02}:{:02}:{:02}",
        days + 1,
        daysec / 3600,
        daysec / 60 % 60,
        daysec % 60
    )
}

/// The complete case-8 time loop (`mpas_init_atm_surface.F:29-101`) and
/// interval check (`mpas_init_atm_cases.F:272-299`). Schema is the immutable
/// surface stream in `core_init_atmosphere/Registry.xml:666-678`.
pub fn build_surface(cfg: &SurfaceConfig) -> MpasResult<SurfaceReceipt> {
    if cfg.fg_interval <= 0 || cfg.fg_interval != cfg.surface_interval {
        return Err(MpasError::Refusal("config_fg_interval must be positive and match the surface stream output_interval (mpas_init_atm_cases.F:277-289); otherwise surface time records are skipped or the clock cannot advance".into()));
    }
    let start = seconds(&cfg.start_time)?;
    let stop = seconds(&cfg.stop_time)?;
    if stop < start {
        return Err(MpasError::Refusal(
            "surface stop time precedes start time; no surface records would be written".into(),
        ));
    }
    let (lat, lon, landmask) = read_mesh(&cfg.static_path)?;
    let n = lat.len();
    let count = ((stop - start) / cfg.fg_interval + 1) as usize;
    let vars = vec![
        NcVarDef::new("xtime", NcType::Char, vec![0, 2]).with_attrs(vec![
            NcAttr::text("units", "YYYY-MM-DD_hh:mm:ss"),
            NcAttr::text("long_name", "Model valid time"),
        ]),
        NcVarDef::new("Time", NcType::Float, vec![0]).with_attrs(vec![
            NcAttr::text(
                "units",
                format!("seconds since {}", cfg.start_time.replace('_', " ")),
            ),
            NcAttr::text("long_name", "CF-compliant valid time"),
            NcAttr::text("standard_name", "time"),
        ]),
        NcVarDef::new("sst", NcType::Float, vec![0, 1]).with_attrs(vec![
            NcAttr::text("units", "K"),
            NcAttr::text("long_name", "sea-surface temperature"),
        ]),
        NcVarDef::new("xice", NcType::Float, vec![0, 1]).with_attrs(vec![
            NcAttr::text("units", "unitless"),
            NcAttr::text("long_name", "fractional area coverage of sea-ice"),
        ]),
    ];
    // Stage beside the output, so a missing later intermediate cannot replace
    // an existing usable file with a truncated sequence. Never stage on input.
    if cfg.out_path == cfg.static_path
        || (cfg.out_path.exists()
            && cfg.out_path.canonicalize()? == cfg.static_path.canonicalize()?)
    {
        return Err(MpasError::Refusal(
            "surface output aliases its static input; replacing it would destroy the mesh".into(),
        ));
    }
    let pending = cfg
        .out_path
        .with_extension(format!("surface-{}.partial", std::process::id()));
    if pending.exists() {
        return Err(MpasError::Refusal(format!(
            "staging path {} already exists; truncating it could destroy another surface output",
            pending.display()
        )));
    }
    let result = (|| {
        let mut writer = NcClassicWriter::create(
            &pending,
            NcFormat::Data64,
            vec![
                NcDim::record("Time"),
                NcDim::fixed("nCells", n),
                NcDim::fixed("StrLen", 64),
            ],
            vec![NcAttr::text("source", "MPAS-A v8.4.2 case 8 Rust port")],
            vars,
            count as u64,
        )?;
        let mut receipt = SurfaceReceipt {
            schema: "rw-mpas.surface-update/v1",
            mpas_source:
                "MPAS-A v8.4.2 mpas_init_atm_surface.F:29-394; mpas_init_atm_cases.F:272-299",
            n_cells: n,
            records: Vec::new(),
            output_bytes: 0,
        };
        let mut sst = vec![0.0; n];
        let mut xice = vec![0.0; n];
        for r in 0..count {
            let elapsed = r as i64 * cfg.fg_interval;
            let xtime = stamp(start + elapsed);
            let mut filename = cfg.sfc_prefix.as_os_str().to_os_string();
            filename.push(":");
            filename.push(&xtime[..13]);
            let intermediate = PathBuf::from(filename);
            if cfg.out_path == intermediate
                || (cfg.out_path.exists()
                    && cfg.out_path.canonicalize()? == intermediate.canonicalize()?)
            {
                return Err(MpasError::Refusal("surface output aliases an intermediate input; replacing it would destroy the source fields".into()));
            }
            let slabs = read_surface_file(&intermediate)?;
            let (fields_used, have_landsea) =
                interp_sfc(&slabs, &lat, &lon, &landmask, &mut sst, &mut xice)?;
            let mut text = xtime.as_bytes().to_vec();
            text.resize(64, b' ');
            writer.put_record("xtime", r as u64, NcData::Chars(&text))?;
            writer.put_record("Time", r as u64, NcData::Floats(&[elapsed as f32]))?;
            writer.put_record("sst", r as u64, NcData::Floats(&sst))?;
            writer.put_record("xice", r as u64, NcData::Floats(&xice))?;
            receipt.records.push(SurfaceRecord {
                xtime,
                elapsed_seconds: elapsed as f32,
                intermediate,
                fields_used,
                have_landsea,
            });
        }
        receipt.output_bytes = writer.file_len();
        writer.finish()?;
        std::fs::rename(&pending, &cfg.out_path)?;
        Ok(receipt)
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&pending);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn calendar_roundtrip_and_rollovers() {
        for date in [
            "2000-02-29_23:59:59",
            "1900-03-01_00:00:00",
            "1999-12-31_23:59:59",
        ] {
            assert_eq!(stamp(seconds(date).unwrap()), date);
        }
        assert_eq!(
            stamp(seconds("2000-02-29_23:59:59").unwrap() + 1),
            "2000-03-01_00:00:00"
        );
        for date in [
            "1900-02-29_00:00:00",
            "2000-00-01_00:00:00",
            "2000-01-01_24:00:00",
            "0000-01-01_00:00:00",
        ] {
            assert!(seconds(date).is_err());
        }
    }
}

// Retain i1's public helper API as well as the complete case-8 driver.
pub use initial::{initialize_sst, interpolate};

/// Case-7 optional surface initialization, retained from the landed i1 port.
pub mod initial {
// Complete optional surface-input driver, MPAS-A v8.4.2
// core_init_atmosphere/mpas_init_atm_surface.F:104-394.
use super::super::{metfile::{MetSlab,IPROJ_POLAR},projection::Projection,hinterp::{self,Slab,InterpContext,Method,Underflow},DEG_PER_RAD};
use crate::error::{MpasError,MpasResult};

/// MPAS v8.4.2 mpas_init_atm_surface.F:104-394. Ordered matching
/// records replace the destination; the last LANDSEA record supplies masks.
pub fn interpolate(slabs:&[MetSlab],lat:&[f32],lon:&[f32],land:&[i32],
    sst:&mut [f32],ice:&mut [f32],underflow:Underflow)->MpasResult<usize> {
    let n=land.len();
    if lat.len()!=n || lon.len()!=n || sst.len()!=n || ice.len()!=n {return Err(MpasError::Refusal("surface-input arrays have different cell counts".into()));}
    if land.iter().copied().max()==Some(0) {return Err(MpasError::Refusal("native interp_sfc_to_MPAS rejects a landmask whose global maximum is zero because surface interpolation would have no initialized land mask".into()));}
    let mask=slabs.iter().rev().find(|s|s.field=="LANDSEA").map(|s|Slab::from_met(&s.values,s.nx,s.ny));
    let mut consumed=0;
    for slab in slabs {
        let is_sst=slab.field=="SKINTEMP" || slab.field=="SST";
        if !is_sst && slab.field!="SEAICE" {continue;}
        let projection=if slab.iproj==IPROJ_POLAR {Projection::auxiliary_polar(slab)?} else {Projection::from_met(slab)?};
        let array=Slab::from_met(&slab.values,slab.nx,slab.ny);
        let methods=if is_sst {vec![Method::FourPoint,Method::Search]} else {vec![Method::FourPoint,Method::WtAverage4,Method::Search]};
        let masked=if is_sst {-1} else {1};let maskval=masked as f32;
        let context=InterpContext {array:&array,msgval:-1e30f32,maskval:Some(maskval),mask:mask.as_ref(),underflow,underflow_hits:None};
        let destination=if is_sst {&mut *sst} else {&mut *ice};
        destination.fill(0.);
        for c in 0..n {
            if land[c]==masked {continue;}
            let (x,y)=projection.locate_surface(lat[c]*DEG_PER_RAD,lon[c]*DEG_PER_RAD);
            destination[c]=hinterp::interp_sequence(x,y,&context,&methods,0);
        }
        consumed+=1;
    }
    for c in 0..n {if sst[c]<271. && land[c]==0 {ice[c]=1.;}}
    for value in ice {*value=value.clamp(0.,1.);}
    Ok(consumed)
}

/// MPAS v8.4.2 mpas_atmphys_initialize_real.F:516-583.
/// Native seaice is scratch here and is rebuilt by physics_init_seaice.
pub fn initialize_sst(frac:bool,land:&[i32],sst:&[f32],skin:&mut [f32],ice:&mut [f32]) {
    let threshold=if frac {0.02f32} else {0.5f32};
    for c in 0..land.len() {
        if land[c]==0 && ice[c]<threshold {skin[c]=sst[c];}
        if (land[c]==1 && ice[c]>0.) || ice[c]>200. {ice[c]=0.;}
    }
}

}
