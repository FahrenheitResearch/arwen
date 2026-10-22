//! `rw_native_wrfout` -- a reference model's own native-level state written
//! into a run's wrfout shape, so the SAME wrf-rust diagnostics run on both.
//!
//! The question it answers: when a run's derived product (sea-level
//! pressure, CAPE, shear, helicity, gust) differs from the reference
//! model's published message, how much of the difference is the model
//! and how much is the post-processor?  The reference's published
//! messages come from its own post-processor (NCEP UPP for HRRR), whose
//! definitions are not wrf-python's.  Feeding the reference's native
//! model-level fields through the run's own diagnostic lane removes that
//! half: what is left is the model.
//!
//! ```text
//! rw_native_wrfout --template WOOF_WRFOUT --native WRFNAT.grib2
//!                  [--surface WRFSFC.grib2] --out OUT_WRFOUT
//!                  [--i0 N --j0 N]
//! ```
//!
//! The template is a run's wrfout AT THE SAME VALID TIME on a grid that is
//! a window of the reference grid starting at reference point (i0, j0)
//! (HRRR's 1799 x 1059 grid less one outer row is i0 = j0 = 1).  The output
//! is a byte copy of the template with every time-varying field replaced:
//!
//! | wrfout | from the reference (hybrid level k = 1..nz, bottom up) |
//! | --- | --- |
//! | PB, P_HYD | PRES; P = 0 |
//! | T | TMP (p0/PRES)^kappa - 300, wrf-core's own p0 and kappa, so its temperature returns TMP |
//! | QVAPOR | SPFH / (1 - SPFH) |
//! | QCLOUD QRAIN QICE QSNOW QGRAUP | CLMR RWMR CIMIXR SNMR GRLE |
//! | PHB | g z on interfaces: z(0) = HGT surface, z(k+1) = 2 HGT(k) - z(k), so the interface mean wrf-core takes is HGT(k); PH = 0 |
//! | U, V | faces whose two-face mean is UGRD/VGRD exactly (grid-relative, as published), the free constant chosen to minimise the 2-grid-length mode |
//! | PSFC HGT TSK | PRES, HGT, TMP surface |
//! | T2 Q2 TH2 | TMP 2 m, SPFH 2 m as mixing ratio, TMP 2 m (p0/PSFC)^kappa |
//! | U10 V10 | UGRD/VGRD 10 m |
//! | PBLH | HPBL surface |
//!
//! Every other time-varying floating field of two or more dimensions is set
//! to NaN, so a diagnostic that reads a field the reference does not
//! publish comes out missing instead of silently reading the run's value.
//! Geography (XLAT, XLONG, map factors, rotation, land masks), the vertical
//! coordinate constants and Times stay the template's.
//!
//! Breakage it prevents: the WOOF-vs-HRRR sheets paired wrf-rust's
//! definitions with UPP's (MAPS sea-level pressure, UPP's parcel CAPE,
//! layer-mean shear, the GSD gust), so calculation differences showed up as
//! forecast differences.

use std::collections::BTreeMap;
use std::error::Error;
use std::path::PathBuf;
use std::process::ExitCode;

use grib_core::grib2::{unpack_message, Grib2File, Grib2Message};
use netcdf_writer::NcPatcher;
use netcdf_writer::{NcType, VarData};

/// Embed the build revision for release artifact verification.
pub static GPUWM_BRIDGE_SOURCE_REV_STAMP: &str =
    concat!("GPUWM_BRIDGE_SOURCE_REV=", env!("GPUWM_BRIDGE_SOURCE_REV"));

const G: f64 = 9.80665; // wrf-core file.rs
const P0: f64 = 100_000.0; // wrf-core file.rs
const KAPPA: f64 = 0.2857142857; // wrf-core file.rs

const HYBRID: u8 = 105;
const SURFACE: u8 = 1;
const HEIGHT_AGL: u8 = 103;

/// Fields that stay the template's: geography and the vertical coordinate.
const KEEP: &[&str] = &[
    "Times", "XLAT", "XLONG", "XLAT_U", "XLONG_U", "XLAT_V", "XLONG_V", "MAPFAC_M", "MAPFAC_U",
    "MAPFAC_V", "MAPFAC_MX", "MAPFAC_MY", "MAPFAC_UX", "MAPFAC_UY", "MAPFAC_VX", "MAPFAC_VY",
    "MF_VX_INV", "SINALPHA", "COSALPHA", "F", "E", "LANDMASK", "XLAND", "LU_INDEX", "LAKEMASK",
    "ZNU", "ZNW", "ZS", "DZS", "FNM", "FNP", "RDNW", "RDN", "DNW", "DN", "CF1", "CF2", "CF3",
    "CFN", "CFN1", "C1H", "C2H", "C3H", "C4H", "C1F", "C2F", "C3F", "C4F", "P_TOP", "T00", "P00",
    "TLP", "TISO", "TLP_STRAT", "P_STRAT", "MAX_MSTFX", "MAX_MSTFY", "XTIME", "ITIMESTEP",
    "IVGTYP", "ISLTYP", "VEGFRA", "LANDUSEF", "SOILCTOP", "SOILCBOT", "SHDMAX", "SHDMIN",
    "ALBBCK", "SNOALB", "MAXSNOWALB", "VAR", "VAR_SSO", "CON", "OA1", "OA2", "OA3", "OA4", "OL1",
    "OL2", "OL3", "OL4",
];

struct Args {
    template: PathBuf,
    native: PathBuf,
    surface: Option<PathBuf>,
    out: PathBuf,
    i0: usize,
    j0: usize,
}

fn parse_args() -> Result<Args, String> {
    let mut map = BTreeMap::new();
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let value = it.next().ok_or_else(|| format!("{flag} needs a value"))?;
        map.insert(flag, value);
    }
    let take = |map: &mut BTreeMap<String, String>, k: &str| map.remove(k);
    let template = take(&mut map, "--template").ok_or("--template WRFOUT is required")?;
    let native = take(&mut map, "--native").ok_or("--native WRFNAT.grib2 is required")?;
    let out = take(&mut map, "--out").ok_or("--out FILE is required")?;
    let surface = take(&mut map, "--surface").map(PathBuf::from);
    let i0 = take(&mut map, "--i0").map(|v| v.parse::<usize>()).transpose().map_err(|e| e.to_string())?.unwrap_or(1);
    let j0 = take(&mut map, "--j0").map(|v| v.parse::<usize>()).transpose().map_err(|e| e.to_string())?.unwrap_or(1);
    if let Some(k) = map.keys().next() {
        return Err(format!("unknown argument {k}"));
    }
    Ok(Args { template: template.into(), native: native.into(), surface, out: out.into(), i0, j0 })
}

/// Messages of one or two files, searched by code and level.
struct Source {
    messages: Vec<Grib2Message>,
}

impl Source {
    fn find(&self, d: u8, c: u8, n: u8, level_type: u8, level: f64) -> Option<&Grib2Message> {
        self.messages.iter().find(|m| {
            m.discipline == d
                && m.product.parameter_category == c
                && m.product.parameter_number == n
                && m.product.level_type == level_type
                && (m.product.level_value - level).abs() < 1e-6
                // instantaneous fields only (template 0); HPBL etc. are template 0 too
                && m.product.template == 0
        })
    }
}

/// One plane cropped to the template window, row-major south to north.
fn crop(m: &Grib2Message, i0: usize, j0: usize, nx: usize, ny: usize) -> Result<Vec<f64>, Box<dyn Error>> {
    let gx = m.grid.nx as usize;
    let gy = m.grid.ny as usize;
    if m.grid.scan_mode != 0x40 {
        return Err(format!("scan mode {:#x}; this tool reads +i +j row-major (0x40) grids", m.grid.scan_mode).into());
    }
    if i0 + nx > gx || j0 + ny > gy {
        return Err(format!("window {nx}x{ny} at ({i0},{j0}) leaves the {gx}x{gy} reference grid").into());
    }
    let values = unpack_message(m)?;
    let mut out = Vec::with_capacity(nx * ny);
    for j in 0..ny {
        let row = (j + j0) * gx + i0;
        out.extend_from_slice(&values[row..row + nx]);
    }
    Ok(out)
}

/// Faces 0..=n along a line whose two-face means are `mass` exactly.  The
/// one free constant (the 2-grid-length mode the means cannot see) is set
/// to minimise that mode's energy.
fn faces_from_means(mass: &[f64]) -> Vec<f64> {
    let n = mass.len();
    let mut face = vec![0.0; n + 1];
    face[0] = mass[0];
    for k in 0..n {
        face[k + 1] = 2.0 * mass[k] - face[k];
    }
    // face[k] = particular[k] + (-1)^k c; the mode amplitude of the
    // particular solution relative to the mass mean is removed.
    let mut mode = 0.0;
    for k in 0..=n {
        let reference = if k == 0 { mass[0] } else if k == n { mass[n - 1] } else { 0.5 * (mass[k - 1] + mass[k]) };
        let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
        mode += sign * (face[k] - reference);
    }
    let c = mode / (n + 1) as f64;
    for (k, f) in face.iter_mut().enumerate() {
        let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
        *f -= sign * c;
    }
    face
}

fn to_f32(v: &[f64]) -> Vec<f32> {
    v.iter().map(|x| *x as f32).collect()
}

fn main() -> ExitCode {
    let _ = std::hint::black_box(GPUWM_BRIDGE_SOURCE_REV_STAMP);
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("rw_native_wrfout: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let args = parse_args().map_err(|e| {
        format!("{e}\nusage: rw_native_wrfout --template WRFOUT --native WRFNAT.grib2 [--surface WRFSFC.grib2] --out FILE [--i0 1 --j0 1]")
    })?;
    let wrf = wrf_core::WrfFile::open(&args.template)?;
    let (nx, ny, nz) = (wrf.nx, wrf.ny, wrf.nz);
    let nxy = nx * ny;
    let terrain_template = wrf.read_var("HGT", 0)?;

    let mut messages = Grib2File::open(args.native.to_str().ok_or("path")?)?.messages;
    if let Some(s) = &args.surface {
        messages.extend(Grib2File::open(s.to_str().ok_or("path")?)?.messages);
    }
    let src = Source { messages };
    let (i0, j0) = (args.i0, args.j0);
    let plane = |name: &str, d: u8, c: u8, n: u8, lt: u8, lv: f64| -> Result<Vec<f64>, Box<dyn Error>> {
        let m = src.find(d, c, n, lt, lv).ok_or_else(|| format!("{name}: no message {d}/{c}/{n} on level type {lt} value {lv}"))?;
        crop(m, i0, j0, nx, ny)
    };
    let column = |name: &str, d: u8, c: u8, n: u8| -> Result<Vec<f64>, Box<dyn Error>> {
        let mut out = Vec::with_capacity(nz * nxy);
        for k in 1..=nz {
            out.extend(plane(name, d, c, n, HYBRID, k as f64)?);
        }
        Ok(out)
    };
    let optional_column = |name: &str, d: u8, c: u8, n: u8| -> Option<Vec<f64>> {
        match column(name, d, c, n) {
            Ok(v) => Some(v),
            Err(e) => {
                eprintln!("rw_native_wrfout: {name} not published ({e}); left NaN");
                None
            }
        }
    };

    // Geography check: the reference's terrain must be the template's.
    let hgt_sfc = plane("HGT surface", 0, 3, 5, SURFACE, 0.0)?;
    let terr_diff = hgt_sfc.iter().zip(&terrain_template).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
    let terr_rms = (hgt_sfc.iter().zip(&terrain_template).map(|(a, b)| (a - b).powi(2)).sum::<f64>() / nxy as f64).sqrt();
    let mut abs: Vec<f64> = hgt_sfc.iter().zip(&terrain_template).map(|(a, b)| (a - b).abs()).collect();
    abs.sort_by(|a, b| a.total_cmp(b));
    let median = abs[abs.len() / 2];
    let same = abs.iter().filter(|d| **d < 0.5).count() as f64 / abs.len() as f64;
    println!("terrain reference - template: max {terr_diff:.3} m, rms {terr_rms:.4} m, median {median:.4} m, {:.2} % within 0.5 m (window i0={i0} j0={j0})", 100.0 * same);
    // A window off by one point puts every slope out by tens of metres; a
    // run whose static terrain differs from the reference at a few points
    // (coastlines, lakes) keeps the median at zero.
    if median > 0.5 {
        return Err(format!("terrain median difference {median:.1} m: the window does not line up with the template grid").into());
    }

    let pres = column("PRES", 0, 3, 0)?;
    let tmp = column("TMP", 0, 0, 0)?;
    let spfh = column("SPFH", 0, 1, 0)?;
    let hgt = column("HGT", 0, 3, 5)?;
    let ugrd = column("UGRD", 0, 2, 2)?;
    let vgrd = column("VGRD", 0, 2, 3)?;

    let theta_pert: Vec<f64> = tmp.iter().zip(&pres).map(|(t, p)| t * (P0 / p).powf(KAPPA) - 300.0).collect();
    let qv: Vec<f64> = spfh.iter().map(|q| q / (1.0 - q)).collect();

    // Interfaces whose means are the published level heights.
    let mut phb = vec![0.0f64; (nz + 1) * nxy];
    for cell in 0..nxy {
        let mut z = hgt_sfc[cell];
        phb[cell] = G * z;
        for k in 0..nz {
            z = 2.0 * hgt[k * nxy + cell] - z;
            phb[(k + 1) * nxy + cell] = G * z;
        }
    }
    // Faces: U on (nz, ny, nx+1), V on (nz, ny+1, nx).
    let mut u_st = vec![0.0f64; nz * ny * (nx + 1)];
    let mut v_st = vec![0.0f64; nz * (ny + 1) * nx];
    for k in 0..nz {
        for j in 0..ny {
            let row: Vec<f64> = (0..nx).map(|i| ugrd[k * nxy + j * nx + i]).collect();
            let faces = faces_from_means(&row);
            let base = k * ny * (nx + 1) + j * (nx + 1);
            u_st[base..base + nx + 1].copy_from_slice(&faces);
        }
        for i in 0..nx {
            let col: Vec<f64> = (0..ny).map(|j| vgrd[k * nxy + j * nx + i]).collect();
            let faces = faces_from_means(&col);
            for (j, f) in faces.iter().enumerate() {
                v_st[k * (ny + 1) * nx + j * nx + i] = *f;
            }
        }
    }

    let psfc = plane("PRES surface", 0, 3, 0, SURFACE, 0.0)?;
    let tsk = plane("TMP surface", 0, 0, 0, SURFACE, 0.0)?;
    let t2 = plane("TMP 2 m", 0, 0, 0, HEIGHT_AGL, 2.0)?;
    let q2s = plane("SPFH 2 m", 0, 1, 0, HEIGHT_AGL, 2.0)?;
    let u10 = plane("UGRD 10 m", 0, 2, 2, HEIGHT_AGL, 10.0)?;
    let v10 = plane("VGRD 10 m", 0, 2, 3, HEIGHT_AGL, 10.0)?;
    // HPBL: the WMO code (0/3/18) or NCEP's local one (0/3/196).
    let pblh = plane("HPBL surface", 0, 3, 18, SURFACE, 0.0)
        .or_else(|_| plane("HPBL surface", 0, 3, 196, SURFACE, 0.0))?;
    let q2: Vec<f64> = q2s.iter().map(|q| q / (1.0 - q)).collect();
    let th2: Vec<f64> = t2.iter().zip(&psfc).map(|(t, p)| t * (P0 / p).powf(KAPPA)).collect();

    let mut fields: BTreeMap<&str, Vec<f64>> = BTreeMap::new();
    fields.insert("PB", pres.clone());
    fields.insert("P_HYD", pres);
    fields.insert("P", vec![0.0; nz * nxy]);
    fields.insert("T", theta_pert);
    fields.insert("QVAPOR", qv);
    fields.insert("PHB", phb);
    fields.insert("PH", vec![0.0; (nz + 1) * nxy]);
    fields.insert("U", u_st);
    fields.insert("V", v_st);
    fields.insert("PSFC", psfc);
    fields.insert("HGT", hgt_sfc);
    fields.insert("TSK", tsk);
    fields.insert("T2", t2);
    fields.insert("Q2", q2);
    fields.insert("TH2", th2);
    fields.insert("U10", u10);
    fields.insert("V10", v10);
    fields.insert("PBLH", pblh);
    for (name, (d, c, n)) in [
        ("QCLOUD", (0u8, 1u8, 22u8)),
        ("QRAIN", (0, 1, 24)),
        ("QICE", (0, 1, 82)),
        ("QSNOW", (0, 1, 25)),
        ("QGRAUP", (0, 1, 32)),
    ] {
        if let Some(v) = optional_column(name, d, c, n) {
            fields.insert(name, v);
        }
    }

    let mut patch = NcPatcher::open(&args.template, &args.out)?;
    let records = patch.num_records();
    if records != 1 {
        return Err(format!("the template holds {records} records; one valid time per file").into());
    }
    let mut written = Vec::new();
    let mut nan = Vec::new();
    for var in patch.variables() {
        let name = var.name.as_str();
        let record = var.is_record.then_some(0);
        if let Some(values) = fields.get(name) {
            if values.len() as u64 != var.elems {
                return Err(format!("{name}: {} values for a {}-element slab", values.len(), var.elems).into());
            }
            match var.ty {
                NcType::Float => patch.put(name, record, VarData::F32(&to_f32(values)))?,
                NcType::Double => patch.put(name, record, VarData::F64(values))?,
                other => return Err(format!("{name} is stored as {other:?}").into()),
            }
            written.push(name.to_string());
            continue;
        }
        if KEEP.contains(&name) || !var.is_record || (var.elems as usize) < nxy {
            continue;
        }
        match var.ty {
            NcType::Float => patch.put(name, record, VarData::F32(&vec![f32::NAN; var.elems as usize]))?,
            NcType::Double => patch.put(name, record, VarData::F64(&vec![f64::NAN; var.elems as usize]))?,
            _ => continue,
        }
        nan.push(name.to_string());
    }
    let missing: Vec<&&str> = fields.keys().filter(|k| !written.iter().any(|w| w == **k)).collect();
    let out = patch.finish()?;
    println!("wrote {} from the reference: {}", written.len(), written.join(" "));
    println!("set {} run-only fields to NaN: {}", nan.len(), nan.join(" "));
    if !missing.is_empty() {
        println!("not in the template (skipped): {missing:?}");
    }
    println!("{}", out.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::faces_from_means;

    #[test]
    fn face_means_reproduce_the_mass_values() {
        let mass: Vec<f64> = (0..1797).map(|i| (i as f64 * 0.37).sin() * 12.0 + (i % 7) as f64 * 0.3).collect();
        let faces = faces_from_means(&mass);
        for (k, m) in mass.iter().enumerate() {
            assert!((0.5 * (faces[k] + faces[k + 1]) - m).abs() < 1e-9, "level {k}");
        }
    }
}
