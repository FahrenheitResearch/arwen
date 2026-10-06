//! `isobaric_height_check` -- a run's stored isobaric height planes against
//! the same surfaces read between the frame's layer interfaces.
//!
//! The store is what every chart and sounding of the run is drawn from; the
//! interface read is `rw_isobaric::isobaric_heights_from_interfaces` on the
//! frame itself: PH + PHB over standard gravity, interpolated in ln p
//! between interfaces whose pressures are linear in eta between the
//! mass-level pressures (P + PB).  A store written by a build that paired
//! layer-mean heights with mass-level pressures shows that pairing here as a
//! mean offset of several metres at 500 hPa; a store that reads between
//! interfaces shows round-off.
//!
//! ```text
//! cargo run --release --example isobaric_height_check -- \
//!     --store-root DIR --frame WRFOUT [--levels 850,700,500,300,250]
//! ```
//!
//! One line per level: `LEVEL hpa points mean_m rms_m max_abs_m`, store
//! minus interface read, over the points where both are finite.
//! `--planes-out DIR` also writes each interface-read plane as
//! `DIR/height_<hpa>hpa.f32` (row-major little-endian f32, NaN where the
//! surface does not exist): the reference the Python copy of the same
//! arithmetic is measured against.  `--store-root` may then be omitted.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use rustwx_core::{CanonicalField, FieldSelector};
use rw_isobaric::{InterfaceStencil, STANDARD_GRAVITY, isobaric_heights_from_interfaces};

fn hour_files(dir: &Path, found: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            hour_files(&path, found);
        } else if path.extension().is_some_and(|ext| ext == "rws") {
            found.push(path);
        }
    }
}

fn run() -> Result<(), String> {
    let mut store_root: Option<PathBuf> = None;
    let mut frame = None;
    let mut planes_out: Option<PathBuf> = None;
    let mut levels: Vec<u16> = vec![850, 700, 500, 300, 250];
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let mut value = || args.next().ok_or_else(|| format!("{arg} needs a value"));
        match arg.as_str() {
            "--store-root" => store_root = Some(PathBuf::from(value()?)),
            "--frame" => frame = Some(PathBuf::from(value()?)),
            "--planes-out" => planes_out = Some(PathBuf::from(value()?)),
            "--levels" => {
                levels = value()?
                    .split(',')
                    .map(|text| text.trim().parse::<u16>().map_err(|e| format!("{text}: {e}")))
                    .collect::<Result<_, _>>()?
            }
            other => return Err(format!("unknown argument {other}")),
        }
    }
    let frame = frame.ok_or("--frame is required")?;
    if store_root.is_none() && planes_out.is_none() {
        return Err("--store-root or --planes-out is required".into());
    }

    let file = wrf_core::WrfFile::open(&frame).map_err(|e| format!("open {}: {e}", frame.display()))?;
    let cells = file.nxy();
    let eta_mass = file.read_var("ZNU", 0).map_err(|e| format!("ZNU: {e}"))?;
    let eta_interface = file.read_var("ZNW", 0).map_err(|e| format!("ZNW: {e}"))?;
    let stencil = InterfaceStencil::new(&eta_mass, &eta_interface)?;
    let pressure = file.full_pressure(0).map_err(|e| format!("P + PB: {e}"))?;
    let geopotential = file.geopotential_stag(0).map_err(|e| format!("PH + PHB: {e}"))?;
    let targets: Vec<f64> = levels.iter().map(|hpa| f64::from(*hpa) * 100.0).collect();
    let interface = isobaric_heights_from_interfaces(
        &geopotential,
        STANDARD_GRAVITY,
        &pressure,
        &stencil,
        cells,
        &targets,
    )?;

    if let Some(dir) = &planes_out {
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        for (level, plane) in levels.iter().zip(&interface) {
            let path = dir.join(format!("height_{level}hpa.f32"));
            let bytes: Vec<u8> = plane.iter().flat_map(|v| v.to_le_bytes()).collect();
            std::fs::write(&path, bytes).map_err(|e| format!("{}: {e}", path.display()))?;
            println!("PLANE\t{level}\t{}\tny={}\tnx={}", path.display(), file.ny, file.nx);
        }
    }
    let Some(store_root) = store_root else {
        return Ok(());
    };
    let mut hours = Vec::new();
    hour_files(&store_root, &mut hours);
    let [hour] = hours.as_slice() else {
        return Err(format!(
            "{} holds {} stored hours; this check reads a store of exactly one frame",
            store_root.display(),
            hours.len()
        ));
    };
    let reader = rw_store::reader::HourReader::open(hour).map_err(|e| e.to_string())?;
    println!("STORE\t{}", hour.display());

    for (level, from_interfaces) in levels.iter().zip(&interface) {
        let name = FieldSelector::isobaric(CanonicalField::GeopotentialHeight, *level).key();
        let stored = reader.read_full_2d(&name).map_err(|e| format!("{name}: {e}"))?;
        if stored.len() != cells {
            return Err(format!("{name} has {} points, the frame {cells}", stored.len()));
        }
        let (mut points, mut sum, mut sum_sq, mut max_abs) = (0usize, 0.0f64, 0.0f64, 0.0f64);
        let (mut stored_sum, mut interface_sum) = (0.0f64, 0.0f64);
        for (a, b) in stored.iter().zip(from_interfaces) {
            if !(a.is_finite() && b.is_finite()) {
                continue;
            }
            let difference = f64::from(*a) - f64::from(*b);
            points += 1;
            sum += difference;
            sum_sq += difference * difference;
            max_abs = max_abs.max(difference.abs());
            stored_sum += f64::from(*a);
            interface_sum += f64::from(*b);
        }
        if points == 0 {
            return Err(format!("{name}: no point where both planes are finite"));
        }
        let n = points as f64;
        println!(
            "LEVEL\t{level}\tpoints={points}\tmean_m={:.4}\trms_m={:.4}\tmax_abs_m={:.4}\tstore_mean_m={:.3}\tinterface_mean_m={:.3}",
            sum / n,
            (sum_sq / n).sqrt(),
            max_abs,
            stored_sum / n,
            interface_sum / n
        );
    }
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("isobaric_height_check: {message}");
            ExitCode::FAILURE
        }
    }
}
