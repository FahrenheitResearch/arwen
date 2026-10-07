//! Black-box comparison of Group A planes with another exporter's GRIB2
//! output for the same history frame: values only, matched by GRIB2
//! identity (discipline, category, number, first-surface type and value).
//!
//! ```text
//! compare_grib FILE.grib2 OUR_DIR [--json OUT.json] [--dump-ref DIR]
//! ```
//!
//! `OUR_DIR` is a `post_frame` output directory.  Prints a Markdown table
//! of per-field maximum and RMS differences over cells where both are
//! finite.

use std::path::PathBuf;

use grib_core::grib2::{Grib2File, unpack_message};
use serde_json::{Value, json};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 2 {
        return Err("usage: compare_grib FILE.grib2 OUR_DIR [--json OUT.json]".into());
    }
    let grib = Grib2File::open(&args[0])?;
    let ours = PathBuf::from(&args[1]);
    let json_out = args.iter().position(|a| a == "--json").and_then(|i| args.get(i + 1)).map(PathBuf::from);
    let dump = args.iter().position(|a| a == "--dump-ref").and_then(|i| args.get(i + 1)).map(PathBuf::from);
    if let Some(d) = &dump {
        std::fs::create_dir_all(d)?;
    }
    let manifest: Value = serde_json::from_slice(&std::fs::read(ours.join("manifest.json"))?)?;
    let nx = manifest["shape"]["nx"].as_u64().ok_or("manifest shape")? as usize;
    let ny = manifest["shape"]["ny"].as_u64().ok_or("manifest shape")? as usize;

    println!("| field | GRIB2 | cells | ref quantum | max abs diff | RMS diff | mean diff (ours - ref) | cells beyond half quantum | ours min..max | ref min..max |");
    println!("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |");
    let mut rows = Vec::new();
    let mut entries: Vec<Value> = manifest["fields"].as_array().ok_or("manifest fields")?.clone();
    if let Some(extra) = manifest["diagnostics"].as_array() {
        entries.extend(extra.iter().cloned());
    }
    for f in &entries {
        let id = f["id"].as_str().unwrap_or("?");
        let g = &f["grib2"];
        let key = (
            g["discipline"].as_u64().unwrap_or(0) as u8,
            g["category"].as_u64().unwrap_or(0) as u8,
            g["number"].as_u64().unwrap_or(0) as u8,
            g["level_type"].as_u64().unwrap_or(0) as u8,
            g["level_value"].as_f64().unwrap_or(0.0),
        );
        let ident = format!("{}.{}.{} {}/{}", key.0, key.1, key.2, key.3, key.4);
        if f["available"] != json!(true) {
            let why = f["omitted"].as_str().unwrap_or("");
            println!("| {id} | {ident} | omitted: {why} | | | | | | | |");
            rows.push(json!({"id": id, "omitted": why}));
            continue;
        }
        let msg = grib.messages.iter().find(|m| {
            m.discipline == key.0
                && m.product.parameter_category == key.1
                && m.product.parameter_number == key.2
                && m.product.level_type == key.3
                && (m.product.level_value - key.4).abs() < 1e-6
                && m.grid.nx as usize == nx
                && m.grid.ny as usize == ny
        });
        let Some(msg) = msg else {
            println!("| {id} | {ident} | not in reference | | | | | | | |");
            rows.push(json!({"id": id, "reference": "absent"}));
            continue;
        };
        let mut refv = unpack_message(msg)?;
        if msg.grid.scan_mode & 0x40 == 0 {
            // rows stored north to south: flip to the WRF south-to-north order
            for j in 0..ny / 2 {
                for i in 0..nx {
                    refv.swap(j * nx + i, (ny - 1 - j) * nx + i);
                }
            }
        }
        if let Some(d) = &dump {
            let b: Vec<u8> = refv.iter().flat_map(|v| (*v as f32).to_le_bytes()).collect();
            std::fs::write(d.join(format!("{id}.f32")), b)?;
        }
        let bytes = std::fs::read(ours.join(format!("{id}.f32")))?;
        let our: Vec<f64> = bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f64).collect();
        let (mut n, mut maxd, mut ss, mut sum) = (0usize, 0.0f64, 0.0f64, 0.0f64);
        let (mut omin, mut omax, mut rmin, mut rmax) = (f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::NEG_INFINITY);
        let mut only_ours = 0usize;
        let mut only_ref = 0usize;
        for (o, r) in our.iter().zip(&refv) {
            match (o.is_finite(), r.is_finite()) {
                (true, true) => {
                    let d = o - r;
                    n += 1;
                    maxd = maxd.max(d.abs());
                    ss += d * d;
                    sum += d;
                    omin = omin.min(*o);
                    omax = omax.max(*o);
                    rmin = rmin.min(*r);
                    rmax = rmax.max(*r);
                }
                (true, false) => only_ours += 1,
                (false, true) => only_ref += 1,
                _ => {}
            }
        }
        // Packing step of the reference: the smallest gap between distinct values.
        let mut distinct: Vec<f64> = refv.iter().copied().filter(|v| v.is_finite()).collect();
        distinct.sort_by(|a, b| a.partial_cmp(b).unwrap());
        distinct.dedup();
        let quantum = distinct.windows(2).map(|w| w[1] - w[0]).fold(f64::INFINITY, f64::min);
        let beyond = our
            .iter()
            .zip(&refv)
            .filter(|(o, r)| o.is_finite() && r.is_finite() && (*o - *r).abs() > 0.5 * quantum * (1.0 + 1e-9) + 1e-12)
            .count();
        let rms = if n > 0 { (ss / n as f64).sqrt() } else { f64::NAN };
        let mean = if n > 0 { sum / n as f64 } else { f64::NAN };
        println!(
            "| {id} | {ident} | {n} | {quantum:.3e} | {maxd:.6} | {rms:.6} | {mean:.6} | {beyond} | {omin:.4}..{omax:.4} | {rmin:.4}..{rmax:.4} |"
        );
        rows.push(json!({
            "id": id, "grib2": ident, "cells": n, "ref_quantum": quantum, "cells_beyond_half_quantum": beyond, "max_abs_diff": maxd, "rms_diff": rms, "mean_diff": mean,
            "ours_min": omin, "ours_max": omax, "ref_min": rmin, "ref_max": rmax,
            "missing_only_in_ref": only_ours, "missing_only_in_ours": only_ref,
        }));
    }
    if let Some(p) = json_out {
        std::fs::write(p, serde_json::to_vec_pretty(&json!({"reference": args[0], "ours": args[1], "rows": rows}))?)?;
    }
    Ok(())
}
