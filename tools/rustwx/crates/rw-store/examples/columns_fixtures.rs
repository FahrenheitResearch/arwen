//! Small store files of every version, and what this crate's readers decode from them, for readers written in other
//! languages (the WOOF soundings server reads all of them) to check themselves against, bit for bit.
//!
//!   cargo run --release -p rw-store --example columns_fixtures -- OUT_DIR
//!
//! Writes OUT_DIR/grid.rwg, v1-native.rws (native-level DA names: pres_native hPa, tmp_native K, ...; psfc Pa,
//! t2m K, ...), v1-iso.rws (isobaric: temperature_iso K, height_iso, rh_iso, u_iso, v_iso; surface_pressure Pa,
//! ...), v2-affine.wcol (v1-native transcoded, every value exact), v2-quantized.wcol and v2-exact.wcol (the canonical
//! fields from the same values), and expected.json: per file, per point (ix, iy), every stored field's column (3-D)
//! or value (2-D) as f32 bit patterns, read by HourReader or ColumnFile.

use rw_store::columns::{self as cs, ColumnFile, ColumnsWrite, Field2, Field3, Values};
use rw_store::reader::HourReader;
use rw_store::writer::HourWriter;
use serde_json::{Value, json};
use std::path::Path;

const NX: usize = 41;
const NY: usize = 35;
const NZ: usize = 9;

fn vol(f: &dyn Fn(usize, usize, usize) -> f32) -> Vec<Vec<f32>> {
    (0..NZ).map(|k| (0..NX * NY).map(|i| f(k, i % NX, i / NX)).collect()).collect()
}

fn main() {
    let out = std::path::PathBuf::from(std::env::args().nth(1).expect("usage: columns_fixtures OUT_DIR"));
    std::fs::create_dir_all(&out).unwrap();
    let n = NX * NY;
    // A rotated, sheared grid (like a Lambert nest).
    let lat: Vec<f32> = (0..n).map(|i| 38.0 + 0.027 * (i / NX) as f32 + 0.004 * (i % NX) as f32).collect();
    let lon: Vec<f32> = (0..n).map(|i| -99.0 + 0.034 * (i % NX) as f32 - 0.006 * (i / NX) as f32).collect();
    let grid = rustwx_core::LatLonGrid::new(rustwx_core::GridShape::new(NX, NY).unwrap(), lat.clone(), lon.clone()).unwrap();
    let hash = rw_store::grid::write_grid(&out.join("grid.rwg"), &grid, None).unwrap();
    let levels: Vec<u16> = (0..NZ).map(|k| 1000 - 100 * k as u16).collect();
    // Native-level fields, lowest level first: p hPa, z m, T K, Td K, u, v, omega (NaN scattered, one all-missing
    // chunk, one constant chunk).
    let native = [
        ("pres_native", "hPa", vol(&|k, x, y| 985.0 - 105.0 * k as f32 + 0.11 * x as f32 - 0.03 * y as f32)),
        ("hght_native", "m", vol(&|k, x, y| 310.0 + 1150.0 * k as f32 + 0.5 * y as f32 + 0.2 * x as f32)),
        ("tmp_native", "K", vol(&|k, x, _| 298.15 - 7.1 * k as f32 + 0.031 * x as f32)),
        ("dpt_native", "K", vol(&|k, _, y| 291.0 - 9.3 * k as f32 - 0.021 * y as f32)),
        ("u_native", "m/s", vol(&|k, x, y| 3.0 + 4.1 * k as f32 + 0.013 * x as f32 - 0.007 * y as f32)),
        ("v_native", "m/s", vol(&|k, _, y| 6.0 - 1.5 * k as f32 + 0.02 * y as f32)),
        ("omega_native", "Pa/s", vol(&|k, x, y| {
            if x >= 32 && y >= 32 {
                f32::NAN
            } else if x < 16 && y < 16 {
                0.25
            } else if (x + 3 * y + k) % 13 == 0 {
                f32::NAN
            } else {
                0.1 * ((x as f32) / 5.0).sin() - 0.01 * k as f32
            }
        })),
    ];
    let sfc2 = |f: &dyn Fn(usize, usize) -> f32| -> Vec<f32> { (0..n).map(|i| f(i % NX, i / NX)).collect() };
    let native_sfc = [
        ("psfc", "Pa", sfc2(&|x, _| 98_500.0 + 3.0 * x as f32)),
        ("hgt_sfc", "m", sfc2(&|_, y| 250.0 + y as f32)),
        ("t2m", "K", sfc2(&|x, y| 300.15 + 0.01 * x as f32 - 0.02 * y as f32)),
        ("d2m", "K", sfc2(&|x, _| 292.15 - 0.01 * x as f32)),
        ("u10", "m/s", sfc2(&|x, _| 2.0 + 0.01 * x as f32)),
        ("v10", "m/s", sfc2(&|_, y| 4.0 - 0.01 * y as f32)),
    ];
    let write_v1 = |path: &Path, vols: &[(&str, &str, Vec<Vec<f32>>)], sfcs: &[(&str, &str, Vec<f32>)]| {
        let mut w = HourWriter::new("wrf", "fixture", 2, NX, NY, &hash, "columns_fixtures");
        for (name, units, planes) in vols {
            let refs: Vec<&[f32]> = planes.iter().map(|p| p.as_slice()).collect();
            w.add_pressure3d(name, units, json!({}), &levels, &refs).unwrap();
        }
        for (name, units, v) in sfcs {
            w.add_surface2d(name, units, json!({}), v).unwrap();
        }
        w.finish(path).unwrap();
    };
    write_v1(&out.join("v1-native.rws"), &native, &native_sfc);
    let iso = [
        ("height_iso", "m", native[1].2.clone()),
        ("temperature_iso", "K", native[2].2.clone()),
        ("rh_iso", "%", vol(&|k, x, y| (85.0 - 6.0 * k as f32 + 0.1 * x as f32 - 0.05 * y as f32).clamp(1.0, 100.0))),
        ("u_iso", "m/s", native[4].2.clone()),
        ("v_iso", "m/s", native[5].2.clone()),
    ];
    let iso_sfc = [
        ("surface_pressure", "Pa", native_sfc[0].2.clone()),
        ("orography", "m", native_sfc[1].2.clone()),
        ("temperature_2m", "K", native_sfc[2].2.clone()),
        ("dewpoint_2m", "K", native_sfc[3].2.clone()),
        ("u_10m", "m/s", native_sfc[4].2.clone()),
        ("v_10m", "m/s", native_sfc[5].2.clone()),
    ];
    write_v1(&out.join("v1-iso.rws"), &iso, &iso_sfc);
    cs::transcode_v1(&out.join("v1-native.rws"), &out.join("grid.rwg"), &out.join("v2-affine.wcol"), "d01", "mem001", 16)
        .unwrap();
    // The canonical v2 fields from the native values (K -> C, Pa -> hPa), quantized and exact.
    let flat: Vec<Vec<f32>> = native.iter().map(|(_, _, p)| p.concat()).collect();
    let off3 = [0.0, 0.0, -273.15, -273.15, 0.0, 0.0, 0.0];
    let sfc_conv = [(0.01, 0.0), (1.0, 0.0), (1.0, -273.15), (1.0, -273.15), (1.0, 0.0), (1.0, 0.0)];
    for exact in [false, true] {
        let w = ColumnsWrite {
            nx: NX,
            ny: NY,
            nz: NZ,
            lat: Values::F32(&lat),
            lon: Values::F32(&lon),
            fields3: (0..7)
                .map(|i| Field3 {
                    name: cs::FIELDS_3D[i],
                    units: cs::UNITS_3D[i],
                    quantum: if exact { None } else { Some(cs::QUANTA_3D[i]) },
                    data: Values::F32(&flat[i]),
                    scale: 1.0,
                    offset: off3[i],
                    affine: None,
                })
                .collect(),
            fields2: (0..6)
                .map(|i| Field2 {
                    name: cs::FIELDS_2D[i + 2],
                    units: cs::UNITS_2D[i + 2],
                    data: Values::F32(&native_sfc[i].2),
                    scale: sfc_conv[i].0,
                    offset: sfc_conv[i].1,
                })
                .collect(),
            domain: "d01",
            member: "mem001",
            valid_unix: 1_776_394_800,
            tile: 16,
            meta: json!({"writer": "columns_fixtures"}),
        };
        let name = if exact { "v2-exact.wcol" } else { "v2-quantized.wcol" };
        cs::write(&out.join(name), &w).unwrap();
    }
    // What the readers decode, at points on tile edges, the grid's last row and column, and inside the missing chunk.
    let points = [(0usize, 0usize), (5, 7), (15, 15), (16, 16), (17, 3), (31, 33), (33, 33), (40, 34), (40, 0), (22, 29)];
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();
    let mut expected = serde_json::Map::new();
    for file in ["v1-native.rws", "v1-iso.rws"] {
        let r = HourReader::open(&out.join(file)).unwrap();
        let mut rows = vec![];
        for (ix, iy) in points {
            let mut f = serde_json::Map::new();
            for v in &r.meta().variables {
                let val: Value = if v.kind == "pressure3d" {
                    json!(bits(&r.read_column_3d(&v.name, ix, iy).unwrap()))
                } else {
                    json!(r.read_full_2d(&v.name).unwrap()[iy * NX + ix].to_bits())
                };
                f.insert(v.name.clone(), val);
            }
            rows.push(json!({"ix": ix, "iy": iy, "fields": f}));
        }
        expected.insert(file.into(), json!(rows));
    }
    for file in ["v2-affine.wcol", "v2-quantized.wcol", "v2-exact.wcol"] {
        let c = ColumnFile::open(&out.join(file)).unwrap();
        let h = &c.header;
        let mut rows = vec![];
        for (ix, iy) in points {
            let t = c.read_tile(h.tile_of(ix, iy).0, h.tile_of(ix, iy).1).unwrap();
            let mut f = serde_json::Map::new();
            for (i, name) in h.fields3d.iter().enumerate() {
                f.insert(name.clone(), json!(bits(&t.column3(i, ix, iy))));
            }
            for (i, name) in h.fields2d.iter().enumerate() {
                f.insert(name.clone(), json!(t.point2(i, ix, iy).to_bits()));
            }
            rows.push(json!({"ix": ix, "iy": iy, "fields": f}));
        }
        expected.insert(file.into(), json!(rows));
    }
    let sizes: serde_json::Map<String, Value> = ["v1-native.rws", "v1-iso.rws", "v2-affine.wcol", "v2-quantized.wcol", "v2-exact.wcol"]
        .iter()
        .map(|f| (f.to_string(), json!(std::fs::metadata(out.join(f)).unwrap().len())))
        .collect();
    std::fs::write(
        out.join("expected.json"),
        serde_json::to_string(&json!({"points": points, "files": expected, "bytes": sizes})).unwrap(),
    )
    .unwrap();
    println!("{}", json!({"out": out, "bytes": sizes}));
}
