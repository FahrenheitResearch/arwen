//! Public fixed-height W products on analytic terrain-following columns.
//! No thermodynamic input is present. AGL, sign, destaggering, units,
//! missing values and the actual renderer artifact are checked together.
use netcdf_writer::{AttrValue, NcFormat, NcType, NcWriter, Schema, VarData};
use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

const NX: usize = 24;
const NY: usize = 18;
const NZ: usize = 4;
const CELLS: usize = NX * NY;
const G: f64 = 9.80665;

fn frame(root: &Path, with_w: bool) -> PathBuf {
    let path = root.join("wrfout_d01_2026-10-17_00_00_00");
    let mut s = Schema::new(NcFormat::Offset64);
    let t = s.def_dim("Time", 0, true).unwrap();
    let date = s.def_dim("DateStrLen", 19, false).unwrap();
    let z = s.def_dim("bottom_top", NZ, false).unwrap();
    let zs = s.def_dim("bottom_top_stag", NZ + 1, false).unwrap();
    let y = s.def_dim("south_north", NY, false).unwrap();
    let x = s.def_dim("west_east", NX, false).unwrap();
    for name in ["START_DATE", "SIMULATION_START_DATE"] {
        s.put_global_attr(name, AttrValue::Text("2026-10-17_00:00:00".into()))
            .unwrap();
    }
    s.put_global_attr("MAP_PROJ", AttrValue::Ints(vec![6]))
        .unwrap();
    s.put_global_attr("GRID_ID", AttrValue::Ints(vec![1]))
        .unwrap();
    let times = s.def_var("Times", NcType::Char, &[t, date]).unwrap();
    let mut planes = Vec::new();
    for (name, units, values) in [
        (
            "XLAT",
            "degree_north",
            (0..CELLS)
                .map(|i| 36. + (i / NX) as f32 * 0.05)
                .collect::<Vec<_>>(),
        ),
        (
            "XLONG",
            "degree_east",
            (0..CELLS).map(|i| -98. + (i % NX) as f32 * 0.05).collect(),
        ),
        (
            "HGT",
            "m",
            (0..CELLS)
                .map(|i| 320. + (i % NX) as f32 * 100. + (i / NX) as f32 * 50.)
                .collect(),
        ),
    ] {
        let id = s.def_var(name, NcType::Float, &[t, y, x]).unwrap();
        s.put_var_attr(id, "units", AttrValue::Text(units.into()))
            .unwrap();
        planes.push((id, values));
    }
    // Only T's shape is needed for the WrfFile dimension probe.
    let id = s.def_var("T", NcType::Float, &[t, z, y, x]).unwrap();
    planes.push((id, vec![0.; NZ * CELLS]));
    for name in ["PH", "PHB", "W"] {
        if name == "W" && !with_w {
            continue;
        }
        let id = s.def_var(name, NcType::Float, &[t, zs, y, x]).unwrap();
        s.put_var_attr(
            id,
            "units",
            AttrValue::Text(if name == "W" { "m s-1" } else { "m2 s-2" }.into()),
        )
        .unwrap();
        let values = (0..(NZ + 1) * CELLS)
            .map(|i| {
                let cell = i % CELLS;
                let height = (i / CELLS) as f64 * 1000.;
                match name {
                    "PH" => 0.,
                    "PHB" => ((f64::from(planes[2].1[cell]) + height) * G) as f32,
                    // Both signs at both heights; the first column is missing.
                    _ if cell == 0 => 9.96921e36,
                    _ => (-8. + 16. * (cell % NX) as f64 / (NX - 1) as f64 + 0.001 * height) as f32,
                }
            })
            .collect();
        planes.push((id, values));
    }
    let mut writer = NcWriter::create(&path, s).unwrap();
    writer
        .write_record(0, times, VarData::Char(b"2026-10-17_00:00:00"))
        .unwrap();
    for (id, values) in planes {
        writer.write_record(0, id, VarData::F32(&values)).unwrap();
    }
    writer.finish().unwrap();
    path
}

fn files(root: &Path, extension: &str) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for item in fs::read_dir(root).unwrap() {
        let path = item.unwrap().path();
        if path.is_dir() {
            out.extend(files(&path, extension));
        } else if path.extension().is_some_and(|e| e == extension) {
            out.push(path);
        }
    }
    out
}

#[test]
fn public_w_products_draw_and_store_upward_positive_m_s_on_agl_surfaces() {
    let retained = std::env::var_os("RUSTWX_VERTICAL_VELOCITY_EVIDENCE_ROOT");
    let root = retained
        .clone()
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join(format!("rw-w-{}", std::process::id())));
    fs::create_dir_all(&root).unwrap();
    let input = frame(&root, true);
    let output = Command::new(env!("CARGO_BIN_EXE_rw_wrfbatch"))
        .args([
            "--products",
            "1km_vertical_velocity,3km_vertical_velocity",
            "--width",
            "1000",
            "--height",
            "700",
        ])
        .arg("--store-root")
        .arg(root.join("store"))
        .arg("--out-dir")
        .arg(root.join("pictures"))
        .arg(&input)
        .env("CUDA_VISIBLE_DEVICES", "")
        .env("RAYON_NUM_THREADS", "2")
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    fs::write(
        root.join("renderer.log"),
        format!("{stdout}\n{}", String::from_utf8_lossy(&output.stderr)),
    )
    .unwrap();
    assert!(
        output.status.success(),
        "{stdout}\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    for slug in ["1km_vertical_velocity", "3km_vertical_velocity"] {
        assert!(stdout.contains(&format!("RENDERED {slug} ")), "{stdout}");
    }
    assert!(!stdout.contains("Computing WRF diagnostic"), "{stdout}");
    assert_eq!(files(&root.join("pictures"), "png").len(), 2);
    let hours = files(&root.join("store"), "rws");
    assert_eq!(hours.len(), 1);
    let hour = rw_store::reader::HourReader::open(&hours[0]).unwrap();
    assert_eq!(
        hour.meta().variables.len(),
        3,
        "only W planes and orography are imported"
    );
    for (name, height) in [
        ("vertical_velocity_1km_agl", 1000.),
        ("vertical_velocity_3km_agl", 3000.),
    ] {
        assert_eq!(hour.variable(name).unwrap().units, "m s-1");
        let values = hour.read_full_2d(name).unwrap();
        assert!(
            values[0].is_nan(),
            "finite missing-value sentinel must stay masked"
        );
        let mut max_error = 0.0f64;
        for (i, value) in values.iter().enumerate().skip(1) {
            let expected = -8. + 16. * (i % NX) as f64 / (NX - 1) as f64 + 0.001 * height;
            max_error = max_error.max((f64::from(*value) - expected).abs());
            assert!(
                (f64::from(*value) - expected).abs() < 1e-5,
                "{name} {i}: {value} vs {expected}"
            );
        }
        println!("{name}: 431 finite cells, one masked; max analytic error {max_error:.9} m s-1");
    }
    if retained.is_none() {
        fs::remove_dir_all(&root).unwrap();
    }
}

#[test]
fn missing_w_is_reported_unavailable_in_the_public_catalog() {
    let root = std::env::temp_dir().join(format!("rw-no-w-{}", std::process::id()));
    fs::create_dir_all(&root).unwrap();
    let input = frame(&root, false);
    let output = Command::new(env!("CARGO_BIN_EXE_rw_wrfbatch"))
        .args([
            "--list-products",
            "--products",
            "1km_vertical_velocity,3km_vertical_velocity",
        ])
        .arg("--store-root")
        .arg(root.join("store"))
        .arg("--out-dir")
        .arg(root.join("pictures"))
        .arg(input)
        .env("CUDA_VISIBLE_DEVICES", "")
        .env("RAYON_NUM_THREADS", "2")
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    for slug in ["1km_vertical_velocity", "3km_vertical_velocity"] {
        let row = stdout
            .lines()
            .find(|row| row.starts_with(&format!("PRODUCT\t{slug}\t")))
            .unwrap_or_else(|| panic!("{stdout}"));
        assert!(!row.contains("\trenderable\t"), "{row}");
    }
    let record = files(&root.join("store"), "json")
        .into_iter()
        .find(|path| path.file_name().unwrap() == "import-record.json")
        .unwrap();
    let notes = fs::read_to_string(record).unwrap();
    assert!(
        notes.contains("vertical_velocity_1km_agl unavailable: W:"),
        "{notes}"
    );
    println!("missing W catalog: both products unavailable; import note: {notes}");
    fs::remove_dir_all(root).unwrap();
}
