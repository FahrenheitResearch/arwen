//! Driver-boundary case-8 replay. WPS/NetCDF fixture data is written and compared
//! in Rust. Optional pristine Fortran oracle includes the entire surface module.
use rw_store::netcdf_classic::{NcClassicWriter, NcData, NcDim, NcFormat, NcType, NcVarDef};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
#[path = "common/surface_native.rs"]
mod surface_native;

const STAMPS: [&str; 5] = [
    "2000-02-29_18:00:00",
    "2000-03-01_00:00:00",
    "2000-03-01_06:00:00",
    "2000-03-01_12:00:00",
    "2000-03-01_18:00:00",
];
const NX: usize = 16;
const NY: usize = 9;
const NC: usize = 640;

fn record(out: &mut Vec<u8>, bytes: &[u8]) {
    out.extend_from_slice(&(bytes.len() as u32).to_be_bytes());
    out.extend_from_slice(bytes);
    out.extend_from_slice(&(bytes.len() as u32).to_be_bytes());
}
fn text(out: &mut Vec<u8>, value: &str, n: usize) {
    let mut b = value.as_bytes().to_vec();
    b.resize(n, b' ');
    out.extend_from_slice(&b);
}
fn f32be(out: &mut Vec<u8>, x: f32) {
    out.extend_from_slice(&x.to_be_bytes());
}
fn field(out: &mut Vec<u8>, name: &str, stamp: &str, values: &[f32], scenario: usize) {
    let version = match scenario {
        3 | 9 => 3i32,
        4 | 10 => 4i32,
        _ => 5i32,
    };
    let gaussian = scenario == 5 || scenario == 11;
    let polar = [6, 7, 9, 10].contains(&scenario);
    let ignored_projection = if scenario == 14 {
        1i32
    } else if scenario == 15 {
        3i32
    } else if name == "IGNORED" {
        match scenario {
            12 => 1i32,
            13 => 3i32,
            _ => 0i32,
        }
    } else {
        0i32
    };
    let ny = if gaussian { 8 } else { NY };
    record(out, &version.to_be_bytes());
    let mut h = Vec::new();
    text(&mut h, stamp, 24);
    f32be(&mut h, 0.0);
    if version >= 4 {
        text(&mut h, "synthetic branch replay", 32);
    }
    text(&mut h, name, 9);
    text(&mut h, "unit", 25);
    text(&mut h, "test", 46);
    f32be(&mut h, 200100.0);
    h.extend_from_slice(&(NX as i32).to_be_bytes());
    h.extend_from_slice(&(ny as i32).to_be_bytes());
    h.extend_from_slice(
        &(if ignored_projection != 0 {
            ignored_projection
        } else if gaussian {
            4i32
        } else if polar {
            5i32
        } else {
            0i32
        })
        .to_be_bytes(),
    );
    record(out, &h);
    let mut p = Vec::new();
    if version >= 4 {
        text(&mut p, "SWCORNER", 8);
    }
    if ignored_projection != 0 {
        for v in [-45.0, 0.0, 100.0, 100.0] {
            f32be(&mut p, v);
        }
        if ignored_projection == 3 {
            f32be(&mut p, 0.0);
        }
        f32be(&mut p, 60.0);
        if ignored_projection == 3 {
            f32be(&mut p, 60.0);
        }
    } else if polar {
        for v in [
            if scenario == 7 { -60.0 } else { 60.0 },
            0.0,
            1500.0,
            1500.0,
            0.0,
            if scenario == 7 { -60.0 } else { 60.0 },
        ] {
            f32be(&mut p, v);
        }
    } else {
        for v in [
            if gaussian {
                if scenario == 11 {
                    -73.79921
                } else {
                    73.79921
                }
            } else if scenario == 8 {
                90.0
            } else {
                -90.0
            },
            if scenario == 8 { 180.0 } else { 0.0 },
            if gaussian {
                4.0
            } else if scenario == 8 {
                -22.5
            } else {
                22.5
            },
            if scenario == 8 { -22.5 } else { 22.5 },
        ] {
            f32be(&mut p, v);
        }
    }
    if version == 5 {
        f32be(&mut p, 6370.0);
    }
    record(out, &p);
    if version == 5 {
        record(out, &0i32.to_be_bytes());
    }
    let mut b = Vec::new();
    for &v in values {
        f32be(&mut b, v);
    }
    record(out, &b);
}
fn mesh(path: &Path, all_water: bool, scenario: usize) {
    let mut lat = Vec::new();
    let mut lon = Vec::new();
    let mut land = Vec::new();
    let deg = 180.0f32 / std::f32::consts::PI;
    let mut seed = 18731u32;
    for i in 0..NC {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        let a = if i < 12 {
            [
                -100.0, -90.0, -78.75, 0.0, 78.75, 90.0, 100.0, 11.25, 22.5, -11.25, -90.0, 90.0,
            ][i]
        } else {
            (seed % 20000) as f32 / 100.0 - 100.0
        };
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        let b = if i < 12 {
            [
                0.0, -180.0, 360.0, 11.25, 348.75, -0.1, 359.99, 45.0, 90.0, 180.0, -180.0, 360.0,
            ][i]
        } else {
            (seed % 72000) as f32 / 100.0 - 180.0
        };
        let a = if [6, 7, 9, 10].contains(&scenario) {
            (a * 0.4 + 45.0) * if scenario == 7 { -1.0 } else { 1.0 }
        } else {
            a
        };
        lat.push(a / deg);
        lon.push(b / deg);
        land.push(if all_water {
            0
        } else {
            match i % 7 {
                0 => 1,
                1 => -1,
                _ => 0,
            }
        });
    }
    let mut w = NcClassicWriter::create(
        path,
        NcFormat::Data64,
        vec![NcDim::fixed("nCells", NC)],
        vec![],
        vec![
            NcVarDef::new("latCell", NcType::Float, vec![0]),
            NcVarDef::new("lonCell", NcType::Float, vec![0]),
            NcVarDef::new("landmask", NcType::Int, vec![0]),
        ],
        0,
    )
    .unwrap();
    w.put("latCell", NcData::Floats(&lat)).unwrap();
    w.put("lonCell", NcData::Floats(&lon)).unwrap();
    w.put("landmask", NcData::Ints(&land)).unwrap();
    w.finish().unwrap();
}
fn inputs(root: &Path, scenario: usize) {
    let mut clock = format!("5 21600\n");
    for (r, stamp) in STAMPS.iter().enumerate() {
        clock.push_str(stamp);
        clock.push('\n');
        let mut out = Vec::new();
        let ny = if scenario == 5 || scenario == 11 {
            8
        } else {
            NY
        };
        let mut sst = vec![0.0; NX * ny];
        let mut ice = vec![0.0; NX * ny];
        let mut land = vec![0.0; NX * ny];
        for i in 0..NX * ny {
            sst[i] = if scenario == 2 {
                -1e30
            } else if r == 0 {
                [270.0, 271.0, 300.0, 280.125][i % 4]
            } else if (i + r) % 4 == 0 {
                -1e30
            } else {
                265.0 + ((i * 17 + r * 31) % 3500) as f32 / 100.0
            };
            ice[i] = if scenario == 2 {
                -1e30
            } else if r == 3 {
                [-0.5, 1.5, 0.0, 1.0][i % 4]
            } else if (i + r) % 3 == 0 {
                -1e30
            } else {
                ((i * 31 + r * 23) % 101) as f32 / 100.0
            };
            land[i] = if (i / NX + i % NX) % 3 == 0 { 1.0 } else { 0.0 };
        }
        if r != 3 && r != 4 {
            field(&mut out, "SKINTEMP", stamp, &vec![310.0; NX * ny], scenario);
            field(&mut out, "SST", stamp, &sst, scenario);
        }
        if r == 2 {
            field(&mut out, "SKINTEMP", stamp, &vec![271.0; NX * ny], scenario);
        }
        if r != 4 {
            field(&mut out, "SEAICE", stamp, &ice, scenario);
        }
        field(&mut out, "IGNORED", stamp, &vec![19.0; NX * ny], scenario);
        if scenario == 0 || scenario >= 3 {
            field(&mut out, "LANDSEA", stamp, &vec![1.0; NX * ny], scenario);
            field(&mut out, "LANDSEA", stamp, &land, scenario);
        }
        fs::write(root.join(format!("SFC:{}", &stamp[..13])), out).unwrap();
    }
    fs::write(root.join("clock.txt"), clock).unwrap();
}
fn command(root: &Path) -> Command {
    let mut c = Command::new(
        std::env::var_os("MPAS_SURFACE_ENGINE")
            .unwrap_or_else(|| env!("CARGO_BIN_EXE_rw_mpas_init").into()),
    );
    c.args([
        "--mode",
        "surface-update",
        "--start-time",
        STAMPS[0],
        "--stop-time",
        STAMPS[4],
        "--fg-interval",
        "21600",
        "--surface-interval",
        "21600",
    ]);
    c.arg("--sfc-prefix")
        .arg(root.join("SFC"))
        .arg("--static")
        .arg(root.join("static.nc"))
        .arg("--out")
        .arg(root.join("surface.nc"))
        .arg("--receipt")
        .arg(root.join("engine.json"));
    c
}
fn root(label: &str) -> PathBuf {
    // Lane runs set an explicit scratch root. Ordinary tests use the crate's
    // own target tree rather than /tmp.
    let base = std::env::var_os("MPAS_SURFACE_TEST_OUT")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/surface-tests"));
    let root = base.join(format!("{label}-{}", std::process::id()));
    fs::create_dir_all(&root).unwrap();
    root
}
#[test]
fn surface_front_door_calendar_schema_and_refusals() {
    let root = root("front-door");
    mesh(&root.join("static.nc"), false, 0);
    inputs(&root, 0);
    let output = command(&root).output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let file = netcrust::File::open(root.join("surface.nc")).unwrap();
    assert_eq!(
        file.variables()
            .unwrap()
            .iter()
            .map(|v| v.name())
            .collect::<Vec<_>>(),
        ["xtime", "Time", "sst", "xice"]
    );
    assert_eq!(
        file.read_array_f64("Time").unwrap().values(),
        [0.0, 21600.0, 43200.0, 64800.0, 86400.0]
    );
    assert_eq!(
        file.read_strings("xtime")
            .unwrap()
            .iter()
            .map(|s| s.trim())
            .collect::<Vec<_>>(),
        STAMPS
    );
    let before = fs::read(root.join("surface.nc")).unwrap();
    let bad = command(&root)
        .args(["--surface-interval", "43200"])
        .output()
        .unwrap();
    assert!(!bad.status.success());
    assert!(String::from_utf8_lossy(&bad.stderr).contains("config_fg_interval"));
    assert_eq!(fs::read(root.join("surface.nc")).unwrap(), before);
    inputs(&root, 0);
    let first = root.join(format!("SFC:{}", &STAMPS[0][..13]));
    let original = fs::read(&first).unwrap();
    let mut corrupt = original.clone();
    corrupt[11] ^= 1;
    fs::write(&first, corrupt).unwrap();
    let bad = command(&root).output().unwrap();
    assert!(!bad.status.success());
    assert!(String::from_utf8_lossy(&bad.stderr).contains("record markers disagree"));
    assert_eq!(fs::read(root.join("surface.nc")).unwrap(), before);
    fs::write(&first, &original).unwrap();
    let bad = command(&root).arg("--out").arg(&first).output().unwrap();
    assert!(!bad.status.success());
    assert!(String::from_utf8_lossy(&bad.stderr).contains("aliases an intermediate"));
    assert_eq!(fs::read(&first).unwrap(), original);
    fs::remove_file(root.join(format!("SFC:{}", &STAMPS[4][..13]))).unwrap();
    let missing = command(&root).output().unwrap();
    assert!(!missing.status.success());
    assert_eq!(fs::read(root.join("surface.nc")).unwrap(), before);
    mesh(&root.join("static.nc"), true, 0);
    let bad = command(&root).output().unwrap();
    assert!(!bad.status.success());
    assert!(String::from_utf8_lossy(&bad.stderr).contains("maximum of landmask is zero"));
    fs::remove_dir_all(root).unwrap();
}

#[test]
#[ignore = "requires pristine MPAS Fortran oracle built under build-slot"]
fn pristine_surface_module_error_paths() {
    let oracle = std::env::var_os("MPAS_SURFACE_ORACLE")
        .expect("MPAS_SURFACE_ORACLE must name the pristine module executable");
    let root = root("native-errors");
    let mut rows = Vec::new();
    for (label, line, expected) in [
        ("zero-landmask", 157, "global maximum"),
        ("missing-first-open", 169, "Error opening surface file"),
        ("missing-second-open", 200, "Error opening surface file"),
    ] {
        let case = root.join(label);
        fs::create_dir_all(&case).unwrap();
        mesh(&case.join("static.nc"), label == "zero-landmask", 0);
        inputs(&case, 0);
        let first = case.join(format!("SFC:{}", &STAMPS[0][..13]));
        if label == "missing-first-open" {
            fs::remove_file(&first).unwrap();
        }
        let mut native = Command::new(&oracle);
        native
            .arg(case.join("static.nc"))
            .arg(case.join("SFC"))
            .arg(case.join("clock.txt"))
            .arg(case.join("native.words"))
            .env_remove("MPAS_SURFACE_ORACLE_DROP_AFTER_SCAN");
        if label == "missing-second-open" {
            native.env("MPAS_SURFACE_ORACLE_DROP_AFTER_SCAN", &first);
        }
        let output = native.output().unwrap();
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert_eq!(output.status.code(), Some(2), "{stdout} {stderr}");
        assert!(stdout.contains(expected), "{stdout} {stderr}");
        assert_eq!(fs::metadata(case.join("native.words")).unwrap().len(), 0);
        if label == "missing-second-open" {
            assert!(
                !first.exists(),
                "the framework I/O fault must remove the source"
            );
        }
        rows.push(serde_json::json!({
            "scenario": label,
            "mpas_source": format!("mpas_init_atm_surface.F:{line}"),
            "exit_code": output.status.code(),
            "diagnostic_matched": true,
            "output_bytes": 0,
            "framework_io_fault": label == "missing-second-open"
        }));
        fs::remove_dir_all(case).unwrap();
    }
    fs::write(
        root.join("errors.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "tier": "standalone-module-error-paths",
            "native_processes": 1,
            "rust_processes": 1,
            "whole_model_proof": false,
            "rows": rows
        }))
        .unwrap(),
    )
    .unwrap();
}

#[test]
#[ignore = "requires pristine MPAS Fortran oracle built under build-slot"]
fn pristine_surface_module_word_replay() {
    let oracle = std::env::var_os("MPAS_SURFACE_ORACLE")
        .expect("MPAS_SURFACE_ORACLE must name the pristine module executable");
    let root = root("native-replay");
    let mut rows = Vec::new();
    let scenario_labels = [
        "latlon-masked",
        "latlon-unmasked",
        "all-missing-unmasked",
        "format-3-latlon",
        "format-4-latlon",
        "gaussian-north-first",
        "polar-north",
        "polar-south",
        "latlon-negative-increments",
        "format-3-polar",
        "format-4-polar",
        "gaussian-south-first",
        "ignored-mercator",
        "ignored-lambert",
    ];
    for scenario in 0..14 {
        let case = root.join(format!("scenario-{scenario}"));
        fs::create_dir_all(&case).unwrap();
        mesh(&case.join("static.nc"), false, scenario);
        inputs(&case, scenario);
        let output = command(&case).output().unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let native = Command::new(&oracle)
            .arg(case.join("static.nc"))
            .arg(case.join("SFC"))
            .arg(case.join("clock.txt"))
            .arg(case.join("native.words"))
            .output()
            .unwrap();
        assert!(
            native.status.success(),
            "{} {}",
            String::from_utf8_lossy(&native.stdout),
            String::from_utf8_lossy(&native.stderr)
        );
        let raw = fs::read(case.join("native.words")).unwrap();
        assert_eq!(raw.len(), 5 * (64 + 4 + 8 * NC));
        let file = netcrust::File::open(case.join("surface.nc")).unwrap();
        let time = file.read_array_f64("Time").unwrap().into_values();
        let sst = file.read_array_f64("sst").unwrap().into_values();
        let ice = file.read_array_f64("xice").unwrap().into_values();
        let labels = file.read_strings("xtime").unwrap();
        let mut mismatches = Vec::new();
        for r in 0..5 {
            let offset = r * (64 + 4 + 8 * NC);
            assert_eq!(labels[r].as_bytes(), &raw[offset..offset + 64]);
            let numbers = &raw[offset + 64..offset + 64 + 4 + 8 * NC];
            let got = std::iter::once(time[r] as f32)
                .chain(sst[r * NC..(r + 1) * NC].iter().map(|&v| v as f32))
                .chain(ice[r * NC..(r + 1) * NC].iter().map(|&v| v as f32));
            for (i, (g, b)) in got.zip(numbers.chunks_exact(4)).enumerate() {
                let want = u32::from_be_bytes(b.try_into().unwrap());
                if g.to_bits() != want {
                    mismatches.push(format!(
                        "record {r}, word {i}: Rust {:08x} native {want:08x}",
                        g.to_bits()
                    ));
                }
            }
        }
        rows.push(serde_json::json!({"scenario":scenario,"native_processes":1,"rust_processes":1,"label":scenario_labels[scenario],"records":5,"cells":NC,"numeric_words":5*(2*NC+1),"xtime_bytes":320,"different_words":mismatches.len(),"native_words_sha256":format!("{:x}",Sha256::digest(&raw)),"rust_output_sha256":format!("{:x}",Sha256::digest(fs::read(case.join("surface.nc")).unwrap()))}));
        fs::write(root.join("replay.json"),serde_json::to_vec_pretty(&serde_json::json!({"tier":"standalone-module-replay","native_processes":1,"rust_processes":1,"whole_model_proof":false,"host":fs::read_to_string("/proc/sys/kernel/hostname").unwrap_or_default().trim(),"rows":rows})).unwrap()).unwrap();
        assert!(
            mismatches.is_empty(),
            "{} differing words:\n{}",
            mismatches.len(),
            mismatches
                .iter()
                .take(20)
                .cloned()
                .collect::<Vec<_>>()
                .join("\n")
        );
        fs::remove_dir_all(&case).unwrap();
    }
}

#[test]
#[ignore = "requires f0 strict native executable and full published mesh input"]
fn strict_native_projection_refusals() {
    let native = PathBuf::from(std::env::var_os("MPAS_SURFACE_NATIVE").unwrap());
    let static_path = PathBuf::from(std::env::var_os("MPAS_SURFACE_NATIVE_STATIC").unwrap());
    let graph_prefix = PathBuf::from(std::env::var_os("MPAS_SURFACE_NATIVE_GRAPH_PREFIX").unwrap());
    let root = root("native-projections");
    let mut rows = Vec::new();
    for scenario in [14, 15] {
        let case = root.join(format!("scenario-{scenario}"));
        fs::create_dir_all(&case).unwrap();
        inputs(&case, scenario);
        for ranks in [1, 2] {
            let work = case.join(format!("native-ranks-{ranks}"));
            let result = surface_native::run(
                &native,
                &static_path,
                &case.join("SFC"),
                &graph_prefix,
                STAMPS[0],
                STAMPS[0],
                21600,
                ranks,
                &work,
            );
            assert_eq!(result["success"], false, "{result}");
            let logs = result["logs"].as_str().unwrap();
            assert!(
                logs.contains("You have not called map_set for this projection"),
                "{logs}"
            );
            assert!(logs.contains("LATLON_TO_IJ"), "{logs}");
            let rust = command(&case)
                .arg("--static")
                .arg(&static_path)
                .args(["--stop-time", STAMPS[0]])
                .output()
                .unwrap();
            assert!(!rust.status.success());
            assert!(String::from_utf8_lossy(&rust.stderr)
                .contains("is not initialized by native case 8"));
            rows.push(serde_json::json!({
                "projection":if scenario==14 {"Mercator"} else {"Lambert"},
                "native":result,
                "rust_refused_same_option":true,
                "mpas_source":"mpas_init_atm_surface.F:324-349; mpas_init_atm_llxy.F:235-272,601-604,2225-2233",
                "undefined_output_words":[],
                "diagnosis":"map_init sets init=false; latlon_to_ij explicitly aborts before reading unset mapping data"
            }));
        }
        fs::remove_dir_all(case).unwrap();
    }
    fs::write(
        root.join("projection-refusals.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "schema":"mpas.surface-native-projection-refusal/v1",
            "native_ranks_tested":[1,2],
            "rust_processes":1,
            "host":fs::read_to_string("/proc/sys/kernel/hostname").unwrap_or_default().trim(),
            "whole_model_proof":false,
            "rows":rows
        }))
        .unwrap(),
    )
    .unwrap();
}
