//! Full case-9 replay against the pristine MPAS v8.4.2 routine, reached
//! through `python -m hexcore.cli lbc` (the gpuwm-hex command entry point).
//! This synthetic oracle is diagnostic evidence, never the f0 forecast proof.
//! Build the Fortran test oracle with tests/lbc_v842/build_oracle.py, then set
//! MPAS_LBC_ORACLE, MPAS_LBC_REPLAY_DIR, and the hex CLI's PYTHONPATH.

use rw_mpas::lbc::compare::compare_lbc;
use rw_mpas::lbc::projection::gaussian_latitudes;
use rw_store::netcdf_classic::{
    NcAttr, NcClassicWriter, NcData, NcDim, NcFormat, NcType, NcVarDef,
};
use sha2::{Digest, Sha256};
use std::fs;
use std::io::{Read, Write};
use std::path::Path;
use std::process::Command;

const NC: usize = 4;
const NE: usize = 5;
const NZ: usize = 4;
const START: &str = "2024-02-28_21:00:00";
const STOP: &str = "2024-02-29_00:00:00";

fn grid(path: &Path, above_top: bool) {
    let dims = vec![
        NcDim::fixed("nCells", NC),
        NcDim::fixed("nEdges", NE),
        NcDim::fixed("nVertLevels", NZ),
        NcDim::fixed("nVertLevelsP1", NZ + 1),
        NcDim::fixed("TWO", 2),
        NcDim::fixed("maxEdges", 2),
    ];
    let mut attrs = [
        "model_name",
        "core_name",
        "version",
        "source",
        "Conventions",
        "git_version",
        "on_a_sphere",
        "is_periodic",
        "mesh_spec",
        "file_id",
    ]
    .iter()
    .map(|n| NcAttr::text(*n, "synthetic-v842-replay"))
    .collect::<Vec<_>>();
    attrs.extend([
        NcAttr::float("sphere_radius", 6371229.0),
        NcAttr::float("x_period", 0.0),
        NcAttr::float("y_period", 0.0),
    ]);
    let mut floats: Vec<(&str, Vec<usize>, Vec<f32>)> = Vec::new();
    let rad = std::f32::consts::PI / 180.0;
    for (name, d, values) in [
        ("latCell", 0, vec![-80.0, -29.75, 7.125, 69.4]),
        ("lonCell", 0, vec![-181.0, -88.4, 47.7, 185.0]),
        ("latEdge", 1, vec![-55.125, -11.31, 38.5, 87.0, -89.0]),
        ("lonEdge", 1, vec![-135.2, -20.35, 115.4, 190.0, -189.0]),
    ] {
        floats.push((name, vec![d], values.iter().map(|x| x * rad).collect()));
    }
    floats.push(("angleEdge", vec![1], vec![0.0, 0.7125, -1.8125, 2.7, -0.0]));
    let z: Vec<f32> = (0..NC)
        .flat_map(|c| {
            [0.0, 450.0, 1800.0, 3700.0, 6400.0]
                .map(|v| v * if above_top { 3.0 } else { 1.0 } + 40.0 * c as f32)
        })
        .collect();
    floats.push(("zgrid", vec![0, 3], z));
    floats.push((
        "zz",
        vec![0, 2],
        (0..NC)
            .flat_map(|c| [1.1, 0.92, 1.04, 0.88].map(|v| v + 0.015 * c as f32))
            .collect(),
    ));
    for (name, values) in [
        ("fzm", vec![0.0, 0.47, 0.43, 0.58, 0.0]),
        ("fzp", vec![0.0, 0.53, 0.57, 0.42, 0.0]),
        ("dzu", vec![0.0, 900.0, 1500.0, 2300.0, 0.0]),
        (
            "rdzw",
            vec![1.0 / 450.0, 1.0 / 1350.0, 1.0 / 1900.0, 1.0 / 2700.0, 0.0],
        ),
    ] {
        floats.push((name, vec![3], values));
    }
    for (name, factor) in [("zb", 0.00019), ("zb3", 0.00031)] {
        floats.push((
            name,
            vec![1, 4, 3],
            (0..NE)
                .flat_map(|e| {
                    (0..2).flat_map(move |s| {
                        (0..NZ + 1).map(move |k| {
                            if k == 0 || k == NZ {
                                0.0
                            } else {
                                factor * (e + 1) as f32 * (s + 1) as f32 * (k + 1) as f32
                            }
                        })
                    })
                })
                .collect(),
        ));
    }
    let ints = [
        (
            "cellsOnEdge",
            vec![1, 4],
            vec![1, 2, 2, 3, 3, 4, 4, 0, 0, 1],
        ),
        ("nEdgesOnCell", vec![0], vec![2; NC]),
        ("edgesOnCell", vec![0, 5], vec![1, 5, 1, 2, 2, 3, 3, 4]),
    ];
    let vars = floats
        .iter()
        .map(|(n, d, _)| NcVarDef::new(*n, NcType::Float, d.clone()))
        .chain(
            ints.iter()
                .map(|(n, d, _)| NcVarDef::new(*n, NcType::Int, d.clone())),
        )
        .collect();
    let mut writer = NcClassicWriter::create(path, NcFormat::Data64, dims, attrs, vars, 0).unwrap();
    for (name, _, values) in floats {
        writer.put(name, NcData::Floats(&values)).unwrap();
    }
    for (name, _, values) in ints {
        writer.put(name, NcData::Ints(&values)).unwrap();
    }
    writer.finish().unwrap();
}

fn record(file: &mut fs::File, payload: &[u8]) {
    let count = (payload.len() as u32).to_be_bytes();
    file.write_all(&count).unwrap();
    file.write_all(payload).unwrap();
    file.write_all(&count).unwrap();
}
fn text(buf: &mut Vec<u8>, value: &str, size: usize) {
    let start = buf.len();
    buf.extend_from_slice(value.as_bytes());
    buf.resize(start + size, b' ');
}
fn float(buf: &mut Vec<u8>, value: f32) {
    buf.extend_from_slice(&value.to_be_bytes());
}

fn met(
    path: &Path,
    gaussian: bool,
    model_level: bool,
    humidity: bool,
    time: &str,
    phase: f32,
    south_first: bool,
    zero_wind: bool,
    version: i32,
    height_ties: u8,
) {
    let (nx, ny) = (12, 8);
    let mut file = fs::File::create(path).unwrap();
    // Permuted levels make every column go through the native sort.
    for k in [4, 0, 7, 2, 5, 1, 6, 3] {
        // Native mpas_sort.F:346-408 randomizes equal-coordinate order.
        // Keep GHT bit-identical for levels 2/3; mode 1 also gives them
        // identical sampled values, mode 2 keeps distinct values at that height.
        let coordinate_k = if height_ties != 0 && k == 3 { 2 } else { k };
        let value_k = if height_ties == 1 { coordinate_k } else { k };
        let height = [0.0, 300.0, 1100.0, 2800.0, 5000.0, 7500.0, 10000.0, 14000.0][coordinate_k];
        let level = if k == 0 {
            200100.0
        } else if model_level {
            k as f32
        } else {
            [
                200100.0, 97000.0, 90000.0, 75000.0, 60000.0, 45000.0, 30000.0, 16000.0,
            ][k]
        };
        for name in [
            "TT", "UU", "VV", "GHT", "RH", "SPECHUMD", "PRESSURE", "PSFC", "SOILHGT",
        ] {
            if name == "SPECHUMD" && !humidity {
                continue;
            }
            if name == "PRESSURE" && !model_level {
                continue;
            }
            record(&mut file, &version.to_be_bytes());
            let mut header = Vec::new();
            text(&mut header, time, 24);
            float(&mut header, 0.0);
            if version != 3 {
                text(&mut header, "synthetic", 32);
            }
            let field_name = match (name, version) {
                ("GHT", 4) => "HGT",
                ("PRESSURE", 3) => "PRES",
                _ => name,
            };
            text(&mut header, field_name, 9);
            text(&mut header, "test", 25);
            text(&mut header, "case-9 branch replay", 46);
            float(&mut header, level);
            let projection_code = if name == "PSFC" {
                [1, 3, 5][k % 3]
            } else if gaussian {
                4
            } else {
                0
            };
            for i in [nx as i32, ny as i32, projection_code] {
                header.extend_from_slice(&i.to_be_bytes());
            }
            record(&mut file, &header);
            let mut proj = Vec::new();
            if version != 3 {
                text(&mut proj, "SWCORNER", 8);
            }
            float(
                &mut proj,
                if gaussian {
                    gaussian_latitudes(ny)[0] * if south_first { -1.0 } else { 1.0 }
                } else {
                    -70.0
                },
            );
            float(&mut proj, -180.0);
            float(&mut proj, if gaussian { ny as f32 / 2.0 } else { 20.0 });
            float(&mut proj, 30.0);
            let extra = match projection_code {
                1 => 1,
                3 => 3,
                5 => 2,
                _ => 0,
            };
            for v in [0.0, 30.0, 60.0].into_iter().take(extra) {
                float(&mut proj, v);
            }
            if version == 5 {
                float(&mut proj, 6371.229);
            }
            record(&mut file, &proj);
            if version == 5 {
                record(&mut file, &0i32.to_be_bytes());
            }
            let mut values = Vec::new();
            for j in 0..ny {
                for i in 0..nx {
                    let perturb = (((i * 13 + j * 7 + value_k * 3) % 17) as f32 - 8.0) * 0.025 + phase;
                    let value = match name {
                        "TT" => 290.0 - height * 0.006 + perturb,
                        "UU" => {
                            if zero_wind || (i + j) % 7 == 0 {
                                0.0
                            } else {
                                11.0 - height * 0.0004 + perturb * 4.0
                            }
                        }
                        "VV" => {
                            if zero_wind {
                                -0.0
                            } else {
                                -3.0 + height * 0.0002 + perturb * 2.0
                            }
                        }
                        "GHT" => height + ((((i * 13 + j * 7 + coordinate_k * 3) % 17)
                            as f32 - 8.0) * 0.025 + phase) * 8.0,
                        "RH" => 52.0 + perturb * 9.0,
                        "SPECHUMD" => {
                            if k == 7 {
                                -0.001
                            } else {
                                0.012 * (-height / 5500.0).exp() + perturb * 0.0001
                            }
                        }
                        "PRESSURE" => 97000.0 * (-height / 7800.0).exp() + perturb * 30.0,
                        _ => 101000.0,
                    };
                    float(&mut values, value);
                }
            }
            record(&mut file, &values);
        }
    }
}

fn run(command: &mut Command) -> std::process::Output {
    let output = command.output().expect("run artifact");
    assert!(
        output.status.success(),
        "{command:?}\n{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    output
}

fn file_hash(path: &Path) -> String {
    let mut input = fs::File::open(path).unwrap();
    let mut hash = Sha256::new();
    let mut buf = [0u8; 65536];
    loop {
        let n = input.read(&mut buf).unwrap();
        if n == 0 {
            break;
        }
        hash.update(&buf[..n]);
    }
    format!("{:x}", hash.finalize())
}

#[test]
#[ignore = "requires published real MET/grid inputs, strict Fortran oracle and hex CLI"]
fn published_real_columns_through_hex_door_are_word_identical_to_fortran() {
    let oracle = std::env::var("MPAS_LBC_ORACLE").expect("MPAS_LBC_ORACLE");
    let root = std::path::PathBuf::from(
        std::env::var("MPAS_LBC_REPLAY_DIR").expect("MPAS_LBC_REPLAY_DIR"),
    );
    let met = std::path::PathBuf::from(
        std::env::var("MPAS_LBC_REAL_MET_FILE").expect("MPAS_LBC_REAL_MET_FILE"),
    );
    let grid = std::path::PathBuf::from(
        std::env::var("MPAS_LBC_REAL_GRID").expect("MPAS_LBC_REAL_GRID"),
    );
    let stamp = std::env::var("MPAS_LBC_REAL_TIME").expect("MPAS_LBC_REAL_TIME");
    let stop = std::env::var("MPAS_LBC_REAL_STOP_TIME").expect("MPAS_LBC_REAL_STOP_TIME");
    let end_met = std::path::PathBuf::from(
        std::env::var("MPAS_LBC_REAL_END_MET_FILE").expect("MPAS_LBC_REAL_END_MET_FILE"),
    );
    let nf = std::env::var("MPAS_LBC_REAL_NFGLEVELS").expect("MPAS_LBC_REAL_NFGLEVELS");
    assert!(rw_mpas::lbc::seconds_between(&stamp, &stop).unwrap() > 0);
    let met_hash = file_hash(&met);
    let end_met_hash = file_hash(&end_met);
    let grid_hash = file_hash(&grid);
    assert_eq!(met_hash, std::env::var("MPAS_LBC_REAL_MET_SHA256").unwrap());
    assert_eq!(end_met_hash, std::env::var("MPAS_LBC_REAL_END_MET_SHA256").unwrap());
    assert_eq!(grid_hash, std::env::var("MPAS_LBC_REAL_GRID_SHA256").unwrap());
    fs::create_dir_all(root.join("met")).unwrap();
    // The published input is immutable; only this lane-owned name is created.
    let link = root.join("met").join(format!("MET:{}", &stamp[..13]));
    assert!(!link.exists(), "use a fresh replay directory");
    #[cfg(unix)]
    std::os::unix::fs::symlink(&met, &link).unwrap();
    #[cfg(not(unix))]
    fs::copy(&met, &link).unwrap();
    let end_link = root.join("met").join(format!("MET:{}", &stop[..13]));
    assert!(!end_link.exists(), "use a fresh replay directory");
    #[cfg(unix)]
    std::os::unix::fs::symlink(&end_met, &end_link).unwrap();
    #[cfg(not(unix))]
    fs::copy(&end_met, &end_link).unwrap();
    let provenance = serde_json::json!({
        "rust_binary_sha256": file_hash(Path::new(env!("CARGO_BIN_EXE_rw_mpas_lbc"))),
        "oracle_sha256": file_hash(Path::new(&oracle)),
        "source_revision": env!("GPUWM_BRIDGE_SOURCE_REV"),
        "optimized": !cfg!(debug_assertions),
        "host": String::from_utf8(run(Command::new("hostname").arg("-f")).stdout).unwrap().trim(),
        "met_file": met, "met_sha256": met_hash,
        "end_met_file": end_met, "end_met_sha256": end_met_hash,
        "grid_file": grid, "grid_sha256": grid_hash, "nfglevels": nf,
    });
    let mut receipts = Vec::new();
    for (index, (use_sh, order, extrap)) in
        [("yes", "3", "lapse-rate"), ("no", "2", "linear")].into_iter().enumerate()
    {
        let dir = root.join(format!("option-{index}"));
        fs::create_dir_all(&dir).unwrap();
        let mut door = Command::new(
            std::env::var("MPAS_LBC_PYTHON").unwrap_or_else(|_| "python3".into()),
        );
        door.args(["-m", "hexcore.cli", "lbc", "--grid"])
            .arg(&grid)
            .arg("--met-dir").arg(root.join("met"))
            .arg("--out-dir").arg(dir.join("ours"))
            .args(["--start-time", &stamp, "--stop-time", &stop,
                "--nfglevels", &nf, "--extrap-airtemp", extrap,
                "--use-spechumd", use_sh, "--theta-adv-order", order,
                "--coef-3rd-order", "0.25", "--oned-underflow", "preserve",
                "--lbc-exe", env!("CARGO_BIN_EXE_rw_mpas_lbc")]);
        run(&mut door);
        for time in [&stamp, &stop] {
        let native = dir.join(format!("native-{}.nc", time.replace(':', ".")));
        run(Command::new(&oracle).arg(&grid).arg(&native).arg(time)
            .arg(root.join("met/MET")).args([extrap, use_sh, order, &nf]));
        let ours = dir.join("ours").join(rw_mpas::lbc::emit::lbc_file_name(time));
        let comparison = compare_lbc(&ours, &native).unwrap();
        assert_eq!(comparison.variables.len(), 7);
        let receipt = dir.join(format!("comparison-{}.json", time.replace(':', ".")));
        fs::write(&receipt, serde_json::to_string_pretty(&comparison).unwrap()).unwrap();
        receipts.push(serde_json::json!({"time":time,"use_spechumd":use_sh,
            "theta_adv_order":order,"extrapolation":extrap,"comparison":receipt,
            "payload_byte_exact":comparison.payload_byte_exact}));
        fs::write(root.join("replay.json"), serde_json::to_string_pretty(
            &serde_json::json!({"tier":"real published columns, full case-9 numerical replay; not regional f0 native deck proof",
                "provenance":provenance,"cases":receipts})).unwrap()).unwrap();
        assert!(comparison.payload_byte_exact, "real option {index}: {receipt:?}");
        fs::remove_file(native).unwrap();
        fs::remove_file(ours).unwrap();
        }
    }
    fs::remove_file(link).unwrap();
    fs::remove_file(end_link).unwrap();
}

#[test]
#[ignore = "requires strict pinned Fortran oracle, hex CLI and lane-owned replay directory"]
fn entire_case9_through_hex_door_is_word_identical_to_fortran() {
    let oracle = std::env::var("MPAS_LBC_ORACLE").expect("MPAS_LBC_ORACLE");
    let root = std::path::PathBuf::from(
        std::env::var("MPAS_LBC_REPLAY_DIR").expect("MPAS_LBC_REPLAY_DIR"),
    );
    fs::create_dir_all(&root).unwrap();
    let provenance = serde_json::json!({
        "rust_binary_sha256": format!("{:x}", Sha256::digest(fs::read(env!("CARGO_BIN_EXE_rw_mpas_lbc")).unwrap())),
        "oracle_sha256": format!("{:x}", Sha256::digest(fs::read(&oracle).unwrap())),
        "source_revision": env!("GPUWM_BRIDGE_SOURCE_REV"),
        "optimized": !cfg!(debug_assertions),
        "host": String::from_utf8(run(Command::new("hostname").arg("-f")).stdout).unwrap().trim(),
    });
    let mut receipts = Vec::new();
    for (
        index,
        (
            gaussian,
            model_level,
            humidity,
            use_sh,
            order,
            extrap,
            above_top,
            south_first,
            zero_wind,
            version,
        ),
    ) in [
        (
            false,
            false,
            true,
            "yes",
            3,
            "lapse-rate",
            false,
            false,
            false,
            5,
        ),
        (
            false, true, true, "yes", 2, "linear", false, false, false, 5,
        ),
        (
            false, false, false, "yes", 3, "constant", false, false, false, 5,
        ),
        (
            false, true, true, "no", 3, "constant", true, false, false, 5,
        ),
        (
            true,
            false,
            true,
            "yes",
            3,
            "lapse-rate",
            false,
            false,
            false,
            5,
        ),
        (true, true, true, "no", 2, "linear", false, false, false, 5),
        (
            true, false, false, "yes", 3, "constant", true, false, false, 5,
        ),
        (true, true, true, "yes", 3, "linear", false, true, false, 5),
        (
            true, false, true, "yes", 3, "constant", false, false, true, 5,
        ),
        (
            false, true, true, "yes", 3, "linear", false, false, false, 3,
        ),
        (
            false,
            false,
            false,
            "no",
            2,
            "lapse-rate",
            false,
            false,
            false,
            4,
        ),
    ]
    .into_iter()
    .enumerate()
    {
        let dir = root.join(format!("branch-{index}"));
        fs::create_dir_all(dir.join("met")).unwrap();
        let grid_path = dir.join("grid.nc");
        grid(&grid_path, above_top);
        for (time, phase) in [(START, 0.0), (STOP, 0.1375)] {
            met(
                &dir.join("met").join(format!("MET:{}", &time[..13])),
                gaussian,
                model_level,
                humidity,
                time,
                phase,
                south_first,
                zero_wind,
                version,
                0,
            );
        }
        let mut door =
            Command::new(std::env::var("MPAS_LBC_PYTHON").unwrap_or_else(|_| "python3".into()));
        door.args(["-m", "hexcore.cli", "lbc", "--grid"])
            .arg(&grid_path)
            .arg("--met-dir")
            .arg(dir.join("met"))
            .arg("--out-dir")
            .arg(dir.join("ours"))
            .args([
                "--start-time",
                START,
                "--stop-time",
                STOP,
                "--nfglevels",
                "8",
                "--extrap-airtemp",
                extrap,
                "--use-spechumd",
                use_sh,
                "--theta-adv-order",
                &order.to_string(),
                "--coef-3rd-order",
                "0.25",
                "--oned-underflow",
                "preserve",
                "--lbc-exe",
                env!("CARGO_BIN_EXE_rw_mpas_lbc"),
            ]);
        run(&mut door);
        for time in [START, STOP] {
            let native = dir.join(format!("native-{}.nc", time.replace(':', ".")));
            let mut reference = Command::new(&oracle);
            reference
                .arg(&grid_path)
                .arg(&native)
                .arg(time)
                .arg(dir.join("met/MET"))
                .args([extrap, use_sh, &order.to_string()]);
            run(&mut reference);
            let ours = dir
                .join("ours")
                .join(rw_mpas::lbc::emit::lbc_file_name(time));
            let comparison = compare_lbc(&ours, &native).unwrap();
            assert_eq!(
                comparison.variables.len(),
                7,
                "complete numerical LBC field set"
            );
            let receipt = dir.join(format!("comparison-{}.json", time.replace(':', ".")));
            fs::write(&receipt, serde_json::to_string_pretty(&comparison).unwrap()).unwrap();
            receipts.push(serde_json::json!({"branch":index,"gaussian":gaussian,"model_level":model_level,
                "specific_humidity_present":humidity,"use_spechumd":use_sh,"theta_adv_order":order,
                "extrapolation":extrap,"above_top":above_top,"south_first":south_first,"zero_wind":zero_wind,"intermediate_version":version,"time":time,"comparison":receipt,
                "payload_byte_exact":comparison.payload_byte_exact}));
            fs::write(root.join("replay.json"),serde_json::to_string_pretty(&serde_json::json!({
                "tier":"synthetic full case-9 numerical replay; not f0 native deck proof", "provenance":provenance,"cases":receipts})).unwrap()).unwrap();
            assert!(comparison.payload_byte_exact, "branch {index}: {receipt:?}");
            fs::remove_file(native).unwrap();
            fs::remove_file(ours).unwrap();
        }
        fs::remove_file(grid_path).unwrap();
        for entry in fs::read_dir(dir.join("met")).unwrap() {
            fs::remove_file(entry.unwrap().path()).unwrap();
        }
    }
}


/// Whole-case diagnostic for randomized native sorting; differences are
/// measured and retained, never converted into an acceptance tolerance.
#[test]
#[ignore = "requires strict native full-case oracle and public hex door"]
fn equal_height_columns_measure_native_repeatability_through_hex_door() {
    let oracle = std::env::var("MPAS_LBC_ORACLE").expect("MPAS_LBC_ORACLE");
    let root = std::path::PathBuf::from(
        std::env::var("MPAS_LBC_REPLAY_DIR").expect("MPAS_LBC_REPLAY_DIR"),
    ).join("equal-heights");
    fs::create_dir_all(&root).unwrap();
    let provenance = serde_json::json!({
        "rust_binary_sha256": file_hash(Path::new(env!("CARGO_BIN_EXE_rw_mpas_lbc"))),
        "oracle_sha256": file_hash(Path::new(&oracle)),
        "source_revision": env!("GPUWM_BRIDGE_SOURCE_REV"),
        "optimized": !cfg!(debug_assertions), "process_count": 1,
        "host": String::from_utf8(run(Command::new("hostname").arg("-f")).stdout).unwrap().trim(),
        "native_source": "mpas_init_atm_cases.F:5434-6218; mpas_sort.F:346-408",
        "native_repeats": 3,
    });
    let mut receipts = Vec::new();
    for (branch, (gaussian, model_level, use_sh, order, extrap, south_first, version)) in [
        (false, false, "yes", 3, "lapse-rate", false, 5),
        (false, true, "yes", 2, "linear", false, 5),
        (true, false, "no", 3, "constant", false, 5),
        (true, true, "yes", 3, "linear", true, 5),
        (false, true, "yes", 3, "linear", false, 3),
        (false, false, "no", 2, "lapse-rate", false, 4),
    ].into_iter().enumerate() {
        for ties in [1u8, 2] {
            let dir = root.join(format!("branch-{branch}-ties-{ties}"));
            fs::create_dir_all(dir.join("met")).unwrap();
            let grid_path = dir.join("grid.nc");
            grid(&grid_path, false);
            for (time, phase) in [(START, 0.0), (STOP, 0.1375)] {
                met(&dir.join("met").join(format!("MET:{}", &time[..13])),
                    gaussian, model_level, true, time, phase, south_first, false, version, ties);
            }
            let mut door = Command::new(
                std::env::var("MPAS_LBC_PYTHON").unwrap_or_else(|_| "python3".into()));
            door.args(["-m", "hexcore.cli", "lbc", "--grid"]).arg(&grid_path)
                .arg("--met-dir").arg(dir.join("met"))
                .arg("--out-dir").arg(dir.join("ours"))
                .args(["--start-time", START, "--stop-time", STOP,
                    "--nfglevels", "8", "--extrap-airtemp", extrap,
                    "--use-spechumd", use_sh, "--theta-adv-order", &order.to_string(),
                    "--coef-3rd-order", "0.25", "--oned-underflow", "preserve",
                    "--lbc-exe", env!("CARGO_BIN_EXE_rw_mpas_lbc")]);
            run(&mut door);
            for time in [START, STOP] {
                let ours = dir.join("ours").join(rw_mpas::lbc::emit::lbc_file_name(time));
                let mut native_paths = Vec::new();
                for repeat in 0..3 {
                    let native = dir.join(format!("native-{repeat}-{}.nc", time.replace(':', ".")));
                    run(Command::new(&oracle).arg(&grid_path).arg(&native).arg(time)
                        .arg(dir.join("met/MET")).args([extrap, use_sh, &order.to_string()]));
                    native_paths.push(native);
                }
                let mut comparisons = Vec::new();
                for (repeat, native) in native_paths.iter().enumerate() {
                    let comparison = compare_lbc(&ours, native).unwrap();
                    // This numerical oracle omits production clock/header
                    // metadata and declares dimensions in a different order.
                    // compare_lbc rejects any field shape or dtype mismatch;
                    // retain its container differences rather than claiming
                    // full production-file identity from this diagnostic.
                    assert_eq!(comparison.variables.len(), 7);
                    let reference_repeat = compare_lbc(native, &native_paths[0]).unwrap();
                    assert!(reference_repeat.dims_equal);
                    comparisons.push(serde_json::json!({"repeat":repeat,
                        "native_sha256":file_hash(native), "rust_vs_native":comparison,
                        "native_vs_repeat_zero":reference_repeat}));
                }
                receipts.push(serde_json::json!({"branch":branch, "height_ties":ties,
                    "values_at_tied_height":if ties == 1 {"identical"} else {"distinct"},
                    "gaussian":gaussian,"model_level":model_level,"use_spechumd":use_sh,
                    "theta_adv_order":order,"extrapolation":extrap,"south_first":south_first,
                    "intermediate_version":version,"time":time,"grid_sha256":file_hash(&grid_path),
                    "met_sha256":file_hash(&dir.join("met").join(format!("MET:{}", &time[..13]))),
                    "rust_sha256":file_hash(&ours),"comparisons":comparisons}));
                fs::write(root.join("repeatability.json"), serde_json::to_string_pretty(
                    &serde_json::json!({"tier":"synthetic full-case randomized-tie diagnostic; not f0 native proof or exclusion witness",
                        "provenance":provenance,"cases":receipts})).unwrap()).unwrap();
            }
        }
    }
    assert_eq!(receipts.len(), 24, "all full-case configurations and valid times completed");
}
