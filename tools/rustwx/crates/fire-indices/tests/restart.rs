//! Exercise the actual executable across an hourly-file boundary and reject
//! state/member/time mismatches. These are contract tests, not forecast proof.
use netcdf_writer::{AttrValue, NcFormat, NcType, NcWriter, Schema, VarData};
use serde_json::{Value, json};
use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};
struct Scratch(PathBuf);
impl Drop for Scratch {
    fn drop(&mut self) {
        use std::io::Write;
        let mut log = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(
                self.0
                    .parent()
                    .unwrap()
                    .join("fire-indices-test-deleted-files.log"),
            )
            .unwrap();
        for e in fs::read_dir(&self.0).unwrap() {
            let e = e.unwrap();
            writeln!(
                log,
                "{} {}",
                e.path().display(),
                e.metadata().unwrap().len()
            )
            .unwrap();
        }
        fs::remove_dir_all(&self.0).unwrap();
    }
}
fn input(p: &Path, start: i64, hours: usize) {
    input_with_night(p, start, hours, -0.25);
}
fn input_with_night(p: &Path, start: i64, hours: usize, night: f64) {
    let mut s = Schema::new(NcFormat::Offset64);
    let t = s.def_dim("time", 0, true).unwrap();
    let y = s.def_dim("latitude", 32, false).unwrap();
    let x = s.def_dim("longitude", 32, false).unwrap();
    let tv = s.def_var("time", NcType::Double, &[t]).unwrap();
    s.put_var_attr(
        tv,
        "units",
        AttrValue::Text("seconds since 1970-01-01 00:00:00".into()),
    )
    .unwrap();
    let lat = s.def_var("latitude", NcType::Double, &[y]).unwrap();
    let lon = s.def_var("longitude", NcType::Double, &[x]).unwrap();
    let mut vars = vec![];
    for (n, u) in [
        ("T2", "K"),
        ("D2", "K"),
        ("U10", "m s-1"),
        ("V10", "m s-1"),
        ("TP", "m"),
        ("SSRD", "J m-2"),
    ] {
        let id = s.def_var(n, NcType::Double, &[t, y, x]).unwrap();
        s.put_var_attr(id, "units", AttrValue::Text(u.into()))
            .unwrap();
        vars.push(id);
    }
    let mut w = NcWriter::create(p, s).unwrap();
    w.write_var(
        lat,
        VarData::F64(&(0..32).map(|i| 49. + i as f64 / 4.).collect::<Vec<_>>()),
    )
    .unwrap();
    w.write_var(
        lon,
        VarData::F64(&(0..32).map(|i| -118. + i as f64 / 4.).collect::<Vec<_>>()),
    )
    .unwrap();
    for k in 0..hours {
        let stamp = start + k as i64 * 3600;
        w.write_record(k as u64, tv, VarData::F64(&[stamp as f64]))
            .unwrap();
        for (j, id) in vars.iter().enumerate() {
            let a: Vec<_> = (0..1024)
                .map(|i| match j {
                    0 => 290. + (i % 32) as f64 / 10. + ((stamp / 3600) % 24) as f64 / 5.,
                    1 => 280.,
                    2 => 4.,
                    3 => -2.,
                    5 => {
                        if (stamp / 3600) % 24 < 8 {
                            night
                        } else {
                            3600. * 400.
                        }
                    }
                    _ => {
                        if (stamp / 3600) % 29 == 0 {
                            0.002
                        } else {
                            0.
                        }
                    }
                })
                .collect();
            w.write_record(k as u64, *id, VarData::F64(&a)).unwrap();
        }
    }
    w.finish().unwrap();
}
fn execute(
    root: &Path,
    name: &str,
    inputs: &[PathBuf],
    state: Option<PathBuf>,
    member: &str,
) -> (bool, Value, String) {
    let req = json!({"inputs":inputs,"state_in":state,"state_out":root.join(format!("{name}.state.json")),"output":root.join(format!("{name}.nc")),"receipt":root.join(format!("{name}.receipt.json")),"member":member,"compute_backend":"cpu","utc_offset_hours":-7,"annual_rain_mm":600.,"annual_rain_source":"contract-test climatology","initial_kbdi":0.});
    let path = root.join(format!("{name}.request.json"));
    fs::write(&path, serde_json::to_vec(&req).unwrap()).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_rw_fire_indices"))
        .arg(&path)
        .env("RAYON_NUM_THREADS", "4")
        .output()
        .unwrap();
    (
        out.status.success(),
        req,
        String::from_utf8_lossy(&out.stderr).into(),
    )
}
#[test]
fn executable_restart_and_invalid_chain() {
    let root = std::env::var_os("FIRE_INDICES_TEST_SCRATCH")
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir)
        .join(format!("fire-indices-{}", std::process::id()));
    fs::create_dir(&root).unwrap();
    let _owned = Scratch(root.clone());
    // 2023-05-01 20Z, first interval after noon MST.
    let start = 1682971200;
    let a = root.join("first.nc");
    let b = root.join("second.nc");
    input(&a, start, 31);
    input(&b, start + 31 * 3600, 41);
    let (ok, full, error) = execute(&root, "whole", &[a.clone(), b.clone()], None, "control");
    assert!(ok, "{error}");
    let (ok, first, error) = execute(&root, "part1", &[a.clone()], None, "control");
    assert!(ok, "{error}");
    let (ok, last, error) = execute(
        &root,
        "part2",
        &[b.clone()],
        Some(first["state_out"].as_str().unwrap().into()),
        "control",
    );
    assert!(ok, "{error}");
    let load = |v: &Value| -> Value {
        serde_json::from_slice(&fs::read(v["state_out"].as_str().unwrap()).unwrap()).unwrap()
    };
    let x = load(&full);
    let y = load(&last);
    assert!(
        x["cells"] == y["cells"],
        "carried cell state differs across restart"
    );
    assert_eq!(x["last_hour"], y["last_hour"]);
    assert_eq!(x["completed_days"], 3);
    // Real ERA5 stores have sub-joule negative SSRD at night. It must give
    // exactly the same physical state as zero, while material negatives fail.
    let zero = root.join("zero-solar-forcing.nc");
    let noise = root.join("noise-solar-forcing.nc");
    let negative = root.join("invalid-solar-forcing.nc");
    input_with_night(&zero, start, 24, 0.);
    input_with_night(&noise, start, 24, -0.25);
    input_with_night(&negative, start, 24, -1.);
    let (ok, z, error) = execute(&root, "zero-solar", &[zero], None, "control");
    assert!(ok, "{error}");
    let (ok, q, error) = execute(&root, "noise-solar", &[noise], None, "control");
    assert!(ok, "{error}");
    assert_eq!(load(&z)["cells"], load(&q)["cells"]);
    let (ok, _, error) = execute(&root, "invalid-solar", &[negative], None, "control");
    assert!(!ok && error.contains("SSRD=-1"), "{error}");
    assert!(!root.join("invalid-solar.nc.partial").exists());

    let (ok, _, error) = execute(
        &root,
        "bad-member",
        &[b.clone()],
        Some(first["state_out"].as_str().unwrap().into()),
        "another",
    );
    assert!(!ok && error.contains("member mismatch"), "{error}");
    let (ok, _, error) = execute(
        &root,
        "duplicate",
        &[a.clone()],
        Some(first["state_out"].as_str().unwrap().into()),
        "control",
    );
    assert!(!ok && error.contains("gap/overlap"), "{error}");
    let gap = root.join("gap-forcing.nc");
    input(&gap, start + 32 * 3600, 40);
    let (ok, _, error) = execute(
        &root,
        "gap",
        &[gap],
        Some(first["state_out"].as_str().unwrap().into()),
        "control",
    );
    assert!(!ok && error.contains("gap/overlap"), "{error}");
    assert!(!root.join("gap.state.json").exists());
    assert!(!root.join("gap.nc.partial").exists());
    // The real archive reuses short IDs at different station sites. Compare both
    // sites once while dropping an exact duplicate source record.
    let csv=b"rep_date,aes,name,lon,lat,ffmc,dmc,dc,isi,bui,fwi,opts,calcstatus\n2023-05-02 12:00:00,L3,Site A,-118,49,85,6,15,2,6,3,M=1,1\n2023-05-02 12:00:00,L3,Site B,-118,50,85,6,15,2,6,3,M=1,1\n2023-05-02 12:00:00,L3,Site B,-118,50,85,6,15,2,6,3,M=1,1\n2023-05-02 12:00:00,L4,Ambiguous,-118,51,85,6,15,2,6,3,M=1,1\n2023-05-02 12:00:00,L4,Ambiguous,-118,51,85,6,15,2,6,4,M=1,1\n";
    let mut compressed = Vec::new();
    {
        use std::io::Write;
        let mut encoder = bzip2::write::BzEncoder::new(&mut compressed, bzip2::Compression::best());
        encoder.write_all(csv).unwrap();
        encoder.finish().unwrap();
    }
    let archive = root.join("stations.csv.bz2");
    fs::write(&archive, compressed).unwrap();
    let verify = json!({"archive":archive,"forecast":[full["output"]],"rows":root.join("pairs.csv"),"receipt":root.join("verification.json"),"start_date":"2023-05-02","end_date":"2023-05-02","utc_offset_hours":-7,"max_distance_km":30.,"research_station_license":"OGL-Canada-2.0"});
    fs::write(
        root.join("verify.request.json"),
        serde_json::to_vec(&verify).unwrap(),
    )
    .unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_rw_fire_indices"))
        .arg("verify")
        .arg(root.join("verify.request.json"))
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let verification: Value =
        serde_json::from_slice(&fs::read(root.join("verification.json")).unwrap()).unwrap();
    assert_eq!(verification["ambiguous_station_days_excluded"], 1);
    assert_eq!(verification["stations"], 2);
    assert_eq!(verification["station_days"], 2);
    assert_eq!(verification["identical_archive_duplicates"], 1);
    assert_eq!(verification["metrics"]["fwi_fwi"]["n"], 2);
    // Same request and bytes are reusable; changing any persisted bytes refuses reuse.
    let receipt_before = fs::read(root.join("whole.receipt.json")).unwrap();
    let replay = Command::new(env!("CARGO_BIN_EXE_rw_fire_indices"))
        .arg(root.join("whole.request.json"))
        .env("RAYON_NUM_THREADS", "4")
        .output()
        .unwrap();
    assert!(
        replay.status.success(),
        "{}",
        String::from_utf8_lossy(&replay.stderr)
    );
    assert_eq!(
        receipt_before,
        fs::read(root.join("whole.receipt.json")).unwrap()
    );
    let mut req: Value =
        serde_json::from_slice(&fs::read(root.join("part2.request.json")).unwrap()).unwrap();
    req["member"] = json!("perturbed-1");
    req["fork_member"] = json!("control");
    for (key, ending) in [
        ("output", "nc"),
        ("state_out", "state.json"),
        ("receipt", "receipt.json"),
    ] {
        req[key] = json!(root.join(format!("fork.{ending}")));
    }
    fs::write(
        root.join("fork.request.json"),
        serde_json::to_vec(&req).unwrap(),
    )
    .unwrap();
    let fork = Command::new(env!("CARGO_BIN_EXE_rw_fire_indices"))
        .arg(root.join("fork.request.json"))
        .env("RAYON_NUM_THREADS", "4")
        .output()
        .unwrap();
    assert!(
        fork.status.success(),
        "{}",
        String::from_utf8_lossy(&fork.stderr)
    );
    let fork_state: Value =
        serde_json::from_slice(&fs::read(root.join("fork.state.json")).unwrap()).unwrap();
    assert!(fork_state["cells"] == x["cells"]);
    assert_eq!(fork_state["member"], "perturbed-1");
    fs::write(full["state_out"].as_str().unwrap(), b"corrupted").unwrap();
    let replay = Command::new(env!("CARGO_BIN_EXE_rw_fire_indices"))
        .arg(root.join("whole.request.json"))
        .output()
        .unwrap();
    assert!(!replay.status.success());
}
