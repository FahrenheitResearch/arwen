//! `rw_mrms frames` on archived-object fixtures written here: small
//! MergedReflectivityQCComposite GRIB2 messages (IEEE packing, the archive's
//! north-to-south row order) under the archive's own file names, with
//! off-cadence seconds.  The binary is run as a user runs it, offline
//! through `--files`.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

const NX: usize = 6;
const NY: usize = 4;
const LAT_NORTH: f64 = 36.0;
const LON_WEST: f64 = -100.0;
const STEP_DEG: f64 = 0.01;

/// Row 0 is the northern row, as the archive writes it.
fn archive_values() -> Vec<f32> {
    let mut v = vec![-99.0f32; NX * NY];
    v[0] = -999.0; // north-west corner: no coverage
    v[2] = 30.5;
    v[3] = 45.0;
    v[NX + 1] = 12.25;
    v[(NY - 1) * NX + NX - 1] = -999.0; // south-east corner
    v[2 * NX + 4] = 61.0;
    v
}

fn section(number: u8, length: usize) -> Vec<u8> {
    let mut bytes = vec![0; length];
    bytes[..4].copy_from_slice(&(length as u32).to_be_bytes());
    bytes[4] = number;
    bytes
}

/// One GRIB2 message: discipline 209, category 10, number 0 (the composite),
/// level type 102 at 500 m, template 3.0 lat/lon grid, template 5.4 IEEE.
fn composite_grib(values: &[f32]) -> Vec<u8> {
    let count = values.len() as u32;
    let micro = |deg: f64| (deg * 1.0e6).round() as i64;
    let lon360 = |deg: f64| if deg < 0.0 { deg + 360.0 } else { deg };
    let mut s1 = section(1, 21);
    s1[12..14].copy_from_slice(&2026u16.to_be_bytes());
    s1[14] = 10;
    s1[15] = 1;
    let mut s3 = section(3, 72);
    s3[6..10].copy_from_slice(&count.to_be_bytes());
    s3[14] = 6;
    s3[30..34].copy_from_slice(&(NX as u32).to_be_bytes());
    s3[34..38].copy_from_slice(&(NY as u32).to_be_bytes());
    s3[46..50].copy_from_slice(&(micro(LAT_NORTH) as u32).to_be_bytes());
    s3[50..54].copy_from_slice(&(micro(lon360(LON_WEST)) as u32).to_be_bytes());
    s3[54] = 0x30;
    let lat_south = LAT_NORTH - (NY - 1) as f64 * STEP_DEG;
    let lon_east = LON_WEST + (NX - 1) as f64 * STEP_DEG;
    s3[55..59].copy_from_slice(&(micro(lat_south) as u32).to_be_bytes());
    s3[59..63].copy_from_slice(&(micro(lon360(lon_east)) as u32).to_be_bytes());
    s3[63..67].copy_from_slice(&(micro(STEP_DEG) as u32).to_be_bytes());
    s3[67..71].copy_from_slice(&(micro(STEP_DEG) as u32).to_be_bytes());
    s3[71] = 0x00; // +i east, -j south: the archive's order
    let mut s4 = section(4, 34);
    s4[9] = 10;
    s4[10] = 0;
    s4[17] = 1;
    s4[22] = 102;
    s4[24..28].copy_from_slice(&500u32.to_be_bytes());
    s4[28..34].fill(255);
    let mut s5 = section(5, 12);
    s5[5..9].copy_from_slice(&count.to_be_bytes());
    s5[9..11].copy_from_slice(&4u16.to_be_bytes());
    s5[11] = 1;
    let mut s6 = section(6, 6);
    s6[5] = 255;
    let mut s7 = section(7, 5 + values.len() * 4);
    for (slot, value) in s7[5..].chunks_exact_mut(4).zip(values) {
        slot.copy_from_slice(&value.to_be_bytes());
    }
    let mut bytes = vec![0; 16];
    bytes[..4].copy_from_slice(b"GRIB");
    bytes[6] = 209;
    bytes[7] = 2;
    for part in [s1, s3, s4, s5, s6, s7] {
        bytes.extend(part);
    }
    bytes.extend_from_slice(b"7777");
    let length = bytes.len() as u64;
    bytes[8..16].copy_from_slice(&length.to_be_bytes());
    bytes
}

fn scratch(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("mrms_frames").join(name);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("archive")).unwrap();
    dir
}

/// Archive objects at 18:10:37 (gzipped), 18:13:30 (a decoy, further from
/// 18:10) and 18:19:02.
fn write_archive(dir: &Path) {
    let bytes = composite_grib(&archive_values());
    let name = |stamp: &str| format!("MRMS_MergedReflectivityQCComposite_00.50_20261001-{stamp}.grib2");
    let mut gz = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
    gz.write_all(&bytes).unwrap();
    std::fs::write(dir.join("archive").join(format!("{}.gz", name("181037"))), gz.finish().unwrap()).unwrap();
    std::fs::write(dir.join("archive").join(name("181330")), &bytes).unwrap();
    std::fs::write(dir.join("archive").join(name("181902")), &bytes).unwrap();
    std::fs::write(dir.join("archive").join("README.txt"), b"not a frame").unwrap();
}

#[test]
fn batch_matches_scalar_packs_for_every_truth_product_and_geometry() {
    let dir = scratch("decode-batch-identity");
    let bin = env!("CARGO_BIN_EXE_rw_mrms");
    let products = [
        ("MergedReflectivityQCComposite_00.50", 10u8, 0u8, 20.0f32),
        ("PrecipRate_00.00", 6, 1, 2.0),
        ("RadarQualityIndex_00.00", 8, 0, 0.8),
    ];
    let mut jobs: Vec<Vec<String>> = Vec::new();
    let mut outputs = Vec::new();
    for (index, (product, category, parameter, value)) in products.iter().enumerate() {
        let mut raw = composite_grib(&vec![*value; NX * NY]);
        let mut offset = 16;
        while offset < raw.len() - 4 {
            let length = u32::from_be_bytes(raw[offset..offset + 4].try_into().unwrap()) as usize;
            if raw[offset + 4] == 4 {
                raw[offset + 9] = *category;
                raw[offset + 10] = *parameter;
                if *category != 10 {
                    raw[offset + 24..offset + 28].copy_from_slice(&0u32.to_be_bytes());
                }
                break;
            }
            offset += length;
        }
        let source = dir.join(format!("MRMS_{product}_20261001-181037.grib2"));
        std::fs::write(&source, raw).unwrap();
        let scalar = dir.join(format!("scalar-{index}.obspack"));
        let batch = dir.join(format!("batch-{index}.obspack"));
        let mut args = vec!["decode".to_string(), "--product".into(), product.to_string(),
            "--file".into(), source.display().to_string(), "--out".into(), scalar.display().to_string(),
            "--fetched-at".into(), "2026-10-08T00:00:00Z".into()];
        let scalar_run = Command::new(bin).args(&args).output().unwrap();
        assert!(scalar_run.status.success(), "{}", String::from_utf8_lossy(&scalar_run.stderr));
        let out_index = args.iter().position(|arg| arg == "--out").unwrap() + 1;
        args[out_index] = batch.display().to_string();
        jobs.push(args);
        outputs.push((scalar, batch));
        if index == 1 {
            let scalar_geo = dir.join("scalar.geopack");
            let batch_geo = dir.join("batch.geopack");
            let mut args = vec!["grid".into(), "--product".into(), product.to_string(),
                "--file".into(), source.display().to_string(), "--out".into(), scalar_geo.display().to_string()];
            assert!(Command::new(bin).args(&args).status().unwrap().success());
            *args.last_mut().unwrap() = batch_geo.display().to_string();
            jobs.push(args);
            outputs.push((scalar_geo, batch_geo));
        }
    }
    let manifest = dir.join("jobs.json");
    std::fs::write(&manifest, serde_json::to_vec(&jobs).unwrap()).unwrap();
    std::thread::sleep(std::time::Duration::from_millis(1100));
    let run = Command::new(bin).arg("decode-batch").arg("--files").arg(&manifest).output().unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    let receipt: serde_json::Value = serde_json::from_slice(&run.stdout).unwrap();
    assert_eq!(receipt["status"], "COMPLETE");
    for (index, (scalar, batch)) in outputs.iter().enumerate() {
        assert_eq!(std::fs::read(scalar).unwrap(), std::fs::read(batch).unwrap());
        assert_eq!(receipt["records"][index]["index"], index);
    }
    let mut duplicate = jobs.clone();
    duplicate.push(jobs[0].clone());
    std::fs::write(&manifest, serde_json::to_vec(&duplicate).unwrap()).unwrap();
    let failed = Command::new(bin).arg("decode-batch").arg("--files").arg(&manifest).output().unwrap();
    assert!(!failed.status.success());
    assert!(String::from_utf8_lossy(&failed.stderr).contains("duplicate batch output"));
}

fn frames(dir: &Path, issue: &str, end: &str, out: &str) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_rw_mrms"))
        .args([
            "frames", "--issue", issue, "--start", "2026-10-01T18:10:00Z", "--end", end,
            "--step-minutes", "10", "--files",
        ])
        .arg(dir.join("archive"))
        .arg("--out")
        .arg(dir.join(out))
        .output()
        .expect("rw_mrms runs")
}

fn sha256(bytes: &[u8]) -> String {
    rw_nexrad::s3::hex_sha256(bytes)
}

fn read_json(path: &Path) -> serde_json::Value {
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

#[test]
fn frames_keep_the_sentinels_record_the_stamps_and_say_not_causal_after_the_issue() {
    let dir = scratch("after-issue");
    write_archive(&dir);
    let run = frames(&dir, "2026-10-01T18:00:00Z", "2026-10-01T18:20:00Z", "out");
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    let out = dir.join("out");
    let receipt = read_json(&out.join("nowcast.json"));
    assert_eq!(receipt["schema"], "gpuwm-obs.nowcast-frames.v1");
    assert_eq!(receipt["status"], "READY");
    assert_eq!(receipt["source"]["kind"], "observed");
    assert_eq!(receipt["members"], 1);
    assert_eq!(receipt["causal"], false, "frames stamped after the issue time");
    assert_eq!(receipt["latest_input_time"], "2026-10-01T18:19:02Z");
    let listed = receipt["frames"].as_array().unwrap();
    assert_eq!(listed.len(), 2);
    // The nearest object, with its own seconds: 18:10:37 beats 18:13:30.
    assert_eq!(listed[0]["stamp"], "2026-10-01T18:10:37Z");
    assert_eq!(listed[0]["nominal"], "2026-10-01T18:10:00Z");
    assert_eq!(listed[0]["offset_seconds"], 37);
    assert_eq!(listed[1]["stamp"], "2026-10-01T18:19:02Z");
    assert_eq!(listed[1]["offset_seconds"], -58);

    let lattice = &receipt["lattice"];
    assert_eq!(lattice["kind"], "latlon");
    assert_eq!((lattice["nx"].as_u64(), lattice["ny"].as_u64()), (Some(NX as u64), Some(NY as u64)));
    let lat0 = lattice["lat0"].as_f64().unwrap();
    let lon0 = lattice["lon0"].as_f64().unwrap();
    let dlat = lattice["dlat"].as_f64().unwrap();
    let dlon = lattice["dlon"].as_f64().unwrap();
    assert_eq!(lattice["row_order"], if dlat > 0.0 { "south_to_north" } else { "north_to_south" });

    for frame in listed {
        let path = out.join(frame["file"].as_str().unwrap());
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(frame["sha256"], sha256(&bytes));
        assert_eq!(frame["shape"], serde_json::json!([1, NY, NX]));
        assert_eq!(frame["dtype"], "float32-le");
        let values: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|w| f32::from_le_bytes([w[0], w[1], w[2], w[3]]))
            .collect();
        // Look each archive cell up by latitude and longitude through the
        // receipt's lattice, so the row order is checked, not assumed.
        let at = |north_row: usize, column: usize| {
            let lat = LAT_NORTH - north_row as f64 * STEP_DEG;
            let lon = LON_WEST + column as f64 * STEP_DEG;
            let j = ((lat - lat0) / dlat).round() as usize;
            let i = ((lon - lon0) / dlon).round() as usize;
            values[j * NX + i]
        };
        let archive = archive_values();
        for r in 0..NY {
            for c in 0..NX {
                let want = archive[r * NX + c];
                let got = at(r, c);
                if want <= -900.0 {
                    assert!(got.is_nan(), "-999 must become NaN at ({r}, {c}), got {got}");
                } else {
                    assert_eq!(got, want, "({r}, {c})");
                }
            }
        }
        assert_eq!(frame["cells"]["no_coverage"], 2);
        assert_eq!(frame["cells"]["echo"], 4);
    }

    let inputs = read_json(&out.join("inputs.json"));
    let rows = inputs["objects"].as_array().unwrap();
    assert_eq!(rows.len(), 2);
    assert!(rows.iter().all(|r| r["after_issue"] == true));
    assert_eq!(rows[0]["stamp"], "2026-10-01T18:10:37Z");
    assert_eq!(rows[0]["slot"], "2026-10-01T18:10:00Z");
    assert_eq!(inputs["slots"].as_array().unwrap().len(), 2);
    let corners = lattice["corners_latlon"].as_object().unwrap();
    assert_eq!(corners.keys().cloned().collect::<Vec<_>>(), ["j0_i0", "j0_in", "jn_i0", "jn_in"]);
    assert!(rows[0]["name"].as_str().unwrap().ends_with(".grib2.gz"));
    assert_eq!(
        receipt["inputs"]["sha256"],
        sha256(&std::fs::read(out.join("inputs.json")).unwrap())
    );

    // The adapter reads this receipt as an observed source and holds its
    // corners to its lattice.
    let loaded = rw_nexrad::grid_composite::Frames::load(&out).expect("the adapter reads it");
    assert_eq!(loaded.kind, "observed");
    assert!(!loaded.causal);
}

#[test]
fn frames_at_or_before_the_issue_are_causal() {
    let dir = scratch("before-issue");
    write_archive(&dir);
    let run = frames(&dir, "2026-10-01T18:30:00Z", "2026-10-01T18:20:00Z", "out");
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    let receipt = read_json(&dir.join("out").join("nowcast.json"));
    assert_eq!(receipt["causal"], true);
}

#[test]
fn a_window_end_with_no_object_within_120_s_is_refused() {
    let dir = scratch("missing");
    write_archive(&dir);
    let run = frames(&dir, "2026-10-01T18:00:00Z", "2026-10-01T18:30:00Z", "out");
    assert!(!run.status.success());
    let stderr = String::from_utf8_lossy(&run.stderr);
    assert!(stderr.contains("no MRMS composite within 120 s of 2026-10-01T18:30:00Z"), "{stderr}");
    assert!(!dir.join("out").join("nowcast.json").exists(), "nothing published on a refusal");
}
