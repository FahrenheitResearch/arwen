//! The executable must accept a reflectivity volume above the former cap.
//! This sparse synthetic CDF-5 file is built at run time and then deleted.

use std::fs::{self, File};
use std::io::{Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use serde_json::{json, Value};

const NX: usize = 2_695;
const NY: usize = 1_585;
const NZ: usize = 50;
const ELEMENTS: usize = NX * NY * NZ;
const FIRST: f32 = 24.125;
const LAST: f32 = 45.5;

struct Scratch(PathBuf);

impl Scratch {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let serial = NEXT.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "rw-scoreboard-large-{}-{serial}",
            std::process::id()
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }

    fn file(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        for entry in fs::read_dir(&self.0).unwrap() {
            let entry = entry.unwrap();
            assert!(entry.file_type().unwrap().is_file());
            println!(
                "fixture_deleted {} {}",
                entry.file_name().to_string_lossy(),
                entry.metadata().unwrap().len()
            );
            fs::remove_file(entry.path()).unwrap();
        }
        fs::remove_dir(&self.0).unwrap();
    }
}

fn word(header: &mut Vec<u8>, value: u32) {
    header.extend_from_slice(&value.to_be_bytes());
}

fn count(header: &mut Vec<u8>, value: u64) {
    header.extend_from_slice(&value.to_be_bytes());
}

fn padded(header: &mut Vec<u8>, bytes: &[u8]) {
    header.extend_from_slice(bytes);
    while header.len() % 4 != 0 {
        header.push(0);
    }
}

fn name(header: &mut Vec<u8>, value: &str) {
    count(header, value.len() as u64);
    padded(header, value.as_bytes());
}

/// A fixed-dimension CDF-5 header uses 64-bit dimension lengths, value counts
/// and offsets. The large array's zero values occupy holes in the file; only
/// its two nonzero endpoints require writes.
fn fixture(path: &Path) {
    assert_eq!(ELEMENTS, 213_578_750);
    assert!(ELEMENTS > 134_217_728);
    let mut header = b"CDF\x05".to_vec();
    count(&mut header, 0); // no unlimited dimension
    word(&mut header, 10);
    count(&mut header, 5);
    for (label, length) in [
        ("Time", 1),
        ("DateStrLen", 19),
        ("bottom_top", NZ),
        ("south_north", NY),
        ("west_east", NX),
    ] {
        name(&mut header, label);
        count(&mut header, length as u64);
    }
    word(&mut header, 12);
    count(&mut header, 1);
    name(&mut header, "SIMULATION_START_DATE");
    word(&mut header, 2); // NC_CHAR
    count(&mut header, 19);
    padded(&mut header, b"2026-01-01_00:00:00");
    word(&mut header, 11);
    count(&mut header, 5);
    let mut offsets = Vec::new();
    for (label, dimids, dtype, bytes) in [
        ("Times", vec![0, 1], 2, 20u64),
        ("XLAT", vec![0, 3, 4], 5, (NX * NY * 4) as u64),
        ("XLONG", vec![0, 3, 4], 5, (NX * NY * 4) as u64),
        ("HGT", vec![0, 3, 4], 5, (NX * NY * 4) as u64),
        ("REFL_10CM", vec![0, 2, 3, 4], 5, (ELEMENTS * 4) as u64),
    ] {
        name(&mut header, label);
        count(&mut header, dimids.len() as u64);
        for dimid in dimids {
            count(&mut header, dimid);
        }
        word(&mut header, 0); // no variable attributes
        count(&mut header, 0);
        word(&mut header, dtype);
        count(&mut header, bytes);
        offsets.push((header.len(), bytes));
        count(&mut header, 0);
    }
    let mut cursor = header.len() as u64;
    let mut begins = Vec::new();
    for (patch, bytes) in offsets {
        header[patch..patch + 8].copy_from_slice(&cursor.to_be_bytes());
        begins.push(cursor);
        cursor = cursor.checked_add(bytes).unwrap();
    }
    let mut file = File::create(path).unwrap();
    file.write_all(&header).unwrap();
    file.set_len(cursor).unwrap();
    file.seek(SeekFrom::Start(begins[0])).unwrap();
    file.write_all(b"2026-01-01_00:00:00\0").unwrap();
    let mut row = Vec::with_capacity(NX * 4);
    for (position, latitude) in [(begins[1], true), (begins[2], false)] {
        file.seek(SeekFrom::Start(position)).unwrap();
        for j in 0..NY {
            row.clear();
            for i in 0..NX {
                let value = if latitude {
                    j as f32 / (NY - 1) as f32
                } else {
                    i as f32 / (NX - 1) as f32
                };
                row.extend_from_slice(&value.to_be_bytes());
            }
            file.write_all(&row).unwrap();
        }
    }
    file.seek(SeekFrom::Start(begins[4])).unwrap();
    file.write_all(&FIRST.to_be_bytes()).unwrap();
    file.seek(SeekFrom::Start(cursor - 4)).unwrap();
    file.write_all(&LAST.to_be_bytes()).unwrap();
    file.sync_all().unwrap();
}

#[test]
fn executable_extracts_stations_from_a_213578750_value_reflectivity_volume() {
    let scratch = Scratch::new();
    let frame = scratch.file("frame.nc");
    let artifact = scratch.file("stations.json");
    let request = scratch.file("request.json");
    fixture(&frame);
    let native = netcrust::File::open(&frame).unwrap();
    assert_eq!(
        native.variable("REFL_10CM").unwrap().shape(),
        [1, NZ, NY, NX]
    );
    drop(native);
    fs::write(
        &request,
        serde_json::to_vec(&json!({
            "action":"extract", "method_id":"scoreboard-v2", "format":"wrf",
            "input":frame, "valid_time":"2026-01-01T00:00:00Z", "output":artifact,
            "stations":[
                {"station_id":"A", "lat":0.0, "lon":0.0, "elevation_m":0.0},
                {"station_id":"B", "lat":1.0, "lon":1.0, "elevation_m":0.0},
                {"station_id":"C", "lat":0.5, "lon":0.5, "elevation_m":0.0}
            ]
        }))
        .unwrap(),
    )
    .unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_rw_scoreboard"))
        .args(["--request", request.to_str().unwrap()])
        .env("CUDA_VISIBLE_DEVICES", "")
        .env("GPUWM_NO_LOCAL_GPU", "1")
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let actual: Value = serde_json::from_slice(&fs::read(&artifact).unwrap()).unwrap();
    assert_eq!(actual["drops"], json!([]));
    let points = actual["points"].as_array().unwrap();
    assert_eq!(points.len(), 3);
    for (index, expected) in [f64::from(FIRST), f64::from(LAST), 0.0]
        .into_iter()
        .enumerate()
    {
        // The existing scientific operator interpolates linear reflectivity
        // and converts back to dBZ, including at a native grid point.
        let expected = 10.0 * 10f64.powf(expected / 10.0).log10();
        let value = points[index]["values"]["reflectivity"].as_f64().unwrap();
        assert_eq!(value.to_bits(), expected.to_bits(), "station {index}");
    }
}
