//! CPU-only replay of regular-grid GRIB export, independent of packing.
//! Run with GPUWM_MAPPED_ENGINE_THREADS=N; hashes are outside timed regions.
//! Synthetic 0.25-degree global geometry, not a forecast-shape qualification.
use std::hint::black_box;
use std::time::Instant;

use grib_core::grib2::{DataRepresentation, Grib2Message, GridDefinition};
use mapped_engine::grib::regular_latlon_frame;

fn cpu_ticks() -> Option<u64> {
    let stat = std::fs::read_to_string("/proc/self/stat").ok()?;
    let fields: Vec<_> = stat.rsplit_once(") ")?.1.split_whitespace().collect();
    Some(fields[11].parse::<u64>().ok()? + fields[12].parse::<u64>().ok()?)
}

fn main() {
    let nx = 1440;
    let ny = 721;
    let planes = 37;
    let raw: Vec<f64> = (0..nx * ny)
        .map(|i| match i % 1009 {
            0 => f64::from_bits(0x7ff8_0000_0000_0123),
            1 => -0.0,
            _ => (i as f64 - 500_000.0) / 17.0,
        })
        .collect();
    // Warm only the worker pool; no timed stage includes compilation or proof.
    mapped_engine::threads::install(|| ());
    for scan in [0x40, 0x00] {
        let message = Grib2Message {
            discipline: 0,
            identification: Default::default(),
            reference_time: chrono::NaiveDate::from_ymd_opt(2026, 10, 18)
                .unwrap()
                .and_hms_opt(0, 0, 0)
                .unwrap(),
            grid: GridDefinition {
                template: 0,
                nx: nx as u32,
                ny: ny as u32,
                num_data_points: (nx * ny) as u32,
                lat1: if scan == 0x40 { -90.0 } else { 90.0 },
                lon1: 0.0,
                dx: 0.25,
                dy: 0.25,
                scan_mode: scan,
                ..Default::default()
            },
            product: Default::default(),
            data_rep: DataRepresentation::default(),
            bitmap: None,
            raw_data: Vec::new(),
        };
        let ticks = cpu_ticks();
        let clock = Instant::now();
        let mut last = None;
        for _ in 0..planes {
            last = Some(black_box(
                regular_latlon_frame(&message, black_box(&raw)).unwrap(),
            ));
        }
        let seconds = clock.elapsed().as_secs_f64();
        let ticks = ticks.zip(cpu_ticks()).map(|(a, b)| b - a);
        let (latitude, longitude, values) = last.unwrap();
        // Hash every output bit (including the NaN payload and signed zero).
        let mut bytes = Vec::with_capacity(values.len() * 8);
        for value in &values {
            bytes.extend_from_slice(&value.to_bits().to_le_bytes());
        }
        println!(
            "{}",
            serde_json::json!({
                "stage": "regular_latlon_frame", "synthetic": true,
                "shape": [planes, ny, nx], "scan_mode": scan,
                "threads": mapped_engine::threads::width(), "seconds": seconds,
                "cpu_ticks": ticks, "words_per_plane": values.len(),
                "values_sha256": mapped_engine::digest::bytes_sha256(&bytes),
                "latitude_sha256": mapped_engine::digest::array_sha256(&[ny], &latitude),
                "longitude_sha256": mapped_engine::digest::array_sha256(&[nx], &longitude),
            })
        );
    }
    // Optional real GRIB objects: measure the complete selected-record
    // unpack + export + metadata path. Read/parse and proof stay outside.
    for path in std::env::args().skip(1) {
        let bytes = std::fs::read(&path).unwrap();
        let file = grib_core::grib2::Grib2File::from_bytes(&bytes).unwrap();
        let wanted: Vec<_> = (0..file.messages.len()).collect();
        let declaration = mapped_engine::model::GridDeclaration::regular();
        let ticks = cpu_ticks();
        let clock = Instant::now();
        let mut last = None;
        for _ in 0..planes {
            last = Some(black_box(
                mapped_engine::grib::grib2_records(&file, &path, &wanted, &declaration).unwrap(),
            ));
        }
        let seconds = clock.elapsed().as_secs_f64();
        let ticks = ticks.zip(cpu_ticks()).map(|(a, b)| b - a);
        let records: Vec<_> = last
            .unwrap()
            .iter()
            .map(|record| {
                let mut bytes = Vec::with_capacity(record.values.len() * 8);
                for value in &record.values {
                    bytes.extend_from_slice(&value.to_bits().to_le_bytes());
                }
                serde_json::json!({
                    "index": record.index, "shape": record.values.shape(),
                    "values_sha256": mapped_engine::digest::bytes_sha256(&bytes),
                    "latitude_sha256": mapped_engine::digest::array_sha256(
                        &[record.latitude.len()], &record.latitude),
                    "longitude_sha256": mapped_engine::digest::array_sha256(
                        &[record.longitude.len()], &record.longitude),
                    "grid_fingerprint": record.grid_fingerprint,
                })
            })
            .collect();
        println!(
            "{}",
            serde_json::json!({
                "stage": "grib2_records", "synthetic": false, "input": path,
                "input_sha256": mapped_engine::digest::bytes_sha256(&bytes),
                "repeats": planes, "threads": mapped_engine::threads::width(),
                "seconds": seconds, "cpu_ticks": ticks, "records": records,
            })
        );
        // One production-style multi-message call, so the existing pool
        // can overlap independent unpackers as it does on a source file.
        let batch = grib_core::grib2::Grib2File {
            messages: (0..planes)
                .flat_map(|_| file.messages.iter().cloned())
                .collect(),
        };
        let wanted: Vec<_> = (0..batch.messages.len()).collect();
        let ticks = cpu_ticks();
        let clock = Instant::now();
        let batch_records =
            mapped_engine::grib::grib2_records(&batch, &path, &wanted, &declaration).unwrap();
        let seconds = clock.elapsed().as_secs_f64();
        let ticks = ticks.zip(cpu_ticks()).map(|(a, b)| b - a);
        let mut proof = Vec::new();
        let mut words = 0;
        for record in &batch_records {
            let mut bytes = Vec::with_capacity(record.values.len() * 8);
            for value in &record.values {
                bytes.extend_from_slice(&value.to_bits().to_le_bytes());
            }
            words += record.values.len();
            proof.push(mapped_engine::digest::bytes_sha256(&bytes));
        }
        println!(
            "{}",
            serde_json::json!({
                "stage": "grib2_records_batch", "synthetic": true,
                "input": path, "input_sha256": mapped_engine::digest::bytes_sha256(&bytes),
                "replicated_messages": batch.messages.len(),
                "threads": mapped_engine::threads::width(), "seconds": seconds, "cpu_ticks": ticks,
                "words": words, "field_sha256": proof,
            })
        );
    }
}
