//! CPU-only array fingerprint replay. Read, decode and reference hashing are untimed.
use mapped_engine::digest::array_sha256;
use std::hint::black_box;
use std::time::Instant;
fn ticks() -> u64 {
    let s = std::fs::read_to_string("/proc/self/stat").unwrap();
    let f: Vec<_> = s.rsplit_once(") ").unwrap().1.split_whitespace().collect();
    f[11].parse::<u64>().unwrap() + f[12].parse::<u64>().unwrap()
}
fn measure(label: &str, shape: &[usize], values: &[f64], repeats: usize, input: serde_json::Value) {
    let start_ticks = ticks();
    let clock = Instant::now();
    let mut digests = Vec::new();
    for _ in 0..repeats {
        digests.push(black_box(array_sha256(shape, black_box(values))));
    }
    let seconds = clock.elapsed().as_secs_f64();
    let cpu_ticks = ticks() - start_ticks;
    assert!(digests.iter().all(|v| v == &digests[0]));
    println!(
        "{}",
        serde_json::json!({"stage":"array_sha256","label":label,"shape":shape,"words":values.len(),"repeats":repeats,"seconds":seconds,"cpu_ticks":cpu_ticks,"digest":digests[0],"input":input})
    );
}
fn main() {
    let shape = [37, 721, 1440];
    let count = shape.iter().product();
    let values: Vec<_> = (0..count)
        .map(|i| f64::from_bits(0x3ff0_0000_0000_0000 + (i as u64 % 0x1000_0000)))
        .collect();
    measure(
        "synthetic-global-37-planes",
        &shape,
        &values,
        1,
        serde_json::Value::Null,
    );
    drop(values);
    for path in std::env::args().skip(1) {
        let bytes = std::fs::read(&path).unwrap();
        let sha = mapped_engine::digest::bytes_sha256(&bytes);
        let file = grib_core::grib2::Grib2File::from_bytes(&bytes).unwrap();
        let wanted: Vec<_> = (0..file.messages.len()).collect();
        let records = mapped_engine::grib::grib2_records(
            &file,
            &path,
            &wanted,
            &mapped_engine::model::GridDeclaration::regular(),
        )
        .unwrap();
        for record in records {
            let values: Vec<_> = record.values.iter().copied().collect();
            measure(
                "real-GEFS-record",
                record.values.shape(),
                &values,
                8,
                serde_json::json!({"path":path,"sha256":sha,"record":record.index,"grid":record.grid_fingerprint}),
            );
        }
    }
}
