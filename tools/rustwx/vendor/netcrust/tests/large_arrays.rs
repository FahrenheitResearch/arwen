//! Large arrays are generated as sparse files and removed after each test.

use std::fs::{File as DiskFile, OpenOptions};
use std::io::{Seek, SeekFrom, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use netcrust::{NcSliceInfo, NcSliceInfoElem};

static NEXT_FILE: AtomicU64 = AtomicU64::new(0);

struct SparseFile(PathBuf);

impl Drop for SparseFile {
    fn drop(&mut self) {
        let bytes = std::fs::metadata(&self.0)
            .map(|metadata| metadata.len())
            .unwrap_or(0);
        match std::fs::remove_file(&self.0) {
            Ok(()) => eprintln!(
                "removed generated fixture {} ({bytes} logical bytes)",
                self.0.display()
            ),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
            Err(err) => panic!("remove generated fixture {}: {err}", self.0.display()),
        }
    }
}

fn push_count(bytes: &mut Vec<u8>, value: u64) {
    bytes.extend_from_slice(&value.to_be_bytes());
}

fn push_name(bytes: &mut Vec<u8>, name: &str) {
    push_count(bytes, name.len() as u64);
    bytes.extend_from_slice(name.as_bytes());
    while bytes.len() % 4 != 0 {
        bytes.push(0);
    }
}

fn sparse_cdf5(shape: &[u64], dtype: u32, width: u64, markers: &[(u64, &[u8])]) -> SparseFile {
    let path = std::env::temp_dir().join(format!(
        "netcrust-large-{}-{}.nc",
        std::process::id(),
        NEXT_FILE.fetch_add(1, Ordering::Relaxed),
    ));
    let mut bytes = b"CDF\x05".to_vec();
    push_count(&mut bytes, 0);
    bytes.extend_from_slice(&10u32.to_be_bytes());
    push_count(&mut bytes, shape.len() as u64);
    for (axis, &size) in shape.iter().enumerate() {
        push_name(&mut bytes, &format!("axis{axis}"));
        push_count(&mut bytes, size);
    }
    bytes.extend_from_slice(&0u32.to_be_bytes());
    push_count(&mut bytes, 0);
    bytes.extend_from_slice(&11u32.to_be_bytes());
    push_count(&mut bytes, 1);
    push_name(&mut bytes, "field");
    push_count(&mut bytes, shape.len() as u64);
    for axis in 0..shape.len() {
        push_count(&mut bytes, axis as u64);
    }
    bytes.extend_from_slice(&0u32.to_be_bytes());
    push_count(&mut bytes, 0);
    bytes.extend_from_slice(&dtype.to_be_bytes());
    let elements = shape.iter().copied().product::<u64>();
    let data_bytes = elements * width;
    push_count(&mut bytes, (data_bytes + 3) & !3);
    let data_start = (bytes.len() + 8) as u64;
    push_count(&mut bytes, data_start);
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)
        .unwrap();
    file.write_all(&bytes).unwrap();
    file.set_len(data_start + data_bytes).unwrap();
    for &(element, marker) in markers {
        assert_eq!(marker.len() as u64, width);
        file.seek(SeekFrom::Start(data_start + element * width))
            .unwrap();
        file.write_all(marker).unwrap();
    }
    file.sync_all().unwrap();
    drop(file);
    SparseFile(path)
}

fn window(start: u64, end: u64, step: u64) -> NcSliceInfo {
    NcSliceInfo {
        selections: vec![NcSliceInfoElem::Slice { start, end, step }],
    }
}

#[test]
fn dense_stored_read_exceeds_retired_array_and_axis_limits() {
    let count = 134_217_737u64;
    let fixture = sparse_cdf5(&[count], 7, 1, &[(0, &[19]), (count - 1, &[247])]);
    let file = netcrust::open(&fixture.0).unwrap();
    assert_eq!(file.variable("field").unwrap().shape(), [count as usize]);
    let array = file.read_array::<u8>("field").unwrap();
    assert_eq!(array.shape(), &[count as usize]);
    assert_eq!(array[ndarray::IxDyn(&[0])], 19);
    assert_eq!(array[ndarray::IxDyn(&[count as usize - 1])], 247);
    assert_eq!(array.iter().map(|&value| value as u64).sum::<u64>(), 266);
}

#[test]
fn selective_reads_use_64_bit_offsets_and_preserve_float_words() {
    let count = 536_870_921u64;
    let samples = [
        f64::from_bits(0x8000_0000_0000_0000),
        f64::from_bits(0x7ff8_0000_0000_0035),
        17.25,
    ];
    let words = samples.map(f64::to_be_bytes);
    let fixture = sparse_cdf5(
        &[count],
        6,
        8,
        &[
            (count - 5, &words[0]),
            (count - 3, &words[1]),
            (count - 1, &words[2]),
        ],
    );
    assert!(
        DiskFile::open(&fixture.0)
            .unwrap()
            .metadata()
            .unwrap()
            .len()
            > u32::MAX as u64
    );
    let file = netcrust::open(&fixture.0).unwrap();
    let selection = window(count - 5, count, 2);
    let typed = file.read_array_slice::<f64>("field", &selection).unwrap();
    let promoted = file.read_array_f64_slice("field", &selection).unwrap();
    assert_eq!(typed.shape(), &[3]);
    for ((actual, promoted), expected) in typed.iter().zip(promoted.values()).zip(samples) {
        assert_eq!(actual.to_bits(), expected.to_bits());
        assert_eq!(promoted.to_bits(), expected.to_bits());
    }
}

#[test]
fn promoted_multislab_read_matches_exact_stored_values() {
    let count = 1_048_577u64;
    let words = [(-0.0f64).to_be_bytes(), 3.75f64.to_be_bytes()];
    let fixture = sparse_cdf5(&[count], 6, 8, &[(0, &words[0]), (count - 1, &words[1])]);
    let file = netcrust::open(&fixture.0).unwrap();
    let output = file.read_array_f64("field").unwrap();
    assert_eq!(output.shape(), &[count as usize]);
    assert_eq!(output.values()[0].to_bits(), (-0.0f64).to_bits());
    assert_eq!(
        output.values()[count as usize - 1].to_bits(),
        3.75f64.to_bits()
    );
    assert!(output.values()[1..count as usize - 1]
        .iter()
        .all(|value| value.to_bits() == 0));
    let via_variable = file.variable("field").unwrap().array_f64().unwrap();
    assert_eq!(
        output
            .values()
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        via_variable
            .values()
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );
}

#[test]
fn compressed_hdf5_dense_and_64_bit_coordinate_reads() {
    // h5py is the independent fixture writer for this format. The generated
    // file stores two small compressed chunks and leaves all other cells at
    // their declared fill values.
    let path = std::env::temp_dir().join(format!(
        "netcrust-large-{}-{}.h5",
        std::process::id(),
        NEXT_FILE.fetch_add(1, Ordering::Relaxed),
    ));
    let fixture = SparseFile(path);
    let script = r#"
import sys
import h5py
import numpy as np
with h5py.File(sys.argv[1], 'w', libver='earliest') as f:
    dimensions = {}
    for name, count in [('time', 2), ('row', 8193), ('column', 8193),
        ('nCells', 4294967303)]:
        dimension = f.create_dataset(name, (count,), dtype='i1', chunks=(min(count, 1024),))
        dimension.make_scale(name)
        dimensions[name] = dimension
    volume = f.create_dataset('volume', (2, 8193, 8193), dtype='u1',
        chunks=(1, 64, 64), compression='gzip', fillvalue=12)
    for axis, name in enumerate(['time', 'row', 'column']):
        volume.dims[axis].attach_scale(dimensions[name])
    volume[1, 8192, 8192] = 203
    count = 4294967303
    coordinate = f.create_dataset('coordinate', (count,), dtype='f8',
        chunks=(1024,), compression='gzip', fillvalue=3.25)
    coordinate.dims[0].attach_scale(dimensions['nCells'])
    words = np.array([0x8000000000000000, 0x7ff8000000000035,
        0x4031400000000000], dtype=np.uint64).view(np.float64)
    coordinate[count-5:count:2] = words
"#;
    let python = std::env::var("NETCRUST_TEST_PYTHON").unwrap_or_else(|_| {
        if cfg!(windows) {
            "python".to_string()
        } else {
            "python3".to_string()
        }
    });
    let generation = std::process::Command::new(python)
        .args(["-c", script])
        .arg(&fixture.0)
        .output()
        .expect("run HDF5 fixture writer with h5py");
    assert!(
        generation.status.success(),
        "HDF5 fixture generation failed: {}",
        String::from_utf8_lossy(&generation.stderr)
    );
    assert!(
        DiskFile::open(&fixture.0)
            .unwrap()
            .metadata()
            .unwrap()
            .len()
            < 1_048_576
    );
    let file = netcrust::open(&fixture.0).unwrap();
    let metadata = file.hdf5_root_datasets().unwrap();
    assert!(metadata
        .iter()
        .any(|dataset| dataset.name() == "volume" && dataset.shape() == [2, 8193, 8193]));
    let volume = file.read_array::<u8>("volume").unwrap();
    assert_eq!(volume.len(), 134_250_498);
    assert_eq!(volume[ndarray::IxDyn(&[1, 8192, 8192])], 203);
    assert!(volume.iter().enumerate().all(|(index, &value)| {
        value == if index + 1 == volume.len() { 203 } else { 12 }
    }));
    drop(volume);
    let selection = window(4_294_967_298, 4_294_967_303, 2);
    let coordinate = file.read_array_f64_slice("coordinate", &selection).unwrap();
    assert_eq!(coordinate.shape(), &[3]);
    assert_eq!(
        coordinate
            .values()
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        [
            0x8000_0000_0000_0000,
            0x7ff8_0000_0000_0035,
            0x4031_4000_0000_0000
        ]
    );
}
