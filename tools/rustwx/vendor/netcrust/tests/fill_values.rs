//! Stored fill values and numeric promotion retain their declared types.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use netcrust::{NcReadable, NcSliceInfo, NcSliceInfoElem};

static NEXT_FILE: AtomicU64 = AtomicU64::new(0);

struct GeneratedFile(PathBuf);

impl Drop for GeneratedFile {
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

fn generate_fixture(kind: &str, script: &str) -> GeneratedFile {
    let fixture = GeneratedFile(std::env::temp_dir().join(format!(
        "netcrust-fill-{}-{}.h5",
        std::process::id(),
        NEXT_FILE.fetch_add(1, Ordering::Relaxed),
    )));
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
        .arg(kind)
        .output()
        .expect("run independent HDF5 fixture writer with h5py");
    assert!(
        generation.status.success(),
        "HDF5 fixture generation failed: {}",
        String::from_utf8_lossy(&generation.stderr)
    );
    fixture
}

const MATRIX_FIXTURE: &str = r#"
import sys
import h5py
import numpy as np

kind = sys.argv[2]
if kind.startswith(('i', 'u')):
    limits = np.iinfo(kind)
    fill = limits.min + 37 if kind.startswith('i') else limits.max - 37
    markers = [limits.min, limits.max, 0, fill]
    tail = 17
elif kind == 'f4':
    fill = -123.25
    markers = np.array([0x80000000, 0x7fc00035, 0, 0xc2f68000],
        dtype=np.uint32).view(np.float32)
    tail = 17.25
else:
    fill = -123.25
    markers = np.array([0x8000000000000000, 0x7ff8000000000035, 0,
        0xc05ed00000000000], dtype=np.uint64).view(np.float64)
    tail = 17.25

with h5py.File(sys.argv[1], 'w', libver='earliest') as f:
    for count in [17, 1048577]:
        dimension = f.create_dataset('axis_' + str(count), (count,), dtype='i1',
            chunks=(min(count, 1024),))
        dimension.make_scale('axis_' + str(count))
        for order, prefix in [('le', '<'), ('be', '>')]:
            dtype = np.dtype(prefix + kind)
            for route in ['listed', 'raw']:
                name = route + '_' + str(count) + '_' + order
                dataset = f.create_dataset(name, (count,), dtype=dtype,
                    chunks=(4 if count == 17 else 1024,),
                    compression='gzip', fillvalue=fill)
                dataset.attrs.create('_FillValue', np.array(fill, dtype=dtype),
                    dtype=dtype)
                dataset[:4] = np.array(markers, dtype=dtype)
                dataset[count - 1] = tail
                if route == 'raw':
                    # Coordinate datasets are dimension scales, so the NetCDF
                    # variable index omits them and the facade uses raw HDF5.
                    dataset.make_scale(name)
                else:
                    dataset.dims[0].attach_scale(dimension)
"#;

trait StoredWord: NcReadable + Copy {
    fn bits(self) -> u64;
    fn promoted(self) -> f64;
}

macro_rules! integer_words {
    ($($kind:ty),+ $(,)?) => {
        $(impl StoredWord for $kind {
            fn bits(self) -> u64 { self as u64 }
            fn promoted(self) -> f64 { self as f64 }
        })+
    };
}

integer_words!(i8, u8, i16, u16, i32, u32, i64, u64);

impl StoredWord for f32 {
    fn bits(self) -> u64 {
        self.to_bits() as u64
    }
    fn promoted(self) -> f64 {
        self as f64
    }
}

impl StoredWord for f64 {
    fn bits(self) -> u64 {
        self.to_bits()
    }
    fn promoted(self) -> f64 {
        self
    }
}

fn expected<T: Copy>(index: usize, count: usize, markers: &[T; 4], tail: T) -> T {
    if index < 4 {
        markers[index]
    } else if index + 1 == count {
        tail
    } else {
        markers[3]
    }
}

fn assert_typed<T: StoredWord>(
    name: &str,
    values: &[T],
    indices: impl Iterator<Item = usize>,
    count: usize,
    markers: &[T; 4],
    tail: T,
) {
    for (value, index) in values.iter().zip(indices) {
        assert_eq!(
            value.bits(),
            expected(index, count, markers, tail).bits(),
            "stored word mismatch for {name} at {index}"
        );
    }
}

fn assert_promoted<T: StoredWord>(
    name: &str,
    values: &[f64],
    indices: impl Iterator<Item = usize>,
    count: usize,
    markers: &[T; 4],
    tail: T,
) {
    for (value, index) in values.iter().zip(indices) {
        assert_eq!(
            value.to_bits(),
            expected(index, count, markers, tail).promoted().to_bits(),
            "promoted word mismatch for {name} at {index}"
        );
    }
}

fn check_matrix_type<T: StoredWord>(kind: &str, markers: [T; 4], tail: T) {
    let fixture = generate_fixture(kind, MATRIX_FIXTURE);
    assert!(std::fs::metadata(&fixture.0).unwrap().len() < 1_048_576);
    let file = netcrust::open(&fixture.0).unwrap();
    let hdf5 = hdf5_reader::Hdf5File::open(&fixture.0).unwrap();
    for count in [17usize, 1_048_577] {
        let selection = NcSliceInfo {
            selections: vec![NcSliceInfoElem::Slice {
                start: 1,
                end: count as u64,
                step: 3,
            }],
        };
        let hdf5_selection = hdf5_reader::SliceInfo {
            selections: vec![hdf5_reader::SliceInfoElem::Slice {
                start: 1,
                end: count as u64,
                step: 3,
            }],
        };
        for order in ["le", "be"] {
            for route in ["listed", "raw"] {
                let name = format!("{route}_{count}_{order}");
                assert_eq!(file.variable(&name).is_some(), route == "listed", "{name}");
                let dataset = hdf5.dataset(&name).unwrap();
                assert_eq!(
                    dataset
                        .attribute("_FillValue")
                        .unwrap()
                        .read_scalar::<T>()
                        .unwrap()
                        .bits(),
                    markers[3].bits(),
                    "{name}"
                );

                let typed = if route == "listed" {
                    file.read_array::<T>(&name).unwrap()
                } else {
                    dataset.read_array::<T>().unwrap()
                };
                assert_eq!(typed.shape(), &[count], "{name}");
                assert_typed(
                    &name,
                    typed.as_slice().unwrap(),
                    0..count,
                    count,
                    &markers,
                    tail,
                );

                let promoted = file.read_array_f64(&name).unwrap();
                assert_eq!(promoted.shape(), &[count], "{name}");
                assert_promoted(&name, promoted.values(), 0..count, count, &markers, tail);

                let typed_slice = if route == "listed" {
                    file.read_array_slice::<T>(&name, &selection).unwrap()
                } else {
                    dataset.read_slice::<T>(&hdf5_selection).unwrap()
                };
                assert_eq!(typed_slice.len(), (count - 1).div_ceil(3), "{name}");
                assert_typed(
                    &name,
                    typed_slice.as_slice().unwrap(),
                    (1..count).step_by(3),
                    count,
                    &markers,
                    tail,
                );

                let promoted_slice = file.read_array_f64_slice(&name, &selection).unwrap();
                assert_eq!(promoted_slice.shape(), &[typed_slice.len()], "{name}");
                assert_promoted(
                    &name,
                    promoted_slice.values(),
                    (1..count).step_by(3),
                    count,
                    &markers,
                    tail,
                );
            }
        }
    }
}

macro_rules! fixed_point_case {
    ($test:ident, $kind:ty, $dtype:literal, $fill:expr) => {
        #[test]
        fn $test() {
            check_matrix_type::<$kind>($dtype, [<$kind>::MIN, <$kind>::MAX, 0, $fill], 17);
        }
    };
}

fixed_point_case!(i8_fill_values_keep_signedness, i8, "i1", i8::MIN + 37);
fixed_point_case!(u8_fill_values_keep_signedness, u8, "u1", u8::MAX - 37);
fixed_point_case!(i16_fill_values_keep_signedness, i16, "i2", i16::MIN + 37);
fixed_point_case!(u16_fill_values_keep_signedness, u16, "u2", u16::MAX - 37);
fixed_point_case!(i32_fill_values_keep_signedness, i32, "i4", i32::MIN + 37);
fixed_point_case!(u32_fill_values_keep_signedness, u32, "u4", u32::MAX - 37);
fixed_point_case!(i64_fill_values_keep_signedness, i64, "i8", i64::MIN + 37);
fixed_point_case!(u64_fill_values_keep_signedness, u64, "u8", u64::MAX - 37);

#[test]
fn f32_fill_values_preserve_stored_and_promoted_words() {
    check_matrix_type::<f32>(
        "f4",
        [-0.0, f32::from_bits(0x7fc0_0035), 0.0, -123.25],
        17.25,
    );
}

#[test]
fn f64_fill_values_preserve_stored_and_promoted_words() {
    check_matrix_type::<f64>(
        "f8",
        [-0.0, f64::from_bits(0x7ff8_0000_0000_0035), 0.0, -123.25],
        17.25,
    );
}

#[test]
fn int64_time_coordinate_with_fill_value_is_decoded() {
    let fixture = generate_fixture(
        "i8",
        r#"
import sys
import h5py
import numpy as np
fill = np.int64(-9223372036854775806)
with h5py.File(sys.argv[1], 'w', libver='earliest') as f:
    time = f.create_dataset('valid_time', (3,), dtype='i8', chunks=(2,),
        fillvalue=fill)
    time.attrs.create('_FillValue', fill, dtype='i8')
    time.attrs['units'] = np.bytes_('seconds since 1970-01-01')
    time.make_scale('valid_time')
    time[0] = np.int64(1590969600)
"#,
    );
    let file = netcrust::open(&fixture.0).unwrap();
    assert!(file.variable("valid_time").is_none());
    let output = file.read_array_f64("valid_time").unwrap();
    assert_eq!(output.shape(), &[3]);
    assert_eq!(
        output
            .values()
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        [
            1590969600.0f64.to_bits(),
            (-9223372036854775806i64 as f64).to_bits(),
            (-9223372036854775806i64 as f64).to_bits()
        ]
    );
}
