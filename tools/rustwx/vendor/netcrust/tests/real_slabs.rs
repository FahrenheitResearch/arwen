//! Optional real-input comparison against the existing unchunked backend.

use netcrust::{NcSliceInfo, NcSliceInfoElem};

fn identical(actual: &[f64], expected: impl IntoIterator<Item = f64>) {
    let mut expected = expected.into_iter();
    for (index, value) in actual.iter().enumerate() {
        assert_eq!(value.to_bits(), expected.next().unwrap().to_bits(), "value {index}");
    }
    assert!(expected.next().is_none());
}

#[test]
fn real_mesh_numeric_reads_are_bit_identical_to_backend() {
    let Some(path) = std::env::var_os("NETCRUST_REAL_FIXTURE") else {
        eprintln!("real mesh comparison requires NETCRUST_REAL_FIXTURE");
        return;
    };
    let file = netcrust::open(&path).unwrap();
    let baseline = netcdf_reader::NcFile::open(&path).unwrap();
    for name in ["theta", "zgrid", "cf1"] {
        let expected = baseline.read_variable_as_f64(name).unwrap();
        let actual = file.read_array_f64(name).unwrap();
        assert_eq!(actual.shape(), expected.shape());
        identical(actual.values(), expected.iter().copied());
        eprintln!("real field {name}: {} exact float64 words", actual.len());
    }
    let shape = file.variable("theta").unwrap().shape();
    let selection = NcSliceInfo { selections: vec![
        NcSliceInfoElem::Index(0),
        NcSliceInfoElem::Slice { start: 1, end: shape[1] as u64, step: 2 },
        NcSliceInfoElem::Slice { start: 0, end: shape[2] as u64, step: 2 },
    ] };
    let expected = baseline.read_variable_slice_as_f64("theta", &selection).unwrap();
    let actual = file.read_array_f64_slice("theta", &selection).unwrap();
    assert_eq!(actual.shape(), expected.shape());
    identical(actual.values(), expected.iter().copied());
    eprintln!("real strided theta: {} exact float64 words", actual.len());

    let empty = NcSliceInfo { selections: vec![
        NcSliceInfoElem::Index(0),
        NcSliceInfoElem::Slice { start: 3, end: 3, step: 1 },
        NcSliceInfoElem::Slice { start: 0, end: u64::MAX, step: 1 },
    ] };
    let actual = file.read_array_f64_slice("theta", &empty).unwrap();
    assert_eq!(actual.shape(), &[0, shape[2]]);
    assert!(actual.is_empty());
    let expected = baseline.read_variable_slice_as_f64("theta", &NcSliceInfo {
        selections: vec![NcSliceInfoElem::Index(0),
            NcSliceInfoElem::Slice { start: 0, end: u64::MAX, step: 1 },
            NcSliceInfoElem::Slice { start: 0, end: u64::MAX, step: 1 }],
    }).unwrap();
    let actual = file.read_array_f64_first_record_or_all("theta").unwrap();
    identical(actual.values(), expected.iter().copied());
    eprintln!("real first-record theta: {} exact float64 words", actual.len());
}
