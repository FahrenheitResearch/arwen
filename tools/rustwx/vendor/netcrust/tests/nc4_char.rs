//! NetCDF-4 stores NC_CHAR as one-byte HDF5 strings. `read_strings` must
//! still answer one string per row of the last axis, as it does for a
//! classic file: a WRF `Times(Time, DateStrLen)` read back as 19 one-letter
//! strings per time made rw_grib2export refuse real netCDF-4 histories
//! ("holds 19 times") and every reader that took the first string see "2".
//!
//! Fixture: tests/fixtures/nc4_char_times.nc, written by netCDF4-python
//! (format NETCDF4) with Times(Time=2, DateStrLen=19), a NUL-padded 1-D
//! char Title(DateStrLen) and an NC_STRING names(one).

use std::path::PathBuf;

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/nc4_char_times.nc")
}

#[test]
fn a_netcdf4_char_matrix_reads_as_one_string_per_row() {
    let file = netcrust::open(fixture()).expect("open NetCDF-4 fixture");
    assert_eq!(
        file.read_strings("Times").expect("Times as text"),
        vec!["2024-05-25_19:00:00".to_string(), "2024-05-25_20:00:00".to_string()]
    );
}

#[test]
fn a_one_dimensional_netcdf4_char_array_is_one_string_without_its_pad() {
    let file = netcrust::open(fixture()).expect("open NetCDF-4 fixture");
    assert_eq!(file.read_strings("Title").expect("Title as text"), vec!["short".to_string()]);
}

#[test]
fn a_netcdf4_string_variable_is_untouched() {
    let file = netcrust::open(fixture()).expect("open NetCDF-4 fixture");
    assert_eq!(file.read_strings("names").expect("names as text"), vec!["a string".to_string()]);
}
