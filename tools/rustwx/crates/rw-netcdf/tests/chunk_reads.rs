//! A NetCDF-4 variable whose zlib chunks are larger than the chunk cache
//! is read with each chunk it needs inflated exactly once.
//!
//! NCAR's v4.4 CONUS 2.5 km restart stores each 3D field as 88 MB zlib
//! chunks, past the 64 MiB chunk cache.  The reader walked such a variable
//! in ~1M-element row slabs and inflated every chunk once per slab: about
//! 60 s for one 3D variable (netCDF-C: about 1 s) and 44 minutes for the
//! warm start.  `big-chunk.nc4` has the same shape of problem at fixture
//! size (two 68 MB chunks; see make_big_chunk_fixture.py).  One test in
//! this file, because the inflate count is per process.

use std::path::PathBuf;

use netcrust::{NcSliceInfo, NcSliceInfoElem};

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/big-chunk.nc4")
}

fn expected(k: usize, j: usize) -> f64 {
    f64::from(k as f32 + (j % 64) as f32 / 64.0)
}

#[test]
fn chunks_larger_than_the_cache_are_inflated_once_per_read() {
    let file = netcrust::File::open(fixture()).expect("open big-chunk.nc4");

    let before = netcrust::inflated_streams();
    let array = file.read_array_f64("U").expect("read U");
    let inflated = netcrust::inflated_streams() - before;
    assert_eq!(array.shape(), &[1, 17, 2000, 1000]);
    for (index, value) in array.values().iter().enumerate() {
        let (k, j) = (index / 2_000_000, (index / 1000) % 2000);
        assert_eq!(value.to_bits(), expected(k, j).to_bits(), "U flat index {index}");
    }
    assert_eq!(inflated, 2, "the full read inflated {inflated} streams for 2 chunks");

    // A window inside the first chunk needs that chunk alone.
    let window = NcSliceInfo { selections: vec![
        NcSliceInfoElem::Slice { start: 0, end: 1, step: 1 },
        NcSliceInfoElem::Slice { start: 0, end: 17, step: 1 },
        NcSliceInfoElem::Slice { start: 990, end: 1000, step: 1 },
        NcSliceInfoElem::Slice { start: 3, end: 8, step: 1 },
    ] };
    let before = netcrust::inflated_streams();
    let cut = file.read_array_f64_slice("U", &window).expect("read window");
    assert_eq!(netcrust::inflated_streams() - before, 1);
    assert_eq!(cut.shape(), &[1, 17, 10, 5]);
    for (index, value) in cut.values().iter().enumerate() {
        let (k, j) = (index / 50, 990 + (index / 5) % 10);
        assert_eq!(value.to_bits(), expected(k, j).to_bits(), "window flat index {index}");
    }
}
