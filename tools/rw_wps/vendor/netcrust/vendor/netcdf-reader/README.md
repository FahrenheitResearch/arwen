# netcdf-rust

Pure-Rust, read-only decoders for HDF5 and NetCDF. No C libraries or build scripts; internal `unsafe` is limited to read-only memory mapping and performance-critical decoding/copy paths.

## Crates

| Crate | Description |
|---|---|
| `hdf5-reader` | Low-level HDF5 decoder (superblock, object headers, B-trees, chunked I/O, filters) |
| `netcdf-reader` | NetCDF reader supporting CDF-1/2/5 classic and NetCDF-4 (HDF5-backed) formats |

## Usage

```rust
use netcdf_reader::{NcFile, NcSliceInfo, NcSliceInfoElem};

let file = NcFile::open("era5.nc")?;
println!("format: {:?}", file.format());

for var in file.variables()? {
    println!("  var: {} {:?}", var.name(), var.shape());
}

// Read typed data (works for both classic and NetCDF-4)
let temp: ndarray::ArrayD<f32> = file.read_variable("temperature")?;

// Type-promoting read (any numeric type → f64)
let data = file.read_variable_as_f64("temperature")?;

// String variables (classic char arrays and NetCDF-4 NC_STRING)
let names = file.read_variable_as_strings("station_name")?;

// CF conventions: unpack packed integer data (scale_factor + add_offset)
let unpacked = file.read_variable_unpacked("temperature")?;

// CF conventions: mask fill values + unpack in one call
let clean = file.read_variable_unpacked_masked("temperature")?;

// Hyperslab: read a single time step from a 4D variable
let sel = NcSliceInfo {
    selections: vec![
        NcSliceInfoElem::Index(0),                                        // time=0
        NcSliceInfoElem::Slice { start: 0, end: u64::MAX, step: 1 },     // all levels
        NcSliceInfoElem::Slice { start: 0, end: u64::MAX, step: 1 },     // all lat
        NcSliceInfoElem::Slice { start: 0, end: u64::MAX, step: 1 },     // all lon
    ],
};
let step: ndarray::ArrayD<f32> = file.read_variable_slice("temperature", &sel)?;

// Lazy iteration over time steps
for slice in file.iter_slices::<f32>("temperature", 0)? {
    let data = slice?;
    println!("  step shape: {:?}", data.shape());
}

// In-memory open with custom NC4 cache/filter options
let bytes = std::fs::read("era5.nc")?;
let file = NcFile::from_bytes_with_options(&bytes, netcdf_reader::NcOpenOptions {
    chunk_cache_bytes: 8 * 1024 * 1024,
    chunk_cache_slots: 257,
    metadata_mode: netcdf_reader::NcMetadataMode::Strict,
    #[cfg(feature = "netcdf4")]
    filter_registry: None,
})?;
```

Using `hdf5-reader` directly:

```rust
use hdf5_reader::Hdf5File;

let file = Hdf5File::open("data.h5")?;
let ds = file.dataset("/group1/temperature")?;
let data: ndarray::ArrayD<f64> = ds.read_array()?;

// Hyperslab selection
use hdf5_reader::{SliceInfo, SliceInfoElem};
let sel = SliceInfo {
    selections: vec![
        SliceInfoElem::Slice { start: 0, end: 10, step: 1 },
        SliceInfoElem::Index(5),
    ],
};
let slice: ndarray::ArrayD<f64> = ds.read_slice(&sel)?;

// String datasets
let labels = file.dataset("/labels")?.read_strings()?;
```

## Features

**HDF5**
- Superblock v0-v3 and object header v1/v2 with checksum verification
- Compact, contiguous, and chunked layouts
- All chunk index types: v1/v2 B-tree, single-chunk, implicit, Fixed Array, Extensible Array
- Deflate, shuffle, Fletcher-32, N-Bit, ScaleOffset, and optional LZ4 filters
- Custom filters via `FilterRegistry`
- Fixed-length strings, HDF5 variable-length strings, and byte-vlen string datasets
- Dense-link resolution, soft-link resolution, committed datatypes, global heap strings, and object references
- Parallel chunk decoding, chunk caching, and object-header caching
- Range-backed opens via `Storage` backends (`BytesStorage`, `FileStorage`, `MmapStorage`)

**NetCDF**
- CDF-1, CDF-2, CDF-5, and NetCDF-4
- Automatic format detection
- Unified typed reads across formats
- Unified string reads for classic char arrays and NetCDF-4 string variables
- Type promotion to `f64`, unpacking, masking, and combined CF helpers
- Slice reads, lazy slice iteration, and parallel NC4 slice reads
- Cache and filter configuration through `NcOpenOptions`, including in-memory and storage-backed opens

## Feature flags

```toml
[dependencies]
netcdf-reader = "0.3"           # CDF-1/2/5 + NetCDF-4 (default)
netcdf-reader = { version = "0.3", default-features = false }  # CDF-1/2/5 only
```

| Flag | Default | Description |
|---|---|---|
| `netcdf4` | yes | NetCDF-4 support via `hdf5-reader` |
| `rayon` | yes | Parallel chunk reading |
| `lz4` | yes | LZ4 filter support (hdf5-reader) |
| `cf` | no | CF Conventions helpers (axis identification, time decoding, CRS extraction, bounds) |

## Custom filters

Register filters before opening files:

```rust
use hdf5_reader::{Hdf5File, OpenOptions};
use hdf5_reader::filters::FilterRegistry;

let mut registry = FilterRegistry::new();
registry.register(32001, Box::new(|_filter, data, _elem_size| {
    // Custom decompression logic
    Ok(data.to_vec())
}));

let file = Hdf5File::open_with_options("data.h5", OpenOptions {
    filter_registry: Some(registry),
    ..Default::default()
})?;
```

## Testing

```sh
# Unit tests (no external dependencies)
cargo test --workspace

# Integration tests with generated fixtures
scripts/generate-fixtures.sh
cargo test --workspace
```

For reference comparisons and current benchmark results against
`georust/netcdf`, see [docs/benchmark-report.md](docs/benchmark-report.md).

## Releasing

See [RELEASING.md](RELEASING.md) for the release checklist and the required
publish order for `hdf5-reader` and `netcdf-reader`.

## Known limitations

- External HDF5 links are skipped (soft links are resolved)
- SZIP is not built in (register via `FilterRegistry` if needed)
- ScaleOffset floating-point E-scale mode is not supported by the HDF5 decoder path
- SOHM (shared object header message table) resolution returns a descriptive error
- Fractal heap huge/tiny objects are not yet supported (managed objects work)
- CF time decoding uses Gregorian approximation for non-standard calendars (noleap, 360_day)

## License

MIT OR Apache-2.0
