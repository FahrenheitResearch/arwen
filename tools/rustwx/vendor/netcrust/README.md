# netcrust

WRF-focused, pure-Rust NetCDF/NetCDF4 reader facade.

Goal:

- replace the C-backed `netcdf` crate usage in WRF weather workflows
- keep the small read surface those workflows actually need
- rely on a Rust NetCDF4/HDF5 decode path instead of native NetCDF/HDF5 libs

Current implemented surface:

- `open(path)` / `File::open`
- NetCDF classic and NetCDF4/HDF5 signature helpers
- root dimensions and variables
- root/global attributes with numeric promotion helpers
- variable metadata: name, dimensions, shape, dtype
- full numeric reads promoted to `f64`
- WRF-style first-record reads for rank >= 3 variables

The reader is intentionally narrower than the full NetCDF C API. It is aimed at
WRF output and processed WRF rollups first: `wrfout_*` files, native WRF fields,
and derived weather products that only need dense numeric reads.

## Example

```rust
let file = netcrust::open("wrfout_d01_2021-02-16_00_00_00")?;

let nx = file.dimension("west_east").unwrap().len();
let ny = file.dimension("south_north").unwrap().len();
let dx = file.attribute("DX").and_then(|attr| attr.as_f64()).unwrap();

let t2 = file.read_array_f64_first_record_or_all("T2")?;
assert_eq!(t2.shape(), &[ny, nx]);
```

## Local Verification

The repository includes portable unit tests plus WRF real-file tests. The real
tests use `NETCRUST_WRF_FIXTURE` when set, otherwise they look for the small
local fixture:

`F:\250m_master\20210216_00z_tx_freeze\wrfout_d01_2021-02-16_00_00_00`

Run:

```powershell
cargo test
```

For a specific large fixture:

```powershell
$env:NETCRUST_WRF_FIXTURE="F:\250m_master\20210216_00z_tx_freeze\wrfout_d03_2021-02-16_00_00_00"
cargo test --test wrf_real -- --nocapture
```

## Scope

This crate is currently read-only. Write support and full drop-in `netcdf` API
compatibility are non-goals until the WRF read path is proven across the local
case archive.
