# Source coordinates for catchment weights

`forcing-zarr-coordinates REQUEST.json` is an optional, setup-only Rust/Rayon
command. It decodes sealed local **coordinate chunks**, preserving their exact
float64 words and index order. Coordinate names come from the request; there
is no AORC adapter. It does not read weather fields, fetch a hydrofabric,
construct catchment weights or change the resident forcing ABI.

Set `CARGO_HOME`, `CARGO_TARGET_DIR` and `TMPDIR` to owned scratch, then build:

```bash
cargo build --offline --locked --release --manifest-path tools/nextgen_forcing/Cargo.toml \
  --features zarr-coordinates --bin forcing-zarr-coordinates -j 8
RAYON_NUM_THREADS=8 forcing-zarr-coordinates REQUEST.json
```

The feature is off by default. Complete selected MIT/BSD decoder notices are
in `ZSTD-NOTICES/`; versions and source archive checksums are in `Cargo.lock`.
Zstandard explicitly offers a BSD option, selected here.

The request shape is below. Replace paths/hashes with actual staged inputs;
the example's two hashes mean exactly two source chunks, not a default:

```json
{
  "schema": "nextgen.forcing.zarr-coordinates-request/v1",
  "metadata": "/owned-scratch/store.zmetadata",
  "metadata_sha256": "SHA256_OF_CONSOLIDATED_METADATA",
  "source_store": "https://provider.example/store.zarr",
  "chunks_root": "/owned-scratch/coordinate-chunks",
  "output_dir": "/owned-scratch/fresh-coordinate-export",
  "axes": [{
    "name": "latitude",
    "bounds": "midpoint_extrapolated",
    "chunk_sha256": ["SHA256_OF_CHUNK_0", "SHA256_OF_CHUNK_1"],
    "selections": []
  }]
}
```

Local files are `chunks_root/AXIS/0`, `.../1`, etc. Supported: consolidated
Zarr-v2 metadata, rank-one C-order `<f8` coordinates compressed with Zstd,
no filters or nonidentity CF packing, named dimensions and units. Coordinates
must be finite, nonmissing and strictly monotone, with at least two centres.
The final source chunk retains the full declared extent; padding outside the
array shape is discarded. All chunk hashes and the metadata hash are checked
before use. Decompression is bounded by the declared extent. Unsupported
encodings fail. Limits: 16 axes, 1,000,000 values per axis/chunk.

Outputs are raw little-endian float64 `AXIS.f64le` and a manifest with shape,
hashes, original attributes, chunk provenance, source order and spacing.
Optional selections have `name`, `min_inclusive` and `max_inclusive`; the
manifest records matching contiguous source-index ranges without resampling.
Descending source order is preserved. Output must be a fresh directory.

`bounds="none"` exports centres only. Explicit `midpoint_extrapolated` also
writes C-order float64 `[n,2]` bounds: adjacent-centre midpoints, with half
spacing extrapolated outside. They retain the centres' direction and are
marked **inferred, not provider-published**. If a bounds variable is declared,
inference fails: that native variable needs a reader. No CRS is inferred
from names or units. Hydrofabric owns CRS, geographic cell areas, intersection,
local subset/index ordering and the native sampling convention. This export
does not qualify inferred bounds or geographic weights as native inputs.

Actual January 2023 AORC metadata SHA:
`eec224d45eb7621ff8f13621f22c733be19c491389d55ab3ceda62530ef76e09`,
from `https://noaa-nws-aorc-v1-1-1km.s3.amazonaws.com/2023.zarr/.zmetadata`.
Only 127 coordinate chunks (111,453 compressed bytes) were fetched; no weather
chunks or fabric. Actual latitude/longitude are ascending, lengths 4,201/8,401.
Time contains 8,760 hourly epoch-second values. Rust selects `[97,385)` for
the 288 endpoint stamps January 5 01Z through January 17 00Z, matching the
current referee event; inclusive discharge samples use `[96,385)`, 289 points.
This is coordinate evidence, not native model replay or an event weather packet.
The source metadata declares no spatial bounds or CRS; midpoint bounds require
hydrofabric/native qualification.

Focused tests cover source words/order, descending axes, padded edge chunks,
bounds and selections; bad hashes/extents, duplicate coordinates, packing and
declared provider bounds must fail. With scratch environment set:

```bash
RAYON_NUM_THREADS=8 cargo test --offline --locked --release \
  --manifest-path tools/nextgen_forcing/Cargo.toml --features zarr-coordinates \
  --bin forcing-zarr-coordinates -j 8
```

The optional [packed-int16 subset command](ZARR-SUBSET.md) now supplies local
weather transport into the NetCDF preparation door. It preserves these source
coordinates and requires matching metadata seals; it infers no geographic CSR. Geographic CSR, complete catchment geometry, native sampling,
parameters, warmup, reservoir controls and initialized states are still needed.
