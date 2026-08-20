# MPAS to wrfout render bridge — SUPERSEDED, reference generator only

**These two scripts are not a product path.** The Rust crate
`tools/rustwx/crates/rw-mpas` (binary `rw_mpas_convert`) owns this route on the
release line. Convert with it:

```
rw_mpas_convert --history HIST.nc [HIST2.nc ...] --mesh MESH.nc --out-dir OUT \
    [--init INIT.nc] [--window focus|global] [--field-set full|surface] \
    [--format cdf2|cdf5] [--json REPORT.json]
```

(built from `tools/rustwx`: `cargo build --release -p rw-mpas`)

A bare run of either Python script refuses with exit code 78 and prints that
command. The breakage the refusal names: this pair reads an MPAS history file,
regrids it and writes the NetCDF frame the renderer draws — read, regrid and
write are all Rust-only under the 2.5 Python boundary, so running it puts
data-path Python back on the render route and produces frames a reader would
take for products.

They are kept, rather than deleted, because
`evidence/rw-mpas-converter-parity.json` is a receipt **about them**: the Rust
crate reproduces this exact pair bit for bit (identical weights digests on both
windows, 404,772,334 float values compared, 0 differing; 127 of 127 rendered
PNGs byte-identical). Delete the reference and that claim stops being
reproducible. Pass `--regenerate-reference` to either script when you are
regenerating that reference. Frames produced that way are fixtures, never
products.

`--print-field-map` reads no data and stays reachable without the
acknowledgement.

---

Two scripts that let the pinned Rust renderer (`rw_wrfbatch`) draw MPAS output.
The renderer has no unstructured path: its real named products, with their
operational colortables, titles and map furniture, come from the real-WRF import
route, which needs a structured grid, `Times`, `XLAT`/`XLONG` and a rank-4 `T`.
These two scripts produce exactly that from an MPAS history file.

They are the Python side of the bridge. The Rust crate `tools/rustwx/crates/rw-mpas`
is the productionised path; this pair is the working reference that path is
checked against, and it is what produced the render-bridge evidence in
`evidence/mpas-render-bridge/`. Since the demotion above, that reference role is
the *only* role: the usage below runs only with `--regenerate-reference`.

| file | role |
| --- | --- |
| `mpas_resample_weights.py` | one-time, per-mesh. Builds and caches the nearest-cell index map from mesh cells onto structured target windows. Reads no forecast field and touches no GPU. |
| `mpas_history_to_wrfout.py` | per-frame. Gathers every MPAS field that has a WRF product equivalent onto a cached window and writes a wrfout-shaped NetCDF frame. |

`mpas_history_to_wrfout.py` imports `mpas_resample_weights` from its own
directory, so the two files must stay side by side.

## Usage

```
python mpas_resample_weights.py --regenerate-reference \
    --mesh x1.40962.static.nc --cache-dir CACHE \
    --windows global,focus --verify --json weights.json

python mpas_history_to_wrfout.py --regenerate-reference \
    --history HIST.nc [HIST2.nc ...] \
    --mesh x1.40962.static.nc --cache-dir CACHE --window focus \
    --field-set full --out-dir OUT --json convert.json
```

Build the weights cache once per mesh, then run the converter per frame against
that cache. `--print-field-map` lists the MPAS-to-WRF field correspondence
without reading a forecast.

## Regridding method: nearest cell centre

Nearest cell centre on the sphere, by 3-D chord distance through a k-d tree over
unit vectors. **This is a deliberate choice, not a convenience.**

A resampled value is exactly the value the model carried in that cell, so
extremes survive the transfer unchanged: a 70 dBZ core stays 70 dBZ, a 35 m/s
gust stays 35 m/s. Every smoothing scheme, inverse-distance included, pulls
maxima down and pushes minima up, which is the wrong trade for meteorological
products whose whole point is the tail of the distribution.

Validated in `evidence/mpas-render-bridge/regrid-validation.json`:

- **No new values.** 36,000 of 36,000 output values are bit-identical to some
  source cell value. Zero points not matching.
- **Spike survival.** A planted single-cell 70.0 spike comes out at exactly
  70.0, amplitude loss 0.0, with **zero** target points strictly between 0 and
  the amplitude. A smoothing interpolant would put a nonzero count in that row.
- **Smooth-field error** sits under the predicted `|grad f| x max nearest-cell
  arc` bound at every one of the 36,000 points.

Neither this scheme nor its planned successor is conservative. **Nothing
produced through this path is a conservative remap and nothing here supports an
accumulation-budget claim.**

### Stated resolution limit

The target grid is finer than the mesh that feeds it, and regridding creates no
information. For the validated configuration:

| quantity | value |
| --- | --- |
| mesh | x1.40962 quasi-uniform global, 40,962 cells |
| mean cell spacing | 111.6 km (~112 km between cell centres) |
| target grid | Lambert 240x150 at dx = 22 km (the `focus` window) |
| mean nearest-cell distance | 41.9 km |
| max nearest-cell distance | 69.8 km |

Each MPAS cell paints a contiguous patch of target pixels roughly 120 km across.
A rendered field is therefore **blocky at cell scale — the visible hexagons are
the mesh cells, not an artifact — and nothing smaller than about two mesh cells
(~240 km) is represented at all.** Read a product from this path as a synoptic
field on a 22 km canvas, never as 22 km detail.

### Deliberately not implemented

Barycentric interpolation on the Delaunay dual of the cell centres. That is the
right choice for smooth fields (geopotential height, MSLP) where nearest-cell
leaves visible hexagon facets, and it is second-order accurate where
nearest-cell is zeroth-order. It needs a triangulation plus per-target
barycentric coordinates, so it is a strictly larger cache with the same
interface. When it lands it should be selectable **per field, not per file**:
nearest for anything judged on extremes, barycentric for anything judged on
gradients.

## What the converter will not do

It will not invent a field the MPAS history does not carry. A WRF variable with
no MPAS source is simply absent, its products report `missing-fields`, and the
absence is recorded in the emitted file's `MPAS_ABSENT_WRF_FIELDS` attribute and
in the run report. **A missing field is a port history-stream gap to fix
upstream, never something to synthesize here.**

Every emitted file carries the source history digest, the mesh digest, the
weights digest, the resample method and the tool's own digest.

## Provenance

Committed 2026-08-14 from the validated working copies. Until then these two
files existed only as loose copies outside any repository; this is their first
git home. Schemas: `mpas-port.nearest-cell-render-weights/v1` and
`mpas-port.history-to-wrfout-render-frame/v1`.
