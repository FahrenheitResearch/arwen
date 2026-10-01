# WPS terrain-smoother oracle, WPS v4.6.0

The reference for `[[domain]] static = { smooth_option, smooth_passes }`
(`gpuwm/static/terrain_smoothing.py`): WPS's own geogrid smoothers, run
byte-unmodified, against which the Python reference and the Rust
static-fields entry point are held bit for bit.

WPS v4.6.0 is the WPS release paired with WRF v4.7.x (there is no WPS 4.7).
The tree is pinned to the tag's commit
`335c76a111f84503e8b963abaf273ea8053645bb`; both scripts refuse any other
commit or a modified source.

| file | sha256 |
|---|---|
| `geogrid/src/smooth_module.F` | `a95b265bd5f93309285303d6c22a1e2e0923a59d58b5e8729090a19e39351336` |
| `geogrid/src/parallel_module.F` | `d0a09c3099977edb54efa6982a59c9329e3ff287bdb0434e7fca53931de67558` |
| `geogrid/GEOGRID.TBL.ARW` | `dd1237f317ed5162d03cd289516e8bbd9e88375f053eb7093e26456a20b10a33` |

## Two fixtures

**Operator fixture** `tests/fixtures/wps_smooth_v460/wps_smooth_v460.npz`:

```
bash build.sh <WPS_SOURCE_ROOT> <BUILD_DIR>
python make_fixture.py <BUILD_DIR> wps_smooth_v460.npz --real NAME=PLANE.npy ...
```

`build.sh` compiles `smooth_module.F` and `parallel_module.F` with WPS's own
GNU flags (`-ffree-form -O -fconvert=big-endian -frecord-marker=4`, cpp
`-P -traditional -D_UNDERSCORE -DBYTESWAP -DLINUX -DIO_NETCDF -DBIT32`; no
`-r8`, so default REAL is single precision; no `-march`) and refuses objects
holding a fused multiply-add.  `-D_MPI` is left out: on one process
`exchange_halo_r` is a no-op in the MPI build too.  `driver.F90` calls
`one_two_one`, `smth_desmth` and `smth_desmth_special` on float32 planes
with the bounds process_tile_module.F:884-914 uses (the halo-extended memory
array).  `make_fixture.py` runs every smoother and pass count over seeded
synthetic planes (a coast at exactly 0 for the special restore, spikes, and
values near the float32 subnormal boundary, which nothing may flush) plus,
with `--real`, crops of geogrid's own unsmoothed Alpine terrain.  The Rust
crate replays the same bytes from
`tools/rustwx/crates/static-fields/golden/lane2/wps_terrain_smoothing/`
(written by `golden/generate_terrain_smoothing_goldens.py`).

**End-to-end fixture** `tests/fixtures/wps_smooth_v460/geogrid_alps_small.npz`:

```
bash geogrid_fixture.sh <WPS_BUILD_DIR> <GEOG_ROOT> <WORK_DIR>
python make_geogrid_fixture.py <WORK_DIR> geogrid_alps_small.npz
```

`WPS_BUILD_DIR` is the pinned tree built with `./configure` (option 1, serial
gfortran) and `./compile geogrid`.  The script runs `geogrid.exe` itself on a
48 x 40 cell, 1 km Lambert domain over Alpine valleys once per setting (the
stock table with only HGT_M's smooth line changed), and once on the same
lattice widened by the 3-cell halo with no smoothing: WPS's own unsmoothed
halo-extended HGT_M, the smoother's input.  The packer refuses unless the
widened run's interior equals the unsmoothed run bit for bit and
`smooth_passes=0` equals no smoothing.
`tests/test_terrain_smoothing.py::test_geogrid_exe_identity_on_a_real_domain`
then holds every non-default setting to `geogrid.exe`'s HGT_M bit for bit.

WPS's default (one `smth-desmth_special` pass) is the one setting the engine
does not take through WPS's arithmetic by default: it keeps the static
builders' historical float64 smoother, so every default build keeps its
bytes.  That smoother is within 1.5 mm of `geogrid.exe` (measured: 1.22 mm
on the fixture domain, 1.31 mm and 1.07 mm on 1 km and 3 km Alpine domains,
1.435 mm on a 3 km Alpine GFS root); `smooth_precision = "wps-float32"`
selects WPS's float32 arithmetic for it instead, and the fixture test pins
both that bound and the option's bit-for-bit result, through the Python
reference and the Rust entry point.

## What the fixtures do not cover

The static builders' sampler (GMTED2010 30" through
`average_gcell(4.0)+four_pt+average_4pt`) is a separate port with its own
contract; the full engine build differs from `geogrid.exe` by that sampler
alone (0.50 m RMS unsmoothed on the 1 km Alpine domain), which each smoothing
setting carries through its smoother.

Last generated with GNU Fortran 15.2.0 on x86_64 Ubuntu.
