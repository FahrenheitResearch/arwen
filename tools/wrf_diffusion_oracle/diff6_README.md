# Sixth-order diffusion oracle

`diff6_build.py` extracts the byte-unmodified `sixth_order_diffusion` routine
from WRF v4.7.1 `dyn_em/module_big_step_utilities_em.F`, lines 6220 to 6636.
The full source hash and the exact extracted slice hash are pinned. WRF commit
`f52c197ed39d12e087d02c50f412d90d418f6186` is the `v4.7.1` tag.
WRF's own `module_model_constants.F` is compiled as well. The routine uses its
own literal `9.81` for slope thresholds, exactly as WRF wrote it.

Only the configuration service type is supplied by the wrapper. It defines the
eight members the routine reads with their WRF scalar types. The original
numerical argument list, array dimensions, staggering, loop limits, arithmetic,
and constants are unchanged. Gfortran builds at `-O0 -ffp-contract=off
-fno-tree-vectorize -fcheck=bounds`; reference REAL values are binary32.

The input is a raw Rust `rw_netcdf` export of a real WRF initial state valid at
2024-05-25 18Z. Input decoding uses Rust. Python only assembles the standalone
test cases and launches the Fortran executable. The source NetCDF file's hash
is in the fixture metadata. Edge probes include a steep ridge, map factors
from 0.25 to 2.75, zero and near-zero values, and a wind/latitude hemisphere
reversal. This routine has no latitude or Coriolis argument, so that reversal
is a symmetry probe, not a second observed southern-hemisphere state.

The real source state uses terrain coordinates. The additional hybrid
transition case retains its upper atmospheric state and supplies a coefficient
edge probe with C1 from 1 to 0 and C2 equal to `(1-C1)*mean(real MUT)`.
The periodic case supplies true three-cell periodic halos and starts with a
nonzero accumulated tendency. The other cases cycle specified, nested, and
four-sided open boundaries. There are 200 cases, 40 per u/v/w/theta/moisture
staggering, with both flux-limit and slope options and varied dt, factor, dx,
and dy. All 376160 tendency words are retained and compared as uint32.

To rebuild, first copy the two pinned WRF source files into `SOURCE`, and copy
the real input file into the build scratch. Then run on a CPU host:

```sh
nice -n 10 rw_netcdf dump --raw wrfinput_d01 dump U V W T QVAPOR MU MUB C1H C2H C1F C2F PHB MAPFAC_U MAPFAC_V MAPFAC_M XLAT
nice -n 10 python tools/wrf_diffusion_oracle/diff6_build.py SOURCE BUILD
nice -n 10 python tools/wrf_diffusion_oracle/diff6_cases.py dump BUILD tests/data/wrf471_diffusion --real-file wrfinput_d01
```

Under the host's GPU ownership protocol, run the engine's actual launcher:

```sh
PYTHONPATH=. python -m pytest -q -s tests/test_diff6_wrf471_parity.py tests/test_diff6.py tests/test_diff6_boundary_face.py
```

The compiled oracle exposed omitted tendency map factors and arithmetic that
contracted multiplies or coupled hybrid mass after averaging. The default
product now passes the field's own map factor even when slopeopt is zero,
couples every mass before averaging, and pins REAL multiplication rounding.
Host coefficient and slope-threshold calculations also use WRF's REAL order.
The two seam kernels use the same corrected arithmetic. The omitted-map
mutation produces 5723 wrong words on ten projected real cases with slopeopt
zero. The old GPU golden equality assertions were replaced by compiled WRF
replays, retaining periodic and boundary coverage.
