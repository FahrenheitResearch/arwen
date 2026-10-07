# Native ideal-fire initialization controls

`extract_geometry.py` byte-extracts the vertical-coordinate, hybrid,
mountain and gradient blocks from WRF v4.7.1's ideal-fire initializer.
It also extracts `process_soil_ideal` and its original helper routines.
The harness compiles original and corrected soil controls separately.

Sixty-five controls grade 36,525 output words with zero differences and
0 ULP on the RTX 5090: four vertical-grid choices, all four native hybrid
options, three heights/counts, hills/ridges on anisotropic refined grids,
native boundary gradients, coordinates, five soil geometries, and the
terrain-interpolation edge correction. The ridge controls also retain
zero widths on their unused coordinate axis. The exact source hashes and
per-field grades are recorded beside the fixtures.

Two original defects are retained as negative controls. The RUC depth
helper overwrites its first midpoint boundary and makes its second soil
thickness include the first interval. Preserving that boundary fixes one
thickness word for each of the six- and nine-level tables. Its native
forecast does not use DZS, but the output geometry must be consistent.
The original terrain interpolation leaves half-cell edge strips unwritten;
changing the input fill changes those outputs. Continuing the coarse
terrain before calling the original interpolation initializes every strip.
Both corrections are default behavior.

Reproduce with `build_geometry.sh WRF_SOURCE_ROOT NATIVE_ORACLE_BUILD OUTPUT`,
then `python -m tools.sfire_coupled_ideal.initialization.pack_geometry OUTPUT
CORPUS`. Run `tests/test_sfire_ideal_geometry_wrf471_parity.py` on the GPU.
The atmospheric sounding, hydrostatics, winds and bubble use separate
compiled initializer controls.

The complete original `landuse_init` routine supplies another 34 controls
and 21,420 output words with zero differences and 0 ULP on the RTX 5090.
They cover seasons, hemispheres, snow, sea ice, monthly albedo, category
rounding and reuse of its saved lookup state. The original ideal driver
sets `ISWATER=0`; a zero no-data category then indexes table row zero.
An original bounds-checking control fails. The corrected caller substitutes
the configured ideal land-use category for no-data cells and produces
valid surface and vegetation fields. Positive categories retain original
behavior. Native table limits still apply to the configured fallback.
Reproduce these controls with `build_landuse.sh` and `pack_fixtures.py`.
The packaged USGS section matches the retained 793 native binary32 table
values word for word; its complete file differs outside that section.

Four additional controls compile the exact native three-level ground
temperature assignment. The corrected moisture caller applies it per
column to current temperature and dry potential temperature when no
surface or land model supplies diagnostics. Vapor uses the lowest mass
level and pressure uses the ground interface. All 1,008 words match the
compiled reference on both GPUs. The atmosphere and heat/moisture exchange
fluxes remain byte-identical. This supplies physical moisture inputs in a
configuration where original WRF retains an uninitialized zero T2.
