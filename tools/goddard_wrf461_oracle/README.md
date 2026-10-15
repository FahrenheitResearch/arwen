# New Goddard radiation qualification

The paired (5,5) registry row runs the WRF 4.6.1 Goddard driver, both bands,
cloud optics, heating and the surface/TOA fluxes published by that driver.
A bare imported namelist selecting 5/5 uses the scheme. The driver uses
binary32. WRF fp_kind in the private band solvers is binary64; both engine
arithmetic modes retain that precision. Every transcendental uses WOOF's own
math routines. GPUWM_WRF_EXACT=1 selects the strict operation-order arm.

The public source is WRF 4.6.1 commit d66e442fccc04111067e29274c9f9eaccc3cef28.
The 64-column, 32-layer driver oracle covers clear sky, low cloud, deep ice,
cirrus, polar night, low sun, terrain and edge cases. It compares 5,056 output
words in 17 fields. Strict qualification requires zero ULP. Default limits are
0.1 W/m2 plus 1e-4 relative for fluxes, and 1e-7 K/s plus 1e-4 relative for
heating. Separate one-bit SW and LW flux mutations must fail this comparison.
A saved real 50-layer column additionally checks dynamic solver storage.
The solver tier define precedes the private header's storage declarations.

Declared source divergences: stock swrad reads unassigned taux scratch state
in its fast-overcast reduction. The port defines taux from visible-band cloud
optical depth; GODDARD_DEFINED_TAUX=1 makes the same definition in the oracle.
This does not claim identity to undefined stock scratch state. The sounding
interpolation clamps the equatorial endpoint rather than reading outside the
WRF latitude table.

The cloud table is decoded by Rust, SHA256-bound and cached once per device.
Its source SHA256 is ad4bb1dd8446a2b00559ac425e21255a01b79a4cac7b5e3e693a3c98c222a1c8.
WRF's three direct-access reads ignore the final 24 source bytes; so does the
decoder. The small oracle fixtures have a .keep file for reproducibility.
Aerosol qualification is aer_opt=0 with WRF_CHEM=0. No aerosol NetCDF arm is
reached or admitted. The six RE_*_GSFC inputs and non-7 radius fallbacks are
covered by focused tests; the base engine does not admit mp_physics=7 itself.
Stock goddardrad does not publish SWDOWNC/SWDDNIC carriers. Its clear-column
and night branches are covered without inventing those output fields.

Build and test on the assigned CPU box:

```sh
GODDARD_DEFINED_TAUX=1 bash tools/goddard_wrf461_oracle/build_driver.sh "$WRF_SOURCE_ROOT" /work/port-ra5-goddard/driver-build
GPUWM_NO_LOCAL_GPU=1 python -m pytest -q --noconftest tests/test_goddard_column_check.py tests/test_goddard_driver.py
```

On an assigned GPU box, inside its card queue, run run_driver_gpu.py in strict
and default mode, then with --mutate-band 1 and --mutate-band 2. The mutation
runner succeeds only when it detects the planted error. Run run_forecast.sh
for the public namelist fixture and hash_forecast.py on its native outputs.
Render with gpuwm.cli render --engine rust using var:wrf_swdown,var:wrf_glw;
rw_compare produces the two-time radiation sheet. Compare all existing 4/4
forecast fields before and after this series with hash_forecast.py.

The final mechanical gates run where their receipts were produced, without
copying data between boxes:

```sh
python tools/goddard_wrf461_oracle/assess_qualification.py --stage driver --receipts "$LANE_ROOT"
python tools/goddard_wrf461_oracle/assess_qualification.py --stage forecast --receipts "$LANE_ROOT"
```

Both partitions must pass. --stage all can check a colocated run. The older
assess_runs.py describes only the earlier private-band/cloud stage; its
full_port_complete=false is deliberately not a final driver verdict.
