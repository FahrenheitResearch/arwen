WRF v4.8 urban generation and independent column checks

The opt-in `wrf_480` generation carries vegetated ground, root uptake and tree
flux feedback through the existing urban runtime. The default remains
`wrf_471`. Qualification is recorded in the lane report. A native column pass
is not a GPU or whole-forecast qualification.

On the authorized CPU box, set T to the lane source and L to its owned root,
source /work/env.sh, and disable local GPU visibility for CPU checks.

```sh
python tools/urban_wrf480_oracle/build.py --sources "$L/sources" --build "$L/build" --receipt "$L/source-build.json"
python tools/urban_wrf480_oracle/build_wrappers.py "$L/pristine-wrapper"
python tools/urban_wrf480_oracle/defined_state.py "$L/sources/v4.8.0/phys/module_sf_urban.F" "$L/defined/module_sf_urban.F"
python tools/urban_wrf480_oracle/transcribe.py "$L/sources/v4.8.0/phys/module_sf_urban.F" "$L/urban480-final.cuh"
python tools/urban_wrf480_oracle/compile_generated.py "$L/urban480-final.cuh"
python tools/urban_wrf480_oracle/grade_full.py --listing "$L/urban480-final.cuh" --csv tests/data/oracles/urban/ucm480/nlcd-tt0.csv.gz --tables gpuwm/data/urban/wrf480 --out "$L/column-grade.json" --switch groption=0
python tools/urban_wrf480_oracle/source_contract.py --receipt "$L/runtime-contract.json"
```

The structural contract grades state/table/control registration, not numerical
answers. `grade_full.py --gpu` uses the production shared no-FTZ loader and the
same source listing. GPU checks require the box card queue. The checked source
reference and default arithmetic are measured separately.

The pristine release sources stay unchanged. `defined_state.py` produces a
separate reference with explicit corrections for undefined entry state and the
unassigned bottom moisture output. Its diff and hash declare every correction.
Signaling-NaN initialization and floating-point traps remain enabled in the
checked reference. Undefined upstream words are never an exact reference.
The reference-only rounding wrapper is separately declared in receipts; the
stock host-library result is retained independently.

`transcribe.py` preserves source loops and explicitly rounded operation order.
It emits only physics statements. WOOF's own float32 math routines supply its
math, and no external math source is vendored. `--trace` emits an isolated
scalar/array diagnostic listing; that listing is not the production kernel.

`check.py` retains the root/water helper checks and one-bit root mutation.
`cpu_modes.py` grades each state and diagnostic at each carried step across
source selectors, moisture bounds and temperature regimes. `plant_full.py`
plants a one-bit error in the new FLXHUMVG term and grades the resulting full
column against unchanged recorded reference words.

`build_wrappers.py` compiles the pristine release Noah wrapper and the exact
pinned Noah-MP urban wrapper with recorded providers, flags and hashes.
`make_noah_driver.py` extends the existing WRF-facing driver for the eight new
state arguments. A successful wrapper compile or recording is not a coupling
parity claim; the report states which comparisons were completed.

Runtime controls are `[shared] urban_ucm_generation`, `urban_treeoption`,
`urban_ttscheme`, `urban_fvg` and `urban_table_dir`. Defaults of -1 retain the
selected table control. The real import and raw-wrfinput CLI expose these
controls. Table SHA and packed coefficients bind the new restart identity.
The pinned standard and LCZ v4.8 tables are bundled. The table-only APIs also
retain explicit `wrf_461`, `wrf_470` and `wrf_471` source packs.

TS_SCHEME=2 with TREEOPTION=1 is refused because the upstream arm does not
solve vegetated-ground heat/water and reads undefined ALPHAVG/ETAG. NUDAPT
morphology and distributed drag retain their concrete calculation refusals.
The new generation with Noah mosaic is refused because that coupling does not
carry its eight new state fields and flux feedback.

`census-dispositions.csv` retains exact hunk IDs, ranges and payload hashes.
Each disposition states implementation and qualification limits separately.