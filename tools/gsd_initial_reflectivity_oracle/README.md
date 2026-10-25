# HRRR initial-time reflectivity oracle

Authority: NOAA-EMC/HRRR
`40ee6058c2fc6624cbfbbe8cf1c20c59e6a45827`,
`sorc/hrrr_wrfarw.fd/WRFV3.9/phys/`.

Place the pinned `phys/module_diag_gsd.F`, `phys/module_mp_thompson.F`,
`phys/module_mp_radar.F` and `share/module_model_constants.F` (flat names)
in a source directory. On the desktop with WSL
Ubuntu-24.04 and gfortran installed:

```powershell
python tools/gsd_initial_reflectivity_oracle/run.py SOURCE_DIR OUTPUT_DIR
Copy-Item OUTPUT_DIR/vectors.npz tests/data/oracles/gsd_initial_reflectivity/
Copy-Item OUTPUT_DIR/receipt.json tests/data/oracles/gsd_initial_reflectivity/
$env:PYTHONPATH=(Get-Location).Path
python -m pytest -q -p no:cacheprovider tests/test_gsd_initial_reflectivity.py
```

The harness extracts `gsd_refl_first_time_step` and `calc_refl10cm`
unchanged, retains the complete native radar module, and initializes its
constants using unchanged statements from Thompson's initializer. Unused
microphysics lookup tables are not built. Framework-only stubs come from
the existing Thompson fork oracle; no diagnostic arithmetic is replaced.
The generated sources, compiled oracle executable, input text, output words and
source/extraction SHA-256 receipt stay in OUTPUT_DIR for inspection.

Sixteen calls cover Thompson 8/28, four mixed-phase profiles, nonpositive
and supplied positive rain numbers, and rain masses on both sides of the
1.e-12 fallback threshold. Half the calls additionally prepare theta,
Exner and moist density with the fork's `phy_prep` expressions
(`module_big_step_utilities_em.F:5118-5122`; Earth `rcp=r_d/cp`). Every
binary32 output word is frozen (3,456 total). The GPU test compares words,
checks every input for mutation, and writes resident/host/store-direct
initial history values through the Rust NetCDF writer. Store-direct cases
span two GPU slabs, use a one-dimensional theta base, and poison the stale
attachment state's carriers. Delayed activation and later-frame consumption
are checked too.

The GPU unit reuses `thompson_fork.cu` as a pinned device header. Its older
`thompson_fork_column` entry is compiled but never launched by this helper;
only `gsd_refl_first_time_step` is selected. The new entry's frame is 0 B
on RTX 3080 / sm_86 / NVRTC 13.0.48. Three transient global arenas replace
per-thread column arrays and are priced by preflight. IEEE subnormals are
retained; the default fast policy permits FMA. Word identity is asserted
with the battery's explicit strict numerics, not across all compilers or
complete operational forecasts.
