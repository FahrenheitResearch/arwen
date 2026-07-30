# WRF v4.6.1 radiation oracle and RRTM coefficients

These files are exact copies from the provenance-bound WRF v4.6.1 source
tree. The table below records canonical repository-relative paths; a
machine-local staging path is not part of the authority.

| Packaged file | Canonical relative path | SHA-256 |
|---|---|---|
| `RRTM_DATA` | `run/RRTM_DATA` | `45dc91514cf018e133301a1b667deeb17f5b65c49ccfaa9140761a8028bc8ae0` |
| `RRTMG_LW_DATA` | `run/RRTMG_LW_DATA` | `bcfdee24b63a4c909522a329b8e16c539f0173c7e5aea2caf933ab4fe28c5c97` |
| `RRTMG_SW_DATA` | `run/RRTMG_SW_DATA` | `a7d25f5b4d33be8629cbef7ecacc1ff413bf398a021297793e843ba1cc627baf` |
| `ozone.formatted` | `run/ozone.formatted` | `bcd5316d213625cf28805a83dd6703cdebc836422c9c89ae7ec0fccf4d0c4389` |
| `ozone_lat.formatted` | `run/ozone_lat.formatted` | `00ca601bf61c40ae06fccb692ff3d1ab98a60be581892ec2c8a168d58512aadb` |
| `ozone_plev.formatted` | `run/ozone_plev.formatted` | `5c9cc9855b12f699f25213a351f8cb7526dcb31b784cdddd9a951d2578b56b3a` |
| `module_ra_rrtm.F` | `phys/module_ra_rrtm.F` | `07ebd4e83b2a571837a27748a29ca678440c363f96a3694fc9549ae0911335da` |
| `module_ra_sw.F` | `phys/module_ra_sw.F` | `87597a93e4121e26a4d365f41510fa9a5b52fd019d8b8b3c308ef598813e9248` |
| `LICENSE-WRF.txt` | `LICENSE.txt` | `87a6e3a357ae54b081b9afca2fcd86080c1584d7907fedcb4a18634d3ac2843a` |

`RRTM_DATA` is a big-endian, sequential-unformatted Fortran file with 16
records. `gpuwm.core.rrtm.load_rrtm_raw_tables` validates both record markers,
the exact declared array shapes from `module_ra_rrtm.F`, coefficient
finiteness/non-negativity, the absence of trailing bytes, and the pinned file
digest before exposing native FP32 arrays.

`RRTMG_LW_DATA` and `RRTMG_SW_DATA` are the big-endian,
sequential-unformatted coefficient files that WRF's FP32 (`RWORDSIZE=4`)
legacy RRTMG build (`ra_lw_physics = 4` / `ra_sw_physics = 4`) reads through
`rrtmg_lwlookuptable` / `rrtmg_swlookuptable` (16 and 14 band records; the
`_DBL` variants are not used by that build).
`gpuwm.ingest.rrtmg_coeffs.load_rrtmg_lw_coefficients` /
`load_rrtmg_sw_coefficients` validate the pinned digests, parse the records
against the exact `rrlw_kg01..16` / `rrsw_kg16..29` declarations, and
reproduce WRF's init-time g-point reduction bit-for-bit.
`rrtmg_coeffs_oracle_manifest.json` pins the SHA-256 of every resulting
array as dumped from the unmodified WRF Fortran by
`tools/rrtmg_wrf461_oracle/` (see that directory's build.sh; dump digests
are recorded inside the manifest under `dumps`).

`ozone.formatted`, `ozone_lat.formatted`, and `ozone_plev.formatted` are
WRF's CAM monthly zonal ozone climatology (VOLUME mixing ratio on 59
pressure levels x 64 latitudes x 12 months — the values are consumed by
RRTMG as vmr verbatim, with no amdo or 1e-6 conversion anywhere on the
o3input=2 path) exactly as `phy_init` reads them for
`o3input = 2` (the Registry default consumed by the legacy RRTMG option-4
radiation driver).  `gpuwm.ingest.wrf_ozone` validates the pinned digests
and reproduces WRF's read plus the radiation driver's time and pressure
interpolation (`ozn_time_int` / `ozn_p_int`) bit-for-bit against a compiled
control from the unmodified Fortran.

`module_ra_sw.F` is retained as the line-level oracle for the Dudhia
transcription in `gpuwm/core/dudhia.py`. The Fortran sources are provenance
oracles, not runtime-compiled code.

`rrtmg_lw_statics.npz` (SHA-256
`edd2508db89180667b0f80b4cdd991f4aa1447e711bbfe3a6db8de5fdb778d62`)
packages the 13 compile-time DATA arrays of WRF v4.6.1's
`phys/module_ra_rrtmg_lw.F` — `totplnk`, `totplk16`, `delwave`, `ngb`
(rrlw_wvn), `preflog`, `tref`, `chi_mls` (rrlw_ref), and `absice0`,
`absice1`, `absice2`, `absice3`, `absliq0`, `absliq1` (rrlw_cld) —
6,129 float32/int32 values total, exactly as the Fortran modules hold
them. They are part of the algorithm, not of the `RRTMG_LW_DATA`
coefficient file, so they are packaged separately. Provenance chain:
WRF v4.6.1 module DATA statements -> module state of the compiled
UNMODIFIED Fortran, dumped as the LW oracle fixture `lw_coeffs.bin`
(SHA-256
`3d01af85e255fbe02881494eea2797fb9d43e33bee416618483fe24628f1172a`,
written by `tools/rrtmg_wrf461_oracle/lw_extract.F90` + `lw_binio.F90`
via `lw_build.sh`, gfortran FP32 build) -> packaged bit-exact as this
npz (one C-ordered little-endian member per array). Regenerate with
`python tools/rrtmg_wrf461_oracle/lw_statics_package.py [fixdir]`
(fixdir defaulting to `GPUWM_RRTMG_LW_FIXTURES`).
`gpuwm.core.rrtmg_legacy._lw_static_tables` enforces the pinned file
digest plus the exact member roster/order, shapes, and dtypes, and
`tests/test_rrtmg_legacy_wiring.py` re-gates every member bitwise
against `lw_coeffs.bin` whenever the LW oracle fixture deck is present.

The WRF distribution license copied as `LICENSE-WRF.txt` applies to the WRF
source and data copies in this directory.
