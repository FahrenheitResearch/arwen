# Analyzed aerosol initial and boundary fields

`mp28_aerosol_source = "analysis"` requires both water-friendly and ice-friendly
aerosol number mixing ratios on every initial and lateral boundary frame.
Missing fields stop preparation rather than substitute a climatology.
The source mapping declares `water_friendly_aerosol_number` and
`ice_friendly_aerosol_number`, both in `kg-1`. The regular source join carries
them as QNWFA and QNIFA through the existing scalar interpolation and specified
boundary operators. Cloud droplet, cloud ice and rain numbers use the same
mapping table.

`use_rap_aero_icbc = true` admits the operational namelist spelling. It selects
analyzed three-dimensional initial and boundary aerosol and retains operational
WRF's monthly two-dimensional surface source. The surface source uses the
monthly near-surface number multiplied by `0.000196 * (airmass * 2e-10)`, where
`airmass = (1/alt) * z1 * dx * dy`. It requires the monthly WIF dataset staged by
`gpuwm fetch-tables --wif`; the dataset supplies surface emissions only.
The Rust surface operator has a separate Fortran REAL comparison in
`tests/test_aerosol_analysis_input.py`.
The staged file is the engine's pinned `QNWFA_QNIFA_SIGMA_MONTHLY.dat`.
Its monthly values have not been compared byte-for-byte with the operational
deployment's `QNWFA_QNIFA_Monthly_GFS` constants file.

With a separate initial meteorological analysis, the aerosol donor keeps its
own pressure and moisture columns through vertical interpolation. The target
dry eta pressure comes from the meteorological initialization. The interpolated
aerosol is installed before the cold-start droplet number closure. Copying a
donor aerosol array onto a different source pressure ladder is not equivalent.

The operational source authority is NOAA-EMC/HRRR commit
`40ee6058c2fc6624cbfbbe8cf1c20c59e6a45827`:

- `parm/conus/hrrr_vtable:19-20` maps GRIB2 `0/13/193` and `0/13/192`, level
  type 105, to QNWFA and QNIFA in `kg-1`. Generic GRIB inventories label these
  local parameters PMTF and PMTC with mass-concentration units. The operational
  Vtable and postprocessor establish their number-mixing-ratio meaning here.
- `sorc/hrrr_wrfpost.fd/INITPOST.F:635-655` reads QNWFA and QNIFA directly;
  `MDLFLD.f:1142-1188` writes those arrays without a density conversion.
- `parm/conus/hrrr_METGRID.TBL:619-630` selects nearest-neighbor horizontal
  interpolation, then four-point and average-of-available-corners fallbacks,
  zero only if those cannot answer, and the deepest source layer for the
  surface pseudo-level. The source mapping pins the product authority and fill.
- `sorc/hrrr_wrfarw.fd/WRFV3.9/dyn_em/module_initialize_real.F:2125-2200`
  interpolates the analyzed aerosol on its source dry pressure. Lines 4424-4430
  retain the monthly surface source separately.

The WRF public-domain notice is retained in
`licenses/LICENSE-WRF-public-domain.txt`. These input and operator checks do
not establish whole-model equivalence or forecast skill.
