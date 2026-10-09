# Analyzed aerosol initial and boundary fields

`mp28_aerosol_source = "analysis"` requires both water-friendly and ice-friendly
aerosol number mixing ratios on every initial and lateral boundary frame.
Missing fields stop preparation rather than substitute a climatology.
On the native HRRR route the bridge reads the pair only for a configuration
that requests it (this key, or `use_rap_aero_icbc`); any other configuration
never selects it, so a masked or partial pair cannot stop it. For a
configuration that requests it, bitmap shape and present packed count must
agree. Masked cells outside the requested native crop are accepted. Missing
QNWFA/QNIFA cells inside it take nearest_neighbor, four_pt, average_4pt, then
zero only when the chain has no answer. Finite source values retain the existing
packing bounds policy and output bytes. Any other selected field with masked
cells inside the crop refuses with its field, level, message index, crop and
masked count.

The bridge emits finite native payloads and, for each requested aerosol field,
a level-major byte bitmap in `QNWFA.mask` or `QNIFA.mask`.
The bitmap is part of the sealed bridge manifest. Native floor/ceiling corners
coincide at integer grid points, so masked native cells exhaust the chain and
receive zero in the numeric payload. Those zeros never become valid aerosol
donors: the Python boundary carries the original bitmap to the Rust chain at
the fractional target coordinates. Finite neighboring corners are tried before
zero is used there. A field without an in-crop mask retains its existing nearest
mapping and output bits.

`inventory.tsv` records each selected message's masked count and repair stage
counts. The completed `gate.txt` sums them per aerosol field across every lead.
The prepare report carries those counts as `input.aerosol_missing` and, when
streaming, `pipeline.aerosol_missing`. Mapping reports add the target counts under
`aerosol_missing`, including finite nearest answers. A later masked lead is repaired under
the same policy; no series-wide withholding or masked-later-lead refusal
remains. Historical bridges that omitted the pair still refuse analyzed
initialization: their actual missing payload cannot be repaired retrospectively.
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
