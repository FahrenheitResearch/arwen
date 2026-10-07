# Landuse input convention

The plume accepts explicit `mean_fct(4)` and `firesize(4)` arrays. A dominant
model land-cover category is insufficient to reconstruct them.

The public preparation code accumulates per-fire size and fire counts, then
divides accumulated size by count. It derives group flaming fractions from
species emission totals and averages the fractions over emitted species:
[preparation source, lines 1186-1279](https://github.com/NOAA-GSL/GSL-prep-chem/blob/a3301d1dc6eef5c023873f9b30017b003d6168bc/prep-chem/Prep_smoke_FRP/src/3bem_emissions_oper.f90#L1186-L1279).
Those fire observations and emission weights cannot be inferred from `ivgtyp`.

The same public revision has two different category aggregations at
[lines 1438-1472](https://github.com/NOAA-GSL/GSL-prep-chem/blob/a3301d1dc6eef5c023873f9b30017b003d6168bc/prep-chem/Prep_smoke_FRP/src/3bem_emissions_oper.f90#L1438-L1472).
Its active FRP preparation declares all flaming fractions zero at
[3bem_plumerise.f90:46-47](https://github.com/NOAA-GSL/GSL-prep-chem/blob/a3301d1dc6eef5c023873f9b30017b003d6168bc/prep-chem/Prep_smoke_FRP/src/3bem_plumerise.f90#L46-L47),
while the archived Original variant uses different fractions. Selecting those
archived numbers as a WRF model-category default would invent a convention.

No model-dataset/category conversion or assumed fire area is supplied.
`fire_groups.v1.json` carries only the pinned WRF heat fluxes and single-pass
rule. The fire preparation lane supplies the observational group arrays.
