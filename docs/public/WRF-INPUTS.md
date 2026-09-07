# Use existing WPS or WRF inputs

Native ArWen preparation remains the recommended path for new forecasts. Existing
WPS and `real.exe` scripts can also hand their files to the same ArWen forecast
runtime.

After `real.exe`, keep `wrfinput_d0*`, `wrfbdy_d01`, and the producing
`namelist.input` together:

```console
gpuwm run --wrfinput wrf-run --outdir forecast
```

After WPS `metgrid.exe`, keep `met_em.d0*.nc` and the intended
`namelist.input` together. ArWen performs its native initialization:

```console
gpuwm run --met-em metgrid-run --outdir forecast
```

Explicit `eta_levels` are preserved exactly. When they are absent, native
initialization generates the requested `e_vert` grid with WRF's automatic
algorithm. Both `auto_levels_opt = 1` and `2` are supported; `max_dz`, `dzbot`,
`dzstretch_s`, and `dzstretch_u` retain their namelist values. Omitted controls
take the WRF defaults (option 2, 1000 m, 50 m, 1.3, and 1.1). The resolved
grid and control values are recorded with the forecast. `--vertical-grid native`
explicitly chooses ArWen's native profile instead when no eta list is given.

The external WRF doors preserve WRF RRTMG when radiation scheme 4 is selected.
Changing it to RTE+RRTMGP requires `--rrtmg-variant rte-rrtmgp`. Other unsupported
controls fail with the missing capability named; the adapter does not silently
replace explicit physics. `--run-seconds 600` shortens a run within its supplied
forcing coverage.

Both routes use ArWen's shared clocks, physics setup, output, health checks, and
restart machinery. Input hashes, actual humidity/soil interpretation, selected
physics, and initialization choices are recorded under `forecast/input`; users
do not copy hashes into a launch command. The metgrid path prices initialization
and forecast memory separately before preparing a state.

Metgrid soil can contain depth-node stacks (`SOILT`, `SOILM`, `SOIL_LEVELS`) or
layer stacks (`ST`, `SM`, `SOIL_LAYERS`) with the corresponding WPS layer names.
The actual depths must support the selected land model's target geometry.
Flagged analyzed mass categories are retained according to the active physics
package. If a flagged number concentration is active but its native
initialization is unavailable, use `real.exe` and the `--wrfinput` door to retain
that state. Likewise, a request for WRF's sea-level-pressure reconstruction is
distinct from the implemented source-pressure/terrain reconstruction.

Moving nests need forcing and static coverage at their future positions. A
directory containing only initial footprints cannot supply that coverage;
prepare a native statics corridor for those requests. These rectangular WRF
files do not establish compatibility with other mesh layouts.
