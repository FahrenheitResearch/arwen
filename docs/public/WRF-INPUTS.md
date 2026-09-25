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

## `use_theta_m` on the WRF doors

`import-namelist`, `run --wrfinput` and `run --met-em` all accept a namelist that selects `use_theta_m = 1` (WRF's default when the key is omitted), and all announce it at the terminal as a declared divergence: ArWen integrates dry potential temperature and has no moist-theta branch. The initial and boundary state is recovered exactly on every one of them (metgrid TT is physical temperature, a moist wrfbdy's THM/QV/MU are converted at each forcing time, and native initialization builds dry theta from physical temperature), but the integration differs from what WRF would do with `use_theta_m = 1`. The import receipt (`input/wrf-import.json` or `input/metgrid-import.json`) lists it under "Physics substitutions" with the reason. Set `use_theta_m = 0` in the producing namelist to run WRF on the same variable.

What the files hold: WRF 4.0 through 4.7.1 `real.exe` writes `wrfinput` `T` as dry theta-300 under both settings and `THM` as the prognostic (moist theta-300 under `use_theta_m = 1`, equal to `T` otherwise), and `wrfbdy` `T_BXS` and its siblings couple the prognostic with dry column mass. ArWen reads `T` as its dry initial theta, converts a moist boundary back to dry theta with the boundary's own vapour and mass at each forcing time, and checks at import that the boundary's first record equals the coupling of the initial file's own representation: under `USE_THETA_M = 1` moist theta rebuilt from `T` and `QVAPOR` with WRF's formula, `(T + 300) * (1 + Rv/Rd * QVAPOR) - 300`, to WRF's FP32 rounding. A stock WRF 4.6.1 pair passes at both settings bit for bit. The refusal `wrfbdy T west does not match initial wrfinput_d01` states what was compared and how far apart the two files are, and names the case where the boundary holds dry theta under `USE_THETA_M = 1`: WRF 3.7 to 3.9.1.1 `real.exe` wrote that (`use_theta_m` entered the namelist in 3.7; no `THM` variable; its solver converted at run time), WRF 4.x never does. `use_theta_m = 0` in the producing namelist remains a fine setting: the files then hold dry theta everywhere and the import applies no vapour conversion.

## `fine_input_stream` on the WRF doors

`fine_input_stream = 2` is WRF's delayed-nest-start stream: the nest takes only its static and masked land-surface fields from its own input and interpolates the rest from the parent. `import-namelist`, `run --wrfinput` and `run --met-em` all accept it and announce it at the terminal as a declared divergence. The delayed child starts at its declared start time either way; what differs is where its masked surface state comes from, which in ArWen is the child's own-grid analysis rather than a `real.exe` `wrfinput`. `fine_input_stream = 0`, WRF's other defined value, is what ArWen already does for every domain and is recorded without a divergence. An index WRF does not define is refused by name, with both defined values stated. The RW-WPS support report and the importer read this answer from one function, so a namelist pair the report passes is a pair the importer imports.
