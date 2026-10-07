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

Native model-level mappings can declare `model_top_pressure_pa`, the interface
above the highest mass level. This lets a native analysis initialize the model
on its own eta ladder and lid. Each target mass level must still be supported
by the decoded pressure column. A declared numerical divergence from WRF real's
strict top test co-locates a target with the highest source mass level when
the target lies above it by at most 2^-16 (1.53e-5) times that source pressure.
The gap is physical as well as numerical: the native top level is a full
pressure that carries the water vapour of its top half-layer, while the target
is the same level's dry pressure, and WRF's moisture integration leaves the top
level undried. At HRRR's 1731.475 Pa top mass level above a 1500 Pa lid this
is 231.475 Pa times the top-level vapour mixing ratio, plus GRIB and FP32
rounding: 0.00061 Pa on 2026-10-03 00Z and 0.00085 Pa on 2025-03-14 06Z, the
second past the earlier four-FP32-epsilon bound. 2^-16 holds a top-level vapour
mixing ratio up to 1e-5 kg/kg and is about 0.1 m in height. The target takes
the source endpoint value and no atmospheric layer is extrapolated. A target
further above the source top is refused, because its values would have to be
invented above the analysis. WRF real.exe has no top tolerance to copy here: it
refuses this start earlier, because its `find_p_top` takes the highest pressure
on the top source level and requires `p_top_requested` to be at least that.
The preprocessing receipt records
`native-pressure-top-colocation-relative-2pow-16-v2`.

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

`import-namelist`, `run --wrfinput` and `run --met-em` all accept a namelist that selects `use_theta_m = 1` (WRF 4's default when the key is omitted), and all announce it at the terminal as a declared divergence: ArWen integrates dry potential temperature and has no moist-theta branch. The initial and boundary state is recovered exactly on every one of them (metgrid TT is physical temperature, a moist wrfbdy's THM/QV/MU are converted at each forcing time, and native initialization builds dry theta from physical temperature), but the integration differs from what WRF would do with `use_theta_m = 1`. The import receipt (`input/wrf-import.json` or `input/metgrid-import.json`) lists it under "Physics substitutions" with the reason. Set `use_theta_m = 0` in the producing namelist to run WRF on the same variable.

An omitted key takes the default of the WRF version the namelist was written for, and WRF 3.9 (the line operational HRRR v4 runs, NOAA-EMC/HRRR v4.1.21 `Registry.EM_COMMON:2633`) defaults it to 0, dry theta, the variable ArWen integrates. `run --wrfinput` reads the version from the files' own `TITLE` ("OUTPUT FROM REAL_EM V3.9..."), so such a run directory imports with no substitution and no flag. `import-namelist` sees only the namelist and assumes WRF 4 unless told `--wrf-version 3`. Either way the import report and the receipt list the applied default under "Namelist defaults", with the WRF version that was read, what chose it (the flag, the files' `TITLE` or the default) and the Registry row it came from, so a namelist read on the wrong version shows.

The files ArWen exports for stock WRF (the companion `wrfinput_d0N` and `wrfbdy_d01`) hold dry theta throughout and declare it: `USE_THETA_M = 0`, `THM` equal to `T`, `T_B*` dry-coupled. Initial `T` and `THM` use the model's float32 sum of base and perturbation theta, rounded before subtracting 300 K, so they match the boundary coupling when read back through `run --wrfinput`. Run WRF on them with `use_theta_m = 0`; WRF 4 stops at its input check when the namelist omits the key or says 1. Exports written by 2.8.4 or earlier declared 1 over dry boundary rows and should be written again ([WRF-INTEROP.md](WRF-INTEROP.md)).

The first `wrfbdy_d01` record of an export is the coupling of the exported `wrfinput_d01` itself, as `real.exe` always writes it. A preparation whose start state comes from a separate analysis (`--initial-inputs`) built its boundary from the boundary source, so that first record and its tendency are rewritten the way WRFDA's `da_update_bc` rewrites them after an analysis: the values become the new start's coupling and the tendency still reaches the prepared boundary at the end of the first interval; later records are the prepared ones unchanged. When the preparation forces the Thompson-aerosol numbers at the boundary (`mp_physics = 28` with aerosols from input), `QNWFA_B*` and `QNIFA_B*` carry those tables; run WRF with `use_aero_icbc = .true.` to use them. Exports written by 2.8.6 or earlier with `--initial-inputs` held another atmosphere in that first record (stock WRF's outer row jumped by up to 10 m/s and 8 K in its first minute on a 3 km RAP crop, and `run --wrfinput` refused the pair), and mp 28 exports held zero aerosol tables; write them again.

What the files hold: WRF 4.0 through 4.7.1 `real.exe` writes `wrfinput` `T` as dry theta-300 under both settings and `THM` as the prognostic (moist theta-300 under `use_theta_m = 1`, equal to `T` otherwise), and `wrfbdy` `T_BXS` and its siblings couple the prognostic with dry column mass. ArWen reads `T` as its dry initial theta, converts a moist boundary back to dry theta with the boundary's own vapour and mass at each forcing time, and checks at import that the boundary's first record equals the coupling of the initial file's own representation: under `USE_THETA_M = 1` moist theta rebuilt from `T` and `QVAPOR` with WRF's formula, `(T + 300) * (1 + Rv/Rd * QVAPOR) - 300`, to WRF's FP32 rounding. A stock WRF 4.6.1 pair passes at both settings bit for bit. A `real.exe` build that fuses multiply-adds rounds that rebuild differently; the difference is bounded by a few FP32 steps of the full theta (a few hundred K) times the column mass, and the check allows that bound at each point. The refusal `wrfbdy T west does not match initial wrfinput_d01` states what was compared and how far apart the two files are, and names the case where the boundary holds dry theta under `USE_THETA_M = 1`: WRF 3.7 to 3.9.1.1 `real.exe` wrote that (`use_theta_m` entered the namelist in 3.7; no `THM` variable; its solver converted at run time), WRF 4.x never does. `use_theta_m = 0` in the producing namelist remains a fine setting: the files then hold dry theta everywhere and the import applies no vapour conversion.

## `fine_input_stream` on the WRF doors

`fine_input_stream = 2` is WRF's delayed-nest-start stream: the nest takes only its static and masked land-surface fields from its own input and interpolates the rest from the parent. `import-namelist`, `run --wrfinput` and `run --met-em` all accept it and announce it at the terminal as a declared divergence. The delayed child starts at its declared start time either way; what differs is where its masked surface state comes from, which in ArWen is the child's own-grid analysis rather than a `real.exe` `wrfinput`. `fine_input_stream = 0`, WRF's other defined value, is what ArWen already does for every domain and is recorded without a divergence. An index WRF does not define is refused by name, with both defined values stated. The RW-WPS support report and the importer read this answer from one function, so a namelist pair the report passes is a pair the importer imports.
