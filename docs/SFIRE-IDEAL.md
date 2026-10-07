The sounding-driven fire initializer is reachable through `gpuwm fire-ideal`
and the assembled `woof fire-ideal` command. It prepares the native WRF ideal
vertical coordinate, atmosphere, surface and refined fire terrain on the GPU.
Rust reads the sounding and optional native text grids. Python selects inputs,
launches kernels and writes receipts through the existing native output path.
The complete short recipe is under `examples/sfire-ideal`.

```sh
gpuwm fire-ideal fire.toml --sounding input_sounding --outdir out/fire
gpuwm fire-ideal namelist.input --namelist --input-directory inputs \
  --outdir out/native-fire
```

A TOML configuration uses the existing `[grid]`, `[dynamics]`, `[run]` and
`[fire]` RunConfig tables. Declare `ifire = 2`, `moist = true`, positive
`sr_x` and `sr_y`, fuel selection and the desired ignition lines. The native
namelist form imports `&fire`, grid dimensions, fixed fractional timestep,
physics, output cadence and `&dynamics tracer_opt = 3` for optional bulk smoke.
An adjacent `namelist.fire` supplies fuel and moisture parameters. The
`--input-directory` option selects that file and the native optional grids.

The native `input_sounding` header contains surface pressure in mb, surface
potential temperature in K and surface water vapor in g/kg. Each subsequent
record contains height in m, potential temperature in K, water vapor in g/kg,
and u/v wind in m/s. The original initializer does not consume the header's
water-vapor value; the receipt retains this fact. Profile values drive the
dry base and moist atmosphere calculations.

The initialization controls include `stretch_grd`, `stretch_hyp`,
`z_grd_scale`, explicit `eta_levels`, the four native `hybrid_opt` coefficient
sets, `fire_mountain_*`, the five perturbation controls, `sfc_full_init` and
the surface initialization fields. The ideal path uses Cartesian coordinates
(`map_proj = 0`) and native ideal `hypsometric_opt = 1`. Its per-column base
arrays retain terrain geometry even for a flat domain.

Optional native two-dimensional REAL grids begin with their two integer
dimensions. Each following record supplies all y values for one x index.
Rust preserves that source layout while returning contiguous engine fields.
The `fire_read_*` switches select the same fixed filenames as WRF. Fine fields
must match the configured refinement. Source-inert namelist fields retain
their source status; source branches which terminate without implementation
retain a concrete refusal.

The command writes native `wrfout` histories, preparation and run receipts,
and checkpoints at `restart_interval_s`. A checkpoint restores the carried
atmosphere, fire, moisture and smoke arrays. Continue into a fresh folder:

```sh
gpuwm fire-ideal fire.toml --sounding input_sounding \
  --restart out/fire/gpuwmrst_d01_0001-01-01_00:00:02.npz \
  --outdir out/continued
```

The restart checks the sounding, fuel namelist and optional grid hashes in
addition to the engine's configuration and setup identities. Forecast length
and output cadence follow the existing restart compatibility contract.
`--prepare-only` writes the initialization receipt after full state and
physics construction. Fixed-clock run length and history/restart alarms must
lie on the configured timestep lattice.

When moisture evolution is enabled with every surface-layer and land-surface
scheme disabled, the original allocated T2/Q2 values have no producer and T2
remains 0K. The corrected default supplies a near-ground atmospheric proxy:
the native three-point ground extrapolation of current temperature and dry
potential temperature, lowest-level vapor and diagnosed surface pressure.
It enables no surface exchange physics. `FIRE_MOISTURE_SURFACE_SOURCE` records
this source in histories; it is not a surface-scheme 2m diagnosis. Active
surface schemes retain their own diagnostics.

Nested real forecasts use the normal experiment preparation and `gpuwm sim`
or `woof sim` doors. Native initialized nested inputs use `gpuwm run
--wrfinput`. The refined fire grid can belong to the innermost domain; these
paths retain the parent forcing and nesting state.
