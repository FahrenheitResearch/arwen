# Sub-grid terrain drag

These optional WRF v4.7.1 schemes use terrain statistics below the model
grid scale. Both selectors default to 0. Turning them on changes the
forecast; column agreement with WRF does not establish forecast skill.

| Selector | Values | Behavior |
|---|---|---|
| `topo_wind` | 0 off, 1, 2 | Corrects YSU surface momentum drag and 10 m wind using terrain variance and terrain curvature. Option 1 uses `VAR_SSO`; option 2 uses `VAR`. Requires `bl_pbl_physics = 1`. |
| `gwd_opt` | 0 off, 1, 3 | Adds orographic drag after the PBL scheme at its cadence. Option 1 provides gravity-wave drag and blocking; option 3 provides large-scale gravity-wave drag, blocking, small-scale gravity-wave drag and turbulent orographic form drag. Requires an active PBL scheme. |

GSL components taper with grid spacing as in WRF. Its large-scale drag
vanishes at 3 km and its small-scale wave and form drag vanish at 1 km.
The port uses WRF's default `gsl_diss_ht_opt = 0`, with no dissipative
heating. `gwd_opt = 2` is unavailable.

With SASE, GSL uses SASE's bulk-Richardson PBL height and its corresponding
upper-bracket level. This BL definition differs from a WRF PBL package.
The diagnostic supplies drag locally and leaves the surface-layer PBL
height and top-level carriers unchanged.

Select an option before preparing the run so the static builder includes
its fields. For example, in the experiment TOML:

```toml
[shared]
bl_pbl_physics = 1
sf_sfclay_physics = 1
topo_wind = 1
gwd_opt = 0
```

The GEOG root must contain the selected WPS datasets:

| Option | Default dataset | Produced fields |
|---|---|---|
| `topo_wind = 1` | `varsso_10m` | `VAR_SSO` |
| `topo_wind = 2` | `orogwd_10m/var` | `VAR` |
| `gwd_opt = 1` | `orogwd_10m` | `VAR`, `CON`, `OA1` to `OA4`, `OL1` to `OL4` |
| `gwd_opt = 3` | `orogwd3_10m` | `VAR`, `CON`, `OA1` to `OA4`, `OL1` to `OL4`, each with `LS` and `SS` suffixes |

Unpack the WPS orographic datasets beneath the same GEOG root used for
terrain and land use. `geog_data_res` in `namelist.wps` selects the WPS
table's resolution alternatives. Missing requested fields stop preparation
with the dataset name; they are never replaced with zero drag.

```sh
gpuwm prep --source SOURCE --input-list inputs.txt \
  --experiment-config experiment.toml --wps-namelist namelist.wps \
  --geog-root WPS_GEOG --output-root prep
gpuwm sim prep --experiment-config experiment.toml \
  --wps-namelist namelist.wps --outdir forecast
```

Replace `SOURCE` with the forcing source and include its required
supplementary inputs, as for a run without terrain drag.

The current routes support a prepared resident root domain and a separate
downscale of saved history on its own geography. The downscale builds the
requested statistics from WPS_GEOG and passes them to child physics.
Parent-terrain downscales and children inside a running domain tree are
refused because those initializers do not supply the required statistics.
Streamed tiles are refused because the statistics are not
gathered for each tile. `topo_wind` with BEP or BEP+BEM is refused because
that YSU arm does not contain the surface momentum term it would scale.

The reproducible column authority is
`tools/terrain_drag_wrf471_oracle/build.sh`; its WRF sources and fixture
files are pinned by SHA-256. The comparison gates are
`tests/test_terrain_drag_wrf471_parity.py`, and option/import/default
identity gates are `tests/test_terrain_drag_config.py`.
