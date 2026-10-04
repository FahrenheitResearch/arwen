# Radar colour table provenance

`crates/rustwx-render/src/radar_tables.rs` carries the colour tables the
renderer draws radar moments with. Each is transcribed value for value from
the owner's radar application, BowEcho (MIT OR Apache-2.0, the same licence
as the simulation extraction recorded in `SIMULATED-RADAR.md`):

| Source | Revision |
| --- | --- |
| BowEcho `crates/color_tables/src/lib.rs` | `66ceb9c4aea5da2bd857d3178f08a13c4c08a8d0` |

| Renderer table (`RadarTable`, name) | BowEcho definition | Origin |
| --- | --- | --- |
| `Reflectivity`, `radar_reflectivity` | its AWIPS reflectivity preset, the `.pal` table at line 2637 | A `.pal` preset of the NWS AWIPS reflectivity colour table, -30 to 95 dBZ, with per-row gradient end colours |
| `RadialVelocity`, `radar_velocity` | its green-red velocity preset, the `.pal` table at line 2875 | A `.pal` velocity preset in knots (`Scale: 1.9426`): greens toward the radar, reds away, a narrow dark grey band at zero |
| `DifferentialReflectivity`, `differential_reflectivity` | `builtin_differential_reflectivity_table` | BowEcho's own default ZDR table, -4 to 8 dB |
| `CorrelationCoefficient`, `correlation_coefficient` | `builtin_correlation_coefficient_table` | BowEcho's own default CC table, 0.2 to 1.05 |
| `SpecificDifferentialPhase`, `specific_differential_phase` | `builtin_specific_differential_phase_table` | BowEcho's own default KDP table, -1 to 7 deg/km |
| `DifferentialPhase`, `differential_phase` | `builtin_differential_phase_table` | BowEcho's own default PHIDP table, 0 to 360 deg |

The two `.pal` presets keep BowEcho's interval sampling: each row ramps
from its colour to its own end colour, or to the next row's colour when it
has none. The four dual-polarization tables keep its linear interpolation.
The renderer bins each table into a discrete scale and colours every bin at
its centre.

The scales these tables replaced stay selectable: `reflectivity_classic`
(the twelve-step product ladder) and `radial_velocity_classic` (the ten-step
blue-red scale), by table name through
`rw_wrfbatch::scales::radar_scale_named`, and together as the `classic`
radar colour set (`[simulated_radar] color_tables = "classic"`,
`gpuwm render --radar-colors classic`, `RUSTWX_RADAR_COLORS=classic`).
