# Solar energy products

`gpuwm solar catalog` prints the product rows, default PV system rows, and
NOAA observation acquisition/verification rows embedded in `rw_wrfbatch`.
`gpuwm solar products --wrfout FRAME ... --out DIR` computes and renders a
native, chronological, fixed-grid sequence. The Python door calls
`gpuwm.rustwx.run_solar_product_renderer`; Rust reads, computes and renders.
Multi-record history files expand automatically; advanced frame rows can select
`time_index`. Optional `bounds` are `[west,east,south,north]` and affect only the
picture, preserving computation on the complete native grid.
Use `--products solar_ghi,solar_pv_ac_fixed` to select pictures. All products
still receive per-frame statistics in `receipt.json`.

The default rows describe 1 MWdc fixed 30-degree south-facing and horizontal
north/south single-axis tracking systems, with 60-degree limits, backtracking
and GCR 0.35 on level cross-axis terrain. Each row carries DC/AC ratio, inverter
efficiency, temperature coefficient, aggregate loss fraction, ground albedo,
Faiman thermal coefficients, module height and ramp threshold. Supply a JSON
list with `--systems FILE` for other systems; no new model code is needed.
Power units are MW for the row's actual nameplate, not a capacity factor.

GHI uses SWDOWN. Paired SWDDNI/SWDDIF are used when present; otherwise the receipt
names the Erbs hourly decomposition. A half-pair or unreadable present native
field fails instead of silently mixing methods. Radiation history preserves
native direct and diffuse output by default when the producer supplies it.

POA uses isotropic sky and ground reflection. PV power uses PVWatts v5 DC and
inverter equations with a Faiman steady-state module/cell temperature estimate,
module-height wind adjustment and the row's losses. This is not a claim of
full PVWatts simulator equivalence: Fuentes thermal inertia, Perez sky diffuse,
external shading, snow and a separate angle-dependent glass model are not used.
The clear-sky index uses empirical Haurwitz GHI; clear-sky GHI below 20 W/m2 is
missing in the ratio rather than producing an unstable night-time index.

Ramp maps are signed MW/min between native endpoints. Event maps apply the row's
fraction of AC nameplate per minute. The first frame and gaps beyond the
configured maximum are missing. A coarse history cannot resolve intra-interval
ramps. Daily GHI is kWh/m2 and daily AC energy is MWh, with UTC day boundaries.
Trapezoids are split at midnight, gaps poison affected cells, and a day without
all 86400 seconds of coverage receives no daily map and missing daily statistics.

For observations, `gpuwm solar fetch-observations --network surfrad --prefix bon
--directory Bondville_IL --date 2023-03-31 --out OBS` acquires the public daily file
and writes its acquisition receipt. SOLRAD uses `--network solrad` and its archive
station prefix. These rows describe the post-2015 one-minute archive. Both
formats use paired value/QC columns, but their direct/diffuse indices differ.
Only finite QC=0 values above the missing sentinel enter verification. Valid
negative nighttime instrument offsets are preserved. Missing data stay missing. NOAA terms and
acknowledgement are in the acquisition receipt and `licenses/solar-sources.md`.

Advanced requests use `gpuwm solar products --request REQUEST.json`:

```json
{
  "out": "solar-output",
  "frames": [{"label": "model", "kind": "wrfout", "path": "FRAME",
    "fields": {"solar_ghi": "SWDOWN", "solar_dni": "SWDDNI",
      "solar_dhi": "SWDDIF", "temperature_2m": "T2", "wind_speed_10m": "WSPD10"}}],
  "stations": [{"id": "bon", "network": "surfrad", "latitude": 40.05,
    "longitude": -88.37, "files": ["OBS/bon23090.dat"]}],
  "products": ["solar_ghi", "solar_pv_ac_fixed", "solar_daily_energy_fixed"],
  "max_gap_seconds": 3600,
  "domain": "d01"
}
```

Frames can also use the existing native GRIB or WxStore reader and explicit
valid-time metadata. GRIB field selection is canonical and does not branch on
model names. Station matching uses the nearest native model cell after checking
station containment, and an observation mean within the table's bounded time
window. The receipt distinguishes instantaneous model flux from the short
observation mean. `verification.csv` retains matched GHI/DNI/DHI and endpoint GHI
ramps; the receipt gives all-time and daylight error statistics. These compare
irradiance, not measured PV generation.

Pictures follow `DIR/domain/product/UTC-day/time.png`. Receipts name source
hashes, system parameters, methods, dimensions, stage wall times and parallelism.
The CPU/Rayon product route uses no GPU and reports zero GPU VRAM.
