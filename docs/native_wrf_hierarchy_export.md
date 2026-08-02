# Native nested WRF final export

The hierarchy exporter is the first multi-domain component of the native
WPS/`real.exe` replacement.  It translates standard WPS/WRF namelists in
memory, verifies every domain against its independently prepared native
artifacts, and atomically publishes:

```text
wrfinput_d01
wrfinput_d02
...
wrfinput_dNN
wrfbdy_d01
manifest.json
```

Only the root gets an external boundary file.  Child domains carry WRF nest
identity (`GRID_ID`, `PARENT_ID`, starts, ratio, child `DX`/`DY`/`DT`) and are
forced by their parents when stock `wrf.exe` runs.

The generic manifest join remains an internal integration interface. HRRR
also has a public, fail-closed end-to-end hierarchy mode for one sealed
Oklahoma root and arbitrary valid one-way Lambert nest layouts declared by
standard WPS/WRF namelists. It accepts `max_dom=1..21`, verifies a previously
prepared 12-hour root, initializes d02..dNN through the bounded worker
scheduler, and publishes the complete stock-WRF input directory atomically.
Invalid, non-contiguous, cyclic, orphaned, or out-of-parent layouts fail before
publication rather than being silently approximated.

## Public HRRR hierarchy command

```bash
gpuwm-wrf-init --source hrrr \
  --root-preparation /case/root-preparation \
  --domain-spec /case/hrrr-d01-199x199-3km.json \
  --wps-namelist /case/namelist.wps \
  --namelist-input /case/namelist.native.input \
  --stock-wrf-namelist-input /case/namelist.stock.input \
  --geog-root /data/WPS_GEOG \
  --source-sha256s /data/hrrr/SHA256SUMS \
  --source-sha256s-sha256 EXPECTED_MANIFEST_HASH \
  --valid-time 2026-07-18_00:00:00 \
  --output-root /case/wrf-hierarchy \
  --child-workers 8
```

`--valid-time` on this door is the **cycle**; add `--forecast-start-hour K`
when the sealed root preparation began at a lead, and every stage derives
model time zero (cycle + K) for itself.  The underlying
`gpuwm.hrrr_hierarchy_direct` takes `--cycle`/`--forecast-start-hour`; its
own `--valid-time`, which meant model time zero rather than the cycle,
remains accepted with that meaning for compatibility and refuses to be
combined with a lead.

The native and stock namelists must be semantically identical after
normalizing exactly three receipt-bound runtime differences: WRF RRTM
longwave replaces disabled native longwave, stock WRF consumes the exported
moist-theta representation, and `ghg_input=0` pins RRTM to fixed gases.  Any
other raw namelist difference fails before output publication.

The live 199x199x49 d01 plus five 300x300x49 children proof completed native
d01..d06 generation and stock-WRF export in 135.093 seconds with eight
workers. The parallel child phase took 75.911 seconds and final export took
44.664 seconds. Parser, raw-runtime-contract, and atomic-export tests cover
every `max_dom` from 1 through 21; the larger counts are cardinality gates,
not claims that every possible geometry has received a stock-WRF forecast.

A pinned unchanged WRF v4.6.1 dmpar binary accepted the exported d01..d06
hierarchy and root boundary directly. A 24-rank acceptance run advanced every
domain through 30 simulated seconds in 15:12.99 wall time, exited zero, and
reported `SUCCESS COMPLETE WRF` on all 24 ranks with no fatal marker. All six
initial history files passed `ncdump`, and SHA-256 verification proved that the
binary, namelists, six `wrfinput` files, and `wrfbdy_d01` were unchanged by the
run. Stock-WRF acceptance depth is recorded separately from the 1..21
parser/export bound so public claims cannot confuse synthetic cardinality
coverage with a completed forecast at every depth.

## Artifact inventory

Expensive decode/interpolation/static-build jobs may run concurrently.  Each
job writes one prepared cache, one static cache, and one geometry receipt.  A
small relocatable manifest joins them only at the final dependency barrier:

```json
{
  "schema": "gpuwm-native-domain-artifacts-v1",
  "domains": [
    {
      "grid_id": 1,
      "prepared_cache": "cache/d01",
      "static_cache": "static/d01.npz",
      "geometry_receipt": "geometry/d01.json"
    },
    {
      "grid_id": 2,
      "prepared_cache": "cache/d02",
      "static_cache": "static/d02.npz",
      "geometry_receipt": "geometry/d02.json"
    }
  ]
}
```

Relative paths resolve against the artifact manifest.  Unknown keys,
duplicate/missing IDs, non-contiguous namelist IDs, an invalid parent order,
config drift, eta drift, static/geometry digest drift, and unsupported physics
all fail before the output directory is published.

Every receipt geometry value is also regenerated from the namelist hierarchy
and compared exactly; an editable center or projection cannot become output
metadata merely because the receipt itself is well-formed.  Final NetCDF
validation rereads the domain identity, nest placement, spacing, timestep, and
projection attributes before assigning `READY`.

## Internal command

```bash
python -m gpuwm.wrf_direct \
  --wps-namelist namelist.wps \
  --namelist-input namelist.input \
  --domain-artifacts domain-artifacts.json \
  --boundary-interval-seconds 3600 \
  --output wrf-ready
```

The valid time defaults to the namelist start.  If `--valid-time` is supplied,
it must match the namelist and every prepared cache.  Existing output is
preserved unless `--overwrite` is explicit.

Explicit replacement uses an exclusive publication lock and moves the prior
validated tree to `<output>.previous-valid` before installing the candidate.
A failed install rolls the prior tree back.  An interrupted process therefore
leaves the former result recoverable; a later invocation restores a lone
backup and refuses to choose automatically if both trees are present.

## Parallelization boundary

The manifest is deliberately a join point rather than a work queue.  Source
download/decode, horizontal interpolation, and per-domain static generation
are candidates for bounded parallel execution.  The public HRRR orchestrator
now schedules independent child initialization with a bounded worker count
and feeds this same deterministic manifest join.  WRF-specific child terrain
blending and parent-dependent balance adjustments remain in an explicit
parent-before-child phase, so the final-export contract stays deterministic
regardless of worker count or CPU/CUDA preprocessing backend.
