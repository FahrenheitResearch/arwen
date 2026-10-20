# StormScope beside WOOF DA

StormScope is a separate forecast family. It is not a WOOF initialization,
three-dimensional analysis, or replacement for the DA cycle. This comparison
measures composite reflectivity against MRMS on common covered interior cells.
It does not establish DA quality, six-hour forecast skill, calibration,
lightning skill, or improvement from blending.

## Chain and source rows

`tools/nowcast/sources.toml` row `stormscope-3km-10min` feeds the existing
`tools/nowcast/run_nowcast.py`. The driver couples forecast GOES channels to
the radar model and writes `gpuwm-obs.nowcast-frames.v1`: `nowcast.json`, a
hashed input-object ledger, and raw little-endian float32 `refc` frames with
shape `[samples, ny, nx]`. See [nowcast-frames.md](nowcast-frames.md).
Only the initial six observed frames condition the free forecast. Later GOES
and lightning are model predictions. The matcher verifies READY status, `refc`,
dBZ, frame digest, input-ledger digest, exact frame time, sample bounds,
causal flag and latest-input time at or before issue. The runner owns the
object-level causality checks; the matcher does not independently re-download
or audit every input object.

A second supported model variant belongs in a new source-table row. Regional
`3km_10min_flash` is an open gap: the runner loads `row["variant"]`, but its
metadata and CLI expose no regional bounds, context pixels or `set_domain`
call. A variant row alone cannot request a regional run. No regional Flash
row is advertised and no second StormScope driver is introduced here.

WOOF DA members enter the same `rw_ensemble_match match-grid` binary as
`format: wrf-history`. Each member's `REFL_10CM` column maximum supplies dBZ.
A missing vertical value masks its column. AI members use `format:
nowcast-frames` with an explicit zero-based `sample`. Required member IDs
are declared separately from supplied files. A missing file, sample or roster
entry fails the entire scored frame. No available-member probability is made.

MRMS enters as the actual `MergedReflectivityQCComposite_00.50` GRIB2,
optionally gzip wrapped. The reader checks the native product and 500 m
level, regular-grid scan and timestamp. It accepts the nearest supplied
frame at or before valid time, at most 120 seconds old. The acquisition
step must select the latest such archive object. MRMS -999 is no coverage;
-99 remains valid observed clear air. This is composite reflectivity, not
QPE, base reflectivity or rainfall.

## Mapping, families and scores

The target is a declared 3 km WRF mass grid at the scored time. Every other
grid maps to it by nearest **valid** source centre within exactly 5,000 m,
using great-circle distance on a 6,371,000 m sphere. Equal distances choose
the lower source index. Identical coordinate hashes use identity mapping.
No dBZ averaging or interpolation occurs in this remap. Hashes of coordinates,
source-index mapping, inputs and the final common mask remain in the receipt.

The scored mask intersects MRMS, every required member in both families,
the declared specified/relaxation boundary rows, and the extra interior rim.
Use the actual boundary width from the producer's run configuration. The
derecho proof declares five rows and 45 km extra; it is an explicit verification
choice, not a recovered namelist claim. Missing cells do not count as clear
air and do not shrink the ensemble denominator.

At 20, 35 and 45 dBZ, each family reports probability as fraction of its own
required members meeting `>= threshold`, Brier score, and neighbourhood FSS.
FSS reuses the native masked FSS boxcar and reduction definitions: 11 by 11
cells, 33 km on this grid, zero boundary, neighbourhood sums divided by
covered counts. Family FSS compares the neighbourhood ensemble probability
with the neighbourhood observed binary event. Per-member deterministic FSS
and contingency metrics are also retained. These definitions are stated in
the receipts, so probability FSS is not mistaken for FSS of a thresholded
ensemble mean.

Family CSI, POD, FAR and frequency bias use the explicit probability cutoff
`>= 0.5`. Receipts include hits, misses, false alarms, correct negatives,
observed and forecast event counts, and expected forecast events (sum of
probabilities). Undefined ratios are JSON null. All-clear FSS is one. Low
Brier scores in rare-event domains alone do not demonstrate useful skill.

`physics` and `ai` are always separate. An omitted `blend` disables blending.
`"blend": {}` declares fixed weights 0.5 physics / 0.5 AI. Explicit nonnegative
weights summing to one may replace those defaults. Every declared family must
be present. The separately labelled experimental blend combines family
probabilities, never member counts. Weights must be chosen before scoring.

The matcher writes `gpuwm-resident-ensemble-products.v1` CDF5 products per
family and an observed singleton. The existing renderer already has generic
2-D XLAT/XLONG planes, the table reflectivity product, probability planes and
member postage. `tools.heat_eval.sheets --receipt` calls
`gpuwm.rustwx.run_ensemble_product_renderer` with `rw_wrfbatch`, then composes
its PNGs. The comparison is MRMS composite | StormScope probability >=35 |
DA forecast probability >=35; missing families have a labelled blank. Python
never reads weather arrays. Analysis charts in `ensemble_campaign_plot.py`
use only native score JSON. The CDF5 family mean exists for the renderer's
units contract; only the observed singleton mean is requested for drawing.

## Native request and campaigns

```json
{
  "quantity": "composite_reflectivity",
  "valid_time": "2024-01-08T18:30:00Z",
  "target_grid": "/workspace/e2c-20240108/physics/m001/wrfout/wrfout_d01_2024-01-08_18_30_00",
  "observation": "/workspace/e2c-20240108/observations/MRMS_MergedReflectivityQCComposite_00.50_20240108-182840.grib2.gz",
  "required_families": {"physics": ["p001"], "ai": ["a000"]},
  "members": [
    {"id": "p001", "family": "physics", "format": "wrf-history", "file": "/workspace/e2c-20240108/physics/m001/wrfout/wrfout_d01_2024-01-08_18_30_00"},
    {"id": "a000", "family": "ai", "format": "nowcast-frames", "file": "/workspace/e2c-20240108/stormscope", "sample": 0}
  ],
  "boundary_rows": 5,
  "interior_rim_m": 45000,
  "max_distance_m": 5000,
  "blend": {"physics": 0.5, "ai": 0.5}
}
```

This one-member example illustrates the contract only. A grid campaign uses
`{"schema":"gpuwm-ensemble-grid-campaign.v1","frames":[REQUEST, ...]}`.
`ensemble_campaign_score.py` invokes the same Rust binary for each frame,
freezes the roster and target geometry across times, and writes family rows
plus experimental-blend rows to `score-table.tsv`. Archived station/rainfall
campaigns retain their existing three-field product contract. Reflectivity
is registered in the same quantity table, using the gridded family contract.

## Exact planned 2024-01-08 18Z experiment

No 2024-01-08 WOOF DA product exists yet. The DA member set that exists for
this lane is the 2022-06-13 derecho set. The following commands are a future
run specification, not a claim that the 2024 experiment ran. No forecast,
weight download or GPU work is part of this lane's proof.

Before these commands, an authorized DA producer must supply accepted
2024-01-08 18Z analyses, complete 36-member histories every 30 minutes to
2024-01-09 00Z, GEFS-cycle/DA provenance, the real boundary width, and a
domain containing the verification region after boundary exclusions. This
recipe expects histories under `/workspace/e2c-20240108/physics/mNNN/wrfout/`,
the actual boundary width in `/workspace/e2c-20240108/physics/boundary-rows.txt`,
and an anonymously acquired MRMS index mapping each ISO valid time to the
latest composite path at or before that time in
`/workspace/e2c-20240108/observations/mrms-index.json`. All 12 verifying MRMS
objects must be within 120 seconds; the matcher refuses otherwise. The
example native request above is not that index.

From the engine root on an explicitly authorized inference box:

```sh
/workspace/venv-e2s/bin/python tools/nowcast/run_nowcast.py \
  --source stormscope-3km-10min --issue 2024-01-08T18:00Z \
  --minutes 360 --members 16 --batch-members 4 --seed 2024010818 \
  --revision 62f0fd2fa52c3cff67c931daac18cdc0d9f58d2a \
  --satellite goes16 --device cuda:0 --out /workspace/e2c-20240108/stormscope
python3 - <<'PY'
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
root = Path('/workspace/e2c-20240108')
obs = json.loads((root/'observations/mrms-index.json').read_text())
boundary = int((root/'physics/boundary-rows.txt').read_text())
frames = []
for lead in range(30, 361, 30):
    valid = datetime(2024, 1, 8, 18, tzinfo=timezone.utc) + timedelta(minutes=lead)
    iso = valid.strftime('%Y-%m-%dT%H:%M:%SZ')
    name = 'wrfout_d01_' + valid.strftime('%Y-%m-%d_%H_%M_%S')
    physics = [{'id':f'p{i:03}', 'family':'physics', 'format':'wrf-history',
        'file':str(root/f'physics/m{i:03}/wrfout'/name)} for i in range(1,37)]
    ai = [{'id':f'a{i:03}', 'family':'ai', 'format':'nowcast-frames',
        'file':str(root/'stormscope'), 'sample':i} for i in range(16)]
    frames.append(dict(quantity='composite_reflectivity', valid_time=iso,
        target_grid=physics[0]['file'], observation=obs[iso],
        required_families={'physics':[m['id'] for m in physics], 'ai':[m['id'] for m in ai]},
        members=physics+ai, boundary_rows=boundary, interior_rim_m=45000,
        max_distance_m=5000, blend={'physics':0.5,'ai':0.5}))
(root/'campaign-request.json').write_text(json.dumps(
    {'schema':'gpuwm-ensemble-grid-campaign.v1','frames':frames}, indent=2)+'\n')
PY
RAYON_NUM_THREADS=24 python3 tools/ensemble_campaign_score.py \
  --request /workspace/e2c-20240108/campaign-request.json \
  --matcher tools/rustwx/target/release/rw_ensemble_match \
  --out /workspace/e2c-20240108/scores
python3 tools/ensemble_campaign_plot.py \
  --scores /workspace/e2c-20240108/scores/campaign-scores.json \
  --out /workspace/e2c-20240108/analysis-charts
RAYON_NUM_THREADS=24 python3 -m tools.heat_eval.sheets \
  --receipt /workspace/e2c-20240108/scores/frame-011.receipt.json \
  --renderer tools/rustwx/target/release/rw_wrfbatch \
  --out /workspace/e2c-20240108/panels-6h
```

This is 16 StormScope samples for six hours, beside 36 required physics
members. Thirty-minute score times lie on both the AI ten-minute and the
planned physics fifteen-minute output clocks. The ordinary full-grid row
is used because regional Flash is not yet expressible. The last three
hours extend beyond the upstream three-hour evaluation described in the
prior audit; report them separately. No persistence-baseline forecast was
scored by this lane. The existing nowcast FSS/persistence verifier remains
available for a later time-sequence experiment.

## Reuse of the heating evaluation

`integrate/2.9.3` already contains the WP1 runner, WP2 and WP3 work from
`compose/291-final`. Read-only comparisons of `tools/heat_eval/score_case.py`
and `sheets.py` between those branches are byte-identical. The difference
in `rw-wrfbatch/src/bin/compare.rs` is the integration branch's extra
`required_fields: None` initializer, not a missing comparison renderer.

The campaign delegates each native family frame to
`tools.heat_eval.score_case.score_family_frame`. That seam is also available
as `python3 -m tools.heat_eval.score_case --grid-request REQUEST --matcher
BINARY --out DIRECTORY`. All weather data processing remains in Rust.
The heating decision registration stays separate: its 20/35 dBZ thresholds,
9/27/63 km neighbourhoods and nearest-within-two-minutes MRMS rule are not
the family registration's 20/35/45 dBZ, 33 km and causal at-or-before rule.

`tools.heat_eval.sheets --receipt` draws family sheets through
`gpuwm.rustwx.run_ensemble_product_renderer`. The existing `rw_wrfbatch`
ensemble product table supplies probability and postage. The heating
`rw_compare` panel source table already supplies WRF field/mean/PMM/max,
`frames:` and `ttenref:` comparisons, but no probability or member postage
source. Reuse the existing ensemble products rather than introducing a
second Rust probability implementation in `compare.rs`. The old
`ensemble_reflectivity_panels.py` entry point delegates to this sheet module.
Python pastes rendered PNGs only. Missing families stay labelled blanks.

## Licence and distribution

StormScope weights use the [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/).
Commercial use is allowed under its terms, and redistribution requires the
licence and required notices. The model weights are not Apache-licensed
engine code. Never bundle them in a public wheel. The existing runner fetches
them at runtime from `hf://nvidia/stormscope-goes-mrms` at the source row's
pin `62f0fd2fa52c3cff67c931daac18cdc0d9f58d2a` and refuses a revision override
that differs. Preserve that pin, model notices and source receipts with any
result. The prior desk audit supplies the licence/source review; this lane
does not repeat it or claim checkpoint access was tested.

## Scoring the AI family alone

When no physics history covers a case, `target_grid` may name the StormScope
nowcast-frames root itself: its 3 km Lambert lattice becomes the scoring grid
(the frame at the valid time must exist). `target_window: [y0, y1, x0, x1]`
cuts any target grid (WRF or nowcast) to one region, in target cells, so the
1024 x 1792 HRRR crop is scored over the case region only. Members on the same
lattice still go through the declared nearest-valid-centre remap.
