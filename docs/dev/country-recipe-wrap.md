# Country recipe wrap and resume

The country integration is complete. All four used boxes are scheduled for
release. This lane used G3 only. Do not depend on any old box address, build,
prepared folder or borrowed geography tree. No new GPU job was started for
wrap-up. The last GPU forecast and control pair had already completed.

The dated Downloads handoff contains the branch bundle, patch, config,
METAR/CMORPH score requests and receipts, map PNGs, hashes, scripts and a
small receipt/observation archive. All needed source is in the desktop engine
repository and the forward bundle. The archive contains no model history,
prepared cache, aerosol binary or geography tiles.

## Regenerating inputs after a new approved box is available

Use the same box image requirements as the proof: `/work/env.sh`, Rust 1.94,
a free-threaded Python 3.14 environment with the project's installed scientific
dependencies, matching CUDA wheel libraries, and the GPU mutex helper.
Restore source into `/work/box-country-recipe/src`, and the handoff scripts
into `/work/box-country-recipe/`. Create a fresh owned venv instead of reusing
paths on a released box. The archived `setup-box.sh` and `box-env.sh` document
the editable installation and library selection. Set `TMPDIR` and caches
inside the lane folder. Build tools from the restored checkout; do not bypass
stale-build checks.

```sh
mkdir -p /work/box-country-recipe/tmp /work/box-country-recipe/configs
export TMPDIR=/work/box-country-recipe/tmp
/work/venv-ft/bin/python -m venv --system-site-packages /work/box-country-recipe/venv-ft
bash /work/box-country-recipe/setup-box.sh
bash /work/box-country-recipe/build-box.sh
source /work/box-country-recipe/box-env.sh
export GPUWM_NO_LOCAL_GPU=1 CUDA_VISIBLE_DEVICES=
cd "$T/tools/rw_wps"
cargo build --release --locked --offline -j 8 --bin rw-wps --bin gpuwm_mapped_engine
cd "$T"
python -m gpuwm.cli domain --recipe hrrr_v4_gsd41 --region PH --spacing-km 3 --source gfs --cycle 2025-11-03T00 --hours 6 --cadence 1 --out "$L/configs/country.toml"
python -m gpuwm.cli fetch-geog --config "$L/configs/country.toml" --root /workspace/shared-prepared/box-country-recipe/geog --source ncar
python -m gpuwm.cli fetch-tables --wif --wif-only --wif-root "$GPUWM_WIF_DATA_ROOT"
python -m gpuwm.cli fetch-tables --thompson-fork --thompson-fork-only --thompson-fork-root "$GPUWM_THOMPSON_FORK_TABLE_ROOT"
python -m gpuwm.cli go "$L/configs/country.toml" --prepare-only --outdir /workspace/shared-prepared/box-country-recipe/preparation --data-dir /workspace/shared-prepared/box-country-recipe/forcing --geog-root /workspace/shared-prepared/box-country-recipe/geog --devices 2 --products none
```

The optional geography is pinned. Fetch the baseline independently on a new
box; the old proof's borrowed baseline belongs to another lane and must not
be copied or modified. No account credential is needed for the staged public
GFS, monthly aerosol, geography, METAR or CMORPH sources used here.
Rebuilding and refetching after box release were not measured in this wrap.

Restore METAR from the small archive, including the corrected endpoint
window, without refetching. To regenerate satellite observation packs:

```sh
python /work/box-country-recipe/fetch-rain.py
bash /work/box-country-recipe/prepare-rain.sh
```

If the short US control is specifically needed again, run the archived
`prepare-control.sh` on CPU. Base and tip must use one shared sealed bundle
and independently rebuilt native tools. A repetition is not needed to retain
the measured result already in the handoff.

## Scope at stop

The direct recipe door is qualified by one six-hour country run and a short
US integration identity pair. Monthly aerosol is a declared forcing difference
from operational analysed HRRR. The separate imported suite-planner test group
still has three failures. No ensemble-admission or exact-chain changes were
made here. Do not start another forecast until a new session authorizes the
box and the specific run. Old scripts naming timestamped preparation folders
are experiment records; resolve the newly prepared folder before using them.