"""Finding 4 settling measurement: the two sizing arms of the standalone-child fit.

Arm RETIRED: gpuwm.downscale._fit_child_size as it ships -- flat
vram_reserve_gib budget, ALLOCATOR_HEADROOM x subtotal through the retired
multiplicative observed_peak_envelope_bytes.

Arm AFFINE: the same search loop with the fit criterion the live sizing
path uses (domain wizard / gpuwm check): estimate_experiment(...)
.peak_envelope_bytes against card_assumed_free_gib - EXTERNAL_MARGIN,
stopping short by fit_headroom_bytes.

Parent: the real 386x308 12 km GFS parent in parent-rrtmgp-run (its own
authority record supplies the config a restart would carry).  Card: the
desktop RTX 3080, 10 GiB, declared.
"""

import json
import math
import sys
import tomllib
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

from gpuwm import downscale as ds
from gpuwm.config import RunConfig
from gpuwm.core.preflight import (
    EXTERNAL_MARGIN_BYTES, GIB, estimate_experiment)
from gpuwm.domain_wizard import card_assumed_free_gib, fit_headroom_bytes
from gpuwm.experiment import experiment_from_run_config

AUTHORITY = tomllib.loads(
    (HERE / "parent-rrtmgp-run" / "authority" / "experiment.toml")
    .read_text(encoding="utf-8"))

domain = AUTHORITY["domain"][0]
parent = {"nx": domain["nx"], "ny": domain["ny"],
          "dx": float(domain["dx"]), "dy": float(domain.get("dy", domain["dx"]))}
parent_config = dict(AUTHORITY["shared"])
parent_config.update({k: v for k, v in domain.items()})
parent_config.setdefault("run_seconds", AUTHORITY["experiment"]["run_seconds"])
parent_config.setdefault("output_interval_s", domain.get("history_interval_s", 900.0))
# The authority spells the clock as WRF's time_step; a restart's config
# carries it as dt.  Same number, this parent's own.
parent_config.setdefault("dt", float(domain["time_step"]))

VRAM_GIB = 10.0
RATIO = 3
RUN_SECONDS = float(AUTHORITY["experiment"]["run_seconds"])
OUTPUT_INTERVAL = 900.0
J0, I0 = parent["ny"] // 2, parent["nx"] // 2
START = AUTHORITY["experiment"]["start_time"]
if not isinstance(START, datetime):
    START = datetime.fromisoformat(str(START))


def fit_generic(fits) -> int:
    """The door's own doubling+bisection search, criterion injected."""
    unit = 2 * RATIO
    low, high = 0, 1
    while fits(unit * (high + 1)) and unit * high < 4096:
        high *= 2
    if high == 1 and not fits(unit * 2):
        raise SystemExit("no child fits at all")
    low = high // 2
    while low + 1 < high:
        mid = (low + high + 1) // 2
        if fits(unit * mid):
            low = mid
        else:
            high = mid
    return unit * (high if fits(unit * high) else low)


def derived_run(size: int) -> RunConfig:
    ds._centered_placement(parent, j0=J0, i0=I0, ratio=RATIO,
                           child_nx=size, child_ny=size)
    merged = ds._derive_child_run_config(
        parent_config, parent=parent, ratio=RATIO,
        child_nx=size, child_ny=size, run_seconds=RUN_SECONDS,
        output_interval_s=OUTPUT_INTERVAL)
    return RunConfig(**merged), merged


def affine_fits(size: int) -> bool:
    try:
        run, _ = derived_run(size)
    except Exception:
        return False
    exp = experiment_from_run_config(run, START)
    estimate = estimate_experiment(exp, vram_gib=VRAM_GIB)
    budget = int(card_assumed_free_gib(VRAM_GIB) * GIB) - EXTERNAL_MARGIN_BYTES
    return estimate.peak_envelope_bytes <= budget - fit_headroom_bytes(budget)


# Arm RETIRED: the shipping fit, verbatim.
retired = ds._fit_child_size(
    parent, parent_config, j0=J0, i0=I0, ratio=RATIO,
    run_seconds=RUN_SECONDS, output_interval_s=OUTPUT_INTERVAL,
    vram_gib=VRAM_GIB)

affine = fit_generic(affine_fits)

report = {"card": "RTX 3080 10 GiB (declared --vram-gib 10)",
          "parent": parent, "ratio": RATIO,
          "retired_arm_child": retired, "affine_arm_child": affine}
for name, size in (("retired", retired), ("affine", affine)):
    run, merged = derived_run(size)
    exp = experiment_from_run_config(run, START)
    estimate = estimate_experiment(exp, vram_gib=VRAM_GIB)
    report[f"{name}_peak_envelope_gib"] = round(
        estimate.peak_envelope_bytes / GIB, 3)
    report[f"{name}_cells"] = size * size
    toml_path = HERE / f"child-{name}.toml"
    toml_path.write_text(ds._render_child_toml(merged), encoding="utf-8")
    placement = ds._centered_placement(parent, j0=J0, i0=I0, ratio=RATIO,
                                       child_nx=size, child_ny=size)
    report[f"{name}_i_parent_start"] = placement.i_parent_start
    report[f"{name}_j_parent_start"] = placement.j_parent_start

print(json.dumps(report, indent=2))
(HERE / "fit_arms_report.json").write_text(
    json.dumps(report, indent=2), encoding="utf-8")
