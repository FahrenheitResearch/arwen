"""Finding 4, legacy-RRTMG arm: the two sizing criteria on the legacy parent.

The lane tip's _fit_child_size is already the affine fit, so the RETIRED
criterion (flat vram_reserve_gib budget; ALLOCATOR_HEADROOM x
(resident+transient) through the 1.75x multiplicative envelope) is
replicated here verbatim from the pre-fix code for the comparison.
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
    ALLOCATOR_HEADROOM, GIB, estimate_domain, observed_peak_envelope_bytes)
from gpuwm.domain_wizard import vram_reserve_gib
from gpuwm.experiment import DomainConfig

AUTHORITY = tomllib.loads(
    (HERE / "parent-legacy-run" / "authority" / "experiment.toml")
    .read_text(encoding="utf-8"))

domain = AUTHORITY["domain"][0]
parent = {"nx": domain["nx"], "ny": domain["ny"],
          "dx": float(domain["dx"]), "dy": float(domain.get("dy", domain["dx"]))}
parent_config = dict(AUTHORITY["shared"])
parent_config.update({k: v for k, v in domain.items()})
parent_config.setdefault("run_seconds", AUTHORITY["experiment"]["run_seconds"])
parent_config.setdefault("output_interval_s", domain.get("history_interval_s", 900.0))
parent_config.setdefault("dt", float(domain["time_step"]))

VRAM_GIB = 10.0
RATIO = 3
RUN_SECONDS = float(AUTHORITY["experiment"]["run_seconds"])
OUTPUT_INTERVAL = 900.0
J0, I0 = parent["ny"] // 2, parent["nx"] // 2


def derived_run(size: int):
    ds._centered_placement(parent, j0=J0, i0=I0, ratio=RATIO,
                           child_nx=size, child_ny=size)
    merged = ds._derive_child_run_config(
        parent_config, parent=parent, ratio=RATIO,
        child_nx=size, child_ny=size, run_seconds=RUN_SECONDS,
        output_interval_s=OUTPUT_INTERVAL)
    return RunConfig(**merged), merged


def retired_fits(size: int) -> bool:
    """The pre-fix criterion, verbatim (audit finding 4)."""
    try:
        run, _ = derived_run(size)
    except Exception:
        return False
    budget = (VRAM_GIB - vram_reserve_gib(VRAM_GIB)) * GIB
    dc = DomainConfig(
        grid_id=run.grid_id, parent_id=0, i_parent_start=1,
        j_parent_start=1, parent_grid_ratio=1, parent_time_step_ratio=1,
        history_interval_s=OUTPUT_INTERVAL, run=run)
    estimate = estimate_domain(dc, n_lbc_intervals=1)
    subtotal = estimate.resident_bytes + estimate.transient_bytes
    peak = observed_peak_envelope_bytes(
        math.ceil(ALLOCATOR_HEADROOM * subtotal))
    return peak <= budget


def fit_generic(fits) -> int:
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


retired = fit_generic(retired_fits)
# The lane-tip door IS the affine fit now.
affine = ds._fit_child_size(
    parent, dict(parent_config), j0=J0, i0=I0, ratio=RATIO,
    run_seconds=RUN_SECONDS, output_interval_s=OUTPUT_INTERVAL,
    vram_gib=VRAM_GIB)

report = {"card": "RTX 3080 10 GiB (declared --vram-gib 10)",
          "arm": "thompson-mp8 legacy-RRTMG",
          "parent": parent, "ratio": RATIO,
          "retired_arm_child": retired, "affine_arm_child": affine}
for name, size in (("retired", retired), ("affine", affine)):
    _, merged = derived_run(size)
    toml_path = HERE / f"child-legacy-{name}.toml"
    toml_path.write_text(ds._render_child_toml(merged), encoding="utf-8")
    placement = ds._centered_placement(parent, j0=J0, i0=I0, ratio=RATIO,
                                       child_nx=size, child_ny=size)
    report[f"{name}_cells"] = size * size
    report[f"{name}_i_parent_start"] = placement.i_parent_start
    report[f"{name}_j_parent_start"] = placement.j_parent_start

print(json.dumps(report, indent=2))
(HERE / "fit_arms_legacy_report.json").write_text(
    json.dumps(report, indent=2), encoding="utf-8")
