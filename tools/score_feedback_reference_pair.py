#!/usr/bin/env python3
"""Score a stock-WRF feedback A/B pair into a conservation-receipt tier.

Usage::

    python tools/score_feedback_reference_pair.py \
        --provenance  <pair>/provenance.json \
        --operator-map <pair>/feedback_operator_map.json \
        --run-a       <dir holding run A's parent history frames> \
        --run-b       <dir holding run B's parent history frames> \
        --frame       <parent history filename>   [--frame ...] \
        --out         <tier json>

``--run-a``/``--run-b`` are given separately from the provenance record
because the pair is normally scored from a copied subset: the provenance
carries the paths the pair was PRODUCED at, and rewriting those into
local paths would falsify the record.  The two are cross-checked instead
-- the tier reports both.

Two checks run before any scoring, and both refuse rather than warn:

* the pair's single-variable guarantee (identical inputs, feedback 0 vs
  1, pristine build) is read out of the provenance record;
* every scored field's staggering is derived from the file's own
  dimensions and compared against the operator map's declaration.  The
  map is generated from WRF's own include file, so a disagreement means
  the file and the map are not describing the same run and the scoring
  would silently use the wrong parent region.

The null control (run B scored against run A's own file) is computed by
this tool, not asserted by it: a scorer that returned zeros everywhere
would otherwise be indistinguishable from a pair with no feedback signal.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gpuwm.verify.feedback_reference import (
    ReferencePairError, build_feedback_ab_tier, load_operator_map,
    load_reference_pair_layout, operator_class_census, score_field_regions,
)

#: History-file variable names whose registry key is not the lowercased
#: output name.  Every entry is a name change WRF itself makes on
#: output; an unresolved field raises rather than joining this table by
#: guesswork.
_REGISTRY_ALIASES: dict[str, str] = {"HGT": "ht"}

#: The horizontal staggered dimension names a WRF history file uses.
_X_STAGGERED_DIM = "west_east_stag"
_Y_STAGGERED_DIM = "south_north_stag"


def resolve_registry_key(name: str, classes: dict[str, str]) -> str:
    """Map a history variable name onto its operator-map key."""
    for candidate in (_REGISTRY_ALIASES.get(name), name.lower(),
                      f"{name.lower()}_2"):
        if candidate and candidate in classes:
            return candidate
    raise ReferencePairError(
        f"history variable {name!r} has no operator-map entry: the parent "
        "region it is written over is unknown, so it is not scored")


def read_frame(path: Path, names: list[str]):
    """``{name: (array, xstag, ystag)}`` for one history frame."""
    import netCDF4

    out = {}
    with netCDF4.Dataset(path) as dataset:
        for name in names:
            if name not in dataset.variables:
                raise ReferencePairError(f"{path} carries no {name!r}")
            variable = dataset.variables[name]
            values = np.asarray(variable[:], dtype=np.float64)
            if values.shape[0] == 1:            # leading unlimited Time
                values = values[0]
            out[name] = (values,
                         _X_STAGGERED_DIM in variable.dimensions,
                         _Y_STAGGERED_DIM in variable.dimensions)
    return out


def check_staggering(name: str, declared: str, xstag: bool,
                     ystag: bool) -> None:
    """The file's dimensions and the map's declaration must agree."""
    from_file = ("u_face" if xstag else "v_face" if ystag else None)
    if from_file is None:
        if declared in ("u_face", "v_face"):
            raise ReferencePairError(
                f"{name}: the operator map declares {declared} but the "
                "history file carries it unstaggered")
        return
    if declared != from_file:
        raise ReferencePairError(
            f"{name}: the history file's dimensions say {from_file}, the "
            f"operator map says {declared}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--operator-map", type=Path, required=True)
    parser.add_argument("--run-a", type=Path, required=True)
    parser.add_argument("--run-b", type=Path, required=True)
    parser.add_argument("--frame", action="append", required=True,
                        help="parent history filename, repeatable")
    parser.add_argument("--field", action="append", required=True,
                        help="history variable to score, repeatable")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    layout = load_reference_pair_layout(args.provenance)
    classes, map_digest = load_operator_map(args.operator_map)
    census = operator_class_census(classes)

    frames = []
    null_max = 0.0
    for frame_name in args.frame:
        path_a = args.run_a / frame_name
        path_b = args.run_b / frame_name
        fields_a = read_frame(path_a, args.field)
        fields_b = read_frame(path_b, args.field)
        scored: dict[str, dict] = {}
        for name in args.field:
            key = resolve_registry_key(name, classes)
            class_name = classes[key]
            values_a, xstag, ystag = fields_a[name]
            check_staggering(name, class_name, xstag, ystag)
            scored[name] = score_field_regions(
                values_a, fields_b[name][0], layout=layout,
                class_name=class_name)
            scored[name]["registry_key"] = key
            # null control: run A scored against its own array
            null = score_field_regions(values_a, values_a, layout=layout,
                                       class_name=class_name)
            null_max = max(null_max,
                           max(null[region]["max_abs"]
                               for region in ("fb_zone", "footprint", "full")))
        frames.append({"frame": frame_name,
                       "run_a_file": str(path_a),
                       "run_b_file": str(path_b),
                       "fields": scored})
        print(f"scored {frame_name}: {len(scored)} fields")

    tier = build_feedback_ab_tier(
        layout=layout, operator_map_sha256=map_digest, class_census=census,
        frames=frames,
        null_control={
            "method": "each field of run A scored against its own array",
            "max_abs_over_all_fields_and_regions": null_max,
        })
    tier["scored_from"] = {"run_a": str(args.run_a), "run_b": str(args.run_b)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(tier, indent=2, sort_keys=True,
                                   allow_nan=False) + "\n", encoding="utf-8")
    print(f"null control max_abs {null_max}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
